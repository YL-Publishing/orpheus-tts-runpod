"""
RunPod Serverless Handler for Orpheus TTS 3B.

Direct vllm + SNAC integration — bypasses orpheus-speech OrpheusModel
for full control over engine parameters (max_model_len, etc).

GPU: RTX 4090 (24GB VRAM) recommended.
Model: canopylabs/orpheus-tts-0.1-finetune-prod (~6GB)
"""

import runpod
import base64
import io
import wave
import time
import os
import traceback
import asyncio
import threading
import queue as thread_queue

import torch

# ── Model loading (happens once on cold start) ─────────────────────

print("=" * 60)
print("Orpheus TTS 3B — RunPod Serverless Worker")
print("=" * 60)

load_start = time.time()
LOAD_ERROR = None
ENGINE = None
TOKENIZER = None
SNAC_MODEL = None

try:
    print("Step 1: Loading vllm engine...")
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

    engine_args = AsyncEngineArgs(
        model="canopylabs/orpheus-tts-0.1-finetune-prod",
        dtype=torch.bfloat16,
        max_model_len=8192,
    )
    ENGINE = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"  vllm engine OK ({time.time() - load_start:.1f}s)")

    print("Step 2: Loading tokenizer...")
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-pretrained")
    print(f"  Tokenizer OK ({time.time() - load_start:.1f}s)")

    print("Step 3: Loading SNAC decoder...")
    from snac import SNAC
    SNAC_MODEL = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")
    print(f"  SNAC OK ({time.time() - load_start:.1f}s)")

    print(f"All models loaded in {time.time() - load_start:.1f}s")
    print("Ready for requests.")

except Exception as e:
    LOAD_ERROR = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    print(f"FATAL: {LOAD_ERROR}")

# ── Constants ───────────────────────────────────────────────────────

VOICES = {"tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"}
SAMPLE_RATE = 24000
MAX_TEXT_LENGTH = 4000
REQ_COUNTER = 0


# ── Token-to-audio decoder (from orpheus_tts/decoder.py) ───────────

def _convert_to_audio(multiframe):
    """Decode Orpheus tokens to audio samples using SNAC."""
    if len(multiframe) < 7:
        return None

    codes_0 = torch.tensor([], device="cuda", dtype=torch.int32)
    codes_1 = torch.tensor([], device="cuda", dtype=torch.int32)
    codes_2 = torch.tensor([], device="cuda", dtype=torch.int32)

    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames * 7]

    for j in range(num_frames):
        i = 7 * j
        codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device="cuda", dtype=torch.int32)])
        codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device="cuda", dtype=torch.int32)])
        codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device="cuda", dtype=torch.int32)])
        codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device="cuda", dtype=torch.int32)])
        codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device="cuda", dtype=torch.int32)])
        codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device="cuda", dtype=torch.int32)])
        codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device="cuda", dtype=torch.int32)])

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or
        torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or
        torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        return None

    with torch.inference_mode():
        audio_hat = SNAC_MODEL.decode(codes)

    audio_slice = audio_hat[:, :, 2048:4096]
    audio_np = audio_slice.detach().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype("int16")
    return audio_int16.tobytes()


def _turn_token_into_id(token_string, index):
    """Convert vllm token string to audio token ID."""
    token_string = token_string.strip()
    last_token_start = token_string.rfind("<custom_token_")
    if last_token_start == -1:
        return None
    last_token = token_string[last_token_start:]
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    return None


# ── Prompt formatting ───────────────────────────────────────────────

def _format_prompt(text, voice):
    """Format text + voice into Orpheus model prompt."""
    adapted_prompt = f"{voice}: {text}"
    prompt_tokens = TOKENIZER(adapted_prompt, return_tensors="pt")
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
    all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
    return TOKENIZER.decode(all_input_ids[0])


# ── Audio generation ────────────────────────────────────────────────

def _generate_one(text: str, voice: str) -> bytes:
    """Generate WAV audio for a single text segment."""
    global REQ_COUNTER
    REQ_COUNTER += 1
    request_id = f"req-{REQ_COUNTER}-{int(time.time())}"

    prompt_string = _format_prompt(text, voice)

    sampling_params = SamplingParams(
        temperature=0.4,
        top_p=0.9,
        max_tokens=2000,
        stop_token_ids=[128258],
        repetition_penalty=1.1,
    )

    # Run async vllm engine from sync context
    token_queue = thread_queue.Queue()

    async def async_producer():
        async for result in ENGINE.generate(
            prompt=prompt_string,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            token_queue.put(result.outputs[0].text)
        token_queue.put(None)

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.start()

    # Decode tokens to audio
    buffer = []
    count = 0
    audio_chunks = []

    while True:
        token_sim = token_queue.get()
        if token_sim is None:
            break
        token = _turn_token_into_id(token_sim, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = _convert_to_audio(buffer_to_proc)
                if audio_samples is not None:
                    audio_chunks.append(audio_samples)

    thread.join()

    # Write WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for chunk in audio_chunks:
            wf.writeframes(chunk)

    return buf.getvalue()


# ── Handler ─────────────────────────────────────────────────────────

def handler(event):
    if ENGINE is None:
        return {"error": f"Model failed to load: {LOAD_ERROR}"}

    try:
        inp = event["input"]

        # Batch mode
        if "segments" in inp:
            segments = inp["segments"]
            results = []
            total_start = time.time()

            for seg in segments:
                seg_id = seg.get("id", "unknown")
                text = seg.get("text", "").strip()
                voice = seg.get("voice", "tara")
                if voice not in VOICES:
                    voice = "tara"
                if not text:
                    results.append({"id": seg_id, "error": "empty text"})
                    continue
                if len(text) > MAX_TEXT_LENGTH:
                    text = text[:MAX_TEXT_LENGTH]
                try:
                    t0 = time.time()
                    wav_bytes = _generate_one(text, voice)
                    gen_time = time.time() - t0
                    audio_duration = max(0, (len(wav_bytes) - 44)) / (SAMPLE_RATE * 2)
                    results.append({
                        "id": seg_id,
                        "audio_base64": base64.b64encode(wav_bytes).decode(),
                        "duration_s": round(audio_duration, 2),
                        "gen_time_s": round(gen_time, 2),
                        "voice": voice,
                    })
                    print(f"  {seg_id}: {audio_duration:.1f}s audio in {gen_time:.1f}s")
                except Exception as e:
                    results.append({"id": seg_id, "error": str(e)})
                    print(f"  {seg_id}: ERROR - {e}")
                    traceback.print_exc()

            total_time = time.time() - total_start
            return {"results": results, "total_segments": len(results), "total_time_s": round(total_time, 2)}

        # Single mode
        text = inp.get("text", "").strip()
        voice = inp.get("voice", "tara")
        if not text:
            return {"error": "No text provided"}
        if voice not in VOICES:
            voice = "tara"
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]

        t0 = time.time()
        wav_bytes = _generate_one(text, voice)
        gen_time = time.time() - t0
        audio_duration = max(0, (len(wav_bytes) - 44)) / (SAMPLE_RATE * 2)

        return {
            "audio_base64": base64.b64encode(wav_bytes).decode(),
            "format": "wav",
            "sample_rate": SAMPLE_RATE,
            "duration_s": round(audio_duration, 2),
            "gen_time_s": round(gen_time, 2),
            "voice": voice,
            "text_length": len(text),
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Handler error: {e}\n{tb}")
        return {"error": str(e), "traceback": tb}


runpod.serverless.start({"handler": handler})
