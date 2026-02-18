"""
RunPod Serverless Handler for Orpheus TTS 3B.

Loads the Orpheus model once on cold start, then processes TTS requests.
Supports single and batch modes for efficient chapter rendering.

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

# ── Model loading (happens once on cold start) ─────────────────────

print("=" * 60)
print("Orpheus TTS 3B — RunPod Serverless Worker")
print("=" * 60)

load_start = time.time()

try:
    from orpheus_tts import OrpheusModel

    MODEL = OrpheusModel(
        model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
    )
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s")
    print("Ready for requests.")
except Exception as e:
    print(f"FATAL: Model loading failed: {e}")
    traceback.print_exc()
    MODEL = None

# ── Constants ───────────────────────────────────────────────────────

VOICES = {"tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"}
SAMPLE_RATE = 24000
MAX_TEXT_LENGTH = 4000


# ── Audio generation ────────────────────────────────────────────────

def _generate_one(text: str, voice: str) -> bytes:
    """Generate WAV audio for a single text segment."""
    # Use default parameters from orpheus_tts (known working)
    syn_tokens = MODEL.generate_speech(
        prompt=text,
        voice=voice,
    )

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for chunk in syn_tokens:
            wf.writeframes(chunk)

    return buf.getvalue()


# ── Handler ─────────────────────────────────────────────────────────

def handler(event):
    """
    RunPod handler. Two modes:

    SINGLE: input: { "text": "...", "voice": "tara" }
    BATCH:  input: { "segments": [{ "id": "N1", "text": "...", "voice": "tara" }, ...] }
    """
    if MODEL is None:
        return {"error": "Model failed to load. Check build logs."}

    try:
        inp = event["input"]

        # ── Batch mode ──────────────────────────────────────────
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
            print(f"Batch: {len(results)} segments in {total_time:.1f}s")
            return {"results": results, "total_segments": len(results), "total_time_s": round(total_time, 2)}

        # ── Single mode ─────────────────────────────────────────
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
