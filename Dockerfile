# Orpheus TTS 3B — RunPod Serverless Worker
# GPU: RTX 4090 (24GB) | Model: ~6GB
#
# Uses official vllm image as base (vllm 0.7.3 + PyTorch + CUDA pre-installed).
# This avoids all CUDA/PyTorch/vllm version conflicts.
#
# Model downloads at first cold start (~60s), then cached by FlashBoot.

FROM vllm/vllm-openai:v0.7.3

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# snac = audio decoder for Orpheus, runpod = serverless handler SDK
RUN pip install --no-cache-dir runpod snac

# orpheus-speech without deps (vllm + torch + transformers already in base)
RUN pip install --no-cache-dir --no-deps orpheus-speech

COPY handler.py /app/handler.py

# Override vllm's ENTRYPOINT (it runs the OpenAI server, not our handler)
ENTRYPOINT []
CMD ["python3", "-u", "/app/handler.py"]
