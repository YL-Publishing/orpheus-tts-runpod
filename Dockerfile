# Orpheus TTS 3B — RunPod Serverless Worker
# GPU: RTX 4090 (24GB) | Model: ~6GB | Image: ~20GB total
#
# Build:
#   docker build --platform linux/amd64 -t YOUR_DOCKERHUB/orpheus-tts-runpod:latest .
#   docker push YOUR_DOCKERHUB/orpheus-tts-runpod:latest
#
# Or use RunPod GitHub integration for auto-builds.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install orpheus-speech first (may pull its own vllm)
RUN pip install --no-cache-dir runpod orpheus-speech

# Force vllm==0.7.3 (orpheus-speech March 2025 known-good version)
RUN pip install --no-cache-dir "vllm==0.7.3"

# Pre-download model AND tokenizer into HF cache (standard format)
# This eliminates cold-start downloads (~6GB model + tokenizer)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('canopylabs/orpheus-tts-0.1-finetune-prod'); \
snapshot_download('canopylabs/orpheus-3b-0.1-pretrained')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
