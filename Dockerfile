# Orpheus TTS 3B — RunPod Serverless Worker
# GPU: RTX 4090 (24GB) | Model: ~6GB | Image: ~15GB total
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

# Install core dependencies
RUN pip install --no-cache-dir \
    runpod \
    orpheus-speech \
    "vllm==0.7.3"

# Pre-download model into image (eliminates cold-start download)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('canopylabs/orpheus-tts-0.1-finetune-prod', \
                  local_dir='/app/hf_cache/hub/models--canopylabs--orpheus-tts-0.1-finetune-prod')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
