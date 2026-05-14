# Build from repository root: docker compose build backend
# For x86_64: nvcr.io/nvidia/pytorch:25.01-py3
# For ARM64 / DGX Spark: use NVIDIA ARM64 PyTorch image, e.g. nvcr.io/nvidia/pytorch:25.01-py3 (check NGC for ARM64 tags)
FROM nvcr.io/nvidia/pytorch:25.01-py3
RUN set -e; \
  for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do \
    [ -f "$f" ] && sed -i -e 's|https\?://ports\.ubuntu\.com/ubuntu-ports/\?|http://mirrors.mit.edu/ubuntu-ports/|g' -e 's|https\?://ports\.ubuntu\.com/ubuntu-ports|http://mirrors.mit.edu/ubuntu-ports/|g' "$f" || true; \
  done; \
  apt-get update && apt-get install -y --no-install-recommends --fix-missing ffmpeg libopus-dev \
  || (sleep 20 && apt-get update && apt-get install -y --no-install-recommends --fix-missing ffmpeg libopus-dev); \
  rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=10
ARG USE_FAISS_GPU=1
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# NeMo ASR (Nemotron streaming) — keep NGC PyTorch/CUDA wheels; avoid multi-GB PyPI torch downloads.
COPY docker/install_nemo_asr.sh /tmp/install_nemo_asr.sh
RUN chmod +x /tmp/install_nemo_asr.sh && /tmp/install_nemo_asr.sh

# NeMo upgrades torch; NGC torchvision targets the old torch — upgrade torchvision to match.
# NeMo pulls NumPy 2.x; faiss-cpu targets NumPy 1.x — pin NumPy and reinstall FAISS.
RUN pip install --no-cache-dir "numpy>=1.26.4,<2.0" --force-reinstall \
 && pip install --no-cache-dir --no-deps --force-reinstall "faiss-cpu==1.8.0.post1" \
 && pip install --no-cache-dir --upgrade torchvision

COPY nemotron_asr /app/nemotron_asr
RUN pip install --no-cache-dir /app/nemotron_asr

# Pre-download Nemotron weights during build (offline runs use HF_HUB_OFFLINE=1 and local cache only).
ARG SKIP_MODEL_DOWNLOAD=0
ARG SKIP_NEMOTRON_WARMUP=0
ARG SKIP_BOARD_ROOM_NEMOTRON_WARMUP=1
ARG ECHOMIND_ASR_MODEL_NAME=nvidia/nemotron-speech-streaming-en-0.6b
ARG BOARD_ROOM_ASR_MODEL_NAME=nvidia/multitalker-parakeet-streaming-0.6b-v1
ARG WARMUP_ASR_DEVICE=cpu
ARG HF_TOKEN=
RUN export HF_TOKEN="${HF_TOKEN}" && \
    if [ "$SKIP_MODEL_DOWNLOAD" != "1" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('${ECHOMIND_ASR_MODEL_NAME}'); snapshot_download('${BOARD_ROOM_ASR_MODEL_NAME}')"; \
    fi
RUN export HF_TOKEN="${HF_TOKEN}" && \
    if [ "$SKIP_MODEL_DOWNLOAD" != "1" ] && [ "$SKIP_NEMOTRON_WARMUP" != "1" ]; then \
      TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 TORCH_COMPILE_DISABLE=1 \
      python -m nemotron_asr.warmup_build \
        --model "${ECHOMIND_ASR_MODEL_NAME}" \
        --device "${WARMUP_ASR_DEVICE}"; \
      if [ "$SKIP_BOARD_ROOM_NEMOTRON_WARMUP" != "1" ]; then \
        TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 TORCH_COMPILE_DISABLE=1 \
        python -m nemotron_asr.warmup_build \
          --model "${BOARD_ROOM_ASR_MODEL_NAME}" \
          --device "${WARMUP_ASR_DEVICE}"; \
      fi; \
    fi

COPY backend/app /app/app
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
