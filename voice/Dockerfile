# Build from repository root: docker compose build voice
FROM nvcr.io/nvidia/pytorch:25.01-py3
RUN set -e; \
  for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do \
    [ -f "$f" ] && sed -i -e 's|https\?://ports\.ubuntu\.com/ubuntu-ports/\?|http://mirrors.mit.edu/ubuntu-ports/|g' -e 's|https\?://ports\.ubuntu\.com/ubuntu-ports|http://mirrors.mit.edu/ubuntu-ports/|g' "$f" || true; \
  done; \
  apt-get update && apt-get install -y --no-install-recommends --fix-missing ffmpeg wget libsndfile1 \
  || (sleep 20 && apt-get update && apt-get install -y --no-install-recommends --fix-missing ffmpeg wget libsndfile1); \
  rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=10
RUN python -m pip install --upgrade pip && pip install "setuptools>=70,<82" wheel

COPY voice/requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY docker/install_nemo_asr.sh /tmp/install_nemo_asr.sh
RUN chmod +x /tmp/install_nemo_asr.sh && /tmp/install_nemo_asr.sh

RUN pip install --no-cache-dir "numpy>=1.26.4,<2.0" --force-reinstall \
 && pip install --no-cache-dir --upgrade torchvision

COPY nemotron_asr /workspace/nemotron_asr
RUN pip install --no-cache-dir /workspace/nemotron_asr

# Piper: onnxruntime. onnxruntime-gpu has no manylinux aarch64 wheel from PyPI — keep CPU ORT on ARM64 (Spark).
RUN set -e; \
    ARCH="$(dpkg --print-architecture)"; \
    if [ "$ARCH" = "amd64" ]; then \
      pip uninstall -y onnxruntime 2>/dev/null || true; \
      pip install --no-cache-dir "onnxruntime-gpu>=1.17.0,<2"; \
    else \
      echo "voice image ($ARCH): using onnxruntime CPU for Piper; CUDA EP unavailable via pip on this arch"; \
    fi

ARG SKIP_MODEL_DOWNLOAD=0
ARG SKIP_NEMOTRON_WARMUP=0
ARG ECHOMIND_ASR_MODEL_NAME=nvidia/nemotron-speech-streaming-en-0.6b
ARG WARMUP_ASR_DEVICE=cpu
# Optional: HF_TOKEN for gated Nemotron weights during build (same as runtime .env)
ARG HF_TOKEN=
RUN export HF_TOKEN="${HF_TOKEN}" && \
    if [ "$SKIP_MODEL_DOWNLOAD" != "1" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('${ECHOMIND_ASR_MODEL_NAME}')"; \
    fi
# Load NeMo + one forward pass on CPU (docker build has no GPU by default); fills HF/NeMo caches for fast runtime GPU load.
RUN export HF_TOKEN="${HF_TOKEN}" && \
    if [ "$SKIP_MODEL_DOWNLOAD" != "1" ] && [ "$SKIP_NEMOTRON_WARMUP" != "1" ]; then \
      TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 TORCH_COMPILE_DISABLE=1 \
      python -m nemotron_asr.warmup_build \
        --model "${ECHOMIND_ASR_MODEL_NAME}" \
        --device "${WARMUP_ASR_DEVICE}"; \
    fi

RUN mkdir -p /voices && cd /voices && \
    wget -q -O en_US-lessac-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx && \
    wget -q -O en_US-lessac-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

COPY voice /workspace

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000"]
