# Build from repository root: docker compose build voice
FROM nvcr.io/nvidia/pytorch:25.01-py3
RUN set -e; \
  for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do \
    [ -f "$f" ] && sed -i -e 's|https\?://ports\.ubuntu\.com/ubuntu-ports/\?|http://mirrors.mit.edu/ubuntu-ports/|g' -e 's|https\?://ports\.ubuntu\.com/ubuntu-ports|http://mirrors.mit.edu/ubuntu-ports/|g' "$f" || true; \
  done; \
  apt-get update && apt-get install -y --no-install-recommends --fix-missing ffmpeg wget libsndfile1 espeak-ng \
  || (sleep 20 && apt-get update && apt-get install -y --no-install-recommends --fix-missing ffmpeg wget libsndfile1 espeak-ng); \
  rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN python -m pip install --upgrade pip && pip install "setuptools>=70,<82" wheel

COPY voice/requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

RUN pip install --no-cache-dir Cython packaging sentencepiece \
 && pip install --no-cache-dir "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"

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
# Accurate final-decode model (Parakeet-TDT) for the voice path — public model, baked for offline runtime.
ARG VOICE_FINAL_ASR_MODEL=nvidia/parakeet-tdt-0.6b-v2
RUN export HF_TOKEN="${HF_TOKEN}" && \
    if [ "$SKIP_MODEL_DOWNLOAD" != "1" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('${VOICE_FINAL_ASR_MODEL}')"; \
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

# Kokoro-82M TTS (selectable alternative to Piper) — install + bake model/voice/G2P caches for offline runtime.
RUN pip install --no-cache-dir kokoro soundfile
RUN if [ "$SKIP_MODEL_DOWNLOAD" != "1" ]; then \
      python - <<'PY' ; \
from kokoro import KPipeline
p = KPipeline(lang_code="a")
for v in ["af_heart","af_bella","af_nicole","af_sarah","am_michael","am_fenrir","am_adam"]:
    for _ in p("Warming up the Kokoro voice cache.", voice=v):
        pass
print("kokoro baked")
PY
    fi

COPY voice /workspace

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000"]
