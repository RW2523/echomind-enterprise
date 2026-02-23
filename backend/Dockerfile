# For x86_64: nvcr.io/nvidia/pytorch:25.01-py3
# For ARM64 / DGX Spark: use NVIDIA ARM64 PyTorch image, e.g. nvcr.io/nvidia/pytorch:25.01-py3 (check NGC for ARM64 tags)
FROM nvcr.io/nvidia/pytorch:25.01-py3
# Replace all ports.ubuntu.com (http/https, with/without trailing slash) to work around 403; include .sources (DEB822)
RUN set -e; \
  for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do \
    [ -f "$f" ] && sed -i -e 's|https\?://ports\.ubuntu\.com/ubuntu-ports/\?|http://mirrors.mit.edu/ubuntu-ports/|g' -e 's|https\?://ports\.ubuntu\.com/ubuntu-ports|http://mirrors.mit.edu/ubuntu-ports|g' "$f" || true; \
  done; \
  apt-get update && apt-get install -y --no-install-recommends --fix-missing ffmpeg libopus-dev \
  || (sleep 20 && apt-get update && apt-get install -y --no-install-recommends --fix-missing ffmpeg libopus-dev); \
  rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-gpu.txt /app/
ARG USE_FAISS_GPU=1
# faiss-gpu has no PyPI wheel for aarch64/ARM64 (e.g. DGX Spark); use GPU only on x86_64.
RUN ARCH=$(dpkg --print-architecture) && \
    echo "Installing FAISS: USE_FAISS_GPU=$USE_FAISS_GPU, arch=$ARCH" && \
    if [ "$USE_FAISS_GPU" = "1" ] && [ "$ARCH" = "amd64" ]; then \
      pip install -r requirements-gpu.txt; \
    else \
      pip install -r requirements.txt; \
    fi

RUN pip install --no-cache-dir "bitsandbytes==0.46.0" sentencepiece \
 && pip install --no-cache-dir --no-deps "moshi==0.2.9"

# Pre-download Kyutai STT model during build (avoids 2–5 min load at first Live Transcript use).
# HF_HUB_DISABLE_PROGRESS_BARS=1 avoids tqdm progress output that can cause "read |0: file already closed" with buildx.
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('kyutai/stt-1b-en_fr')"

COPY app /app/app
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV UVICORN_WORKERS=2
# Shell form so UVICORN_WORKERS is expanded at runtime (set per-host for CPU-based concurrency).
CMD uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers ${UVICORN_WORKERS:-2} \
  --loop uvloop \
  --http httptools
