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
# Use requirements-gpu.txt for FAISS GPU (faster search). Default: faiss-cpu.
ARG USE_FAISS_GPU=1
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# NeMo ASR (Nemotron streaming). Cython/packaging are NeMo prerequisites.
RUN pip install --no-cache-dir Cython packaging sentencepiece \
 && pip install --no-cache-dir "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"

# NeMo upgrades torch; NGC torchvision targets the old torch — upgrade torchvision to match.
# NeMo pulls NumPy 2.x; faiss-cpu targets NumPy 1.x — pin NumPy and reinstall FAISS.
RUN pip install --no-cache-dir "numpy>=1.26.4,<2.0" --force-reinstall \
 && pip install --no-cache-dir --no-deps --force-reinstall "faiss-cpu==1.8.0.post1" \
 && pip install --no-cache-dir --upgrade torchvision

# Pre-download Nemotron weights during build (offline runs use HF_HUB_OFFLINE=1 and local cache only).
ARG SKIP_MODEL_DOWNLOAD=0
ARG ECHOMIND_ASR_MODEL_NAME=nvidia/nemotron-speech-streaming-en-0.6b
RUN if [ "$SKIP_MODEL_DOWNLOAD" != "1" ]; then python -c "from huggingface_hub import snapshot_download; snapshot_download('${ECHOMIND_ASR_MODEL_NAME}')"; fi

COPY app /app/app
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
