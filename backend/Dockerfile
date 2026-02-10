FROM nvcr.io/nvidia/pytorch:25.01-py3
# Replace all ports.ubuntu.com (http/https, with/without trailing slash) to work around 403; include .sources (DEB822)
RUN set -e; \
  for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do \
    [ -f "$f" ] && sed -i -e 's|https\?://ports\.ubuntu\.com/ubuntu-ports/\?|http://mirrors.mit.edu/ubuntu-ports/|g' -e 's|https\?://ports\.ubuntu\.com/ubuntu-ports|http://mirrors.mit.edu/ubuntu-ports|g' "$f" || true; \
  done; \
  apt-get update && apt-get install -y --no-install-recommends --fix-missing ffmpeg \
  || (sleep 20 && apt-get update && apt-get install -y --no-install-recommends --fix-missing ffmpeg); \
  rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Use requirements-gpu.txt for FAISS GPU (faster search). Default: faiss-cpu.
ARG USE_FAISS_GPU=0
COPY requirements.txt requirements-gpu.txt /app/
# Do not upgrade setuptools to 82+ (pkg_resources removed). Install whisper with base setuptools then rest.
RUN pip install --no-build-isolation openai-whisper==20240930 && pip install -r requirements.txt

COPY app /app/app
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
