FROM nvcr.io/nvidia/pytorch:24.10-py3
# Use MIT mirror if ports.ubuntu.com returns 403; retry once on failure
RUN for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do [ -f "$f" ] && sed -i 's|http://ports.ubuntu.com/ubuntu-ports|http://mirrors.mit.edu/ubuntu-ports|g' "$f"; done; \
  apt-get update && apt-get install -y --no-install-recommends --fix-missing ffmpeg wget libsndfile1 \
  || (sleep 15 && apt-get update && apt-get install -y --no-install-recommends --fix-missing ffmpeg wget libsndfile1) \
  && rm -rf /var/lib/apt/lists/*

# Piper TTS model (expected at /voices/en_US-lessac-medium.onnx by config)
RUN mkdir -p /voices && cd /voices && \
    wget -q -O en_US-lessac-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx && \
    wget -q -O en_US-lessac-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

WORKDIR /workspace
COPY . /workspace
# setuptools>=82 removed pkg_resources; openai-whisper needs it. Pin setuptools<82 and use --no-build-isolation for whisper.
RUN python -m pip install --upgrade pip && pip install "setuptools>=70,<82" wheel
RUN pip install fastapi==0.115.6 "uvicorn[standard]==0.32.1" numpy==1.26.4 webrtcvad==2.0.10 websockets==12.0 requests==2.32.3 soundfile==0.12.1 piper-tts==1.2.0
RUN pip install --no-build-isolation openai-whisper==20240930
EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000"]
