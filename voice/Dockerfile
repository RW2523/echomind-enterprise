FROM nvcr.io/nvidia/pytorch:24.10-py3
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg wget libsndfile1 && rm -rf /var/lib/apt/lists/*

# Piper TTS model (expected at /voices/en_US-lessac-medium.onnx by config)
RUN mkdir -p /voices && cd /voices && \
    wget -q -O en_US-lessac-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx && \
    wget -q -O en_US-lessac-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

WORKDIR /workspace
COPY . /workspace
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn","app.server:app","--host","0.0.0.0","--port","8000"]
