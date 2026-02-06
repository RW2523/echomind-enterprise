FROM nvcr.io/nvidia/pytorch:25.01-py3
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Use requirements-gpu.txt for FAISS GPU (faster search). Default: faiss-cpu.
ARG USE_FAISS_GPU=0
COPY requirements.txt requirements-gpu.txt /app/
RUN python -m pip install --upgrade pip && \
    if [ "$USE_FAISS_GPU" = "1" ]; then pip install -r requirements-gpu.txt; else pip install -r requirements.txt; fi
COPY app /app/app
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
