# Extends official Ollama: pre-pull models during build so first container start is fast.
FROM ollama/ollama:latest

# Models to pre-pull (match backend/voice env)
ENV OLLAMA_LLM_MODEL=qwen2.5:7b-instruct
ENV OLLAMA_EMBED_MODEL=nomic-embed-text

# Preload embedding + LLM models during build (stored in /opt/ollama-models; entrypoint copies to volume on first run).
ENV OLLAMA_MODELS=/opt/ollama-models
RUN mkdir -p /opt/ollama-models && \
    (ollama serve &) && sleep 20 && \
    ollama pull nomic-embed-text && \
    ollama pull qwen2.5:7b-instruct && \
    (pkill ollama || true)

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
