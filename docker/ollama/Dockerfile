# Extends official Ollama: pre-pulls models at startup so no runtime loading.
FROM ollama/ollama:latest

# curl needed by entrypoint to warm LLM and embed models via API
# Use MIT mirror + retry to avoid transient "mirror sync in progress" failures from ports.ubuntu.com
RUN for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do [ -f "$f" ] && sed -i 's|http://ports.ubuntu.com/ubuntu-ports|http://mirrors.mit.edu/ubuntu-ports|g' "$f"; done; \
  apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
  || (sleep 20 && apt-get update && apt-get install -y --no-install-recommends curl ca-certificates) \
  && rm -rf /var/lib/apt/lists/*

# Models to pre-pull (match backend/voice env)
ENV OLLAMA_LLM_MODEL=qwen2.5:7b-instruct
ENV OLLAMA_EMBED_MODEL=nomic-embed-text

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
