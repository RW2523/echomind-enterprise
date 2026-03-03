# Ollama embed-only: nomic-embed-text for RAG. No GPU required.
FROM ollama/ollama:latest

# curl for warming embed model (AlmaLinux base uses dnf)
RUN dnf install -y curl 2>/dev/null || yum install -y curl 2>/dev/null || true

ENV OLLAMA_EMBED_MODEL=nomic-embed-text

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
