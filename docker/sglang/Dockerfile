# SGLang for DGX Spark - optimized NVIDIA container
# Supports both LLM (chat) and embedding models
FROM lmsysorg/sglang:spark

# Models (override via env in docker-compose)
ENV SGLANG_LLM_MODEL=nvidia/Qwen3-8B-FP8
ENV SGLANG_EMBED_MODEL=Alibaba-NLP/gte-Qwen2-1.5B-instruct

# Copy entrypoint scripts
COPY entrypoint-llm.sh /entrypoint-llm.sh
COPY entrypoint-embed.sh /entrypoint-embed.sh
RUN chmod +x /entrypoint-llm.sh /entrypoint-embed.sh

# Default: run LLM server (override with command for embed)
ENTRYPOINT ["/entrypoint-llm.sh"]
