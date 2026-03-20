# Thin wrapper on NVIDIA TensorRT-LLM release image: stable entrypoint + compose-friendly defaults.
FROM nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
