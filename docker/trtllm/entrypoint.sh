#!/usr/bin/env bash
# TensorRT-LLM OpenAI-compatible server (matches manual nvcr.io/nvidia/tensorrt-llm/run flow).
set -euo pipefail

MODEL_HANDLE="${MODEL_HANDLE:-nvidia/Llama-3.1-8B-Instruct-FP4}"

if [ "${TRTLLM_SKIP_DOWNLOAD:-0}" = "1" ]; then
  echo "[trtllm] TRTLLM_SKIP_DOWNLOAD=1 — skipping hf download (model must exist in /root/.cache/huggingface)."
elif [ "${HF_HUB_OFFLINE:-0}" = "1" ]; then
  echo "[trtllm] HF_HUB_OFFLINE=1 — skipping hf download (offline; cache must be pre-populated)."
else
  echo "[trtllm] Downloading ${MODEL_HANDLE} (one-time into mounted HF cache)..."
  hf download "${MODEL_HANDLE}"
fi

cat > /tmp/extra-llm-api-config.yml <<'EOF'
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF

echo "[trtllm] Starting trtllm-serve on 0.0.0.0:8355 ..."
exec trtllm-serve "${MODEL_HANDLE}" \
  --max_batch_size 64 \
  --trust_remote_code \
  --host 0.0.0.0 \
  --port 8355 \
  --extra_llm_api_options /tmp/extra-llm-api-config.yml
