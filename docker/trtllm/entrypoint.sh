#!/usr/bin/env bash
# TensorRT-LLM OpenAI-compatible server (matches manual nvcr.io/nvidia/tensorrt-llm/run flow).
set -euo pipefail

MODEL_HANDLE="${MODEL_HANDLE:-nvidia/Qwen3-30B-A3B-FP4}"
# Memory management: cap the KV cache so the LLM coexists with Nemotron STT + SDXL on one Spark.
# 0.9 previously grabbed ~83 GB of unified memory; 0.5 leaves headroom. Override via TRTLLM_KV_FRACTION.
KVF="${TRTLLM_KV_FRACTION:-0.5}"
MAXB="${TRTLLM_MAX_BATCH:-64}"

if [ "${TRTLLM_SKIP_DOWNLOAD:-0}" = "1" ]; then
  echo "[trtllm] TRTLLM_SKIP_DOWNLOAD=1 — skipping hf download (model must exist in /root/.cache/huggingface)."
elif [ "${HF_HUB_OFFLINE:-0}" = "1" ]; then
  echo "[trtllm] HF_HUB_OFFLINE=1 — skipping hf download (offline; cache must be pre-populated)."
else
  echo "[trtllm] Downloading ${MODEL_HANDLE} (one-time into mounted HF cache)..."
  hf download "${MODEL_HANDLE}"
fi

cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: ${KVF}
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF

echo "[trtllm] Starting trtllm-serve: model=${MODEL_HANDLE} kv_fraction=${KVF} max_batch=${MAXB} on 0.0.0.0:8355 ..."
exec trtllm-serve "${MODEL_HANDLE}" \
  --max_batch_size "${MAXB}" \
  --trust_remote_code \
  --host 0.0.0.0 \
  --port 8355 \
  --extra_llm_api_options /tmp/extra-llm-api-config.yml
