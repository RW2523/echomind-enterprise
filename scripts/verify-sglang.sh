#!/bin/bash
# Preliminary SGLang verification for DGX Spark (per NVIDIA instructions).
# Run this before docker compose to ensure SGLang works.
set -e

echo "=== Step 1: Verify prerequisites ==="
docker --version
nvidia-smi
echo "Verifying Docker GPU support..."
docker run --rm --gpus all lmsysorg/sglang:spark nvidia-smi
echo "Disk space:"
df -h /

echo ""
echo "=== Step 2: Pull SGLang container ==="
docker pull lmsysorg/sglang:spark
docker images | grep sglang

echo ""
echo "=== Step 3 & 4: Launch SGLang with GB10 patch ==="
echo "Starting SGLang (model: nvidia/Llama-3.1-8B-Instruct-FP8, mem-fraction 0.75)..."
echo "This may take 2-5 min for model download on first run."
echo ""

# Clean up any previous run (keeps container on failure so we can inspect logs)
docker rm -f echomind-sglang-verify 2>/dev/null || true

# Use our GB10 patch to avoid mem_get_info failure; no --rm so we can get logs on crash
docker run --gpus all -d \
  --name echomind-sglang-verify \
  -p 30000:30000 \
  -v /tmp:/tmp \
  -v "$(pwd)/docker/sglang/launch_gb10.py:/tmp/launch_gb10.py" \
  -e SGLANG_GB10_MEM_MB=100000 \
  --shm-size 32g \
  --ipc host \
  lmsysorg/sglang:spark \
  python3 /tmp/launch_gb10.py \
    --model-path nvidia/Llama-3.1-8B-Instruct-FP8 \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    --tp 1 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.75 \
    --log-level info

echo "Waiting for server (poll /health every 15s, max 10 min)..."
for i in $(seq 1 40); do
  if curl -sf http://localhost:30000/health >/dev/null 2>&1; then
    echo "Server ready after $((i * 15))s"
    break
  fi
  [ $i -eq 40 ] && { echo "Health check failed after 10 min. Container logs (container kept for inspection):"; docker logs echomind-sglang-verify 2>&1 | tail -80; exit 1; }
  sleep 15
done

echo ""
echo "=== Step 5: Test /health ==="
curl -sf http://localhost:30000/health && echo " OK" || { echo "Health check failed. Container logs:"; docker logs echomind-sglang-verify 2>&1 | tail -80; exit 1; }

echo ""
echo "=== Step 6: Test /generate (native API) ==="
curl -s -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "What does NVIDIA love?", "sampling_params": {"temperature": 0.7, "max_new_tokens": 50}}' | head -c 500
echo ""
echo ""

echo "=== Step 7: Test OpenAI-compatible /v1/chat/completions ==="
curl -s -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "nvidia/Llama-3.1-8B-Instruct-FP8", "messages": [{"role": "user", "content": "Say hello in 5 words."}], "max_tokens": 20}' | head -c 500
echo ""
echo ""

echo "=== SGLang verification PASSED ==="
echo ""
echo "Next steps:"
echo "  1. Stop the test container:  docker stop echomind-sglang-verify"
echo "  2. Run full stack:           docker compose up --build"
echo ""
echo "Or run compose now (sglang-llm will fail to bind port 30000 if verify container is still running)."
