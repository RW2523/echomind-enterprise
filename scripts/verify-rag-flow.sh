#!/bin/bash
# Verify the entire RAG flow: LLM, embeddings, retrieval, and backend.
# Run after: docker compose up -d (or with full stack running)
set -e

echo "=== RAG Flow Verification ==="
echo ""

# Default ports (adjust if different)
BACKEND_PORT="${BACKEND_PORT:-8000}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
OLLAMA_EMBED_PORT="${OLLAMA_EMBED_PORT:-11435}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

# API base: frontend proxies /api to backend. Backend /health is at root (not under /api).
# Try frontend first (localhost:3000), then direct backend if published.
API_BASE="${API_BASE:-http://localhost:${FRONTEND_PORT}/api}"
if [ -n "$SKIP_PROXY" ]; then
  API_BASE="http://localhost:${BACKEND_PORT}/api"
fi

echo "=== 1. SGLang LLM (GPU) ==="
if curl -sf "http://localhost:${SGLANG_PORT}/health" >/dev/null 2>&1; then
  echo "  OK: SGLang health check passed"
else
  echo "  FAIL: SGLang not responding on port ${SGLANG_PORT}"
  echo "  Run: ./scripts/verify-sglang.sh first, or docker compose up"
  exit 1
fi

echo ""
echo "=== 2. Ollama Embeddings (CPU) ==="
EMBED_RESP=$(curl -sf -X POST "http://localhost:${OLLAMA_EMBED_PORT}/api/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"model":"nomic-embed-text","prompt":"test"}' 2>/dev/null || true)
if echo "$EMBED_RESP" | grep -q '"embedding"'; then
  DIM=$(echo "$EMBED_RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('embedding',d.get('embeddings',[[]])[0])))" 2>/dev/null || echo "?")
  echo "  OK: Embeddings returned (dim=${DIM:-?})"
else
  echo "  FAIL: Ollama embed not responding on port ${OLLAMA_EMBED_PORT}"
  echo "  Ensure ollama-embed container is running"
  exit 1
fi

echo ""
echo "=== 3. Backend API ==="
CREATE=$(curl -sf -X POST "${API_BASE}/chat/create" -H "Content-Type: application/json" -d '{"title":"Verify"}' 2>/dev/null || true)
if echo "$CREATE" | grep -q '"chat_id"'; then
  echo "  OK: Backend API responding"
else
  echo "  FAIL: Backend not responding at ${API_BASE}"
  echo "  Ensure docker compose up and frontend is on port ${FRONTEND_PORT}"
  echo "  Response: ${CREATE:0:200}"
  exit 1
fi

echo ""
echo "=== 4. Chat/RAG Flow ==="
if echo "$CREATE" | grep -q '"chat_id"'; then
  CHAT_ID=$(echo "$CREATE" | python3 -c "import json,sys; print(json.load(sys.stdin)['chat_id'])" 2>/dev/null)
  if [ -n "$CHAT_ID" ]; then
    ASK=$(curl -sf -X POST "${API_BASE}/chat/ask" -H "Content-Type: application/json" \
      -d "{\"chat_id\":\"$CHAT_ID\",\"message\":\"What is 2+2?\"}" 2>/dev/null || true)
    if echo "$ASK" | grep -q '"answer"'; then
      echo "  OK: Chat/ask returned answer"
    else
      echo "  FAIL: /ask did not return valid answer"
      echo "  Response: $ASK"
      exit 1
    fi
  else
    echo "  FAIL: Could not parse chat_id"
    exit 1
  fi
else
  echo "  FAIL: Could not create chat"
  echo "  Response: $CREATE"
  exit 1
fi

echo ""
echo "=== 5. RAG Timing Logs ==="
if docker compose logs backend 2>&1 | grep -q "RAG_TIMING"; then
  echo "  OK: RAG timing logs present"
  echo "  Sample:"
  docker compose logs backend 2>&1 | grep "RAG_TIMING" | tail -5 | sed 's/^/    /'
else
  echo "  INFO: No RAG_TIMING logs yet (send a chat message to generate)"
fi

echo ""
echo "=== RAG Flow Verification PASSED ==="
echo ""
echo "Summary:"
echo "  - LLM (SGLang):     GPU - port ${SGLANG_PORT}"
echo "  - Embeddings:       CPU - port ${OLLAMA_EMBED_PORT}"
echo "  - FAISS:            CPU (faiss-cpu, USE_FAISS_GPU=0)"
echo "  - Backend:          GPU for Kyutai STT, CPU for RAG"
echo ""
echo "See docs/RAG_FLOW_GPU_AUDIT.md for details."
