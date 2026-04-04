#!/usr/bin/env bash
# Run the entire Live Transcript application with Docker Compose.
# For DGX Spark: uses 1 NVIDIA GPU (nvidia-container-toolkit required).
set -e
cd "$(dirname "$0")"
echo "Building and starting services (backend with GPU on DGX Spark)..."
docker compose up --build -d
echo ""
echo "Waiting for backend to become healthy (model load can take 2–3 minutes on first run)..."
for i in {1..36}; do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null | grep -q 200; then
    echo "Backend is ready."
    break
  fi
  if [ "$i" -eq 36 ]; then
    echo "Backend health check timed out. Check: docker compose logs backend"
    exit 1
  fi
  sleep 5
done
echo ""
docker compose ps
echo ""
echo "App is running:"
echo "  Frontend (HTTPS): https://localhost  (use this for microphone access)"
echo "  Frontend (HTTP):  http://localhost   (redirects to HTTPS)"
echo "  Backend:          http://localhost:8000"
echo "  Health:           http://localhost:8000/health"
echo ""
echo "Note: Browser will warn about self-signed certificate; accept to use the mic."
echo ""
echo "To view logs: docker compose logs -f"
