# EchoMind — Enterprise Edition (Clean v2)

This build removes ALL Gemini code and connects the UI to your backend APIs.

## Services
- Frontend: http://<DGX_IP>:3000 (HTTP) or https://<DGX_IP>:3443 (HTTPS)
- Backend API: proxied under /api
- Voice bot: proxied under /voice (direct: http://<DGX_IP>:8001)
- Ollama: http://<DGX_IP>:11434

## HTTPS without a domain
The frontend image includes a **self-signed certificate** so you can use HTTPS with no domain:
- **HTTPS:** https://localhost:3443 (or https://\<your-ip\>:3443)
- Your browser will show a certificate warning (e.g. "Your connection is not private"); choose **Advanced** → **Proceed** to continue. This is expected when using a self-signed cert.
- The app and Voice WebSocket (wss) work over this HTTPS port. HTTP remains on port 3000 if you prefer.

## Run
```bash
docker compose up --build
```

## Pull models (once)
```bash
docker exec -it echomind-ollama ollama pull qwen2.5:7b-instruct
docker exec -it echomind-ollama ollama pull nomic-embed-text
```

If you still see Gemini calls in the browser console:
1) Hard refresh (Ctrl+Shift+R) / clear site data
2) Ensure you rebuilt images: `docker compose up --build`

## FAISS GPU (faster RAG search)

By default the backend uses **faiss-cpu**. For faster vector search you can use **faiss-gpu** (requires an NVIDIA GPU and CUDA).

1. In `docker-compose.yml`, set the backend build arg: `USE_FAISS_GPU: "1"`.
2. Rebuild: `docker compose build --no-cache backend && docker compose up -d backend`.

The backend service already has GPU access in `docker-compose.yml`. No code changes are needed—the same `faiss` API is used; the GPU build just runs the index on the GPU.

**Note:** The PyPI `faiss-gpu` package (1.7.2) is archived and only provides wheels for Python ≤3.10. If the backend image uses Python 3.11+, the GPU build may fail; in that case keep `faiss-cpu` or use a conda base image with `faiss-gpu`.
