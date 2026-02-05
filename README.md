# EchoMind — Enterprise Edition (Clean v2)

This build removes ALL Gemini code and connects the UI to your backend APIs.

## Services
- Frontend: http://<DGX_IP>:3000
- Backend API: proxied under /api
- Voice bot: proxied under /voice (direct: http://<DGX_IP>:8001)
- Ollama: http://<DGX_IP>:11434

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
