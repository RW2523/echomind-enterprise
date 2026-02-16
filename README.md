# EchoMind — Enterprise Edition (Clean v2)

This build removes ALL Gemini code and connects the UI to your backend APIs.

## Services
- Frontend: http://<DGX_IP>:3000 (HTTP) or https://<DGX_IP>:3443 (HTTPS)
- Backend API: proxied under /api
- Voice bot: proxied under /voice (direct: http://<DGX_IP>:8001). **Voice AI is connected to RAG** (via `BACKEND_CHAT_URL`): questions about your transcripts or uploaded PDFs are answered from the knowledge base.
- Ollama: http://<DGX_IP>:11434

## HTTPS

**Recommended: Option B – Trusted certificate (no browser warning)**  
Use a free domain and Let's Encrypt so the browser trusts the certificate:
- Free subdomain: [DuckDNS](https://www.duckdns.org/) → **echomind.duckdns.org**
- Trusted cert: **`sudo certbot --nginx -d echomind.duckdns.org`**
- **HTTPS:** https://echomind.duckdns.org — no warning, fully trusted. Free.
- Full steps: **[docs/HTTPS_TRUSTED_CERTIFICATE.md](docs/HTTPS_TRUSTED_CERTIFICATE.md)**

**Option A – Self-signed (quick local HTTPS)**  
If you don't need a domain, the image includes a self-signed cert:
- **HTTPS:** https://localhost:3443 (or https://\<your-ip\>:3443)
- Browser will show a certificate warning; choose **Advanced** → **Proceed**. Use Option B above for no warning.

## Run
```bash
docker compose up --build
```

### Build fails with "failed to execute bake: read |0: file already closed"
This can happen at the end of a Buildx build when writing provenance metadata. Disable provenance and rebuild:

```bash
BUILDX_METADATA_PROVENANCE=disabled docker compose build
docker compose up -d
```

Or in one go: `BUILDX_METADATA_PROVENANCE=disabled docker compose up --build`.  
If you use `docker buildx bake` instead of `docker compose build`, run it with the same env var: `BUILDX_METADATA_PROVENANCE=disabled docker buildx bake`.

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
