# Docker flow and troubleshooting

## Startup order

1. **ollama** – Builds (or uses image), pulls LLM + embed models, runs healthcheck until both models are listed. This can take several minutes (healthcheck: 80 retries × 15s, start_period 300s).
2. **backend** – Starts only after `ollama` is **healthy**. Uses `/data` (volume `echomind_data`) for SQLite and FAISS. Listens on port 8000.
3. **voice** – Starts after `ollama` is healthy and **backend** has started. Port 8001→8000.
4. **frontend** – Starts after **backend** and **voice** have started. Builds with Vite, serves via nginx on 3000 (HTTP) and 3443 (HTTPS). Nginx proxies `/api/` → backend:8000, `/voice/` → voice:8000.

So if you only see **ollama** logs, backend is still waiting for ollama’s healthcheck to pass. Once it does, backend and then voice and frontend should start.

## If frontend or backend don’t come up

### 1. Check status of all services

```bash
docker compose ps -a
```

Look for `exited`, `unhealthy`, or `starting` and which service is affected.

### 2. Inspect backend logs

```bash
docker compose logs backend
```

- **GPU reservation errors** – Previously the backend reserved 1 GPU. If the GPU was already used by ollama or nvidia-container-toolkit wasn’t available, the backend could fail to start. The compose file now has the backend GPU block **commented out**, so the backend runs without a GPU (FAISS CPU). If you need FAISS GPU, uncomment the `deploy.resources` block and ensure a GPU is available.
- **Module/import errors** – Usually from a failed or partial build. Run `docker compose build --no-cache backend` and try again.
- **Address already in use** – If port 8000 is taken on the host, change `ports: ["8000:8000"]` or stop the other process.

### 3. Inspect frontend logs

```bash
docker compose logs frontend
```

- **Build failures** – Often `npm install` or `vite build` (e.g. missing deps, Node version). Fix errors in the Dockerfile or frontend app and rebuild: `docker compose build --no-cache frontend`.
- **nginx errors** – Check that `frontend/nginx.conf` exists and is copied into the image.

### 4. Rebuild and start clean

```bash
docker compose down
docker compose build --no-cache backend frontend
docker compose up -d
docker compose logs -f backend frontend
```

### 5. Backend port

Backend is exposed as **8000:8000** so you can call `http://localhost:8000/health` from the host. The frontend (nginx) proxies `/api/` to the backend container; with `VITE_API_BASE=` the browser talks to the same host (e.g. localhost:3000) and nginx forwards `/api/` to backend.

## Optional: run only some services

To bring up only ollama and backend (no voice, no frontend):

```bash
docker compose up -d ollama backend
```

Then open or proxy backend at port 8000.
