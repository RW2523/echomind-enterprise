# Going Public with Cloudflare Tunnel — `echomind-ajace.com`

This guide exposes the EchoMind stack running on your DGX Spark to the public internet at **https://echomind-ajace.com**, safely, using a **Cloudflare Tunnel** (no router port-forwarding, your home IP stays hidden) **gated by Cloudflare Access** (a login wall in front of the app).

> ⚠️ **Read this first.** EchoMind ships with **no built-in authentication** and wildcard CORS — by design, for trusted networks. On a public URL that means *anyone* could use your GPU and read/delete/upload to your knowledge base. **Cloudflare Access is therefore mandatory**, not optional: it is the login wall that makes "public" safe. Do **not** publish the URL until Step 4 is done.

## What's already wired up (in this repo)
- `docker-compose.yml` — a `cloudflared` service behind the **`public` profile** (it does **not** start with a normal `docker compose up`).
- `frontend/nginx.conf` — `server_name` includes `echomind-ajace.com`.
- The frontend builds all API/WebSocket URLs **same-origin**, so it works at `https://echomind-ajace.com` with **no rebuild**.

## What only you can do (Cloudflare account — auth/account steps)
Claude cannot log into your Cloudflare account or change account settings. Do these in the dashboard:

### Prerequisite — domain on Cloudflare
`echomind-ajace.com` must be a **zone in your Cloudflare account** (registered via Cloudflare Registrar, or its nameservers pointed to Cloudflare and showing **Active**). Dashboard → **Add a site** if it isn't.

### Step 1 — Open Zero Trust (free)
Dashboard → **Zero Trust**. Complete the one-time team setup (free plan covers up to 50 users).

### Step 2 — Create the tunnel and get its token
Zero Trust → **Networks → Tunnels → Create a tunnel** → **Cloudflared** → name it `echomind` → **Save**.
On the install screen, copy the **token** (the long string after `--token`). Put it in your `.env` on the DGX:

```dotenv
# .env  (do NOT commit this file)
TUNNEL_TOKEN=eyJ...your-tunnel-token...
```

### Step 3 — Route the public hostname to the app
In the tunnel's **Public Hostname** tab → **Add a public hostname**:
- **Subdomain:** *(leave blank)* **Domain:** `echomind-ajace.com`
- **Service Type:** `HTTP` **URL:** `frontend:80`

(Cloudflare provides HTTPS at the edge; the internal hop to nginx is plain HTTP on the Docker network. Add a second hostname `www` → same service if you want.)

### Step 4 — Put a login wall in front (MANDATORY)
Zero Trust → **Access → Applications → Add an application → Self-hosted**:
- **Application name:** EchoMind **Domain:** `echomind-ajace.com`
- Add a **policy**: Action **Allow**, and an **Include** rule — e.g. *Emails* = your allow-list, or *Emails ending in* `@ajace.com`, or a Google/GitHub identity provider.
- Save. Now visitors must authenticate before they ever reach EchoMind.

> This is your real authentication layer. Until EchoMind has its own auth, keep Access **on** for every public hostname.

### Step 5 — Abuse protection (recommended)
- **WebSockets:** zone **Network** settings → ensure **WebSockets** is **On** (default) — required for voice/live transcription.
- **Rate limiting:** **Security → WAF → Rate limiting rules** → cap requests per IP to protect the single GPU.
- Keep Cloudflare's proxy (orange cloud) on so your home IP is never exposed and you get DDoS protection.

### Step 6 — Start the tunnel on the DGX
```bash
cd /home/echomind/Documents/echomind/echomind-enterprise
docker compose --profile public up -d cloudflared      # start the tunnel
docker logs -f echomind-cloudflared                     # should show 4 "Registered tunnel connection" lines
```

### Step 7 — Test
Open **https://echomind-ajace.com** → you should hit the **Cloudflare Access login** → after sign-in, the EchoMind UI. Verify Knowledge Chat, then Voice/Live Transcript (mic works because the page is HTTPS).

## Take it down / pause
```bash
docker compose --profile public stop cloudflared        # offline again, instantly
```
Or disable the public hostname / Access app in the dashboard.

## Good to know
- **Upload size:** Cloudflare's **free plan caps request bodies at ~100 MB**; larger document uploads will fail *through the tunnel* (they still work on the LAN). nginx itself allows 200 MB.
- **Capacity:** one DGX Spark serves only a handful of concurrent users — the LLM/STT largely run one job at a time. Treat this as an **invite-gated beta**, not internet scale. Use the Access allow-list + rate limits to keep numbers sane.
- **Tighten CORS later (optional):** with Access in front it's low-risk, but for defense-in-depth you can restrict the backend's CORS from `*` to `https://echomind-ajace.com` once everything works.
- **Local access is unchanged:** LAN/Tailscale access via `:3000`/`:3443` keeps working.
- **Privacy/ToS:** if outside users upload data, add a basic privacy policy and an abuse-handling contact.
