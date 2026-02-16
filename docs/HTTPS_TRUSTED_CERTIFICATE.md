# Trusted HTTPS (no browser warning) with free domain

This flow removes the browser warning **"This certificate is not trusted"** by using a free subdomain and a **Let's Encrypt** certificate. No need to buy a domain.

**Result:**
- Works fully on HTTPS  
- Free  
- No browser warnings (trusted certificate)

---

## 1. Get a free domain with DuckDNS

1. Go to [DuckDNS](https://www.duckdns.org/) and sign in (e.g. with Google/GitHub).
2. Create a subdomain, e.g. **echomind** → you get **echomind.duckdns.org**.
3. Set the domain to point to your server’s **public IP**:
   - In DuckDNS, set the IP to your machine’s public IP (the one that receives internet traffic).
   - If your app runs on a home or office machine, ensure your router forwards **ports 80 and 443** to that machine (or use a tunnel if you prefer).

---

## 2. Install Nginx and Certbot on the host

On the **same machine** where you run EchoMind (or where you will proxy to it), install Nginx and Certbot. Example for **Ubuntu/Debian**:

```bash
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx
```

---

## 3. Point Nginx at EchoMind (before getting the cert)

Nginx on the host will terminate HTTPS and proxy to the EchoMind frontend. The frontend container serves both the app and proxies `/api/` and `/voice/` to the backend and voice services.

Create a simple HTTP server block so Certbot can respond to the ACME challenge (Certbot will add SSL for you in the next step). Example config:

```bash
sudo nano /etc/nginx/sites-available/echomind
```

Paste (replace `echomind.duckdns.org` if you used a different subdomain):

```nginx
server {
  listen 80;
  server_name echomind.duckdns.org;
  client_max_body_size 200m;

  location / {
    proxy_pass http://127.0.0.1:3000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 86400;
  }
}
```

Enable the site and reload Nginx:

```bash
sudo ln -sf /etc/nginx/sites-available/echomind /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

Ensure EchoMind is running and listening on port 3000:

```bash
docker compose up -d
# Frontend should be on 3000 (HTTP) and 3443 (HTTPS). Host Nginx will use 3000.
```

---

## 4. Get a trusted certificate with Let's Encrypt

Run Certbot with the Nginx plugin; it will obtain a certificate and adjust your Nginx config for HTTPS:

```bash
sudo certbot --nginx -d echomind.duckdns.org
```

- Follow the prompts (email, agree to terms).  
- Certbot will add an HTTPS server block and redirect HTTP → HTTPS if you choose that option.

After this, open **https://echomind.duckdns.org** in your browser. You should see a **trusted** connection with no certificate warning.

---

## 5. (Optional) Auto-renewal

Let's Encrypt certificates expire after about 90 days. Certbot installs a timer/cron for renewal. Test renewal with:

```bash
sudo certbot renew --dry-run
```

---

## Summary

| Step | Action |
|------|--------|
| 1 | Create **echomind.duckdns.org** (or your subdomain) at DuckDNS and set its IP to your server. |
| 2 | Install **nginx** and **certbot** (and **python3-certbot-nginx**) on the host. |
| 3 | Add an Nginx HTTP server block for **echomind.duckdns.org** that proxies to **http://127.0.0.1:3000**. |
| 4 | Run **`sudo certbot --nginx -d echomind.duckdns.org`** to get a trusted certificate and enable HTTPS. |

Result: **HTTPS works with a trusted certificate, no browser warning, and no cost.**
