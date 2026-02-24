# Local HTTPS with no browser warning (mkcert)

Use **mkcert** to create a certificate that your machine trusts. The dev server then serves HTTPS that the browser accepts with **no warning**.

---

## 1. Install mkcert

**Linux (Debian/Ubuntu):**
```bash
sudo apt install -y mkcert libnss3-tools
# or: download from https://github.com/FiloSottile/mkcert/releases
mkcert -install
```

**macOS:**
```bash
brew install mkcert nss
mkcert -install
```

**Windows:**  
Download from [mkcert releases](https://github.com/FiloSottile/mkcert/releases) and run `mkcert -install`.

---

## 2. Create certificates for localhost

From the **frontend** directory (or project root):

```bash
cd frontend
mkcert -key-file .cert/key.pem -cert-file .cert/cert.pem localhost 127.0.0.1 ::1
```

Or put certs in a folder of your choice and set env vars in the next step (e.g. `frontend/.cert/` or `./.cert/`).

---

## 3. Run the dev server with trusted HTTPS

Point Vite at the mkcert cert and key:

```bash
cd frontend
VITE_DEV_HTTPS=1 VITE_SSL_CERT=.cert/cert.pem VITE_SSL_KEY=.cert/key.pem npm run dev
```

Or from the project root (paths relative to shell cwd):

```bash
VITE_DEV_HTTPS=1 VITE_SSL_CERT=frontend/.cert/cert.pem VITE_SSL_KEY=frontend/.cert/key.pem npm run dev --workspace=frontend
```

If you use a single `npm run dev` from the repo root, adjust paths accordingly. Example with certs in `frontend/.cert/`:

```bash
cd /path/to/echomind-enterprise
VITE_DEV_HTTPS=1 VITE_SSL_CERT=frontend/.cert/cert.pem VITE_SSL_KEY=frontend/.cert/key.pem npm run dev
```

Open **https://localhost:3000**. The browser should show a trusted lock (no warning).

---

## 4. Add `.cert/` to .gitignore

So the private key is never committed:

```bash
echo ".cert/" >> frontend/.gitignore
# or, if .cert is in repo root:
echo ".cert/" >> .gitignore
```

---

## Summary

| Step | Action |
|------|--------|
| 1 | Install **mkcert** and run **mkcert -install** |
| 2 | From `frontend/`: **mkcert -key-file .cert/key.pem -cert-file .cert/cert.pem localhost 127.0.0.1 ::1** |
| 3 | Run dev server with **VITE_DEV_HTTPS=1** and **VITE_SSL_CERT** / **VITE_SSL_KEY** pointing to those files |
| 4 | Add **.cert/** to **.gitignore** |

Result: **https://localhost:3000** with a trusted certificate and no browser warning.
