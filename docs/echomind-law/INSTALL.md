# EchoMind Legal — first installation

For the person installing EchoMind Legal on a firm's appliance (one DGX Spark). Day-to-day operation is RUNBOOK.md;
incidents are INCIDENT_RESPONSE.md. Every step is run from the clone. Nothing here touches another stack on the host.

## 1. Before you start

| Need | Check |
|---|---|
| Full-disk encryption on the disk that holds Docker's data | `scripts/law/check_disk_encryption.sh` prints "yes". The database holds extracted document text; only the original files and backups are encrypted by EchoMind itself. Without it, the owner must accept the risk in writing (`LAW_ALLOW_UNENCRYPTED_DISK=1`). |
| The answer model and embeddings served on this host | `curl -s localhost:8355/v1/models` and `curl -s localhost:11435/api/tags` answer. The session then reports "Answered on this server". |
| Docker, at least 12 GB of free memory for the stack | `docker info`, `free -g` |
| A TLS certificate the firm's browsers trust, for the name people will use | the firm's IT. The image ships a self-signed certificate for `localhost` only. |
| A place off this host for backups | a mounted disk or share (`LAW_BACKUP_OFFHOST_DIR`) |

## 2. Secrets and settings

```bash
cp .env.law.example .env.law && chmod 600 .env.law
install -d -m 700 .law-secrets
install -m 600 /dev/null .law-secrets/pilot-master.key && openssl rand -base64 32 > .law-secrets/pilot-master.key
openssl rand -hex 32        # → LAW_AUTH_SECRET in .env.law
```
- Set `LAW_AUTH_SECRET`, `LAW_ADMIN_PASSWORD` (the password policy applies), `LAW_CORS_ORIGINS` (the HTTPS address
  people will use) and the feature switches in `.env.law`.
- Keep `LAW_PROFILE=production` and `LAW_DEMO_SIGNIN=0`. The backend refuses to start with demo sign-in outside a demo
  instance.
- Copy the master key off this host now, apart from everything else: without it the documents cannot be opened.

## 3. Build and start

```bash
LAW_DEPLOY_WITHOUT_BACKUP=1 scripts/law/deploy_pilot.sh      # first install only: there is nothing to back up yet
```
The backend checks its settings at start (`core/edition.py`) and refuses anything insecure, saying which setting.

## 4. Lock the network

```bash
sudo scripts/law/egress_lock.sh                  # prints "T11 enforced: PASS"
sudo apt install iptables-persistent && sudo netfilter-persistent save
```
Later deployments refuse to run while the lock does not hold.

## 5. Certificate

Replace the self-signed certificate with the firm's: mount the firm's `cert.pem` and `key.pem` over `/etc/nginx/ssl/` of
the frontend (a compose override kept with `.env.law`), and add the firm's address to `LAW_CORS_ORIGINS`. Once browsers
trust it, HTTPS is enforced for a year (HSTS). Renew before it expires; the self-signed one lasts 365 days from the build.

## 6. Backups

```bash
scripts/law/backup_pilot.sh --keygen             # once; copy ~/.echomind-law/backup.key off this host, apart from the master key
scripts/law/backup_pilot.sh --restore-test       # a first archive, restored into a scratch folder and checked
crontab -e                                        # the two lines in RUNBOOK §1, with LAW_BACKUP_OFFHOST_DIR set
```

## 7. First administrator and people

1. Read the enrolment token: `scripts/law/compose.sh exec backend cat /data/legal/bootstrap_enrollment_token`.
2. Open `https://<address>/signin?enrol=<token>`, sign in as the administrator and set up two-step sign-in.
3. Create accounts in **Users & roles**. The first managing partner may be appointed there; every further one is
   appointed by a managing partner (two of them approve walls and emergency access).
4. Create clients and matters; each matter's lead adds its members.

## 8. Before real client data goes in

- [ ] `check_disk_encryption.sh` says yes, or the owner's written acceptance is on file.
- [ ] `egress_lock.sh --check` passes.
- [ ] The firm's certificate is in place; plain HTTP is only on this server (`LAW_HTTP_BIND=127.0.0.1`).
- [ ] A backup was made, copied off the host and restored in a test.
- [ ] `docker exec echomind-law-backend python -m app.legal.audit verify` passes.
- [ ] The release's evaluation gate passed (`cd backend && python3 ../eval/law/run_gate.py`), with thresholds signed off.
- [ ] Demo sign-in is off and no demo data is in this installation (start from a fresh volume if it ever held the demo).
