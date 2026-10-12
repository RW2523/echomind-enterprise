# EchoMind Law — pilot operations runbook

For the person who runs the pilot on this host. Every command below is run from the clone,
`/home/echomind/Documents/echomind/law/echomind-enterprise`. **Production EchoMind shares this host:** nothing here
touches it, and nothing else should either (see "Never" at the end).

## 1. What runs where

| What | Where |
|---|---|
| The pilot | Docker compose project `echomind-law`: `echomind-law-backend`, `-voice`, `-frontend` |
| Addresses | `https://localhost:3553` on the server; `https://<server address>:3553` on the office network. No public address (THREAT_MODEL T-17) |
| Data | Docker volume `echomind-law_echomind_law_data` |
| Settings | `.env.law` (mode 600, never committed). Feature switches are `LAW_*`; images are pinned in `LAW_*_IMAGE` |
| Master key | `.law-secrets/pilot-master.key` (documents cannot be read without it); a second copy is in `~/.echomind-law-master/` |
| Backup key | `~/.echomind-law/backup.key` (archives cannot be read without it) |
| Backups | `~/echomind-law-backups/` (newest 14 kept); log `backup.log` |
| Regulatory bundles | `~/echomind-law-feeds/` (60 days kept); log `regulatory.log` |
| Daily jobs | `crontab -l`: 02:30 backup (D-067), 06:15 regulatory updates (D-068) |

**Keep a copy of both keys off this host,** apart from each other and from the archives. Losing the master key loses the
documents; losing the backup key loses the backups.

## 2. Is it healthy?

```bash
scripts/law/compose.sh ps
```
All three containers should be `Up`; backend and voice should be `(healthy)`. Then:
- **The end-to-end check:** `python3 scripts/law/demo_check.py` checks every feature with the demo data (about 10 minutes). It uses the real models, so `reference` and `followup` can vary between runs; run a single check again with `python3 scripts/law/demo_check.py reference`.
- **Logs:** `scripts/law/compose.sh logs --tail 200 backend`. Logs hold no matter content.
- **The audit chain:** `docker exec echomind-law-backend python -m app.legal.audit verify` exits 1 if anything was altered.

## 3. Update the pilot

1. Commit the change.
2. Run the deploy script:
   ```bash
   scripts/law/deploy_pilot.sh
   ```
   - It builds the images from the commit, refusing uncommitted changes.
   - It stops the pilot, pins the new tags in `.env.law` and starts it.
   - If the pilot is not healthy within four minutes, it pins the previous tags again.
   - The pilot is down for about a minute.
3. Then check production is unaffected: the `echomind-*` containers keep their start times and restart counts (`docker inspect`).

**Go back to a previous build by hand:**
1. Set `LAW_BACKEND_IMAGE`, `LAW_VOICE_IMAGE` and `LAW_FRONTEND_IMAGE` in `.env.law` to the earlier tag (`docker images | grep echomind-law`).
2. Then run:
   ```bash
   scripts/law/compose.sh up -d backend voice frontend
   ```
- Migrations only move forward on start. A rollback of the code does not undo them; the migration runner refuses rollbacks that would delete kept records (restore a backup instead).

**`compose.sh up` refuses under 12 GB of free memory.** Production shares this host's memory; stop the pilot first (`scripts/law/compose.sh stop`), as the deploy script does.

## 4. Backups and restores

- **Take one now:** `scripts/law/backup_pilot.sh`. It makes the archive inside the backend container, verifies it, copies it out with its sha256 checked, and keeps the newest 14.
- **Practise a restore:** `scripts/law/backup_pilot.sh --restore-test` also restores the new archive into a scratch folder in the container and checks the counts match. Do this monthly.
- **A real restore** replaces the pilot's data. It needs the owner's approval, the backup key and the master key.
  1. `scripts/law/compose.sh stop`.
  2. Restore the archive into a fresh folder with `python -m app.legal.backup restore --archive … --key-file … --target …`.
  3. Put that folder in a new volume, and point the backend at it.
  4. Start the pilot and verify the audit chain.
  5. Practise the whole procedure on a copy first (`scripts/law/dr_drill.sh` is the development drill).
- **Never use `scripts/backup_data.sh` or `scripts/restore_data.sh`:** they act on production.

## 5. Regulatory updates (D-068)

- **Run today's fetch now:** `scripts/law/regulatory_daily.sh`. It fetches on the host and imports into the pilot; the log is in `~/echomind-law-feeds/regulatory.log`. A source that fails is noted and the other is still imported; the next day's run covers three days, so a missed day is caught up.
- **After changing the practice-area terms** (`backend/app/legal/regulatory/topics.py`, with a lawyer):
  1. Deploy.
  2. Run `docker exec echomind-law-backend python -m app.legal.regulatory rematch`. Undecided matches follow the new terms; decided ones are kept.
- **Turn it off:** set `LAW_REGULATORY=0` in `.env.law` and deploy, then remove the 06:15 line with `crontab -e`.

## 6. The network lock (T11, D-071)

```bash
sudo scripts/law/egress_lock.sh
```
- It adds one firewall rule so the Law containers can reach only each other and this host (the model servers), then tests it. It prints PASS or FAIL.
- `--check` tests only; `--remove` takes the rule away.
- **A reboot clears the rule:** run it again after every reboot, or save it with the host's firewall tool.
- If the SIEM forwarder is turned on (D-069) for a SIEM on another machine, that address needs its own exception first.

## 7. People and access

- **The first administrator** signs in through the one-time enrolment link `https://localhost:3553/signin?enrol=<token>`. The token is read with `scripts/law/compose.sh exec backend cat /data/legal/bootstrap_enrollment_token`.
- **New people, a lost authenticator, or someone leaving:** the firm administrator uses **Users & roles**: an enrolment link, a reset of two-step sign-in, sign everywhere out, or disable.
  - Disabling ends every session at once.
  - Matter access is given and taken on each matter's **Members** page.
- **Demo sign-ins** (fictional people only): with one-click demo sign-in on (D-077), the sign-in page lists them, with no password. Otherwise `python3 scripts/law/demo_signin.py`, on the server, shows their passwords and codes; never paste these anywhere else.

## 8. When something goes wrong

| Situation | Do |
|---|---|
| The pilot does not come up after a deploy | The deploy script already went back to the previous build. Read `scripts/law/compose.sh logs --tail 300 backend`; fix, commit, deploy again |
| Production is short of memory or slow | `scripts/law/compose.sh stop`: production comes first. The pilot is capped and is ended first by the host under pressure |
| A live transcript is "behind" | Speech recognition on CPU is overloaded (D-066). Fewer recordings at once, or a GPU for speech. Gaps are marked in the transcript |
| A person's access must end now | **Users & roles**: disable. Their sessions, streams and recordings stop at the next check (seconds) |
| Suspected misuse or tampering | 1. Disable the account.<br>2. Take a backup (`scripts/law/backup_pilot.sh`).<br>3. Run `docker exec echomind-law-backend python -m app.legal.audit verify`.<br>4. Export the audit trail from **Audit trail**.<br>5. Keep the archives and tell the owner. Do not delete anything; legal holds bind every deletion path |
| The backup job failed | `~/echomind-law-backups/backup.log`. Usually the backend was down or the disk was full. Run `scripts/law/backup_pilot.sh` by hand once fixed |

## Never

- Run `docker compose` in this clone except through `scripts/law/compose.sh` (`-p echomind-law`). The project name would otherwise be production's.
- Run `scripts/law/compose.sh down -v`, or remove the `echomind-law_*` volume, without the owner's approval and a verified backup.
- Run `scripts/backup_data.sh` or `scripts/restore_data.sh` from this clone.
- Put real client data into the pilot while it holds the demo world (DEMO_FLOW.md); start from a fresh volume first, and turn
  one-click demo sign-in off (`LAW_DEMO_SIGNIN=0`, then deploy; D-077).
- Print passwords, keys or authenticator secrets into logs, chats or tickets.
