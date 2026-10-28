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
| Daily jobs | `crontab -l` must show both lines below (D-067, D-068). **On 2026-10-28 there was no crontab: the last archive was 9 days old (review R2). Put them back:** |

```cron
MAILTO=<the operator's address>
30 2 * * * cd <clone> && scripts/law/backup_pilot.sh >> ~/echomind-law-backups/backup.log 2>&1
15 6 * * * cd <clone> && scripts/law/regulatory_daily.sh >> ~/echomind-law-feeds/regulatory.log 2>&1
```
Each backup appends one line (`OK …` or `FAILED …`) to `~/echomind-law-backups/status.log`; a failure exits non-zero, so
cron mails it.

**Keep a copy of both keys off this host,** apart from each other and from the archives. Losing the master key loses the
documents; losing the backup key loses the backups.

**Recovery targets** (proposed by engineering, for the owner to confirm with the firm; review R2):

| Target | Value | How |
|---|---|---|
| RPO — most data that may be lost | 24 hours | the 02:30 backup; set `LAW_BACKUP_OFFHOST_DIR` so a copy also leaves this host |
| RTO — longest the pilot may be down | 4 hours | `scripts/law/restore_pilot.sh` onto this host (about 15 minutes for the pilot's size), or onto a replacement DGX Spark with the two keys |

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
   - It refuses an unencrypted data disk unless `LAW_ALLOW_UNENCRYPTED_DISK=1` records the owner's acceptance
     (`scripts/law/check_disk_encryption.sh`), and an unlocked network unless `LAW_ALLOW_OPEN_EGRESS=1` (§6).
   - It builds the images from the commit, refusing uncommitted changes.
   - **It takes a verified backup and notes the database's migration head first** (review R3).
   - It stops the pilot, pins the new tags in `.env.law` and starts it.
   - If the pilot is not healthy within four minutes, it reverts the migrations the new version applied (with the new
     image's down steps) and pins the previous tags again. If the revert is refused because it would delete records kept
     since, it stops and names the backup to restore (§4); it never restores by itself.
   - The pilot is down for about a minute.
3. Then check production is unaffected: the `echomind-*` containers keep their start times and restart counts (`docker inspect`).

**Go back to a previous build by hand:**
1. Set `LAW_BACKEND_IMAGE`, `LAW_VOICE_IMAGE` and `LAW_FRONTEND_IMAGE` in `.env.law` to the earlier tag (`docker images | grep echomind-law`).
2. Then run:
   ```bash
   scripts/law/compose.sh up -d backend voice frontend
   ```
- Migrations move forward on start, and the old code refuses a newer database. Before starting an older build, revert
  with the **newer** image: `docker exec echomind-law-backend python -m app.legal.migrations down --to <old head>`. The
  runner refuses a revert that would delete kept records (audit trail, holds, decisions, conversations): restore the
  pre-deploy backup instead (§4).

**`compose.sh up` refuses under 12 GB of free memory.** Production shares this host's memory; stop the pilot first (`scripts/law/compose.sh stop`), as the deploy script does.

## 4. Backups and restores

- **Take one now:** `scripts/law/backup_pilot.sh`. It makes the archive inside the backend container, verifies it, copies it out with its sha256 checked, and keeps the newest 14.
- **Practise a restore:** `scripts/law/backup_pilot.sh --restore-test` also restores the new archive into a scratch folder in the container and checks the counts match. Do this monthly.
- **A real restore** replaces the pilot's data. It needs the owner's approval, the backup key and the master key the
  archive was made with.
  ```bash
  scripts/law/restore_pilot.sh ~/echomind-law-backups/law-pilot-<time>.emlb          # checks the archive, changes nothing
  scripts/law/restore_pilot.sh ~/echomind-law-backups/law-pilot-<time>.emlb --yes    # does it
  ```
  It checks the archive, stops the pilot, keeps the current data in a volume `echomind-law_prerestore_<time>`, restores
  the archive into the data volume, starts the pilot and verifies the audit chain. Remove the kept volume only once the
  restored pilot is confirmed. Its steps were drilled on scratch volumes with a demo archive on 2026-10-28 (audit chain
  verified, 728 events).
- **Backups verify the audit chain first** and fail if it is broken (review R7): treat that as §8 "tampering".
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
- **A reboot clears the rule:** run it again after every reboot, or save it with the host's firewall tool
  (`sudo apt install iptables-persistent && sudo netfilter-persistent save`). The deploy script checks it before and after.
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
| The pilot does not come up after a deploy | The deploy script went back to the previous build, or stopped and named the backup to restore (§3). Read `scripts/law/compose.sh logs --tail 300 backend`; fix, commit, deploy again |
| Production is short of memory or slow | `scripts/law/compose.sh stop`: production comes first. The pilot is capped and is ended first by the host under pressure |
| A live transcript is "behind" | Speech recognition on CPU is overloaded (D-066). Fewer recordings at once, or a GPU for speech. Gaps are marked in the transcript |
| A person's access must end now | **Users & roles**: disable. Their sessions, streams and recordings stop at the next check (seconds) |
| Suspected misuse or tampering | 1. Disable the account.<br>2. Take a backup (`scripts/law/backup_pilot.sh`).<br>3. Run `docker exec echomind-law-backend python -m app.legal.audit verify`.<br>4. Export the audit trail from **Audit trail**.<br>5. Keep the archives and tell the owner. Do not delete anything; legal holds bind every deletion path |
| The backup job failed | Last line of `~/echomind-law-backups/status.log`, then `backup.log`. Usually the backend was down or the disk was full; "audit chain broken" is the row above. Run `scripts/law/backup_pilot.sh` by hand once fixed |
| A security incident, a breach, or a client's data seen by the wrong person | `docs/echomind-law/INCIDENT_RESPONSE.md` |

## Never

- Run `docker compose` in this clone except through `scripts/law/compose.sh` (`-p echomind-law`). The project name would otherwise be production's.
- Run `scripts/law/compose.sh down -v`, or remove the `echomind-law_*` volume, without the owner's approval and a verified backup.
- Run `scripts/backup_data.sh` or `scripts/restore_data.sh` from this clone.
- Put real client data into the pilot while it holds the demo world (DEMO_FLOW.md); start from a fresh volume first, and turn
  one-click demo sign-in off (`LAW_DEMO_SIGNIN=0`, then deploy; D-077).
- Print passwords, keys or authenticator secrets into logs, chats or tickets.
