# EchoMind Legal — incident response

For the firm's operator and the person at Ajace AI who supports the installation. It complements RUNBOOK §8. Names and
contact details are filled in per installation and kept **outside this public repository** (in the firm's own copy).

## Severity

| Level | Examples | First response |
|---|---|---|
| **S1 — data exposed or integrity lost** | a matter's content seen by someone without access; the audit chain does not verify; a lost or copied master key, backup key or archive; ransomware on the host | within 1 hour, around the clock |
| **S2 — security control failed, no known exposure** | the egress lock not holding; a demo instance found with real data; an account that should be disabled still signs in; a failed backup two nights running | within 4 working hours |
| **S3 — service down or degraded** | the pilot does not start; answers fail; speech is behind | within 1 working day |

## Who

| Role | Who (fill in per installation) | Reaches |
|---|---|---|
| Incident lead at the firm | the managing partner named in the order form | everyone below |
| Operator | the person who runs RUNBOOK on the appliance | the incident lead |
| Supplier support | Ajace AI support contact in the support agreement | the operator |
| Data protection | the firm's DPO or privacy counsel | the incident lead |

## First hour (S1)

1. **Contain without destroying evidence.** Disable the account(s) involved (Users & roles); if the cause is unknown, stop
   the pilot (`scripts/law/compose.sh stop`). Do not delete anything: legal holds bind every deletion path.
2. **Preserve.** `scripts/law/backup_pilot.sh` (the backup refuses a broken audit chain: then copy the data volume as it is,
   `docker run --rm -v echomind-law_echomind_law_data:/d -v <safe folder>:/out <backend image> cp -a /d /out/`).
   Export the audit trail (Audit trail page, or `python -m app.legal.audit` export) and keep the backend logs
   (`scripts/law/compose.sh logs --no-color backend > <safe folder>/backend.log`).
3. **Verify.** `docker exec echomind-law-backend python -m app.legal.audit verify`; note where a chain breaks.
4. **Record** a timeline: who noticed what, when, and every action taken, with times in UTC.

## Notify

- The incident lead decides, with data-protection advice, whether the firm must notify a regulator, clients or courts.
  Under the GDPR a personal-data breach is notified to the supervisory authority **within 72 hours** of becoming aware
  of it, unless it is unlikely to result in a risk; other jurisdictions and professional rules may apply. EchoMind does
  not decide this.
- Ajace AI tells the firm without undue delay of any incident in its software or support access that may affect the
  firm's data, with what is known and what is being done.

## Recover and learn

- Restore from a verified archive if data was lost or altered (RUNBOOK §4, `scripts/law/restore_pilot.sh`).
- Rotate what may be exposed: passwords and two-step sign-in (Users & roles), `AUTH_SECRET` (ends every session), the
  backup key (new archives only; keep the old key for old archives). Re-keying the master key needs a maintenance
  window and a tested procedure: not yet automated (see go/no-go report).
- Within two weeks: a written review (cause, impact, what changed), filed with the firm's ISO 9001 records as a
  nonconformity where applicable.
