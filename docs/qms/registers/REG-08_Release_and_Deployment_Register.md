# REG-08 — Release and Deployment Register

| Field | Value |
|---|---|
| Document ID | REG-08 |
| Revision | 1.0 |
| Status | **LIVE REGISTER** |
| Owner | Lead Engineer |
| ISO 9001:2015 clauses | 8.5.1, 8.5.2, 8.6 |
| Governing procedures | SOP-07, SOP-08 |

---

## 1. Purpose

Two things ISO 9001:2015 requires and this project currently cannot do: evidence that a release was
verified and authorised before it was delivered (8.6), and the ability to say which release a given
customer instance is running (8.5.2).

## 2. Releases

| Release ID | Date | Source revision (sha) | Git tag | Image tags / digests | Model versions | Verification (unit / golden eval / manual) | Known issues accepted | Authorised by | Record |
|---|---|---|---|---|---|---|---|---|---|
| `v1.4.0` | 2026-09-22 | `eaa7b1a` | `v1.4.0` (annotated) | Not recorded — no images were built for this release | Chat `nvidia/Qwen3-30B-A3B-FP4`; others per `REG-02` | Unit: **fail** — 4/40 backend, 1/127 voice (run 2026-09-28 on code-identical `b3f3a35`; NC-2026-010) · Golden: **9/52**, corpus absent (NC-2026-008) · Manual: not recorded | NC-2026-008, NC-2026-010 — concession **not yet recorded** | `________` | `records/releases/2026-10-06_v1.4.0.md` (retrospective, prepared 2026-10-06) |

## 3. Deployments

| Instance | Customer | Environment | Release ID deployed | Deployed on | Deployed by | Post-deployment verification | Current |
|---|---|---|---|---|---|---|---|
| `echomind-ajace.com` — public reference instance on the DGX Spark | None — Ajace AI demonstration | Public via Cloudflare Tunnel, application login (`AUTH_ENABLED=1`) | **`v1.4.0` application code** (`eaa7b1a`) for backend and voice, applied by `docker cp` + `docker commit` onto images first built 2026-09-04 (backend `2103b5467c9a`, voice `6e044f7842d0`); frontend is the previously running build re-saved as `17e8946f59fe`, not rebuilt. `GET /api/version` → `1.4.0` / `eaa7b1a` | 2026-10-06 | `________` | 2026-10-06 — see `records/releases/2026-10-06_v1.4.0.md` §7 | Yes |
| `echomind-ajace.com` (previous) | — | Public via Cloudflare Tunnel, **no access control** (NC-2026-009) | Untagged build that predates `v1.4.0` (backend started 2026-09-22T04:55Z, before the tag at 17:53Z); `/api/version` → 404 | Backend and voice 2026-09-22; other services 2026-09-10 | `________` | Not recorded | No — replaced 2026-10-06 |

## 4. Current state

Updated 2026-10-06. The first tag exists (`v1.4.0`, cut with `scripts/release.sh`), `frontend/package.json`
carries `1.4.0`, and `GET /api/version` exists in the code (`backend/app/main.py:213`). Since 2026-10-06
the reference instance runs the `v1.4.0` application code and reports it. Still missing: a from-scratch
image build per release (the instance carries `v1.4.0` code on older base images), and an `FRM-03`
completed and authorised **before** tagging and deploying rather than afterwards.

The original assessment, as at 2026-09-21:

- **No git tags exist** (`git tag` is empty).
- `frontend/package.json` is `"version": "0.0.0"`.
- Backend, voice and frontend images are compose-default and effectively `:latest`.
- **No endpoint reports a build identifier**, so a running instance cannot be asked what it is.

Until the release-identification scheme in SOP-08 §6 is adopted, the *Release ID*, *Git tag* and
*Image tags* columns cannot be filled in honestly. This is Gap Analysis G-01 and quality objective
QO-3. The first entry should be made by tagging the current HEAD as the first identified release.
