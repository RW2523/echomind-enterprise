# REG-08 — Release and Deployment Register

| Field | Value |
|---|---|
| Document ID | REG-08 |
| Revision | 1.0 |
| Status | **LIVE REGISTER — currently empty** |
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
| | | | | | | | | | `forms/FRM-03` |

## 3. Deployments

| Instance | Customer | Environment | Release ID deployed | Deployed on | Deployed by | Post-deployment verification | Current |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## 4. Current state — this register cannot yet be completed

The columns above are not aspirational padding; they name exactly what is missing. As at 2026-09-21:

- **No git tags exist** (`git tag` is empty).
- `frontend/package.json` is `"version": "0.0.0"`.
- Backend, voice and frontend images are compose-default and effectively `:latest`.
- **No endpoint reports a build identifier**, so a running instance cannot be asked what it is.

Until the release-identification scheme in SOP-08 §6 is adopted, the *Release ID*, *Git tag* and
*Image tags* columns cannot be filled in honestly. This is Gap Analysis G-01 and quality objective
QO-3. The first entry should be made by tagging the current HEAD as the first identified release.
