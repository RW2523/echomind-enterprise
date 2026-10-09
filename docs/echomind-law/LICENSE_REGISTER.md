# EchoMind Law — Model, Dataset & Dependency Licence Register

> **Nothing in this register is a licence clearance.** "Reported licence" means what the publisher states as Claude understands it at audit time. Each row must be checked against the actual model card or licence file of the exact version deployed, by a person responsible for licensing, before the pilot. No dataset has been downloaded.

## 1. Models currently used by EchoMind

| Model | Used for | Where | Reported licence (verify) | Concern for Law pilot |
|---|---|---|---|---|
| `nvidia/Qwen3-30B-A3B-FP4` (NVFP4 quantization of Qwen3-30B-A3B) | Chat LLM | trtllm | Base Qwen3 weights: Apache-2.0. **NVIDIA quantized checkpoint may carry NVIDIA terms. Verify.** | Verify redistribution terms for on-prem delivery |
| `qwen2.5:7b-instruct-q4_K_M` | Fallback LLM / eval judge | ollama | Qwen2.5-7B: Apache-2.0 (verify) | — |
| `nomic-embed-text` | Embeddings | ollama | Apache-2.0 (verify) | — |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker | backend | Apache-2.0 (verify); trained on MS MARCO, **whose terms restrict commercial use of the dataset. Verify whether that affects the model.** | Legal review recommended |
| `nvidia/nemotron-speech-streaming-en-0.6b` | Live ASR | backend, voice | NVIDIA model licence (verify exact terms) | Verify on-prem commercial use |
| `nvidia/parakeet-tdt-0.6b-v2` | Final STT (voice) | voice | CC-BY-4.0 (verify) | Attribution required |
| VibeVoice-ASR-HF | Boardroom diarized ASR | backend `/data/hf_cache` | **Unknown. Verify.** | Must be resolved before MEET-01 pilot |
| Piper TTS (`piper-tts>=1.4.2`, the `piper1-gpl` fork) + voices amy/joe/john/kusal/lessac | TTS | voice | Engine: GPL-3.0 (reported); **each voice has its own licence (verify per voice)** | GPL distribution implications; consider Kokoro-only for Law |
| Kokoro-82M | TTS | voice | Apache-2.0 (verify) | — |
| `stabilityai/sdxl-turbo` | Document Studio images | backend | Stability AI licence. **Historically non-commercial research; terms have changed over time. Verify current terms.** | **Proposed: disabled in Law edition** (images not needed for legal documents) |
| Tesseract 5 + `eng` traineddata (proposed, D-010) | OCR | backend | Apache-2.0 (verify) | — |

## 2. New dependencies proposed by the plan (to approve before install)

| Package | Purpose | Reported licence (verify) | Slice |
|---|---|---|---|
| `react-router` v7 | Law workspace routing | MIT | S1.3 |
| `@tanstack/react-query` | Server state, matter-keyed caches | MIT | S1.3 |
| `lucide-react` | Icons | ISC | S1.3 |
| `@fontsource-variable/inter` | UI font (self-hosted) | OFL-1.1 (font), MIT (package) | S1.3 |
| `@fontsource-variable/source-serif-4` | Document font (self-hosted) | OFL-1.1 / MIT | S1.3 |
| `qrcode` | Local MFA enrolment QR | MIT | S1.1 |
| `pdfjs-dist` | In-app document viewer with highlights | Apache-2.0 | S1.6/S2.2 |
| `vitest`, `@testing-library/react`, `jsdom`, `@playwright/test`, `axe-core` | Tests (dev only) | MIT / MIT / MIT / Apache-2.0 / MPL-2.0 | S1.0/S1.3 |
| `cryptography` (Python) | AES-GCM blobs, key wrapping | Apache-2.0 / BSD | S1.6 |
| `pytesseract` (Python) | OCR wrapper | Apache-2.0 | S2.1 |
| `python-docx` upgrade 1.1.2 → 1.2.x | Comments API for redlines | MIT | S3.3 |

## 3. Candidate evaluation datasets (spec §10). **None downloaded or integrated.**

| Dataset | Intended use | Reported licence / terms (verify) | Status |
|---|---|---|---|
| CUAD (Contract Understanding Atticus Dataset) | Clause extraction eval | CC BY 4.0 (verify) | Not downloaded; needs approval |
| LegalBench | Legal reasoning tasks | **Per-task licences vary** | Only tasks with compatible licences, after review |
| AMI Meeting Corpus | Diarization / WER eval | CC BY 4.0 (verify) | Not downloaded |
| TREC 2010 Legal Track (privilege) | Privilege classification eval | **Usage agreement required; terms need review** | Not used |
| CourtListener / Caselaw Access Project | Citation-format fixtures | Varies by source; verify | Not used; P-009 governs any authority use |
| SEC EDGAR filings | Public contracts / exhibits | Public records (verify any terms of access) | Not downloaded |
| Public court recordings | ASR eval | Varies by court; verify | Not used |

**Default:** the Law test suite uses **synthetic, fictional fixtures written for this project** (`eval/fixtures/law/`, `backend/tests/law/fixtures/`), with a `PROVENANCE.md` per folder. External datasets are added only after a row here is marked verified and the owner approves.

## 4. Review log

| Date | Item | Reviewer | Result |
|---|---|---|---|
| 2026-10-08 | `react-router` 7, `@tanstack/react-query`, `lucide-react`, `@fontsource-variable/inter`, `@fontsource-variable/source-serif-4`, `qrcode` (S1.1–S1.3) | Claude (from package metadata) | MIT / ISC / OFL-1.1 as listed in §2; added when the owner approved starting S1.3 |
| 2026-10-08 | `cryptography` `>=44,<47` (S1.6; 46.0.3 on the dev host) | Claude (from package metadata) | Apache-2.0 OR BSD-3-Clause; added under the standing approval; offline, no network use |
