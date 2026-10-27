# 08 — NVIDIA DGX Spark deployment playbook
## Hardware realities
DGX Spark GB10 Grace Blackwell uses an ARM CPU and coherent CPU/GPU unified memory, **not** a typical x86 CUDA workstation. NVIDIA's published configurations include 64 GB and 128 GB; verify the actual appliance. Reference: https://www.nvidia.com/en-us/products/workstations/dgx-spark/ and https://docs.nvidia.com/dgx/dgx-spark/hardware.html. On a 128 GB system, model weights, KV cache, CPU processes, document processing, OS and graphical services share memory. Published maximum model sizes do not guarantee concurrent real-time performance.

## First commands (run on device; adapt to installed NVIDIA stack)
```bash
uname -m                  # expect aarch64
cat /etc/os-release
nvidia-smi               # available on NVIDIA software stack; record full output
free -h
df -h
lsblk
sudo docker version
sudo docker info --format '{{.Architecture}}'
ss -tulpen              # check exposed listeners, requires permissions
```
Pin exact container digests after validating architecture (`linux/arm64`), CUDA compatibility, TensorRT-LLM wheels/images, drivers, Python ABI, and model quantization artifacts. Never presume an x86 wheel runs on Spark. Use existing stable EchoMind TensorRT serving if proven. Try an offline smoke test before adopting new inference runtime.

## Scheduling priorities
P0 active audio transcription + safety critical control actions; P1 interactive chat; P2 retrieval/embedding for current user; P3 queued large draft; P4 council/graph extract/batch meeting ASR/image generation. Enforce admission controls, timeouts, maximum concurrent generation slots, memory headroom and cancellation. Do **not** assume simultaneous Qwen3-30B + VibeVoice + SDXL + another reasoning model will be acceptable.

## Recommended process layout
- Local reverse proxy + React static UI
- FastAPI application (auth, matter APIs, streaming orchestrator)
- PostgreSQL (+ pgvector if evaluated/compatible ARM build) + optionally local lexical FTS
- File/object storage with encrypted volume and host file permissions
- Existing optimized LLM service; one heavy LLM resident by default
- Single job worker with GPU workload semaphore; scalable pool for CPU parsing
- Optional speech process only while explicitly authorized sessions active
- Watchdog, healthcheck, metrics, encrypted backup agent.

## Deployment sequence
1. Document current OS/drivers/available RAM/SSD free space and measured existing EchoMind baseline.
2. Prepare encrypted data mount and distinct non-root service accounts; local TLS via trusted firm CA/device enrollment.
3. Build/preload ARM64 images and model weights; verify all required dependencies are cached for offline installs.
4. Configure network isolation: deny outbound by default and selectively allow known LAN clients; ensure no public exposure or telemetry.
5. Deploy databases and perform migrations/backups; check restore into disposable test instance.
6. Deploy API, workers and model gateway; run `/healthz`, `/readyz`, access-control integration tests.
7. Deploy UI with locally bundled assets and Content Security Policy; no remote requests.
8. Ingest tiny synthetic matter; run RAG/citation/ACL/demo sequence offline.
9. Benchmark chat, document ingestion, STT and combined load; capture GPU+CPU+RAM+thermal metrics.
10. Only then add complex council/GraphRAG workloads.

## Resource monitoring and service policy
Track RAM available, committed CPU/GPU memory, inference TPS, time-to-first-token, prefill/decode time, GPU activity, disk IOPS, queue wait, STT lag, temperature and model swapping costs. Establish **measured** per-mode budgets with 20–25% desired safety margin (tune based on tests). Preempt/queue low priority background work. Do not present guessed 0.6s voice latency as guaranteed on this deployment.

## Offline operation
Pre-download/pin apt/container deps, local pip/npm mirrors or vendored dependencies, pre-packaged models/license, local DNS/time and local CA. Install scripts must fail clearly without network. Test by blocking WAN at firewall and running all MVP journeys. Office Wi-Fi hotspot only when secured by WPA and site policy; do not default to broadcasting a hotspot in a law office.

## Backup and disaster recovery
Encrypted backups must include DB, originals, embeddings/index rebuild plan, graph, drafts, secrets recovery procedure (separate secure escrow), permissions and audit records. Encrypt backup destinations, use immutable/off-device rotation where allowed, test **restoring and opening a source citation** on a clean instance. Decide retention and legal hold with firm's counsel. Backups not operational in the supplied v1.4.1: **production blocker**.

## Benchmark matrix
single chat; multi-turn legal chat; two simultaneous users; 500-page PDF import while chat active; live STT while Qwen responds; transcript post-processing; 3-agent council sequential; index rebuild; model cold start; power interruption recovery. Record P50/P95, throughput, memory and failure modes; retain script and fixture hashes.
