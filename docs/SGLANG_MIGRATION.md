# SGLang on DGX Spark (GB10)

EchoMind uses **SGLang** for LLM and **Ollama** for embeddings.

## Preliminary verification (run first)

Before `docker compose up`, verify SGLang works on your DGX Spark:

```bash
./scripts/verify-sglang.sh
```

This follows the [NVIDIA DGX Spark SGLang instructions](https://build.nvidia.com/spark/sglang):
1. Verifies Docker, GPU, nvidia-smi
2. Pulls `lmsysorg/sglang:spark`
3. Launches SGLang with nvidia/Llama-3.1-8B-Instruct-FP8 (mem-fraction 0.75)
4. Tests `/health` and `/generate`

If verification passes, run `docker stop echomind-sglang-verify` then `docker compose up --build`.

## Architecture

- **sglang-llm** (port 30000): Chat/completions via `nvidia/Llama-3.1-8B-Instruct-FP8`. Uses GPU.
- **ollama-embed** (port 11435): Embeddings via `nomic-embed-text`. Runs on CPU.

## GB10 UMA Workaround

DGX Spark (GB10) uses **unified memory (UMA)**. Both `nvidia-smi --query-gpu=memory.total` and `torch.cuda.mem_get_info()` can fail or report incorrectly. We use a patch script (`docker/sglang/launch_gb10.py`) that:

1. Patches `get_nvgpu_memory_capacity` to catch failures
2. Returns a fallback of **100GB** (configurable via `SGLANG_GB10_MEM_MB`)

This allows SGLang to start and allocate memory correctly.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://sglang-llm:30000/v1` | LLM base URL |
| `LLM_MODEL` | `nvidia/Llama-3.1-8B-Instruct-FP8` | Chat model |
| `EMBED_URL` | `http://ollama-embed:11434/api/embeddings` | Embeddings endpoint |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBED_FORMAT` | `ollama` | `ollama` or `openai` |
| `SGLANG_GB10_MEM_MB` | `100000` | Fallback GPU memory (MB) when detection fails |

## Reindexing

If you previously used a different embedding model, clear and reindex the knowledge base (Settings → Knowledge Base → Clear index).

## Troubleshooting

### Multiprocessing spawn error (fixed)

If you see `RuntimeError: ... bootstrapping phase`, the launch script uses `multiprocessing.set_start_method("fork")` to avoid spawn/bootstrap issues on Linux.

### SGLang still fails with OOM

1. **Free GPU memory**: Close other GPU apps (browsers, remote desktop). Check with `nvidia-smi`.
2. Reduce fallback: `SGLANG_GB10_MEM_MB=80000` (80GB)
3. Reduce `--mem-fraction-static` to 0.6–0.7 in the sglang-llm command
4. Try a smaller model: `nvidia/Qwen3-8B-FP8` or `nvidia/Llama-3.1-8B-Instruct-FP8`

### FP4 vs FP8

The `lmsysorg/sglang:spark` image does not support ModelOpt FP4 (`nvidia/Llama-3.1-8B-Instruct-FP4`). Use the FP8 variant (`nvidia/Llama-3.1-8B-Instruct-FP8`) instead.

### Patch not applied

Ensure `./docker/sglang/launch_gb10.py` exists and is mounted. Check container logs for `[sglang-gb10]` messages.
