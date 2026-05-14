#!/usr/bin/env bash
# Install NeMo ASR in NGC PyTorch images without replacing the prebuilt CUDA torch stack.
set -euo pipefail

NEMO_REF="${NEMO_GIT_REF:-v2.2.1}"
PIP_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-1000}"
PIP_RETRIES="${PIP_RETRIES:-10}"

python3 - <<'PY' > /tmp/ngc-pytorch-constraints.txt
import importlib.metadata as m

seen: set[str] = set()
for dist in m.distributions():
    name = (dist.metadata.get("Name") or "").strip()
    version = (dist.metadata.get("Version") or "").strip()
    if not name or not version:
        continue
    key = name.lower().replace("_", "-")
    if key in {"torch", "torchvision", "torchaudio", "triton", "pytorch-triton"} or key.startswith("nvidia-"):
        line = f"{name}=={version}"
        if line not in seen:
            seen.add(line)
            print(line)
PY

pip install --default-timeout="${PIP_TIMEOUT}" --retries="${PIP_RETRIES}" --no-cache-dir \
  Cython packaging sentencepiece

pip install --default-timeout="${PIP_TIMEOUT}" --retries="${PIP_RETRIES}" --no-cache-dir \
  -c /tmp/ngc-pytorch-constraints.txt \
  "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@${NEMO_REF}"
