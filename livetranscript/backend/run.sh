#!/usr/bin/env bash
# Run backend (optionally with NeMo). Install NeMo first for ASR:
# pip install Cython packaging && pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
cd "$(dirname "$0")"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
