#!/usr/bin/env bash
set -euo pipefail
mkdir -p voices
cd voices
echo "Downloading Piper voice: en_US-lessac-medium"
wget -q -O en_US-lessac-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -q -O en_US-lessac-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
echo "Done."
ls -lh
