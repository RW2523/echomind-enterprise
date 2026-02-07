#!/usr/bin/env bash
# Download a Piper TTS voice from Hugging Face (rhasspy/piper-voices).
# Usage: ./download_voice.sh [voice_id]
# Example: ./download_voice.sh en_US-lessac-medium
# Default: en_US-lessac-medium
set -euo pipefail
VOICE_ID="${1:-en_US-lessac-medium}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICES_DIR="${VOICES_DIR:-$SCRIPT_DIR/../voices}"
mkdir -p "$VOICES_DIR"
cd "$VOICES_DIR"

# Parse voice_id (e.g. en_US-lessac-medium or en_US-libritts_r-medium) -> en_US, speaker, quality
IFS='-' read -ra PARTS <<< "$VOICE_ID"
LOCALE="${PARTS[0]}_${PARTS[1]}"
QUALITY="${PARTS[-1]}"
SPEAKER="${PARTS[2]}"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/${LOCALE}/${SPEAKER}/${QUALITY}"
echo "Downloading Piper voice: $VOICE_ID"
wget -q -O "${VOICE_ID}.onnx" "${BASE}/${VOICE_ID}.onnx"
wget -q -O "${VOICE_ID}.onnx.json" "${BASE}/${VOICE_ID}.onnx.json"
echo "Done: $VOICES_DIR"
ls -lh "${VOICE_ID}.onnx" "${VOICE_ID}.onnx.json"
