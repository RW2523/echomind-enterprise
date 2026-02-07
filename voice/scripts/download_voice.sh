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

# Parse voice_id: locale-speaker-quality (locale e.g. en_US). See https://github.com/rhasspy/piper/blob/master/VOICES.md
IFS='-' read -ra PARTS <<< "$VOICE_ID"
LOCALE="${PARTS[0]}"
QUALITY="${PARTS[-1]}"
SPEAKER="${PARTS[1]}"
for i in $(seq 2 $((${#PARTS[@]} - 2))); do SPEAKER="${SPEAKER}-${PARTS[$i]}"; done
LANG="${LOCALE%%_*}"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/${LANG}/${LOCALE}/${SPEAKER}/${QUALITY}"
echo "Downloading Piper voice: $VOICE_ID"
wget -q -O "${VOICE_ID}.onnx" "${BASE}/${VOICE_ID}.onnx"
wget -q -O "${VOICE_ID}.onnx.json" "${BASE}/${VOICE_ID}.onnx.json"
echo "Done: $VOICES_DIR"
ls -lh "${VOICE_ID}.onnx" "${VOICE_ID}.onnx.json"
