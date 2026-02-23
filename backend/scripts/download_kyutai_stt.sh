#!/usr/bin/env bash
# Pre-download Kyutai STT model (kyutai/stt-1b-en_fr) to a local folder so the backend
# does not download or re-fetch at runtime. Run once; then set ECHOMIND_KYUTAI_MODEL_DIR
# to the same path so Live Transcript loads from disk only.
set -e

REPO_ID="kyutai/stt-1b-en_fr"

# Target directory: env override, or $ECHOMIND_DATA_DIR/kyutai-stt, or backend/kyutai-stt (user-writable)
if [ -n "${ECHOMIND_KYUTAI_MODEL_DIR}" ]; then
  TARGET_DIR="${ECHOMIND_KYUTAI_MODEL_DIR}"
elif [ -n "${ECHOMIND_DATA_DIR}" ]; then
  TARGET_DIR="${ECHOMIND_DATA_DIR}/kyutai-stt"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  TARGET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/kyutai-stt"
fi

mkdir -p "$TARGET_DIR"
echo "Downloading Kyutai STT to: $TARGET_DIR"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO_ID', local_dir='$TARGET_DIR')
print('Done. Model files are in:', '$TARGET_DIR')
"

echo ""
echo "Kyutai STT model saved to: $TARGET_DIR"
echo "Set this so the backend uses it (no download at runtime):"
echo "  export ECHOMIND_KYUTAI_MODEL_DIR=$TARGET_DIR"
echo ""
echo "In Docker, use: -e ECHOMIND_KYUTAI_MODEL_DIR=$TARGET_DIR"
echo "Or in .env: ECHOMIND_KYUTAI_MODEL_DIR=$TARGET_DIR"
