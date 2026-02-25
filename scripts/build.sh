#!/usr/bin/env bash
# Build EchoMind images, avoiding Docker BuildKit "file already closed" bug.
# The bug occurs when backend's long Hugging Face download runs in parallel
# with voice finishing; building backend first isolates it.
set -e
cd "$(dirname "$0")/.."

echo "Building backend first (avoids parallel-build pipe bug)..."
docker compose build backend

echo "Building remaining services..."
BUILDX_METADATA_PROVENANCE=disabled docker compose build

echo "Build complete."
if [[ "${1:-}" == "--up" ]]; then
  docker compose up -d
else
  echo "Run: docker compose up -d"
fi
