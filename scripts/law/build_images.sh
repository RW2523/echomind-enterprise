#!/usr/bin/env bash
# Build the EchoMind Law pilot images (D-060) from one commit, on top of the 1.4.1 images on this host:
#   scripts/law/build_images.sh        # tags echomind-law-{backend,voice,frontend}:<commit> and :latest
# Starts nothing and touches no running container. Needs the network once, for tesseract (apt) and
# cryptography (PyPI); the reranker comes from the host's model cache and the frontend is built here.
# Refuses uncommitted changes, so every image matches a commit. Pin the tags in .env.law (LAW_*_IMAGE).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "refusing: uncommitted changes; an image must match a commit" >&2
  exit 2
fi
COMMIT="$(git rev-parse --short HEAD)"
RERANKER="${LAW_RERANKER_CACHE:-$HOME/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2}"
if [[ ! -d "$RERANKER/snapshots" ]]; then
  echo "reranker model missing from $RERANKER (see scripts/law/dev_backend.sh to fetch it once)" >&2
  exit 4
fi

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
git archive HEAD backend/app voice/app frontend/nginx.law.conf | tar -x -C "$CTX"
(cd frontend && npx vite build --outDir "$CTX/frontend/dist" --emptyOutDir --logLevel warn)

for svc in backend voice frontend; do
  extra=()
  [[ "$svc" == backend ]] && extra=(--build-context "reranker=$RERANKER")
  docker build -f "$ROOT/$svc/Dockerfile.law" "${extra[@]}" --build-arg BUILD_COMMIT="$COMMIT" \
    -t "echomind-law-$svc:$COMMIT" -t "echomind-law-$svc:latest" "$CTX"
done
# The manifest (2026-11-07 review point 8): each image's id and the base image it was built on, with a SHA-256 of the list,
# kept outside the repository; scripts/law/verify_images.sh checks running tags against it before a deploy uses them.
MAN="${LAW_IMAGE_MANIFESTS:-$HOME/.echomind-law-images}"
mkdir -p -m 700 "$MAN"
{
  echo "commit $COMMIT $(git rev-parse HEAD)"
  for svc in backend voice frontend; do
    base="$(sed -n 's/^ARG BASE_IMAGE=//p' "$ROOT/$svc/Dockerfile.law" | head -1)"
    echo "image echomind-law-$svc:$COMMIT $(docker image inspect -f '{{.Id}}' "echomind-law-$svc:$COMMIT")"
    echo "base $base $(docker image inspect -f '{{.Id}}' "$base")"
  done
} > "$MAN/$COMMIT.txt"
(cd "$MAN" && sha256sum "$COMMIT.txt" > "$COMMIT.txt.sha256")
echo "built echomind-law-backend:$COMMIT echomind-law-voice:$COMMIT echomind-law-frontend:$COMMIT (manifest $MAN/$COMMIT.txt)"
