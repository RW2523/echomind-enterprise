#!/usr/bin/env bash
# Check the EchoMind Law images of one commit against the manifest written when they were built (2026-11-07 review point 8):
#   scripts/law/verify_images.sh <commit>
# The manifest's own SHA-256 must match, and every tag must still point at the image id it was built as, so a tag that was
# moved or an image replaced after the build is refused. scripts/law/deploy_pilot.sh runs it before using the images.
set -euo pipefail
COMMIT="${1:?usage: $0 <commit>}"
MAN="${LAW_IMAGE_MANIFESTS:-$HOME/.echomind-law-images}"
[[ -f "$MAN/$COMMIT.txt" && -f "$MAN/$COMMIT.txt.sha256" ]] || { echo "no manifest for $COMMIT in $MAN" >&2; exit 2; }
(cd "$MAN" && sha256sum --quiet -c "$COMMIT.txt.sha256") || { echo "the manifest for $COMMIT was changed" >&2; exit 3; }
bad=0
while read -r kind tag id; do
  [[ "$kind" == image ]] || continue
  now="$(docker image inspect -f '{{.Id}}' "$tag" 2>/dev/null || echo missing)"
  if [[ "$now" == "$id" ]]; then echo "ok    $tag"; else echo "WRONG $tag (built as ${id:7:12}, now ${now:7:12})"; bad=1; fi
done < "$MAN/$COMMIT.txt"
exit $bad
