#!/usr/bin/env bash
# EchoMind Enterprise — release procedure (ISO 9001:2015 clauses 8.5.2 and 8.6).
#
# Creates the release record: a semver-tagged source revision that every built
# image is stamped with, so a running instance can always be traced back to the
# revision it was built from.
#
# This script deliberately does NOT build, push, or deploy anything. It prepares
# and records the release, then prints the exact commands to run. A person
# authorises the release of the product (clause 8.6), not a script.
#
# Usage:  ./scripts/release.sh 1.4.0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

die() { printf 'release: %s\n' "$1" >&2; exit 1; }
note() { printf '  %s\n' "$1"; }

# ── 1. Arguments ──────────────────────────────────────────────────────────────
VERSION="${1:-}"
[ -n "$VERSION" ] || die "usage: ./scripts/release.sh <version>   e.g. ./scripts/release.sh 1.4.0"

# Semver (semver.org): MAJOR.MINOR.PATCH with optional pre-release and build metadata.
SEMVER_RE='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$'
[[ "$VERSION" =~ $SEMVER_RE ]] || die "'$VERSION' is not a valid semver version (expected e.g. 1.4.0 or 1.4.0-rc.1)"

TAG="v$VERSION"

# ── 2. Pre-conditions: a release must be reproducible from the recorded revision ──
command -v git >/dev/null 2>&1 || die "git is not available"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git working tree"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "main" ] || die "releases are cut from 'main' only (currently on '$BRANCH')"

if [ -n "$(git status --porcelain)" ]; then
  git status --short >&2
  die "working tree is dirty — commit or stash the changes above before releasing"
fi

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  die "tag $TAG already exists — releases are immutable; choose a new version"
fi

# ── 3. Derive the build identity ──────────────────────────────────────────────
BUILD_COMMIT="$(git rev-parse HEAD)"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PREV_TAG="$(git describe --tags --abbrev=0 2>/dev/null || true)"

echo "EchoMind Enterprise release $TAG"
note "commit : $BUILD_COMMIT"
note "date   : $BUILD_DATE (UTC)"
note "since  : ${PREV_TAG:-<no previous tag — first release>}"
echo

# ── 4. Record the version in the front-end package manifest ───────────────────
PKG="frontend/package.json"
if [ -f "$PKG" ]; then
  python3 - "$PKG" "$VERSION" <<'PY'
import json, sys
path, version = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as fh:
    raw = fh.read()
data = json.loads(raw)
if data.get("version") == version:
    sys.exit(0)
data["version"] = version
# Preserve key order and the trailing newline convention of the original file.
with open(path, "w", encoding="utf-8") as fh:
    fh.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
PY
  if [ -n "$(git status --porcelain -- "$PKG")" ]; then
    git add "$PKG"
    git commit -q -m "chore(release): $TAG" \
      -m "Set frontend package version to $VERSION for release $TAG."
    BUILD_COMMIT="$(git rev-parse HEAD)"
    echo "Recorded version $VERSION in $PKG (commit ${BUILD_COMMIT:0:12})."
  else
    echo "$PKG already at version $VERSION — nothing to record."
  fi
fi

# ── 5. Regenerate the change log from real git history ────────────────────────
if [ -x scripts/gen_changelog.sh ]; then
  # Pass the version so the commits being released are headed [$VERSION] rather
  # than "Unreleased": the tag does not exist yet at this point.
  ./scripts/gen_changelog.sh "$VERSION" > CHANGELOG.md
  if [ -n "$(git status --porcelain -- CHANGELOG.md)" ]; then
    git add CHANGELOG.md
    git commit -q -m "docs(changelog): regenerate for $TAG"
    BUILD_COMMIT="$(git rev-parse HEAD)"
    echo "Regenerated CHANGELOG.md (commit ${BUILD_COMMIT:0:12})."
  fi
fi

# ── 6. The release record: an annotated tag carrying the evidence ─────────────
if [ -n "$PREV_TAG" ]; then
  RANGE="$PREV_TAG..HEAD"
  RANGE_LABEL="Changes since $PREV_TAG"
else
  RANGE="HEAD"
  RANGE_LABEL="Changes (full history — first tagged release)"
fi
CHANGES="$(git log --no-merges --pretty='  - %h %s' "$RANGE")"
[ -n "$CHANGES" ] || CHANGES="  - (no changes recorded)"

TAG_MSG="EchoMind Enterprise $VERSION

Commit: $BUILD_COMMIT
Date:   $BUILD_DATE
Branch: $BRANCH

$RANGE_LABEL:
$CHANGES

Released under ISO 9001:2015 clause 8.6 — release authorised by the person
running this procedure. Build identity is reported at runtime by
GET /health, GET /api/version (backend), GET /health (voice) and
GET /build.json (front end)."

git tag -a "$TAG" -m "$TAG_MSG"
echo "Created annotated tag $TAG."
echo

# ── 7. Print — do not run — the build, deploy and verification steps ──────────
cat <<EOF
────────────────────────────────────────────────────────────────────────────────
Nothing has been built, pushed or deployed. Run the steps below to authorise and
release this build (ISO 9001:2015 clause 8.6).

1. Build the images with the release identity baked in:

   export BUILD_VERSION=$VERSION
   export BUILD_COMMIT=$BUILD_COMMIT
   export BUILD_DATE=$BUILD_DATE
   docker compose build backend voice frontend

2. Deploy:

   BUILD_VERSION=$VERSION BUILD_COMMIT=$BUILD_COMMIT BUILD_DATE=$BUILD_DATE \\
     docker compose up -d backend voice frontend

3. Verify the deployed build reports this revision (audit evidence):

   docker exec echomind-backend python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/api/version').read().decode())"
   docker exec echomind-backend python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
   docker exec echomind-voice   python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
   docker exec echomind-frontend cat /usr/share/nginx/html/build.json

   Each must report version $VERSION and commit $BUILD_COMMIT.

4. Publish the release record:

   git push origin main
   git push origin $TAG

To abandon this release before pushing:  git tag -d $TAG
────────────────────────────────────────────────────────────────────────────────
EOF
