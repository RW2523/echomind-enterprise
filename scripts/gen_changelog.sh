#!/usr/bin/env bash
# EchoMind Enterprise — generate CHANGELOG.md from real git history.
#
# Output is Keep-a-Changelog-style Markdown written to stdout, so the change log
# is always derived from the recorded history rather than maintained by hand
# (ISO 9001:2015 clause 7.5.3 — controlled, trustworthy documented information).
#
# Commits are grouped per tag range (newest first) and, within a range, by
# conventional-commit prefix. Anything that does not carry a recognised prefix is
# listed under "Other changes" rather than being dropped or reworded.
#
# Usage:  ./scripts/gen_changelog.sh [version] > CHANGELOG.md
#
# With no argument, commits beyond the newest tag are listed as "Unreleased".
# With a version (as scripts/release.sh passes when cutting a release), those
# same commits are headed "[version] - <date of HEAD>" instead, because the tag
# does not exist yet at the moment the change log has to be written into it.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PENDING_VERSION="${1:-}"

# Conventional-commit prefix -> Keep-a-Changelog-ish heading.
section_for() {
  case "$1" in
    feat)   echo "Added" ;;
    fix)    echo "Fixed" ;;
    perf)   echo "Performance" ;;
    docs)   echo "Documentation" ;;
    build)  echo "Build" ;;
    revert) echo "Reverted" ;;
    *)      echo "Other changes" ;;
  esac
}

# Order sections are printed in; "Other changes" always comes last.
SECTION_ORDER=("Added" "Fixed" "Performance" "Documentation" "Build" "Reverted" "Other changes")

# Conventional-commit subject: type(scope)!: description. Held in a variable
# because bash will not parse this pattern inline inside [[ =~ ]].
CC_RE='^([a-zA-Z]+)(\(([^)]*)\))?!?:[[:space:]]*(.*)$'

# Emit one release section for the commits in a git range.
#   $1 = heading (e.g. "[1.4.0]" or "Unreleased")
#   $2 = date suffix (e.g. " - 2026-09-22", or empty)
#   $3 = git range (e.g. "v1.3.0..v1.4.0" or "HEAD")
emit_range() {
  local heading="$1" datestr="$2" range="$3"
  local -A buckets=()
  local line sha subject prefix scope desc section entry

  while IFS=$'\x1f' read -r sha subject; do
    [ -n "$sha" ] || continue
    prefix=""; scope=""; desc="$subject"
    # type(scope)!: description  /  type!: description  /  type: description
    if [[ "$subject" =~ $CC_RE ]]; then
      prefix="$(printf '%s' "${BASH_REMATCH[1]}" | tr '[:upper:]' '[:lower:]')"
      scope="${BASH_REMATCH[3]}"
      desc="${BASH_REMATCH[4]}"
    fi
    section="$(section_for "$prefix")"
    # An unrecognised prefix is not a real conventional commit: keep the subject verbatim.
    if [ "$section" = "Other changes" ] && [ -n "$prefix" ]; then
      case "$prefix" in
        feat|fix|perf|docs|build|revert) ;;
        *) desc="$subject"; scope="" ;;
      esac
    fi
    if [ -n "$scope" ]; then
      entry="- **${scope}**: ${desc} (\`${sha}\`)"
    else
      entry="- ${desc} (\`${sha}\`)"
    fi
    buckets["$section"]+="${entry}"$'\n'
  done < <(git log --no-merges --date=short --pretty=format:'%h%x1f%s' "$range")

  # A range with no commits contributes nothing.
  local any=0
  for section in "${SECTION_ORDER[@]}"; do
    [ -n "${buckets[$section]:-}" ] && any=1
  done
  [ "$any" -eq 1 ] || return 0

  printf '## %s%s\n\n' "$heading" "$datestr"
  for section in "${SECTION_ORDER[@]}"; do
    [ -n "${buckets[$section]:-}" ] || continue
    printf '### %s\n\n' "$section"
    printf '%s\n' "${buckets[$section]}"
  done
}

# Root commit date (avoids piping git log into head, which trips pipefail on SIGPIPE).
FIRST_COMMIT_DATE="$(git log --max-parents=0 -1 --date=short --pretty=format:'%ad')"
LAST_COMMIT_DATE="$(git log -1 --date=short --pretty=format:'%ad')"

cat <<EOF
# Changelog

All notable changes to EchoMind Enterprise are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

This file is generated from the git history by \`scripts/gen_changelog.sh\`; do
not edit it by hand. Versions are cut by \`scripts/release.sh\`, which tags the
source revision that each released image is built from (ISO 9001:2015, 8.5.2).

History covered: ${FIRST_COMMIT_DATE} to ${LAST_COMMIT_DATE}.

EOF

# Tags newest first. Empty when the repository has no tags yet.
mapfile -t TAGS < <(git tag --list --sort=-creatordate 'v*')
if [ "${#TAGS[@]}" -eq 0 ]; then
  mapfile -t TAGS < <(git tag --list --sort=-creatordate)
fi

# Heading for the commits that are not yet covered by a tag.
if [ -n "$PENDING_VERSION" ]; then
  PENDING_HEADING="[${PENDING_VERSION}]"
  PENDING_DATE=" - $(git log -1 --date=short --pretty=format:'%ad')"
else
  PENDING_HEADING="Unreleased"
  PENDING_DATE=""
fi

if [ "${#TAGS[@]}" -eq 0 ]; then
  # No releases have been cut yet: everything is unreleased.
  emit_range "$PENDING_HEADING" "$PENDING_DATE" "HEAD"
else
  # Anything on HEAD beyond the newest tag.
  emit_range "$PENDING_HEADING" "$PENDING_DATE" "${TAGS[0]}..HEAD"
  for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"
    prev="${TAGS[$((i + 1))]:-}"
    tag_date="$(git log -1 --date=short --pretty=format:'%ad' "$tag")"
    if [ -n "$prev" ]; then
      emit_range "[${tag#v}]" " - ${tag_date}" "${prev}..${tag}"
    else
      emit_range "[${tag#v}]" " - ${tag_date}" "$tag"
    fi
  done
fi
