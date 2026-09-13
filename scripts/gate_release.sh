#!/bin/bash
# Clean-checkout gate: does a fresh extract of the committed tree still
# reproduce the manuscript headline with committed code only?
set -euo pipefail
REF="${1:-HEAD}"
IMG="${FEDCORE_IMAGE:-fedcore-c400r:latest}"
R="$(cd "$(dirname "$0")/.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
git -C "$R" archive "$REF" | tar -x -C "$T"
echo "extracted $REF -> $T"
docker run --rm -v "$T":/w -w /w "$IMG" python paper/wr-v3/scripts/verify_release.py
