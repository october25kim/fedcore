#!/usr/bin/env bash
set -euo pipefail

asset_dir="${1:-.}"
cd "$asset_dir"
sha256sum -c RELEASE_ASSETS.sha256

cat fedcore-pacs-v3-image-fa016c19.tar.gz.part-00 \
    fedcore-pacs-v3-image-fa016c19.tar.gz.part-01 \
    | gzip -t

resnet_count="$(tar -tzf fedcore-pacs-v3-checkpoints-resnet18.tar.gz | grep -c '/latest.pt$')"
convnext_count="$(tar -tzf fedcore-pacs-v3-checkpoints-convnext-tiny.tar.gz | grep -c '/latest.pt$')"

test "$resnet_count" -eq 15
test "$convnext_count" -eq 15

echo "PASS: release hashes and 30 checkpoint members verified"
