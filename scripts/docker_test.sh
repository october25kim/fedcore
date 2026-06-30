#!/usr/bin/env bash
# Docker-first golden regression suite (structure-only refactor gate).
# Runs the bit-for-bit golden check + the CPU sanity scripts inside the torch container.
# Must stay GREEN before every refactor commit.  Usage: bash scripts/docker_test.sh
set -euo pipefail
IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIP_INSTALL="${PIP_INSTALL:-pip install -q --no-cache-dir scipy scikit-learn}"
echo "[docker_test] golden_check + CPU sanity in ${IMAGE}"
docker run --rm -e CUDA_VISIBLE_DEVICES="" -v "${REPO_ROOT}:/workspace" -w /workspace "${IMAGE}" \
  bash -c "${PIP_INSTALL} && python tests/golden_check.py"
