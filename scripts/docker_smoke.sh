#!/usr/bin/env bash
# Docker-first CPU smoke: the torch-free sanity scripts (no GPU, no training).
# Usage: bash scripts/docker_smoke.sh
set -euo pipefail
IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIP_INSTALL="${PIP_INSTALL:-pip install -q --no-cache-dir scipy scikit-learn}"
echo "[docker_smoke] exp_lemma_L + exp_pooling_fail + run_smoke in ${IMAGE}"
docker run --rm -e CUDA_VISIBLE_DEVICES="" -v "${REPO_ROOT}:/workspace" -w /workspace "${IMAGE}" \
  bash -c "${PIP_INSTALL} && cd experiments/fedcore && \
    python exp_lemma_L.py && python exp_pooling_fail.py && python run_smoke.py"
