#!/usr/bin/env bash
# Fed-CORE: multi-GPU grid dispatcher inside ONE CUDA torch container with all
# GPUs visible. Spawns one subprocess per cell (isolated seed/RNG/CUDA context),
# packing JOBS_PER_GPU cells per GPU via CUDA_VISIBLE_DEVICES pinning. Thin wrapper
# around scripts/ws4090/dispatch.py -- all scheduling/guardrails live there.
#
# Usage (env-driven; see scripts/ws4090/RUNBOOK.md for the grids):
#   MANIFEST=scripts/ws4090/manifest_R2.txt JOBS_PER_GPU=3 RESUME=1 bash scripts/docker_grid.sh
#   MANIFEST=scripts/ws4090/manifest_R2.txt DRY_RUN=1 bash scripts/docker_grid.sh
#   MANIFEST=scripts/ws4090/manifest_R6_fedpd.txt JOBS_PER_GPU=2 bash scripts/docker_grid.sh
#
# Never commits runs/ or data/. NO numeric/logic change vs a standalone run --
# each cell is the exact same device-agnostic command from the manifest.
set -euo pipefail

# 2.6.0-cu118 is the image present on the 4xTITAN-RTX box and verified against the
# golden gate (numpy 2.2.2, scipy 1.17.1 -> GOLDEN CHECK PASS). Override with IMAGE=
# to pin the REPRODUCE image (pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime).
IMAGE="${IMAGE:-pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime}"
MANIFEST="${MANIFEST:?set MANIFEST=scripts/ws4090/manifest_R*.txt}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[docker_grid] image=${IMAGE}"
echo "[docker_grid] manifest=${MANIFEST} num_gpus=${NUM_GPUS} jobs_per_gpu=${JOBS_PER_GPU} resume=${RESUME} dry_run=${DRY_RUN}"

# The runtime image ships torch+torchvision but not scipy/scikit-learn, which the
# CP core needs. Pin to the golden-verified versions before dispatching.
PIP_INSTALL="${PIP_INSTALL:-pip install -q --no-cache-dir scipy==1.17.1 scikit-learn==1.9.0 && pip install -q -e .}"

# --gpus all so every GPU is visible; dispatch.py pins each cell to one via
# CUDA_VISIBLE_DEVICES. Env knobs (NUM_GPUS/JOBS_PER_GPU/RESUME/DRY_RUN) are read
# by dispatch.py directly.
docker run --rm --gpus all \
  -e NUM_GPUS="${NUM_GPUS}" \
  -e JOBS_PER_GPU="${JOBS_PER_GPU}" \
  -e RESUME="${RESUME}" \
  -e DRY_RUN="${DRY_RUN}" \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  bash -c "${PIP_INSTALL} && python scripts/ws4090/dispatch.py '${MANIFEST}' ${EXTRA_ARGS:-}"
