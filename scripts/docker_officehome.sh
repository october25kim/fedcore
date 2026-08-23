#!/usr/bin/env bash
# Fed-CORE Office-Home arm: train ONE immutable cell inside the pinned CUDA
# torch container. Env-driven so the 50-cell scheduler can setsid-detach it.
#
#   SPLIT_ID=officehome_split_0 TRAIN_REP=0 PIPELINE=A GPU=1 bash scripts/docker_officehome.sh
#
# GPU policy: GPU 0 is EXCLUDED. Use physical GPUs 1, 2, or 3 with
# CUDA_DEVICE_ORDER=PCI_BUS_ID. Never commits runs/ or data/.
set -euo pipefail

IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
SPLIT_ID="${SPLIT_ID:-officehome_split_0}"
TRAIN_REP="${TRAIN_REP:-0}"
PIPELINE="${PIPELINE:-A}"
GPU="${GPU:-1}"
ROUNDS="${ROUNDS:-30}"

if [[ "${GPU}" == "0" ]]; then
  echo "[docker_officehome] refusing GPU 0 (excluded); use 1, 2, or 3" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${MANIFEST:-results/officehome/dedup/retained_canonical_manifest.csv}"
FOLDS="${FOLDS:-results/officehome/folds/folds_${SPLIT_ID}.csv}"
CLASS_SPLITS="${CLASS_SPLITS:-results/officehome/preflight/class_splits.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-data/officehome/OfficeHomeDataset}"
TORCH_CACHE="${TORCH_CACHE:-/workspace/data/officehome/torch_cache}"

PIP_INSTALL="${PIP_INSTALL:-pip install -q --no-cache-dir scipy && pip install -q -e .}"

echo "[docker_officehome] image=${IMAGE} split=${SPLIT_ID} rep=${TRAIN_REP} pipeline=${PIPELINE} gpu=${GPU} rounds=${ROUNDS}"

docker run --rm --gpus "\"device=${GPU}\"" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e TORCH_HOME="${TORCH_CACHE}" \
  -e PYTHONUNBUFFERED=1 \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  bash -c "${PIP_INSTALL} && python -m fedcore.experiments.run_officehome \
    --manifest-csv '${MANIFEST}' \
    --folds-csv '${FOLDS}' \
    --class-splits-csv '${CLASS_SPLITS}' \
    --split-id '${SPLIT_ID}' \
    --image-root '${IMAGE_ROOT}' \
    --pipeline '${PIPELINE}' \
    --train-rep '${TRAIN_REP}' \
    --rounds '${ROUNDS}' ${EXTRA_ARGS:-}"
