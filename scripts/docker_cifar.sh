#!/usr/bin/env bash
# docker_cifar.sh — run the Fed-CORE real CIFAR FedOSR pipeline inside a CUDA
# torch container on the 4070. Mirrors the project's Docker-first discipline.
#
# Usage:
#   bash scripts/docker_cifar.sh
#   DATASET=cifar10 DIRICHLET_ALPHA=0.1 SEED=0 bash scripts/docker_cifar.sh
#   IMAGE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime bash scripts/docker_cifar.sh
#
# Adapt IMAGE to whatever torch+torchvision image your repo already uses. The
# script mounts the repo at /workspace and runs experiments/fedcore/run_cifar.py.
set -euo pipefail

# ---- config (override via env) ----------------------------------------------
IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
DATASET="${DATASET:-cifar10}"
N_KNOWN="${N_KNOWN:-6}"
N_CLIENTS="${N_CLIENTS:-5}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.1}"
ROUNDS="${ROUNDS:-50}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
ALPHA="${ALPHA:-0.10}"
DELTA="${DELTA:-0.10}"
NOISE_TYPE="${NOISE_TYPE:-none}"          # none | symmetric | asymmetric
NOISE_RATE="${NOISE_RATE:-0.0}"           # e.g. 0.35 (sym) | 0.20 (asym)
SEED="${SEED:-0}"
OUT="${OUT:-runs/${DATASET}_a${DIRICHLET_ALPHA}_${NOISE_TYPE}${NOISE_RATE}_seed${SEED}_cert.csv}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${REPO_ROOT}/runs" "${REPO_ROOT}/data"

echo "[docker_cifar] image=${IMAGE}"
echo "[docker_cifar] dataset=${DATASET} n_known=${N_KNOWN} clients=${N_CLIENTS} "\
"dir_alpha=${DIRICHLET_ALPHA} rounds=${ROUNDS} alpha=${ALPHA} delta=${DELTA} seed=${SEED}"
echo "[docker_cifar] out=${OUT}"

# ---- run --------------------------------------------------------------------
# --gpus all requires nvidia-container-toolkit. Drop it to run on CPU (slow).
docker run --rm -it \
  --gpus all \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  -e PYTHONUNBUFFERED=1 \
  "${IMAGE}" \
  bash -lc "
    set -e
    pip install --no-cache-dir torchvision scipy numpy >/dev/null 2>&1 || true
    python experiments/fedcore/run_cifar.py \
      --dataset '${DATASET}' \
      --n_known ${N_KNOWN} \
      --n_clients ${N_CLIENTS} \
      --dirichlet_alpha ${DIRICHLET_ALPHA} \
      --rounds ${ROUNDS} \
      --local_epochs ${LOCAL_EPOCHS} \
      --alpha ${ALPHA} \
      --delta ${DELTA} \
      --noise_type '${NOISE_TYPE}' \
      --noise_rate ${NOISE_RATE} \
      --seed ${SEED} \
      --out '${OUT}'
  "

echo "[docker_cifar] done -> ${OUT}"
# Note: runs/ and data/ are gitignored per project policy; do not commit them.
