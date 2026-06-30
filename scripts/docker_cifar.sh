#!/usr/bin/env bash
# Fed-CORE: run the real CIFAR FedOSR experiment inside a CUDA torch container.
#
# Env-driven so it composes with the run ladder in CLAUDE.md / HANDOFF.md, e.g.:
#   bash scripts/docker_cifar.sh
#   NOISE_TYPE=symmetric  NOISE_RATE=0.35 bash scripts/docker_cifar.sh
#   NOISE_TYPE=asymmetric NOISE_RATE=0.20 bash scripts/docker_cifar.sh
#   DATASET=cifar100 N_KNOWN=60 bash scripts/docker_cifar.sh
#
# Mounts the repo at /workspace and uses --gpus all. Never commits runs/ or data/.
set -euo pipefail

IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
DATASET="${DATASET:-cifar10}"
N_KNOWN="${N_KNOWN:-6}"
N_CLIENTS="${N_CLIENTS:-5}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.1}"
ROUNDS="${ROUNDS:-50}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
ALPHA="${ALPHA:-0.10}"
DELTA="${DELTA:-0.10}"
NOISE_TYPE="${NOISE_TYPE:-none}"
NOISE_RATE="${NOISE_RATE:-0.0}"
SEED="${SEED:-0}"
BACKBONE="${BACKBONE:-simplecnn}"
NORM="${NORM:-bn}"

# repo root = parent of this scripts/ directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TAG="${DATASET}_d${DIRICHLET_ALPHA}_${NOISE_TYPE}${NOISE_RATE}_seed${SEED}"
OUT="${OUT:-runs/${TAG}.csv}"

echo "[docker_cifar] image=${IMAGE}"
echo "[docker_cifar] tag=${TAG} -> ${OUT}"

# The pytorch/pytorch runtime image ships torch+torchvision but not scipy,
# which the CP core requires. Install it (cached in pip's dir) before running.
PIP_INSTALL="${PIP_INSTALL:-pip install -q --no-cache-dir scipy && pip install -q -e .}"

docker run --rm --gpus all \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  bash -c "${PIP_INSTALL} && python experiments/fedcore/run_cifar.py \
    --dataset '${DATASET}' \
    --n_known '${N_KNOWN}' \
    --n_clients '${N_CLIENTS}' \
    --dirichlet_alpha '${DIRICHLET_ALPHA}' \
    --rounds '${ROUNDS}' \
    --local_epochs '${LOCAL_EPOCHS}' \
    --alpha '${ALPHA}' \
    --delta '${DELTA}' \
    --noise_type '${NOISE_TYPE}' \
    --noise_rate '${NOISE_RATE}' \
    --seed '${SEED}' \
    --backbone '${BACKBONE}' \
    --norm '${NORM}' \
    --data_root data \
    --out '${OUT}' ${EXTRA_ARGS:-}"
