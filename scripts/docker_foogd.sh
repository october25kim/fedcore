#!/usr/bin/env bash
# Fed-CORE: train FOOGD SM3D score head on a shared FedAvg backbone inside the CUDA
# torch container, exporting the native open-set score on the audit folds.
#   bash scripts/docker_foogd.sh                       # d=5 seed0
#   DIRICHLET_ALPHA=0.5 SEED=1 bash scripts/docker_foogd.sh
#   SMOKE=1 bash scripts/docker_foogd.sh               # tiny wiring check
set -euo pipefail

IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
DATASET="${DATASET:-cifar10}"
N_KNOWN="${N_KNOWN:-6}"
N_CLIENTS="${N_CLIENTS:-5}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-5}"
ROUNDS="${ROUNDS:-50}"
SCORE_ROUNDS="${SCORE_ROUNDS:-30}"
SEED="${SEED:-0}"
NORM="${NORM:-gn}"
SMOKE_FLAG=""
[ "${SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIP_INSTALL="${PIP_INSTALL:-pip install -q --no-cache-dir scipy scikit-learn && pip install -q -e .}"

echo "[docker_foogd] image=${IMAGE} d=${DIRICHLET_ALPHA} seed=${SEED} smoke=${SMOKE:-0}"

docker run --rm --gpus all \
  -v "${REPO_ROOT}:/workspace" -w /workspace "${IMAGE}" \
  bash -c "${PIP_INSTALL} && python experiments/fedcore/run_foogd_cifar.py \
    --dataset '${DATASET}' --n_known '${N_KNOWN}' --n_clients '${N_CLIENTS}' \
    --dirichlet_alpha '${DIRICHLET_ALPHA}' --rounds '${ROUNDS}' \
    --score_rounds '${SCORE_ROUNDS}' --seed '${SEED}' --norm '${NORM}' \
    --data_root data ${SMOKE_FLAG} ${EXTRA_ARGS:-}"
