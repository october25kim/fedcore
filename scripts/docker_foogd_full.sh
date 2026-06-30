#!/usr/bin/env bash
# FULL FOOGD (SAG): run FOOGD's real ODGClient training (WideResNet+score+KSD/MMD) on our
# split inside the CUDA container, mounting third_party/FOOGD-main at /foogd.
#   bash scripts/docker_foogd_full.sh                       # d=5 seed0
#   DIRICHLET_ALPHA=0.5 SEED=1 bash scripts/docker_foogd_full.sh
#   SMOKE=1 bash scripts/docker_foogd_full.sh
set -euo pipefail
IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
DATASET="${DATASET:-cifar10}"; N_KNOWN="${N_KNOWN:-6}"; N_CLIENTS="${N_CLIENTS:-5}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-5}"; ROUNDS="${ROUNDS:-30}"; SEED="${SEED:-0}"
LAMBDA1="${LAMBDA1:-0.1}"; LAMBDA2="${LAMBDA2:-0.1}"
SMOKE_FLAG=""; [ "${SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIP_INSTALL="${PIP_INSTALL:-pip install -q --no-cache-dir scipy scikit-learn && pip install -q -e .}"
echo "[docker_foogd_full] d=${DIRICHLET_ALPHA} seed=${SEED} lambda1=${LAMBDA1} lambda2=${LAMBDA2} smoke=${SMOKE:-0}"
docker run --rm --gpus all \
  -v "${REPO_ROOT}:/workspace" -v "${REPO_ROOT}/third_party/FOOGD-main:/foogd" \
  -w /workspace "${IMAGE}" \
  bash -c "${PIP_INSTALL} && python experiments/fedcore/run_foogd_full_cifar.py \
    --dataset '${DATASET}' --n_known '${N_KNOWN}' --n_clients '${N_CLIENTS}' \
    --dirichlet_alpha '${DIRICHLET_ALPHA}' --rounds '${ROUNDS}' --seed '${SEED}' \
    --lambda1 '${LAMBDA1}' --lambda2 '${LAMBDA2}' --data_root data ${SMOKE_FLAG} ${EXTRA_ARGS:-}"
