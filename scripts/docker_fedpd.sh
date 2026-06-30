#!/usr/bin/env bash
# FedPD (PROSER) on our CIFAR-10 FedOSR split, inside the CUDA container, mounting
# third_party/FedPD at /fedpd.  SMOKE=1 for a tiny wiring check.
set -euo pipefail
IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
DATASET="${DATASET:-cifar10}"; N_KNOWN="${N_KNOWN:-6}"; N_CLIENTS="${N_CLIENTS:-5}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-5}"; ROUNDS="${ROUNDS:-15}"; PRETRAIN_ROUNDS="${PRETRAIN_ROUNDS:-50}"; SEED="${SEED:-0}"
SMOKE_FLAG=""; [ "${SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIP_INSTALL="${PIP_INSTALL:-pip install -q --no-cache-dir scipy scikit-learn thop}"
echo "[docker_fedpd] d=${DIRICHLET_ALPHA} seed=${SEED} rounds=${ROUNDS} smoke=${SMOKE:-0}"
docker run --rm --gpus all \
  -v "${REPO_ROOT}:/workspace" -v "${REPO_ROOT}/third_party/FedPD:/fedpd" \
  -w /workspace "${IMAGE}" \
  bash -c "${PIP_INSTALL} && python experiments/fedcore/run_fedpd_cifar.py \
    --dataset '${DATASET}' --n_known '${N_KNOWN}' --n_clients '${N_CLIENTS}' \
    --dirichlet_alpha '${DIRICHLET_ALPHA}' --rounds '${ROUNDS}' \
    --pretrain_rounds '${PRETRAIN_ROUNDS}' --seed '${SEED}' \
    --data_root data ${SMOKE_FLAG} ${EXTRA_ARGS:-}"
