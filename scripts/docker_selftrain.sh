#!/usr/bin/env bash
# Fed-CORE: certified federated self-training (Proposition 4) inside a CUDA torch
# container. Env-driven; smoke-size by default. Mirrors docker_cifar.sh.
#
#   bash scripts/docker_selftrain.sh
#   NOISE_TYPE=symmetric NOISE_RATE=0.35 bash scripts/docker_selftrain.sh
set -euo pipefail

IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
DATASET="${DATASET:-cifar10}"
N_KNOWN="${N_KNOWN:-6}"
N_CLIENTS="${N_CLIENTS:-5}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.1}"
T_ROUNDS="${T_ROUNDS:-4}"
FEDAVG_ROUNDS="${FEDAVG_ROUNDS:-5}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
ALPHA="${ALPHA:-0.10}"
DELTA="${DELTA:-0.10}"
GAMMA="${GAMMA:-0.7}"
SCORE="${SCORE:-energy}"
NOISE_TYPE="${NOISE_TYPE:-none}"
NOISE_RATE="${NOISE_RATE:-0.0}"
SEED="${SEED:-0}"
BACKBONE="${BACKBONE:-simplecnn}"
NORM="${NORM:-bn}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="selftrain_${DATASET}_${BACKBONE}_d${DIRICHLET_ALPHA}_${NOISE_TYPE}${NOISE_RATE}_seed${SEED}"
OUT="${OUT:-runs/${TAG}.csv}"
PIP_INSTALL="${PIP_INSTALL:-pip install -q --no-cache-dir scipy && pip install -q -e .}"

echo "[docker_selftrain] image=${IMAGE} tag=${TAG} -> ${OUT}"

docker run --rm --gpus all \
  -v "${REPO_ROOT}:/workspace" -w /workspace "${IMAGE}" \
  bash -c "${PIP_INSTALL} && python experiments/fedcore/run_selftrain_cifar.py \
    --dataset '${DATASET}' --n_known '${N_KNOWN}' --n_clients '${N_CLIENTS}' \
    --dirichlet_alpha '${DIRICHLET_ALPHA}' --T '${T_ROUNDS}' \
    --fedavg_rounds '${FEDAVG_ROUNDS}' --local_epochs '${LOCAL_EPOCHS}' \
    --alpha '${ALPHA}' --delta '${DELTA}' --gamma '${GAMMA}' --score '${SCORE}' \
    --noise_type '${NOISE_TYPE}' --noise_rate '${NOISE_RATE}' --seed '${SEED}' \
    --backbone '${BACKBONE}' --norm '${NORM}' \
    --data_root data --out '${OUT}'"
