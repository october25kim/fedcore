#!/usr/bin/env bash
# FedOSS (DUSS+FOSS) on our CIFAR-10 FedOSR split, inside the CUDA container, mounting
# third_party/FedOSS at /fedoss.  SMOKE=1 for a tiny wiring check.
# GPUS pins the device (default all); use CUDA_VISIBLE_DEVICES-style single id for one GPU.
set -euo pipefail
IMAGE="${IMAGE:-pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime}"
DATASET="${DATASET:-cifar10}"; N_KNOWN="${N_KNOWN:-6}"; N_CLIENTS="${N_CLIENTS:-5}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-5}"; ROUNDS="${ROUNDS:-30}"; PRETRAIN_ROUNDS="${PRETRAIN_ROUNDS:-40}"; SEED="${SEED:-0}"
GPUS="${GPUS:-all}"
SMOKE_FLAG=""; [ "${SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIP_INSTALL="${PIP_INSTALL:-pip install -q --no-cache-dir scipy==1.17.1 scikit-learn==1.9.0 && pip install -q -e .}"
echo "[docker_fedoss] image=${IMAGE} d=${DIRICHLET_ALPHA} seed=${SEED} pretrain=${PRETRAIN_ROUNDS} finetune=${ROUNDS} smoke=${SMOKE:-0} gpus=${GPUS}"
if [ "${GPUS}" = "all" ]; then GPU_ARG="all"; else GPU_ARG="device=${GPUS}"; fi
docker run --rm --gpus "${GPU_ARG}" \
  -v "${REPO_ROOT}:/workspace" -v "${REPO_ROOT}/third_party/FedOSS:/fedoss" \
  -w /workspace "${IMAGE}" \
  bash -c "${PIP_INSTALL} && python -m fedcore.experiments.run_fedoss_cifar \
    --dataset '${DATASET}' --n_known '${N_KNOWN}' --n_clients '${N_CLIENTS}' \
    --dirichlet_alpha '${DIRICHLET_ALPHA}' --rounds '${ROUNDS}' \
    --pretrain_rounds '${PRETRAIN_ROUNDS}' --seed '${SEED}' \
    --data_root data ${SMOKE_FLAG} ${EXTRA_ARGS:-}"
