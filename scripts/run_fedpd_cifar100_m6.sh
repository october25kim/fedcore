#!/usr/bin/env bash
# M6: FedPD-PROSER on CIFAR-100 (n_known=60, d=5), seeds 0,1,2. Pretrain-then-finetune
# recipe (CE FedAvg pretrain + PROSER traindummy finetune), native -sm detector score.
# Produces runs/fedpd_cifar100_d5_seed{s}.npz (certified later by exp_m6_cifar100_fedpd.py).
#
#   SEEDS="0 1 2" GPUS="0 1 2" bash scripts/run_fedpd_cifar100_m6.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"

IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
SEEDS="${SEEDS:-0 1 2}"; GPUS="${GPUS:-0 1 2}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-5}"; N_KNOWN="${N_KNOWN:-60}"; N_CLIENTS="${N_CLIENTS:-5}"
PRETRAIN_ROUNDS="${PRETRAIN_ROUNDS:-50}"; ROUNDS="${ROUNDS:-15}"
PIP_INSTALL="${PIP_INSTALL:-pip install -q --no-cache-dir scipy scikit-learn thop && pip install -q -e .}"

read -r -a SEED_ARR <<< "$SEEDS"; read -r -a GPU_ARR <<< "$GPUS"
pids=()
for i in "${!SEED_ARR[@]}"; do
  S="${SEED_ARR[$i]}"; GPU="${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}"
  OUT="runs/fedpd_cifar100_d${DIRICHLET_ALPHA}_seed${S}.npz"
  LOG="runs/fedpd_cifar100_m6_seed${S}.log"
  if [ -f "$OUT" ]; then echo "[m6] SKIP seed=$S (exists)"; continue; fi
  echo "[m6] seed=$S -> GPU $GPU (log $LOG) $(date '+%H:%M:%S')"
  docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES="${GPU}" \
    -v "${REPO_ROOT}:/workspace" -v "${REPO_ROOT}/third_party/FedPD:/fedpd" \
    -w /workspace "${IMAGE}" \
    bash -c "${PIP_INSTALL} && python -m fedcore.experiments.run_fedpd_cifar \
      --dataset cifar100 --n_known ${N_KNOWN} --n_clients ${N_CLIENTS} \
      --dirichlet_alpha '${DIRICHLET_ALPHA}' --pretrain_rounds ${PRETRAIN_ROUNDS} --rounds ${ROUNDS} \
      --seed '${S}' --data_root data --out '${OUT}'" > "$LOG" 2>&1 &
  pids+=($!)
done
fail=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" || fail=1; done
echo "[m6] finished (fail=$fail) $(date '+%H:%M:%S')"; exit $fail
