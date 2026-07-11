#!/usr/bin/env bash
# Task E seed extensions (single GPU, sequential):
#   1. FedPD-PROSER npz export, d in {5,0.5} x seed in {3,4}  (recipe = run_fedpd_all.sh)
#   2. FedPD self-training 4x-budget cell, seed in {3,4}      (recipe = run_selftrain_fedpd_seeds.sh)
# Idempotent: FedPD npz skips if present. Logs to the path in $LOG.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
LOG="${LOG:-/tmp/taskE.log}"

echo "[taskE] START $(date '+%F %H:%M')" | tee -a "$LOG"

# --- 1. FedPD-PROSER npz seeds 3,4 (match run_fedpd_all.sh recipe) ---
export PRETRAIN_ROUNDS=40 ROUNDS=15 EXTRA_ARGS="--local_epochs 2"
for D in 5 0.5; do
  for S in 3 4; do
    NPZ="runs/fedpd_cifar10_d${D}_seed${S}.npz"
    if [ -f "$NPZ" ]; then echo "[skip] $NPZ" | tee -a "$LOG"; continue; fi
    echo "[run] FedPD npz d=$D seed=$S $(date '+%H:%M')" | tee -a "$LOG"
    DIRICHLET_ALPHA="$D" SEED="$S" bash scripts/docker_fedpd.sh >>"$LOG" 2>&1 \
      && echo "[ok] $NPZ" | tee -a "$LOG" \
      || echo "[FAIL] FedPD npz d=$D seed=$S" | tee -a "$LOG"
  done
done

# --- 2. FedPD self-training 4x-budget seeds 3,4 (match run_selftrain_fedpd_seeds.sh recipe) ---
for S in 3 4; do
  echo "[run] FedPD self-training 4x seed=$S $(date '+%H:%M')" | tee -a "$LOG"
  ALPHA=0.20 AUDIT=4 SEED="$S" DIRICHLET_ALPHA=5 MODES="none certified oracle" \
    PRETRAIN_ROUNDS=40 PROSER_ROUNDS=15 FINETUNE_ROUNDS=8 \
    OUT="runs/selftrain_pkg_5seed.csv" \
    bash scripts/docker_selftrain_fedpd.sh >>"$LOG" 2>&1 \
    && echo "[ok] selftrain seed=$S" | tee -a "$LOG" \
    || echo "[FAIL] selftrain seed=$S" | tee -a "$LOG"
done

echo "[taskE] DONE $(date '+%F %H:%M')" | tee -a "$LOG"
