#!/usr/bin/env bash
# Sequentially export FedPD-PROSER (closed-set pretrain + PROSER fine-tune) scored npz
# for cifar10 d in {5,0.5} x seeds {0,1,2}. Single GPU -> sequential. Idempotent.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
export PRETRAIN_ROUNDS="${PRETRAIN_ROUNDS:-40}" ROUNDS="${ROUNDS:-15}"
export EXTRA_ARGS="${EXTRA_ARGS:---local_epochs 2}"
for D in 5 0.5; do
  for S in 0 1 2; do
    NPZ="runs/fedpd_cifar10_d${D}_seed${S}.npz"
    [ -f "$NPZ" ] && { echo "[skip] $NPZ"; continue; }
    echo "[run] FedPD d=$D seed=$S -> $NPZ"
    DIRICHLET_ALPHA="$D" SEED="$S" bash scripts/docker_fedpd.sh || { echo "[FAIL] d=$D seed=$S"; continue; }
  done
done
echo "[done] fedpd export batch"
