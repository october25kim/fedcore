#!/usr/bin/env bash
# Sequentially export FOOGD scored npz for cifar10 d in {5,0.5} x seeds {0,1,2}.
# Single GPU -> sequential. Idempotent: skips a cell whose npz already exists.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export ROUNDS="${ROUNDS:-80}" SCORE_ROUNDS="${SCORE_ROUNDS:-30}" NORM="${NORM:-gn}"
for D in 5 0.5; do
  for S in 0 1 2; do
    NPZ="runs/foogd_cifar10_d${D}_seed${S}.npz"
    if [ -f "$NPZ" ]; then
      echo "[skip] $NPZ exists"; continue
    fi
    echo "[run] d=$D seed=$S -> $NPZ"
    DIRICHLET_ALPHA="$D" SEED="$S" bash scripts/docker_foogd.sh \
      || { echo "[FAIL] d=$D seed=$S"; continue; }
  done
done
echo "[done] foogd export batch"
