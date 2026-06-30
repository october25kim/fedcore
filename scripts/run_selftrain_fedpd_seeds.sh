#!/usr/bin/env bash
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
for S in 1 2; do
  echo "[run] FedPD self-training seed=$S $(date '+%H:%M')"
  ALPHA=0.20 AUDIT=4 SEED="$S" MODES="none certified oracle" \
    PRETRAIN_ROUNDS=40 PROSER_ROUNDS=15 FINETUNE_ROUNDS=8 \
    bash scripts/docker_selftrain_fedpd.sh || echo "[FAIL] seed=$S"
done
echo "[done] fedpd selftrain seeds $(date '+%H:%M')"
