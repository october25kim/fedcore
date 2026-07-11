#!/usr/bin/env bash
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
for S in 1 2; do
  echo "[run] STEP3 lf=0.10 seed=$S $(date '+%H:%M')"
  LABELED_FRAC=0.10 ALPHA=0.20 AUDIT=4 AUDIT_DIV=8 PROP_FRAC=0.2 TEST_FRAC=0.2 SEED="$S" \
    MODES="none certified oracle" PRETRAIN_ROUNDS=40 PROSER_ROUNDS=15 FINETUNE_ROUNDS=8 \
    OUT=runs/selftrain_lowlabel.csv \
    bash scripts/docker_selftrain_fedpd.sh || echo "[FAIL] seed=$S"
done
echo "[done] STEP3 lowlabel seeds $(date '+%H:%M')"
