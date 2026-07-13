#!/usr/bin/env bash
# M0: complete the 5-seed certified self-training gain cell (FedPD-PROSER base, 4x audit,
# one-shot delta certification). Runs the SAME recipe as seeds 0-2 for the requested seeds
# and appends to runs/selftrain_pkg_5seed.csv (the extension source that
# fedcore.experiments.build_selftrain_gain_5seed reads for seeds 3,4).
#
#   SEEDS="3 4" GPUS="0 1" bash scripts/run_selftrain_fedpd_5seed.sh
#
# GPU pinning: seed i runs in its own container with CUDA_VISIBLE_DEVICES set to the
# matching entry of GPUS (round-robin). Concurrent appends are file-locked (append_csv_locked).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"

IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
SEEDS="${SEEDS:-3 4}"
GPUS="${GPUS:-0 1}"
OUT="${OUT:-runs/selftrain_pkg_5seed.csv}"
# recipe (identical to seeds 0-2)
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-5}"; ALPHA="${ALPHA:-0.20}"; DELTA="${DELTA:-0.10}"
AUDIT="${AUDIT:-4}"; AUDIT_DIV="${AUDIT_DIV:-4}"; LABELED_FRAC="${LABELED_FRAC:-0.5}"
PROP_FRAC="${PROP_FRAC:-0.4}"; TEST_FRAC="${TEST_FRAC:-0.3}"
PRETRAIN_ROUNDS="${PRETRAIN_ROUNDS:-40}"; PROSER_ROUNDS="${PROSER_ROUNDS:-15}"; FINETUNE_ROUNDS="${FINETUNE_ROUNDS:-8}"
MODES="${MODES:-none certified oracle}"
PIP_INSTALL="${PIP_INSTALL:-pip install -q --no-cache-dir scipy scikit-learn thop && pip install -q -e .}"

read -r -a SEED_ARR <<< "$SEEDS"
read -r -a GPU_ARR  <<< "$GPUS"

pids=()
for i in "${!SEED_ARR[@]}"; do
  S="${SEED_ARR[$i]}"
  GPU="${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}"
  LOG="runs/selftrain_fedpd_5seed_seed${S}.log"
  echo "[m0] seed=$S -> GPU $GPU  (log: $LOG)  $(date '+%H:%M:%S')"
  docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES="${GPU}" \
    -v "${REPO_ROOT}:/workspace" -v "${REPO_ROOT}/third_party/FedPD:/fedpd" \
    -w /workspace "${IMAGE}" \
    bash -c "${PIP_INSTALL} && python -m fedcore.experiments.run_selftrain_fedpd \
      --dirichlet_alpha '${DIRICHLET_ALPHA}' --alpha '${ALPHA}' --delta '${DELTA}' \
      --audit ${AUDIT} --audit_div '${AUDIT_DIV}' --seed '${S}' \
      --labeled_frac '${LABELED_FRAC}' --prop_frac '${PROP_FRAC}' --test_frac '${TEST_FRAC}' \
      --pretrain_rounds '${PRETRAIN_ROUNDS}' --proser_rounds '${PROSER_ROUNDS}' --finetune_rounds '${FINETUNE_ROUNDS}' \
      --modes ${MODES} --data_root data --out '${OUT}'" > "$LOG" 2>&1 &
  pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[m0] seed=${SEED_ARR[$i]} DONE  $(date '+%H:%M:%S')";
  else echo "[m0] seed=${SEED_ARR[$i]} FAILED (see log)"; fail=1; fi
done
echo "[m0] all seeds finished (fail=$fail) $(date '+%H:%M:%S')"
exit $fail
