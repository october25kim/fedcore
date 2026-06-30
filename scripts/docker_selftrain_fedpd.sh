#!/usr/bin/env bash
# P2: one-shot certified self-training with the FedPD-PROSER base (mounts /fedpd).
#   ALPHA=0.20 AUDIT=4 SEED=0 MODES="none certified oracle" bash scripts/docker_selftrain_fedpd.sh
#   SMOKE=1 ... for a tiny wiring check.
set -euo pipefail
IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-5}"; ALPHA="${ALPHA:-0.20}"; AUDIT="${AUDIT:-4}"; AUDIT_DIV="${AUDIT_DIV:-4}"; SEED="${SEED:-0}"
PRETRAIN_ROUNDS="${PRETRAIN_ROUNDS:-40}"; PROSER_ROUNDS="${PROSER_ROUNDS:-15}"; FINETUNE_ROUNDS="${FINETUNE_ROUNDS:-8}"
MODES="${MODES:-none certified oracle}"; OUT="${OUT:-runs/selftrain_pkg.csv}"; LABELED_FRAC="${LABELED_FRAC:-0.5}"
PROP_FRAC="${PROP_FRAC:-0.4}"; TEST_FRAC="${TEST_FRAC:-0.3}"
SMOKE_FLAG=""; [ "${SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIP_INSTALL="${PIP_INSTALL:-pip install -q --no-cache-dir scipy scikit-learn thop && pip install -q -e .}"
echo "[docker_selftrain_fedpd] alpha=${ALPHA} audit='${AUDIT}'(div=${AUDIT_DIV}) lf=${LABELED_FRAC} seed=${SEED} modes='${MODES}' smoke=${SMOKE:-0}"
docker run --rm --gpus all -v "${REPO_ROOT}:/workspace" -v "${REPO_ROOT}/third_party/FedPD:/fedpd" \
  -w /workspace "${IMAGE}" \
  bash -c "${PIP_INSTALL} && python experiments/fedcore/run_selftrain_fedpd.py \
    --dirichlet_alpha '${DIRICHLET_ALPHA}' --alpha '${ALPHA}' --audit ${AUDIT} --audit_div '${AUDIT_DIV}' --seed '${SEED}' \
    --labeled_frac '${LABELED_FRAC}' --prop_frac '${PROP_FRAC}' --test_frac '${TEST_FRAC}' \
    --pretrain_rounds '${PRETRAIN_ROUNDS}' --proser_rounds '${PROSER_ROUNDS}' --finetune_rounds '${FINETUNE_ROUNDS}' \
    --modes ${MODES} --data_root data --out '${OUT}' ${SMOKE_FLAG} ${EXTRA_ARGS:-}"
