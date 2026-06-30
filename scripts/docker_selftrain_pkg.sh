#!/usr/bin/env bash
# One-shot certified self-training package (P1-P4) inside the CUDA container.
#   MODES="none naive certified oracle" ALPHAS=0.20 BETAS=0.25 AUDIT=1 SEEDS=0 \
#     bash scripts/docker_selftrain_pkg.sh
set -euo pipefail
IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
BACKBONE="${BACKBONE:-resnet18}"; NORM="${NORM:-gn}"; SCORE="${SCORE:-msp}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-5}"; FEDAVG_ROUNDS="${FEDAVG_ROUNDS:-40}"; FINETUNE_ROUNDS="${FINETUNE_ROUNDS:-15}"
MODES="${MODES:-none naive certified oracle}"; ALPHAS="${ALPHAS:-0.20}"; BETAS="${BETAS:-0.25}"
AUDIT="${AUDIT:-1}"; SEEDS="${SEEDS:-0}"; OUT="${OUT:-runs/selftrain_pkg.csv}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIP_INSTALL="${PIP_INSTALL:-pip install -q --no-cache-dir scipy scikit-learn}"
echo "[docker_selftrain_pkg] modes='${MODES}' alphas='${ALPHAS}' betas='${BETAS}' audit='${AUDIT}' seeds='${SEEDS}'"
docker run --rm --gpus all -v "${REPO_ROOT}:/workspace" -w /workspace "${IMAGE}" \
  bash -c "${PIP_INSTALL} && python experiments/fedcore/run_selftrain_pkg.py \
    --backbone '${BACKBONE}' --norm '${NORM}' --score '${SCORE}' --dirichlet_alpha '${DIRICHLET_ALPHA}' \
    --fedavg_rounds '${FEDAVG_ROUNDS}' --finetune_rounds '${FINETUNE_ROUNDS}' \
    --modes ${MODES} --alphas ${ALPHAS} --betas ${BETAS} --audit ${AUDIT} --seeds ${SEEDS} \
    --data_root data --out '${OUT}' ${EXTRA_ARGS:-}"
