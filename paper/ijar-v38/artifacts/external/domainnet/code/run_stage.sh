#!/usr/bin/env bash
# usage: run_stage.sh <stage> [extra args]
set -euo pipefail
F=/home/sanghoon/Desktop/Workspace/Fedcore
R=$F/outputs/fedcore_s1v2_run_20260925
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
cd $F/work/s1v2/s1v2
exec ~/miniconda3/envs/torch/bin/python runner.py "$@" --run-dir $R --data-dir $F/data/domainnet_s1v2/prepared --zips $F/data/domainnet_s1v2/zips \
  --protocol $F/outputs/fedcore_s1v2_prereg_20260925/PROTOCOL.json --model-matrix $F/work/v1_pacs_preflight/MODEL_MATRIX.csv \
  --v1-run-state $F/outputs/fedcore_v1_run/RUN_STATE.json --ckpt-sha $F/outputs/fedcore_s1v2_prereg_20260925/refs/V1_CHECKPOINT_SHA256.txt
