#!/usr/bin/env bash
# Fresh-box bootstrap for the 4x4090 Workstation1 run of the Fed-CORE GPU queue.
# Idempotent: safe to re-run. Sets up a torch venv, installs the package, and
# PRE-DOWNLOADS CIFAR-10/100 once (serially) so the 4 parallel dispatch jobs don't
# race the same download. Run from the fedcore/ package root:
#     bash scripts/ws4090/bootstrap.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."     # -> fedcore/ package root
ROOT="$(pwd)"
echo "[bootstrap] repo root: $ROOT"

# 1) GPUs
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[bootstrap] ERROR: nvidia-smi not found -- need NVIDIA driver + CUDA runtime"; exit 2
fi
NGPU=$(nvidia-smi -L | wc -l)
echo "[bootstrap] GPUs detected: $NGPU"; nvidia-smi -L
[ "$NGPU" -ge 1 ] || { echo "[bootstrap] ERROR: no GPUs"; exit 2; }

# 2) venv + torch + package
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo "[bootstrap] creating .venv"; "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
if ! python -c "import torch" 2>/dev/null; then
  echo "[bootstrap] installing torch (cu121)"
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi
echo "[bootstrap] installing fedcore + deps (editable)"
pip install -q scipy scikit-learn thop
pip install -q -e .
python - <<'PY'
import torch
print(f"[bootstrap] torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  ndev={torch.cuda.device_count()}")
assert torch.cuda.is_available(), "torch cannot see CUDA -- check driver/toolkit match"
PY

# 3) pre-download CIFAR-10/100 once (serial) into data/ to avoid concurrent-download races
echo "[bootstrap] pre-downloading CIFAR-10/100 into data/ (serial, one-time)"
python - <<'PY'
import torchvision
for cls in (torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR100):
    cls(root="data", train=True, download=True)
    cls(root="data", train=False, download=True)
print("[bootstrap] CIFAR data ready under data/")
PY

# 4) CPU smoke (wiring only) -- must pass before spending GPU time
echo "[bootstrap] CPU smoke: exp_lemma_L"
python -m fedcore.experiments.exp_lemma_L >/dev/null && echo "[bootstrap] smoke OK"

echo "[bootstrap] DONE. Next: generate manifests and dispatch (see scripts/ws4090/RUNBOOK.md):"
echo "    python scripts/ws4090/gen_manifest.py"
echo "    python scripts/ws4090/dispatch.py scripts/ws4090/manifest_R1.txt --gpus 0,1,2,3 --skip-existing"
