#!/usr/bin/env bash
# Fed-CORE: run the full CIFAR experiment ladder (CLAUDE.md sec 4 / HANDOFF sec 2)
# sequentially via docker_cifar.sh. Idempotent: a rung whose CSV already exists is
# skipped, so the script is safe to re-run after an interruption.
#
#   bash scripts/run_ladder.sh
#
# Each rung's full output is appended to runs/ladder.log; per-rung CSVs land in
# runs/<tag>.csv (+ <tag>_logits.npz).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p runs
LADDER_LOG="runs/ladder.log"

lad() {
  export DATASET="$1" DIRICHLET_ALPHA="$2" NOISE_TYPE="$3" NOISE_RATE="$4" SEED="$5"
  export N_KNOWN="${6:-6}"
  local tag="${DATASET}_d${DIRICHLET_ALPHA}_${NOISE_TYPE}${NOISE_RATE}_seed${SEED}"
  if [ -f "runs/${tag}.csv" ]; then
    echo "[ladder $(date +%H:%M:%S)] SKIP ${tag} (csv exists)" | tee -a "$LADDER_LOG"
    return
  fi
  echo "[ladder $(date +%H:%M:%S)] RUN  ${tag}" | tee -a "$LADDER_LOG"
  if bash scripts/docker_cifar.sh >> "$LADDER_LOG" 2>&1; then
    local cov
    cov="$(grep -aE 'CertifiedCoverage@alpha' "$LADDER_LOG" | tail -1)"
    echo "[ladder $(date +%H:%M:%S)] DONE ${tag} :: ${cov}" | tee -a "$LADDER_LOG"
  else
    echo "[ladder $(date +%H:%M:%S)] FAIL ${tag} (see $LADDER_LOG)" | tee -a "$LADDER_LOG"
  fi
}

# --- the ladder ----------------------------------------------------------------
# 1) cifar10 clean seed0 d0.1  (run separately already; skipped if present)
lad cifar10 0.1 none       0.0  0
# 2) client-side corruption, seed0
lad cifar10 0.1 symmetric  0.35 0
lad cifar10 0.1 asymmetric 0.20 0
# 3) seeds 1,2 for the three conditions
lad cifar10 0.1 none       0.0  1
lad cifar10 0.1 none       0.0  2
lad cifar10 0.1 symmetric  0.35 1
lad cifar10 0.1 symmetric  0.35 2
lad cifar10 0.1 asymmetric 0.20 1
lad cifar10 0.1 asymmetric 0.20 2
# 4) dirichlet sweep (clean seed0): 0.1 done above; add 0.5 and 5
lad cifar10 0.5 none       0.0  0
lad cifar10 5   none       0.0  0
# 5) cifar100 (60 known / 40 unknown), clean seed0
lad cifar100 0.1 none      0.0  0 60

echo "[ladder $(date +%H:%M:%S)] LADDER COMPLETE" | tee -a "$LADDER_LOG"
echo "=== certified-coverage summary ===" | tee -a "$LADDER_LOG"
for f in runs/cifar*_d*.csv; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY' | tee -a "$LADDER_LOG"
import csv, sys
f = sys.argv[1]
rows = list(csv.DictReader(open(f)))
cert = [r for r in rows if r["certified"] == "True"]
best = max(cert, key=lambda r: float(r["cert_coverage_lcb"]), default=None)
tag = f.split("/")[-1].replace(".csv", "")
if best:
    print(f"{tag:42} cov={float(best['cert_coverage_lcb']):.4f} "
          f"score={best['score_name']} gamma={best['gamma']} L={best['Lambda']}")
else:
    print(f"{tag:42} cov=0.0000 (none certified)")
PY
done
