# Workstation1 (4×4090) runbook — Fed-CORE GPU queue R1/R2/R5 (+R3/R6/R7)

Portable bundle to run the remaining GPU experiments on the fresh 4×4090 box.
Everything here is device-agnostic; `dispatch.py` pins one job per GPU and keeps
all four busy. **R8 and R4 are already done on the other host — do NOT redo them.**

## 0. Get the code + data
```bash
git clone git@github.com:october25kim/ssrc.git      # or your existing remote
cd ssrc && git checkout refactor/hoist-dedup && cd fedcore
```
`data/` (CIFAR) and `third_party/` are NOT in git. Bootstrap downloads CIFAR;
`third_party/` (FedPD/FOOGD) is only needed for R3/R6 (see §4).

## 1. Bootstrap (once)
```bash
bash scripts/ws4090/bootstrap.sh          # venv + torch(cu121) + pip install -e . + CIFAR predownload + CPU smoke
source .venv/bin/activate
```

## 2. Generate manifests
```bash
python scripts/ws4090/gen_manifest.py     # -> scripts/ws4090/manifest_{R1,R2,R5}.txt
```

## 3. Priority order (GPU): R1 → R2 → R5  (R3/R6/R7 in §4)
`dispatch.py` is idempotent with `--skip-existing` (skips a job whose logits npz
already exists), so you can stop/resume freely. Failures are logged and reported,
never silently retried.

### R1 — CIFAR-10 client scaling (20 runs, 10 seeds each J)
```bash
python scripts/ws4090/dispatch.py scripts/ws4090/manifest_R1.txt --gpus 0,1,2,3 --skip-existing
python scripts/ws4090/certify_client_scaling.py            # -> runs/client_scaling.csv
```
Acceptance: Theorem-3 pattern — per-client (G=J) bounds degrade with J; grouping
(G→2) restores certification. The aggregator prints CertCov@0.20 by (J,G).
**Partial-participation variant: SKIP** — `run_cifar` has no native client
sampling (verified), so per R1 that variant is not run. Say so in the report.

### R2 — CIFAR-100 multi-model (60 runs, 10 seeds; DO NOT cut seeds)
```bash
python scripts/ws4090/dispatch.py scripts/ws4090/manifest_R2.txt --gpus 0,1,2,3 --skip-existing
python scripts/ws4090/certify_grid.py --task r2           # -> runs/cifar100_multimodel.csv
```
Backbone order in the manifest is resnet18gn → resnet18bn → simplecnn; if GPU time
runs short, report any incomplete backbone cell as INCOMPLETE rather than at <10
seeds. (Optional: also append per-seed G∈{2,J} rows into runs/T9_diagnostics.csv.)

### R5 — corruption curve (60 runs, 10 seeds) — lowest GPU priority
```bash
python scripts/ws4090/dispatch.py scripts/ws4090/manifest_R5.txt --gpus 0,1,2,3 --skip-existing
python scripts/ws4090/certify_grid.py --task r5           # -> runs/corruption_curve_seeded.csv
```
Split hygiene is built into `run_cifar` (TRAIN labels only; calibration folds
clean) — no extra care needed.

## 4. Detector / covtype tasks (need extra setup — phase 2)
- **R3 (FOOGD seed extension)** and **R6 (FedPD simplex / small-J)** retrain
  detectors and need `third_party/` (FedPD, FOOGD) which is NOT in git. Copy it
  from the other host or clone upstream, then use the existing wrappers
  (`scripts/docker_fedpd.sh`, `scripts/docker_foogd.sh` / `run_foogd_cifar.py`,
  `run_fedpd_cifar.py`). Detector cells have a **floor of 5 seeds** (extend toward
  10 if budget remains); flag any cell <10.
- **R7 (covtype stable second positive)** is CPU-heavy: needs seeds 5..9 trained
  (locate the covtype trainer that produced `runs/covtype_seed*_logits.npz`), then
  `python experiments/fedcore/exp_covtype_valid_multiscore.py` (A2-compliant:
  proposal-fold score selection). Grouped G=2, alpha∈{0.20,0.25,0.30}, 10 seeds.

## Guardrails (inviolable)
- proposal / certification / test folds disjoint; never use test labels upstream.
- corruption on TRAIN labels only; calibration folds stay clean.
- contiguous pre-declared grouping (client c → group `c*G//J`); stated in every CSV header.
- **never commit** `runs/`, `data/`, weights; DO commit new scripts.
- no silent retries — `dispatch.py` reports failures; investigate the per-job log
  under `runs/ws4090_logs/<label>.log`.
- Judge by `cert_*` risk/coverage, never accuracy/AUROC. Seed floor = 10 for
  R1/R2/R5; 5 for detector retraining (R3/R6).

## Cost sketch
R1+R2+R5 = 140 `run_cifar` jobs. At 4-way parallel (~8–10 min/run on a 4090)
≈ 140/4 × ~9 min ≈ **5–6 h** wall-clock for the whole training grid; aggregation
is minutes (CPU). Collect result CSVs from `runs/` back to the paper host.
