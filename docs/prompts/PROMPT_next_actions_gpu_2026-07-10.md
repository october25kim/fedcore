# PROMPT — Fed-CORE next actions (GPU server, 2026-07-10)

> Paste this whole file as the task brief for Claude Code on the GPU server.
> Execute the tasks **in order**. Tasks 1 and 5 are CPU-only; 2–4 need GPUs.

---

Read `CLAUDE.md` and `AGENTS.md` at the repo root **first** and obey them. In
particular:

- **Docker-first.** All GPU training runs inside the project container
  (`bash scripts/docker_cifar.sh` or the documented `python -m fedcore.experiments.*`
  entry points inside a torch container). Never train with host Python.
- **Smoke-first.** Before any multi-seed GPU campaign, run one smallest-config
  sanity run and confirm the output schema.
- **Split hygiene.** proposal / certification / test folds stay disjoint; test
  labels never touch proposal or certification. Corruption is injected into
  TRAIN labels only; trusted calibration folds stay clean.
- **Canonical metric schema** (do not rename): `certified, cert_risk_ucb,
  cert_coverage_lcb, cert_n, cert_k, prop_coverage, prop_risk, test_coverage,
  test_risk, score_name, gamma, alpha, delta, Lambda, dirichlet_alpha, n_clients`.
- **Division of labor.** This server produces `runs/*.csv`, `runs/*_logits.npz`,
  and reports only. Do **not** edit `docs/Fed-CORE_draft.md`, `docs/*_KO.*`, or
  regenerate manuscript figures (`experiments/fedcore/figs/paper_figs_v2.py` is
  run on the laptop after artifacts sync).
- **Git.** Never commit `runs/`, `data/`, `checkpoints/`, `*.npz`, `*.pt`.
  Commit only code and reports, in English, one logical change per commit.
- Report every task in the fixed format (진단 요약 / 확인한 명령 / 핵심 결과 /
  판정 / 다음 행동). Never hide failed commands.

Context for all tasks: the manuscript (48pp, 2026-07-10 revision) now includes
(i) an FCP coverage-rule recast necessity result on 18 clean ResNet runs
(`fedcore/experiments/exp_fcp_recast.py`, `runs/fcp_recast.csv`; realized
accepted risk 0.225–0.382, exceeding alpha in 18/18 runs), and (ii) Figure 6:
(a) a real-data certified-coverage frontier (GN/BN 5 seeds, FedPD-PROSER native
3 seeds, d=5) and (b) an AUROC-vs-CertifiedCoverage scatter from
`runs/T8_fedosr_bases_agg.csv`. The tasks below harden exactly these additions.

---

## Task 1 (CPU, ~hours) — Fold FCP-recast into the resampling validity study

**Goal.** The 18-run FCP-recast result is a point estimate; give it the same
resampling treatment as the certificate so the paper can quote a violation
*rate* with a confidence bound.

1. Locate the generator of `runs/resampling_validity.csv` (grep for
   `resampling` under `fedcore/experiments/`). Reuse its redraw machinery:
   for each stored clean CIFAR-10 ResNet run (the 18 GN/BN clean runs used by
   `exp_fcp_recast.py`), treat the held-out pool as the population and re-draw
   per-client audit folds of the original sizes B=1,000 times.
2. Per redraw, recompute the split-conformal quantile on the redrawn
   certification fold's known-class points (a_cov=0.10, protocol identical to
   `exp_fcp_recast.py`), apply the singleton-set acceptance rule, and evaluate
   the realized accepted risk on the population.
3. Record, per run and aggregated: Pr(realized risk > alpha) for
   alpha in {0.10, 0.20}, plus mean/quantiles of realized risk and acceptance.
4. Write `runs/fcp_recast_resampling.csv` + append a short
   `reports/REPORT_fcp_recast_resampling.md` (protocol, numbers, one-paragraph
   reading). New script: `fedcore/experiments/exp_fcp_recast_resampling.py`.

**Acceptance.** 18,000 evaluations complete; CSV has per-run and aggregate
rows; the report states the violation rate with a Clopper–Pearson 95% bound.
Expected (not required): violation rate ≈ 1.0 — the point is to quantify it.

## Task 2 (GPU, highest value) — FedPD-PROSER seeds 3–4 with logit export

**Goal.** Figure 6(a)'s FedPD frontier uses 3 seeds (`runs/fedpd_cifar10_d5_seed{0,1,2}.npz`);
the Table-5(b) aggregate claims 5 seeds. Export the missing seeds so the
frontier matches the headline seed count.

1. Read `fedcore/experiments/run_fedpd_cifar.py` for the exact CLI and confirm
   the recipe used for seeds 0–2 from `runs/_fedpd_batch.log`: WideResNet-28-10,
   closed-set CE pretraining (~8 rounds) then PROSER fine-tuning — **from-scratch
   PROSER does not converge**, so keep the pretrain-then-fine-tune recipe.
2. Smoke: rerun seed 0 config for 1–2 rounds, confirm the npz schema matches
   (`prop/cert/test_{logits,sm,y_open,client}`); note that the native `sm`
   score is oriented LOWER = more known (downstream consumers negate it).
3. Run seeds 3 and 4 at d=5, then (if GPU budget allows) seeds 3–4 at d=0.5,
   exporting `runs/fedpd_cifar10_d{5,0.5}_seed{3,4}.npz` and per-run CSVs.
4. Re-run the T8 aggregation (`fedcore/aggregate/` — locate the T8 aggregator)
   so `runs/T8_fedosr_bases_agg.csv` reflects 5 seeds consistently.

**Acceptance.** New npz files load; AUROC of `-sm` on the test fold is ≈0.8
(sanity vs seeds 0–2); T8 aggregate row for FedPD-PROSER d=5 reports n_seeds=5
with CertCov within ±0.05 of the 3-seed value (report honestly if not).

## Task 3 (GPU) — Asymmetric-corruption multi-seed sweep

**Goal.** Figure 5(b)'s asymmetric curve is single-seed (dashed). Match the
symmetric protocol: 10 seeds × rates {0.1, 0.2, 0.35} × d ∈ {0.5, 5},
ResNet-GN, asymmetric TRAIN-label noise only, calibration folds clean.

1. Confirm the symmetric protocol from the generator of
   `runs/corruption_curve_seeded.csv` (G=2 grouped headline, best-gamma,
   delta=0.10; CertCov@a = cov_lcb if certified else 0) and mirror it exactly
   with `--noise_type asymmetric`.
2. Smoke one cell (rate 0.2, d=5, seed 0), then launch the 60-run sweep
   (4 GPUs → parallelize by seed).
3. Append rows to `runs/corruption_curve_seeded.csv` (same columns, noise_type
   = asymmetric) — do not overwrite the symmetric rows.

**Acceptance.** 60 rows appended; expected pattern (report honestly either
way): CertCov@0.20 ≈ 0 at rates ≥ 0.2, marginal at 0.1 — the manuscript
sentence "the single-seed asymmetric sweep collapsed identically" gets a
10-seed footing, and the laptop will move the asym curve out of dashed style.

## Task 4 (GPU, major-revision card — start only after 1–3 are done)

Two sub-cards; open a separate report for each.

**4a. CIFAR-100 alpha-frontier.** The CIFAR-100 grid (Table 6) has per-run
CSVs (`runs/r2_cifar100_*`) but only 2 stored npz. Re-export logits npz for
the GN d∈{0.5,5} cells (10 seeds each) with the same training config as the
r2 grid, so the laptop can extend Figure 6(a) with a CIFAR-100 panel
(alphas {0.10, 0.15, 0.20, 0.25, 0.30}).

**4b. FedOSS reproduction.** `third_party` FedOSS is a medical-imaging
codebase without a CIFAR loader (that is why it was deferred). Write a CIFAR-10
loader shim, reproduce at d=5 for 3 seeds, export native-score npz in the
FedPD schema. If AUROC stays at chance after a good-faith budget (as happened
with full FOOGD-SAG), record it as an honest single-paragraph negative in
`reports/` and stop — do not tune indefinitely.

## Task 5 (CPU, optional) — delta-sensitivity appendix table

Sweep delta ∈ {0.05, 0.10, 0.20} at alpha ∈ {0.10, 0.20} on the stored clean
GN/BN logits (G=2 grouped, best-gamma, simultaneous delta/2 budget), starting
from `fedcore/experiments/exp_delta_split.py` and
`runs/delta_split_recompute.csv`. Output `runs/delta_sensitivity.csv`
(columns: backbone, d, alpha, delta, n_seeds, cert_frac_seeds, CertCov_mean,
CertCov_std, certucb_median). Expected: CertCov degrades gracefully as delta
shrinks; the point is an appendix table showing the guarantee level is not a
hidden tuning knob.

---

## After all tasks

1. `git add` the new/changed **code and reports only**; commit per task, e.g.
   `exp: fcp-recast resampling study (B3 hardening)`.
2. Push, then list the produced artifacts (`runs/fcp_recast_resampling.csv`,
   new `fedpd_*.npz`, appended `corruption_curve_seeded.csv`, …) in the final
   report so the laptop can pull/rsync them and regenerate Figures 5–6 via
   `experiments/fedcore/figs/paper_figs_v2.py`.
3. Final report in the fixed format, one 판정 per task
   (strong go / moderate go / warning / fail).

Do NOT: overclaim from smoke runs; judge success by accuracy/AUROC alone; mix
folds; pool accepted points across heterogeneous clients into a single binomial
CP; promote the pooled certificate; reintroduce SRCC/RC-OWPL/pseudo-labeling as
the main object; hide failed commands.
