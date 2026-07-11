# Fed-CORE execution report — 2026-06-28 (4070 session)

Sync target: Mac-side `Fed-CORE_draft.md` §5 / Table 1 / headline. All numbers below
are reproducible CPU post-hoc runs on already-exported logits unless noted. Protocol
matched to the CIFAR headline: **fixed single score, fixed worst-group G=2,
cert_frac=0.5, box-Lambda best-gamma, margin=0.01, delta=0.10**. Judge by `cert_*`.

---

## PRIORITY 1 — alpha=0.20 multi-seed (CLOSED Gap 1: was single-config)

Artifact: `runs/agg_alpha20.csv` — `python experiments/fedcore/aggregate.py --alpha 0.20`.
cifar10, ResNet-GN clean, 5 clients, worst-group **G=2**, fixed **MSP**, cert_frac=0.5.

| cell | CertCov@0.20 | n_pass | per-seed | max cert_ucb | max test_risk |
|------|--------------|--------|----------|--------------|---------------|
| d=5   | **0.392 +/- 0.097** | 5/5 | 0.312 / 0.445 / 0.246 / 0.457 / 0.502 | 0.166 | 0.122 |
| d=0.5 | **0.353 +/- 0.130** | 5/5 | 0.127 / 0.460 / 0.285 / 0.457 / 0.437 | 0.180 | 0.122 |

- **0/10 false certificates** across the 10 runs (every cert_ucb <= 0.20 AND every
  empirical test_risk <= 0.20).
- DRAFT EDIT: replace "single-config ~0.29" (old T4 matched-risk, seed0) with the two
  multi-seed cells above. The ~0.29 T4 number stays valid as a *matched-risk vs SOTA*
  comparison point, but it is no longer the evidence for the alpha=0.20 positive.

## PRIORITY 4 — real-data ablations (mirror synthetic Figs 9/10)

### A4-real — calibration-budget sweep
Artifacts: `runs/ablation_calib_budget.csv`, `figs/ablation_calib_budget.png`
(`exp_ablation_calib_budget.py`). d=5 ResNet-GN, model fixed, prop/test fixed, cert
fold grown. G=2 @ alpha=0.10, 5 model seeds.

| per-group accepted | cert_ucb (mean) | CertCov@0.10 | n_pass |
|--------------------|-----------------|--------------|--------|
| 15  | 0.575 | 0 | 0/5 |
| 90  | 0.287 | 0 | 0/5 |
| 148 | 0.208 | 0 | 0/5 |
| 185 | **0.182** | **0.061 +/- 0.100** | 2/5 |

Reading: cert_ucb falls monotonically with audit budget and alpha=0.10 turns
non-vacuous at the largest budget (Theorem-2 floor). Confirms synthetic A4. Honest
caveat: only the largest budget crosses; a full-seed crossover needs a larger trusted
pool than CIFAR-10's test split affords.

### A5-real — unknown-proportion sweep
Artifacts: `runs/ablation_unknown_prop.csv`, `figs/ablation_unknown_prop.png`
(`exp_ablation_unknown_prop.py`). Cert-fold unknowns subsampled to rho * p_deploy
(p=0.30); prop/test keep full rate. d=5 ResNet-GN, alpha=0.20, 5 seeds x 40 reps.

| rho | cert unk frac | cert_ucb | realized risk | empirical coverage | verdict |
|-----|---------------|----------|---------------|--------------------|---------|
| 0.25 | 0.075 | 0.070 | 0.141 | **0.005** | anti-conservative |
| 0.50 | 0.150 | 0.102 | 0.141 | **0.010** | anti-conservative |
| 0.75 | 0.225 | 0.137 | 0.141 | **0.364** | anti-conservative |
| 1.00 | 0.300 | 0.174 | 0.141 | **1.000** | ok (>= 1-delta) |

Reading: matched unknowns -> coverage >= 1-delta; under-representing unknowns drops
cert_ucb below the (constant) realized risk and coverage collapses. Confirms synthetic
A5 — supports the §3 "calibration must contain LABELED unknowns at the deployment rate;
distribution-free is w.r.t. the calibration distribution" caveat with real data.

## PRIORITY 2 — covtype multi-seed (REFRAMED: not a stable positive)

Artifact: `runs/agg_covtype.csv` — `python experiments/fedcore/aggregate_covtype.py`
(5 seeds; `run_tabular.py --dataset covtype --seed {0..4}`, CPU ~1.5s each).

| protocol | alpha=0.20 | alpha=0.25 | alpha=0.30 |
|----------|------------|------------|------------|
| honest fixed-MSP, G=2 (= CIFAR protocol) | 0/5 | 0/5 | 0/5 |
| honest fixed neg_entropy, G=2 (best single score) | 0.07+/-0.14 (1/5) | 0.12+/-0.17 (2/5) | 0.20+/-0.25 (3/5) |
| old selection (best score x G in {1,2,3}, incl pooled) | 0.10+/-0.17 (2/5) | 0.15+/-0.22 (3/5) | 0.22+/-0.25 (4/5) |

- The old headline "covtype 0.433 @ alpha=0.20" = `make_handoff.py::covtype_frontier`
  on a SINGLE seed with best-of-4-scores x best-of-G in {1,2,3} (incl. pooled G=1).
- All positives concentrate in seed 0 (and weakly seed 3); std > mean everywhere.
- Cause: federated LINEAR logreg realized r_hat ~ 0.14-0.24, too close to alpha
  (Theorem-2 sample requirement ~ (alpha - r_hat)^-2 explodes).
- DRAFT EDIT (decision: reframe with full variance, do NOT drop): present covtype as a
  **seed-variable, selection-optimistic breadth probe**, e.g. "under a selection
  protocol covtype reaches CertCov@0.20 = 0.10+/-0.17 (2/5 seeds), entirely driven by
  one seed; under the matched fixed-score worst-group protocol it is 0/5 — the tabular
  domain is at the feasibility edge, not a stable positive." Do NOT cite "0.43" as a
  second-domain positive. Stabilizing it honestly would need a stronger federated
  tabular model (lower r_hat).

## PRIORITY 3 — real FedOSR base model (DEFERRED this session)

- The current `score_norm` in `scores.py` is an explicit **logit-space proxy** (L2 norm
  of the logit vector), not the faithful FOOGD feature-space score-norm. A faithful
  FOOGD detector needs **penultimate features exported** from the trained ResNet-GN =
  a GPU re-run; deferred to a GPU session per stop-and-ask.
- INTERIM LABELING (apply in draft now): wherever `score_norm` appears, call it a
  "representative FedOSR-style score (logit-space score-norm proxy)", NOT a faithful
  FOOGD/FedPD/FedOSS open-set score. Do not claim certification "on top of a real
  FedOSR base model" until the feature-space export is done.
- TODO (next GPU session): add penultimate-feature export to `run_cifar.py`, implement
  feature-space FOOGD score-norm, and certify CertCov@alpha on that genuine score.

---

## Writing fixes to apply in the Mac draft

1. **Denominators explicit.** State "0/N false certificates across <N> runs" with the
   number, not "no false certificate". This session: PRIORITY 1 = **0/10**; A5 matched
   (rho=1) coverage 1.00 over 200 evals.
2. **"valid" usage.** Keep "valid" = theorem-validity, plus the empirical
   "valid in all tested settings (0/N false certificates)". Do NOT write
   "valid everywhere".
3. **Proposition 3 stays subordinate** (Gap 2 / roster-composition coupling still
   open). Do not promote it; G=1 pooled remains a near-IID-only bonus.
4. **Self-training = contamination-control only.** No accuracy-gain claim (F8 honesty
   stands).
5. **Distribution-free wording.** Keep "distribution-free w.r.t. the audited deployment
   distribution"; A5-real now empirically backs the "calibration must match the
   deployment unknown rate" caveat.
6. **covtype** — see PRIORITY 2 reframe above (remove the "0.43" positive claim).
7. **FOOGD/score_norm** — see PRIORITY 3 labeling above (proxy, not faithful).

## Reproduce

```bash
python experiments/fedcore/aggregate.py --alpha 0.20            # runs/agg_alpha20.csv
python experiments/fedcore/exp_ablation_calib_budget.py         # A4-real
python experiments/fedcore/exp_ablation_unknown_prop.py         # A5-real
python experiments/fedcore/run_tabular.py --dataset covtype --seed {0..4}
python experiments/fedcore/aggregate_covtype.py                 # runs/agg_covtype.csv
```
