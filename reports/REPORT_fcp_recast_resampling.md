# FCP coverage-rule recast — resampling validity (B3 hardening)

> Status (read first). The point estimate in `exp_fcp_recast.py` (18 clean
> CIFAR-10 ResNet runs; realized accepted risk **0.225–0.382**, exceeding
> `alpha` in 18/18 runs) is now given the same resampling treatment the
> certificate receives in `exp_resampling_validity.py`. Recasting federated
> conformal prediction (FCP) as a singleton-set selector violates the risk
> target on **essentially every** redraw: the empirical violation rate is
> **1.0000** at `alpha=0.10` and **0.9999** at `alpha=0.20`, with Clopper–Pearson
> 95% intervals `[0.9998, 1.0000]` and `[0.9997, 1.0000]`. The wrong-functional
> claim is not an artifact of one calibration draw.

## 1. What this quantifies

FCP certifies closed-set prediction-set *coverage*. Its authors' own
selective-classification demonstration — accept a point iff its conformal
prediction set is a singleton — is a heuristic with **no** guarantee on the
accepted *error* rate. `exp_fcp_recast.py` measures the realized accepted risk
of that selector once per run; a single number cannot separate "the rule is
unsafe" from "this particular calibration fold was unlucky." Resampling the
audit fold at a **fixed model** turns the one number into a distribution and a
violation *rate* with a confidence bound — the estimand the certificate is held
to (Pr[false certificate] ≤ delta), applied here to the FCP recast.

## 2. Protocol

Identical calibration/acceptance rule to `exp_fcp_recast.py`; redraw machinery
identical to `exp_resampling_validity.py`.

- **Runs.** The 18 clean CIFAR-10 ResNet runs used by `exp_fcp_recast.py`:
  GN d5 seeds 0–4, GN d0.5 seeds 0–4, BN(resnet18) d5 seeds 0–4, BN d0.5
  seeds 0–2. All 18 npz present on ws4090 (0 missing, 0 substituted).
- **Population.** Per run, the held-out pool = certification ∪ test folds
  (as in `exp_resampling_validity.py`).
- **Redraws.** `B = 1000` per-client audit folds of the **original**
  certification-fold per-client sizes, bootstrap (with replacement) *within*
  each client, drawn from the pool; single shared `np.random.default_rng(0)`.
  Total `18 × 1000 = 18,000` evaluations.
- **Per redraw.** (i) recompute the split-conformal quantile `q` on the
  redrawn fold's **known-class** points — nonconformity `s = 1 − p_true`,
  `a_cov = 0.10`, `k = ceil((n+1)(1−a_cov))`; (ii) apply the singleton rule
  `C(x) = {y : 1 − p_y(x) ≤ q}`, ACCEPT iff `|C(x)| = 1`, on the **population**;
  (iii) record realized accepted risk (an accepted unknown-class point is an
  accepted error) and acceptance rate.
- **Report.** Pr(realized risk > `alpha`) for `alpha ∈ {0.10, 0.20}`, per run
  and aggregate, with a two-sided Clopper–Pearson 95% interval
  (`cp_lower/cp_upper` at eps = 0.025).

Outputs: `runs/fcp_recast_resampling.csv` (18 per-run × 2 alpha rows + 2
aggregate rows), generator `fedcore/experiments/exp_fcp_recast_resampling.py`.

## 3. Numbers

Aggregate over all 18,000 redraws:

| alpha | violations / draws | violation rate | CP95 interval |
|------:|-------------------:|---------------:|:--------------|
| 0.10  | 18000 / 18000      | 1.000000       | [0.999795, 1.000000] |
| 0.20  | 17999 / 18000      | 0.999944       | [0.999691, 0.999999] |

Pooled realized accepted risk: **mean 0.3260, median 0.3447**, 5–95%
`[0.2324, 0.3663]`; pooled acceptance rate **0.828**. Exactly one redraw (out
of 18,000) achieved realized risk ≤ 0.20; none achieved ≤ 0.10.

Per-run resampling means track the `exp_fcp_recast.py` point estimates closely
(the small gap is because the point estimate scores the test fold only, while
resampling scores the pool and averages over 1000 bootstraps):

| run | point est. (test) | resampling mean (pool) | acceptance |
|-----|------------------:|-----------------------:|-----------:|
| GN d5 s0    | 0.2568 | 0.2597 | 0.558 |
| GN d0.5 s0  | 0.2752 | 0.2614 | 0.475 |
| BN d5 s0    | 0.2579 | 0.2505 | 0.578 |
| BN d0.5 s0  | 0.2254 | 0.2261 | 0.419 |
| GN d5 s4    | 0.3819 | 0.3585 | 0.958 |

(The lowest-risk runs are also the lowest-acceptance runs — seed 0 across
backbones — consistent with a threshold that only accepts the most confident
points; even there the risk is 2.3–2.8×, never below, the `alpha=0.10` target.)

## 4. Reading (one paragraph)

Recasting federated conformal prediction as a singleton-set accept rule
controls the **wrong functional**: across 18,000 fixed-model calibration
redraws it deploys an accepted-error rate of ≈0.23–0.37 while a 90%-coverage
rule tries to hold set-coverage, exceeding a 10% accepted-risk target in
**every** redraw and a 20% target in all but one (violation rate 1.0000 and
0.9999; Clopper–Pearson 95% lower bounds 0.9998 and 0.9997). The manuscript's
point-estimate sentence ("realized accepted risk 0.225–0.382, exceeding alpha
in 18/18 runs") is therefore not a calibration-draw artifact but a stable
property of the coverage rule, now quotable as a violation rate with a
distribution-free confidence bound — the necessity of certifying the accepted
*risk* directly, rather than reusing a coverage guarantee, holds under
resampling.
