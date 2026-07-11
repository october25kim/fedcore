# PROMPT — ws4090 follow-up: decisions + canonical payload (2026-07-10)

> Response to the ws4090 checkpoint report. All blockers are resolved below;
> this brief is **self-contained** (the WORK tree is not a git repo, so every
> required file is embedded verbatim). Execute in order. Keep the fixed report
> format; never hide failed commands.

## Decisions

- **D1 (Task 1): GO — with canonical code, not blind reconstruction.**
  The missing `exp_fcp_recast.py`, its reference output `runs/fcp_recast.csv`
  (ground truth, 18 rows), and the resampling generator
  `exp_resampling_validity.py` are embedded below. The resampling generator was
  deleted from the tree during the package refactor and has been **recovered
  from git history (commit dfafded) with imports ported** to the current
  `fedcore` package; it is the generator of `runs/resampling_validity*.csv`.
- **D2 (Task 2): already satisfied — verification only, no retraining.**
  Your seed-3/4 npz findings are accepted (AUROC(-sm) 0.843/0.848/0.850/0.757;
  note the low d0.5 s4 value in the report, no action). Add the artifacts to
  the sync-back manifest (Section "Sync").
- **D3 (Task 3): GO** at the confirmed scope: 60 runs, asymmetric TRAIN-label
  noise, ResNet-GN, 10 seeds x rates {0.1, 0.2, 0.35} x d in {0.5, 5},
  docker only, mirroring the symmetric protocol of
  `scripts/ws4090/certify_grid.py`; **append** rows to
  `runs/corruption_curve_seeded.csv` (never overwrite symmetric rows).
- **D4 (Task 5): GO** (CPU, low risk, as originally specified).
- **D5 (Task 4): HOLD** until Tasks 1, 3, 5 are done.

## Task 1 — exact procedure

**Phase 0 (reproduce the point estimates).**
1. Write the two embedded scripts to `fedcore/experiments/exp_fcp_recast.py`
   and `fedcore/experiments/exp_resampling_validity.py` (byte-for-byte).
2. Run `python3 -m fedcore.experiments.exp_fcp_recast` (CPU; needs numpy only).
3. Diff the produced `runs/fcp_recast.csv` against the reference CSV embedded
   below. Rows must match to 4 decimals **for every npz present on this
   server**. The reference set is 18 runs: GN d5 seeds 0-4, GN d0.5 seeds 0-4,
   BN(`resnet18`) d5 seeds 0-4, BN d0.5 seeds 0-2. You reported only 4 plain-BN
   clean npz: run the intersection, list every missing npz in the sync
   manifest, and do NOT substitute other runs.
4. Any mismatch beyond float noise -> stop, report, wait.

**Phase 1 (resampling extension).**
5. Smoke `python3 -m fedcore.experiments.exp_resampling_validity 0 1` and
   confirm the output schema against the docstring.
6. Write `fedcore/experiments/exp_fcp_recast_resampling.py`: reuse the redraw
   machinery of `exp_resampling_validity.py` (held-out pool = population,
   per-client audit folds of original sizes, B=1,000, rng seed 0), but per
   redraw recompute the split-conformal quantile on the redrawn certification
   fold's known-class points (a_cov=0.10, protocol identical to
   `exp_fcp_recast.py`) and evaluate the singleton-rule realized accepted risk
   on the population. Tally Pr(realized risk > alpha) for alpha in
   {0.10, 0.20} per run and aggregate, with a Clopper-Pearson 95% bound.
7. Outputs: `runs/fcp_recast_resampling.csv` +
   `reports/REPORT_fcp_recast_resampling.md`.

## Sync (produce at the end, before the final report)

`runs/SYNC_MANIFEST_ws4090_2026-07-10.txt` with md5sums, listing at minimum:
- server -> laptop: `fedpd_cifar10_d5_seed{3,4}.npz`,
  `fedpd_cifar10_d0.5_seed{3,4}.npz`, current `T8_fedosr_bases_agg.csv`,
  `fcp_recast_resampling.csv`, appended `corruption_curve_seeded.csv`,
  `delta_sensitivity.csv`, any clean BN npz absent from the reference-run list,
  and every log of this campaign;
- laptop -> server (already delivered inline here): `exp_fcp_recast.py`,
  `exp_resampling_validity.py`, reference `fcp_recast.csv`.

---

## EMBEDDED FILE 1/3 — `fedcore/experiments/exp_fcp_recast.py`

```python
"""B3: coverage-rule recast of federated conformal prediction controls the
wrong functional.

Claim under test. Federated CP certifies closed-set prediction-set coverage.
Its authors' own selective-classification demonstration — accept a point when
the conformal prediction set is a singleton — is a heuristic *without* a
guarantee on the accepted error rate. We recast FCP as such a selector on the
stored real logits and measure the realized accepted selective risk on the
held-out test fold, asking whether anything ties it to a risk target alpha.

Protocol (post-hoc, model fixed; split conformal on the certification fold):
  - calibration = known-class points of the certification fold (unknowns carry
    no closed-set label, so closed-set CP cannot use them — that is the point);
    nonconformity s_i = 1 - softmax_prob(true class).
  - q = the ceil((n+1)(1-a_cov))/n empirical quantile of {s_i}, a_cov = 0.10
    (the standard 90%-coverage rule), pooled across clients as in the natural
    single-quantile recast.
  - test fold: C(x) = {y : 1 - p_y(x) <= q}; ACCEPT iff |C(x)| = 1, predict its
    element; an accepted unknown-class point is always an accepted error.
  - record acceptance rate, realized accepted risk, and whether the realized
    risk exceeds alpha in {0.10, 0.20}.

Runs: the 18 clean CIFAR-10 ResNet runs (GN/BN, d in {0.5, 5}).
Output: runs/fcp_recast.csv + aggregate printout.

Run: python -m fedcore.experiments.exp_fcp_recast   (CPU, no torch)
"""
from __future__ import annotations

import csv
import glob
import math
import os

import numpy as np

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "runs")
A_COV = 0.10
ALPHAS = (0.10, 0.20)

PATTERNS = [
    ("GN", 5, "cifar10_d5_resnet18gn_none0.0_seed*_logits.npz"),
    ("GN", 0.5, "cifar10_d0.5_resnet18gn_none0.0_seed*_logits.npz"),
    ("BN", 5, "cifar10_d5_resnet18_seed*_logits.npz"),
    ("BN", 0.5, "cifar10_d0.5_resnet18_none0.0_seed*_logits.npz"),
]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def recast_one(path):
    z = np.load(path)
    p_cal = softmax(z["cert_logits"])
    y_cal = z["cert_y_open"]
    known = y_cal >= 0
    s = 1.0 - p_cal[known, y_cal[known]]
    n = len(s)
    k = min(n, int(math.ceil((n + 1) * (1 - A_COV))))
    q = np.sort(s)[k - 1]

    p_te = softmax(z["test_logits"])
    y_te = z["test_y_open"]
    sets = (1.0 - p_te) <= q                      # per-class membership
    sizes = sets.sum(axis=1)
    accept = sizes == 1
    pred = p_te.argmax(axis=1)
    err = (pred != y_te) | (y_te < 0)             # accepted unknown = error
    acc_rate = float(accept.mean())
    risk = float(err[accept].mean()) if accept.any() else float("nan")
    return acc_rate, risk, n


def main():
    rows = []
    for bb, d, pat in PATTERNS:
        for path in sorted(glob.glob(os.path.join(RUNS, pat))):
            seed = path.split("seed")[-1].split("_")[0]
            acc, risk, n_cal = recast_one(path)
            rows.append(dict(backbone=bb, d=d, seed=seed, a_cov=A_COV,
                             n_cal_known=n_cal,
                             accept_rate=round(acc, 4),
                             realized_accepted_risk=round(risk, 4),
                             **{f"exceeds_a{int(a*100)}": int(risk > a)
                                for a in ALPHAS}))
    out = os.path.join(RUNS, "fcp_recast.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    risks = np.array([r["realized_accepted_risk"] for r in rows])
    print(f"wrote {out} ({len(rows)} runs)")
    print(f"realized accepted risk: min={risks.min():.3f} "
          f"median={np.median(risks):.3f} max={risks.max():.3f}")
    for a in ALPHAS:
        exc = sum(r[f"exceeds_a{int(a*100)}"] for r in rows)
        print(f"exceeds alpha={a:.2f}: {exc}/{len(rows)} runs")
    print(f"mean acceptance rate: {np.mean([r['accept_rate'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
```

## EMBEDDED FILE 2/3 — `fedcore/experiments/exp_resampling_validity.py`
(recovered from commit dfafded; imports ported to the current package)

```python
"""Resampling validity experiment (E1): empirical coverage of the certificate
over calibration randomness, on REAL federated logits.

Rationale. The certificate's guarantee is a probability over the draw of the
certification fold, not over training seeds. Five training seeds cannot
estimate a false-certificate rate; resampling the audit fold at a FIXED model
can. This script treats each run's held-out pool (certification + test folds)
as the ground-truth population, re-draws per-client audit folds B times, and
counts violations of the stratified certificate (Theorem 1 / grouped variant)
against the population worst-stratum conditional risk
sup_lambda R_sel(lambda) = max_g r_g.

Grouping rule (fixed a priori, matching A6): clients are merged into G
contiguous groups via np.array_split(range(J), G).

CPU-only: consumes runs/*_logits.npz produced by run_cifar.py.

Usage:
  python exp_resampling_validity.py [shard_start] [shard_end]

Outputs:
  runs/resampling_validity[_s_e].csv   per (run, alpha, G, gamma) tallies
  stdout aggregate with the Clopper-Pearson bound on the violation rate
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

from fedcore.certificate.cp import cp_upper
from fedcore.certify import conditional_risk_certificate
from fedcore.scores import compute_score
from fedcore.selector import choose_threshold, open_set_error

B = 1000                       # resamples per run (shared across configs)
DELTA = 0.10
ALPHAS = (0.10, 0.20)
GROUPS = (5, 2)                # 5 = per-client stratified; 2 = headline grouping
GAMMAS = (0.5, 0.7, 1.0)
SCORE = "msp"                  # fixed-score protocol (headline)
SEED = 0


def load_run(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    out = {}
    for fold in ("prop", "cert", "test"):
        logits = z[f"{fold}_logits"]
        out[fold] = {
            "score": compute_score(SCORE, logits),
            "pred": logits.argmax(axis=1),
            "y_open": z[f"{fold}_y_open"],
            "client": z[f"{fold}_client"],
        }
    return out


def group_of_client(n_clients: int, G: int) -> np.ndarray:
    g = np.empty(n_clients, dtype=int)
    for gi, block in enumerate(np.array_split(np.arange(n_clients), G)):
        g[block] = gi
    return g


def worst_group_risk(err: np.ndarray, acc: np.ndarray, grp: np.ndarray,
                     G: int) -> float:
    worst = 0.0
    for g in range(G):
        m = (grp == g) & acc
        if m.sum() > 0:
            worst = max(worst, float(err[m].mean()))
    return worst


def group_counts(err: np.ndarray, acc: np.ndarray, grp: np.ndarray,
                 G: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.zeros(G, dtype=int)
    K = np.zeros(G, dtype=int)
    n = np.zeros(G, dtype=int)
    for g in range(G):
        m = grp == g
        n[g] = int(m.sum())
        a = m & acc
        A[g] = int(a.sum())
        K[g] = int(err[a].sum())
    return A, K, n


def main() -> None:
    rng = np.random.default_rng(SEED)
    paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..",
                                          "runs", "*_logits.npz")))
    if not paths:
        paths = sorted(glob.glob("runs/*_logits.npz"))
    if len(sys.argv) >= 3:
        paths = paths[int(sys.argv[1]):int(sys.argv[2])]

    rows = []
    tot_dep = tot_viol = 0
    for path in paths:
        run = os.path.basename(path).replace("_logits.npz", "")
        d = load_run(path)
        n_clients = int(max(d["cert"]["client"].max(),
                            d["test"]["client"].max())) + 1
        pool = {k: np.concatenate([d["cert"][k], d["test"][k]])
                for k in ("score", "pred", "y_open", "client")}
        pool_err = open_set_error(pool["pred"], pool["y_open"])
        n_j = np.array([(d["cert"]["client"] == j).sum()
                        for j in range(n_clients)])
        client_idx = [np.flatnonzero(pool["client"] == j)
                      for j in range(n_clients)]
        # one shared set of B bootstrap audit folds per run
        boots = [np.concatenate([rng.choice(client_idx[j], size=n_j[j],
                                            replace=True)
                                 for j in range(n_clients) if n_j[j] > 0])
                 for _ in range(B)]

        for alpha in ALPHAS:
            for gamma in GAMMAS:
                sel = choose_threshold(d["prop"]["score"], d["prop"]["pred"],
                                       d["prop"]["y_open"],
                                       gamma=gamma, alpha=alpha)
                if not sel.feasible:
                    for G in GROUPS:
                        rows.append((run, alpha, G, gamma,
                                     "infeasible-proposal", 0, 0, np.nan))
                    continue
                pool_acc = pool["score"] >= sel.threshold
                for G in GROUPS:
                    grp_c = group_of_client(n_clients, G)
                    pool_grp = grp_c[pool["client"]]
                    true_worst = worst_group_risk(pool_err, pool_acc,
                                                  pool_grp, G)
                    n_dep = n_viol = 0
                    for idx in boots:
                        acc = pool_acc[idx]
                        err = pool_err[idx]
                        grp = pool_grp[idx]
                        A, K, n = group_counts(err, acc, grp, G)
                        res = conditional_risk_certificate(
                            A, K, n, DELTA, Lambda="simplex")
                        if res.U <= alpha:
                            n_dep += 1
                            if true_worst > alpha:
                                n_viol += 1
                    tot_dep += n_dep
                    tot_viol += n_viol
                    rows.append((run, alpha, G, gamma, "ok", n_dep, n_viol,
                                 round(true_worst, 4)))
        print(f"{run}: done")

    suffix = f"_{sys.argv[1]}_{sys.argv[2]}" if len(sys.argv) >= 3 else ""
    out = os.path.join("runs", f"resampling_validity{suffix}.csv")
    with open(out, "w") as f:
        f.write("run,alpha,G,gamma,status,n_deploy_of_B,n_violation,"
                "true_worst_group_risk\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    ucb = cp_upper(tot_viol, max(tot_dep, 1), 0.05)
    print("\n=== aggregate (this shard) ===")
    print(f"deployments: {tot_dep}  violations: {tot_viol}  "
          f"CP95 UCB on violation rate: {ucb:.5f}  (delta = {DELTA})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

## EMBEDDED FILE 3/3 — reference `runs/fcp_recast.csv` (ground truth)

```csv
backbone,d,seed,a_cov,n_cal_known,accept_rate,realized_accepted_risk,exceeds_a10,exceeds_a20
GN,5,0,0.1,1800,0.5545,0.2568,1,1
GN,5,1,0.1,1800,0.9693,0.3633,1,1
GN,5,2,0.1,1800,0.8891,0.3427,1,1
GN,5,3,0.1,1800,0.9864,0.3665,1,1
GN,5,4,0.1,1800,0.9903,0.3819,1,1
GN,0.5,0,0.1,1800,0.4864,0.2752,1,1
GN,0.5,1,0.1,1800,0.9253,0.3469,1,1
GN,0.5,2,0.1,1800,0.8728,0.3366,1,1
GN,0.5,3,0.1,1800,0.9463,0.3635,1,1
GN,0.5,4,0.1,1800,0.9054,0.358,1,1
BN,5,0,0.1,1800,0.5914,0.2579,1,1
BN,5,1,0.1,1800,0.9389,0.3539,1,1
BN,5,2,0.1,1800,0.9272,0.3584,1,1
BN,5,3,0.1,1800,0.9498,0.354,1,1
BN,5,4,0.1,1800,0.9638,0.3605,1,1
BN,0.5,0,0.1,1800,0.4195,0.2254,1,1
BN,0.5,1,0.1,1800,0.9366,0.3585,1,1
BN,0.5,2,0.1,1800,0.8086,0.3191,1,1
```
