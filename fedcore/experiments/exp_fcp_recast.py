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
