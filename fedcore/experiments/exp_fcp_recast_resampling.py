"""B3 hardening: resampling the FCP coverage-rule recast.

The point estimate (``exp_fcp_recast.py``) shows that recasting federated
conformal prediction as a singleton-set selector realizes an accepted risk of
0.225-0.382 on 18 clean CIFAR-10 ResNet runs -- far above any risk target
alpha. That is a single number per run. This script gives it the SAME
resampling treatment the certificate gets in ``exp_resampling_validity.py``,
so the paper can quote a *violation rate* (Pr[realized accepted risk > alpha])
with a Clopper-Pearson confidence bound instead of a point estimate.

Protocol (per run, model fixed):
  - population = the held-out pool (certification + test folds concatenated),
    exactly as in ``exp_resampling_validity.py``.
  - re-draw B = 1000 per-client audit folds of the ORIGINAL certification-fold
    per-client sizes, bootstrap (with replacement) within each client, from the
    pool; a single shared rng (np.random.default_rng(0)).
  - per redraw: recompute the split-conformal quantile q on the redrawn audit
    fold's KNOWN-class points (nonconformity s = 1 - softmax_prob(true class),
    a_cov = 0.10, k = ceil((n+1)(1-a_cov)); protocol identical to
    ``exp_fcp_recast.py``), then apply the singleton-set acceptance rule
    C(x) = {y : 1 - p_y(x) <= q}, ACCEPT iff |C(x)| = 1, on the POPULATION and
    record the realized accepted risk (an accepted unknown-class point is an
    accepted error).
  - tally Pr(realized risk > alpha) for alpha in {0.10, 0.20}, plus the mean
    and 5/50/95 quantiles of realized risk and of acceptance.

Runs: the 18 clean CIFAR-10 ResNet runs used by ``exp_fcp_recast.py``
(GN d5 seeds 0-4, GN d0.5 seeds 0-4, BN d5 seeds 0-4, BN d0.5 seeds 0-2) =>
18 x 1000 = 18,000 evaluations. Missing npz are reported and skipped, never
substituted.

Output: runs/fcp_recast_resampling.csv (per-run + aggregate rows) and an
aggregate printout with the Clopper-Pearson 95% interval on the violation rate.

Run: python -m fedcore.experiments.exp_fcp_recast_resampling   (CPU, no torch)
"""
from __future__ import annotations

import csv
import glob
import math
import os

import numpy as np

from fedcore.certificate.cp import cp_lower, cp_upper
from fedcore.experiments.exp_fcp_recast import A_COV, RUNS, softmax

B = 1000
SEED = 0
ALPHAS = (0.10, 0.20)

# The 18 reference runs of exp_fcp_recast.py, capped to the canonical seed sets
# (mirrors exp_fcp_recast.PATTERNS but restricted so the aggregate matches the
# manuscript's 18-run headline exactly => 18,000 redraws).
REF_RUNS = (
    [("GN", 5, s, f"cifar10_d5_resnet18gn_none0.0_seed{s}_logits.npz") for s in range(5)] +
    [("GN", 0.5, s, f"cifar10_d0.5_resnet18gn_none0.0_seed{s}_logits.npz") for s in range(5)] +
    [("BN", 5, s, f"cifar10_d5_resnet18_seed{s}_logits.npz") for s in range(5)] +
    [("BN", 0.5, s, f"cifar10_d0.5_resnet18_none0.0_seed{s}_logits.npz") for s in range(3)]
)


def _fcp_quantile(one_minus_p_true: np.ndarray) -> float:
    """Split-conformal quantile q on known-class nonconformity (identical rule
    to exp_fcp_recast.recast_one)."""
    s = np.sort(one_minus_p_true)
    n = len(s)
    k = min(n, int(math.ceil((n + 1) * (1 - A_COV))))
    return float(s[k - 1])


def resample_one(path: str) -> dict:
    z = np.load(path)
    # population pool = certification + test folds
    logits = np.concatenate([z["cert_logits"], z["test_logits"]], axis=0)
    y_pool = np.concatenate([z["cert_y_open"], z["test_y_open"]])
    cl_pool = np.concatenate([z["cert_client"], z["test_client"]])
    cert_client = z["cert_client"]

    p_pool = softmax(logits)
    one_minus_p = 1.0 - p_pool
    pred_pool = p_pool.argmax(axis=1)
    err_pool = (pred_pool != y_pool) | (y_pool < 0)

    n_clients = int(max(cl_pool.max(), cert_client.max())) + 1
    n_j = np.array([(cert_client == j).sum() for j in range(n_clients)])
    client_idx = [np.flatnonzero(cl_pool == j) for j in range(n_clients)]

    rng = np.random.default_rng(SEED)
    boots = [np.concatenate([rng.choice(client_idx[j], size=n_j[j], replace=True)
                             for j in range(n_clients) if n_j[j] > 0])
             for _ in range(B)]

    risks = np.empty(B)
    accepts = np.empty(B)
    n_nan = 0
    for b, idx in enumerate(boots):
        yk = y_pool[idx]
        known = yk >= 0
        s = one_minus_p[idx][known, yk[known]]          # nonconformity of true class
        q = _fcp_quantile(s)
        sizes = (one_minus_p <= q).sum(axis=1)          # singleton rule on population
        accept = sizes == 1
        if accept.any():
            risks[b] = err_pool[accept].mean()
            accepts[b] = accept.mean()
        else:
            risks[b] = np.nan
            accepts[b] = 0.0
            n_nan += 1
    return dict(risks=risks, accepts=accepts, n_nan=n_nan, n_cal_known=int((z["cert_y_open"] >= 0).sum()))


def _stats(risks, accepts, n_nan):
    valid = risks[~np.isnan(risks)]
    return dict(
        mean_realized_risk=float(np.mean(valid)) if valid.size else float("nan"),
        q05_realized_risk=float(np.quantile(valid, 0.05)) if valid.size else float("nan"),
        median_realized_risk=float(np.median(valid)) if valid.size else float("nan"),
        q95_realized_risk=float(np.quantile(valid, 0.95)) if valid.size else float("nan"),
        mean_acceptance=float(np.mean(accepts)),
        n_nan_draws=int(n_nan),
    )


FIELDS = ["scope", "backbone", "d", "seed", "alpha", "n_draws", "n_violations",
          "violation_rate", "cp95_lo", "cp95_hi", "mean_realized_risk",
          "q05_realized_risk", "median_realized_risk", "q95_realized_risk",
          "mean_acceptance", "n_nan_draws"]


def main() -> None:
    rows = []
    pooled_risks, pooled_accepts = [], []
    pooled_nan = 0
    present = 0
    missing = []
    for bb, d, seed, fname in REF_RUNS:
        path = os.path.join(RUNS, fname)
        if not os.path.exists(path):
            missing.append(fname)
            continue
        present += 1
        r = resample_one(path)
        st = _stats(r["risks"], r["accepts"], r["n_nan"])
        pooled_risks.append(r["risks"])
        pooled_accepts.append(r["accepts"])
        pooled_nan += r["n_nan"]
        for a in ALPHAS:
            viol = int(np.nansum(r["risks"] > a))
            row = dict(scope="run", backbone=bb, d=d, seed=seed, alpha=a,
                       n_draws=B, n_violations=viol,
                       violation_rate=round(viol / B, 6),
                       cp95_lo=round(cp_lower(viol, B, 0.025), 6),
                       cp95_hi=round(cp_upper(viol, B, 0.025), 6),
                       **{k: round(v, 6) for k, v in st.items() if k != "n_nan_draws"},
                       n_nan_draws=st["n_nan_draws"])
            rows.append(row)
        print(f"{bb} d{d} seed{seed}: mean_risk={st['mean_realized_risk']:.4f} "
              f"accept={st['mean_acceptance']:.3f} nan={r['n_nan']}")

    all_risks = np.concatenate(pooled_risks) if pooled_risks else np.array([])
    all_accepts = np.concatenate(pooled_accepts) if pooled_accepts else np.array([])
    agg_st = _stats(all_risks, all_accepts, pooled_nan)
    N = all_risks.size
    for a in ALPHAS:
        viol = int(np.nansum(all_risks > a))
        rows.append(dict(scope="aggregate", backbone="", d="", seed="", alpha=a,
                         n_draws=N, n_violations=viol,
                         violation_rate=round(viol / N, 6) if N else float("nan"),
                         cp95_lo=round(cp_lower(viol, N, 0.025), 6),
                         cp95_hi=round(cp_upper(viol, N, 0.025), 6),
                         **{k: round(v, 6) for k, v in agg_st.items() if k != "n_nan_draws"},
                         n_nan_draws=agg_st["n_nan_draws"]))

    out = os.path.join(RUNS, "fcp_recast_resampling.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {out}  ({present}/{len(REF_RUNS)} reference runs present, "
          f"{N} redraws)")
    if missing:
        print("MISSING reference npz (skipped, not substituted):")
        for m in missing:
            print("   ", m)
    for a in ALPHAS:
        viol = int(np.nansum(all_risks > a))
        lo, hi = cp_lower(viol, N, 0.025), cp_upper(viol, N, 0.025)
        print(f"alpha={a:.2f}: violations {viol}/{N} = {viol/N:.4f}  "
              f"CP95 = [{lo:.4f}, {hi:.4f}]")
    print(f"realized accepted risk (pooled): mean={agg_st['mean_realized_risk']:.4f} "
          f"median={agg_st['median_realized_risk']:.4f} "
          f"[q05={agg_st['q05_realized_risk']:.4f}, q95={agg_st['q95_realized_risk']:.4f}]")


if __name__ == "__main__":
    main()
