"""M9: oracle-comparison consolidation on stored GN (resnet18gn) CIFAR-10 d=5 logits.

Consolidates -- under ONE protocol -- three accepted-coverage numbers per seed at
alpha=0.20, so a single comparison figure can be drawn:

  1. fedcore_grouped_G2   Fed-CORE grouped-stratified certificate (G=2, Theorem 1/1',
                          Lambda=box). Reports the certified coverage LCB. Distribution-
                          free, secure-aggregatable within groups. Does NOT peek at test
                          labels. `valid` = did the risk certificate hold on the test fold
                          (test_risk <= cert_risk_ucb).
  2. pooled_matched       Matched-mixture POOLED diagnostic (Proposition 3, G=1: all
                          accepted points as ONE binomial). Tighter coverage LCB, but only
                          valid under matched i.i.d. mixture. Same `valid` check.
  3. oracle_test_peek     Test-peeking oracle: the largest test coverage whose realized
                          test risk <= alpha, computed WITH test labels. The unachievable
                          upper bound (uses_test_labels=True, valid=True by construction).

Same MSP score head, same seed-0 pooled repartition (cert_frac 0.5, test_frac 0.2),
same alpha/delta as the T9 diagnostics -> the three methods are directly comparable.

CSV: runs/oracle_comparison.csv
  columns: method,seed,coverage_or_lcb,valid,uses_test_labels,cert_risk_ucb,test_risk,alpha

Run: python experiments/fedcore/exp_oracle_comparison.py   (CPU, no torch)
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

from fedcore.certify import certify_best_gamma_grouped
from fedcore.grouping import make_group_map, repartition_trusted_pool, views_from_parts
from fedcore.io_utils import atomic_write_csv
from fedcore.selector import open_set_error

GLOB = "runs/cifar10_d5_resnet18gn_none0.0_seed[0-9]_logits.npz"
ALPHA, DELTA = 0.20, 0.10
CERT_FRAC, TEST_FRAC = 0.5, 0.2
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
BOX, MARGIN, SEED = 0.15, 0.01, 0
OUT = "runs/oracle_comparison.csv"
FIELDS = ["method", "seed", "coverage_or_lcb", "valid", "uses_test_labels",
          "cert_risk_ucb", "test_risk", "alpha"]


def _pool(npz):
    z = np.load(npz)
    return {k: np.concatenate([z[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}


def _oracle_test_peek(test_view, alpha):
    """Largest test coverage with realized test risk <= alpha (uses test labels)."""
    score = test_view["score"]
    err = open_set_error(test_view["pred"], test_view["y_open"]).astype(float)
    order = np.argsort(-score)                       # accept highest-score first
    cum_err = np.cumsum(err[order])
    k = np.arange(1, len(order) + 1)
    risk = cum_err / k
    ok = risk <= alpha
    if not ok.any():
        return 0.0, float("nan")
    kbest = int(np.max(np.where(ok)[0]))
    cov = (kbest + 1) / len(order)
    return float(cov), float(risk[kbest])


def _certify_G(parts, n_clients, G):
    gmap = make_group_map(n_clients, G)
    views = views_from_parts(parts, "msp")
    return certify_best_gamma_grouped(
        views["prop"], views["cert"], views["test"], score_name="msp",
        group_map=gmap, G=G, gammas=GAMMAS, alpha=ALPHA, delta=DELTA,
        Lambda="box", box=BOX, seed=SEED, margin=MARGIN)


def main():
    rows = []
    for npz in sorted(glob.glob(GLOB)):
        seed = int(re.search(r"seed(\d+)", os.path.basename(npz)).group(1))
        pool = _pool(npz)
        n_clients = int(pool["client"].max()) + 1
        parts = repartition_trusted_pool(pool, CERT_FRAC, TEST_FRAC, seed=SEED)

        g2 = _certify_G(parts, n_clients, G=2)
        pooled = _certify_G(parts, n_clients, G=1)
        test_view = views_from_parts(parts, "msp")["test"]
        ocov, orisk = _oracle_test_peek(test_view, ALPHA)

        rows.append({"method": "fedcore_grouped_G2", "seed": seed,
                     "coverage_or_lcb": round(float(g2["cert_coverage_lcb"]), 4),
                     "valid": int(float(g2["test_risk"]) <= float(g2["cert_risk_ucb"]) + 1e-12),
                     "uses_test_labels": 0,
                     "cert_risk_ucb": round(float(g2["cert_risk_ucb"]), 4),
                     "test_risk": round(float(g2["test_risk"]), 4), "alpha": ALPHA})
        rows.append({"method": "pooled_matched", "seed": seed,
                     "coverage_or_lcb": round(float(pooled["cert_coverage_lcb"]), 4),
                     "valid": int(float(pooled["test_risk"]) <= float(pooled["cert_risk_ucb"]) + 1e-12),
                     "uses_test_labels": 0,
                     "cert_risk_ucb": round(float(pooled["cert_risk_ucb"]), 4),
                     "test_risk": round(float(pooled["test_risk"]), 4), "alpha": ALPHA})
        rows.append({"method": "oracle_test_peek", "seed": seed,
                     "coverage_or_lcb": round(ocov, 4), "valid": 1, "uses_test_labels": 1,
                     "cert_risk_ucb": float("nan"), "test_risk": round(orisk, 4), "alpha": ALPHA})
        print(f"seed {seed}: G2 lcb={g2['cert_coverage_lcb']:.3f} (valid={rows[-3]['valid']}) | "
              f"pooled lcb={pooled['cert_coverage_lcb']:.3f} (valid={rows[-2]['valid']}) | "
              f"oracle cov={ocov:.3f}")

    if not rows:
        print(f"[warn] no GN d5 logits matched {GLOB}"); return
    atomic_write_csv(OUT, FIELDS, rows)
    print(f"\nsaved {OUT}  ({len(rows)} rows)")
    for m in ("fedcore_grouped_G2", "pooled_matched", "oracle_test_peek"):
        vals = [r["coverage_or_lcb"] for r in rows if r["method"] == m]
        valid = [r["valid"] for r in rows if r["method"] == m]
        print(f"  {m:>20}: cov {np.mean(vals):.3f}+/-{np.std(vals):.3f}  "
              f"valid {sum(valid)}/{len(valid)}")


if __name__ == "__main__":
    main()
