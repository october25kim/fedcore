#!/usr/bin/env python3
"""R6 attempt-1: client-SIMPLEX (Theorem 1) certificate on the stored FedPD-PROSER
J=5 detector logits -- CPU, no retraining. Reuses the EXACT T8 detector protocol
(native -sm accept score, cert_frac=0.5 enlarged audit, gammas {0.5,0.7,1.0},
delta=0.10, seed-0 pooled repartition) but certifies under the FULL SIMPLEX
(Lambda="simplex", per-client G=J, Ubar = max_j rbar_j) instead of grouped G=2.

This is the hardest deployment set (worst-client), so a non-vacuous cell here is a
strong client-simplex positive; a clean failure is the measured PRICE of full
client-simplex robustness (reportable, per the R6 brief).

  -> runs/simplex_positive.csv
     schema: J,seed,alpha,cert_risk_ucb,cert_coverage_lcb,cert_n_min_client,certified

Run: python experiments/fedcore/exp_r6_simplex_positive.py   (CPU, no torch)
"""
from __future__ import annotations

import csv
import glob
import os
import re

import numpy as np

from fedcore.certify import certify_best_gamma
from fedcore.selector import choose_threshold, counts_per_client
# reuse the T8 detector helpers verbatim -> identical score head / repartition / knobs
from fedcore.aggregate.t8 import (
    CERT_FRAC, DELTA, GAMMAS, MARGIN,
    _accept_from_sm, _pool, _repartition_score_pool, _view,
)

ALPHA = 0.20
TEST_FRAC = 0.2
SEED = 0
BASE = "FedPD-PROSER"
GLOB = "runs/fedpd_cifar10_d5_seed[0-9].npz"   # J=5 (native n_clients), d=5
OUT = "runs/simplex_positive.csv"


def certify_simplex(npz_path):
    pool = _pool(npz_path, ("logits", "sm"))
    pool["_accept"] = _accept_from_sm(pool)
    parts = _repartition_score_pool(pool, ("logits", "sm", "_accept"),
                                    CERT_FRAC, TEST_FRAC, seed=SEED)
    J = int(np.asarray(parts["cert"]["client"]).max()) + 1
    views = {fn: _view(parts[fn], np.asarray(parts[fn]["_accept"], float), BASE)
             for fn in ("prop", "cert", "test")}
    res = certify_best_gamma(
        views["prop"], views["cert"], views["test"], score_name=BASE,
        gammas=GAMMAS, alpha=ALPHA, delta=DELTA, n_clients=J,
        dirichlet_alpha=float("nan"), Lambda="simplex", box=0.15,
        seed=SEED, margin=MARGIN)
    sel = choose_threshold(views["prop"]["score"], views["prop"]["pred"],
                           views["prop"]["y_open"], res["gamma_star"], ALPHA)
    A, _K, _n = counts_per_client(views["cert"]["score"], views["cert"]["pred"],
                                  views["cert"]["y_open"],
                                  np.asarray(views["cert"]["client"]), sel, J)
    return J, res, int(A.min())


def main():
    rows = []
    for f in sorted(glob.glob(GLOB)):
        s = int(re.search(r"seed(\d+)", os.path.basename(f)).group(1))
        J, res, nmin = certify_simplex(f)
        rows.append({"J": J, "seed": s, "alpha": ALPHA,
                     "cert_risk_ucb": round(float(res["cert_risk_ucb"]), 4),
                     "cert_coverage_lcb": round(float(res["cert_coverage_lcb"]), 4),
                     "cert_n_min_client": nmin,
                     "certified": int(bool(res["certified"]))})
        print(f"[ok] {os.path.basename(f)}  J={J} certified={rows[-1]['certified']} "
              f"cov_lcb={rows[-1]['cert_coverage_lcb']} risk_ucb={rows[-1]['cert_risk_ucb']} "
              f"n_min={nmin}")
    if not rows:
        print(f"[warn] no FedPD npz matched {GLOB}"); return
    npass = sum(r["certified"] for r in rows)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        fh.write("# R6 client-SIMPLEX (Thm 1, Lambda=simplex, per-client G=J) on FedPD-PROSER J=5 d=5; "
                 "native -sm, cert_frac=0.5, gammas{0.5,0.7,1.0}, delta=0.10, seed-0 repartition\n")
        w = csv.DictWriter(fh, fieldnames=["J", "seed", "alpha", "cert_risk_ucb",
                                           "cert_coverage_lcb", "cert_n_min_client", "certified"])
        w.writeheader(); w.writerows(rows)
    mean_cov = float(np.mean([r["cert_coverage_lcb"] if r["certified"] else 0.0 for r in rows]))
    print(f"\nwrote {OUT}  ({len(rows)} rows)  n_pass={npass}/{len(rows)}  "
          f"mean CertCov@{ALPHA}(cert else 0)={mean_cov:.4f}")


if __name__ == "__main__":
    main()
