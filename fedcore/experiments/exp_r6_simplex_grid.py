#!/usr/bin/env python3
"""M3: full-simplex (Theorem 1, Lambda=simplex, per-client G=J) expansion on the
stored FedPD-PROSER J=5 detector logits -- CPU, NO retraining. Extends the single
d=5/alpha=0.20 simplex-positive cell to the (d, alpha) grid the review asks for:

    d in {5, 0.5}  x  alpha in {0.20, 0.10}   x  seeds {0..4}

Reuses the EXACT T8 detector protocol verbatim (native -sm accept score,
cert_frac=0.5 enlarged audit budget, gammas {0.5,0.7,1.0}, delta=0.10, seed-0 pooled
repartition), certifying under the FULL client simplex (the hardest, worst-client
deployment set). A non-vacuous cell is a strong client-simplex positive; a clean
alpha=0.10 failure is the measured PRICE of full client-simplex robustness at the
hard target (reportable per the R6/M3 brief).

Overwrites runs/simplex_positive.csv, ADDING a `d` column so d=5 and d=0.5 rows are
distinguishable. The d=5/alpha=0.20 rows reproduce the prior file exactly (same npz,
same code path).

  schema: J,d,seed,alpha,cert_risk_ucb,cert_coverage_lcb,cert_n_min_client,certified

Run: python -m fedcore.experiments.exp_r6_simplex_grid   (CPU, no torch)
"""
from __future__ import annotations

import csv
import glob
import os
import re

import numpy as np

from fedcore.certify import certify_best_gamma
from fedcore.selector import choose_threshold, counts_per_client
from fedcore.aggregate.t8 import (
    CERT_FRAC, DELTA, GAMMAS, MARGIN,
    _accept_from_sm, _pool, _repartition_score_pool, _view,
)

TEST_FRAC = 0.2
SEED = 0
BASE = "FedPD-PROSER"
OUT = "runs/simplex_positive.csv"
DS = ("5", "0.5")
ALPHAS = (0.20, 0.10)
GLOB = "runs/fedpd_cifar10_d{d}_seed[0-9].npz"


def certify_simplex(npz_path, alpha):
    pool = _pool(npz_path, ("logits", "sm"))
    pool["_accept"] = _accept_from_sm(pool)
    parts = _repartition_score_pool(pool, ("logits", "sm", "_accept"),
                                    CERT_FRAC, TEST_FRAC, seed=SEED)
    J = int(np.asarray(parts["cert"]["client"]).max()) + 1
    views = {fn: _view(parts[fn], np.asarray(parts[fn]["_accept"], float), BASE)
             for fn in ("prop", "cert", "test")}
    res = certify_best_gamma(
        views["prop"], views["cert"], views["test"], score_name=BASE,
        gammas=GAMMAS, alpha=alpha, delta=DELTA, n_clients=J,
        dirichlet_alpha=float("nan"), Lambda="simplex", box=0.15,
        seed=SEED, margin=MARGIN)
    sel = choose_threshold(views["prop"]["score"], views["prop"]["pred"],
                           views["prop"]["y_open"], res["gamma_star"], alpha)
    A, _K, _n = counts_per_client(views["cert"]["score"], views["cert"]["pred"],
                                  views["cert"]["y_open"],
                                  np.asarray(views["cert"]["client"]), sel, J)
    return J, res, int(A.min())


def main():
    rows = []
    for d in DS:
        files = sorted(glob.glob(GLOB.format(d=d)))
        if not files:
            print(f"[warn] no FedPD npz for d={d} ({GLOB.format(d=d)})")
            continue
        for alpha in ALPHAS:
            npass = 0
            for f in files:
                s = int(re.search(r"seed(\d+)", os.path.basename(f)).group(1))
                J, res, nmin = certify_simplex(f, alpha)
                cert = int(bool(res["certified"]))
                npass += cert
                rows.append({"J": J, "d": d, "seed": s, "alpha": alpha,
                             "cert_risk_ucb": round(float(res["cert_risk_ucb"]), 4),
                             "cert_coverage_lcb": round(float(res["cert_coverage_lcb"]), 4),
                             "cert_n_min_client": nmin, "certified": cert})
            mean_cov = float(np.mean([r["cert_coverage_lcb"] if r["certified"] else 0.0
                                      for r in rows if r["d"] == d and abs(r["alpha"] - alpha) < 1e-9]))
            print(f"[d={d:>3} alpha={alpha:.2f}] n_pass={npass}/{len(files)}  "
                  f"mean CertCov(cert else 0)={mean_cov:.4f}")
    if not rows:
        print("[warn] no rows produced"); return
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        fh.write("# M3 client-SIMPLEX (Thm 1, Lambda=simplex, per-client G=J) on FedPD-PROSER J=5; "
                 "native -sm, cert_frac=0.5, gammas{0.5,0.7,1.0}, delta=0.10, seed-0 repartition; "
                 "grid d in {5,0.5} x alpha in {0.20,0.10}\n")
        w = csv.DictWriter(fh, fieldnames=["J", "d", "seed", "alpha", "cert_risk_ucb",
                                           "cert_coverage_lcb", "cert_n_min_client", "certified"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
