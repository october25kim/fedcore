#!/usr/bin/env python3
"""R2 / R5 post-processing: grouped selective-risk certificates on the trained
CIFAR logits. CPU, no torch. Same headline protocol as certify_client_scaling.py
(fixed MSP, cert_frac=0.5, box-Lambda best-gamma over {0.2,0.3,0.5,0.7,1.0},
delta=0.10, margin=0.01, contiguous groups c*G//J, seed-0 fold repartition).

  --task r2 : CIFAR-100 multi-model. Glob r2_cifar100_<bb>_d<d>_seed<s>_logits.npz;
              G in {2, J=5}, alpha in {0.10, 0.20}. -> runs/cifar100_multimodel.csv
              schema: backbone,d,seed,alpha,G,cert_risk_ucb,cert_coverage_lcb,
                      cert_n_min_group,certified,test_risk,test_coverage
  --task r5 : corruption curve. Glob r5_cifar10_d<d>_sym<rate>_seed<s>_logits.npz;
              G=2, alpha in {0.10, 0.20}. -> runs/corruption_curve_seeded.csv
              schema: noise_type,rate,d,seed,CertCov@0.10,CertCov@0.20,test_risk
              (CertCov@a = G=2 cert_coverage_lcb if certified else 0.0)

Run: python scripts/ws4090/certify_grid.py --task r2
     python scripts/ws4090/certify_grid.py --task r5
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re

import numpy as np

from fedcore.certify import certify_best_gamma
from fedcore.grouping import make_group_map, repartition_trusted_pool
from fedcore.scores import scored_views
from fedcore.selector import choose_threshold, counts_per_client

GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
DELTA, MARGIN, BOX, CERT_FRAC, TEST_FRAC = 0.10, 0.01, 0.15, 0.5, 0.2
SCORE, SEED = "msp", 0
ALPHAS = (0.10, 0.20)


def certify_grouped(npz_path, G, alpha):
    d = np.load(npz_path)
    n_clients = int(d["cert_client"].max()) + 1
    pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}
    parts = repartition_trusted_pool(pool, CERT_FRAC, TEST_FRAC, seed=SEED)
    views = {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                              parts[fn]["client"], [SCORE])[SCORE]
             for fn in ("prop", "cert", "test")}
    gmap = make_group_map(n_clients, G)
    gviews = {}
    for fn in ("prop", "cert", "test"):
        v = dict(views[fn]); v["client"] = gmap[np.asarray(views[fn]["client"])]
        gviews[fn] = v
    res = certify_best_gamma(gviews["prop"], gviews["cert"], gviews["test"],
                             score_name=SCORE, gammas=GAMMAS, alpha=alpha, delta=DELTA,
                             n_clients=G, dirichlet_alpha=float("nan"), Lambda="box",
                             box=BOX, seed=SEED, margin=MARGIN)
    sel = choose_threshold(gviews["prop"]["score"], gviews["prop"]["pred"],
                           gviews["prop"]["y_open"], res["gamma_star"], alpha)
    A, _K, _n = counts_per_client(gviews["cert"]["score"], gviews["cert"]["pred"],
                                  gviews["cert"]["y_open"], gviews["cert"]["client"], sel, G)
    return res, int(A.min()), n_clients


def _write(out, header_comment, fields, rows):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        f.write(header_comment + "\n")
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")


def run_r2():
    rows = []
    for f in sorted(glob.glob("runs/r2_cifar100_*_seed*_logits.npz")):
        m = re.search(r"r2_cifar100_([a-z0-9]+)_d([0-9.]+)_seed(\d+)_logits\.npz$", os.path.basename(f))
        if not m:
            print(f"[skip] {f}"); continue
        bb, d, s = m.group(1), m.group(2), int(m.group(3))
        for alpha in ALPHAS:
            for G in (2, 5):    # {2, J=5}
                res, nmin, J = certify_grouped(f, G, alpha)
                rows.append({"backbone": bb, "d": d, "seed": s, "alpha": alpha, "G": G,
                             "cert_risk_ucb": round(float(res["cert_risk_ucb"]), 4),
                             "cert_coverage_lcb": round(float(res["cert_coverage_lcb"]), 4),
                             "cert_n_min_group": nmin,
                             "certified": int(bool(res["certified"])),
                             "test_risk": round(float(res["test_risk"]), 4),
                             "test_coverage": round(float(res["test_coverage"]), 4)})
        print(f"[ok] {bb} d={d} seed={s}")
    if not rows:
        print("[warn] no r2 npz matched -- run manifest_R2 first"); return
    _write("runs/cifar100_multimodel.csv",
           "# grouping=contiguous c*G//J; G in {2,J=5}; protocol=msp,cert_frac=0.5,box=0.15,delta=0.10",
           ["backbone", "d", "seed", "alpha", "G", "cert_risk_ucb", "cert_coverage_lcb",
            "cert_n_min_group", "certified", "test_risk", "test_coverage"], rows)


def run_r5():
    rows = []
    for f in sorted(glob.glob("runs/r5_cifar10_d*_sym*_seed*_logits.npz")):
        m = re.search(r"r5_cifar10_d([0-9.]+)_sym([0-9.]+)_seed(\d+)_logits\.npz$", os.path.basename(f))
        if not m:
            print(f"[skip] {f}"); continue
        d, rate, s = m.group(1), m.group(2), int(m.group(3))
        cov = {}
        test_risk = None
        for alpha in ALPHAS:
            res, _nmin, _J = certify_grouped(f, 2, alpha)   # G=2 headline
            cov[alpha] = round(float(res["cert_coverage_lcb"]) if res["certified"] else 0.0, 4)
            test_risk = round(float(res["test_risk"]), 4)
        rows.append({"noise_type": "symmetric", "rate": rate, "d": d, "seed": s,
                     "CertCov@0.10": cov[0.10], "CertCov@0.20": cov[0.20], "test_risk": test_risk})
        print(f"[ok] d={d} rate={rate} seed={s}")
    if not rows:
        print("[warn] no r5 npz matched -- run manifest_R5 first"); return
    _write("runs/corruption_curve_seeded.csv",
           "# G=2 grouped headline (contiguous c*G//J); CertCov@a = cov_lcb if certified else 0; delta=0.10",
           ["noise_type", "rate", "d", "seed", "CertCov@0.10", "CertCov@0.20", "test_risk"], rows)


def run_r5asym():
    """Asymmetric arm of the corruption curve. Same G=2 headline protocol as
    run_r5, over r5_cifar10_d*_asym*_seed* npz, but APPENDS noise_type=asymmetric
    rows to the existing runs/corruption_curve_seeded.csv WITHOUT touching the
    symmetric rows (idempotent: skips (rate,d,seed) triples already present)."""
    out = "runs/corruption_curve_seeded.csv"
    if not os.path.exists(out):
        print(f"[warn] {out} missing -- run certify_grid.py --task r5 (symmetric) first")
        return
    fields = ["noise_type", "rate", "d", "seed", "CertCov@0.10", "CertCov@0.20", "test_risk"]
    with open(out) as f:
        comment = f.readline().rstrip("\n")
        existing = list(csv.DictReader(f))
    have = {(r["noise_type"], r["rate"], r["d"], r["seed"]) for r in existing}
    new_rows = []
    for f in sorted(glob.glob("runs/r5_cifar10_d*_asym*_seed*_logits.npz")):
        m = re.search(r"r5_cifar10_d([0-9.]+)_asym([0-9.]+)_seed(\d+)_logits\.npz$", os.path.basename(f))
        if not m:
            print(f"[skip] {f}"); continue
        d, rate, s = m.group(1), m.group(2), int(m.group(3))
        if ("asymmetric", rate, d, str(s)) in have:
            print(f"[have] asym d={d} rate={rate} seed={s} (already in csv)"); continue
        cov = {}
        test_risk = None
        for alpha in ALPHAS:
            res, _nmin, _J = certify_grouped(f, 2, alpha)   # G=2 headline
            cov[alpha] = round(float(res["cert_coverage_lcb"]) if res["certified"] else 0.0, 4)
            test_risk = round(float(res["test_risk"]), 4)
        new_rows.append({"noise_type": "asymmetric", "rate": rate, "d": d, "seed": s,
                         "CertCov@0.10": cov[0.10], "CertCov@0.20": cov[0.20], "test_risk": test_risk})
        print(f"[ok] asym d={d} rate={rate} seed={s}")
    if not new_rows:
        print("[warn] no new asymmetric rows to append (no r5 asym npz, or all present)"); return
    with open(out, "a", newline="") as f:      # APPEND, preserve symmetric rows + header
        w = csv.DictWriter(f, fieldnames=fields)
        w.writerows(new_rows)
    print(f"appended {len(new_rows)} asymmetric rows to {out} "
          f"({len(existing)} pre-existing rows preserved)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["r2", "r5", "r5asym"], required=True)
    args = ap.parse_args()
    {"r2": run_r2, "r5": run_r5, "r5asym": run_r5asym}[args.task]()


if __name__ == "__main__":
    main()
