#!/usr/bin/env python3
"""R1 post-processing: grouped selective-risk certificates on the real CIFAR-10
client-scaling logits (J in {10,20}). CPU, no torch.

For each trained npz (J, seed) it certifies at grouping G in {J, J/2, 5, 2}
(contiguous PRE-DECLARED groups: client c -> group c*G//J) for alpha in
{0.10, 0.20}, using the EXACT CIFAR headline protocol (fixed MSP, cert_frac=0.5,
test_frac=0.2, box-Lambda best-gamma over {0.2,0.3,0.5,0.7,1.0}, delta=0.10,
margin=0.01, seed-0 fold repartition). This is the Theorem-3 pattern on real data:
per-client (G=J) bounds degrade with J at fixed total data; coarser grouping
raises per-group counts and restores certification.

Output: runs/client_scaling.csv  (a leading '# grouping=...' comment line states
the rule; read with comment='#'). Schema:
  J,seed,alpha,G,cert_risk_ucb,cert_coverage_lcb,cert_n_min_group,certified,test_risk,test_coverage

Run:
  python scripts/ws4090/certify_client_scaling.py
  python scripts/ws4090/certify_client_scaling.py --smoke <one_logits.npz>   # code check
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


def _groupings(J):
    """G in {J, J/2, 5, 2}, deduped, contiguous; drop G>J or G<1."""
    cand = {J, max(1, J // 2), 5, 2}
    return sorted({g for g in cand if 1 <= g <= J}, reverse=True)


def certify_grouped(npz_path, J, G, alpha):
    d = np.load(npz_path)
    n_clients = int(d["cert_client"].max()) + 1
    pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}
    parts = repartition_trusted_pool(pool, CERT_FRAC, TEST_FRAC, seed=SEED)
    views = {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                              parts[fn]["client"], [SCORE])[SCORE]
             for fn in ("prop", "cert", "test")}
    gmap = make_group_map(n_clients, G)              # contiguous blocks c*G//n
    gviews = {}
    for fn in ("prop", "cert", "test"):
        v = dict(views[fn]); v["client"] = gmap[np.asarray(views[fn]["client"])]
        gviews[fn] = v
    res = certify_best_gamma(gviews["prop"], gviews["cert"], gviews["test"],
                             score_name=SCORE, gammas=GAMMAS, alpha=alpha, delta=DELTA,
                             n_clients=G, dirichlet_alpha=float("nan"), Lambda="box",
                             box=BOX, seed=SEED, margin=MARGIN)
    # per-group accepted counts under the chosen selector -> min group count
    sel = choose_threshold(gviews["prop"]["score"], gviews["prop"]["pred"],
                           gviews["prop"]["y_open"], res["gamma_star"], alpha)
    A, _K, _n = counts_per_client(gviews["cert"]["score"], gviews["cert"]["pred"],
                                  gviews["cert"]["y_open"], gviews["cert"]["client"], sel, G)
    return {"J": J, "seed": None, "alpha": alpha, "G": G,
            "cert_risk_ucb": round(float(res["cert_risk_ucb"]), 4),
            "cert_coverage_lcb": round(float(res["cert_coverage_lcb"]), 4),
            "cert_n_min_group": int(A.min()),
            "certified": int(bool(res["certified"])),
            "test_risk": round(float(res["test_risk"]), 4),
            "test_coverage": round(float(res["test_coverage"]), 4)}


def _write(rows, out):
    fields = ["J", "seed", "alpha", "G", "cert_risk_ucb", "cert_coverage_lcb",
              "cert_n_min_group", "certified", "test_risk", "test_coverage"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        f.write("# grouping=contiguous blocks (client c -> group c*G//J); "
                "protocol=msp,cert_frac=0.5,box=0.15,delta=0.10,gamma-grid-selected\n")
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/r1_cifar10_J*_d0.5_resnet18gn_seed*_logits.npz")
    ap.add_argument("--out", default="runs/client_scaling.csv")
    ap.add_argument("--smoke", default=None, help="certify one npz as J=5 (code check) and exit")
    args = ap.parse_args()

    if args.smoke:
        print(f"[smoke] {args.smoke} as J=5, G in {_groupings(5)}")
        for alpha in ALPHAS:
            for G in _groupings(5):
                r = certify_grouped(args.smoke, 5, G, alpha)
                print(f"  a={alpha} G={G}: cov_lcb={r['cert_coverage_lcb']} "
                      f"ucb={r['cert_risk_ucb']} n_min={r['cert_n_min_group']} cert={r['certified']}")
        return

    rows = []
    for f in sorted(glob.glob(args.glob)):
        m = re.search(r"_J(\d+)_.*seed(\d+)_logits\.npz$", os.path.basename(f))
        if not m:
            print(f"[skip] cannot parse J/seed: {f}"); continue
        J, s = int(m.group(1)), int(m.group(2))
        for alpha in ALPHAS:
            for G in _groupings(J):
                r = certify_grouped(f, J, G, alpha); r["seed"] = s
                rows.append(r)
        print(f"[ok] J={J} seed={s}")
    if not rows:
        print(f"[warn] no npz matched {args.glob} -- run manifest_R1 first"); return
    _write(rows, args.out)
    # quick Theorem-3 readout: mean CertCov by (J,G) at alpha=0.20
    print("\n=== CertCov@0.20 mean by (J,G) [Theorem-3 pattern] ===")
    for J in sorted({r["J"] for r in rows}):
        for G in sorted({r["G"] for r in rows if r["J"] == J}, reverse=True):
            cell = [r for r in rows if r["J"] == J and r["G"] == G and r["alpha"] == 0.20]
            cov = np.mean([r["cert_coverage_lcb"] if r["certified"] else 0.0 for r in cell])
            npass = sum(r["certified"] for r in cell)
            print(f"  J={J:>2} G={G:>2}: CertCov@0.20={cov:.4f}  pass={npass}/{len(cell)}")


if __name__ == "__main__":
    main()
