#!/usr/bin/env python3
"""Task 4a: CIFAR-100 ResNet-GN certified-coverage frontier raw material.

Reuses the EXACT Table-6 / CIFAR-10-frontier protocol (certify_grid.certify_grouped:
grouped G=2, best-gamma over {0.2,0.3,0.5,0.7,1.0}, box-Lambda 0.15, msp, cert_frac=0.5,
delta=0.10 -> simultaneous delta/J=delta/2 across the 2 groups, seed-0 fold repartition)
over the frozen r2 GN logit exports -- i.e. the very models behind Table 6. No retraining.

Inputs : runs/r2_cifar100_resnet18gn_d{0.5,5}_seed{0..9}_logits.npz  (20 npz)
Outputs:
  runs/cifar100_frontier_gn_perseed.csv     per-seed CertCov@alpha over
      alpha in {0.10,0.15,0.20,0.25,0.30}   (raw material for Fig 6(a); laptop draws the figure)
  runs/cifar100_frontier_gn_consistency.csv Step-2 check: OLD (runs/cifar100_multimodel.csv,
      G=2) vs NEW (recompute from npz) cell aggregates at alpha in {0.10,0.20}:
      certified fraction + CertCov mean/sd (ddof=1, paper convention).

Judged by cert_* only. CertCov@a := cert_coverage_lcb if certified else 0.0 (same as run_r5).
If NEW certified fraction differs from OLD by >2 seeds in any cell -> prints a STOP banner.

Run inside the golden docker (scipy==1.17.1): python scripts/ws4090/frontier_cifar100_gn.py
"""
from __future__ import annotations

import csv
import glob
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certify_grid import certify_grouped  # noqa: E402  (reuse the accepted protocol verbatim)

FRONTIER_ALPHAS = (0.10, 0.15, 0.20, 0.25, 0.30)
CHECK_ALPHAS = (0.10, 0.20)
G = 2
PERSEED = "runs/cifar100_frontier_gn_perseed.csv"
CONSIST = "runs/cifar100_frontier_gn_consistency.csv"
OLD_CSV = "runs/cifar100_multimodel.csv"
NPZ_GLOB = "runs/r2_cifar100_resnet18gn_d*_seed*_logits.npz"


def _certcov(res):
    return round(float(res["cert_coverage_lcb"]), 4) if res["certified"] else 0.0


def compute_perseed():
    rows = []
    files = sorted(glob.glob(NPZ_GLOB))
    for f in files:
        m = re.search(r"r2_cifar100_resnet18gn_d([0-9.]+)_seed(\d+)_logits\.npz$", os.path.basename(f))
        d, s = m.group(1), int(m.group(2))
        for a in FRONTIER_ALPHAS:
            res, nmin, J = certify_grouped(f, G, a)
            rows.append({"backbone": "resnet18gn", "d": d, "seed": s, "alpha": a, "G": G,
                         "certified": int(bool(res["certified"])),
                         "CertCov": _certcov(res),
                         "cert_coverage_lcb": round(float(res["cert_coverage_lcb"]), 4),
                         "cert_risk_ucb": round(float(res["cert_risk_ucb"]), 4),
                         "cert_n_min_group": nmin,
                         "test_risk": round(float(res["test_risk"]), 4),
                         "test_coverage": round(float(res["test_coverage"]), 4)})
        print(f"[ok] d={d} seed={s}")
    return rows


def _agg(vals):
    """(certified_count, n, mean, sd) with sample sd (ddof=1) to match the paper."""
    covs = [v[0] for v in vals]
    certs = sum(v[1] for v in vals)
    n = len(vals)
    mean = sum(covs) / n if n else 0.0
    sd = statistics.stdev(covs) if n > 1 else 0.0
    return certs, n, mean, sd


def load_old():
    """OLD side: G=2 GN rows from cifar100_multimodel.csv -> {(d,alpha): [(certcov,cert)]}"""
    out = {}
    with open(OLD_CSV) as f:
        f.readline()
        for r in csv.DictReader(f):
            if r["backbone"] != "resnet18gn" or int(r["G"]) != 2:
                continue
            a = round(float(r["alpha"]), 2)
            if a not in CHECK_ALPHAS:
                continue
            cert = int(r["certified"])
            cov = float(r["cert_coverage_lcb"]) if cert else 0.0
            out.setdefault((r["d"], a), []).append((cov, cert))
    return out


def main():
    perseed = compute_perseed()
    os.makedirs("runs", exist_ok=True)
    with open(PERSEED, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "d", "seed", "alpha", "G", "certified",
                                          "CertCov", "cert_coverage_lcb", "cert_risk_ucb",
                                          "cert_n_min_group", "test_risk", "test_coverage"])
        w.writeheader(); w.writerows(perseed)
    print(f"wrote {PERSEED} ({len(perseed)} rows)")

    # NEW aggregates at the check alphas
    new = {}
    for r in perseed:
        a = round(float(r["alpha"]), 2)
        if a in CHECK_ALPHAS:
            new.setdefault((r["d"], a), []).append((r["CertCov"], r["certified"]))
    old = load_old()

    crows, stop = [], []
    for d in ("0.5", "5"):
        for a in CHECK_ALPHAS:
            oc, on, om, osd = _agg(old.get((d, a), []))
            nc, nn, nm, nsd = _agg(new.get((d, a), []))
            crows.append({"d": d, "alpha": a,
                          "old_certfrac": f"{oc}/{on}", "new_certfrac": f"{nc}/{nn}",
                          "old_CertCov_mean": round(om, 4), "old_CertCov_sd": round(osd, 4),
                          "new_CertCov_mean": round(nm, 4), "new_CertCov_sd": round(nsd, 4),
                          "certfrac_delta_seeds": abs(oc - nc)})
            if abs(oc - nc) > 2:
                stop.append((d, a, oc, nc))
    with open(CONSIST, "w", newline="") as f:
        f.write("# Task-4a consistency: OLD=cifar100_multimodel.csv(G=2) vs NEW=recompute from r2 GN npz; "
                "sd is sample sd (ddof=1). SAME models (no retraining) -> expect exact match.\n")
        w = csv.DictWriter(f, fieldnames=["d", "alpha", "old_certfrac", "new_certfrac",
                                          "old_CertCov_mean", "old_CertCov_sd",
                                          "new_CertCov_mean", "new_CertCov_sd", "certfrac_delta_seeds"])
        w.writeheader(); w.writerows(crows)
    print(f"wrote {CONSIST} ({len(crows)} rows)")
    print("\n=== consistency (old vs new) ===")
    for r in crows:
        print(f"  d={r['d']:>3} a={r['alpha']:.2f}  certfrac {r['old_certfrac']}->{r['new_certfrac']}  "
              f"CertCov {r['old_CertCov_mean']:.4f}+-{r['old_CertCov_sd']:.4f} -> "
              f"{r['new_CertCov_mean']:.4f}+-{r['new_CertCov_sd']:.4f}")
    if stop:
        print("\n*** STOP: certified fraction drifted >2 seeds in:")
        for d, a, oc, nc in stop:
            print(f"    d={d} alpha={a}: old {oc} -> new {nc}")
        sys.exit(3)
    print("\n[consistency OK] no cell drifted >2 seeds.")


if __name__ == "__main__":
    main()
