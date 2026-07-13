"""M4: unknown-class split robustness (CPU aggregation of the manifest_M4 logits).

All primary CIFAR-10 results use ONE (seed-driven) known/unknown split. This certifies
TWO PRE-DECLARED alternative splits that hold a fixed unknown-class set across seeds:
  splitB: unknown={0,1,2,3}  known={4,5,6,7,8,9}
  splitC: unknown={6,7,8,9}  known={0,1,2,3,4,5}
GN (resnet18gn) backbone, d=0.5, clean, seeds 0..9, grouped G=2, alpha in {0.10,0.20},
via the IDENTICAL T9 grouped protocol (MSP head, seed-0 pooled repartition, box, delta=0.10) --
inlined here so the script imports only from the fedcore package.

Acceptance: CertCov@0.20 within seed-noise of the primary split, OR an honest report of
split sensitivity. Primary-split GN d=0.5 numbers live in runs/T9_diagnostics.csv
(backbone=resnet18gn, d=0.5); compare against them.

CSV: runs/unknown_split_robustness.csv  (T9 schema + backbone,d,split columns)

Run: python -m fedcore.experiments.exp_unknown_split_robustness   (CPU, no torch)
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

from fedcore.certify import certify_best_gamma
from fedcore.certificate import cp_upper
from fedcore.grouping import make_group_map, repartition_trusted_pool
from fedcore.io_utils import atomic_write_csv
from fedcore.scores import scored_views
from fedcore.selector import choose_threshold, counts_per_client

DELTA, G, SEED = 0.10, 2, 0
CERT_FRAC, TEST_FRAC = 0.5, 0.2
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
BOX, MARGIN = 0.15, 0.01
ALPHAS = (0.10, 0.20)
SPLITS = {"B": "0,1,2,3", "C": "6,7,8,9"}
PAT = "runs/cifar10_d0.5_resnet18gn_none0.0_split{sp}_seed{s}_logits.npz"
OUT = "runs/unknown_split_robustness.csv"
FIELDS = ["backbone", "d", "split", "unknown_classes", "alpha", "seed",
          "cert_risk_ucb_G2", "cert_n_min_group", "cert_k_worst_group",
          "cert_coverage_lcb", "test_risk", "test_coverage", "selected_gamma", "certified"]


def _diagnostics(npz_path, alpha):
    d = np.load(npz_path)
    n_clients = int(d["cert_client"].max()) + 1
    pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}
    parts = repartition_trusted_pool(pool, CERT_FRAC, TEST_FRAC, seed=SEED)
    views = {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                              parts[fn]["client"], ["msp"])["msp"]
             for fn in ("prop", "cert", "test")}
    gmap = make_group_map(n_clients, G)
    gviews = {}
    for fn in ("prop", "cert", "test"):
        v = dict(views[fn]); v["client"] = gmap[np.asarray(views[fn]["client"])]
        gviews[fn] = v
    res = certify_best_gamma(gviews["prop"], gviews["cert"], gviews["test"],
                             score_name="msp", gammas=GAMMAS, alpha=alpha, delta=DELTA,
                             n_clients=G, dirichlet_alpha=float("nan"), Lambda="box",
                             box=BOX, seed=SEED, margin=MARGIN)
    sel = choose_threshold(gviews["prop"]["score"], gviews["prop"]["pred"],
                           gviews["prop"]["y_open"], res["gamma_star"], alpha)
    A, K, _n = counts_per_client(gviews["cert"]["score"], gviews["cert"]["pred"],
                                 gviews["cert"]["y_open"], gviews["cert"]["client"], sel, G)
    eps = DELTA / (3.0 * G)
    rbar = np.array([cp_upper(int(K[g]), int(A[g]), eps) if A[g] > 0 else np.inf for g in range(G)])
    worst = int(np.argmax(rbar))
    return {"cert_risk_ucb_G2": round(float(res["cert_risk_ucb"]), 4),
            "cert_n_min_group": int(A.min()), "cert_k_worst_group": int(K[worst]),
            "cert_coverage_lcb": round(float(res["cert_coverage_lcb"]), 4),
            "test_risk": round(float(res["test_risk"]), 4),
            "test_coverage": round(float(res["test_coverage"]), 4),
            "selected_gamma": float(res["gamma_star"]), "certified": int(bool(res["certified"]))}


def main():
    rows = []
    for sp, unk in SPLITS.items():
        files = sorted(glob.glob(PAT.format(sp=sp, s="[0-9]")))
        if not files:
            print(f"[warn] no split{sp} logits yet ({PAT.format(sp=sp, s='*')})")
            continue
        for f in files:
            s = int(re.search(r"seed(\d+)", os.path.basename(f)).group(1))
            for alpha in ALPHAS:
                rows.append({"backbone": "resnet18gn", "d": "0.5", "split": sp,
                             "unknown_classes": unk, "alpha": alpha, "seed": s,
                             **_diagnostics(f, alpha)})
    if not rows:
        print("[warn] no split logits found; run manifest_M4 first."); return
    atomic_write_csv(OUT, FIELDS, rows)
    print(f"saved {OUT}  ({len(rows)} rows)\n")
    print(f"{'split':>5} {'alpha':>5} {'n':>3} {'CertCov mean+/-sd':>20} {'frac_cert':>9}")
    for sp in SPLITS:
        for alpha in ALPHAS:
            sub = [r for r in rows if r["split"] == sp and abs(r["alpha"] - alpha) < 1e-9]
            if not sub:
                continue
            cov = np.array([r["cert_coverage_lcb"] if r["certified"] else 0.0 for r in sub])
            fc = np.mean([r["certified"] for r in sub])
            print(f"{sp:>5} {alpha:>5.2f} {len(sub):>3} {cov.mean():>10.4f} +/-{cov.std():<7.4f} {fc:>9.3f}")


if __name__ == "__main__":
    main()
