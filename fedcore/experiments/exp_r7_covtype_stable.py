#!/usr/bin/env python3
"""R7: covtype STABLE second-domain positive (A2-compliant). CPU, no torch.

Extends the covtype cell to 10 seeds with the PROCEDURALLY-VALID protocol:
  * score chosen on the PROPOSAL fold only (max proposal coverage among
    proposal-certified scores; else least-infeasible proxy) -> A2 compliant,
    independent of the certification labels;
  * enlarged audit budget cert_frac=0.5;
  * worst-group grouped certificate G=2 (contiguous pre-declared c*G//J),
    box-Lambda best-gamma, delta=0.10;
  * alpha in {0.20, 0.25, 0.30}, seeds {0..9}.

Emits per-seed rows in the T9 schema (+ selected_score for provenance) ->
runs/covtype_stable.csv. Goal: >= 8/10 non-vacuous at some alpha.

Run: python -m fedcore.experiments.exp_r7_covtype_stable   (CPU, no torch)
"""
from __future__ import annotations

import csv
import os

import numpy as np

from fedcore.certificate import cp_upper
from fedcore.certify import certify_best_gamma
from fedcore.grouping import make_group_map, repartition_trusted_pool
from fedcore.scores import scored_views
from fedcore.selector import choose_threshold, counts_per_client

SEEDS = tuple(range(10))
ALPHAS = (0.20, 0.25, 0.30)
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
DELTA, MARGIN, BOX, CERT_FRAC, TEST_FRAC = 0.10, 0.01, 0.15, 0.5, 0.2
SCORES = ("msp", "energy", "neg_entropy", "margin")
G = 2
D = "0.5"          # covtype training dirichlet_alpha (matches seeds 0..9)
SEED = 0           # fold-repartition seed (fixed, as in the headline protocol)
OUT = "runs/covtype_stable.csv"


def _gviews(parts, score):
    views = {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                              parts[fn]["client"], [score])[score]
             for fn in ("prop", "cert", "test")}
    gmap_n = int(np.asarray(parts["cert"]["client"]).max()) + 1
    gmap = make_group_map(gmap_n, G)
    gv = {}
    for fn in ("prop", "cert", "test"):
        v = dict(views[fn]); v["client"] = gmap[np.asarray(views[fn]["client"])]
        gv[fn] = v
    return gv


def _certify(gv, alpha, delta):
    return certify_best_gamma(
        gv["prop"], gv["cert"], gv["test"], score_name="s",
        gammas=GAMMAS, alpha=alpha, delta=delta, n_clients=G,
        dirichlet_alpha=float("nan"), Lambda="box", box=BOX, seed=SEED, margin=MARGIN)


def t9_row(gv, res, alpha, delta):
    """Recompute the worst-/min-group diagnostics for the chosen score (== T9 gen)."""
    gamma_star = res["gamma_star"]
    sel = choose_threshold(gv["prop"]["score"], gv["prop"]["pred"],
                           gv["prop"]["y_open"], gamma_star, alpha)
    A, K, _n = counts_per_client(gv["cert"]["score"], gv["cert"]["pred"],
                                 gv["cert"]["y_open"], gv["cert"]["client"], sel, G)
    eps = delta / (3.0 * G)
    rbar = np.array([cp_upper(int(K[g]), int(A[g]), eps) if A[g] > 0 else np.inf
                     for g in range(G)])
    worst = int(np.argmax(rbar))
    return {
        "cert_risk_ucb_G2": float(res["cert_risk_ucb"]),
        "cert_n_min_group": int(A.min()),
        "cert_k_worst_group": int(K[worst]),
        "cert_coverage_lcb": float(res["cert_coverage_lcb"]),
        "test_risk": float(res["test_risk"]),
        "test_coverage": float(res["test_coverage"]),
        "selected_gamma": float(gamma_star),
        "certified": int(bool(res["certified"])),
    }


def a2_select_and_certify(npz, alpha):
    d = np.load(npz)
    pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}
    parts = repartition_trusted_pool(pool, CERT_FRAC, TEST_FRAC, seed=SEED)
    per = {s: (lambda gv: (gv, _certify(gv, alpha, DELTA)))(_gviews(parts, s)) for s in SCORES}
    # A2 selection on the PROPOSAL fold only
    feas = [s for s in SCORES if per[s][1]["u_proxy"] <= alpha]
    chosen = (max(feas, key=lambda s: per[s][1]["prop_coverage"]) if feas
              else min(SCORES, key=lambda s: per[s][1]["u_proxy"]))
    gv, res = per[chosen]
    row = t9_row(gv, res, alpha, DELTA)
    row["selected_score"] = chosen
    return row


def main():
    rows = []
    for s in SEEDS:
        npz = f"runs/covtype_seed{s}_logits.npz"
        if not os.path.exists(npz):
            print(f"[missing] {npz}"); continue
        for alpha in ALPHAS:
            r = a2_select_and_certify(npz, alpha)
            rows.append({"backbone": "covtype", "d": D, "alpha": alpha, "seed": s, **r})
            print(f"covtype d{D} a{alpha:.2f} s{s} score={r['selected_score']:>11} "
                  f"cov_lcb={r['cert_coverage_lcb']:.4f} ucb={r['cert_risk_ucb_G2']:.3f} "
                  f"n_min={r['cert_n_min_group']} certified={r['certified']}")
    if not rows:
        print("[warn] no covtype npz found"); return
    fields = ["backbone", "d", "alpha", "seed", "cert_risk_ucb_G2", "cert_n_min_group",
              "cert_k_worst_group", "cert_coverage_lcb", "test_risk", "test_coverage",
              "selected_gamma", "certified", "selected_score"]
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w", newline="") as f:
        f.write("# R7 covtype STABLE (A2 proposal-fold score select, enlarged audit cert_frac=0.5, "
                "worst-group G=2 contiguous, box best-gamma, delta=0.10); T9 schema + selected_score\n")
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    # per-alpha n_pass summary
    print(f"\nwrote {OUT}  ({len(rows)} rows)")
    for alpha in ALPHAS:
        cell = [r for r in rows if r["alpha"] == alpha]
        npass = sum(r["certified"] for r in cell)
        covs = np.array([r["cert_coverage_lcb"] if r["certified"] else 0.0 for r in cell])
        print(f"  alpha={alpha:.2f}: n_pass={npass}/{len(cell)}  "
              f"CertCov mean±std={covs.mean():.4f}±{covs.std():.4f}")


if __name__ == "__main__":
    main()
