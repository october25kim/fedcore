"""R4: deployment-knob sensitivity on stored resnet18gn (GN) logits (CPU, no torch).

Two ablations over the certification knobs, on the SAME headline pipeline used by
fedcore/aggregate/main.py (MSP score, G=2 grouped, cert_frac=0.5, test_frac=0.2,
delta=0.10, seed-0 fold repartition). GN cells d in {5, 0.5}, all stored seeds,
alpha in {0.10, 0.20}.

(1) rho sweep -- Lambda deployment-mixture-set width.
    rho in {0.05, 0.10, 0.15, 0.25} is the box half-width (0.15 = headline);
    rho="simplex" is the full-simplex worst case. Best-gamma selection kept (the
    deployment protocol). -> runs/rho_sensitivity.csv
    (rho,seed,d,alpha,cert_risk_ucb,cert_coverage_lcb,certified)

(2) gamma ablation -- FIXED gamma, NO best-gamma selection, at the headline
    Lambda (box=0.15). Shows why the proposal-side risk buffer matters: do not
    conclude gamma=1.0 suffices without checking). -> runs/gamma_ablation.csv
    (gamma,seed,d,alpha,cert_n,cert_risk_ucb,cert_coverage_lcb,certified,test_risk)

NOTE: GN has 5 stored seeds (0..4); this ablation reuses stored logits (no new
training), so the 10-seed floor for NEW cells does not apply -- flagged in the
report.

Run: python -m fedcore.experiments.exp_r4_knob_sensitivity   (CPU, no torch)
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

from fedcore.certify import certify_best_gamma_grouped, certify_for_score
from fedcore.grouping import make_group_map, repartition_trusted_pool
from fedcore.scores import scored_views
from fedcore.io_utils import atomic_write_csv

DELTA, MARGIN, CERT_FRAC, TEST_FRAC = 0.10, 0.01, 0.5, 0.2
SCORE, G, SEED = "msp", 2, 0
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)   # main.py headline grid (selection)
ALPHAS = (0.10, 0.20)
RHOS = (0.05, 0.10, 0.15, 0.25, "simplex")
FIXED_GAMMAS = (0.5, 0.7, 1.0)
DS = ("5", "0.5")
PAT = "runs/cifar10_d{d}_resnet18gn_none0.0_seed{s}_logits.npz"


def _grouped_views(npz):
    """MSP scored views with client ids remapped to G=2 groups (headline fold split)."""
    d = np.load(npz)
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
    return gviews


def _seeds_for(d):
    out = []
    for f in sorted(glob.glob(PAT.format(d=d, s="*"))):
        m = re.search(r"seed(\d+)", f)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def rho_sweep():
    rows = []
    for d in DS:
        for s in _seeds_for(d):
            gviews = _grouped_views(PAT.format(d=d, s=s))
            for alpha in ALPHAS:
                for rho in RHOS:
                    Lambda = "simplex" if rho == "simplex" else "box"
                    box = 0.15 if rho == "simplex" else float(rho)
                    r = certify_best_gamma_grouped(
                        gviews["prop"], gviews["cert"], gviews["test"], score_name=SCORE,
                        group_map=np.arange(G), G=G, gammas=GAMMAS, alpha=alpha, delta=DELTA,
                        Lambda=Lambda, box=box, seed=SEED, margin=MARGIN)
                    rows.append({"rho": rho, "seed": s, "d": d, "alpha": alpha,
                                 "cert_risk_ucb": round(float(r["cert_risk_ucb"]), 4),
                                 "cert_coverage_lcb": round(float(r["cert_coverage_lcb"]), 4),
                                 "certified": int(bool(r["certified"]))})
    fields = ["rho", "seed", "d", "alpha", "cert_risk_ucb", "cert_coverage_lcb", "certified"]
    atomic_write_csv("runs/rho_sensitivity.csv", fields, rows)
    print(f"saved runs/rho_sensitivity.csv ({len(rows)} rows)")
    return rows


def gamma_ablation():
    rows = []
    for d in DS:
        for s in _seeds_for(d):
            gviews = _grouped_views(PAT.format(d=d, s=s))
            for alpha in ALPHAS:
                for gamma in FIXED_GAMMAS:
                    r = certify_for_score(
                        SCORE, gviews["prop"], gviews["cert"], gviews["test"],
                        gamma=gamma, alpha=alpha, delta=DELTA, Lambda="box",
                        n_clients=G, dirichlet_alpha=float("nan"), box=0.15, seed=SEED)
                    rows.append({"gamma": gamma, "seed": s, "d": d, "alpha": alpha,
                                 "cert_n": int(r["cert_n"]),
                                 "cert_risk_ucb": round(float(r["cert_risk_ucb"]), 4),
                                 "cert_coverage_lcb": round(float(r["cert_coverage_lcb"]), 4),
                                 "certified": int(bool(r["certified"])),
                                 "test_risk": round(float(r["test_risk"]), 4)})
    fields = ["gamma", "seed", "d", "alpha", "cert_n", "cert_risk_ucb",
              "cert_coverage_lcb", "certified", "test_risk"]
    atomic_write_csv("runs/gamma_ablation.csv", fields, rows)
    print(f"saved runs/gamma_ablation.csv ({len(rows)} rows)")
    return rows


def _summary(rho_rows, gam_rows):
    print("\n=== rho sweep: mean CertCov_lcb (certified->cov, else 0), d=5 ===")
    print(f"{'alpha':>5} " + " ".join(f"{str(r):>8}" for r in RHOS))
    for alpha in ALPHAS:
        line = []
        for rho in RHOS:
            vals = [r["cert_coverage_lcb"] if r["certified"] else 0.0
                    for r in rho_rows if r["d"] == "5" and r["alpha"] == alpha and r["rho"] == rho]
            line.append(f"{np.mean(vals):>8.4f}" if vals else f"{'--':>8}")
        print(f"{alpha:>5.2f} " + " ".join(line))
    print("\n=== gamma ablation: mean CertCov_lcb / n_pass, d=5 ===")
    print(f"{'alpha':>5} " + " ".join(f"{'g='+str(g):>14}" for g in FIXED_GAMMAS))
    for alpha in ALPHAS:
        line = []
        for g in FIXED_GAMMAS:
            cell = [r for r in gam_rows if r["d"] == "5" and r["alpha"] == alpha and r["gamma"] == g]
            cov = np.mean([r["cert_coverage_lcb"] if r["certified"] else 0.0 for r in cell]) if cell else 0.0
            npass = sum(r["certified"] for r in cell)
            line.append(f"{cov:>7.4f}/{npass}/{len(cell):<3}")
        print(f"{alpha:>5.2f} " + " ".join(f"{x:>14}" for x in line))


def main():
    print(f"R4 knob sensitivity: GN d{DS} seeds{{{_seeds_for('5')}}} score={SCORE} G={G} delta={DELTA}")
    rho_rows = rho_sweep()
    gam_rows = gamma_ablation()
    _summary(rho_rows, gam_rows)


if __name__ == "__main__":
    main()
