"""PRIORITY 2: covtype multi-seed CertifiedCoverage at alpha in {0.20,0.25,0.30}.

Stabilizes (or fails to stabilize) the second (tabular) domain. Re-runs the covtype
federated logreg pipeline across 5 seeds (run_tabular.py --seed S) and certifies
post-hoc under THREE protocols, to separate an honest worst-group positive from the
selection-optimistic single-seed number that produced the old "covtype 0.43" headline:

  honest_msp_G2   -- fixed MSP, fixed worst-group G=2 (EXACTLY the CIFAR headline protocol);
  honest_best1_G2 -- fixed single best score (chosen once across the panel), worst-group G=2;
  selection_G123  -- best-of-4-scores x best-of-G in {1,2,3} incl. pooled G=1
                     (the OLD make_handoff frontier protocol; selection-optimistic).

All use cert_frac=0.5 on the pooled trusted points, box-Lambda best-gamma, margin=0.01.
Emits runs/agg_covtype.csv with per-seed coverage so nothing is hidden.

Run: python experiments/fedcore/aggregate_covtype.py
"""

from __future__ import annotations

import csv
import glob
import os

import numpy as np

from certify import certify_best_gamma_grouped
from make_handoff import _group_map, _repartition, _views_from_parts

SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.20, 0.25, 0.30)
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
DELTA, MARGIN, CERT_FRAC = 0.10, 0.01, 0.5
ALL_SCORES = ("msp", "energy", "neg_entropy", "margin")


def _cov_one(npz, score, G, alpha):
    d = np.load(npz)
    n_clients = int(d["cert_client"].max()) + 1
    pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}
    parts = _repartition(pool, CERT_FRAC, 0.2, seed=0)
    v = _views_from_parts(parts, score)
    gmap = _group_map(n_clients, G)
    r = certify_best_gamma_grouped(v["prop"], v["cert"], v["test"], score_name=score,
                                   group_map=gmap, G=G, gammas=GAMMAS, alpha=alpha,
                                   delta=DELTA, Lambda="box", box=0.15, seed=0, margin=MARGIN)
    return (r["cert_coverage_lcb"] if r["certified"] else 0.0)


def _cov_selection(npz, alpha):
    """best over scores x G in {1,2,3} (incl pooled) -- old optimistic protocol."""
    best = 0.0
    for s in ALL_SCORES:
        for G in (1, 2, 3):
            best = max(best, _cov_one(npz, s, G, alpha))
    return best


def main() -> None:
    base = "" if glob.glob("runs/*_logits.npz") else "../../"
    files = [base + f"runs/covtype_seed{s}_logits.npz" for s in SEEDS]
    files = [f for f in files if os.path.exists(f)]
    rows = []
    print(f"covtype multi-seed ({len(files)} seeds), cert_frac={CERT_FRAC}, box best-gamma\n")
    print(f"{'protocol':>16} {'alpha':>5} {'CertCov mean+/-std':>20} {'n_pass':>7}  per-seed")
    print("-" * 90)
    for alpha in ALPHAS:
        protocols = {
            "honest_msp_G2": [_cov_one(f, "msp", 2, alpha) for f in files],
            "honest_negent_G2": [_cov_one(f, "neg_entropy", 2, alpha) for f in files],
            "selection_G123": [_cov_selection(f, alpha) for f in files],
        }
        for name, covs in protocols.items():
            a = np.array(covs)
            rows.append({"protocol": name, "alpha": alpha, "n_seeds": len(a),
                         "CertCov_mean": round(float(a.mean()), 4),
                         "CertCov_std": round(float(a.std()), 4),
                         "n_pass": int((a > 0).sum()),
                         "per_seed": "|".join(f"{x:.3f}" for x in a)})
            print(f"{name:>16} {alpha:>5.2f} {a.mean():>9.3f}+/-{a.std():<9.3f} "
                  f"{int((a>0).sum()):>3}/{len(a)}  [{', '.join(f'{x:.2f}' for x in a)}]")
        print()

    out = base + "runs/agg_covtype.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
