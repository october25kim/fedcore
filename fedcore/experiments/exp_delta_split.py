"""P0: delta-split recomputation of the Table 5 headline (Corollary 1).

The headline currently certifies the risk UCB and the coverage LCB EACH at the
full delta=0.10 (the "auxiliary-level" convention: each statement holds at level
delta on its own). Corollary 1 gives the SIMULTANEOUS statement -- both bounds
hold jointly at overall level delta -- by a union bound with
delta_r = delta_c = delta/2. Since certify_best_gamma applies its `delta`
argument to BOTH the conditional risk certificate (eps=delta/3J under box) and
the coverage LCB (eps=delta/2J), the simultaneous recomputation is exactly a run
at delta=0.05.

This script reports, per headline cell x alpha, the OLD (delta=0.10, auxiliary)
vs NEW (delta=0.05, simultaneous Cor. 1) CertifiedCoverage and median worst-group
UCB, so the draft can decide whether the (expected-marginal) tightening is worth
switching the Section 5.1 metric definition.

Output: runs/delta_split_recompute.csv
Run: python -m fedcore.experiments.exp_delta_split   (CPU, no torch)
"""

from __future__ import annotations

import glob
import os

import numpy as np

from fedcore.certify import certify_best_gamma
from fedcore.grouping import make_group_map, repartition_trusted_pool
from fedcore.io_utils import atomic_write_csv
from fedcore.scores import scored_views

GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
MARGIN, BOX, CERT_FRAC, TEST_FRAC = 0.01, 0.15, 0.5, 0.2
SCORE, G, SEED = "msp", 2, 0
ALPHAS = (0.10, 0.20)
DELTA_OLD = 0.10          # auxiliary: each bound at full delta
DELTA_NEW = 0.05          # simultaneous (Cor. 1): delta_r = delta_c = delta/2

CELLS = [
    ("resnet18gn", "5",   "runs/cifar10_d5_resnet18gn_none0.0_seed{s}_logits.npz"),
    ("resnet18gn", "0.5", "runs/cifar10_d0.5_resnet18gn_none0.0_seed{s}_logits.npz"),
    ("resnet18",   "5",   "runs/cifar10_d5_resnet18_seed{s}_logits.npz"),
]
SEEDS = (0, 1, 2, 3, 4)


def _cov_ucb(npz, alpha, delta):
    """(cert_coverage_lcb-if-certified, cert_risk_ucb, certified) at worst-group G=2."""
    d = np.load(npz)
    n_clients = int(d["cert_client"].max()) + 1
    pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}
    parts = repartition_trusted_pool(pool, CERT_FRAC, TEST_FRAC, seed=SEED)
    gmap = make_group_map(n_clients, G)
    gviews = {}
    for fn in ("prop", "cert", "test"):
        v = scored_views(parts[fn]["logits"], parts[fn]["y_open"], parts[fn]["client"], [SCORE])[SCORE]
        v = dict(v); v["client"] = gmap[np.asarray(v["client"])]
        gviews[fn] = v
    r = certify_best_gamma(gviews["prop"], gviews["cert"], gviews["test"], score_name=SCORE,
                           gammas=GAMMAS, alpha=alpha, delta=delta, n_clients=G,
                           dirichlet_alpha=float("nan"), Lambda="box", box=BOX, seed=SEED, margin=MARGIN)
    certified = bool(r["certified"])
    return (r["cert_coverage_lcb"] if certified else 0.0), float(r["cert_risk_ucb"]), certified


def _summ(vals):
    # HEADLINE CertifiedCoverage convention (as in fedcore/aggregate/main.py):
    # mean over ALL seeds of (cert_coverage_lcb if certified else 0).
    covs = [v[0] for v in vals]                          # v[0] already 0.0 when not certified
    ucbs = [v[1] for v in vals if v[2] and np.isfinite(v[1])]
    n_cert = sum(v[2] for v in vals)
    cov_mean = float(np.mean(covs)) if covs else 0.0
    med_ucb = float(np.median(ucbs)) if ucbs else float("inf")
    return cov_mean, med_ucb, n_cert


def main() -> None:
    rows = []
    print(f"delta-split recompute  (worst-group G={G}, fixed {SCORE}, cert_frac={CERT_FRAC}, "
          f"box, margin={MARGIN})")
    print(f"OLD=auxiliary delta={DELTA_OLD} (each bound); NEW=simultaneous Cor.1 delta={DELTA_NEW} (delta/2 each)\n")
    hdr = (f"{'cell':>18} {'alpha':>5} | {'CertCov_old':>11} {'CertCov_new':>11} {'dCov':>7} | "
           f"{'medUCB_old':>10} {'medUCB_new':>10} {'dUCB':>7} | {'pass_old':>8} {'pass_new':>8}")
    print(hdr); print("-" * len(hdr))
    for backbone, dd, pat in CELLS:
        for alpha in ALPHAS:
            vold, vnew = [], []
            for s in SEEDS:
                p = pat.format(s=s)
                if not os.path.exists(p):
                    continue
                vold.append(_cov_ucb(p, alpha, DELTA_OLD))
                vnew.append(_cov_ucb(p, alpha, DELTA_NEW))
            co, uo, no = _summ(vold)
            cn, un, nn = _summ(vnew)
            rows.append({"backbone": backbone, "d": dd, "alpha": alpha, "n_seeds": len(vold),
                         "CertCov_old_aux": round(co, 4), "CertCov_new_simul": round(cn, 4),
                         "d_CertCov": round(cn - co, 4),
                         "medUCB_old_aux": round(uo, 4), "medUCB_new_simul": round(un, 4),
                         "d_medUCB": round(un - uo, 4),
                         "n_pass_old": no, "n_pass_new": nn})
            uo_s = f"{uo:.4f}" if np.isfinite(uo) else "inf"
            un_s = f"{un:.4f}" if np.isfinite(un) else "inf"
            print(f"{backbone+'/d'+dd:>18} {alpha:>5.2f} | {co:>11.4f} {cn:>11.4f} {cn-co:>+7.4f} | "
                  f"{uo_s:>10} {un_s:>10} {un-uo:>+7.4f} | {no:>7}/{len(vold)} {nn:>7}/{len(vnew)}")

    out = "runs/delta_split_recompute.csv"
    atomic_write_csv(out, list(rows[0].keys()), rows)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
