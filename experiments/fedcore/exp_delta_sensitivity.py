"""Appendix: delta-sensitivity of CertifiedCoverage (is the guarantee level a
hidden tuning knob?).

Extends exp_delta_split.py from a single OLD/NEW comparison to a full sweep of
the overall guarantee level delta in {0.05, 0.10, 0.20} at alpha in {0.10, 0.20},
on the stored clean GN/BN CIFAR-10 logits. Protocol is byte-for-byte that of
exp_delta_split.py: worst-group G=2, fixed MSP, best-gamma over {0.2,..,1.0},
cert_frac=0.5, box-Lambda, margin=0.01, seed-0 fold repartition. Each cell is
certified under the SIMULTANEOUS (Corollary 1) budget delta_r = delta_c = delta/2,
so a target overall level `delta` is a certify_best_gamma call at delta/2 (the
same reduction exp_delta_split uses: overall 0.10 == a delta=0.05 run).

The point is an appendix table: CertifiedCoverage should degrade GRACEFULLY as
delta shrinks (a stronger guarantee costs coverage) rather than collapse or
require re-tuning -- i.e. the guarantee level is an honest dial, not a hidden
knob propping up the headline.

Output: runs/delta_sensitivity.csv
  columns: backbone, d, alpha, delta, n_seeds, cert_frac_seeds,
           CertCov_mean, CertCov_std, certucb_median
Run: python experiments/fedcore/exp_delta_sensitivity.py   (CPU, no torch)
"""

from __future__ import annotations

import os

import numpy as np

from fedcore.certify import certify_best_gamma
from fedcore.grouping import make_group_map, repartition_trusted_pool
from fedcore.io_utils import atomic_write_csv
from fedcore.scores import scored_views

# --- protocol constants: identical to exp_delta_split.py ---
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
MARGIN, BOX, CERT_FRAC, TEST_FRAC = 0.01, 0.15, 0.5, 0.2
SCORE, G, SEED = "msp", 2, 0
ALPHAS = (0.10, 0.20)
DELTAS = (0.05, 0.10, 0.20)          # overall guarantee levels to sweep

CELLS = [
    ("resnet18gn", "5",   "runs/cifar10_d5_resnet18gn_none0.0_seed{s}_logits.npz"),
    ("resnet18gn", "0.5", "runs/cifar10_d0.5_resnet18gn_none0.0_seed{s}_logits.npz"),
    ("resnet18",   "5",   "runs/cifar10_d5_resnet18_seed{s}_logits.npz"),
]
SEEDS = (0, 1, 2, 3, 4)


def _cov_ucb(npz, alpha, delta):
    """(cert_coverage_lcb-if-certified, cert_risk_ucb, certified) at worst-group
    G=2. `delta` is the certify_best_gamma argument (== overall_level / 2 for the
    simultaneous Corollary-1 budget). Identical to exp_delta_split._cov_ucb."""
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
    # HEADLINE CertifiedCoverage convention: mean over ALL seeds of
    # (cert_coverage_lcb if certified else 0).
    covs = [v[0] for v in vals]                           # v[0] already 0.0 when not certified
    ucbs = [v[1] for v in vals if v[2] and np.isfinite(v[1])]
    n_cert = sum(v[2] for v in vals)
    cov_mean = float(np.mean(covs)) if covs else 0.0
    cov_std = float(np.std(covs)) if covs else 0.0
    med_ucb = float(np.median(ucbs)) if ucbs else float("inf")
    return cov_mean, cov_std, med_ucb, n_cert


def main() -> None:
    rows = []
    print(f"delta-sensitivity sweep  (worst-group G={G}, fixed {SCORE}, cert_frac={CERT_FRAC}, "
          f"box, margin={MARGIN}; simultaneous Cor.1 budget delta/2 per bound)")
    hdr = (f"{'cell':>18} {'alpha':>5} {'delta':>6} | {'CertCov_mean':>12} {'CertCov_std':>11} "
           f"{'certucb_med':>11} {'cert_frac':>9}")
    print(hdr); print("-" * len(hdr))
    for backbone, dd, pat in CELLS:
        for alpha in ALPHAS:
            for delta in DELTAS:
                vals = []
                for s in SEEDS:
                    p = pat.format(s=s)
                    if not os.path.exists(p):
                        continue
                    vals.append(_cov_ucb(p, alpha, delta / 2.0))   # simultaneous delta/2 budget
                cov_mean, cov_std, med_ucb, n_cert = _summ(vals)
                n_seeds = len(vals)
                rows.append({"backbone": backbone, "d": dd, "alpha": alpha, "delta": delta,
                             "n_seeds": n_seeds,
                             "cert_frac_seeds": round(n_cert / n_seeds, 4) if n_seeds else 0.0,
                             "CertCov_mean": round(cov_mean, 4), "CertCov_std": round(cov_std, 4),
                             "certucb_median": round(med_ucb, 4) if np.isfinite(med_ucb) else float("inf")})
                mu = f"{med_ucb:.4f}" if np.isfinite(med_ucb) else "inf"
                print(f"{backbone+'/d'+dd:>18} {alpha:>5.2f} {delta:>6.2f} | {cov_mean:>12.4f} "
                      f"{cov_std:>11.4f} {mu:>11} {n_cert}/{n_seeds:>7}")

    out = "runs/delta_sensitivity.csv"
    atomic_write_csv(out, list(rows[0].keys()), rows)
    print(f"\nsaved {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
