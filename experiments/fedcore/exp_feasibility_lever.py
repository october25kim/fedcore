"""STEP 5.1: the feasibility lever -- grouped-stratified + calibration size.

Post-hoc (no retraining) test of the alpha=0.1 diagnosis: CertifiedCoverage is
bound by per-GROUP accepted counts vs the Theorem-2 floor, not by the operating
point. From the SAME exported cert/test logits (e.g. cifar10 d=5 clean) we pool
the trusted folds and sweep:

  (a) grouping G in {5 (per-client), 3, 2, 1 (pooled)}  -- a PUBLIC, fixed
      client->group map; larger groups carry more accepted points;
  (b) calibration size: cert-fold fraction in {0.33, 0.5, 0.7} of the trusted pool.

For each (G, frac) we run the validity-preserving certify_best_gamma over groups
(box-Lambda) at alpha=0.10 and record per-group accepted count, cert_risk_ucb,
CertifiedCoverage@0.1, and test_risk. We overlay the Theorem-2 floor
ln(G/delta)/(-ln(1-alpha)) and the normal (alpha - rhat)^2 count requirement
z^2 rhat(1-rhat)/(alpha-rhat)^2. This turns the honest null into a quantitative
feasibility law / staircase.

VALIDITY: the model is fixed and all trusted points are exchangeable clean
test-set points, so repartitioning them into disjoint prop/cert/test is valid; the
group map is public and data-independent; G=1 (pooled) is valid only under matched
mixture (near-IID d=5) and is flagged as the bonus, kept subordinate to G>=2.

Run: ``python experiments/fedcore/exp_feasibility_lever.py [--npz runs/<tag>_logits.npz]``
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import norm

from certify import certify_best_gamma_grouped
from scores import scored_views

ALPHA = 0.10
DELTA = 0.10
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
SCORES = ("msp", "neg_entropy", "margin", "energy")
DEFAULT_NPZ = "runs/cifar10_d5_none0.0_seed0_logits.npz"


def _group_map(n_clients: int, G: int) -> np.ndarray:
    """Public, data-independent client->group map (contiguous balanced blocks)."""
    return np.array([c * G // n_clients for c in range(n_clients)], dtype=int)


def _repartition(pool, cert_frac, test_frac, seed):
    """Split the pooled trusted points into disjoint prop/cert/test folds."""
    rng = np.random.default_rng(seed)
    n = len(pool["y_open"])
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_cert = int(round(n * cert_frac))
    idx = {"test": perm[:n_test],
           "cert": perm[n_test:n_test + n_cert],
           "prop": perm[n_test + n_cert:]}
    out = {}
    for fold, ix in idx.items():
        out[fold] = {k: pool[k][ix] for k in ("logits", "y_open", "client")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--score", default="msp", help="fixed score (avoids selection bias)")
    args = ap.parse_args()

    d = np.load(args.npz)
    # pool all trusted folds (model is fixed; points are exchangeable)
    pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}
    n_clients = int(pool["client"].max()) + 1
    z = norm.ppf(1 - DELTA)
    print(f"feasibility lever (npz={args.npz}, score={args.score}, alpha={ALPHA}, "
          f"delta={DELTA}, n_clients={n_clients}, pooled trusted={len(pool['y_open'])})")
    print(f"{'G':>2} {'cert_frac':>9} {'avg_n/grp':>9} {'rhat':>6} {'thm2_floor':>10} "
          f"{'(a-rhat)^2 req':>14} {'cert_ucb':>9} {'CertCov@.1':>11} {'test_risk':>9} {'cert?':>6}")
    print("-" * 100)

    for G in (5, 3, 2, 1):
        gmap = _group_map(n_clients, G)
        thm2 = np.log(G / DELTA) / (-np.log(1 - ALPHA))
        for frac in (0.33, 0.5, 0.7):
            parts = _repartition(pool, frac, 0.2, seed=0)
            views = {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                                      parts[fn]["client"], [args.score])[args.score]
                     for fn in ("prop", "cert", "test")}
            r = certify_best_gamma_grouped(
                views["prop"], views["cert"], views["test"], score_name=args.score,
                group_map=gmap, G=G, gammas=GAMMAS, alpha=ALPHA, delta=DELTA,
                Lambda="box", box=0.15, seed=0)
            avg_n = r["cert_n"] / G
            rhat = (r["cert_k"] / r["cert_n"]) if r["cert_n"] else float("nan")
            if rhat == rhat and rhat < ALPHA:
                req = z ** 2 * rhat * (1 - rhat) / (ALPHA - rhat) ** 2
            else:
                req = float("inf")
            cov = r["cert_coverage_lcb"] if r["certified"] else 0.0
            tag = "G=1*" if G == 1 else ""
            print(f"{G:>2} {frac:>9.2f} {avg_n:>9.1f} {rhat:>6.3f} {thm2:>10.1f} "
                  f"{req:>14.0f} {r['cert_risk_ucb']:>9.3f} {cov:>11.4f} "
                  f"{r['test_risk']:>9.3f} {str(r['certified']):>6} {tag}")
            if r["certified"]:
                assert r["test_risk"] <= ALPHA + 1e-9, "certified but test_risk>alpha"

    print("-" * 100)
    print("* G=1 is the pooled certificate: valid only under matched mixture "
          "(near-IID d=5) -> report as bonus, subordinate to G>=2.")
    print("Reading: cert_ucb falls and CertCov@0.1 turns positive once avg per-group "
          "accepted count clears the (alpha-rhat)^2 requirement (Theorem-2 staircase).")


if __name__ == "__main__":
    main()
