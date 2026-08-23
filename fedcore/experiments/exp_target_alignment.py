"""P0-6 target-alignment check: is the reported held-out risk the certificate's
target functional?

The grouped headline rows certify the BOX-Lambda robust functional
``sup_{mu in Lambda_G} (sum_g mu_g a_g r_g) / (sum_g mu_g a_g)`` (Theorem 1'),
but the canonical ``test_risk`` metric (selector.empirical_risk_coverage) is the
POOLED risk of the test fold -- the functional at the single realized mixture
``mu_g \\propto n_g a_g`` -- which is neither the simplex target ``max_g r_g``
nor the box-robust target. This script recomputes, per stored run, the three
empirical held-out functionals under the EXACT headline protocol
(pool original folds -> repartition seed=0 -> G=2 box-Lambda best-gamma):

  pooled   = sum_g A_g r_g / sum_g A_g          (current ``test_risk``)
  max_g    = max_g r_g                          (full-simplex target)
  robust   = sup_{mu in Lambda_G(rho)} ...      (box-Lambda target, EXACT)

and tallies violations (> alpha) of each among certified runs.

Exactness of ``robust``: the certificate's Lambda is {raw/sum(raw) : raw in
[1/G-rho, 1/G+rho]^G} (cp._sample_lambdas). The ratio functional is scale-
invariant, so sup over the normalized set equals sup over the raw box; the
objective is linear-fractional in raw, so the sup is attained at a prefix
vertex in r-descending order (same argument as theorem1._inner_sup_over_a with
the roles of lambda and a swapped and a FIXED at its empirical value). We scan
the G+1 prefix vertices -- exact, no sampling.

Run (CPU, no torch):
  python -m fedcore.experiments.exp_target_alignment \
      [--alphas 0.10 0.20] [--gammas 0.5 0.7 1.0] [--out runs/target_alignment.csv]
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

from fedcore.certify import certify_best_gamma_grouped
from fedcore.grouping import make_group_map, repartition_trusted_pool, views_from_parts
from fedcore.selector import Selector, choose_threshold, open_set_error

CERT_FRAC, TEST_FRAC = 0.5, 0.2
DELTA, BOX, MARGIN, G, SCORE = 0.10, 0.15, 0.01, 2, "msp"

# Table 5(a) cells: (label, filename glob)
CELLS = (
    ("cifar10/GN/d5",   "runs/cifar10_d5_resnet18gn_none0.0_seed*_logits.npz"),
    ("cifar10/GN/d0.5", "runs/cifar10_d0.5_resnet18gn_none0.0_seed*_logits.npz"),
    ("cifar10/BN/d5",   "runs/cifar10_d5_resnet18_seed*_logits.npz"),
)


def robust_box_sup(a: np.ndarray, r: np.ndarray, G: int, rho: float) -> tuple[float, np.ndarray]:
    """EXACT sup_{mu in Lambda_G(rho)} (sum mu a r)/(sum mu a) at FIXED empirical a, r.

    Scale-invariance reduces the normalized Lambda to the raw box
    [max(0, 1/G - rho), 1/G + rho]^G; linear-fractionality puts the sup at a
    prefix vertex in r-descending order. Groups with a_g == 0 drop out.
    Returns (sup, argmax raw-mu vertex, normalized).
    """
    lo, hi = max(0.0, 1.0 / G - rho), 1.0 / G + rho
    live = a > 0
    if not live.any():
        return float("nan"), np.full(G, np.nan)
    order = np.argsort(-np.where(live, r, -np.inf), kind="stable")
    mu = np.full(G, lo)
    best, best_mu = -np.inf, None
    for k in range(G + 1):
        if k > 0:
            mu[order[k - 1]] = hi
        denom = float(np.sum(mu * a))
        if denom <= 0.0:
            continue
        val = float(np.sum(mu * a * r) / denom)
        if val > best:
            best, best_mu = val, mu.copy()
    return best, best_mu / best_mu.sum()


def group_stats(view: dict, sel: Selector, G: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-group (A, K, n, a) on a fold whose ``client`` is already group-relabeled."""
    err = open_set_error(view["pred"], view["y_open"])
    acc = sel.accept(view["score"])
    grp = np.asarray(view["client"])
    A = np.zeros(G, dtype=int); K = np.zeros(G, dtype=int); n = np.zeros(G, dtype=int)
    for g in range(G):
        m = grp == g
        n[g] = int(m.sum())
        A[g] = int((m & acc).sum())
        K[g] = int((m & acc & err).sum())
    a = np.where(n > 0, A / np.maximum(n, 1), 0.0)
    return A, K, n, a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.10, 0.20])
    # default = aggregate/main.py GAMMAS, so the no-arg invocation is protocol-exact
    # for the stored Table-5(a) aggregates (the draft caption's {0.5,0.7,1.0} is the
    # t8.py detector grid and does NOT reproduce the 5(a) alpha=0.10 counts).
    ap.add_argument("--gammas", type=float, nargs="*", default=[0.2, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--out", default="runs/target_alignment.csv")
    args = ap.parse_args()

    rows = []
    for label, pattern in CELLS:
        for path in sorted(glob.glob(pattern)):
            seed = int(re.search(r"seed(\d+)", path).group(1))
            d = np.load(path)
            n_clients = int(d["cert_client"].max()) + 1
            pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
                    for k in ("logits", "y_open", "client")}
            parts = repartition_trusted_pool(pool, CERT_FRAC, TEST_FRAC, seed=0)
            views = views_from_parts(parts, SCORE)
            gmap = make_group_map(n_clients, G)
            gviews = {fn: {**v, "client": gmap[np.asarray(v["client"])]}
                      for fn, v in views.items()}

            for alpha in args.alphas:
                res = certify_best_gamma_grouped(
                    views["prop"], views["cert"], views["test"], score_name=SCORE,
                    group_map=gmap, G=G, gammas=tuple(args.gammas), alpha=alpha,
                    delta=DELTA, Lambda="box", box=BOX, seed=0, margin=MARGIN)

                # rebuild the chosen selector to compute group-resolved TEST stats
                sel = choose_threshold(views["prop"]["score"], views["prop"]["pred"],
                                       views["prop"]["y_open"], res["gamma_star"], alpha)
                A, K, n, a = group_stats(gviews["test"], sel, G)
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.where(A > 0, K / np.maximum(A, 1), np.nan)

                # canonical convention (selector.empirical_risk_coverage): risk = 0.0
                # when nothing is accepted (vacuously safe)
                pooled = float(K.sum() / A.sum()) if A.sum() > 0 else 0.0
                live = A > 0
                maxg = float(np.nanmax(np.where(live, r, np.nan))) if live.any() else float("nan")
                robust, mu_star = robust_box_sup(a, np.nan_to_num(r, nan=0.0), G, BOX)

                assert abs(pooled - res["test_risk"]) < 1e-12, \
                    f"pooled != canonical test_risk on {path}"  # semantic check

                rows.append({
                    "cell": label, "seed": seed, "alpha": alpha,
                    "certified": int(res["certified"]), "gamma_star": res["gamma_star"],
                    "cert_risk_ucb": round(res["cert_risk_ucb"], 4),
                    "pooled_test_risk": round(pooled, 4),
                    "max_g_r_test": round(maxg, 4) if np.isfinite(maxg) else "",
                    "robust_box_test": round(robust, 4) if np.isfinite(robust) else "",
                    "mu_star_g0": round(float(mu_star[0]), 3) if np.isfinite(mu_star[0]) else "",
                    "A_g": "|".join(map(str, A)), "K_g": "|".join(map(str, K)),
                    "viol_pooled": int(res["certified"] and np.isfinite(pooled) and pooled > alpha),
                    "viol_maxg": int(res["certified"] and np.isfinite(maxg) and maxg > alpha),
                    "viol_robust": int(res["certified"] and np.isfinite(robust) and robust > alpha),
                })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = list(rows[0].keys())
    with open(args.out, "w") as f:
        f.write(",".join(fields) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fields) + "\n")

    print(f"{'cell':>16} {'alpha':>6} {'cert':>5} {'viol P/M/R':>11}  "
          f"{'max robust (certified)':>22}  {'max max_g':>10}  {'max pooled':>10}")
    for label, _ in CELLS:
        for alpha in args.alphas:
            sub = [x for x in rows if x["cell"] == label and x["alpha"] == alpha]
            cert = [x for x in sub if x["certified"]]
            vp = sum(x["viol_pooled"] for x in cert)
            vm = sum(x["viol_maxg"] for x in cert)
            vr = sum(x["viol_robust"] for x in cert)
            mx = lambda k: (max((float(x[k]) for x in cert if x[k] != ""), default=float("nan")))
            print(f"{label:>16} {alpha:>6.2f} {len(cert):>2}/{len(sub):<2} "
                  f"{vp:>3}/{vm}/{vr}   {mx('robust_box_test'):>22.4f}  "
                  f"{mx('max_g_r_test'):>10.4f}  {mx('pooled_test_risk'):>10.4f}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
