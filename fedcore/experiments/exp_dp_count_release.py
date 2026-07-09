"""M7: DP count-release ablation -- empirical cost of a formal DP release of the grouped
counts (A_g, K_g). NO DP theorem is claimed; this is an EMPIRICAL cost curve only.

Grouped G=2 on the stored GN (resnet18gn) CIFAR-10 d=5 logits. The certificate needs only
the per-group pairs (A_g, K_g). We release them under Laplace mechanism DP:

  * Each released count has add/remove-one sensitivity 1; we split the per-group budget
    across its two counts (eps/2 each) -> Laplace scale b = 2/eps per count. Groups are a
    DISJOINT partition of the audit points, so parallel composition keeps the WHOLE release
    eps-DP (not G*eps).
  * "Widen the CP to absorb the noise": spend a small delta_dp on a Laplace tail bound
    t = b * ln(2G / delta_dp) (union over the 2G released counts, P(|Lap(b)|>t)=e^{-t/b}),
    then feed the CONSERVATIVE counts A_g^ = max(0, A_g+noise - t), K_g^ = clip(K_g+noise + t)
    (fewer accepted, more errors) to the certificate at the remaining budget delta_cert.
    Group sizes n_g are treated as PUBLIC (grouped-stratified releases group membership).

delta budget: delta_total = 0.10 = delta_cert (0.08) + delta_dp (0.02).
For each (seed, epsilon, alpha) we Monte-Carlo N_NOISE noise draws and report the MEAN
risk UCB, MEAN certified coverage LCB (0 when a draw is not certified), and the
certification RATE over draws. epsilon = inf is the no-DP baseline.

CSV: runs/dp_count_release.csv
  columns: epsilon,seed,alpha,cert_risk_ucb,cert_coverage_lcb,certified

Run: python experiments/fedcore/exp_dp_count_release.py   (CPU, no torch)
"""
from __future__ import annotations

import glob
import math
import os
import re

import numpy as np

from fedcore.certify import certify_best_gamma_grouped
from fedcore.certificate import conditional_risk_certificate, cp_lower, _sample_lambdas
from fedcore.grouping import make_group_map, repartition_trusted_pool, views_from_parts
from fedcore.selector import choose_threshold, counts_per_client

GLOB = "runs/cifar10_d5_resnet18gn_none0.0_seed[0-9]_logits.npz"
G = 2
DELTA_TOTAL, DELTA_DP = 0.10, 0.02
DELTA_CERT = DELTA_TOTAL - DELTA_DP
CERT_FRAC, TEST_FRAC = 0.5, 0.2
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
BOX, MARGIN, SEED = 0.15, 0.01, 0
ALPHAS = (0.10, 0.20)
EPSILONS = (1.0, 3.0, 10.0, float("inf"))
N_NOISE = 300
OUT = "runs/dp_count_release.csv"
FIELDS = ["epsilon", "seed", "alpha", "cert_risk_ucb", "cert_coverage_lcb", "certified"]


def _coverage_lcb_box(A, n, delta, box, rng):
    J = len(A)
    eps = delta / (2.0 * J)
    alow = np.array([cp_lower(int(A[j]), int(n[j]), eps) for j in range(J)])
    lams = _sample_lambdas(J, box, 256, rng)
    return float(min(np.sum(l * alow) for l in lams))


def _clean_group_counts(npz, alpha):
    """Selector-on-proposal + per-group (A,K,n) on the cert fold, grouped G=2."""
    z = np.load(npz)
    pool = {k: np.concatenate([z[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}
    n_clients = int(pool["client"].max()) + 1
    parts = repartition_trusted_pool(pool, CERT_FRAC, TEST_FRAC, seed=SEED)
    gmap = make_group_map(n_clients, G)
    views = views_from_parts(parts, "msp")
    gviews = {}
    for fn in ("prop", "cert", "test"):
        v = dict(views[fn]); v["client"] = gmap[np.asarray(views[fn]["client"])]
        gviews[fn] = v
    res = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"],
                                     score_name="msp", group_map=gmap, G=G, gammas=GAMMAS,
                                     alpha=alpha, delta=DELTA_CERT, Lambda="box", box=BOX,
                                     seed=SEED, margin=MARGIN)
    sel = choose_threshold(gviews["prop"]["score"], gviews["prop"]["pred"],
                           gviews["prop"]["y_open"], res["gamma_star"], alpha)
    A, K, n = counts_per_client(gviews["cert"]["score"], gviews["cert"]["pred"],
                                gviews["cert"]["y_open"], gviews["cert"]["client"], sel, G)
    return np.asarray(A, float), np.asarray(K, float), np.asarray(n, float)


def _dp_certificate(A, K, n, alpha, eps, rng):
    """One noisy release -> conservative counts -> box certificate. Returns (ucb, cov_lcb, cert)."""
    if math.isinf(eps):
        Ac, Kc = A.copy(), K.copy()
    else:
        b = 2.0 / eps                                   # per-count Laplace scale (eps/2 each)
        t = b * math.log(2 * G / DELTA_DP)              # union tail bound over 2G counts
        An = A + rng.laplace(0.0, b, size=G)
        Kn = K + rng.laplace(0.0, b, size=G)
        Ac = np.maximum(0.0, An - t)                    # conservative: fewer accepted
        Kc = np.clip(Kn + t, 0.0, Ac)                   # conservative: more errors, <= Ac
    cert = conditional_risk_certificate(Ac, Kc, n, DELTA_CERT, Lambda="box", box=BOX, seed=SEED)
    certified = bool(cert.feasible and cert.U <= alpha)
    cov = _coverage_lcb_box(Ac, n, DELTA_CERT, BOX, rng) if certified else 0.0
    # a vacuous / infeasible certificate is a risk bound of 1.0 (can't certify anything)
    ucb = min(float(cert.U), 1.0) if np.isfinite(cert.U) else 1.0
    return ucb, float(cov), certified


def main():
    files = sorted(glob.glob(GLOB))
    if not files:
        print(f"[warn] no GN d5 logits ({GLOB})"); return
    print(f"M7 DP count-release: GN d5, G={G}, delta_cert={DELTA_CERT}, delta_dp={DELTA_DP}, "
          f"N_noise={N_NOISE}, seeds={len(files)}")
    rng = np.random.default_rng(0)
    rows = []
    for f in files:
        seed = int(re.search(r"seed(\d+)", os.path.basename(f)).group(1))
        for alpha in ALPHAS:
            A, K, n = _clean_group_counts(f, alpha)
            for eps in EPSILONS:
                if math.isinf(eps):
                    ucb, cov, cert = _dp_certificate(A, K, n, alpha, eps, rng)
                    ucb_m, cov_m, cert_rate = ucb, cov, float(cert)
                else:
                    ucbs, covs, certs = [], [], []
                    for _ in range(N_NOISE):
                        u, c, ct = _dp_certificate(A, K, n, alpha, eps, rng)
                        ucbs.append(u); covs.append(c); certs.append(ct)
                    ucb_m, cov_m, cert_rate = np.mean(ucbs), np.mean(covs), np.mean(certs)
                rows.append({"epsilon": eps, "seed": seed, "alpha": alpha,
                             "cert_risk_ucb": round(float(ucb_m), 4),
                             "cert_coverage_lcb": round(float(cov_m), 4),
                             "certified": round(float(cert_rate), 4)})
    from fedcore.io_utils import atomic_write_csv
    atomic_write_csv(OUT, FIELDS, rows)
    print(f"saved {OUT}  ({len(rows)} rows)\n")
    print(f"{'eps':>5} {'alpha':>5} {'risk_ucb':>9} {'cov_lcb':>9} {'cert_rate':>9}")
    for eps in EPSILONS:
        for alpha in ALPHAS:
            sub = [r for r in rows if r["epsilon"] == eps and abs(r["alpha"] - alpha) < 1e-9]
            if sub:
                print(f"{eps:>5} {alpha:>5.2f} {np.mean([r['cert_risk_ucb'] for r in sub]):>9.4f} "
                      f"{np.mean([r['cert_coverage_lcb'] for r in sub]):>9.4f} "
                      f"{np.mean([r['certified'] for r in sub]):>9.4f}")


if __name__ == "__main__":
    main()
