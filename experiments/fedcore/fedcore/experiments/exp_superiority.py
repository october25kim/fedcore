"""STEP 4(g): superiority -- price of federation + baseline comparison harness.

Two parts:

1. **Price of federation.** As heterogeneity -> 0, the federated conditional
   certificate (Theorem 1, valid under heterogeneity) should approach the
   centralized oracle bound (pooled CP, valid ONLY under matched i.i.d.). We
   sweep the per-client error spread to zero and report the median certified-U
   gap ``cond_simplex - pooled``; it shrinks toward 0 as clients homogenize.

2. **Baseline comparison harness.** :func:`compare_to_baselines` tabulates
   Fed-CORE matched-risk certified coverage against externally supplied
   oracle-tuned baselines (FedPD / FedOSS / FOOGD). Those numbers must come from
   the baselines' own codebases (they are not reimplemented here); the harness
   only formats the comparison and flags whether Fed-CORE is the only method with
   a finite-sample risk certificate.

Run: ``python experiments/fedcore/exp_superiority.py``  (CPU, no torch)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from fedcore.certificate import conditional_risk_certificate, pooled_cp
from fedcore.data.clients import ClientPopulation, draw_counts

DELTA = 0.10
N_TRIALS = 1500
N_CLIENTS = 5
N_PER_CLIENT = 300
SEED = 0


def price_of_federation() -> None:
    a = np.full(N_CLIENTS, 0.6)
    n = np.full(N_CLIENTS, N_PER_CLIENT)
    base_r = 0.10
    print("STEP 4(g.1) price of federation: certified-U gap vs heterogeneity")
    print(f"{'r_spread':>9} {'pooled':>8} {'cond_sx':>8} {'gap':>8}")
    print("-" * 36)
    for spread in (0.30, 0.20, 0.10, 0.05, 0.0):
        # error rates spread symmetrically around base_r; spread=0 => homogeneous
        r = np.linspace(base_r - spread, base_r + spread, N_CLIENTS)
        r = np.clip(r, 0.001, 0.999)
        pop = ClientPopulation(a=a, r=r)
        rng = np.random.default_rng(SEED)
        up, us = [], []
        for _ in range(N_TRIALS):
            A, K = draw_counts(pop, n, rng)
            up.append(pooled_cp(A, K, DELTA))
            us.append(conditional_risk_certificate(A, K, n, DELTA, Lambda="simplex").U)
        mp, ms = np.median(up), np.median(us)
        print(f"{spread:>9.2f} {mp:>8.3f} {ms:>8.3f} {ms - mp:>8.3f}")
    print("-" * 36)
    print("Gap -> 0 as heterogeneity -> 0: the price of federation vanishes when "
          "clients homogenize (pooled oracle valid only there).")


def compare_to_baselines(
    fedcore_coverage: float,
    baselines: Optional[Dict[str, float]] = None,
    alpha: float = 0.10,
) -> None:
    """Tabulate matched-risk certified coverage vs oracle-tuned baselines.

    ``baselines`` maps method name -> achieved coverage at matched risk ``alpha``
    (obtained from each baseline's own codebase). Only Fed-CORE provides a
    finite-sample risk CERTIFICATE; the others are oracle-tuned point estimates.
    """
    print(f"\nSTEP 4(g.2) matched-risk coverage @ alpha={alpha} "
          f"(baselines from their own codebases)")
    print(f"{'method':>12} {'coverage':>9} {'risk certificate?':>18}")
    print("-" * 42)
    print(f"{'Fed-CORE':>12} {fedcore_coverage:>9.4f} {'YES (Thm 1/1prime)':>18}")
    if baselines:
        for name, cov in baselines.items():
            print(f"{name:>12} {cov:>9.4f} {'no (oracle-tuned)':>18}")
    else:
        print("(no baseline numbers supplied; plug FedPD/FedOSS/FOOGD coverage "
              "from their repos via compare_to_baselines(...))")


# --------------------------------------------------------------------------- #
# C3 superiority table T4 (matched-risk: oracle vs naive vs Fed-CORE)
# --------------------------------------------------------------------------- #
def _oracle_coverage(test_score, test_err, alpha, n_grid=400):
    """Upper bound: max test coverage s.t. realized TEST risk <= alpha.

    PEEKS AT TEST LABELS -- used only to define the oracle-tuned upper-bound row.
    """
    qs = np.unique(np.quantile(test_score, np.linspace(0, 1, n_grid)))
    best_cov, best_risk = 0.0, 0.0
    for t in qs:
        acc = test_score >= t
        n = int(acc.sum())
        if n == 0:
            continue
        cov = n / len(test_score)
        risk = float(test_err[acc].mean())
        if risk <= alpha and cov > best_cov:
            best_cov, best_risk = cov, risk
    return best_cov, best_risk


def build_T4(npz_path, dlabel, alphas=(0.10, 0.20), scores=("msp", "energy", "score_norm"),
             delta=0.10, gammas=(0.2, 0.3, 0.5, 0.7, 1.0), margin=0.01):
    """Build T4 rows for one exported-logits npz."""
    from certify import certify_best_gamma, certify_best_gamma_grouped
    from scores import compute_score, scored_views
    from selector import choose_threshold, empirical_risk_coverage, open_set_error

    d = np.load(npz_path)
    n_clients = int(d["cert_client"].max()) + 1
    rows = []
    for s in scores:
        views = {fn: scored_views(d[f"{fn}_logits"], d[f"{fn}_y_open"],
                                  d[f"{fn}_client"], [s])[s] for fn in ("prop", "cert", "test")}
        test_err = open_set_error(views["test"]["pred"], views["test"]["y_open"])
        prop_err = open_set_error(views["prop"]["pred"], views["prop"]["y_open"])
        for alpha in alphas:
            # (A) oracle-tuned (test-peeking upper bound)
            o_cov, o_risk = _oracle_coverage(views["test"]["score"], test_err, alpha)
            rows.append([dlabel, alpha, s, "oracle(test-peek)", o_cov, o_risk,
                         o_risk <= alpha, "NO (peeks test)"])
            # (B) naive no-peek: empirical risk<=alpha on PROPOSAL, eval on TEST
            sel = choose_threshold(views["prop"]["score"], views["prop"]["pred"],
                                   views["prop"]["y_open"], gamma=1.0, alpha=alpha)
            n_cov, n_risk = empirical_risk_coverage(views["test"]["score"], test_err, sel.threshold)
            rows.append([dlabel, alpha, s, "naive(no-cert)", n_cov, n_risk,
                         n_risk <= alpha, "NO (no certificate)"])
            # (C) Fed-CORE per-client (G=J) box
            r = certify_best_gamma(views["prop"], views["cert"], views["test"],
                                   score_name=s, gammas=gammas, alpha=alpha, delta=delta,
                                   n_clients=n_clients, dirichlet_alpha=float("nan"),
                                   Lambda="box", box=0.15, seed=0, margin=margin)
            fc_cov = r["test_coverage"] if r["certified"] else 0.0
            rows.append([dlabel, alpha, s, "Fed-CORE(G=J)", fc_cov, r["test_risk"],
                         r["certified"], "YES (Thm 1/1prime)" if r["certified"] else "n/a (uncertified)"])
            # (C') Fed-CORE grouped G=2 (worst-group, legitimate)
            gmap = np.array([c * 2 // n_clients for c in range(n_clients)])
            rg = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"],
                                            score_name=s, group_map=gmap, G=2, gammas=gammas,
                                            alpha=alpha, delta=delta, Lambda="box", box=0.15,
                                            seed=0, margin=margin)
            g_cov = rg["test_coverage"] if rg["certified"] else 0.0
            rows.append([dlabel, alpha, s, "Fed-CORE(G=2)", g_cov, rg["test_risk"],
                         rg["certified"], "YES (worst-group)" if rg["certified"] else "n/a (uncertified)"])
    return rows


def main() -> None:
    import csv
    import glob
    price_of_federation()

    runs = {
        "cifar10_d5": "runs/cifar10_d5_none0.0_seed0_logits.npz",
        "cifar10_d0.5": "runs/cifar10_d0.5_none0.0_seed0_logits.npz",
    }
    # allow running from repo root or experiments/fedcore
    base = "" if glob.glob("runs/*.npz") else "../../"
    all_rows = []
    print("\nSTEP 5 C3 superiority -- T4 (matched-risk coverage; oracle peeks test)")
    print(f"{'d':>12} {'alpha':>6} {'score':>11} {'method':>20} {'cov':>7} "
          f"{'test_risk':>9} {'risk<=a':>8} {'guarantee':>20}")
    print("-" * 100)
    for dlabel, p in runs.items():
        path = base + p
        if not glob.glob(path):
            print(f"  (skip {dlabel}: {path} not found)")
            continue
        rows = build_T4(path, dlabel)
        for r in rows:
            print(f"{r[0]:>12} {r[1]:>6.2f} {r[2]:>11} {r[3]:>20} {r[4]:>7.3f} "
                  f"{r[5]:>9.3f} {str(r[6]):>8} {r[7]:>20}")
        all_rows.extend(rows)

    if all_rows:
        out = (base + "runs/T4.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["d", "alpha", "score", "method", "coverage", "test_risk",
                        "risk_le_alpha", "guarantee"])
            w.writerows(all_rows)
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
