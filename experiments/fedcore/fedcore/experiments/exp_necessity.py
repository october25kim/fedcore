"""Necessity-of-certificate ablation.

Without a certificate, would a practitioner deploy an unsafe selector? We sweep
the high-risk client's error rate ``r_bad`` so the true matched-mixture selective
risk ``R_sel`` crosses the target ``alpha``, and measure the *unsafe-deploy rate*
``P(deploy | R_sel > alpha)`` for three deploy rules:

* ``naive``    -- deploy iff the pooled EMPIRICAL risk ``sum K / sum A <= alpha``.
* ``pooled``   -- deploy iff the pooled CP bound ``<= alpha``.
* ``Fed-CORE`` -- deploy iff the conditional certificate (known matched lambda) ``<= alpha``.

Acceptance gate (alpha=0.05, delta=0.1): at the boundary ``R_sel = alpha``,
naive unsafe-deploy ~0.52, pooled ~0.08, Fed-CORE ~0.00.

Run: ``python experiments/fedcore/exp_necessity.py``  (CPU, no torch)
"""

from __future__ import annotations

import numpy as np

from fedcore.certificate import conditional_risk_certificate, pooled_cp, true_selective_risk
from fedcore.data.clients import ClientPopulation, draw_counts

ALPHA = 0.05
DELTA = 0.10
N_TRIALS = 3000
SEED = 0


def main() -> None:
    n_good = 4
    a_good, r_good = 0.7, 0.02
    a_bad = 0.5
    # larger bad client so the matched mixture can cross alpha within the sweep
    n = np.array([300, 300, 300, 300, 400])
    J = len(n)
    lam_matched = n / n.sum()

    r_bad_grid = np.linspace(0.02, 0.40, 25)

    print(f"Necessity ablation (alpha={ALPHA}, delta={DELTA}, matched mixture)")
    print(f"{'r_bad':>7} {'R_sel':>8} {'unsafe':>7} {'naive':>8} {'pooled':>8} {'fedcore':>8}")
    print("-" * 50)

    boundary = None
    for r_bad in r_bad_grid:
        a = np.array([a_good] * n_good + [a_bad])
        r = np.array([r_good] * n_good + [r_bad])
        pop = ClientPopulation(a=a, r=r)
        R_true = true_selective_risk(a, r, lam_matched)
        unsafe = R_true > ALPHA

        rng = np.random.default_rng(SEED)
        dep_naive = dep_pooled = dep_fed = 0
        for _ in range(N_TRIALS):
            A, K = draw_counts(pop, n, rng)
            tot_A, tot_K = int(A.sum()), int(K.sum())
            emp = tot_K / tot_A if tot_A > 0 else 0.0
            dep_naive += emp <= ALPHA
            dep_pooled += pooled_cp(A, K, DELTA) <= ALPHA
            U = conditional_risk_certificate(
                A, K, n, DELTA, Lambda="known", lam=lam_matched
            ).U
            dep_fed += U <= ALPHA
        row = (
            r_bad, R_true, unsafe,
            dep_naive / N_TRIALS, dep_pooled / N_TRIALS, dep_fed / N_TRIALS,
        )
        print(
            f"{r_bad:>7.3f} {R_true:>8.4f} {str(unsafe):>7} "
            f"{row[3]:>8.3f} {row[4]:>8.3f} {row[5]:>8.3f}"
        )
        if unsafe and boundary is None:
            boundary = row

    print("-" * 50)
    if boundary is not None:
        print(
            f"boundary (R_sel just > alpha): r_bad={boundary[0]:.3f} R_sel={boundary[1]:.4f}\n"
            f"  unsafe-deploy rate: naive={boundary[3]:.3f} (expect ~0.52), "
            f"pooled={boundary[4]:.3f} (expect ~0.08), fedcore={boundary[5]:.3f} (expect ~0.00)"
        )
    else:
        print("WARNING: R_sel never crossed alpha; widen r_bad_grid.")


if __name__ == "__main__":
    main()
