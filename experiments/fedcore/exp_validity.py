"""STEP 4(a): validity plot -- empirical P(R_sel <= Ubar) vs heterogeneity.

The conditional simplex certificate ``Ubar = max_j rbar_j`` upper-bounds
``sup_{lambda} R_sel(lambda) = max_j r_j``. Validity requires the empirical
coverage ``P(Ubar >= max_j r_j) >= 1 - delta`` for EVERY heterogeneity level.
We sweep the high-risk client's error rate (heterogeneity) and confirm coverage
never drops below ``1 - delta``.

Run: ``python experiments/fedcore/exp_validity.py``  (CPU, no torch)
"""

from __future__ import annotations

import numpy as np

from certificates import conditional_risk_certificate
from clients import ClientPopulation, draw_counts

DELTA = 0.10
N_TRIALS = 2000
N_CLIENTS = 5
N_PER_CLIENT = 300
SEED = 0


def main() -> None:
    print(f"STEP 4(a) validity: P(Ubar >= sup_lam R_sel) vs heterogeneity "
          f"(delta={DELTA}, target >= {1 - DELTA:.2f})")
    print(f"{'r_bad':>7} {'sup R_sel':>10} {'coverage':>10} {'>=1-delta':>10}")
    print("-" * 42)

    a = np.array([0.7] * (N_CLIENTS - 1) + [0.5])
    n = np.full(N_CLIENTS, N_PER_CLIENT)
    worst = 1.0
    for r_bad in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        r = np.array([0.02] * (N_CLIENTS - 1) + [r_bad])
        pop = ClientPopulation(a=a, r=r)
        sup_R = float(np.max(r))  # simplex worst-case target
        rng = np.random.default_rng(SEED)
        cov = 0
        for _ in range(N_TRIALS):
            A, K = draw_counts(pop, n, rng)
            U = conditional_risk_certificate(A, K, n, DELTA, Lambda="simplex").U
            cov += U >= sup_R
        cov /= N_TRIALS
        worst = min(worst, cov)
        ok = "yes" if cov >= 1 - DELTA else "NO"
        print(f"{r_bad:>7.2f} {sup_R:>10.3f} {cov:>10.3f} {ok:>10}")

    print("-" * 42)
    print(f"worst-case coverage across heterogeneity: {worst:.3f} "
          f"({'PASS' if worst >= 1 - DELTA else 'FAIL'}; must stay >= {1 - DELTA:.2f})")


if __name__ == "__main__":
    main()
