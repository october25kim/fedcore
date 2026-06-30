"""STEP 4(d): heterogeneity sweep -> certified-coverage-collapse curve (Theorem 2).

Theorem 2 says a client can be certified only if its OBSERVED accepted count
``A_j >= ln(J/delta) / (-ln(1-alpha))``. As heterogeneity grows, the smallest
high-risk client shrinks; once its accepted count drops below the Theorem-2
threshold, the certificate becomes infeasible and certified coverage COLLAPSES.

We shrink the high-risk client's total size ``n_bad`` (a proxy for increasing
non-IID skew / smaller dirichlet_alpha) and report the certified selective-risk
feasibility and the achieved certified coverage, with the Theorem-2 threshold
marked.

Run: ``python experiments/fedcore/exp_hetero_collapse.py``  (CPU, no torch)
"""

from __future__ import annotations

import numpy as np

from fedcore.certificate import conditional_risk_certificate, cp_lower
from fedcore.data.clients import ClientPopulation, draw_counts

DELTA = 0.10
ALPHA = 0.10
N_TRIALS = 1500
N_CLIENTS = 5
SEED = 0


def main() -> None:
    J = N_CLIENTS
    thm2 = np.log(J / DELTA) / (-np.log(1 - ALPHA))
    print(f"STEP 4(d) heterogeneity collapse vs Theorem 2 "
          f"(alpha={ALPHA}, delta={DELTA}, J={J})")
    print(f"Theorem 2 min observed accepted count A_j >= {thm2:.1f}")
    # low-risk clients so the bound is feasible when counts suffice
    a = np.array([0.7] * (J - 1) + [0.5])
    r = np.array([0.02] * (J - 1) + [0.05])

    print(f"\n{'n_bad':>7} {'E[A_bad]':>9} {'feasible%':>10} {'cert_cov_lcb':>13} {'certified%':>11}")
    print("-" * 56)
    for n_bad in (400, 200, 100, 50, 25, 12):
        n = np.array([400] * (J - 1) + [n_bad])
        pop = ClientPopulation(a=a, r=r)
        rng = np.random.default_rng(SEED)
        feas = certified = 0
        covs = []
        eps_cov = DELTA / (2 * J)
        for _ in range(N_TRIALS):
            A, K = draw_counts(pop, n, rng)
            cert = conditional_risk_certificate(A, K, n, DELTA, Lambda="simplex")
            feasible = cert.feasible and np.isfinite(cert.U)
            feas += feasible
            is_cert = feasible and cert.U <= ALPHA
            certified += is_cert
            if is_cert:
                alow = np.array([cp_lower(int(A[j]), int(n[j]), eps_cov) for j in range(J)])
                covs.append(float(np.min(alow)))  # simplex coverage LCB
        cov_lcb = np.mean(covs) if covs else 0.0
        print(f"{n_bad:>7} {a[-1]*n_bad:>9.1f} {100*feas/N_TRIALS:>9.1f}% "
              f"{cov_lcb:>13.4f} {100*certified/N_TRIALS:>10.1f}%")

    print("-" * 56)
    print(f"Collapse sets in as E[A_bad] approaches the Theorem-2 floor "
          f"({thm2:.1f}); below it, certified coverage -> 0.")


if __name__ == "__main__":
    main()
