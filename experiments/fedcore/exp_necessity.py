"""Necessity-of-certificate experiment.

CLAIM: the obvious non-certified ways to "control" accepted selective risk fail.
We compare, at matched trusted-calibration budget, the **unsafe-deployment rate**
P(deploy | true risk > alpha) of:
  * naive-empirical : deploy iff pooled point estimate K/A <= alpha   (no certificate)
  * pooled-CP       : deploy iff U+(sum K, sum A; delta) <= alpha     (valid, matched-mix)
  * Fed-CORE        : deploy iff conditional certificate (known lambda) <= alpha
A valid method must keep this rate <= delta; a method that ignores finite-sample
uncertainty (naive-empirical) deploys unsafely ~half the time near the boundary.
We also report P(deploy) below the boundary (power) to show the certificate is not
vacuous.
"""
from __future__ import annotations

import numpy as np

from certificates import cp_upper, conditional_risk_certificate, true_selective_risk
from clients import ClientPopulation, draw_counts


def main() -> None:
    rng = np.random.default_rng(20260626)
    J = 5
    n = np.full(J, 300)
    delta, alpha = 0.10, 0.05
    a = np.array([0.70, 0.70, 0.70, 0.70, 0.60])
    lam = np.full(J, 1.0 / J)          # deployment = uniform (matched here)
    T = 2000

    print(f"Necessity of the certificate | J={J} alpha={alpha} delta={delta} "
          f"(valid => P(deploy | R>alpha) <= {delta})")
    hdr = f"{'r_bad':>6} {'R_true':>7} {'region':>7} | {'naive-emp':>10} {'pooled-CP':>10} {'Fed-CORE':>9}"
    print(hdr); print("-" * len(hdr))
    for r_bad in (0.03, 0.05, 0.08, 0.12, 0.19, 0.30):
        r = np.array([0.02, 0.02, 0.02, 0.02, r_bad])
        pop = ClientPopulation(a=a, r=r)
        R_true = true_selective_risk(a, r, lam)
        region = "R<=a" if R_true <= alpha else "R>a"
        d_naive = d_cp = d_fc = 0
        for _ in range(T):
            A, K = draw_counts(pop, n, rng)
            sumA, sumK = int(A.sum()), int(K.sum())
            if sumA > 0 and sumK / sumA <= alpha:
                d_naive += 1
            if cp_upper(sumK, sumA, delta) <= alpha:
                d_cp += 1
            if conditional_risk_certificate(A, K, n, delta, "known", lam=lam).U <= alpha:
                d_fc += 1
        tag = "(VIOLATION rate)" if R_true > alpha else "(power)"
        print(f"{r_bad:>6.2f} {R_true:>7.4f} {region:>7} | "
              f"{d_naive/T:>10.3f} {d_cp/T:>10.3f} {d_fc/T:>9.3f}  {tag}")

    print("\nReading: above the boundary (R>alpha) the entries are UNSAFE-deploy")
    print("rates; naive-empirical stays high (ignores finite-sample noise) while")
    print("pooled-CP and Fed-CORE respect <= delta. Below the boundary they are")
    print("power; the certified methods deploy when confident, not vacuously.")


if __name__ == "__main__":
    main()
