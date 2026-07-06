"""Lemma L verification.

CLAIM (Lemma L, used by Theorem 3): the binomial Clopper-Pearson UPPER limit,
applied to a Poisson-binomial accepted-error count (a sum of independent but
NON-identical Bernoulli accepted-error indicators), is a valid 1 - delta upper
confidence bound for the mean accepted risk
        rbar = (1 / A) * sum_i r_i .

If TRUE: pooling is internally valid for the calibration-weighted mean, so
Theorem 3 (the tighter pooled certificate) stands -- *under matched-lambda*.
If FALSE: we must report a counterexample and drop / repair Theorem 3.

We test across heterogeneity profiles that share a fixed mean ``rbar``, including
the adversarial two-point profiles that are the extreme points for
Poisson-binomial dispersion (Hoeffding, 1956). The binomial (equal r_i) is the
exact-CP reference and should sit at coverage >= 1 - delta by construction.
"""
from __future__ import annotations

import numpy as np

from certificates import cp_upper


def coverage_for_profile(
    r_vec: np.ndarray,
    delta: float,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Empirical P(rbar <= cp_upper(K, A, delta)) and the median certificate width.

    ``r_vec`` is the length-A vector of per-accepted-point error probabilities.
    Vectorized: draw all trials at once, then map each distinct K -> cp_upper(K)
    via a precomputed lookup table (cp_upper depends only on K for fixed A, delta).
    """
    A = len(r_vec)
    rbar = float(r_vec.mean())
    # Poisson-binomial draws: K_t = sum_i 1{u_ti < r_i}
    draws = (rng.random((n_trials, A)) < r_vec).sum(axis=1)  # shape (n_trials,)
    lut = np.array([cp_upper(k, A, delta) for k in range(A + 1)])
    U = lut[draws]
    coverage = float(np.mean(rbar <= U))
    return coverage, float(np.median(U))


def make_profiles(A: int, rbar: float) -> dict[str, np.ndarray]:
    """Several length-A error-probability profiles, all with mean == rbar."""
    profiles: dict[str, np.ndarray] = {}

    # 1) homogeneous (this is exactly Binomial(A, rbar) -> exact CP reference)
    profiles["homogeneous"] = np.full(A, rbar)

    # 2) mild heterogeneity: r_i spread uniformly in [rbar/2, 3rbar/2]
    lo, hi = rbar * 0.5, rbar * 1.5
    mild = np.linspace(lo, hi, A)
    mild *= rbar / mild.mean()
    profiles["mild_spread"] = np.clip(mild, 0, 1)

    # 3) strong heterogeneity: half near 0, half elevated, mean preserved
    strong = np.where(np.arange(A) < A // 2, rbar * 0.1, 0.0)
    rem = rbar * A - strong.sum()
    strong[A // 2:] = rem / (A - A // 2)
    profiles["strong_bimodal"] = np.clip(strong, 0, 1)

    # 4) adversarial two-point: a fraction rho of points at prob p, rest at 0.
    #    Sweep rho so that rho*p = rbar; p in {0.25, 0.5, 0.75}.
    for p in (0.25, 0.5, 0.75):
        rho = rbar / p
        k_hot = int(round(rho * A))
        v = np.zeros(A)
        if k_hot > 0:
            v[:k_hot] = p
            # fix mean exactly
            v[:k_hot] *= rbar * A / v.sum()
            v = np.clip(v, 0, 1)
        profiles[f"two_point_p{p}"] = v

    return profiles


def main() -> None:
    rng = np.random.default_rng(20260626)
    delta = 0.10
    n_trials = 8000
    target = 1 - delta

    print(f"Lemma L verification | delta={delta}  target coverage >= {target:.3f}")
    print(f"(coverage below {target:.3f} would REFUTE Lemma L)\n")

    worst = 1.0
    worst_where = ""
    header = f"{'A':>5} {'rbar':>6} {'profile':>16} {'coverage':>9} {'med_U':>7}"
    for A in (100, 300, 1000):
        for rbar in (0.05, 0.02):
            print(header)
            print("-" * len(header))
            for name, r_vec in make_profiles(A, rbar).items():
                cov, width = coverage_for_profile(r_vec, delta, n_trials, rng)
                flag = "  <-- REFUTES" if cov < target - 0.005 else ""
                print(f"{A:>5} {rbar:>6.3f} {name:>16} {cov:>9.4f} {width:>7.3f}{flag}")
                if cov < worst:
                    worst, worst_where = cov, f"A={A}, rbar={rbar}, {name}"
            print()

    print("=" * 60)
    print(f"WORST observed coverage = {worst:.4f}  at  {worst_where}")
    verdict = "SUPPORTED" if worst >= target - 0.005 else "REFUTED"
    print(f"Lemma L verdict (numerical): {verdict}")
    print("Interpretation: binomial CP is conservative for the Poisson-binomial")
    print("mean under heterogeneity (binomial maximizes dispersion at fixed mean;")
    print("cf. Hoeffding 1956). Formal proof to accompany Theorem 3.")


if __name__ == "__main__":
    main()
