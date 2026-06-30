"""Pooling-fail ablation: the non-reducibility of Theorem 1.

Four low-risk clients + one high-risk client. We measure the empirical coverage
of the true selective risk ``R_sel(lambda*)`` (i.e. P(certificate >= truth))
under four deployment mixtures, for three certificates:

* ``pooled``            -- Proposition 3 (pooled CP).
* ``conditional simplex`` -- Theorem 1 (full-simplex, worst client).
* ``conditional box``   -- Theorem 1' (mixtures in a box around uniform).

Acceptance gate (delta=0.1): pooled coverage ~1.0 at 'matched' but COLLAPSES to
~0.0 under a deployment shift toward the high-risk client; conditional simplex
stays ~1.0 for EVERY mixture; box is valid for in-box mixtures and tighter than
simplex. We also confirm the conditional simplex bound is TIGHTER (smaller
median U) than the Appendix-C mass-ratio baseline.

Run: ``python experiments/fedcore/exp_pooling_fail.py``  (CPU, no torch)
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from certificates import (
    conditional_risk_certificate,
    pooled_cp,
    stratified_certificate,
    true_selective_risk,
)
from clients import draw_counts, heterogeneous_population

DELTA = 0.1
N_TRIALS = 1500
N_PER_CLIENT = 300
BOX_RADIUS = 0.10
BOX_SAMPLES = 64
SEED = 0


def _mixtures(n: np.ndarray, J: int) -> Dict[str, np.ndarray]:
    """Deployment mixtures over clients (bad client is the last index)."""
    uniform = np.full(J, 1.0 / J)
    matched = n / n.sum()
    shift = np.full(J, 0.4 / (J - 1))
    shift[-1] = 0.6
    allbad = np.zeros(J)
    allbad[-1] = 1.0
    return {"matched": matched, "uniform": uniform, "shift->bad": shift, "all->bad": allbad}


def main() -> None:
    pop = heterogeneous_population()  # a=[.7,.7,.7,.7,.5], r=[.02,.02,.02,.02,.3]
    J = pop.J
    n = np.full(J, N_PER_CLIENT)
    mixtures = _mixtures(n, J)

    print(f"Pooling-fail ablation (delta={DELTA}, J={J}, n_j={N_PER_CLIENT})")
    print(f"population a={pop.a.tolist()}  r={pop.r.tolist()}")
    print(f"\n{'mixture':>12} {'R_sel*':>8} {'cov_pooled':>11} {'cov_cond_sx':>12} {'cov_cond_box':>13}")
    print("-" * 60)

    for name, lam in mixtures.items():
        rng = np.random.default_rng(SEED)
        R_true = true_selective_risk(pop.a, pop.r, lam)
        cov_p = cov_s = cov_b = 0
        for t in range(N_TRIALS):
            A, K = draw_counts(pop, n, rng)
            up = pooled_cp(A, K, DELTA)
            us = conditional_risk_certificate(A, K, n, DELTA, Lambda="simplex").U
            ub = conditional_risk_certificate(
                A, K, n, DELTA, Lambda="box", box=BOX_RADIUS,
                n_lam_samples=BOX_SAMPLES, seed=t,
            ).U
            cov_p += up >= R_true
            cov_s += us >= R_true
            cov_b += ub >= R_true
        print(
            f"{name:>12} {R_true:>8.4f} {cov_p / N_TRIALS:>11.3f} "
            f"{cov_s / N_TRIALS:>12.3f} {cov_b / N_TRIALS:>13.3f}"
        )

    # --- tightness: conditional simplex vs mass-ratio (Appendix C) -----------
    rng = np.random.default_rng(SEED)
    cond_U, mass_U = [], []
    for _ in range(N_TRIALS):
        A, K = draw_counts(pop, n, rng)
        cond_U.append(conditional_risk_certificate(A, K, n, DELTA, Lambda="simplex").U)
        mass_U.append(stratified_certificate(A, K, n, DELTA, Lambda="simplex").U)
    print("\nTightness (simplex), median U over trials:")
    print(f"  conditional (Thm 1) : {np.median(cond_U):.4f}  (expect ~0.37, TIGHTER)")
    print(f"  mass-ratio  (App C) : {np.median(mass_U):.4f}  (expect ~0.45)")
    tighter = np.median(cond_U) < np.median(mass_U)
    print(f"  conditional tighter than mass-ratio: {tighter}")


if __name__ == "__main__":
    main()
