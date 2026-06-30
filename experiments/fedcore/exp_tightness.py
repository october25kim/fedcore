"""STEP 4(b): tightness -- conditional vs mass-ratio vs box-Lambda vs pooled.

At a fixed heterogeneity we compare the median certified upper bound ``U`` of:

* ``pooled``       -- Proposition 3 (TIGHTEST but INVALID off matched-mixture);
* ``conditional box`` -- Theorem 1' (valid for in-box mixtures, tighter than simplex);
* ``conditional simplex`` -- Theorem 1 (valid for all mixtures);
* ``mass-ratio``   -- Appendix-C baseline (valid, loosest).

The ordering among the VALID certificates should be box <= simplex <= mass-ratio.
Pooled is reported for reference but its validity collapses under shift
(see ``exp_pooling_fail.py``). We also sweep heterogeneity to show conditional
stays uniformly tighter than mass-ratio.

Run: ``python experiments/fedcore/exp_tightness.py``  (CPU, no torch)
"""

from __future__ import annotations

import numpy as np

from certificates import (
    conditional_risk_certificate,
    pooled_cp,
    stratified_certificate,
)
from clients import ClientPopulation, draw_counts

DELTA = 0.10
N_TRIALS = 1500
N_CLIENTS = 5
N_PER_CLIENT = 300
BOX_RADIUS = 0.10
BOX_SAMPLES = 64
SEED = 0


def main() -> None:
    print(f"STEP 4(b) tightness: median certified U by method (delta={DELTA})")
    print(f"{'r_bad':>7} {'pooled*':>9} {'cond_box':>9} {'cond_sx':>9} {'mass_ratio':>11}")
    print("-" * 50)

    a = np.array([0.7] * (N_CLIENTS - 1) + [0.5])
    n = np.full(N_CLIENTS, N_PER_CLIENT)
    for r_bad in (0.10, 0.20, 0.30, 0.40):
        r = np.array([0.02] * (N_CLIENTS - 1) + [r_bad])
        pop = ClientPopulation(a=a, r=r)
        rng = np.random.default_rng(SEED)
        up, ub, us, um = [], [], [], []
        for t in range(N_TRIALS):
            A, K = draw_counts(pop, n, rng)
            up.append(pooled_cp(A, K, DELTA))
            ub.append(conditional_risk_certificate(
                A, K, n, DELTA, Lambda="box", box=BOX_RADIUS,
                n_lam_samples=BOX_SAMPLES, seed=t).U)
            us.append(conditional_risk_certificate(A, K, n, DELTA, Lambda="simplex").U)
            um.append(stratified_certificate(A, K, n, DELTA, Lambda="simplex").U)
        print(f"{r_bad:>7.2f} {np.median(up):>9.3f} {np.median(ub):>9.3f} "
              f"{np.median(us):>9.3f} {np.median(um):>11.3f}")

    print("-" * 50)
    print("* pooled is TIGHTEST but INVALID off the matched mixture "
          "(see exp_pooling_fail). Among valid: box <= simplex <= mass-ratio.")


if __name__ == "__main__":
    main()
