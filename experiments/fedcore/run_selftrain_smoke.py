"""Proposition 4 smoke -- validate the delta/T temporal split itself (CPU, no torch).

Self-training makes the round-t model depend on what was accepted at round t-1, so
a certification fold reused across rounds is NOT independent of (f_t, A_t). Fed-CORE
fixes this by DATA-SPLITTING IN TIME: T disjoint audit folds C^(1..T), certifying
round t only on the FRESH fold C^(t) at level delta/T. The contract is

    P( for all t<=T : R_sel(A_t) <= Ubar^(t) ) >= 1 - delta.

This smoke validates THAT statement directly, with no model: each round draws an
audit fold whose TRUE selective risk is exactly alpha, certifies it via the
conditional certificate, and we estimate the SIMULTANEOUS unsafe rate
    P( exists t : R_sel(A_t) > Ubar^(t) ).
We show it is <= delta WITH the delta/T split, and that using delta every round
(no /T) inflates it ABOVE delta. (Here each audit fold is a single trusted
population, J=1, for which the conditional simplex certificate equals
cp_upper(K, A; delta_round) -- this isolates the TEMPORAL union bound, which is
Proposition 4's content; multi-client validity is covered by exp_pooling_fail /
exp_validity.)

Run: ``python experiments/fedcore/run_selftrain_smoke.py``  (CPU, no torch)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as _beta

ALPHA = 0.10
DELTA = 0.10
T_ROUNDS = 5
A_PER_FOLD = 1000          # accepted points per audit fold (large => CP tight)
N_TRIALS = 20000
SEED = 0


def _cp_upper_vec(K: np.ndarray, A: int, eps: float) -> np.ndarray:
    """Vectorized one-sided binomial Clopper-Pearson upper limit."""
    K = np.asarray(K)
    u = np.ones(len(K), dtype=float)
    ns = K < A
    u[ns] = _beta.ppf(1.0 - eps, K[ns] + 1, A - K[ns])
    return u


def simultaneous_unsafe_rate(delta_round: float, rng) -> tuple:
    """Return (simultaneous unsafe rate, mean per-round failure) over N_TRIALS.

    Each of T rounds: draw K_t ~ Bin(A, alpha) (true R_sel = alpha), certify
    Ubar = cp_upper(K_t, A; delta_round); a round FAILS if R_sel > Ubar, i.e.
    Ubar < alpha. A trial is unsafe if ANY round fails.
    """
    A = A_PER_FOLD
    # draw all rounds x trials at once
    K = rng.binomial(A, ALPHA, size=(N_TRIALS, T_ROUNDS))
    U = _cp_upper_vec(K.ravel(), A, delta_round).reshape(N_TRIALS, T_ROUNDS)
    round_fail = U < ALPHA - 1e-12               # R_sel(=alpha) > Ubar
    any_fail = round_fail.any(axis=1)
    return float(any_fail.mean()), float(round_fail.mean())


def main() -> None:
    print(f"Proposition 4 smoke: simultaneous validity of the temporal split "
          f"(alpha={ALPHA}, delta={DELTA}, T={T_ROUNDS}, A/fold={A_PER_FOLD})")
    print(f"{'scheme':>16} {'delta_round':>12} {'per-round fail':>15} "
          f"{'simultaneous unsafe':>20} {'<=delta':>9}")
    print("-" * 78)

    rng = np.random.default_rng(SEED)
    sim_split, pr_split = simultaneous_unsafe_rate(DELTA / T_ROUNDS, rng)
    rng = np.random.default_rng(SEED)
    sim_nosplit, pr_nosplit = simultaneous_unsafe_rate(DELTA, rng)

    print(f"{'WITH delta/T':>16} {DELTA / T_ROUNDS:>12.4f} {pr_split:>15.4f} "
          f"{sim_split:>20.4f} {str(sim_split <= DELTA):>9}")
    print(f"{'WITHOUT (/T)':>16} {DELTA:>12.4f} {pr_nosplit:>15.4f} "
          f"{sim_nosplit:>20.4f} {str(sim_nosplit <= DELTA):>9}")
    print("-" * 78)
    print(f"union-bound prediction (no split): 1-(1-delta)^T = "
          f"{1 - (1 - DELTA) ** T_ROUNDS:.4f}")
    ok = (sim_split <= DELTA) and (sim_nosplit > DELTA)
    print(f"VERDICT: {'PASS' if ok else 'CHECK'} "
          f"(delta/T keeps simultaneous unsafe <= delta; delta-each-round does not)")


if __name__ == "__main__":
    main()
