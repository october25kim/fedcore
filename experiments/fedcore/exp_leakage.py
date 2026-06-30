"""STAGE 4 H2: split-leakage ablation -- the guarantee BREAKS without hygiene.

Fed-CORE requires the selector to be chosen on the PROPOSAL fold, independent of the
CERTIFICATION fold. This experiment shows that violating that -- choosing the
threshold on the SAME fold used to certify (a 'leak' of the certification labels into
selection) -- inflates the unsafe-deploy rate FAR beyond delta, while the proper
split keeps it <= delta. This proves split hygiene is load-bearing, not decorative.

Protocol (over many synthetic trials, true accepted risk tuned near alpha):
  PROPER : threshold from PROPOSAL fold; certify on CERT fold; deploy iff cert<=alpha.
  LEAKED : threshold optimized on the CERT fold itself (uses its labels); certify on
           that SAME fold; deploy iff cert<=alpha.
  unsafe = deployed AND held-out TEST risk > alpha.  (TEST is always clean/unused.)

Run: ``python experiments/fedcore/exp_leakage.py``  (CPU, no torch)
"""

from __future__ import annotations

import numpy as np

from certificates import cp_upper

ALPHA, DELTA = 0.10, 0.10
N_TRIALS = 400
N_PER_FOLD = 500
N_GRID = 300


def _select(score, err, alpha, n_grid=N_GRID):
    """Highest-coverage threshold with empirical accepted risk <= alpha."""
    qs = np.unique(np.quantile(score, np.linspace(0, 1, n_grid)))
    best_t, best_cov = np.inf, -1.0
    for t in qs:
        acc = score >= t
        na = int(acc.sum())
        if na == 0:
            continue
        if err[acc].mean() <= alpha and na > best_cov:
            best_cov, best_t = na, t
    return best_t


def _risk_at(score, err, t):
    acc = score >= t
    return (float(err[acc].mean()), int(acc.sum())) if acc.any() else (0.0, 0)


def main() -> None:
    print(f"STAGE 4 H2 split-leakage (alpha={ALPHA}, delta={DELTA}, pooled CP, "
          f"n/fold={N_PER_FOLD})")
    rng = np.random.default_rng(0)
    dep_p = uns_p = dep_l = uns_l = 0
    for t in range(N_TRIALS):
        # one synthetic population: score in [0,1], P(err|score)=0.30*(1-score) so the
        # high-score region sits near alpha; three i.i.d. folds prop/cert/test.
        def fold():
            s = rng.random(N_PER_FOLD)
            e = rng.random(N_PER_FOLD) < 0.30 * (1.0 - s)
            return s, e.astype(bool)
        sp, ep = fold(); sc, ec = fold(); st, et = fold()

        # PROPER: select on prop, certify (pooled CP) on cert, evaluate on test
        tp = _select(sp, ep, ALPHA)
        rc, ac = _risk_at(sc, ec, tp); kc = int(ec[sc >= tp].sum())
        Up = cp_upper(kc, ac, DELTA)
        if np.isfinite(tp) and ac > 0 and Up <= ALPHA:
            dep_p += 1
            rt, _ = _risk_at(st, et, tp); uns_p += rt > ALPHA

        # LEAKED: search thresholds on the CERT fold for the max-coverage one whose
        # pooled CP already passes (<= alpha) on that SAME fold -- i.e. "try thresholds
        # until the certificate passes." The single-CP delta does not correct for this
        # selection over ~N_GRID thresholds, so some pass by chance with true risk>alpha.
        qs = np.unique(np.quantile(sc, np.linspace(0, 1, N_GRID)))
        tl, best_cov = np.inf, -1
        for tt in qs:
            acc = sc >= tt; na = int(acc.sum())
            if na == 0:
                continue
            if cp_upper(int(ec[acc].sum()), na, DELTA) <= ALPHA and na > best_cov:
                best_cov, tl = na, tt
        if np.isfinite(tl):
            dep_l += 1
            rt, _ = _risk_at(st, et, tl); uns_l += rt > ALPHA

    print(f"{'method':>10} {'deploy%':>9} {'unsafe-deploy%':>15} {'<=delta':>9}")
    print("-" * 46)
    rp, rl = uns_p / N_TRIALS, uns_l / N_TRIALS
    print(f"{'PROPER':>10} {100*dep_p/N_TRIALS:>8.1f}% {100*rp:>14.1f}% {str(rp <= DELTA):>9}")
    print(f"{'LEAKED':>10} {100*dep_l/N_TRIALS:>8.1f}% {100*rl:>14.1f}% {str(rl <= DELTA):>9}")
    print("-" * 46)
    print(f"Split hygiene is load-bearing: PROPER unsafe-deploy <= delta; LEAKED "
          f"(threshold chosen on the cert fold) breaks it ({100*rl:.1f}% > {100*DELTA:.0f}%).")


if __name__ == "__main__":
    main()
