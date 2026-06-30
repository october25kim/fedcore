"""Proposition 4 demonstration: certified vs naive federated self-training.

Synthetic CPU demonstration of the central claim: gating pseudo-label admission
on the Fed-CORE certificate keeps per-round contamination certified ``<= alpha``,
so accuracy improves safely; the naive (uncertified) loop admits the high-risk
client's confidently-wrong pseudo-labels and DIVERGES under non-IID corruption.

Three trajectories are compared under the same corruption:
  * ``certified`` -- admit only if the conditional certificate (level delta/T) <= alpha;
  * ``naive``     -- admit whatever the threshold accepts;
  * ``none``      -- never self-train (baseline).

Acceptance expectation: ``certified`` ends safe and >= its start; ``naive`` ends
below its start (diverges); ``none`` stays flat.

Run: ``python experiments/fedcore/exp_self_training.py``  (CPU, no torch)
"""

from __future__ import annotations

import numpy as np

from fedcore.experiments.self_training import simulate_self_training

ALPHA = 0.10
DELTA = 0.10
N_ROUNDS = 12
SEED = 0


def _run(mode: str):
    return simulate_self_training(
        mode=mode, n_clients=5, n_rounds=N_ROUNDS, audit_n_per_client=300,
        corruption_bad=0.45, corruption_good=0.0, alpha=ALPHA, delta=DELTA,
        gamma=0.7, init_acc=0.85, eta=0.6, beta=2.5, Lambda="simplex", seed=SEED,
    )


def main() -> None:
    print(f"Proposition 4: certified federated self-training "
          f"(alpha={ALPHA}, delta={DELTA}, T={N_ROUNDS} audit folds)")

    traj = {m: _run(m) for m in ("certified", "naive", "none")}

    print(f"\n{'round':>5} | {'certified acc':>13} {'cert U':>7} {'adm/J':>5} | "
          f"{'naive acc':>10} {'contam':>7} {'adm/J':>5} | {'none acc':>9}")
    print("-" * 78)
    for t in range(N_ROUNDS):
        c, nv, no = traj["certified"][t], traj["naive"][t], traj["none"][t]
        print(
            f"{t:>5} | {c['acc']:>13.4f} {c['cert_risk_ucb']:>7.3f} "
            f"{c['n_admit_clients']:>5} | {nv['acc']:>10.4f} {nv['true_contam']:>7.3f} "
            f"{nv['n_admit_clients']:>5} | {no['acc']:>9.4f}"
        )

    end = {m: traj[m][-1]["acc"] for m in traj}
    start = traj["none"][0]["acc"]
    print("-" * 78)
    print(f"start acc={start:.4f}")
    print(f"final acc: certified={end['certified']:.4f}  "
          f"naive={end['naive']:.4f}  none={end['none']:.4f}")

    cert_safe = end["certified"] >= start - 1e-9
    naive_diverged = end["naive"] < start
    print(f"\ncertified safe & improved (>= start): {cert_safe}")
    print(f"naive diverged (< start):            {naive_diverged}")
    verdict = "PASS" if (cert_safe and naive_diverged) else "CHECK"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
