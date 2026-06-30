"""STEP 4(h): Utilization A -- automation rate at guaranteed risk.

The practical payoff of Fed-CORE: at a guaranteed selective risk ``alpha``, what
fraction of predictions can be AUTOMATED (accepted without human review)? That
automation rate IS CertifiedCoverage@alpha. We report it per score together with
the held-out TEST risk to confirm the guarantee is honored (test_risk <= alpha).

Run: ``python experiments/fedcore/exp_utilization.py``  (CPU, no torch)
"""

from __future__ import annotations

import numpy as np

from certify import certify_grid
from run_smoke import SmokeSpec, generate_smoke
from scores import scored_views

ALPHA = 0.10
DELTA = 0.10
GAMMAS = (0.5, 0.7, 1.0)
SCORES = ("msp", "neg_entropy", "margin", "energy")
BOX_RADIUS = 0.10


def main() -> None:
    spec = SmokeSpec()
    data = generate_smoke(spec)
    views = {fn: scored_views(data[fn]["logits"], data[fn]["y_open"],
                              data[fn]["client"], list(SCORES))
             for fn in ("prop", "cert", "test")}
    rows = certify_grid(views["prop"], views["cert"], views["test"],
                        scores=SCORES, gammas=GAMMAS, alpha=ALPHA, delta=DELTA,
                        Lambdas=("simplex", "box"), n_clients=spec.n_clients,
                        dirichlet_alpha=float("nan"), box=BOX_RADIUS, seed=spec.seed)

    print(f"STEP 4(h) Utilization A: automation rate = CertifiedCoverage@alpha "
          f"(alpha={ALPHA}, delta={DELTA})")

    cert = [r for r in rows if r["certified"]]
    if not cert:
        print("No combo certified -> automation rate 0 at this alpha.")
        return

    best = max(cert, key=lambda r: r["cert_coverage_lcb"])
    print(f"\nHEADLINE automation rate (best certified): "
          f"{best['cert_coverage_lcb']:.4f}")
    print(f"  via score={best['score_name']} gamma={best['gamma']} "
          f"Lambda={best['Lambda']}  (TEST risk={best['test_risk']:.4f} <= alpha)")

    print(f"\n{'score':>12} {'Lambda':>8} {'automation':>11} {'guaranteed(test<=a)':>20}")
    print("-" * 54)
    for s in SCORES:
        for L in ("simplex", "box"):
            sub = [r for r in rows if r["score_name"] == s and r["Lambda"] == L and r["certified"]]
            if sub:
                b = max(sub, key=lambda r: r["cert_coverage_lcb"])
                print(f"{s:>12} {L:>8} {b['cert_coverage_lcb']:>11.4f} "
                      f"{str(b['test_risk'] <= ALPHA):>20}")


if __name__ == "__main__":
    main()
