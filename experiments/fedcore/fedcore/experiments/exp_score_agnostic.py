"""STEP 4(e): score-agnostic validity -- 4 scores keep validity, change only coverage.

Fed-CORE certifies any monotone open-set score. We run all four scores
(MSP, neg-entropy, margin, energy) through the identical certification path and
show: (i) VALIDITY holds for every score (whenever a combo is certified, its
held-out TEST risk is <= alpha), while (ii) the achieved certified COVERAGE
differs across scores. The certificate's guarantee is score-agnostic; the score
only affects how much coverage you can certify.

Run: ``python experiments/fedcore/exp_score_agnostic.py``  (CPU, no torch)
"""

from __future__ import annotations

import numpy as np

from fedcore.certify import certify_grid
from fedcore.experiments.run_smoke import SmokeSpec, generate_smoke
from fedcore.scores import scored_views

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

    print(f"STEP 4(e) score-agnostic (alpha={ALPHA}, delta={DELTA})")
    print(f"{'score':>12} {'best_cov':>9} {'test_risk':>10} {'valid(test<=a)':>15} {'#certified':>11}")
    print("-" * 60)

    all_valid = True
    for s in SCORES:
        cert = [r for r in rows if r["score_name"] == s and r["certified"]]
        if cert:
            best = max(cert, key=lambda r: r["cert_coverage_lcb"])
            valid = best["test_risk"] <= ALPHA
            all_valid &= all(r["test_risk"] <= ALPHA for r in cert)
            print(f"{s:>12} {best['cert_coverage_lcb']:>9.4f} {best['test_risk']:>10.4f} "
                  f"{str(valid):>15} {len(cert):>11}")
        else:
            print(f"{s:>12} {'--':>9} {'--':>10} {'(none certified)':>15} {0:>11}")

    print("-" * 60)
    print(f"All certified combos valid on TEST (risk <= alpha): {all_valid}")
    print("Validity is score-agnostic; coverage is what varies across scores.")


if __name__ == "__main__":
    main()
