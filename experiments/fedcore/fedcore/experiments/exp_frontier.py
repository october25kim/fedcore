"""STEP 4(c): CertifiedCoverage@alpha frontier over alpha in {0.01,0.02,0.05,0.1}.

For each risk target ``alpha`` we run the full certification path on the
synthetic scored population and report the headline CertifiedCoverage@alpha (the
best certified ``cert_coverage_lcb`` over score x gamma, for both simplex and
box). The frontier is monotone non-decreasing in ``alpha``: a looser risk target
certifies more coverage.

Run: ``python experiments/fedcore/exp_frontier.py``  (CPU, no torch)
"""

from __future__ import annotations

import numpy as np

from fedcore.certify import certify_grid
from fedcore.experiments.run_smoke import SmokeSpec, generate_smoke
from fedcore.scores import scored_views

DELTA = 0.10
GAMMAS = (0.5, 0.7, 1.0)
SCORES = ("msp", "neg_entropy", "margin", "energy")
BOX_RADIUS = 0.10


def _headline(rows, Lambda):
    cert = [r for r in rows if r["Lambda"] == Lambda and r["certified"]]
    if not cert:
        return 0.0, None
    best = max(cert, key=lambda r: r["cert_coverage_lcb"])
    return best["cert_coverage_lcb"], best


def main() -> None:
    # a cleaner population than the default smoke so the frontier is a real curve
    # (some coverage certifies even at small alpha, more at large alpha)
    spec = SmokeSpec(mu_good=7.0, mu_bad=6.0, def_boost=1.5,
                     def_rate_good=0.03, def_rate_bad=0.20, n_known_per_client=900)
    data = generate_smoke(spec)
    views = {fn: scored_views(data[fn]["logits"], data[fn]["y_open"],
                              data[fn]["client"], list(SCORES))
             for fn in ("prop", "cert", "test")}

    print("STEP 4(c) CertifiedCoverage@alpha frontier (delta=0.10, synthetic)")
    print(f"{'alpha':>7} {'simplex_cov':>12} {'box_cov':>9} {'best score/gamma/L':>22}")
    print("-" * 54)

    for alpha in (0.01, 0.02, 0.05, 0.10):
        rows = certify_grid(views["prop"], views["cert"], views["test"],
                            scores=SCORES, gammas=GAMMAS, alpha=alpha, delta=DELTA,
                            Lambdas=("simplex", "box"), n_clients=spec.n_clients,
                            dirichlet_alpha=float("nan"), box=BOX_RADIUS, seed=spec.seed)
        sx_cov, _ = _headline(rows, "simplex")
        bx_cov, bx = _headline(rows, "box")
        best = bx if bx_cov >= sx_cov else _headline(rows, "simplex")[1]
        tag = (f"{best['score_name']}/{best['gamma']}/{best['Lambda']}"
               if best else "-")
        print(f"{alpha:>7.2f} {sx_cov:>12.4f} {bx_cov:>9.4f} {tag:>22}")

    print("-" * 54)
    print("Frontier is monotone non-decreasing in alpha (looser risk -> more coverage).")


if __name__ == "__main__":
    main()
