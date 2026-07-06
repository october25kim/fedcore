"""Pooling-fail ablation -- the empirical core of Theorem 1's non-reducibility.

CLAIM: Under client heterogeneity, naively pooling the federated accepted set
into a single binomial and applying Clopper-Pearson (the obvious "just reuse the
centralized certificate" approach) FAILS to control the *deployment* accepted
selective risk R_sel(lambda*) whenever the deployment mixture lambda* overweights
a high-error client. The stratified worst-case-mixture certificate (Theorem 1)
remains valid (>= 1 - delta) for every lambda* in Lambda simultaneously.

This is exactly why the certificate cannot be reduced to "centralized CP on the
pooled data": pooling is only valid under matched-lambda partial exchangeability.

Outputs a table of empirical coverage of R_sel(lambda*) for:
  * naive pooled CP            (Theorem-3 formula, mis-applied off-matched-lambda)
  * stratified, Lambda=simplex (Theorem 1, full robustness)
  * stratified, Lambda=box     (Theorem 1, tightened around known client sizes)
across several deployment-mixture shifts, plus the matched-lambda control where
pooling is legitimate.
"""
from __future__ import annotations

import csv
import numpy as np

from certificates import pooled_cp, stratified_certificate, true_selective_risk
from clients import draw_counts, heterogeneous_population


def matched_lambda(n: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Deployment weights that match the calibration accepted-count proportions.

    Under these weights the pooled certificate is the legitimate Theorem-3 bound.
    """
    w = n * a
    return w / w.sum()


def run(
    delta: float = 0.10,
    n_per_client: int = 400,
    n_trials: int = 2500,
    seed: int = 20260626,
    n_box_samples: int = 300,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    pop = heterogeneous_population()          # 4 good clients + 1 bad (high-risk)
    J = pop.J
    n = np.full(J, n_per_client)

    # deployment mixtures to evaluate (lambda*)
    bad = J - 1
    e_bad = np.zeros(J); e_bad[bad] = 1.0
    shift_moderate = np.full(J, 0.1); shift_moderate[bad] = 1.0 - 0.1 * (J - 1)
    mixtures = {
        "matched (control)": matched_lambda(n, pop.a),
        "uniform":           np.full(J, 1.0 / J),
        "shift->bad(0.6)":   shift_moderate,
        "all->bad":          e_bad,
    }

    # box Lambda around the (assumed known) client data fractions +/- 0.15
    base = matched_lambda(n, pop.a)
    box = (np.clip(base - 0.15, 0, 1), np.clip(base + 0.15, 0, 1))

    rows: list[dict] = []
    for name, lam in mixtures.items():
        R_true = true_selective_risk(pop.a, pop.r, lam)
        cov_pool = cov_simplex = cov_box = 0
        Up, Us, Ub = [], [], []
        for _ in range(n_trials):
            A, K = draw_counts(pop, n, rng)
            U_pool = pooled_cp(A, K, delta)
            U_simplex = stratified_certificate(A, K, n, delta, "simplex").U
            U_box = stratified_certificate(
                A, K, n, delta, "box", box=box, n_box_samples=n_box_samples, rng=rng
            ).U
            cov_pool += R_true <= U_pool
            cov_simplex += R_true <= U_simplex
            cov_box += R_true <= U_box
            Up.append(U_pool); Us.append(U_simplex); Ub.append(U_box)
        rows.append({
            "mixture": name,
            "R_true": round(R_true, 4),
            "cov_pooled": round(cov_pool / n_trials, 4),
            "cov_stratified_simplex": round(cov_simplex / n_trials, 4),
            "cov_stratified_box": round(cov_box / n_trials, 4),
            "medU_pooled": round(float(np.median(Up)), 4),
            "medU_simplex": round(float(np.median(Us)), 4),
            "medU_box": round(float(np.median(Ub)), 4),
        })
    return rows


def main() -> None:
    delta = 0.10
    target = 1 - delta
    rows = run(delta=delta)

    cols = ["mixture", "R_true", "cov_pooled", "cov_stratified_simplex",
            "cov_stratified_box", "medU_pooled", "medU_simplex", "medU_box"]
    print(f"Pooling-fail ablation | delta={delta}  target coverage >= {target:.2f}")
    print("(cov_* = empirical P(R_sel(lambda*) <= certificate); below target = VIOLATION)\n")
    widths = {c: max(len(c), 10) for c in cols}
    print("  ".join(c.rjust(widths[c]) for c in cols))
    print("-" * (sum(widths.values()) + 2 * (len(cols) - 1)))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            s = f"{v}"
            if c == "cov_pooled" and isinstance(v, float) and v < target - 0.005:
                s += "!"
            cells.append(s.rjust(widths[c]))
        print("  ".join(cells))

    out = "lemma_pooling_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {out}")
    print("\nReading: pooled CP holds only for the matched control; it VIOLATES")
    print("coverage as lambda* shifts toward the high-risk client. The stratified")
    print("simplex certificate holds for every mixture (price: larger U); the box")
    print("certificate recovers much of the tightness when client sizes are known.")


if __name__ == "__main__":
    main()
