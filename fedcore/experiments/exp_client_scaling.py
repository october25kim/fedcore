"""Task C.4 / FJ: "the price of federation" -- client-scaling of the certificate.

CPU-only synthetic. A FIXED total audit budget of ``N_TOTAL`` labeled
certification points is split equally across ``J`` clients (n_j = N_TOTAL / J).
As J grows, two things tighten the screw on the STRATIFIED certificate
(Theorem 1/1', full simplex): (i) each per-client CP interval is taken at the
Bonferroni level delta/J, and (ii) the per-client accepted count A_j = a * n_j
shrinks toward -- and eventually below -- the Theorem 3 feasibility floor
ln(J/delta)/(-ln(1-alpha)). J=1 is the centralized anchor (a single binomial at
the full delta, no split). The curve therefore reads as the certified-coverage
COST of splitting one fixed audit budget across J clients.

Homogeneous clients (a, r fixed, r < alpha) isolate the federation price from
heterogeneity: any decline here is purely the cost of stratifying a fixed budget.

Output: runs/client_scaling.csv + experiments/fedcore/figs/FJ_client_scaling.{pdf,png}
Run: python experiments/fedcore/exp_client_scaling.py   (CPU, no torch)
"""

from __future__ import annotations

import glob
import os

import numpy as np

from fedcore.certificate import (
    conditional_risk_certificate,
    cp_lower,
    thm2_floor,
)
from fedcore.data.clients import ClientPopulation, draw_counts

ALPHA, DELTA = 0.10, 0.10
A_ACC, R_TRUE = 0.60, 0.05          # per-client accept prob / true accepted-risk (r<alpha)
N_TOTAL = 1200                      # FIXED total audit budget (split across J)
JS = (1, 2, 3, 4, 5, 8, 12, 16, 20)
N_MC = 400                          # Monte-Carlo resamples per J
LAMBDA = "simplex"                  # full-simplex worst-group (unconditional main result)


def _coverage_lcb(A, n, delta):
    """Worst-case-over-simplex accepted coverage LCB = min_j U-(A_j,n_j; delta/2J)."""
    J = len(A)
    eps = delta / (2.0 * J)
    return float(min(cp_lower(int(A[j]), int(n[j]), eps) for j in range(J)))


def certified_coverage(J, rng):
    """Mean CertifiedCoverage over MC draws at split J (0 for non-certified draws)."""
    n_j = N_TOTAL // J
    pop = ClientPopulation(a=np.full(J, A_ACC), r=np.full(J, R_TRUE))
    n_arr = np.full(J, n_j, dtype=int)
    covs, cert_flags, ucbs = [], [], []
    for _ in range(N_MC):
        A, K = draw_counts(pop, n_arr, rng)
        cert = conditional_risk_certificate(A, K, n_arr, DELTA, Lambda=LAMBDA)
        certified = bool(cert.feasible and cert.U <= ALPHA)
        cov = _coverage_lcb(A, n_arr, DELTA) if certified else 0.0
        covs.append(cov)
        cert_flags.append(certified)
        ucbs.append(cert.U if np.isfinite(cert.U) else np.nan)
    return {
        "J": J, "n_per_client": n_j,
        "expected_A_per_client": round(A_ACC * n_j, 1),
        "theorem3_floor": round(float(thm2_floor(J, DELTA, ALPHA)), 1),
        "cert_coverage_mean": round(float(np.mean(covs)), 4),
        "frac_certified": round(float(np.mean(cert_flags)), 4),
        "cert_ucb_median": round(float(np.nanmedian(ucbs)), 4),
    }


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"'Price of federation': N_total={N_TOTAL} split across J clients, "
          f"a={A_ACC}, r={R_TRUE}, alpha={ALPHA}, delta={DELTA}, Lambda={LAMBDA}\n")
    print(f"{'J':>3} {'n/client':>8} {'E[A]/client':>11} {'T3 floor':>10} "
          f"{'CertCov':>8} {'frac_cert':>9} {'ucb_med':>8}")
    print("-" * 66)
    rows = []
    for J in JS:
        r = certified_coverage(J, rng)
        rows.append(r)
        print(f"{r['J']:>3} {r['n_per_client']:>8} {r['expected_A_per_client']:>11} "
              f"{r['theorem3_floor']:>10} {r['cert_coverage_mean']:>8.3f} "
              f"{r['frac_certified']:>9.3f} {r['cert_ucb_median']:>8.3f}")

    base = "" if glob.glob("runs") else "../../"
    from fedcore.io_utils import atomic_write_csv
    out = base + "runs/client_scaling.csv"
    atomic_write_csv(out, list(rows[0].keys()), rows)
    print(f"\nsaved {out}")
    _plot(rows, base)


def _plot(rows, base):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipping figure: {exc})")
        return
    J = [r["J"] for r in rows]
    cov = [r["cert_coverage_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(J, cov, "o-", color="#D55E00", ms=6, label="CertifiedCoverage@0.1")
    # annotate the centralized anchor
    ax.scatter([1], [cov[0]], s=110, facecolors="none", edgecolors="#009E73",
               linewidths=2, zorder=5, label="J=1 centralized anchor")
    ax.set_xlabel("number of clients J (fixed total audit budget)")
    ax.set_ylabel(r"CertifiedCoverage@$\alpha=0.1$")
    ax.set_title("The price of federation: splitting a fixed audit budget across J clients")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    stem = base + "experiments/fedcore/figs/FJ_client_scaling"
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=130, bbox_inches="tight")
    print(f"saved {stem}.pdf (+png)")


if __name__ == "__main__":
    main()
