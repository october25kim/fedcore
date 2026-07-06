"""Fig. 1 — Non-reducibility (pooling collapse).

x = deployment mixture shifted toward the high-risk client;
y = empirical coverage of the certificate, P(R_sel(lambda*) <= U).
Curves: naive pooled CP (collapses), stratified simplex (Thm 1), box-Lambda (Thm 1').
Saves PDF + PNG to this directory.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from certificates import (pooled_cp, conditional_risk_certificate,
                          true_selective_risk)
from clients import heterogeneous_population, draw_counts

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def coverage_curve(delta=0.10, n_per_client=400, T=700, seed=0):
    rng = np.random.default_rng(seed)
    pop = heterogeneous_population()            # 4 good + 1 bad
    J = pop.J
    n = np.full(J, n_per_client)
    bad = J - 1
    base = (n * pop.a) / (n * pop.a).sum()      # matched mixture
    # box around the (known) data fractions
    box = (np.clip(base - 0.10, 0, 1), np.clip(base + 0.10, 0, 1))
    # shift parameter s in [0,1]: lambda = (1-s)*matched + s*e_bad
    e_bad = np.zeros(J); e_bad[bad] = 1.0
    shifts = np.linspace(0.0, 1.0, 11)
    cov_pool, cov_strat, cov_box, Rtrue = [], [], [], []
    for s in shifts:
        lam = (1 - s) * base + s * e_bad
        lam = lam / lam.sum()
        Rt = true_selective_risk(pop.a, pop.r, lam)
        cp = cs = cb = 0
        for _ in range(T):
            A, K = draw_counts(pop, n, rng)
            up = pooled_cp(A, K, delta)
            us = conditional_risk_certificate(A, K, n, delta, "simplex").U
            ub = conditional_risk_certificate(A, K, n, delta, "box",
                                              box=box, n_lam_samples=80, rng=rng).U
            cp += Rt <= up; cs += Rt <= us; cb += Rt <= ub
        cov_pool.append(cp / T); cov_strat.append(cs / T); cov_box.append(cb / T)
        Rtrue.append(Rt)
    return shifts, np.array(cov_pool), np.array(cov_strat), np.array(cov_box), delta


def main():
    s, cp, cs, cb, delta = coverage_curve()
    plt.figure(figsize=(5.2, 3.4))
    plt.axhline(1 - delta, ls="--", c="0.4", lw=1, label=f"target $1-\\delta$={1-delta:.2f}")
    plt.plot(s, cp, "-o", ms=4, c="#c0392b", label="naive pooled CP")
    plt.plot(s, cs, "-s", ms=4, c="#27ae60", label="stratified simplex (Thm 1)")
    plt.plot(s, cb, "-^", ms=4, c="#2980b9", label="box-$\\Lambda$ (Thm 1$'$)")
    plt.xlabel("deployment mixture shift toward high-risk client")
    plt.ylabel(r"empirical coverage  $\Pr(R_{\mathrm{sel}}\leq \bar U)$")
    plt.ylim(-0.03, 1.05)
    plt.title("Non-reducibility: pooling collapses, stratified holds")
    plt.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    here = os.path.dirname(os.path.abspath(__file__))
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(here, f"fig1_pooling_collapse.{ext}"), dpi=200)
    print("saved fig1_pooling_collapse.pdf/.png")
    print("shift  pooled  stratified  box")
    for i in range(len(s)):
        print(f"{s[i]:.2f}  {cp[i]:.3f}   {cs[i]:.3f}     {cb[i]:.3f}")


if __name__ == "__main__":
    main()
