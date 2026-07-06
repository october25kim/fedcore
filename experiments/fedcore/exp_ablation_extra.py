"""Extra synthetic ablations (reviewer checklist 4/5 + Theorem-2 J-scaling).

All pure-numpy, no retraining. Directly probe the *knobs of the method*:
  (A4) calibration-budget sweep   -> feasibility law in the per-client-count axis
  (A5) unknown-audit-proportion   -> validity needs representative labeled unknowns
  (J)  number-of-clients scaling  -> Theorem-2 log(J)/per-client starvation
Saves figures to figs/ and prints the numbers.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certificates import cp_upper, conditional_risk_certificate, true_selective_risk
from clients import ClientPopulation, draw_counts

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig); print("saved", name)


# ---------- A4: calibration-budget sweep ----------
def a4(alpha=0.10, delta=0.10, T=1500, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array([0.7, 0.7, 0.7, 0.7, 0.6]); r = np.array([0.02, 0.02, 0.02, 0.02, 0.06])
    pop = ClientPopulation(a=a, r=r); J = pop.J
    lam = np.full(J, 1.0 / J); Rtrue = true_selective_risk(a, r, lam)
    ns = [40, 80, 160, 320, 640, 1280]
    frac_cert, med_U, cov = [], [], []
    for nper in ns:
        n = np.full(J, nper); c = 0; Us = []
        for _ in range(T):
            A, K = draw_counts(pop, n, rng)
            res = conditional_risk_certificate(A, K, n, delta, "simplex")
            Us.append(res.U); c += (res.U <= alpha)
        frac_cert.append(c / T); med_U.append(float(np.median(Us)))
    print(f"[A4] Rtrue={Rtrue:.3f} alpha={alpha}")
    for nper, f, u in zip(ns, frac_cert, med_U):
        print(f"   n/client={nper:5d}  P(certified)={f:.3f}  median U={u:.3f}")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(ns, med_U, "-o", color="#2980b9", label="median cert_ucb")
    ax.axhline(alpha, ls="--", color="#c0392b", label=r"$\alpha=0.10$")
    ax2 = ax.twinx()
    ax2.plot(ns, frac_cert, "-s", color="#27ae60", label="P(certified)")
    ax.set_xscale("log"); ax.set_xlabel("calibration points per client")
    ax.set_ylabel("median cert_ucb"); ax2.set_ylabel("P(certified)", color="#27ae60")
    ax.set_title("A4  Calibration-budget feasibility (Thm 2)")
    ax.legend(loc="upper right", fontsize=8); save(fig, "FA4_calibration_budget")


# ---------- A5: unknown-audit-proportion ----------
def a5(alpha=0.10, delta=0.10, T=3000, A=300, seed=1):
    # accepted point is error w.p. r_eff = u*1.0 + (1-u)*r_known (unknown leak = error)
    rng = np.random.default_rng(seed)
    r_known = 0.02; u_dep = 0.06                       # deployment unknown-among-accepted
    R_true = u_dep * 1.0 + (1 - u_dep) * r_known       # ~0.078 < alpha
    u_cals = [0.02, 0.04, 0.06, 0.08]
    cov = []
    for u_cal in u_cals:
        r_cal = u_cal * 1.0 + (1 - u_cal) * r_known
        K = rng.binomial(A, r_cal, size=T)
        U = np.array([cp_upper(int(k), A, delta) for k in K])
        cov.append(float(np.mean(R_true <= U)))
    print(f"[A5] R_true(dep)={R_true:.3f} (u_dep={u_dep}), target cov>= {1-delta}")
    for u, cv in zip(u_cals, cov):
        flag = "  <-- ANTI-CONSERVATIVE" if cv < 1 - delta - 0.01 else ""
        print(f"   calib unknown frac={u:.2f}  coverage of true R={cv:.3f}{flag}")
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.plot(u_cals, cov, "-o", color="#2980b9")
    ax.axhline(1 - delta, ls="--", color="#c0392b", label=r"target $1-\delta$")
    ax.axvline(u_dep, ls=":", color="0.4", label="deployment unknown frac")
    ax.set_xlabel("calibration unknown-among-accepted fraction")
    ax.set_ylabel("coverage of true deployment risk")
    ax.set_title("A5  Audit must represent unknowns"); ax.legend(fontsize=8)
    save(fig, "FA5_unknown_proportion")


# ---------- J: number-of-clients scaling ----------
def jscale(alpha=0.10, delta=0.10, T=1500, N_total=1500, seed=2):
    rng = np.random.default_rng(seed)
    Js = [2, 3, 5, 10, 20]; med_U = []
    for J in Js:
        nper = max(N_total // J, 1)
        a = np.full(J, 0.6); r = np.full(J, 0.04); pop = ClientPopulation(a=a, r=r)
        n = np.full(J, nper); Us = []
        for _ in range(T):
            A, K = draw_counts(pop, n, rng)
            Us.append(conditional_risk_certificate(A, K, n, delta, "simplex").U)
        med_U.append(float(np.median(Us)))
    print(f"[J] N_total={N_total} fixed; alpha={alpha}")
    for J, u in zip(Js, med_U):
        print(f"   J={J:3d}  n/client={N_total//J:4d}  median U={u:.3f}")
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.plot(Js, med_U, "-o", color="#8e44ad")
    ax.axhline(alpha, ls="--", color="#c0392b", label=r"$\alpha=0.10$")
    ax.set_xlabel("number of clients J (fixed total calibration)")
    ax.set_ylabel("median cert_ucb (simplex)")
    ax.set_title("J  Per-client starvation / log(J) penalty (Thm 2)")
    ax.legend(fontsize=8); save(fig, "FJ_client_scaling")


if __name__ == "__main__":
    a4(); print(); a5(); print(); jscale(); print("done")
