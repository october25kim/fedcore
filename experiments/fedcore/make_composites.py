"""Composite publication figures (recombine EXISTING values; no new numbers, CPU).

TASK A -> figs/F6_feasibility_law.png (+pdf): ONE figure, three panels, same data as the
  standalone F6 (grouped staircase) + the calibration-budget sweep:
    (a) worst-group certified risk-UCB vs per-group accepted count (log x), G=5,3,2,1,
        with alpha=0.10 and the Theorem-2 floor   [source: make_figures._staircase_by_G on
        runs/cifar10_d5_resnet18_seed{0..4}_logits.npz (5-seed band) + simplecnn ref];
    (b) CertifiedCoverage@0.10 vs per-group accepted count                 [same source];
    (c) calibration-budget sweep: cert_ucb (left y) and P(certified) (right y) vs per-client
        calibration count (log x), alpha=0.10  [source: runs/ablation_calib_budget.csv].
  Message: grouping (G) and audit budget are the SAME sample-size lever of Theorem 2.

TASK B -> figs/F7_hetero_collapse.png (+pdf): ONE figure, two panels:
    (a) heterogeneity axis: min certified risk-UCB vs Dirichlet d (SimpleCNN stress)
        [source: runs/cifar10_d{0.1,0.5,5}_none0.0_seed0.csv];
    (b) corruption axis: worst-group CertCov vs client-side train-label noise rate, d in
        {5,0.5} (trusted calibration stays clean)  [source: runs/corruption_curve.csv].
  Message: each axis pushes the model's r_hat past alpha or starves the per-group count.

The standalone figs/ablation_calib_budget.png and figs/F9_corruption_curve.png are kept.

Run: python experiments/fedcore/make_composites.py   (CPU, no torch)
"""

from __future__ import annotations

import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import _staircase_by_G, CB, ALPHA, DELTA, BASE, FIGS

DPI = 200
GS = (5, 3, 2, 1)


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIGS}/{name}.{ext}", bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"saved {FIGS}/{name}.png (+pdf)")


# ---------------------------------------------------------------------------- #
# TASK A: F6 (staircase a,b + calibration budget c)
# ---------------------------------------------------------------------------- #
def fig_F6_composite():
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(15, 4.0))
    thm2 = np.log(2 / DELTA) / (-np.log(1 - ALPHA))   # ~37 (per-group Thm-2 floor, G=2)

    # (a)+(b) grouped staircase: ResNet d=5 5-seed band + simplecnn ref
    res = sorted(glob.glob(BASE + "runs/cifar10_d5_resnet18_seed*_logits.npz"))
    src_ab = []
    if res:
        per_seed = [_staircase_by_G(f) for f in res]
        x = np.array([np.mean([s[G][0] for s in per_seed]) for G in GS])
        for ax, idx, ylab, ttl in (
            (axa, 1, r"certified $\bar{U}$ (risk UCB)", "(a) certificate vs per-group count"),
            (axb, 2, r"CertifiedCoverage@$\alpha{=}0.1$", "(b) certified coverage vs per-group count")):
            mean = np.array([np.mean([s[G][idx] for s in per_seed]) for G in GS])
            std = np.array([np.std([s[G][idx] for s in per_seed]) for G in GS])
            ax.plot(x, mean, "o-", color=CB["resnet18"], ms=6, label=f"resnet18 ({len(res)} seeds)")
            ax.fill_between(x, mean - std, mean + std, color=CB["resnet18"], alpha=0.2)
            # annotate G at each point
            for xi, mi, G in zip(x, mean, GS):
                ax.annotate(f"G={G}", (xi, mi), textcoords="offset points", xytext=(4, 6), fontsize=7)
            ax.set_xscale("log"); ax.set_xlabel("per-group accepted count")
            ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=10)
        src_ab = res
    sc = BASE + "runs/cifar10_d5_none0.0_seed0_logits.npz"
    if glob.glob(sc):
        s = _staircase_by_G(sc); xs = [s[G][0] for G in GS]
        axa.plot(xs, [s[G][1] for G in GS], "s--", color=CB["simplecnn"], ms=5, label="simplecnn (1 seed)")
        axb.plot(xs, [s[G][2] for G in GS], "s--", color=CB["simplecnn"], ms=5, label="simplecnn (1 seed)")
    axa.axhline(ALPHA, ls="--", color=CB["alpha"], label=r"$\alpha=0.1$")
    axa.axvline(thm2, ls=":", color=CB["floor"], label=r"Thm 2 floor")
    axb.axvline(thm2, ls=":", color=CB["floor"])
    axa.legend(fontsize=8); axb.legend(fontsize=8)

    # (c) calibration-budget sweep from runs/ablation_calib_budget.csv
    cb_csv = BASE + "runs/ablation_calib_budget.csv"
    rows = list(csv.DictReader(open(cb_csv)))
    n_clients = 5
    xc = np.array([float(r["n_cert"]) / n_clients for r in rows])   # per-client calibration count
    ucb = np.array([float(r["cert_ucb_mean_all"]) for r in rows])   # mean over seeds (finite)
    pcert = np.array([int(r["n_pass"]) / int(r["n_seeds"]) for r in rows])
    axc.plot(xc, ucb, "o-", color=CB["resnet18"], ms=6, label=r"cert_ucb (mean)")
    axc.axhline(ALPHA, ls="--", color=CB["alpha"], label=r"$\alpha=0.1$")
    axc.set_xscale("log"); axc.set_xlabel("per-client calibration count")
    axc.set_ylabel(r"certified $\bar{U}$ (risk UCB)", color=CB["resnet18"])
    axc.tick_params(axis="y", labelcolor=CB["resnet18"])
    axc.set_title("(c) calibration-budget sweep", fontsize=10)
    axr = axc.twinx()
    axr.plot(xc, pcert, "^--", color=CB["floor"], ms=6, label="P(certified)")
    axr.set_ylabel("P(certified)", color=CB["floor"]); axr.tick_params(axis="y", labelcolor=CB["floor"])
    axr.set_ylim(-0.05, 1.05)
    lines = axc.get_lines()[:2] + axr.get_lines()
    axc.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")

    fig.suptitle("Feasibility law (Theorem 2): grouping and audit budget are the same sample-size lever",
                 fontsize=12, y=1.02)
    _save(fig, "F6_feasibility_law")
    print(f"  (a,b) source: {[os.path.basename(f) for f in src_ab]} + simplecnn ref")
    print(f"  (c) source: runs/ablation_calib_budget.csv (cert_ucb_mean_all, n_pass/n_seeds)")


# ---------------------------------------------------------------------------- #
# TASK B: F7 (heterogeneity a + corruption b)
# ---------------------------------------------------------------------------- #
def fig_F7_composite():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10, 4.0))

    # (a) heterogeneity: min cert_ucb vs Dirichlet d (SimpleCNN)
    cells = {"0.1": "runs/cifar10_d0.1_none0.0_seed0.csv",
             "0.5": "runs/cifar10_d0.5_none0.0_seed0.csv",
             "5": "runs/cifar10_d5_none0.0_seed0.csv"}
    ds, us = [], []
    for dd, f in cells.items():
        p = BASE + f
        if not glob.glob(p):
            continue
        rows = list(csv.DictReader(open(p)))
        feas = [r for r in rows if int(r["cert_n"]) > 0]
        u = min(float(r["cert_risk_ucb"]) for r in feas) if feas else 1.0
        ds.append(float(dd)); us.append(u)
    order = np.argsort(ds); ds = np.array(ds)[order]; us = np.array(us)[order]
    axa.plot(ds, us, "o-", color=CB["simplecnn"], ms=7)
    axa.axhline(ALPHA, ls="--", color=CB["alpha"], label=r"$\alpha=0.1$")
    axa.set_xscale("log"); axa.set_xlabel(r"Dirichlet $d$ (smaller = more non-IID)")
    axa.set_ylabel(r"min certified $\bar{U}$")
    axa.set_title("(a) heterogeneity axis (SimpleCNN stress)", fontsize=10)
    axa.legend(fontsize=8)
    axa.annotate("collapse at d=0.1", (ds[0], us[0]), textcoords="offset points",
                 xytext=(12, -4), fontsize=8, color=CB["alpha"])

    # (b) corruption: worst-group CertCov vs noise rate (symmetric, d in {5,0.5})
    crows = list(csv.DictReader(open(BASE + "runs/corruption_curve.csv")))
    color = {"5": "#0072B2", "5.0": "#0072B2", "0.5": "#D55E00"}
    for d in ("5", "0.5"):
        sub = [r for r in crows if r["noise_type"] == "symmetric" and r["d"] in (d, d + ".0")]
        sub.sort(key=lambda r: float(r["rate"]))
        if not sub:
            continue
        xr = [float(r["rate"]) for r in sub]
        c20 = [float(r["CertCov@0.20"]) for r in sub]
        c10 = [float(r["CertCov@0.10"]) for r in sub]
        axb.plot(xr, c20, "o-", color=color[d], ms=6, label=f"d={d}, $\\alpha$=0.20")
        axb.plot(xr, c10, "s--", color=color[d], ms=5, alpha=0.7, label=f"d={d}, $\\alpha$=0.10")
    axb.set_xlabel("symmetric client-label noise rate")
    axb.set_ylabel("worst-group CertCov (G=2)"); axb.set_ylim(bottom=-0.02)
    axb.set_title("(b) corruption axis", fontsize=10); axb.legend(fontsize=8)
    axb.annotate("trusted calibration stays clean", (0.5, 0.5), xycoords="axes fraction",
                 fontsize=8, color=CB["alpha"], ha="center")

    fig.suptitle("Stress axes of the feasibility law: each pushes $\\hat{r}$ past $\\alpha$ "
                 "or starves the per-group count", fontsize=12, y=1.02)
    _save(fig, "F7_hetero_collapse")
    print(f"  (a) source: runs/cifar10_d{{0.1,0.5,5}}_none0.0_seed0.csv (min cert_ucb)")
    print(f"  (b) source: runs/corruption_curve.csv (symmetric, d in {{5,0.5}})")


def main():
    fig_F6_composite()
    fig_F7_composite()


if __name__ == "__main__":
    main()
