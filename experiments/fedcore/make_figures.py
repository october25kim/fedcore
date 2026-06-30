"""P2: publication figures from existing CPU/exported data.

Generates (PDF + PNG, mathtext-safe, colorblind-safe) into figs/:
  F5 alpha-frontier  : CertCov@alpha vs alpha (d=5), SimpleCNN vs ResNet, proxy margin.
  F6 feasibility law : cert_ucb & CertCov@0.1 vs per-group accepted count, with the
                       Theorem-2 floor and the alpha line -- THE signature figure.
  F7 hetero collapse : min cert_ucb vs Dirichlet d (SimpleCNN), with the alpha line.

Run: ``python experiments/fedcore/make_figures.py``  (CPU, no torch)
"""

from __future__ import annotations

import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from certify import certify_best_gamma, certify_best_gamma_grouped
from exp_feasibility_lever import _group_map
from scores import scored_views

ALPHA, DELTA = 0.10, 0.10
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
CB = {"simplecnn": "#0072B2", "resnet18": "#D55E00", "floor": "#009E73", "alpha": "#444444"}
BASE = "" if glob.glob("runs/*.npz") else "../../"
FIGS = BASE + "experiments/fedcore/figs"


def _load_views(npz, score):
    d = np.load(npz)
    n_clients = int(d["cert_client"].max()) + 1
    views = {fn: scored_views(d[f"{fn}_logits"], d[f"{fn}_y_open"], d[f"{fn}_client"], [score])[score]
             for fn in ("prop", "cert", "test")}
    return views, n_clients, d


def _pool(d):
    return {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}


def _repartition(pool, cert_frac, test_frac, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pool["y_open"]); perm = rng.permutation(n)
    n_test = int(round(n * test_frac)); n_cert = int(round(n * cert_frac))
    ix = {"test": perm[:n_test], "cert": perm[n_test:n_test + n_cert], "prop": perm[n_test + n_cert:]}
    return {f: {k: pool[k][i] for k in ("logits", "y_open", "client")} for f, i in ix.items()}


def staircase_points(npz, score="msp"):
    """(per_group_n, cert_ucb, CertCov@0.1) over G x cert_frac (worst-group, box)."""
    d = np.load(npz)
    n_clients = int(d["cert_client"].max()) + 1
    pool = _pool(d)
    pts = []
    for G in (5, 3, 2, 1):
        gmap = np.array([c * G // n_clients for c in range(n_clients)])
        for frac in (0.33, 0.5, 0.7):
            parts = _repartition(pool, frac, 0.2)
            views = {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                                      parts[fn]["client"], [score])[score] for fn in ("prop", "cert", "test")}
            r = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"],
                                           score_name=score, group_map=gmap, G=G, gammas=GAMMAS,
                                           alpha=ALPHA, delta=DELTA, Lambda="box", box=0.15, seed=0, margin=0.0)
            cov = r["cert_coverage_lcb"] if r["certified"] else 0.0
            pts.append((r["cert_n"] / G, r["cert_risk_ucb"], cov, G))
    return pts


def _staircase_by_G(npz, cert_frac=0.5, score="msp"):
    """{G: (per_group_n, cert_ucb, CertCov@0.1)} at a fixed cert_frac."""
    d = np.load(npz); n_clients = int(d["cert_client"].max()) + 1
    pool = _pool(d); parts = _repartition(pool, cert_frac, 0.2)
    views = {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                              parts[fn]["client"], [score])[score] for fn in ("prop", "cert", "test")}
    out = {}
    for G in (5, 3, 2, 1):
        gmap = _group_map(n_clients, G)
        r = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"], score_name=score,
                                       group_map=gmap, G=G, gammas=GAMMAS, alpha=ALPHA, delta=DELTA,
                                       Lambda="box", box=0.15, seed=0, margin=0.01)
        cov = r["cert_coverage_lcb"] if r["certified"] else 0.0
        out[G] = (r["cert_n"] / G, min(r["cert_risk_ucb"], 1.0), cov)
    return out


def fig_F6():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # ResNet d=5: 5-seed band (mean +/- std over seeds, per G)
    res = sorted(glob.glob(BASE + "runs/cifar10_d5_resnet18_seed*_logits.npz"))
    if res:
        per_seed = [_staircase_by_G(f) for f in res]
        Gs = (5, 3, 2, 1)
        x = np.array([np.mean([s[G][0] for s in per_seed]) for G in Gs])
        for ax, idx in ((ax1, 1), (ax2, 2)):
            mean = np.array([np.mean([s[G][idx] for s in per_seed]) for G in Gs])
            std = np.array([np.std([s[G][idx] for s in per_seed]) for G in Gs])
            ax.plot(x, mean, "o-", color=CB["resnet18"], ms=6, label=f"resnet18 ({len(res)} seeds)")
            ax.fill_between(x, mean - std, mean + std, color=CB["resnet18"], alpha=0.2)
    # SimpleCNN d=5: single-seed reference
    sc = BASE + "runs/cifar10_d5_none0.0_seed0_logits.npz"
    if glob.glob(sc):
        s = _staircase_by_G(sc); Gs = (5, 3, 2, 1)
        x = [s[G][0] for G in Gs]
        ax1.plot(x, [s[G][1] for G in Gs], "s--", color=CB["simplecnn"], ms=5, label="simplecnn (1 seed)")
        ax2.plot(x, [s[G][2] for G in Gs], "s--", color=CB["simplecnn"], ms=5, label="simplecnn (1 seed)")
    thm2 = np.log(2 / DELTA) / (-np.log(1 - ALPHA))
    ax1.axhline(ALPHA, ls="--", color=CB["alpha"], label=r"$\alpha=0.1$")
    ax1.axvline(thm2, ls=":", color=CB["floor"], label=r"Thm 2 floor")
    ax1.set_xlabel("per-group accepted count"); ax1.set_ylabel(r"certified $\bar{U}$ (risk UCB)")
    ax1.set_xscale("log"); ax1.set_title("(a) certificate vs per-group count"); ax1.legend(fontsize=8)
    ax2.axvline(thm2, ls=":", color=CB["floor"])
    ax2.set_xlabel("per-group accepted count"); ax2.set_ylabel(r"CertifiedCoverage@$\alpha=0.1$")
    ax2.set_xscale("log"); ax2.set_title("(b) certified coverage vs per-group count"); ax2.legend(fontsize=8)
    fig.suptitle("F6  Feasibility law (Theorem 2): per-group count moves the certificate (ResNet 5-seed band)")
    _save(fig, "F6_feasibility_law")


def fig_F5():
    runs = {"simplecnn": BASE + "runs/cifar10_d5_none0.0_seed0_logits.npz",
            "resnet18": BASE + "runs/cifar10_d5_resnet18_seed0_logits.npz"}
    alphas = [0.10, 0.15, 0.20, 0.25]
    fig, ax = plt.subplots(figsize=(6, 4))
    for bb, npz in runs.items():
        if not glob.glob(npz):
            continue
        views, n_clients, _ = _load_views(npz, "msp")
        covs = []
        for a in alphas:
            r = certify_best_gamma(views["prop"], views["cert"], views["test"], score_name="msp",
                                   gammas=GAMMAS, alpha=a, delta=DELTA, n_clients=n_clients,
                                   dirichlet_alpha=float("nan"), Lambda="box", box=0.15, seed=0, margin=0.01)
            covs.append(r["cert_coverage_lcb"] if r["certified"] else 0.0)
        ax.plot(alphas, covs, "o-", color=CB[bb], label=bb, ms=6)
    ax.set_xlabel(r"risk target $\alpha$"); ax.set_ylabel(r"CertifiedCoverage@$\alpha$ (G=5, box, margin)")
    ax.set_title("F5  Certified-coverage frontier (cifar10 d=5)"); ax.legend()
    _save(fig, "F5_alpha_frontier")


def fig_F7():
    import csv
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
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ds, us, "o-", color=CB["simplecnn"], ms=7)
    ax.axhline(ALPHA, ls="--", color=CB["alpha"], label=r"$\alpha=0.1$")
    ax.set_xscale("log"); ax.set_xlabel(r"Dirichlet $d$ (smaller = more non-IID)")
    ax.set_ylabel(r"min certified $\bar{U}$"); ax.set_title("F7  Heterogeneity collapse (SimpleCNN)")
    ax.legend()
    _save(fig, "F7_hetero_collapse")


def fig_F2():
    """Necessity: deploy rate vs true selective risk, naive vs pooled vs Fed-CORE."""
    from certificates import conditional_risk_certificate, pooled_cp, true_selective_risk
    from clients import ClientPopulation, draw_counts
    alpha = 0.05
    n = np.array([300, 300, 300, 300, 400]); lam = n / n.sum()
    a = np.array([0.7] * 4 + [0.5])
    Rs, dn, dp, df = [], [], [], []
    for r_bad in np.linspace(0.02, 0.40, 20):
        r = np.array([0.02] * 4 + [r_bad]); pop = ClientPopulation(a=a, r=r)
        Rs.append(true_selective_risk(a, r, lam))
        rng = np.random.default_rng(0); N = 1500; cn = cp = cf = 0
        for _ in range(N):
            A, K = draw_counts(pop, n, rng)
            tA, tK = int(A.sum()), int(K.sum())
            cn += (tK / tA if tA else 0) <= alpha
            cp += pooled_cp(A, K, DELTA) <= alpha
            cf += conditional_risk_certificate(A, K, n, DELTA, Lambda="known", lam=lam).U <= alpha
        dn.append(cn / N); dp.append(cp / N); df.append(cf / N)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Rs, dn, "o-", color="#D55E00", label="naive (empirical)")
    ax.plot(Rs, dp, "s-", color="#0072B2", label="pooled-CP")
    ax.plot(Rs, df, "^-", color="#009E73", label="Fed-CORE (cond.)")
    ax.axvline(alpha, ls="--", color=CB["alpha"], label=r"$\alpha=0.05$")
    ax.axvspan(alpha, max(Rs), color="red", alpha=0.06)
    ax.set_xlabel(r"true selective risk $R_{\mathrm{sel}}$"); ax.set_ylabel("deploy rate")
    ax.set_title(r"F2  Necessity: unsafe-deploy (shaded $R_{\mathrm{sel}}>\alpha$)")
    ax.legend(fontsize=8)
    _save(fig, "F2_necessity")


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGS}/{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {FIGS}/{name}.pdf (+png)")


def main():
    fig_F2()
    fig_F6()
    fig_F5()
    fig_F7()


if __name__ == "__main__":
    main()
