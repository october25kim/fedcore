"""P2: publication figures from existing CPU/exported data.

Generates (PDF + PNG, mathtext-safe, colorblind-safe) into figs/:
  F5 alpha-frontier  : CertCov@alpha vs alpha (d=5), SimpleCNN vs ResNet, proxy margin.
  F6 feasibility law : cert_ucb & CertCov@0.1 vs per-group accepted count, with the
                       Theorem-2 floor and the alpha line -- THE signature figure.
  F7 hetero collapse : min cert_ucb vs Dirichlet d (SimpleCNN), with the alpha line.

Run: ``python -m fedcore.plotting.make_figures``  (CPU, no torch)
"""

from __future__ import annotations

import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Journal column-width readability: larger axis/legend/title fonts.
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    "figure.titlesize": 15,
})

from fedcore.certify import certify_best_gamma, certify_best_gamma_grouped
from fedcore.grouping import make_group_map, repartition_trusted_pool
from fedcore.scores import scored_views

ALPHA, DELTA = 0.10, 0.10
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
CB = {"simplecnn": "#0072B2", "resnet18": "#D55E00", "floor": "#009E73", "alpha": "#444444"}
BASE = "" if glob.glob("runs/*.npz") else "../../"
FIGS = BASE + "runs/figs"


def _load_views(npz, score):
    d = np.load(npz)
    n_clients = int(d["cert_client"].max()) + 1
    views = {fn: scored_views(d[f"{fn}_logits"], d[f"{fn}_y_open"], d[f"{fn}_client"], [score])[score]
             for fn in ("prop", "cert", "test")}
    return views, n_clients, d


def _pool(d):
    return {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}


def staircase_points(npz, score="msp"):
    """(per_group_n, cert_ucb, CertCov@0.1) over G x cert_frac (worst-group, box)."""
    d = np.load(npz)
    n_clients = int(d["cert_client"].max()) + 1
    pool = _pool(d)
    pts = []
    for G in (5, 3, 2, 1):
        gmap = make_group_map(n_clients, G)
        for frac in (0.33, 0.5, 0.7):
            parts = repartition_trusted_pool(pool, frac, 0.2, seed=0)
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
    pool = _pool(d); parts = repartition_trusted_pool(pool, cert_frac, 0.2, seed=0)
    views = {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                              parts[fn]["client"], [score])[score] for fn in ("prop", "cert", "test")}
    out = {}
    for G in (5, 3, 2, 1):
        gmap = make_group_map(n_clients, G)
        r = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"], score_name=score,
                                       group_map=gmap, G=G, gammas=GAMMAS, alpha=ALPHA, delta=DELTA,
                                       Lambda="box", box=0.15, seed=0, margin=0.01)
        cov = r["cert_coverage_lcb"] if r["certified"] else 0.0
        out[G] = (r["cert_n"] / G, min(r["cert_risk_ucb"], 1.0), cov)
    return out


def _certcov_vs_budget(npz, G=2, score="msp", fracs=(0.2, 0.33, 0.5, 0.6, 0.7)):
    """CertifiedCoverage@0.1 at worst-group G vs audit budget (cert_frac)."""
    d = np.load(npz); n_clients = int(d["cert_client"].max()) + 1
    pool = _pool(d); gmap = make_group_map(n_clients, G)
    out = []
    for frac in fracs:
        parts = repartition_trusted_pool(pool, frac, 0.2, seed=0)
        views = {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                                  parts[fn]["client"], [score])[score] for fn in ("prop", "cert", "test")}
        r = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"], score_name=score,
                                       group_map=gmap, G=G, gammas=GAMMAS, alpha=ALPHA, delta=DELTA,
                                       Lambda="box", box=0.15, seed=0, margin=0.01)
        out.append((frac, r["cert_coverage_lcb"] if r["certified"] else 0.0))
    return out


def _client_scaling_cells(csv_path, alpha=0.20):
    """{J: [(G, mean_certcov, std_certcov, n_pass, n)]} from runs/client_scaling.csv.

    CertCov convention: uncertified seeds contribute zero (Section 5.1 headline metric).
    """
    import csv as _csv
    rows = [r for r in _csv.DictReader(
        (l for l in open(csv_path) if not l.startswith("#")))]
    out = {}
    for J in sorted({int(r["J"]) for r in rows}):
        cells = []
        Gs = sorted({int(r["G"]) for r in rows if int(r["J"]) == J}, reverse=True)
        for G in Gs:
            c = [r for r in rows if int(r["J"]) == J and int(r["G"]) == G
                 and abs(float(r["alpha"]) - alpha) < 1e-9]
            if not c:
                continue
            cov = [float(r["cert_coverage_lcb"]) if r["certified"] == "1" else 0.0 for r in c]
            cells.append((G, float(np.mean(cov)), float(np.std(cov)),
                          sum(r["certified"] == "1" for r in c), len(c)))
        out[J] = cells
    return out


def fig_F6():
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4.4))
    Gs = (5, 3, 2, 1)
    res = sorted(glob.glob(BASE + "runs/cifar10_d5_resnet18_seed*_logits.npz"))
    n_clients = None
    if res:
        per_seed = [_staircase_by_G(f) for f in res]
        n_clients = int(np.load(res[0])["cert_client"].max()) + 1
        # (a) certificate UCB vs per-group accepted count
        x = np.array([np.mean([s[G][0] for s in per_seed]) for G in Gs])
        ucb_m = np.array([np.mean([s[G][1] for s in per_seed]) for G in Gs])
        ucb_s = np.array([np.std([s[G][1] for s in per_seed]) for G in Gs])
        ax1.plot(x, ucb_m, "o-", color=CB["resnet18"], ms=6, label=f"resnet18 ({len(res)} seeds)")
        ax1.fill_between(x, ucb_m - ucb_s, ucb_m + ucb_s, color=CB["resnet18"], alpha=0.2)
        # (b) CertifiedCoverage@0.1 vs grouping G
        cov_m = np.array([np.mean([s[G][2] for s in per_seed]) for G in Gs])
        cov_s = np.array([np.std([s[G][2] for s in per_seed]) for G in Gs])
        gx = np.arange(len(Gs))
        ax2.bar(gx, cov_m, yerr=cov_s, color=CB["resnet18"], alpha=0.8, capsize=4)
        ax2.set_xticks(gx); ax2.set_xticklabels([f"G={G}" for G in Gs])
    sc = BASE + "runs/cifar10_d5_none0.0_seed0_logits.npz"
    if glob.glob(sc):
        s = _staircase_by_G(sc)
        ax1.plot([s[G][0] for G in Gs], [s[G][1] for G in Gs], "s--",
                 color=CB["simplecnn"], ms=5, label="simplecnn (1 seed)")
    # per-CLIENT Theorem-3 floor: ln(J/delta) / (-ln(1-alpha)), J = n_clients (~37/client)
    J = n_clients if n_clients else 5
    thm3 = np.log(J / DELTA) / (-np.log(1 - ALPHA))
    ax1.axhline(ALPHA, ls="--", color=CB["alpha"], label=r"$\alpha=0.1$")
    ax1.axvline(thm3, ls=":", color=CB["floor"], label=f"Thm 3 floor ({thm3:.0f}/client)")
    ax1.set_xscale("log"); ax1.set_xlabel("per-group accepted count")
    ax1.set_ylabel(r"certified $\bar{U}$ (risk UCB)")
    ax1.set_title("(a) cert_risk_ucb vs per-group count"); ax1.legend(fontsize=11)
    ax2.set_xlabel("grouping (G=1 pooled .. G=5 per-client)")
    ax2.set_ylabel(r"CertifiedCoverage@$\alpha=0.1$")
    ax2.set_title("(b) cert_coverage_lcb vs grouping")
    # (c) budget sweep: CertifiedCoverage@0.1 at worst-group G=2 vs audit budget
    if res:
        sweeps = [_certcov_vs_budget(f) for f in res]
        fracs = [p[0] for p in sweeps[0]]
        bm = np.array([np.mean([sw[i][1] for sw in sweeps]) for i in range(len(fracs))])
        bs = np.array([np.std([sw[i][1] for sw in sweeps]) for i in range(len(fracs))])
        ax3.plot(fracs, bm, "o-", color=CB["resnet18"], ms=6, label=f"resnet18 ({len(res)} seeds)")
        ax3.fill_between(fracs, bm - bs, bm + bs, color=CB["resnet18"], alpha=0.2)
    ax3.set_xlabel("audit budget (cert_frac)"); ax3.set_ylabel(r"CertifiedCoverage@$\alpha=0.1$ (G=2)")
    ax3.set_title("(c) cert_coverage_lcb vs audit budget"); ax3.legend(fontsize=11)
    # (d) client scaling J in {10, 20}: CertCov@0.20 vs grouping G (10 seeds, d=0.5)
    cs_csv = BASE + "runs/client_scaling.csv"
    if glob.glob(cs_csv):
        marker = {10: ("o-", "#0072B2"), 20: ("s--", "#D55E00")}
        for J_, cells in _client_scaling_cells(cs_csv, alpha=0.20).items():
            mk, col = marker.get(J_, ("^-.", "#009E73"))
            gx = [c[0] for c in cells]
            m = np.array([c[1] for c in cells]); s = np.array([c[2] for c in cells])
            n = cells[0][4]
            ax4.plot(gx, m, mk, color=col, ms=6, label=f"J={J_} ({n} seeds)")
            ax4.fill_between(gx, m - s, m + s, color=col, alpha=0.15)
            for G_, mi, _, npass, ntot in cells:
                ax4.annotate(f"{npass}/{ntot}", (G_, mi), textcoords="offset points",
                             xytext=(4, 6), fontsize=9)
        ax4.set_xscale("log")
        ax4.set_xticks([2, 5, 10, 20]); ax4.set_xticklabels(["2", "5", "10", "20"])
        ax4.invert_xaxis()
        ax4.set_xlabel("grouping G (left = per-client, right = coarser)")
        ax4.set_ylabel(r"CertifiedCoverage@$\alpha=0.2$")
        ax4.set_title("(d) client scaling ($d=0.5$)"); ax4.legend(fontsize=11)
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
    ax.axvline(0.10, ls=":", color=CB["floor"], lw=1.5, label="feasibility edge")
    ax.axvline(0.20, ls="--", color=CB["alpha"], lw=1.5, label="feasibility demonstration point")
    ax.set_xlabel(r"risk target $\alpha$"); ax.set_ylabel(r"CertifiedCoverage@$\alpha$ (G=5, box, margin)")
    ax.set_title("Certified coverage frontier on CIFAR-10"); ax.legend(fontsize=11)
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
    ax.set_ylabel(r"min cert_risk_ucb $\bar{U}$"); ax.set_title("min cert_risk_ucb vs heterogeneity (SimpleCNN)")
    ax.legend()
    _save(fig, "F7_hetero_collapse")


def fig_F2():
    """Necessity: deploy rate vs true selective risk, naive vs pooled vs Fed-CORE."""
    from fedcore.certificate import conditional_risk_certificate, pooled_cp, true_selective_risk
    from fedcore.data.clients import ClientPopulation, draw_counts
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
