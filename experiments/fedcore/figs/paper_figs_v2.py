"""Publication figure set for the Fed-CORE manuscript (Figures 1-5), v2.

Ported to the refactored ``fedcore`` package (the v1 ``paper_figs.py`` imports
the pre-refactor module layout and no longer runs). Changes vs v1:

  * Figure 3 (A4 stress) is now TWO panels: (a) the synthetic Monte-Carlo sweep
    around the deployment unknown fraction 0.06 (matches Section 5.3 text), and
    (b) the real-logit rho-sweep from runs/ablation_unknown_prop.csv (deployment
    rate 0.30). This fixes the caption/figure mismatch in the previous build,
    where the synthetic caption sat on the real-data plot.
  * Figure 4 (feasibility law) gains panel (d) client scaling from
    runs/client_scaling.csv, and panel (c) now plots runs/ablation_calib_budget.csv
    (the audit-budget sweep quoted in Section 5.4) instead of an ad-hoc
    cert_frac recomputation, so text and figure share one data source.
  * Figure 5 (stress axes) panel (b) adds the asymmetric-noise curves from
    runs/corruption_curve.csv (single seed, dashed) next to the ten-seed
    symmetric means from runs/corruption_curve_seeded.csv; the rate-0 anchor is
    the Table-5 clean headline (same G=2 grouped protocol).

Outputs overwrite the filenames referenced by docs/Fed-CORE_draft.md.
Everything is recomputed from local artifacts (runs/*.csv, runs/*_logits.npz);
no numbers are hand-entered except the synthetic panel of Figure 3, which is
recomputed deterministically (seed=1), and the Figure-5 clean anchors, which
mirror runs/T9_diagnostics_simul.csv.

Run:  python experiments/fedcore/figs/paper_figs_v2.py   (CPU, no torch)
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
RUNS = os.path.join(ROOT, "runs")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fedcore.certificate.cp import cp_lower, cp_upper
from fedcore.certify import conditional_risk_certificate
from fedcore.scores import compute_score
from fedcore.selector import choose_threshold, open_set_error

# ---------------------------------------------------------------- style
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#CC79A7", sky="#56B4E9", grey="#7F7F7F", black="#222222")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "figure.dpi": 200,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def refline(ax, y, label, color=C["red"]):
    ax.axhline(y, ls="--", lw=1.3, color=color, zorder=1)
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"),
                xytext=(-2, 3), textcoords="offset points",
                ha="right", fontsize=9, color=color)


# ================================================================ Figure 1
def fig1_concept():
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc="#F4F6F8", ec=C["grey"], lw=1.2, fs=9.5,
            bold=False, tc=C["black"]):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                     fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, color=tc,
                fontweight="bold" if bold else "normal")

    def arrow(p, q, color=C["grey"], lw=1.6, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", lw=lw,
                     color=color, mutation_scale=13, zorder=1,
                     connectionstyle=f"arc3,rad={rad}"))

    ax.text(0.2, 7.62, "deployment stream: known + unseen classes, client"
            " mixture $\\lambda$ unknown", fontsize=9.5, color=C["grey"])

    box(0.20, 5.7, 2.7, 1.4, "heterogeneous clients\ntrain $\\hat h$\n"
        "(FedAvg / any FedOSR)", fs=9)
    box(3.55, 5.7, 2.5, 1.4, "deployed model $\\hat h$\n+ selector $A$")
    box(6.85, 6.45, 2.1, 0.85, "accept: act on $\\hat y$", fc="#E7F2E9",
        ec=C["green"], fs=9)
    box(6.75, 5.45, 2.3, 0.85, "reject: abstain / defer", fc="#FBEEE6",
        ec=C["orange"], fs=9)
    arrow((3.00, 6.4), (3.50, 6.4))
    arrow((6.15, 6.6), (6.80, 6.85), color=C["green"])
    arrow((6.15, 6.2), (6.80, 5.9), color=C["orange"])

    box(9.65, 6.45, 2.15, 0.85, "proposal fold:\npick threshold $t$",
        fc="#EDF3FA", ec=C["blue"], fs=8.5)
    box(9.65, 5.45, 2.15, 0.85, "certification fold:\ncounts $(A_j, K_j)$",
        fc="#EDF3FA", ec=C["blue"], fs=8.5)
    arrow((9.62, 7.25), (5.4, 7.18), color=C["blue"], lw=1.1, rad=-0.18)
    ax.text(8.15, 4.95, "test fold: deployment estimate only",
            fontsize=8, color=C["blue"], ha="center")

    ax.text(0.2, 4.35, "four questions about the same stream, of which only"
            " one is the deployment risk:", fontsize=10, color=C["black"])
    q = [("ranking quality\nAUROC / FPR95\n(all points)", C["grey"], 0.20),
         ("prediction-set\ncoverage\n(FCP, closed-set)", C["grey"], 3.15),
         ("batch FDR\n(novelty batch,\nno classifier)", C["grey"], 6.10),
         ("$\\Pr(\\hat y \\ne Y \\mid \\mathrm{accept})$\nunder unknown"
          " $\\lambda$\n= $R_{\\mathrm{sel}}(\\lambda)$", C["red"], 9.05)]
    for t, ec, x in q:
        hot = ec == C["red"]
        box(x, 1.9, 2.75, 1.9, t, fc="#FDF3E7" if hot else "#F4F6F8",
            ec=ec, lw=2.0 if hot else 1.2, fs=9, bold=hot)
    arrow((11.35, 5.35), (11.35, 3.95), color=C["blue"], lw=1.3)
    ax.text(11.62, 4.65, "certify", fontsize=8.5, color=C["blue"],
            rotation=90, va="center")
    ax.text(10.42, 1.15, "Fed-CORE: deploy iff  $\\bar U \\leq \\alpha$",
            fontsize=10.5, color=C["red"], ha="center", fontweight="bold")
    arrow((10.42, 1.8), (10.42, 1.45), color=C["red"], lw=1.6)
    save(fig, "fig0_problem_diagram")


# ================================================================ Figure 2
def fig2_pooling(delta=0.10):
    """Monte Carlo coverage curves cached in fig2_coverage.json
    (strat/pooled: 700 trials per shift; box: 80 trials, LFP-limited)."""
    import json
    data = json.load(open(os.path.join(HERE, "fig2_coverage.json")))
    shifts = np.array([d["shift"] for d in data])
    cov = {k: np.array([d[k] for d in data]) for k in ("pool", "strat", "box")}

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.axvspan(0, 0.10, color=C["sky"], alpha=0.15, lw=0)
    ax.text(0.05, 0.06, "inside\ndeclared $\\Lambda$", ha="center",
            fontsize=8.5, color=C["blue"])
    ax.plot(shifts, cov["strat"], color=C["blue"], lw=2.2, marker="o", ms=4)
    ax.plot(shifts, cov["box"], color=C["green"], lw=2.0, marker="s", ms=4)
    ax.plot(shifts, cov["pool"], color=C["red"], lw=2.2, marker="^", ms=4)
    refline(ax, 1 - delta, "target $1-\\delta$", color=C["black"])
    ax.annotate("stratified (Thm 1): valid for every mixture",
                xy=(0.35, 1.025), fontsize=9.5, color=C["blue"])
    ax.annotate("box-$\\Lambda$ (Thm 2):\nvalid inside its box",
                xy=(0.62, 0.60), fontsize=9.5, color=C["green"])
    ax.annotate("pooled CP:\ncollapses off the\nmatched mixture",
                xy=(0.24, 0.18), fontsize=9.5, color=C["red"])
    ax.set_xlabel("deployment shift toward the high-risk client\n"
                  "($0$ = calibration-matched mixture)")
    ax.set_ylabel("coverage  $\\Pr(R_{\\mathrm{sel}} \\leq \\bar U)$")
    ax.set_ylim(-0.04, 1.12)
    save(fig, "fig1_pooling_collapse")


# ================================================================ Figure 3
def fig3_a4(delta=0.10, T=3000, A=300, seed=1):
    """(a) synthetic sweep around deployment unknown fraction 0.06;
    (b) real-logit rho-sweep (deployment rate 0.30) from
        runs/ablation_unknown_prop.csv."""
    rng = np.random.default_rng(seed)
    r_known, u_dep = 0.02, 0.06
    r_dep = u_dep * 1.0 + (1 - u_dep) * r_known
    fracs = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
    covs = []
    for u in fracs:
        r_cal = u * 1.0 + (1 - u) * r_known
        k = rng.binomial(A, r_cal, size=T)
        ucb = np.array([cp_upper(int(ki), A, delta) for ki in k])
        covs.append(float((ucb >= r_dep).mean()))
    covs = np.array(covs)
    print("  synthetic A4 coverages:",
          dict(zip(fracs.tolist(), covs.round(3).tolist())))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))

    ax = axes[0]
    ax.axvspan(0.0, u_dep, color=C["red"], alpha=0.10, lw=0)
    ax.axvspan(u_dep, fracs.max() + 0.005, color=C["green"], alpha=0.08, lw=0)
    ax.text(0.033, 0.30, "under-represented:\nanti-conservative",
            fontsize=8.5, color=C["red"], ha="center")
    ax.text(0.097, 0.30, "matched / over-repr.:\ncoverage held (A4/A4$'$)",
            fontsize=8.5, color=C["green"], ha="center")
    ax.plot(fracs, covs, color=C["blue"], lw=2.2, marker="o", ms=5, zorder=3)
    for x, y in zip(fracs, covs):
        ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.axvline(u_dep, ls=":", lw=1.4, color=C["black"])
    ax.annotate("deployment fraction", xy=(u_dep, 0.02), xytext=(4, 0),
                textcoords="offset points", fontsize=8.5, rotation=90,
                va="bottom", color=C["black"])
    refline(ax, 1 - delta, "$1-\\delta$", color=C["black"])
    ax.set_xlabel("calibration unknown fraction")
    ax.set_ylabel("coverage of true deployment risk")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("(a) synthetic ($3{,}000$ draws/point)", fontsize=10.5)

    ax = axes[1]
    rows = list(csv.DictReader(open(os.path.join(RUNS, "ablation_unknown_prop.csv"))))
    x = np.array([float(r["cert_unknown_frac"]) for r in rows])
    y = np.array([float(r["coverage"]) for r in rows])
    p_dep = 0.30
    ax.axvspan(min(x) - 0.02, p_dep, color=C["red"], alpha=0.10, lw=0)
    ax.plot(x, y, color=C["blue"], lw=2.2, marker="o", ms=5, zorder=3)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}", xy=(xi, yi), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.axvline(p_dep, ls=":", lw=1.4, color=C["black"])
    ax.annotate("deployment fraction", xy=(p_dep, 0.02), xytext=(-11, 0),
                textcoords="offset points", fontsize=8.5, rotation=90,
                va="bottom", color=C["black"])
    refline(ax, 1 - delta, "$1-\\delta$", color=C["black"])
    ax.set_xlabel("certification-fold unknown fraction")
    ax.set_ylabel("")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(min(x) - 0.02, p_dep + 0.03)
    ax.set_title("(b) CIFAR-10 logits ($5$ seeds $\\times$ $40$ redraws)",
                 fontsize=10.5)
    fig.tight_layout(w_pad=2.0)
    save(fig, "ablation_unknown_prop")


# ============================================================= Figure 4 util
def _load_msp(path):
    z = np.load(path, allow_pickle=True)
    d = {}
    for f in ("prop", "cert", "test"):
        L = z[f"{f}_logits"]
        d[f] = dict(score=compute_score("msp", L), pred=L.argmax(1),
                    y_open=z[f"{f}_y_open"], client=z[f"{f}_client"])
    return d


def _grouped_cert(cert, sel, G, delta):
    J = int(cert["client"].max()) + 1
    gmap = np.empty(J, int)
    for gi, b in enumerate(np.array_split(np.arange(J), G)):
        gmap[b] = gi
    grp = gmap[cert["client"]]
    acc = cert["score"] >= sel.threshold
    err = open_set_error(cert["pred"], cert["y_open"])
    A = np.array([int(((grp == g) & acc).sum()) for g in range(G)])
    K = np.array([int(err[(grp == g) & acc].sum()) for g in range(G)])
    n = np.array([int((grp == g).sum()) for g in range(G)])
    U = conditional_risk_certificate(A, K, n, delta, Lambda="simplex").U
    lcb = sum(cp_lower(int(A[g]), int(n[g]), delta / (2 * G)) * n[g]
              for g in range(G)) / n.sum()
    return U, lcb, int(A.min())


def fig4_feasibility(alpha=0.10, delta=0.10):
    paths = sorted(glob.glob(os.path.join(
        RUNS, "cifar10_d5_resnet18gn_none0.0_seed*_logits.npz")))
    Gs = [5, 3, 2, 1]
    per_seed = {G: [] for G in Gs}   # (U, lcb, minA)
    for path in paths:
        d = _load_msp(path)
        best = None
        for gamma in (0.5, 0.7, 1.0):
            sel = choose_threshold(d["prop"]["score"], d["prop"]["pred"],
                                   d["prop"]["y_open"], gamma=gamma, alpha=alpha)
            if not sel.feasible:
                continue
            U2, lcb2, _ = _grouped_cert(d["cert"], sel, 2, delta)
            cand = (U2 <= alpha, lcb2, sel)
            if best is None or cand[:2] > best[:2]:
                best = cand
        sel = best[2]
        for G in Gs:
            per_seed[G].append(_grouped_cert(d["cert"], sel, G, delta))

    floor = np.log(5 / delta) / (-np.log(1 - alpha))
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.9))

    ax = axes[0]
    gcol = {5: C["red"], 3: C["orange"], 2: C["green"], 1: C["blue"]}
    for G in Gs:
        xs = [t[2] for t in per_seed[G]]
        ys = [min(t[0], 0.45) for t in per_seed[G]]
        ax.scatter(xs, ys, s=30, color=gcol[G], alpha=0.75, zorder=3,
                   label=f"$G$={G}")
    ax.set_xscale("log")
    ax.set_ylim(0.04, 0.44)
    ax.axvline(floor, ls=":", lw=1.4, color=C["purple"])
    ax.annotate("Thm 3\nfloor", xy=(floor, 0.30), xytext=(3, 0),
                textcoords="offset points", fontsize=8, color=C["purple"],
                va="top")
    ax.axhline(alpha, ls="--", lw=1.3, color=C["black"])
    ax.annotate("$\\alpha$", xy=(0.02, alpha),
                xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", fontsize=9, ha="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2,
              handletextpad=0.1, columnspacing=0.5, borderaxespad=0.1)
    ax.set_xlabel("min per-group accepted count")
    ax.set_ylabel("cert_risk_ucb")
    ax.set_title("(a) bound vs count", fontsize=10.5)

    ax = axes[1]
    covs = {G: [(t[1] if t[0] <= alpha else 0.0) for t in per_seed[G]] for G in Gs}
    means = np.array([np.mean(covs[G]) for G in Gs])
    stds = np.array([np.std(covs[G]) for G in Gs])
    ylo = np.minimum(stds, means)
    ax.bar(range(len(Gs)), means, yerr=[ylo, stds], capsize=3,
           color=[gcol[G] for G in Gs], alpha=0.75,
           edgecolor=C["black"], lw=0.6)
    ax.set_xticks(range(len(Gs)), [f"$G$={G}" for G in Gs], fontsize=9)
    if means[0] == 0:
        ax.annotate("0 (vacuous)", xy=(0, 0.004), ha="center", fontsize=7.5,
                    color=gcol[Gs[0]], rotation=90, va="bottom")
    ax.set_ylabel("CertifiedCoverage@$0.10$")
    ax.set_title("(b) coverage vs grouping", fontsize=10.5)

    ax = axes[2]
    rows = list(csv.DictReader(open(os.path.join(RUNS, "ablation_calib_budget.csv"))))
    x = [int(r["n_cert"]) for r in rows]
    m = np.array([float(r["CertCov_mean"]) for r in rows])
    sd = np.array([float(r["CertCov_std"]) for r in rows])
    pr = [int(r["n_pass"]) / int(r["n_seeds"]) for r in rows]
    ax.fill_between(x, np.maximum(m - sd, 0), m + sd,
                    color=C["sky"], alpha=0.3, lw=0)
    ax.plot(x, m, color=C["blue"], lw=2.0, marker="o", ms=4,
            label="CertCov@$0.10$")
    ax.plot(x, pr, color=C["orange"], lw=1.8, marker="s", ms=4, ls="--",
            label="pass rate")
    ax.set_xlabel("audit budget (cert. points)")
    ax.set_title("(c) coverage vs audit budget", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[3]
    rows = [r for r in csv.DictReader(
        (l for l in open(os.path.join(RUNS, "client_scaling.csv"))
         if not l.startswith("#"))) if r["alpha"] == "0.2"]
    style = {"10": (C["blue"], "o", "-"), "20": (C["orange"], "s", "--")}
    for Jv, (col, mk, ls) in style.items():
        sub = [r for r in rows if r["J"] == Jv]
        Gvals = sorted({int(r["G"]) for r in sub}, reverse=True)
        xs, ys, frac = [], [], []
        for G in Gvals:
            cell = [r for r in sub if int(r["G"]) == G]
            cov = [float(r["cert_coverage_lcb"]) if r["certified"] == "1"
                   else 0.0 for r in cell]
            xs.append(G); ys.append(np.mean(cov))
            frac.append(f"{sum(r['certified'] == '1' for r in cell)}/{len(cell)}")
        ax.plot(xs, ys, color=col, marker=mk, ls=ls, lw=1.9, ms=4.5,
                label=f"$J$={Jv}")
        for x_, y_, f_ in zip(xs, ys, frac):
            ax.annotate(f_, xy=(x_, y_), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xticks([20, 10, 5, 2], ["20", "10", "5", "2"])
    ax.minorticks_off()
    ax.invert_xaxis()
    ax.set_xlabel("grouping $G$ (coarser $\\rightarrow$)")
    ax.set_ylabel("CertifiedCoverage@$0.20$")
    ax.set_title("(d) client scaling", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(w_pad=1.4)
    save(fig, "F6_feasibility_law")
    # console cross-check for the Section 5.4 text
    budget = list(csv.DictReader(open(os.path.join(RUNS, "ablation_calib_budget.csv"))))
    print("  budget sweep: ucb %.3f -> %.3f ; CertCov %.3f -> %.3f ; pass %s/%s -> %s/%s"
          % (float(budget[0]["cert_ucb_mean_all"]),
             float(budget[-1]["cert_ucb_mean_all"]), m[0], m[-1],
             budget[0]["n_pass"], budget[0]["n_seeds"],
             budget[-1]["n_pass"], budget[-1]["n_seeds"]))


# ================================================================ Figure 5
def fig5_stress(alpha=0.10, delta=0.10):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))

    # (a) heterogeneity: best grouped U (G=2, alpha=0.10) vs d, ResNet-18
    ax = axes[0]
    ds = [0.1, 0.5, 5]
    pts = {d: [] for d in ds}
    for d_ in ds:
        pat = (glob.glob(os.path.join(RUNS, f"cifar10_d{d_}_resnet18_none0.0_seed*_logits.npz"))
               + glob.glob(os.path.join(RUNS, f"cifar10_d{d_}_resnet18_seed*_logits.npz")))
        for path in sorted(pat):
            dd = _load_msp(path)
            best_u = np.inf
            for gamma in (0.5, 0.7, 1.0):
                sel = choose_threshold(dd["prop"]["score"], dd["prop"]["pred"],
                                       dd["prop"]["y_open"], gamma=gamma,
                                       alpha=alpha)
                if not sel.feasible:
                    continue
                U, _, _ = _grouped_cert(dd["cert"], sel, 2, delta)
                best_u = min(best_u, U)
            pts[d_].append(best_u)
    xpos = np.arange(len(ds))
    cap = 0.42
    for i, d_ in enumerate(ds):
        vals = np.array(pts[d_])
        fin = vals[np.isfinite(vals)]
        inf_n = int((~np.isfinite(vals)).sum())
        if len(fin):
            ax.scatter([i] * len(fin), np.minimum(fin, cap), s=30,
                       color=C["blue"], alpha=0.6, zorder=3)
            ax.scatter([i], [np.median(fin)], s=90, marker="_",
                       color=C["blue"], lw=2.5, zorder=4)
        if inf_n:
            ax.scatter([i], [cap], s=70, marker="x", color=C["red"], zorder=4)
            ax.annotate("infeasible", xy=(i, cap), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, color=C["red"])
    ax.axhline(alpha, ls="--", lw=1.3, color=C["red"], zorder=1)
    ax.annotate("$\\alpha$", xy=(0.02, alpha), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", fontsize=9,
                ha="left", color=C["red"])
    ax.set_xticks(xpos, [f"$d$={d_}" for d_ in ds])
    ax.set_xlabel("Dirichlet concentration (smaller = more non-IID)")
    ax.set_ylabel("best grouped cert_risk_ucb")
    ax.set_ylim(0, 0.48)
    ax.set_title("(a) heterogeneity axis (ResNet-18)", fontsize=10.5)

    # (b) corruption: ten-seed symmetric means + single-seed asymmetric curves
    ax = axes[1]
    clean_anchor = {"5": 0.269, "0.5": 0.273}   # Table 5 headline (GN, G=2)
    rows = [r for r in csv.DictReader(
        (l for l in open(os.path.join(RUNS, "corruption_curve_seeded.csv"))
         if not l.startswith("#")))]
    sym = {}
    for d_ in ("5", "0.5"):
        rates, means, sds = [0.0], [clean_anchor[d_]], [0.0]
        for rate in ("0.1", "0.2", "0.35"):
            cell = [float(r["CertCov@0.20"]) for r in rows
                    if r["d"] == d_ and r["rate"] == rate]
            if cell:
                rates.append(float(rate)); means.append(np.mean(cell))
                sds.append(np.std(cell))
        sym[d_] = (np.array(rates), np.array(means), np.array(sds))
    # ten-seed means for BOTH noise types (asym arm added 2026-07-11, ws4090)
    curves = {}
    for nt in ("symmetric", "asymmetric"):
        for d_ in ("5", "0.5"):
            rates, means, sds = [0.0], [clean_anchor[d_]], [0.0]
            for rate in ("0.1", "0.2", "0.35"):
                cell = [float(r["CertCov@0.20"]) for r in rows
                        if r["noise_type"] == nt and r["d"] == d_
                        and r["rate"] == rate]
                if cell:
                    rates.append(float(rate)); means.append(np.mean(cell))
                    sds.append(np.std(cell))
            curves[(nt, d_)] = (np.array(rates), np.array(means), np.array(sds))
    style = {("symmetric", "5"): (C["blue"], "o", "-", 2.8),
             ("symmetric", "0.5"): (C["green"], "s", "-", 1.7),
             ("asymmetric", "5"): (C["orange"], "^", "--", 2.2),
             ("asymmetric", "0.5"): (C["purple"], "v", "--", 1.5)}
    for key, (x, m, sd) in curves.items():
        col, mk, ls, lw = style[key]
        nt, d_ = key
        ax.fill_between(x, np.maximum(m - sd, 0), m + sd, color=col,
                        alpha=0.12, lw=0)
        ax.plot(x, m, color=col, marker=mk, ls=ls, lw=lw, ms=4.5,
                label=f"{nt[:4]}., $d$={d_}")
    ax.set_xlabel("client-side training-label noise rate")
    ax.set_ylabel("CertifiedCoverage@$0.20$")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(b) corruption axis (calibration stays clean)",
                 fontsize=10.5)
    fig.tight_layout(w_pad=2.0)
    save(fig, "F7_hetero_collapse")


# ================================================================ Figure 6
def _certcov_at(path, alpha, delta=0.10, score_key=None):
    """Best-gamma grouped (G=2) CertCov@alpha for one run; matches Fig 4 util."""
    z = np.load(path, allow_pickle=True)
    d = {}
    for f in ("prop", "cert", "test"):
        if score_key:                       # native detector score stored in npz
            sc = -z[f"{f}_{score_key}"]     # PROSER sm: lower = more known -> negate
            pred = z[f"{f}_logits"].argmax(1)
        else:
            sc = compute_score("msp", z[f"{f}_logits"])
            pred = z[f"{f}_logits"].argmax(1)
        d[f] = dict(score=sc, pred=pred, y_open=z[f"{f}_y_open"],
                    client=z[f"{f}_client"])
    best = (False, 0.0)
    for gamma in (0.2, 0.3, 0.5, 0.7, 1.0):
        sel = choose_threshold(d["prop"]["score"], d["prop"]["pred"],
                               d["prop"]["y_open"], gamma=gamma, alpha=alpha)
        if not sel.feasible:
            continue
        U, lcb, _ = _grouped_cert(d["cert"], sel, 2, delta)
        cand = (U <= alpha, lcb)
        if cand > best:
            best = cand
    return best[1] if best[0] else 0.0


def fig6_frontier_detectors(delta=0.10):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))

    # (a) real-data alpha-frontier, grouped G=2 best-gamma (headline protocol)
    ax = axes[0]
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    series = [
        ("ResNet-GN + MSP (5 seeds)", C["blue"], "o", "-",
         sorted(glob.glob(os.path.join(RUNS, "cifar10_d5_resnet18gn_none0.0_seed*_logits.npz"))), None),
        ("ResNet-BN + MSP (5 seeds)", C["green"], "s", "-",
         sorted(glob.glob(os.path.join(RUNS, "cifar10_d5_resnet18_seed*_logits.npz"))), None),
        ("FedPD-PROSER native (5 seeds)", C["red"], "^", "--",
         sorted(glob.glob(os.path.join(RUNS, "fedpd_cifar10_d5_seed*.npz"))), "sm"),
    ]
    for lab, col, mk, ls, paths, sk in series:
        m, sd = [], []
        for a in alphas:
            covs = [_certcov_at(p, a, delta, sk) for p in paths]
            m.append(np.mean(covs)); sd.append(np.std(covs))
        m, sd = np.array(m), np.array(sd)
        ax.fill_between(alphas, np.maximum(m - sd, 0), m + sd, color=col,
                        alpha=0.12, lw=0)
        ax.plot(alphas, m, color=col, marker=mk, ls=ls, lw=1.9, ms=4.5, label=lab)
    ax.set_xlabel("risk target $\\alpha$")
    ax.set_ylabel("CertifiedCoverage@$\\alpha$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("(a) certified-coverage frontier ($d$=5)", fontsize=10.5)

    # (b) detector quality vs certified coverage (T8 aggregates, alpha=0.20)
    ax = axes[1]
    rows = list(csv.DictReader(open(os.path.join(RUNS, "T8_fedosr_bases_agg.csv"))))
    rows = [r for r in rows if r["alpha"] == "0.2"]
    style = {"5": ("o", 1.0), "0.5": ("s", 0.55)}
    seen = set()
    for r in rows:
        d_ = r["dirichlet_alpha"]
        if d_ not in style:
            continue
        mk, al = style[d_]
        x, y = float(r["auroc_mean"]), float(r["CertCovG2_mean"])
        lab = f"$d$={d_}" if d_ not in seen else None
        seen.add(d_)
        ax.scatter(x, y, s=55, marker=mk, color=C["blue"], alpha=al,
                   edgecolor=C["black"], lw=0.5, label=lab, zorder=3)
        name = r["base_model"].replace("FOOGD-SM3D-SAG", "FOOGD-SAG") \
                              .replace("FedAvg+MSP", "MSP")
        if d_ == "5":
            ax.annotate(name, xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
    ax.set_xlabel("native-score AUROC (base model)")
    ax.set_ylabel("CertifiedCoverage@$0.20$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("(b) coverage tracks detector quality", fontsize=10.5)
    fig.tight_layout(w_pad=2.0)
    save(fig, "F8_frontier_detectors")


if __name__ == "__main__":
    fig1_concept()
    fig2_pooling()
    fig3_a4()
    fig4_feasibility()
    fig5_stress()
    fig6_frontier_detectors()
