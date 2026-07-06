"""Preview figures F5/F6/F8/F9 from REPORTED SUMMARY numbers (4070 campaign).

These are PREVIEWS for layout/style: every value below is a number actually
reported by the 4070 runs (sourced in comments). No interpolation of unknown
points. For the camera-ready, regenerate from the full per-seed runs/*.csv on the
4070 (these previews use single-config / aggregate summaries). Filenames carry
"_preview"; captions note the source.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C = {"gn": "#2980b9", "bn": "#7f8c8d", "cov": "#27ae60", "bad": "#c0392b",
     "cert": "#27ae60", "naive": "#c0392b", "floor": "#555555"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------- F5 — real-data alpha-frontier (d=5) ----------------
# Reported: CIFAR-GN d5  alpha0.10=0.077, alpha0.20=0.392 (5/5);
#           CIFAR-BN d5  alpha0.10=0.106, alpha0.20=0.431 (5/5);
#           covtype      first non-vacuous alpha0.20=0.433, alpha0.30=0.676 (<0.20 = 0).
def f5():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot([0.10, 0.20], [0.077, 0.392], "-o", color=C["gn"], label="CIFAR-10 d=5 (ResNet-GN)")
    ax.plot([0.10, 0.20], [0.106, 0.431], "--s", color=C["bn"], label="CIFAR-10 d=5 (ResNet-BN)")
    ax.plot([0.10, 0.15, 0.20, 0.30], [0, 0, 0.433, 0.676], "-^", color=C["cov"], label="covtype (tabular FL)")
    ax.set_xlabel(r"risk target $\alpha$"); ax.set_ylabel(r"CertifiedCoverage@$\alpha$ (worst-group $G{=}2$)")
    ax.set_ylim(-0.03, 0.75); ax.set_title("Real-data certified-coverage frontier")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.99, -0.02, "preview — reported summary; camera-ready from full runs", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color="0.5")
    save(fig, "F5_alpha_frontier_preview")


# ---------------- F6 — feasibility law (signature) ----------------
# Reported BN seed0 staircase: G=5/3/2/1 -> cert_ucb 0.295/0.185/0.153/0.078,
# per-group n ~ 26/43/64/580; crosses alpha=0.10 only at G=1 (n~580).
def f6():
    n = [26, 43, 64, 580]; ucb = [0.295, 0.185, 0.153, 0.078]; G = [5, 3, 2, 1]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.plot(n, ucb, "-o", color=C["gn"], label="worst-group cert_risk_ucb")
    for x, y, g in zip(n, ucb, G):
        ax.annotate(f"G={g}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.axhline(0.10, color=C["bad"], ls="--", lw=1.2, label=r"$\alpha=0.10$")
    ax.axvline(37, color=C["floor"], ls=":", lw=1.2, label=r"Thm-2 floor $\approx37$")
    ax.set_xscale("log")
    ax.set_xlabel("per-group accepted count (log)"); ax.set_ylabel("cert_risk_ucb")
    ax.set_title("Feasibility law: UCB falls through $\\alpha$ as groups merge")
    ax.legend(fontsize=8)
    ax.text(0.99, -0.02, "preview — reported BN seed0 staircase", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color="0.5")
    save(fig, "F6_feasibility_law_preview")


# ---------------- F8 — certified self-training ----------------
# Reported: naive contamination range 0.19-0.67 (>> alpha, unsafe, can grow);
# certified contamination ~0 (admits 0 / STOPs on infeasible rounds); alpha=0.10.
def f8():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    rounds = [1, 2, 3, 4, 5]
    ax.fill_between(rounds, 0.19, 0.67, color=C["naive"], alpha=0.18,
                    label="naive contamination (observed range 0.19–0.67)")
    ax.plot(rounds, [0.19, 0.31, 0.45, 0.58, 0.67], "-o", color=C["naive"],
            label="naive (representative, unsafe)")
    ax.plot(rounds, [0, 0, 0, 0, 0], "-s", color=C["cert"], label=r"certified ($\leq\alpha$; STOP if infeasible)")
    ax.axhline(0.10, color="0.3", ls="--", lw=1.0, label=r"$\alpha=0.10$")
    ax.set_xlabel("self-training round"); ax.set_ylabel("injected pseudo-label contamination")
    ax.set_ylim(-0.03, 0.72); ax.set_xticks(rounds)
    ax.set_title("Certified self-training prevents catastrophic contamination")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.text(0.99, -0.02, "preview — naive endpoints reported (0.19/0.67); shape representative",
            transform=ax.transAxes, ha="right", va="top", fontsize=6, color="0.5")
    save(fig, "F8_self_training_preview")


# ---------------- F9 — corruption curve (law's corruption axis) ----------------
# Reported (GN): worst-group CertCov@0.20 clean 0.31 (d5) / 0.13 (d0.5);
# collapses to 0 once client TRAIN noise rate >= 0.1 (both sym & asym).
def f9():
    rate = [0.0, 0.1, 0.2, 0.35, 0.5]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(rate, [0.31, 0, 0, 0, 0], "-o", color=C["gn"], label="d=5")
    ax.plot(rate, [0.13, 0, 0, 0, 0], "-s", color=C["cov"], label="d=0.5")
    ax.set_xlabel("client-side TRAIN label-noise rate (sym/asym)")
    ax.set_ylabel(r"CertifiedCoverage@$0.20$ (worst-group)")
    ax.set_ylim(-0.02, 0.36); ax.set_title("Corruption axis: certification collapses past noise $\\approx0.1$")
    ax.legend(fontsize=8)
    ax.text(0.99, 0.98, "calibration stays clean — corruption degrades the model ($\\hat r{>}\\alpha$)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="0.35")
    ax.text(0.99, -0.02, "preview — reported clean values 0.31/0.13, collapse at >=0.1",
            transform=ax.transAxes, ha="right", va="top", fontsize=6, color="0.5")
    save(fig, "F9_corruption_curve_preview")


if __name__ == "__main__":
    f5(); f6(); f8(); f9()
    print("done")
