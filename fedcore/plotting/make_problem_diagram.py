"""fig0 problem diagram (schematic; no data) -> figs/fig0_problem_diagram.{png,svg}.

NOTE: the original figs/fig0_problem_diagram.svg was NOT present in this repo checkout
(it lives Mac-side), so this RECREATES the schematic from the described content rather
than editing an existing file. It keeps the described elements -- base FedOSR model +
selector A(x) -> accepted / rejected, and the four quantities (AUROC/FPR95, federated-CP
coverage, batch FDR, Fed-CORE R_sel) -- and ADDS the two trusted-fold annotations:
  * proposal fold  -> "selects the threshold of A(x)"
  * certification fold -> "certifies R_sel on a disjoint (independent) fold".
An editable .svg is also written so the original can be refreshed if desired.

Canvas ~780x460 px (figsize 7.8x4.6 @ dpi 100). CPU, no data.
"""

from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = "" if glob.glob("runs") else "../../"
FIGS = BASE + "experiments/fedcore/figs"

INK = "#222222"
C_MODEL = "#0072B2"
C_SEL = "#444444"
C_ACC = "#009E73"
C_REJ = "#D55E00"
C_CERT = "#009E73"
C_FOLD = "#7B3FA0"


def _box(ax, x, y, w, h, text, ec, fc="white", fs=9, lw=1.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                ec=ec, fc=fc, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=3, fontweight="bold" if bold else "normal")


def _arrow(ax, p0, p1, color=INK, lw=1.8, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                                 lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # --- main flow: x -> base model -> selector -> accept/reject ---
    ax.text(0.045, 0.66, "x\n(deployment\nstream)", ha="center", va="center", fontsize=8.5, color=INK)
    _box(ax, 0.12, 0.58, 0.22, 0.17, "Base FedOSR model\n(federated-trained)\n$\\to$ open-set score $s(x)$", C_MODEL)
    _box(ax, 0.41, 0.58, 0.17, 0.17, "Selector $A(x)$\naccept iff\n$s(x)\\geq t$", C_SEL)
    _box(ax, 0.66, 0.71, 0.16, 0.11, "accepted\n$\\to$ used", C_ACC)
    _box(ax, 0.66, 0.52, 0.16, 0.11, "rejected\n$\\to$ abstain", C_REJ)

    _arrow(ax, (0.085, 0.665), (0.12, 0.665))
    _arrow(ax, (0.34, 0.665), (0.41, 0.665), color=C_MODEL)
    _arrow(ax, (0.58, 0.69), (0.66, 0.765), color=C_ACC)
    _arrow(ax, (0.58, 0.64), (0.66, 0.575), color=C_REJ)

    # --- NEW: trusted folds feeding the selector and the certificate ---
    _box(ax, 0.40, 0.30, 0.19, 0.10, "proposal fold", C_FOLD, fc="#F3ECFA", fs=8.5)
    _arrow(ax, (0.495, 0.40), (0.495, 0.58), color=C_FOLD, rad=0.0)
    ax.text(0.605, 0.49, "selects the\nthreshold $t$ of $A(x)$", ha="left", va="center",
            fontsize=8, color=C_FOLD)

    _box(ax, 0.66, 0.20, 0.30, 0.155,
         "Risk certificate (Fed-CORE)\n$\\bar U(\\mathbf{K},\\mathbf{A};\\delta)\\leq\\alpha$\n"
         "$\\Rightarrow R_{\\mathrm{sel}}\\leq\\alpha$ w.p. $1-\\delta$", C_CERT, fc="#EAF6F1", fs=8.5, bold=False)
    _box(ax, 0.40, 0.06, 0.21, 0.10, "certification fold", C_FOLD, fc="#F3ECFA", fs=8.5)
    _arrow(ax, (0.61, 0.11), (0.66, 0.23), color=C_FOLD)
    ax.text(0.505, 0.015, "certifies risk on a disjoint (independent) fold",
            ha="center", va="center", fontsize=8, color=C_FOLD)
    _arrow(ax, (0.74, 0.71), (0.78, 0.355), color=C_ACC, lw=1.4, style="-|>", rad=0.0)
    ax.text(0.815, 0.52, "accepted set", ha="left", va="center", fontsize=7.5, color=C_ACC)

    # --- the four quantities: what each measures (Fed-CORE = the object) ---
    # placed in the empty bottom-left region (clear of the flow + folds).
    ax.text(0.015, 0.45, "What is measured on the accepted predictions:",
            fontsize=8.5, color=INK, fontweight="bold")
    quad = [
        ("AUROC / FPR95", "ranking quality of $s(x)$ — no deployment guarantee", C_MODEL),
        ("federated-CP coverage", "marginal coverage — not a post-selection risk", "#0072B2"),
        ("batch FDR", "per-batch — ignores client heterogeneity", "#D55E00"),
        ("Fed-CORE  $R_{\\mathrm{sel}}$", "accepted selective risk — CERTIFIED (the object)", C_CERT),
    ]
    for i, (q, sub, c) in enumerate(quad):
        y0 = 0.39 - i * 0.075
        ax.text(0.02, y0, "•", fontsize=12, color=c, va="center")
        ax.text(0.045, y0, q, fontsize=8.5, color=INK, va="center", fontweight="bold")
        ax.text(0.045, y0 - 0.032, sub, fontsize=6.8, color="#555555", va="center")

    fig.tight_layout(pad=0.4)
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(f"{FIGS}/fig0_problem_diagram.png", dpi=100)
    fig.savefig(f"{FIGS}/fig0_problem_diagram.svg")
    plt.close(fig)
    print(f"saved {FIGS}/fig0_problem_diagram.png (+svg)  [RECREATED schematic; original svg absent]")


if __name__ == "__main__":
    main()
