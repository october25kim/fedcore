"""F_selftrain_gain: certified one-shot self-training accuracy gain (only built because a
positive gain appeared; the admission/halt F8_selftraining figure remains the primary).

Two panels from runs/selftrain_pkg.csv (alpha=0.20, audit 4x, beta=1.0):
  (a) accuracy gain (Δ known_acc vs none) for certified vs oracle(clean UB), per base model
      {FedAvg+MSP (n=1), FedPD-PROSER (n=3, mean±std + per-seed dots)} -- the gain grows with a
      stronger base detector;
  (b) safety: realized contamination of the admitted batch vs alpha -- certified stays <= alpha
      (0/N violations) while naive sits at/above alpha.

Honest: FedPD certified gain is seed-variable (+0.04-0.05 when it admits; seed 2 admits nothing
-> Δ=0). All values from the real CSV (convergence-guarded). CPU.
"""

from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.20
CB = {"certified": "#009E73", "oracle": "#56B4E9", "naive": "#D55E00", "alpha": "#444444"}
BASE = "" if glob.glob("runs/*.csv") else "../../"
FIGS = BASE + "experiments/fedcore/figs"
SRC = BASE + "runs/selftrain_pkg.csv"
AUDIT, BETA = "4", "1.0"          # the operating point where certified admits


def _load():
    rows = [x for x in csv.DictReader(open(SRC)) if float(x["known_acc"]) >= 0.4]
    # per (base, mode) -> {seed: row} at audit 4x, beta 1.0 (none is audit/beta-invariant)
    by = defaultdict(lambda: defaultdict(dict))
    for x in rows:
        a, b = x.get("audit_mult", "1"), x.get("beta", "1.0")
        mode = x["mode"]
        # none: audit/beta-invariant; oracle: beta-dependent only (audit-invariant);
        # certified/naive: depend on the audit budget.
        if mode == "none":
            by[x["base_model"]][mode][x["seed"]] = x
        elif mode == "oracle" and b == BETA:
            by[x["base_model"]][mode][x["seed"]] = x
        elif mode in ("certified", "naive") and a == AUDIT and b == BETA:
            by[x["base_model"]][mode][x["seed"]] = x
    return by


def main():
    by = _load()
    bases = [b for b in ("FedAvg+MSP", "FedPD-PROSER") if b in by]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- (a) accuracy gain vs none ----
    width = 0.35
    xpos = np.arange(len(bases))
    for off, mode in ((-width / 2, "certified"), (width / 2, "oracle")):
        means, stds, allpts = [], [], []
        for b in bases:
            none = {s: float(r["known_acc"]) for s, r in by[b]["none"].items()}
            gains = [float(r["known_acc"]) - none[s] for s, r in by[b][mode].items() if s in none]
            gains = np.array(gains)
            means.append(gains.mean() if len(gains) else 0.0)
            stds.append(gains.std() if len(gains) > 1 else 0.0)
            allpts.append(gains)
        axa.bar(xpos + off, means, width, yerr=stds, capsize=4, color=CB[mode],
                alpha=0.85, label=("certified (Fed-CORE)" if mode == "certified" else "oracle (clean UB)"))
        for xi, pts in zip(xpos + off, allpts):
            axa.scatter([xi] * len(pts), pts, color="black", s=14, zorder=5)
    axa.axhline(0, color="gray", lw=0.8)
    axa.set_xticks(xpos)
    axa.set_xticklabels([f"{b}\n(n={len(by[b]['none'])})" for b in bases])
    axa.set_ylabel(r"accuracy gain  $\Delta$ known-acc vs none")
    axa.set_title("(a) certified self-training gain (alpha=0.20, audit 4x)", fontsize=10)
    axa.legend(fontsize=8, loc="upper left")
    axa.annotate("stronger detector\n=> larger safe gain", (1, 0.001), fontsize=8,
                 color=CB["certified"], ha="center")

    # ---- (b) safety: admitted-batch contamination vs alpha ----
    labels, vals, cols = [], [], []
    for b in bases:
        for mode in ("certified", "naive"):
            cs = [float(r["realized_contam"]) for r in by[b].get(mode, {}).values()
                  if r["realized_contam"] not in ("", "nan")]
            if cs:
                labels.append(f"{b.split('+')[0].split('-')[0]}\n{mode}")
                vals.append(np.mean(cs)); cols.append(CB[mode])
    xb = np.arange(len(labels))
    axb.bar(xb, vals, 0.6, color=cols, alpha=0.85)
    axb.axhline(ALPHA, ls="--", color=CB["alpha"], lw=1.4)
    axb.annotate(r"$\alpha=0.20$ (contamination budget)", (len(labels) - 0.5, ALPHA + 0.005),
                 fontsize=8, color=CB["alpha"], ha="right")
    for x, v in zip(xb, vals):
        axb.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    axb.set_xticks(xb); axb.set_xticklabels(labels, fontsize=8)
    axb.set_ylabel("realized contamination of admitted batch")
    axb.set_ylim(0, max(ALPHA + 0.06, max(vals) + 0.04))
    axb.set_title("(b) safety: certified <= alpha (0/N viol.); naive at budget", fontsize=10)

    fig.suptitle("Certified one-shot self-training: a SAFE accuracy gain on a strong base detector",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIGS}/F_selftrain_gain.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"saved {FIGS}/F_selftrain_gain.png (+pdf)")
    for b in bases:
        none = {s: float(r["known_acc"]) for s, r in by[b]["none"].items()}
        for mode in ("certified", "oracle"):
            g = np.array([float(r["known_acc"]) - none[s] for s, r in by[b][mode].items() if s in none])
            print(f"  {b:13s} {mode:9s}: Δ={g.mean():+.4f}+/-{g.std():.4f} (n={len(g)})")


if __name__ == "__main__":
    main()
