"""Figure 7: certified pseudo-label ADMISSION prevents unsafe self-training (Prop 4).

Three panels, ALL from a real run_selftrain_cifar CSV (no synthetic curves):
  (a) realized pseudo-label contamination per round -- naive grows past alpha; certified
      admits nothing once infeasible (so it never contaminates).
  (b) ADMISSION / HALT behavior (replaces the old accuracy panel): the admitted
      pseudo-label fraction per round (the gate decision: admit the whole feasible batch
      or halt), with naive admitting every round (counts annotated) and certified marked
      with a HALT at the infeasible round.
  (c) round-wise validity inset: simultaneous unsafe rate for the delta/T temporal split
      vs delta-per-round (no split), against the delta=0.10 contract.

The figure reads as a certified ADMISSION GATE, not an accuracy experiment. Accuracy is
deliberately omitted (it is not the guaranteed quantity).

Run: ``python experiments/fedcore/make_F8.py [--csv runs/<selftrain>.csv]``
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA, DELTA, T = 0.10, 0.10, 5
CB = {"certified": "#009E73", "naive": "#D55E00", "none": "#0072B2",
      "alpha": "#444444", "halt": "#D55E00", "safe": "#009E73", "unsafe": "#D55E00"}
BASE = "" if glob.glob("runs/*.csv") else "../../"
DEFAULT_CSV = "runs/selftrain_cifar10_resnet18_d5_none0.0_seed0.csv"
# Proposition-4 validity contract (run_selftrain_smoke.py, Monte-Carlo; Table 6):
# WITH delta/T split -> simultaneous unsafe ~0.086 (<= delta); WITHOUT -> ~0.386 (> delta).
VALIDITY = {"delta/T split": 0.0856, "no split (delta/round)": 0.3856}


def _f(x):
    return np.nan if x in ("", "nan", "NaN", None) else float(x)


def _b(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def _load(csv_path):
    by = {}
    for r in csv.DictReader(open(csv_path)):
        by.setdefault(r["mode"], []).append(r)
    for m in by:
        by[m].sort(key=lambda r: int(r["round"]))
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    path = args.csv or (BASE + DEFAULT_CSV)
    if not os.path.exists(path):
        cand = sorted(glob.glob(BASE + "runs/selftrain_*.csv"))
        path = cand[-1] if cand else None
    if not path:
        print("no selftrain CSV found"); return
    print(f"F8 (Figure 7) from {path}")
    by = _load(path)
    print(f"  modes available: {list(by)}")

    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [1, 1, 0.72]})

    # ---- (a) contamination (KEEP; numbers unchanged) ------------------------
    for m in ("naive", "certified", "none"):
        if m not in by:
            continue
        rounds = [int(r["round"]) for r in by[m]]
        contam = [_f(r["realized_contam"]) for r in by[m]]
        if np.all(np.isnan(contam)):
            continue  # certified/none admit nothing -> no contamination to plot
        axa.plot(rounds, contam, "o-", color=CB[m], label=m, ms=5)
    axa.axhline(ALPHA, ls="--", color=CB["alpha"], lw=1.2, label=r"$\alpha=0.1$")
    axa.set_xlabel("self-training round"); axa.set_ylabel("pseudo-label contamination")
    axa.set_title("(a) contamination", fontsize=10)
    axa.set_xticks(range(0, T)); axa.legend(fontsize=8, loc="upper left")

    # ---- (b) admission / halt (NEW; replaces accuracy) ----------------------
    # `admitted` is a boolean gate (admit the whole feasible batch, or halt) -> plot as
    # a 0/1 admitted fraction; annotate the admitted COUNT (n_pseudo) for magnitude.
    for m in ("naive", "certified"):
        if m not in by:
            continue
        rounds = [int(r["round"]) for r in by[m]]
        frac = [1.0 if _b(r["admitted"]) else 0.0 for r in by[m]]
        npseudo = [int(float(r["n_pseudo"])) for r in by[m]]
        infeas = [_b(r["infeasible_round"]) for r in by[m]]
        axb.plot(rounds, frac, "o-", color=CB[m], label=m, ms=6, zorder=3)
        for x, f_, n in zip(rounds, frac, npseudo):
            if f_ > 0:  # annotate admitted-batch size on admitted rounds
                axb.annotate(f"n={n}", (x, f_), textcoords="offset points", xytext=(0, 7),
                             ha="center", fontsize=7, color=CB[m])
        # HALT markers at infeasible rounds (red X at y=0)
        halt_rounds = [x for x, ip in zip(rounds, infeas) if ip]
        if halt_rounds:
            axb.scatter(halt_rounds, [0.0] * len(halt_rounds), marker="X", s=130,
                        color=CB["halt"], zorder=5, label="certified HALT (infeasible)")
            axb.annotate("halt", (halt_rounds[0], 0.0), textcoords="offset points",
                         xytext=(10, 12), fontsize=9, color=CB["halt"], fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color=CB["halt"], lw=1))
    axb.set_ylim(-0.08, 1.15); axb.set_xlim(-0.4, T - 0.6); axb.set_xticks(range(0, T))
    axb.set_xlabel("self-training round"); axb.set_ylabel("admitted pseudo-label fraction")
    axb.set_title("(b) admission / halt", fontsize=10); axb.legend(fontsize=7.5, loc="center right")

    # ---- (c) round-wise validity inset (delta/T split) ----------------------
    labels = list(VALIDITY); vals = [VALIDITY[k] for k in labels]
    colors = [CB["safe"] if v <= DELTA + 1e-9 else CB["unsafe"] for v in vals]
    bars = axc.bar([0, 1], vals, color=colors, width=0.6)
    axc.axhline(DELTA, ls="--", color=CB["alpha"], lw=1.2)
    axc.annotate(r"$\delta=0.1$", (1.46, DELTA), fontsize=8, color=CB["alpha"], va="center")
    for b, v in zip(bars, vals):
        axc.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8)
    axc.set_xticks([0, 1]); axc.set_xticklabels([r"$\delta/T$" + "\nsplit", "no\nsplit"], fontsize=8)
    axc.set_ylabel("simultaneous unsafe rate"); axc.set_ylim(0, 0.45)
    axc.set_title("(c) round-wise validity", fontsize=10)

    fig.suptitle("Certified pseudo-label admission prevents unsafe self-training",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    figs = BASE + "experiments/fedcore/figs"
    os.makedirs(figs, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{figs}/F8_selftraining.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)

    # honest run log
    naive = by.get("naive", [])
    fr = [1.0 if _b(r["admitted"]) else 0.0 for r in naive]
    halts = [int(r["round"]) for r in by.get("certified", []) if _b(r["infeasible_round"])]
    print(f"  (a) contamination: naive {[round(_f(r['realized_contam']), 3) for r in naive]}")
    print(f"  (b) admitted fraction range: {min(fr) if fr else '-'}..{max(fr) if fr else '-'}; "
          f"naive n_pseudo {[int(float(r['n_pseudo'])) for r in naive]}; "
          f"certified HALT rounds {halts} (certified halted at round 0; only logged round)")
    print(f"  (c) validity: {VALIDITY} vs delta={DELTA}")
    print(f"saved {figs}/F8_selftraining.png (+pdf)")


if __name__ == "__main__":
    main()
