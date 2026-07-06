"""Figures for the resampling validity experiment (E1) and certificate
diagnostics, all computed from existing artifacts (no GPU):

  FE1_resampling_validity  — (a) per-config per-draw violation probability vs
                             delta; (b) violations concentrate at gamma=1.0
                             (the risk buffer is load-bearing).
  FE2_tightness_gap        — ECDF of cert_risk_ucb - test_risk across all real
                             runs (how conservative is the certificate?).
  FE3_certificate_anatomy  — per-client (A_j, K_j, UCB_j) for one real GN run,
                             per-client stratified vs grouped G=2.

Inputs: runs/resampling_validity.csv, runs/*_seed*.csv, runs/*_logits.npz.
Saves PDF + PNG to this directory.
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RUNS = os.path.join(ROOT, "runs")

from certificates import conditional_risk_certificate, cp_upper  # noqa: E402
from scores import compute_score  # noqa: E402
from selector import choose_threshold, open_set_error  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DELTA = 0.10


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}.pdf/.png")


# ---------------------------------------------------------------- FE1
def fig_resampling_validity() -> None:
    rows = [r for r in csv.DictReader(open(os.path.join(RUNS,
            "resampling_validity.csv"))) if r["status"] == "ok"]
    B = 1000
    pviol = np.array([int(r["n_violation"]) / B for r in rows])
    dep = sum(int(r["n_deploy_of_B"]) for r in rows)
    viol = sum(int(r["n_violation"]) for r in rows)

    by_gamma = {}
    for r in rows:
        d, v = by_gamma.setdefault(r["gamma"], [0, 0])
        by_gamma[r["gamma"]] = [d + int(r["n_deploy_of_B"]),
                                v + int(r["n_violation"])]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    ax.hist(pviol, bins=np.linspace(0, 0.012, 25), color="#4878a8",
            edgecolor="white")
    ax.axvline(DELTA, color="crimson", ls="--", lw=1.5,
               label=r"guarantee level $\delta=0.10$")
    ax.set_xlim(-0.0004, 0.014)
    ax.set_xlabel("per-configuration violation probability per draw")
    ax.set_ylabel("configurations")
    ax.set_title(f"(a) {len(rows)} configs, {B} audit-fold redraws each\n"
                 f"max observed {pviol.max():.3f} vs "
                 r"$\delta=0.10$", fontsize=10)
    ax.legend(fontsize=8, loc="upper center")

    ax = axes[1]
    gammas = sorted(by_gamma)
    rates = [by_gamma[g][1] / max(by_gamma[g][0], 1) for g in gammas]
    deps = [by_gamma[g][0] for g in gammas]
    bars = ax.bar([f"$\\gamma$={g}" for g in gammas], rates,
                  color=["#4878a8", "#4878a8", "#c44e52"])
    for b, d_, v in zip(bars, deps, [by_gamma[g][1] for g in gammas]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v}/{d_}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("violations / deployments")
    ax.set_title("(b) violations concentrate at $\\gamma$=1.0\n"
                 "(no risk buffer); buffered proposals are clean", fontsize=10)
    fig.suptitle(f"Resampling validity on real CIFAR logits: "
                 f"{viol}/{dep} violations over "
                 f"{len(rows) * B:,} certificate evaluations "
                 f"(CP95 UCB {cp_upper(viol, dep, 0.05):.4f})",
                 fontsize=10.5, y=1.04)
    save(fig, "FE1_resampling_validity")


# ---------------------------------------------------------------- FE2
def fig_tightness_gap() -> None:
    """Gap = cert_risk_ucb - test_risk, recomputed from the stored logits with
    the paper's protocol (MSP; gamma in {0.5,0.7,1.0}; alpha in {0.1,0.2};
    G in {5 per-client, 2 grouped}), split by whether the certificate deploys."""
    cert_gaps, other_gaps = [], []
    for path in sorted(glob.glob(os.path.join(RUNS, "*_logits.npz"))):
        z = np.load(path, allow_pickle=True)
        fold = {}
        for f in ("prop", "cert", "test"):
            logits = z[f"{f}_logits"]
            fold[f] = {"score": compute_score("msp", logits),
                       "pred": logits.argmax(axis=1),
                       "y_open": z[f"{f}_y_open"],
                       "client": z[f"{f}_client"]}
        J = int(fold["cert"]["client"].max()) + 1
        c, t = fold["cert"], fold["test"]
        c_err = open_set_error(c["pred"], c["y_open"])
        t_err = open_set_error(t["pred"], t["y_open"])
        for alpha in (0.10, 0.20):
            for gamma in (0.5, 0.7, 1.0):
                sel = choose_threshold(fold["prop"]["score"],
                                       fold["prop"]["pred"],
                                       fold["prop"]["y_open"],
                                       gamma=gamma, alpha=alpha)
                if not sel.feasible:
                    continue
                acc = c["score"] >= sel.threshold
                t_acc = t["score"] >= sel.threshold
                test_risk = float(t_err[t_acc].mean()) if t_acc.sum() else 0.0
                for G in (J, 2):
                    grp_c = np.empty(J, dtype=int)
                    for gi, block in enumerate(
                            np.array_split(np.arange(J), G)):
                        grp_c[block] = gi
                    grp = grp_c[c["client"]]
                    A = np.array([int(((grp == g) & acc).sum())
                                  for g in range(G)])
                    K = np.array([int(c_err[(grp == g) & acc].sum())
                                  for g in range(G)])
                    n = np.array([int((grp == g).sum()) for g in range(G)])
                    res = conditional_risk_certificate(A, K, n, DELTA,
                                                       Lambda="simplex")
                    if not np.isfinite(res.U) or res.U >= 1.0:
                        continue
                    (cert_gaps if res.U <= alpha else other_gaps).append(
                        res.U - test_risk)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for gaps, label, color, lw in (
            (other_gaps, "finite bound, not deployed", "#9fb8d0", 1.4),
            (cert_gaps, "deployed ($\\bar U\\leq\\alpha$)", "#c44e52", 2.0)):
        g = np.sort(gaps)
        ax.step(g, np.arange(1, len(g) + 1) / len(g), where="post",
                label=f"{label} (n={len(g)}, median {np.median(g):.3f})",
                color=color, lw=lw)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"conservatism gap: $\bar U$ $-$ test_risk")
    ax.set_ylabel("ECDF")
    ax.set_title("Tightness on real CIFAR runs (MSP; $\\alpha\\in\\{0.1,0.2\\}$,"
                 " $G\\in\\{J,2\\}$, $\\gamma\\in\\{0.5,0.7,1.0\\}$)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "FE2_tightness_gap")
    cg, og = np.array(cert_gaps), np.array(other_gaps)
    print(f"gap not-deployed: n={len(og)}, min={og.min():.4f}, "
          f"med={np.median(og):.4f}")
    print(f"gap deployed: n={len(cg)}, min={cg.min():.4f}, "
          f"med={np.median(cg):.4f}, max={cg.max():.4f}")


# ---------------------------------------------------------------- FE3
def fig_anatomy(run: str = "cifar10_d5_resnet18gn_none0.0_seed0",
                gamma: float = 0.7, alpha: float = 0.10) -> None:
    z = np.load(os.path.join(RUNS, f"{run}_logits.npz"), allow_pickle=True)
    fold = {}
    for f in ("prop", "cert"):
        logits = z[f"{f}_logits"]
        fold[f] = {"score": compute_score("msp", logits),
                   "pred": logits.argmax(axis=1),
                   "y_open": z[f"{f}_y_open"],
                   "client": z[f"{f}_client"]}
    sel = choose_threshold(fold["prop"]["score"], fold["prop"]["pred"],
                           fold["prop"]["y_open"], gamma=gamma, alpha=alpha)
    c = fold["cert"]
    err = open_set_error(c["pred"], c["y_open"])
    acc = c["score"] >= sel.threshold
    J = int(c["client"].max()) + 1
    A = np.array([int(((c["client"] == j) & acc).sum()) for j in range(J)])
    K = np.array([int(err[(c["client"] == j) & acc].sum()) for j in range(J)])
    n = np.array([int((c["client"] == j).sum()) for j in range(J)])
    res5 = conditional_risk_certificate(A, K, n, DELTA, Lambda="simplex")
    ucb5 = np.array([cp_upper(int(K[j]), int(A[j]), DELTA / J)
                     if A[j] > 0 else 1.0 for j in range(J)])
    # grouped G=2 (contiguous blocks, as in exp_resampling_validity)
    blocks = np.array_split(np.arange(J), 2)
    A2 = np.array([A[b].sum() for b in blocks])
    K2 = np.array([K[b].sum() for b in blocks])
    n2 = np.array([n[b].sum() for b in blocks])
    ucb2 = np.array([cp_upper(int(K2[g]), int(A2[g]), DELTA / 2)
                     for g in range(2)])

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(J)
    bars = ax.bar(x, ucb5, color="#9fb8d0", edgecolor="#4878a8",
                  label=r"per-client $\bar r_j$ (Thm 1, $\varepsilon=\delta/J$)")
    worst = int(np.argmax(ucb5))
    bars[worst].set_color("#e8a0a4"); bars[worst].set_edgecolor("#c44e52")
    for j in range(J):
        ax.text(j, 0.012, f"$A_j$={A[j]}\n$K_j$={K[j]}", ha="center",
                fontsize=7.5, color="#1f3b57")
    ax.axhline(alpha, color="crimson", ls="--", lw=1.4,
               label=fr"target $\alpha$={alpha}")
    ax.axhline(res5.U, color="#4878a8", ls=":", lw=1.4,
               label=fr"simplex certificate $\bar U$={res5.U:.3f} (= worst client)")
    for g, b in enumerate(blocks):
        ax.hlines(ucb2[g], b[0] - 0.4, b[-1] + 0.4, color="#2e7d54", lw=2.2)
        ax.text(np.mean(b), ucb2[g] - 0.012,
                f"group {g}: $\\bar r_g$={ucb2[g]:.3f}",
                ha="center", fontsize=8, color="#2e7d54")
    ax.set_xticks(x, [f"client {j}\n($n_j$={n[j]})" for j in range(J)],
                  fontsize=8)
    ax.set_ylim(0, res5.U * 1.30)
    ax.set_ylabel("Clopper–Pearson UCB on conditional risk")
    ax.set_title("Anatomy of one real certificate "
                 f"(MSP, $\\gamma$={gamma}, $\\alpha$={alpha}, "
                 f"$\\delta$={DELTA})", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "FE3_certificate_anatomy")
    print(f"anatomy: A={A.tolist()} K={K.tolist()} U5={res5.U:.4f} "
          f"U2={ucb2.max():.4f}")


if __name__ == "__main__":
    fig_resampling_validity()
    fig_tightness_gap()
    fig_anatomy()
