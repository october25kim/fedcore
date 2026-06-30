"""Corruption curve: worst-group CertifiedCoverage vs client-side noise rate.

The corruption axis of the feasibility law. Post-hoc on exported GN logits:
for d in {5, 0.5} and noise in {symmetric, asymmetric} at rates {0,0.1,0.2,0.35,0.5}
(rate 0 = the clean run), compute the worst-group G=2 CertifiedCoverage@{0.10,0.20}
(fixed-MSP, box, cert_frac=0.5) and plot vs noise rate. Missing cells are skipped.

Run: ``python experiments/fedcore/make_corruption_curve.py``  (CPU, no torch)
"""

from __future__ import annotations

import csv
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from certify import certify_best_gamma_grouped
from fedcore.grouping import _group_map, _repartition
from scores import scored_views

DELTA = 0.10
RATES = [0.0, 0.1, 0.2, 0.35, 0.5]
BASE = "" if glob.glob("runs/*.npz") else "../../"
CB = {5: "#0072B2", 0.5: "#D55E00"}


def _g2_cov(npz, alpha):
    d = np.load(npz); nc = int(d["cert_client"].max()) + 1
    pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}
    p = _repartition(pool, 0.5, 0.2, 0)
    v = {fn: scored_views(p[fn]["logits"], p[fn]["y_open"], p[fn]["client"], ["msp"])["msp"]
         for fn in ("prop", "cert", "test")}
    r = certify_best_gamma_grouped(v["prop"], v["cert"], v["test"], score_name="msp",
                                   group_map=_group_map(nc, 2), G=2, gammas=(0.2, 0.3, 0.5, 0.7, 1.0),
                                   alpha=alpha, delta=DELTA, Lambda="box", box=0.15, seed=0, margin=0.01)
    return r["cert_coverage_lcb"] if r["certified"] else 0.0


def _npz_for(d, nt, rate):
    if rate == 0.0:
        p = f"{BASE}runs/cifar10_d{d}_resnet18gn_none0.0_seed0_logits.npz"
    else:
        p = f"{BASE}runs/cifar10_d{d}_resnet18gn_{nt}{rate}_seed0_logits.npz"
    return p if glob.glob(p) else None


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    rows = []
    for nt, ax, ls in (("symmetric", ax1, "-"), ("asymmetric", ax2, "--")):
        for d in (5, 0.5):
            xs, c10, c20 = [], [], []
            for rate in RATES:
                npz = _npz_for(d, nt, rate)
                if not npz:
                    continue
                xs.append(rate)
                a10, a20 = _g2_cov(npz, 0.10), _g2_cov(npz, 0.20)
                c10.append(a10); c20.append(a20)
                rows.append([nt, d, rate, round(a10, 3), round(a20, 3)])
            if xs:
                ax.plot(xs, c20, "o-", color=CB[d], label=f"d={d}, $\\alpha$=0.20")
                ax.plot(xs, c10, "s--", color=CB[d], label=f"d={d}, $\\alpha$=0.10", alpha=0.7)
        ax.set_xlabel(f"{nt} noise rate"); ax.set_ylabel("worst-group CertCov (G=2)")
        ax.set_title(f"{nt} corruption"); ax.legend(fontsize=8); ax.set_ylim(bottom=0)
    fig.suptitle("Corruption axis of the feasibility law (ResNet-GN, cifar10)")
    fig.tight_layout()
    import os
    os.makedirs(BASE + "experiments/fedcore/figs", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{BASE}experiments/fedcore/figs/F9_corruption_curve.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    with open(BASE + "runs/corruption_curve.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["noise_type", "d", "rate", "CertCov@0.10", "CertCov@0.20"])
        w.writerows(rows)
    print(f"saved figs/F9_corruption_curve.pdf (+png), runs/corruption_curve.csv ({len(rows)} cells)")


if __name__ == "__main__":
    main()
