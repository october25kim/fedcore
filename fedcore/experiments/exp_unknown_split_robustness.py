"""M4 (+P2 extension): unknown-class split robustness (CPU aggregation of stored logits).

All headline CIFAR-10 results report the (seed-driven) known/unknown split. KEY POINT:
open_set_split is seed-driven, so the 10 GN-d0.5 PRIMARY seeds are ALREADY 10 distinct
random unknown-class splits -- this file certifies them (source=primary) AND six
PRE-DECLARED FIXED splits (same unknown set across all 10 seeds), to separate cross-split
variation from within-split seed noise:
  splitB unknown={0,1,2,3}; splitC {6,7,8,9}  (the two extremes)
  splitD {4,5,6,9}; splitE {0,1,4,9}; splitF {3,5,8,9}; splitG {3,4,7,9}
    (D-G: 4 random rotations pre-declared via rng seed=20260710, P2 extension.)
GN (resnet18gn) d=0.5, clean, seeds 0..9, grouped G=2, alpha in {0.10,0.20}, via the
IDENTICAL T9 grouped protocol (MSP head, seed-0 pooled repartition, box, delta=0.10) --
inlined so the script imports only from the fedcore package.

Finding (honest report of split sensitivity, per the acceptance criterion's alternative):
certified coverage is materially split-dependent; split CHOICE dominates seed noise.

CSV: runs/unknown_split_robustness.csv  (T9 schema + source,split,unknown_classes columns)

Run: python experiments/fedcore/exp_unknown_split_robustness.py   (CPU, no torch)
"""
from __future__ import annotations

import glob
import os
import re
import statistics as st

import numpy as np

from fedcore.certify import certify_best_gamma
from fedcore.certificate import cp_upper
from fedcore.data.fedosr_split import open_set_split
from fedcore.grouping import make_group_map, repartition_trusted_pool
from fedcore.io_utils import atomic_write_csv
from fedcore.scores import scored_views
from fedcore.selector import choose_threshold, counts_per_client

DELTA, G, SEED = 0.10, 2, 0
CERT_FRAC, TEST_FRAC = 0.5, 0.2
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
BOX, MARGIN = 0.15, 0.01
ALPHAS = (0.10, 0.20)
SPLITS = {"B": "0,1,2,3", "C": "6,7,8,9", "D": "4,5,6,9",
          "E": "0,1,4,9", "F": "3,5,8,9", "G": "3,4,7,9"}
PAT = "runs/cifar10_d0.5_resnet18gn_none0.0_split{sp}_seed{s}_logits.npz"
PRIMARY_PAT = "runs/cifar10_d0.5_resnet18gn_none0.0_seed{s}_logits.npz"
OUT = "runs/unknown_split_robustness.csv"
FIELDS = ["backbone", "d", "source", "split", "unknown_classes", "alpha", "seed",
          "cert_risk_ucb_G2", "cert_n_min_group", "cert_k_worst_group",
          "cert_coverage_lcb", "test_risk", "test_coverage", "selected_gamma", "certified"]


def _diagnostics(npz_path, alpha):
    d = np.load(npz_path)
    n_clients = int(d["cert_client"].max()) + 1
    pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}
    parts = repartition_trusted_pool(pool, CERT_FRAC, TEST_FRAC, seed=SEED)
    views = {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                              parts[fn]["client"], ["msp"])["msp"]
             for fn in ("prop", "cert", "test")}
    gmap = make_group_map(n_clients, G)
    gviews = {}
    for fn in ("prop", "cert", "test"):
        v = dict(views[fn]); v["client"] = gmap[np.asarray(views[fn]["client"])]
        gviews[fn] = v
    res = certify_best_gamma(gviews["prop"], gviews["cert"], gviews["test"],
                             score_name="msp", gammas=GAMMAS, alpha=alpha, delta=DELTA,
                             n_clients=G, dirichlet_alpha=float("nan"), Lambda="box",
                             box=BOX, seed=SEED, margin=MARGIN)
    sel = choose_threshold(gviews["prop"]["score"], gviews["prop"]["pred"],
                           gviews["prop"]["y_open"], res["gamma_star"], alpha)
    A, K, _n = counts_per_client(gviews["cert"]["score"], gviews["cert"]["pred"],
                                 gviews["cert"]["y_open"], gviews["cert"]["client"], sel, G)
    eps = DELTA / (3.0 * G)
    rbar = np.array([cp_upper(int(K[g]), int(A[g]), eps) if A[g] > 0 else np.inf for g in range(G)])
    worst = int(np.argmax(rbar))
    return {"cert_risk_ucb_G2": round(float(res["cert_risk_ucb"]), 4),
            "cert_n_min_group": int(A.min()), "cert_k_worst_group": int(K[worst]),
            "cert_coverage_lcb": round(float(res["cert_coverage_lcb"]), 4),
            "test_risk": round(float(res["test_risk"]), 4),
            "test_coverage": round(float(res["test_coverage"]), 4),
            "selected_gamma": float(res["gamma_star"]), "certified": int(bool(res["certified"]))}


def _primary_unknown(seed):
    """Seed-driven unknown set for the PRIMARY split (cifar10: classes 0..9)."""
    _, unk, _ = open_set_split(np.arange(10), 6, seed)
    return ",".join(str(int(c)) for c in unk)


def main():
    rows = []
    # PRIMARY: seed-driven splits (each seed = a distinct random unknown set), from existing logits
    for f in sorted(glob.glob(PRIMARY_PAT.format(s="[0-9]"))):
        s = int(re.search(r"seed(\d+)", os.path.basename(f)).group(1))
        for alpha in ALPHAS:
            rows.append({"backbone": "resnet18gn", "d": "0.5", "source": "primary",
                         "split": f"primary_s{s}", "unknown_classes": _primary_unknown(s),
                         "alpha": alpha, "seed": s, **_diagnostics(f, alpha)})
    # FIXED splits B..G: same unknown set across all seeds
    for sp, unk in SPLITS.items():
        files = sorted(glob.glob(PAT.format(sp=sp, s="[0-9]")))
        if not files:
            print(f"[warn] no split{sp} logits yet ({PAT.format(sp=sp, s='*')})")
            continue
        for f in files:
            s = int(re.search(r"seed(\d+)", os.path.basename(f)).group(1))
            for alpha in ALPHAS:
                rows.append({"backbone": "resnet18gn", "d": "0.5", "source": "fixed",
                             "split": sp, "unknown_classes": unk, "alpha": alpha, "seed": s,
                             **_diagnostics(f, alpha)})
    if not rows:
        print("[warn] no logits found; run manifest_M4 / manifest_M4ext first."); return
    atomic_write_csv(OUT, FIELDS, rows)
    print(f"saved {OUT}  ({len(rows)} rows)\n")

    def _cc(sub):
        return np.array([r["cert_coverage_lcb"] if r["certified"] else 0.0 for r in sub])

    # per FIXED split summary
    print(f"{'split':>10} {'unknown':>14} {'alpha':>5} {'n':>3} {'CertCov mean+/-sd':>19} {'frac_cert':>9}")
    for sp in SPLITS:
        for alpha in ALPHAS:
            sub = [r for r in rows if r.get("split") == sp and abs(r["alpha"] - alpha) < 1e-9]
            if not sub:
                continue
            cc, fc = _cc(sub), np.mean([r["certified"] for r in sub])
            print(f"{sp:>10} {SPLITS[sp]:>14} {alpha:>5.2f} {len(sub):>3} "
                  f"{cc.mean():>9.4f} +/-{cc.std():<7.4f} {fc:>9.3f}")

    # split-sensitivity DISTRIBUTION at alpha=0.20 (the headline): cross-split vs within-split
    a = 0.20
    prim = [r for r in rows if r["source"] == "primary" and abs(r["alpha"] - a) < 1e-9]
    if prim:
        pc = _cc(prim)
        print(f"\n[split sensitivity @alpha=0.20]")
        print(f"  PRIMARY (10 random seed-splits): CertCov mean={pc.mean():.3f} sd={pc.std():.3f} "
              f"range=[{pc.min():.3f}, {pc.max():.3f}]")
        fixed_means = []
        within_sds = []
        for sp in SPLITS:
            sub = [r for r in rows if r.get("split") == sp and abs(r["alpha"] - a) < 1e-9]
            if sub:
                cc = _cc(sub); fixed_means.append(cc.mean()); within_sds.append(cc.std())
        if fixed_means:
            print(f"  FIXED splits ({len(fixed_means)}): per-split-mean CertCov "
                  f"range=[{min(fixed_means):.3f}, {max(fixed_means):.3f}], "
                  f"cross-split sd={st.pstdev(fixed_means):.3f}; "
                  f"mean within-split seed sd={np.mean(within_sds):.3f}")
            print("  => split CHOICE dominates seed noise (cross-split sd >> within-split sd).")


if __name__ == "__main__":
    main()
