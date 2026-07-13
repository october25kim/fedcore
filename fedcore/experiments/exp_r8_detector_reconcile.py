"""R8: reconcile the T9 diagnostics detector cells with the published T8 detector
block (read-only, CPU, no torch).

Motivation. runs/T9_diagnostics.csv disagreed with runs/T8_fedosr_bases.csv on the
FedPD/FOOGD cells (FOOGD d=5 alpha=0.20 mean cert_coverage_lcb 0.3498 in T9 vs
0.0706 in T8 -- a ~5x gap; FedPD 0.4912 vs 0.4828 -- a small gap). This script
isolates the cause by re-certifying each detector npz under all four combinations
of the two candidate protocol axes and comparing to BOTH recorded tables.

Axes.
  (A) SCORE HEAD:  native detector score (-sm, T8) vs fixed MSP on logits (T9).
  (B) GAMMA GRID:  {0.5,0.7,1.0} (T8) vs {0.2,0.3,0.5,0.7,1.0} (T9/main.py).
Everything else is identical (G=2 grouped, cert_frac=0.5, test_frac=0.2,
box=0.15, delta=0.10, seed=0 fold repartition), so any disagreement must come
from (A) and/or (B).

Conclusion (reproduced by the columns below).
  * The FOOGD 5x gap is ENTIRELY the score head: T9's "foogd" row is MSP on the
    FOOGD backbone (byte-identical to T8's controlled FedAvg+MSP baseline), NOT
    the native FOOGD-SM3D detector. The detector's native score certifies 0.0706.
  * FedPD's gap is also the score head, but tiny because PROSER's -sm ~ MSP.
  * The gamma grid moves nothing at alpha=0.20 (gamma*=0.7 in both grids); it only
    diverges for a few seeds at alpha=0.10 where the wider grid picks gamma<=0.3.
  * G=2 is the SAME coverage quantity in both tables; not a source of the gap.

Decision: the detector block reports each detector's NATIVE open-set score (T8).
T9's MSP-scored detector rows measured a different quantity (an MSP baseline on
the detector backbone). exp_alpha20_diagnostics.py now scores detector cells with
-sm + the {0.5,0.7,1.0} grid, reproducing T8 exactly (verified: 0/24 mismatch).

Output: runs/T9_detector_reconciliation.csv
Run:    python -m fedcore.experiments.exp_r8_detector_reconcile
"""
from __future__ import annotations

import csv
import numpy as np

from fedcore.certify import certify_best_gamma_grouped
from fedcore.scores import msp as msp_score
from fedcore.io_utils import atomic_write_csv

DELTA, MARGIN, CERT_FRAC, TEST_FRAC, BOX = 0.10, 0.01, 0.5, 0.2, 0.15
DETECTORS = [("foogd", "FOOGD-SM3D"), ("fedpd", "FedPD-PROSER")]
DS, ALPHAS, SEEDS = ["5", "0.5"], [0.10, 0.20], [0, 1, 2]
GRID_T8 = (0.5, 0.7, 1.0)
GRID_T9 = (0.2, 0.3, 0.5, 0.7, 1.0)


def _pool(npz):
    d = np.load(npz)
    keys = ("logits", "sm", "y_open", "client")
    return {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")]) for k in keys}


def _repartition(pool, seed=0):
    n = len(pool["y_open"])
    perm = np.random.default_rng(seed).permutation(n)
    nt, nc = int(round(n * TEST_FRAC)), int(round(n * CERT_FRAC))
    idx = {"test": perm[:nt], "cert": perm[nt:nt + nc], "prop": perm[nt + nc:]}
    return {f: {k: pool[k][ix] for k in pool} for f, ix in idx.items()}


def certify(npz, score_key, gammas, alpha):
    pool = _pool(npz)
    accept = -np.asarray(pool["sm"], float) if score_key == "native" else msp_score(pool["logits"])
    pool = dict(pool); pool["_accept"] = accept
    parts = _repartition(pool)
    views = {f: {"score": parts[f]["_accept"],
                 "pred": np.asarray(parts[f]["logits"], float).argmax(-1),
                 "y_open": parts[f]["y_open"], "client": parts[f]["client"]}
             for f in ("prop", "cert", "test")}
    n_clients = int(pool["client"].max()) + 1
    G = 2
    gmap = np.array([c * G // n_clients for c in range(n_clients)])
    r = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"],
                                   score_name=score_key, group_map=gmap, G=G, gammas=gammas,
                                   alpha=alpha, delta=DELTA, Lambda="box", box=BOX, seed=0, margin=MARGIN)
    return float(r["cert_coverage_lcb"])


def _recorded():
    t9, t8 = {}, {}
    with open("runs/T9_diagnostics.csv") as f:
        for r in csv.DictReader(f):
            if r["backbone"] in ("foogd", "fedpd"):
                t9[(r["backbone"], r["d"], float(r["alpha"]), int(r["seed"]))] = float(r["cert_coverage_lcb"])
    with open("runs/T8_fedosr_bases.csv") as f:
        for r in csv.DictReader(f):
            b = {"FOOGD-SM3D": "foogd", "FedPD-PROSER": "fedpd"}.get(r["base_model"])
            if b:
                t8[(b, r["dirichlet_alpha"], float(r["alpha"]), int(r["seed"]))] = float(r["cert_coverage_lcb"])
    return t9, t8


def main():
    t9, t8 = _recorded()
    fields = ["det", "d", "alpha", "seed", "rec_T9", "rec_T8",
              "msp_wide", "msp_narrow", "native_narrow", "native_wide"]
    rows = []
    hdr = f"{'det':>6} {'d':>4} {'a':>4} {'s':>2} | {'T9rec':>7} {'msp+w':>7} {'msp+n':>7} | {'T8rec':>7} {'nat+n':>7} {'nat+w':>7}"
    print(hdr); print("-" * len(hdr))
    for det, _ in DETECTORS:
        for d in DS:
            for alpha in ALPHAS:
                for s in SEEDS:
                    npz = f"runs/{det}_cifar10_d{d}_seed{s}.npz"
                    mw = certify(npz, "msp", GRID_T9, alpha)
                    mn = certify(npz, "msp", GRID_T8, alpha)
                    nn = certify(npz, "native", GRID_T8, alpha)
                    nw = certify(npz, "native", GRID_T9, alpha)
                    r9 = t9.get((det, d, alpha, s), float("nan"))
                    r8 = t8.get((det, d, alpha, s), float("nan"))
                    rows.append({"det": det, "d": d, "alpha": alpha, "seed": s,
                                 "rec_T9": round(r9, 4), "rec_T8": round(r8, 4),
                                 "msp_wide": round(mw, 4), "msp_narrow": round(mn, 4),
                                 "native_narrow": round(nn, 4), "native_wide": round(nw, 4)})
                    print(f"{det:>6} {d:>4} {alpha:>4.2f} {s:>2} | {r9:>7.4f} {mw:>7.4f} {mn:>7.4f} "
                          f"| {r8:>7.4f} {nn:>7.4f} {nw:>7.4f}")
    # assertions: msp+wide reproduces T9, native+narrow reproduces T8
    def _close(a, b):
        return all(abs(x[a] - x[b]) < 1e-3 for x in rows if not np.isnan(x["rec_T9"]))
    print(f"\nmsp_wide reproduces rec_T9   : {_close('msp_wide', 'rec_T9')}")
    print(f"native_narrow reproduces rec_T8: {_close('native_narrow', 'rec_T8')}")
    atomic_write_csv("runs/T9_detector_reconciliation.csv", fields, rows)
    print("saved runs/T9_detector_reconciliation.csv")


if __name__ == "__main__":
    main()
