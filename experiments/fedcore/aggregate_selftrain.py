"""Aggregate runs/selftrain_pkg.csv -> runs/selftrain_pkg_agg.csv (seed-aware, guarded).

Two guards prevent contaminated aggregates:
  1. CONVERGENCE GUARD: drop any row with known_acc < MIN_ACC (=0.30). A converged base
     gives ~0.65-0.78; a value near chance (1/n_known ~ 0.17) means the base training did
     NOT converge (e.g. a leftover SMOKE run, or a diverged FedPD-from-insufficient-pretrain)
     -- such rows are TRAINING FAILURES, not seeds, and must never enter a gain mean.
  2. SEED-AWARE grouping: n_seeds = number of DISTINCT seeds (not row count), so a duplicate
     row for the same seed cannot inflate n_seeds.

Excluded rows are printed so nothing is hidden. Run: python experiments/fedcore/aggregate_selftrain.py
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict

import numpy as np

# Convergence guard: drop runs whose base is ~chance (training failure / smoke). For 6 known
# classes chance = 0.167; smoke runs land ~0.20; a LEGIT low-label base (e.g. labeled_frac=0.10)
# reaches ~0.36. 0.30 cleanly separates the two (excludes 0.20 smoke, keeps 0.36 weak-but-real).
MIN_ACC = 0.30
BASE = "" if glob.glob("runs/*.csv") else "../../"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/selftrain_pkg.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min_acc", type=float, default=MIN_ACC)
    args = ap.parse_args()
    min_acc = args.min_acc
    SRC = BASE + args.src
    OUT = BASE + (args.out or args.src.replace(".csv", "_agg.csv"))
    rows = list(csv.DictReader(open(SRC)))
    kept, dropped = [], []
    for x in rows:
        try:
            ok = float(x["known_acc"]) >= min_acc
        except (ValueError, KeyError):
            ok = False
        (kept if ok else dropped).append(x)

    print(f"read {len(rows)} rows; kept {len(kept)}; DROPPED {len(dropped)} non-converged "
          f"(known_acc < {min_acc}):")
    for x in dropped:
        print(f"  DROP base={x['base_model']} mode={x['mode']} seed={x.get('seed')} "
              f"known_acc={float(x['known_acc']):.3f} admitted={x.get('admitted_count')} "
              f"(training failure / smoke -> excluded)")

    # group by (base, labeled_frac, alpha, mode, audit, beta) over DISTINCT seeds
    g = defaultdict(dict)
    for x in kept:
        key = (x["base_model"], x.get("labeled_frac", "0.5"), x["alpha"], x["mode"],
               x.get("audit_mult", "1"), x.get("beta", "1.0"))
        g[key][x.get("seed", "0")] = x  # last write per seed wins (dedupe)

    fields = ["base_model", "labeled_frac", "alpha", "mode", "audit_mult", "beta", "n_seeds", "seeds",
              "known_acc_mean", "known_acc_sd", "contam_mean", "admitted_mean", "certcov_mean"]
    out = []
    for key, seedmap in sorted(g.items()):
        l = list(seedmap.values())
        ka = np.array([float(x["known_acc"]) for x in l])
        ct = [float(x["realized_contam"]) for x in l if x["realized_contam"] not in ("", "nan")]
        out.append({
            "base_model": key[0], "labeled_frac": key[1], "alpha": key[2], "mode": key[3],
            "audit_mult": key[4], "beta": key[5],
            "n_seeds": len(seedmap), "seeds": "|".join(sorted(seedmap)),
            "known_acc_mean": round(float(ka.mean()), 4),
            "known_acc_sd": round(float(ka.std(ddof=1)), 4) if len(ka) > 1 else 0.0,  # SAMPLE SD
            "contam_mean": round(float(np.mean(ct)), 4) if ct else "",
            "admitted_mean": int(np.mean([float(x["admitted_count"]) for x in l])),
            "certcov_mean": round(float(np.mean([float(x.get("certcov_alpha", 0) or 0) for x in l])), 4),
        })
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    print(f"\nsaved {OUT} ({len(out)} cells)")

    # gain summary vs none, per (base, seed-count)
    none = {(o["base_model"], o["labeled_frac"], o["alpha"]): o["known_acc_mean"]
            for o in out if o["mode"] == "none"}
    print("\ngain vs none (mean over distinct seeds):")
    for o in out:
        if o["mode"] in ("certified", "oracle"):
            b = none.get((o["base_model"], o["labeled_frac"], o["alpha"]))
            if b is not None:
                d = o["known_acc_mean"] - b
                print(f"  {o['base_model']:12s} lf={o['labeled_frac']} {o['mode']:9s} a={o['alpha']} "
                      f"audit={o['audit_mult']}x b={o['beta']}: {b:.4f} -> {o['known_acc_mean']:.4f} "
                      f"(Δ={d:+.4f}) n_seeds={o['n_seeds']} [{o['seeds']}] contam={o['contam_mean']}")


if __name__ == "__main__":
    main()
