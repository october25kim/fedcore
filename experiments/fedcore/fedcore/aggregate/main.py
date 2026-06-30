"""P1: aggregate CertifiedCoverage results across seeds from exported logits.

Recomputes the best-gamma headline (per-client G=J and worst-group G=2) for every
exported ``runs/*_logits.npz``, parses the (dataset, d, noise, seed) tag from the
filename, and aggregates mean +/- std across seeds into ``runs/agg_main.csv``. No
single-seed numbers leak into the final tables: cells are reported as mean+/-std
with the seed count.

Run: ``python experiments/fedcore/aggregate.py``  (CPU, no torch)
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np

from fedcore.certify import certify_best_gamma, certify_best_gamma_grouped
from fedcore.atomic_io import atomic_write_csv
from fedcore.grouping import _group_map, _repartition
from fedcore.scores import scored_views

ALPHA, DELTA = 0.10, 0.10
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
SCORES = ("msp", "neg_entropy", "margin", "energy")
MARGIN = 0.01
CERT_FRAC = 0.5   # matches the headline claim (gate cert_frac); repartition pooled trusted

def parse_tag(fn: str):
    """Robustly parse (dataset, backbone, d, noise, seed) from a logits filename,
    tolerant to field order and a missing noise token (clean defaults to none0.0)."""
    ds = "cifar100" if "cifar100" in fn else "cifar10"
    bb = ("resnet18gn" if "resnet18gn" in fn else
          "resnet18" if "resnet18" in fn else "simplecnn")
    md = re.search(r"_d([0-9.]+)", fn)
    mn = re.search(r"(none|symmetric|asymmetric)([0-9.]+)", fn)
    ms = re.search(r"seed(\d+)", fn)
    if not (md and ms):
        return None
    noise = f"{mn.group(1)}{mn.group(2)}" if mn else "none0.0"
    return {"ds": ds, "bb": bb, "d": md.group(1), "noise": noise, "seed": ms.group(1)}


def _headline(npz, G, scores=("msp",), alpha=ALPHA):
    """CertCov@alpha at grouping G (G=None -> per-client).

    MAIN uses the FIXED score MSP (scores=("msp",)) -> no selection bias. Pass the
    full score tuple for the optimistic best-of-N appendix variant.
    """
    d = np.load(npz)
    n_clients = int(d["cert_client"].max()) + 1
    pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}
    parts = _repartition(pool, CERT_FRAC, 0.2, seed=0)
    best = None
    for s in scores:
        views = {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                                  parts[fn]["client"], [s])[s] for fn in ("prop", "cert", "test")}
        if G is None or G >= n_clients:
            r = certify_best_gamma(views["prop"], views["cert"], views["test"],
                                   score_name=s, gammas=GAMMAS, alpha=alpha, delta=DELTA,
                                   n_clients=n_clients, dirichlet_alpha=float("nan"),
                                   Lambda="box", box=0.15, seed=0, margin=MARGIN)
        else:
            gmap = np.array([c * G // n_clients for c in range(n_clients)])
            r = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"],
                                           score_name=s, group_map=gmap, G=G, gammas=GAMMAS,
                                           alpha=alpha, delta=DELTA, Lambda="box", box=0.15,
                                           seed=0, margin=MARGIN)
        cov = r["cert_coverage_lcb"] if r["certified"] else 0.0
        cand = (cov, r["cert_risk_ucb"], r["test_risk"], s)
        if best is None or cand[0] > best[0]:
            best = cand
    return best  # (cov, cert_ucb, test_risk, score)


def _median_certified_ucb(ucbs, covs):
    """Median cert_ucb among CERTIFIED seeds (cov>0); inf if none certified."""
    sel = [u for u, c in zip(ucbs, covs) if c > 0 and np.isfinite(u)]
    return float(np.median(sel)) if sel else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help="target selective-risk level (default 0.10; use 0.20 for the headline)")
    ap.add_argument("--out", default=None,
                    help="output csv (default runs/agg_main.csv, or runs/agg_alpha<NN>.csv for non-0.10)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="restrict to these seeds (default: all exported)")
    args = ap.parse_args()
    alpha = args.alpha

    base = "" if glob.glob("runs/*_logits.npz") else "../../"
    files = sorted(glob.glob(base + "runs/*_logits.npz"))
    cells = defaultdict(list)  # (ds,bb,d,noise) -> list of per-seed dicts
    for f in files:
        m = parse_tag(os.path.basename(f))
        if not m:
            continue
        if args.seeds is not None and int(m["seed"]) not in args.seeds:
            continue
        key = (m["ds"], m["bb"], m["d"], m["noise"])
        covJ, ucbJ, riskJ, _ = _headline(f, None, alpha=alpha)
        cov2, ucb2, risk2, _ = _headline(f, 2, alpha=alpha)
        cells[key].append({"seed": int(m["seed"]),
                           "covJ": covJ, "ucbJ": ucbJ, "cov2": cov2, "ucb2": ucb2,
                           "risk": riskJ})

    if args.out is not None:
        out = base + args.out if not os.path.isabs(args.out) else args.out
    elif abs(alpha - 0.10) < 1e-9:
        out = base + "runs/agg_main.csv"
    else:
        out = base + f"runs/agg_alpha{int(round(alpha * 100)):02d}.csv"
    fields = ["dataset", "backbone", "d", "noise", "n_seeds",
              "CertCovGJ_mean", "CertCovGJ_std", "certucbGJ_median",
              "CertCovG2_mean", "CertCovG2_std", "certucbG2_median",
              "n_pass_G2", "test_risk_mean"]
    print(f"alpha={alpha:.2f}  delta={DELTA:.2f}  cert_frac={CERT_FRAC}  score=msp(fixed)  margin={MARGIN}")
    print(f"{'cell':>42} {'seeds':>5} {'CertCov@a(G=J)':>18} {'CertCov@a(G=2)':>18} {'G2pass':>8}")
    print("-" * 96)
    agg_rows = []
    for key, lst in sorted(cells.items()):
        ds, bb, dd, noise = key
        covJ = np.array([x["covJ"] for x in lst]); cov2 = np.array([x["cov2"] for x in lst])
        ucbJ = [x["ucbJ"] for x in lst]; ucb2 = [x["ucb2"] for x in lst]
        np2 = int(sum(c > 0 for c in cov2))
        row = {"dataset": ds, "backbone": bb, "d": dd, "noise": noise, "n_seeds": len(lst),
               "CertCovGJ_mean": round(covJ.mean(), 4), "CertCovGJ_std": round(covJ.std(), 4),
               "certucbGJ_median": round(_median_certified_ucb(ucbJ, covJ), 4),
               "CertCovG2_mean": round(cov2.mean(), 4), "CertCovG2_std": round(cov2.std(), 4),
               "certucbG2_median": round(_median_certified_ucb(ucb2, cov2), 4),
               "n_pass_G2": np2, "test_risk_mean": round(np.mean([x["risk"] for x in lst]), 4)}
        agg_rows.append(row)
        label = f"{ds}/{bb}/d{dd}/{noise}"
        print(f"{label:>42} {len(lst):>5} "
              f"{covJ.mean():>9.3f}+/-{covJ.std():<6.3f} "
              f"{cov2.mean():>9.3f}+/-{cov2.std():<6.3f} {np2:>4}/{len(lst)}")
        per_seed = ", ".join(f"s{x['seed']}={x['cov2']:.3f}"
                             for x in sorted(lst, key=lambda x: x["seed"]))
        print(f"{'':>42} per-seed G=2: {per_seed}")
    atomic_write_csv(out, fields, agg_rows)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
