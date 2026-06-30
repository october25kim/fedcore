"""Copy-paste handoff summary: seed GATE (cifar10 d=5 ResNet) + covtype frontier.

No retraining -- reads exported npz only. Reuses certify_best_gamma_grouped and the
feasibility-lever repartition (no reimplementation drift). Missing seeds/files are
printed as 'pending' and the aggregate is computed over what exists.

Run: ``python experiments/fedcore/make_handoff.py [--cert_frac 0.5]``
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

from certify import certify_best_gamma_grouped
from exp_feasibility_lever import _group_map, _repartition
from scores import scored_views

ALPHA, DELTA = 0.10, 0.10
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
MARGIN = 0.01
SCORE = "msp"   # fixed (matches the feasibility staircase; avoids score selection)
BASE = "" if glob.glob("runs/*.npz") else "../../"


def _views_from_parts(parts, score):
    return {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                             parts[fn]["client"], [score])[score] for fn in ("prop", "cert", "test")}


def _gate_one(npz, cert_frac, G, score=SCORE):
    d = np.load(npz)
    n_clients = int(d["cert_client"].max()) + 1
    pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}
    parts = _repartition(pool, cert_frac, 0.2, seed=0)
    views = _views_from_parts(parts, score)
    gmap = _group_map(n_clients, G)
    r = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"], score_name=score,
                                   group_map=gmap, G=G, gammas=GAMMAS, alpha=ALPHA, delta=DELTA,
                                   Lambda="box", box=0.15, seed=0, margin=MARGIN)
    cov = r["cert_coverage_lcb"] if r["certified"] else 0.0
    return {"ucb": r["cert_risk_ucb"], "cov": cov, "rhat": r["test_risk"],
            "per_group_n": r["cert_n"] / G}


def covtype_frontier(npz, cert_frac):
    d = np.load(npz)
    n_clients = int(d["cert_client"].max()) + 1
    pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}
    parts = _repartition(pool, cert_frac, 0.2, seed=0)
    rows = []
    for a in (0.10, 0.15, 0.20, 0.25, 0.30):
        best_cov, best_ucb, rhat = 0.0, np.inf, np.nan
        for score in ("msp", "energy", "neg_entropy", "margin"):
            views = _views_from_parts(parts, score)
            for G in (1, 2, 3):
                gmap = _group_map(n_clients, G)
                r = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"],
                                               score_name=score, group_map=gmap, G=G, gammas=GAMMAS,
                                               alpha=a, delta=DELTA, Lambda="box", box=0.15,
                                               seed=0, margin=MARGIN)
                if np.isfinite(r["cert_risk_ucb"]):
                    best_ucb = min(best_ucb, r["cert_risk_ucb"])
                if r["certified"] and r["cert_coverage_lcb"] > best_cov:
                    best_cov, rhat = r["cert_coverage_lcb"], r["test_risk"]
        rows.append({"alpha": a, "cov": best_cov, "ucb": best_ucb, "rhat": rhat})
    return rows


def _fmt(x):
    return "pending" if x is None else (f"{x:.3f}" if np.isfinite(x) else "inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert_frac", type=float, default=0.5,
                    help="crossover cert fraction that produced the seed0 positive")
    args = ap.parse_args()
    cf = args.cert_frac

    print("files read:")
    seed_res, gate_csv = {}, []
    SEEDS = list(range(10))
    for s in SEEDS:
        p = BASE + f"runs/cifar10_d5_resnet18_seed{s}_logits.npz"
        if glob.glob(p):
            print(f"  {p}")
            seed_res[s] = {G: _gate_one(p, cf, G) for G in (2, 3)}
            for G in (2, 3):
                r = seed_res[s][G]
                gate_csv.append([s, G, round(r["ucb"], 3), round(r["cov"], 3),
                                 round(r["rhat"], 3), round(r["per_group_n"], 1)])

    cov_npz = None
    for cand in ("runs/covtype_logits.npz", "runs/tabular_logits.npz"):
        if glob.glob(BASE + cand):
            cov_npz = BASE + cand; print(f"  {cov_npz}"); break
    cov_rows = covtype_frontier(cov_npz, cf) if cov_npz else None

    # aggregate
    def agg(G, key):
        vals = [seed_res[s][G][key] for s in seed_res if np.isfinite(seed_res[s][G][key])]
        return (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)

    n_pass = sum(1 for s in seed_res
                 if any(seed_res[s][G]["ucb"] <= ALPHA and seed_res[s][G]["cov"] > 0 for G in (2, 3)))
    N = len(seed_res)
    per_group = np.mean([seed_res[s][2]["per_group_n"] for s in seed_res]) if seed_res else float("nan")

    # ---- exact handoff block ----
    print()
    print(f"GATE  (cifar10 d=5, ResNet, cert_frac={cf})")
    for s in sorted(seed_res):
        g2, g3 = seed_res[s][2], seed_res[s][3]
        print(f"seed{s}: G2 ucb={_fmt(g2['ucb'])} cov={_fmt(g2['cov'])} | "
              f"G3 ucb={_fmt(g3['ucb'])} cov={_fmt(g3['cov'])} | r_hat={_fmt(g2['rhat'])}")
    a2u, a2us = agg(2, "ucb"); a2c, a2cs = agg(2, "cov")
    a3u, a3us = agg(3, "ucb"); a3c, a3cs = agg(3, "cov")
    print(f"agg : G2 ucb={_fmt(a2u)}+/-{_fmt(a2us)} cov={_fmt(a2c)}+/-{_fmt(a2cs)} ; "
          f"G3 ucb={_fmt(a3u)}+/-{_fmt(a3us)} cov={_fmt(a3c)}+/-{_fmt(a3cs)}")
    import math
    print(f"#seeds with (G2 or G3) ucb<=0.10 & cov>0 :  {n_pass}/{N}")
    print(f"per-group cert_n at crossover ~ {_fmt(per_group)}")
    thresh = math.ceil(2 / 3 * N) if N else 1
    verdict = (f"PROVISIONAL ({n_pass}/{N} seeds)" if N < 3 else
               "PASS" if n_pass >= thresh else "FAIL")
    print(f"GATE: {verdict}  (>= {thresh}/{N} needed)")
    print()
    print("covtype alpha-frontier (post-hoc, no GPU)")
    if cov_rows:
        seg = " | ".join(f"{('a='+format(r['alpha'],'.2f')) if i==0 else format(r['alpha'],'.2f')} "
                         f"cov={_fmt(r['cov'])} ucb={_fmt(r['ucb'])}" for i, r in enumerate(cov_rows))
        print(seg)
        rhats = [r["rhat"] for r in cov_rows if np.isfinite(r["rhat"])]
        first = next((r["alpha"] for r in cov_rows if r["cov"] > 0), None)
        print(f"covtype r_hat ~ {_fmt(rhats[0] if rhats else None)} ; "
              f"first non-vacuous alpha = {_fmt(first)}")
    else:
        print("  (pending: no covtype npz)")
    print()
    print("GPU budget remaining ~ ____ hrs   (fill manually)")

    # ---- csvs ----
    os.makedirs(BASE + "runs", exist_ok=True)
    with open(BASE + "runs/handoff_gate.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seed", "G", "cert_ucb", "CertCov@0.1", "r_hat", "per_group_n"])
        w.writerows(gate_csv)
    if cov_rows:
        with open(BASE + "runs/handoff_covtype.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["alpha", "CertCov", "cert_ucb", "r_hat"])
            for r in cov_rows:
                w.writerow([r["alpha"], round(r["cov"], 3),
                            round(r["ucb"], 3) if np.isfinite(r["ucb"]) else "inf",
                            round(r["rhat"], 3) if np.isfinite(r["rhat"]) else "nan"])
    print("\nsaved runs/handoff_gate.csv" + (", runs/handoff_covtype.csv" if cov_rows else ""))


if __name__ == "__main__":
    main()
