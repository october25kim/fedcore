"""T3 -- bounded-Lambda arm extended to the full wr-v3 cell set (count-only).

Why. The bounded-mixture certificate (Theorem 1') has so far been measured only
on the 100 Office-Home cells, and only as an exploratory sweep. The test axis
runs over all 450 conditions while the target axis does not, which a referee
reads as an unsupported headline. T3 puts the bounded-Lambda arm on the SAME
450 conditions as the full-simplex arm, under a budget-matched comparison, at
every alpha on the grid, using nothing but the already-released count records.

Contract.
  * Estimand and counts are inherited verbatim from wr-v3: the same 12-member
    proposal-frozen family, the same realized (A, K, n) per client and alpha,
    the same Bonferroni split delta/M across family members.
  * BOUNDED arm:  delta_lambda = 0.02 for the mixture set,
                  delta_r = 0.04 for per-client risk,
                  delta_c = 0.04 for acceptance, split 0.02 lower / 0.02 upper.
  * SIMPLEX arm (budget-matched baseline): delta_r = 0.05, delta_c = 0.05, no
    mixture or acceptance-upper spend. Both arms therefore spend 0.10 in total,
    so any difference is attributable to WHERE the budget is spent, not how much.
  * Mixture observation: the deployment mixture declared by the wr-v3 campaign
    is uniform over clients. T3 draws m = 1000 traffic observations from that
    declared mixture with a per-cell pre-registered seed and builds the
    simultaneous multinomial confidence box at delta_lambda. No label, score or
    logit is read from any fold -- the traffic observation is client identities
    only, exactly as `build_identity_only_traffic` specifies. (The diagnostic
    fold's realized client composition is exactly uniform, 514 per client on
    CIFAR, so drawing from the declared mixture and subsampling that fold's
    identities coincide; the declared-mixture route is used because it keeps
    the diagnostic fold untouched.)
  * Certification: BOUNDED uses sup over the mixture box and the acceptance box
    of the accepted-risk ratio, and the coverage infimum over the box; SIMPLEX
    uses max_j risk UCB and min_j acceptance LCB. Both require a strictly
    positive coverage bound.

Pre-registered decision rule (sealed before the run, see PREREGISTRATION.json).

    python -m fedcore.experiments.t3_bounded_lambda seal
    python -m fedcore.experiments.t3_bounded_lambda run \
        --output-dir results/t3_bounded_lambda/primary_run
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import beta as _beta

from fedcore.mixture import (
    multinomial_mixture_confidence_box, solve_coverage_infimum, solve_robust_ratio,
)

COUNTS = "results/theorem_aligned_wr_450_v3/primary_run/primary_candidate_counts.csv.gz"
ALPHA_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
M_FAMILY = 12
TRAFFIC_M = 1000
DELTA_LAMBDA = 0.02
DELTA_R_BOUNDED = 0.04
DELTA_C_BOUNDED = 0.04          # split evenly into lower / upper
DELTA_R_SIMPLEX = 0.05
DELTA_C_SIMPLEX = 0.05
CONTRACT_ID = "fedcore-t3-bounded-lambda"
MASTER_SEED = 20260913

# Pre-registered thresholds (Delta = bounded minus simplex, at every alpha).
GO_MIN_CELLS = 25
GO_MIN_PP = 5.0
GO_MIN_DATASETS = 2
KILL_MAX_PP = 1.0
KILL_MIN_DATASETS = 3


def cp_upper(k: int, n: int, eps: float) -> float:
    if n <= 0 or k >= n or eps <= 0.0:
        return 1.0
    return float(_beta.isf(eps, k + 1, n - k))


def cp_lower(k: int, n: int, eps: float) -> float:
    if n <= 0 or k <= 0:
        return 0.0
    return float(_beta.ppf(eps, k, n - k + 1))


def family_tie_key(row) -> tuple:
    tie = "msp" if row["slot"] == "native" else row["slot"]
    return (float(row["gamma"]), tie, int(row["candidate_index"]))


def traffic_counts(sid: str, J: int) -> list[int]:
    """m=1000 observations of the DECLARED uniform deployment mixture."""
    seed = int.from_bytes(hashlib.sha256(f"{sid}:traffic".encode()).digest()[:4], "big")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([MASTER_SEED, seed])))
    return [int(v) for v in rng.multinomial(TRAFFIC_M, np.full(J, 1.0 / J))]


def load_counts(path: str):
    """-> {(sid, alpha): {'meta': {...}, 'members': {m: {'A':[],'K':[],'n':[],...}}}}"""
    cells = defaultdict(lambda: {"meta": {}, "members": defaultdict(
        lambda: {"A": [], "K": [], "n": [], "feas": 1, "slot": "", "gamma": 0.0,
                 "score": "", "candidate_index": 0})})
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["semantic_id"], float(r["alpha"]))
            cell = cells[key]
            cell["meta"] = {"dataset": r["dataset"], "condition": r["condition"]}
            m = cell["members"][int(r["candidate_index"])]
            m["A"].append(int(r["A"]))
            m["K"].append(int(r["K"]))
            m["n"].append(int(r["n"]))
            m["feas"] = int(r["proposal_feasible"])
            m["slot"], m["gamma"] = r["slot"], float(r["gamma"])
            m["score"], m["candidate_index"] = r["score"], int(r["candidate_index"])
    return cells


def evaluate_cell(members, alpha, box):
    """Return per-arm (certified, selected_index, risk_bound, coverage_bound)."""
    out = {}
    for arm in ("simplex", "bounded"):
        cert_idx, best = [], {}
        for mi, m in sorted(members.items()):
            if not m["feas"]:
                continue
            A, K, n = m["A"], m["K"], m["n"]
            if arm == "simplex":
                rb = [cp_upper(k, a, DELTA_R_SIMPLEX / M_FAMILY) for a, k in zip(A, K)]
                cov = min(cp_lower(a, nn, DELTA_C_SIMPLEX / M_FAMILY) for a, nn in zip(A, n))
                risk = max(rb)
                feasible = True
            else:
                rb = [cp_upper(k, a, DELTA_R_BOUNDED / M_FAMILY) for a, k in zip(A, K)]
                alow = [cp_lower(a, nn, (DELTA_C_BOUNDED / 2) / M_FAMILY) for a, nn in zip(A, n)]
                ahigh = [cp_upper(a, nn, (DELTA_C_BOUNDED / 2) / M_FAMILY) for a, nn in zip(A, n)]
                res = solve_robust_ratio(rb, alow, ahigh, box.mixture)
                risk, feasible = float(res.value), bool(res.feasible)
                cov = float(solve_coverage_infimum(alow, box.mixture).value)
            ok = feasible and (risk <= alpha) and (cov > 0.0)
            if ok:
                cert_idx.append(mi)
                best[mi] = (cov, risk)
        if not cert_idx:
            out[arm] = {"certified": 0, "selected": None, "risk": None, "coverage": 0.0}
            continue
        pick = min(cert_idx, key=lambda i: (-best[i][0],) + family_tie_key(members[i]))
        out[arm] = {"certified": 1, "selected": pick,
                    "risk": best[pick][1], "coverage": best[pick][0]}
    return out


def seal(args) -> None:
    os.makedirs(args.governing, exist_ok=False)
    prereg = {
        "contract_id": CONTRACT_ID, "created_utc": "2026-09-13",
        "question": "Does the bounded-mixture certificate (Theorem 1') certify "
                    "materially more conditions than the full simplex, at MATCHED "
                    "total failure budget, across all wr-v3 datasets?",
        "motivation": "The bounded-Lambda arm was previously measured only on the "
                      "100 Office-Home cells and only as an exploratory sweep, "
                      "while the test axis covers all 450 conditions.",
        "inputs": {"counts": COUNTS, "counts_sha256": sha256_file(COUNTS),
                   "cells": 450, "datasets": ["cifar10", "cifar100", "pathmnist",
                                              "officehome"]},
        "budgets": {
            "bounded": {"delta_lambda": DELTA_LAMBDA, "delta_r": DELTA_R_BOUNDED,
                        "delta_c": DELTA_C_BOUNDED,
                        "delta_c_split": "0.02 lower / 0.02 upper",
                        "total": DELTA_LAMBDA + DELTA_R_BOUNDED + DELTA_C_BOUNDED},
            "simplex": {"delta_r": DELTA_R_SIMPLEX, "delta_c": DELTA_C_SIMPLEX,
                        "total": DELTA_R_SIMPLEX + DELTA_C_SIMPLEX},
            "budget_matched": True,
            "family_bonferroni": "delta/M with M=12, identical in both arms",
        },
        "mixture_observation": {
            "m": TRAFFIC_M, "declared_mixture": "uniform over clients",
            "set": "simultaneous multinomial confidence box at delta_lambda",
            "seed_law": "PCG64(SeedSequence([20260913, sha256(sid+':traffic')[:4]]))",
            "label_free": "client identities only; no label, score or logit from "
                          "any fold is read, and the diagnostic fold is untouched",
        },
        "alpha_grid": list(ALPHA_GRID),
        "decision_rule": {
            "metric": "Delta = certified(bounded) - certified(simplex), per dataset "
                      "and per alpha; pp = percentage-point change in certified rate",
            "GO": f">= +{GO_MIN_CELLS} conditions AND >= +{GO_MIN_PP} pp in at "
                  f"least {GO_MIN_DATASETS} of 4 datasets",
            "CONDITIONAL": "+1 to +5 pp, or gains concentrated in a single dataset",
            "KILL": f"< +{KILL_MAX_PP} pp in at least {KILL_MIN_DATASETS} datasets "
                    "-> pivot to a pure feasibility paper",
        },
        "reporting": {"every_cell": True, "primary_alpha_for_headline": 0.20},
    }
    p = os.path.join(args.governing, "PREREGISTRATION.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(prereg, f, indent=2, sort_keys=True)
    with open(os.path.join(args.governing, "SEALED.sha256"), "w") as f:
        f.write(f"{sha256_file(p)}  PREREGISTRATION.json\n")
    print(f"sealed -> {args.governing}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(args) -> None:
    os.makedirs(args.output_dir, exist_ok=False)
    cells = load_counts(COUNTS)
    boxes = {}
    rows = []
    for (sid, alpha), cell in sorted(cells.items()):
        members = cell["members"]
        J = len(next(iter(members.values()))["A"])
        if sid not in boxes:
            boxes[sid] = multinomial_mixture_confidence_box(
                traffic_counts(sid, J), DELTA_LAMBDA)
        res = evaluate_cell(members, alpha, boxes[sid])
        rows.append({
            "semantic_id": sid, "dataset": cell["meta"]["dataset"],
            "condition": cell["meta"]["condition"], "alpha": alpha, "J": J,
            "traffic_counts": "|".join(str(c) for c in boxes[sid].counts),
            "simplex_certified": res["simplex"]["certified"],
            "simplex_risk_bound": res["simplex"]["risk"],
            "simplex_coverage_lcb": res["simplex"]["coverage"],
            "bounded_certified": res["bounded"]["certified"],
            "bounded_risk_bound": res["bounded"]["risk"],
            "bounded_coverage_lcb": res["bounded"]["coverage"],
            "gain": res["bounded"]["certified"] - res["simplex"]["certified"],
        })
    with open(os.path.join(args.output_dir, "t3_cells.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    agg = defaultdict(lambda: {"n": 0, "s": 0, "b": 0})
    for r in rows:
        for key in ((r["dataset"], r["alpha"]), (r["dataset"], "all"),
                    ("ALL", r["alpha"]), ("ALL", "all")):
            a = agg[key]
            a["n"] += 1
            a["s"] += r["simplex_certified"]
            a["b"] += r["bounded_certified"]
    with open(os.path.join(args.output_dir, "t3_summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "alpha", "cells", "simplex_certified",
                    "bounded_certified", "delta_cells", "simplex_pct",
                    "bounded_pct", "delta_pp"])
        for k in sorted(agg, key=lambda x: (str(x[0]), str(x[1]))):
            a = agg[k]
            sp, bp = 100.0 * a["s"] / a["n"], 100.0 * a["b"] / a["n"]
            w.writerow([k[0], k[1], a["n"], a["s"], a["b"], a["b"] - a["s"],
                        f"{sp:.2f}", f"{bp:.2f}", f"{bp - sp:.2f}"])

    # Pre-registered decision
    datasets = ["cifar10", "cifar100", "pathmnist", "officehome"]
    per_ds = {d: agg[(d, "all")] for d in datasets}
    pp = {d: 100.0 * (v["b"] - v["s"]) / v["n"] for d, v in per_ds.items()}
    dcells = {d: v["b"] - v["s"] for d, v in per_ds.items()}
    total_delta = agg[("ALL", "all")]["b"] - agg[("ALL", "all")]["s"]
    n_go = sum(1 for d in datasets if pp[d] >= GO_MIN_PP)
    n_kill = sum(1 for d in datasets if pp[d] < KILL_MAX_PP)
    if total_delta >= GO_MIN_CELLS and n_go >= GO_MIN_DATASETS:
        verdict = "GO"
    elif n_kill >= KILL_MIN_DATASETS:
        verdict = "KILL"
    else:
        verdict = "CONDITIONAL"
    decision = {"contract_id": CONTRACT_ID, "verdict": verdict,
                "total_delta_cells": total_delta,
                "per_dataset_delta_cells": dcells,
                "per_dataset_delta_pp": {d: round(v, 2) for d, v in pp.items()},
                "datasets_meeting_go_pp": n_go, "datasets_below_kill_pp": n_kill,
                "thresholds": {"GO_MIN_CELLS": GO_MIN_CELLS, "GO_MIN_PP": GO_MIN_PP,
                               "GO_MIN_DATASETS": GO_MIN_DATASETS,
                               "KILL_MAX_PP": KILL_MAX_PP,
                               "KILL_MIN_DATASETS": KILL_MIN_DATASETS}}
    with open(os.path.join(args.output_dir, "DECISION.json"), "w") as f:
        json.dump(decision, f, indent=2, sort_keys=True)
    lines = [f"{sha256_file(os.path.join(args.output_dir, x))}  {x}"
             for x in sorted(os.listdir(args.output_dir)) if x != "SHA256SUMS"]
    with open(os.path.join(args.output_dir, "SHA256SUMS"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(decision, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seal"); s.add_argument("--governing", default="results/t3_bounded_lambda/governing"); s.set_defaults(func=seal)
    r = sub.add_parser("run"); r.add_argument("--output-dir", dest="output_dir", required=True); r.set_defaults(func=run)
    a = ap.parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
