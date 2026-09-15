"""Corrected exploratory T3 replay, preserving original inputs and decisions.

The archived original omitted the J factor for the simultaneous bounded
endpoint box. This repair uses risk tails .04/(JM), acceptance tails
.02/(JM) on each side, and the same .02 shared traffic-box budget.
Their union gives a joint .90 statement, not risk-only .96 coverage.
Simplex, m=1000 archived synthetic uniform traffic, family, alpha grid,
selector eligibility and ties remain unchanged. Original outputs are preserved.
Legacy GO thresholds are recomputed descriptively; this is not prospective
confirmation, actual traffic observation, or a same-protection performance claim.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inputs"))
from collections import defaultdict

import numpy as np
from scipy.stats import beta as _beta

from mixture import (
    multinomial_mixture_confidence_box, solve_coverage_infimum, solve_robust_ratio,
)

COUNTS = str(Path(__file__).resolve().parents[1] / "inputs/counts.csv.gz")
ALPHA_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
M_FAMILY = 12
TRAFFIC_M = 1000
DELTA_LAMBDA = 0.02
DELTA_R_BOUNDED = 0.04
DELTA_C_BOUNDED = 0.04          # split evenly into lower / upper
DELTA_R_SIMPLEX = 0.05
DELTA_C_SIMPLEX = 0.05
CONTRACT_ID = "fedcore-t3-corrected-exploratory"
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


_ARCHIVED_TRAFFIC = None

def traffic_counts(sid: str, J: int) -> list[int]:
    """Reuse the exact saved realization; seed replay differed in one local case."""
    global _ARCHIVED_TRAFFIC
    if _ARCHIVED_TRAFFIC is None:
        values = {}
        path = Path(__file__).resolve().parents[1] / "inputs/t3_cells_original.csv"
        expected = "fa0fef6032562e819e485a0766df9079af4bafd666248f5a9784165c1bfbab01"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Archived traffic source hash changed")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                vector = tuple(int(x) for x in row["traffic_counts"].split("|"))
                key = row["semantic_id"]
                if key in values and values[key] != vector:
                    raise ValueError("Traffic realization varies across alpha")
                values[key] = vector
        if len(values) != 450:
            raise ValueError("Incomplete original traffic roster")
        _ARCHIVED_TRAFFIC = values
    vector = _ARCHIVED_TRAFFIC[sid]
    if len(vector) != J or sum(vector) != TRAFFIC_M or min(vector) < 0:
        raise ValueError("Invalid archived traffic realization")
    return list(vector)


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
                rb = [cp_upper(k, a, DELTA_R_BOUNDED / (len(A) * M_FAMILY)) for a, k in zip(A, K)]
                alow = [cp_lower(a, nn, (DELTA_C_BOUNDED / 2) / (len(A) * M_FAMILY)) for a, nn in zip(A, n)]
                ahigh = [cp_upper(a, nn, (DELTA_C_BOUNDED / 2) / (len(A) * M_FAMILY)) for a, nn in zip(A, n)]
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
    raise RuntimeError("Use the immutable REPAIR_PLAN.json; do not relabel this retrospective repair as preregistered.")


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
            "simplex_selected": res["simplex"]["selected"],
            "simplex_risk_bound": res["simplex"]["risk"],
            "simplex_coverage_lcb": res["simplex"]["coverage"],
            "bounded_certified": res["bounded"]["certified"],
            "bounded_selected": res["bounded"]["selected"],
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

    # Descriptive application of archived thresholds after correcting the invalid endpoint allocation.
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
                "status": "CORRECTED_EXPLORATORY_REPLAY",
                "legacy_threshold_only": True, "prospective_confirmation": False,
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
