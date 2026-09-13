"""Equal-information comparator matrix for the full-simplex certification decision.

Holds fixed: the frozen predictor, the proposal-frozen candidate family, the
realized per-client audit counts, the family procedure (Holm over member-level
intersection-union p-values), the acceptance rule (Clopper-Pearson lower
confidence bound at delta_c / M) and the total confidence budget.

Varies exactly one thing: the per-client test used to build the member p-value
for the unsafe null H0: r_j >= alpha.

Arms
----
exact-cp         one-sided exact binomial (the published H branch)
wilson           one-sided score test (asymptotic; NOT finite-sample valid)
hoeffding        Hoeffding upper bound, inverted for a p-value
emp-bernstein    empirical-Bernstein upper bound, inverted for a p-value
betting          e-value from a uniform mixture over fixed stakes
betting-oracle   max over the stake grid; ANTI-CONSERVATIVE, not a valid test.
                 Reported only as an upper envelope on what any fixed-stake
                 prior could attain.

Consumes archived count triples only. Performs no training, no model inference
and no new audit draws.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
from scipy.stats import beta as _beta
from scipy.stats import binom as _binom
from scipy.stats import norm as _norm

DEFAULT_STAKES = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8])
DENSE_STAKES = np.unique(np.concatenate([np.geomspace(0.001, 0.98, 64), DEFAULT_STAKES]))
COUNT_COLUMNS = ("semantic_id", "dataset", "alpha", "candidate_index",
                 "score", "slot", "gamma", "client", "A", "K", "n")


# --------------------------------------------------------------------------- #
# per-client tests: p-value for H0: r_j >= alpha (reject on small p)
# --------------------------------------------------------------------------- #

def p_exact(K: np.ndarray, A: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    p = np.ones(len(K))
    m = A > 0
    p[m] = _binom.cdf(K[m], A[m], alpha[m])
    return p


def p_wilson(K: np.ndarray, A: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    p = np.ones(len(K))
    m = A > 0
    phat = K[m] / A[m]
    se = np.sqrt(alpha[m] * (1.0 - alpha[m]) / A[m])
    p[m] = _norm.cdf((phat - alpha[m]) / se)
    return p


def p_hoeffding(K: np.ndarray, A: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    p = np.ones(len(K))
    m = A > 0
    gap = alpha[m] - K[m] / A[m]
    p[m] = np.where(gap > 0, np.exp(-2.0 * A[m] * gap ** 2), 1.0)
    return p


def p_empirical_bernstein(K: np.ndarray, A: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    p = np.ones(len(K))
    m = A > 1
    n_acc = A[m].astype(float)
    phat = K[m] / n_acc
    var_hat = n_acc / (n_acc - 1.0) * phat * (1.0 - phat)
    gap = alpha[m] - phat
    lin = 7.0 / (3.0 * (n_acc - 1.0))
    sqrt_c = np.sqrt(2.0 * var_hat / n_acc)
    root = np.where(gap > 0,
                    (-sqrt_c + np.sqrt(sqrt_c ** 2 + 4.0 * lin * np.maximum(gap, 0.0)))
                    / (2.0 * lin), 0.0)
    p[m] = np.where(gap > 0, np.minimum(1.0, 2.0 * np.exp(-(root ** 2))), 1.0)
    return p


def _betting(K, A, alpha, stakes, oracle):
    p = np.ones(len(K))
    m = A > 0
    K_m = K[m].astype(float)[:, None]
    A_m = A[m].astype(float)[:, None]
    a_m = alpha[m][:, None]
    s = np.asarray(stakes)[None, :]
    log_e = K_m * np.log1p(-s) + (A_m - K_m) * np.log1p(s * a_m / (1.0 - a_m))
    peak = log_e.max(axis=1)
    if oracle:
        wealth = np.exp(peak)
    else:
        wealth = np.exp(peak) * np.mean(np.exp(log_e - peak[:, None]), axis=1)
    p[m] = np.minimum(1.0, 1.0 / wealth)
    return p


def p_betting(K, A, alpha, stakes=DEFAULT_STAKES):
    return _betting(K, A, alpha, stakes, oracle=False)


def p_betting_oracle(K, A, alpha, stakes=DENSE_STAKES):
    """NOT a valid test. Upper envelope on any fixed-stake prior."""
    return _betting(K, A, alpha, stakes, oracle=True)


ARMS: Dict[str, Callable[..., np.ndarray]] = {
    "exact-cp": p_exact,
    "wilson": p_wilson,
    "hoeffding": p_hoeffding,
    "emp-bernstein": p_empirical_bernstein,
    "betting": p_betting,
    "betting-oracle": p_betting_oracle,
}
INVALID_ARMS = frozenset({"wilson", "betting-oracle"})


# --------------------------------------------------------------------------- #
# family procedure (identical across arms)
# --------------------------------------------------------------------------- #

def acceptance_lcb(A: np.ndarray, n: np.ndarray, eps: float) -> np.ndarray:
    out = np.zeros(len(A))
    m = A > 0
    out[m] = _beta.ppf(eps, A[m], n[m] - A[m] + 1.0)
    return out


def holm_reject(pvals: Sequence[float], level: float, family_size: int) -> List[bool]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    rejected = [False] * len(pvals)
    for rank, idx in enumerate(order):
        if pvals[idx] <= level / (family_size - rank):
            rejected[idx] = True
        else:
            break
    return rejected


def load_counts(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    cols = {c: [] for c in COUNT_COLUMNS}
    with opener(path, "rt") as handle:
        for row in csv.DictReader(handle):
            for c in COUNT_COLUMNS:
                cols[c].append(row[c])
    return {
        "semantic_id": cols["semantic_id"],
        "dataset": cols["dataset"],
        "alpha": np.array(cols["alpha"], float),
        "candidate_index": np.array(cols["candidate_index"], int),
        "gamma": np.array(cols["gamma"], float),
        "tie_name": ["msp" if s == "native" else sc
                     for s, sc in zip(cols["slot"], cols["score"])],
        "A": np.array(cols["A"], np.int64),
        "K": np.array(cols["K"], np.int64),
        "n": np.array(cols["n"], np.int64),
    }


def run(counts: dict, arms: Iterable[str], delta_r: float, delta_c: float,
        family_size: int) -> List[dict]:
    A, K, alpha = counts["A"], counts["K"], counts["alpha"]
    arms = list(arms)
    pvals = {name: ARMS[name](K, A, alpha) for name in arms}
    lcb = acceptance_lcb(A, counts["n"], delta_c / family_size)

    cells: dict = {}
    for i in range(len(A)):
        key = (counts["semantic_id"][i], counts["dataset"][i], float(alpha[i]))
        members = cells.setdefault(key, {})
        idx = int(counts["candidate_index"][i])
        entry = members.get(idx)
        if entry is None:
            members[idx] = {"gamma": float(counts["gamma"][i]),
                            "tie": counts["tie_name"][i],
                            "lcb": float(lcb[i]),
                            "p": {name: float(pvals[name][i]) for name in arms}}
        else:
            entry["lcb"] = min(entry["lcb"], float(lcb[i]))
            for name in arms:  # intersection-union: worst client
                entry["p"][name] = max(entry["p"][name], float(pvals[name][i]))

    rows = []
    for (sid, dataset, a), members in cells.items():
        idxs = sorted(members)
        row = {"semantic_id": sid, "dataset": dataset, "alpha": a}
        for name in arms:
            rejected = holm_reject([members[i]["p"][name] for i in idxs],
                                   delta_r, family_size)
            passing = [i for j, i in enumerate(idxs)
                       if rejected[j] and members[i]["lcb"] > 0.0]
            if passing:
                # predeclared tie rule: max acceptance LCB, then smaller gamma,
                # then theorem-facing score name, then candidate index
                best = min(passing, key=lambda i: (-members[i]["lcb"],
                                                   members[i]["gamma"],
                                                   members[i]["tie"], i))
                row[name] = members[best]["lcb"]
            else:
                row[name] = 0.0
        rows.append(row)
    return rows


def summarize(rows: List[dict], arms: Sequence[str]) -> List[dict]:
    out = []
    for a in sorted({r["alpha"] for r in rows}):
        at_alpha = [r for r in rows if r["alpha"] == a]
        scopes = ["ALL"] + sorted({r["dataset"] for r in at_alpha})
        for scope in scopes:
            subset = at_alpha if scope == "ALL" else [r for r in at_alpha
                                                      if r["dataset"] == scope]
            for name in arms:
                v = np.array([r[name] for r in subset])
                out.append({"alpha": a, "arm": name, "scope": scope,
                            "N": len(v), "certified": int((v > 0).sum()),
                            "ECA": float(v.mean()),
                            "finite_sample_valid": name not in INVALID_ARMS})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", required=True, help="archived candidate count CSV(.gz)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--arms", nargs="+", default=["exact-cp", "wilson", "hoeffding",
                                                  "emp-bernstein", "betting"])
    ap.add_argument("--delta-r", type=float, default=0.05)
    ap.add_argument("--delta-c", type=float, default=0.05)
    ap.add_argument("--family-size", type=int, default=12)
    ap.add_argument("--report-alpha", type=float, nargs="+", default=[0.10, 0.20])
    args = ap.parse_args(argv)

    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        ap.error("unknown arm(s): %s; choose from %s" % (unknown, sorted(ARMS)))

    counts = load_counts(args.counts)
    rows = run(counts, args.arms, args.delta_r, args.delta_c, args.family_size)
    summary = summarize(rows, args.arms)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "per_cell_eca.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["semantic_id", "dataset", "alpha"] + args.arms)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["alpha", "arm", "scope", "N",
                                           "certified", "ECA", "finite_sample_valid"])
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(args.out, "contract.json"), "w") as fh:
        json.dump({"counts": os.path.basename(args.counts), "arms": args.arms,
                   "delta_r": args.delta_r, "delta_c": args.delta_c,
                   "family_size": args.family_size,
                   "stakes_betting": DEFAULT_STAKES.tolist(),
                   "stakes_oracle_grid": len(DENSE_STAKES),
                   "family_procedure": "holm over member-level max-p IUT",
                   "acceptance_rule": "Clopper-Pearson LCB at delta_c / M",
                   "invalid_arms": sorted(INVALID_ARMS)}, fh, indent=2)

    for a in args.report_alpha:
        at = [s for s in summary if s["scope"] == "ALL" and abs(s["alpha"] - a) < 1e-9]
        if not at:
            continue
        base = next(s for s in at if s["arm"] == "exact-cp")
        print("\nalpha = %.2f" % a)
        print("  %-16s%16s%12s%9s%12s  %s"
              % ("arm", "certified", "ECA", "d_cert", "d_ECA(pp)", "valid"))
        for s in at:
            print("  %-16s%16s%12.6f%9d%12.4f  %s"
                  % (s["arm"], "%d/%d" % (s["certified"], s["N"]), s["ECA"],
                     s["certified"] - base["certified"],
                     (s["ECA"] - base["ECA"]) * 100.0,
                     "yes" if s["finite_sample_valid"] else "NO"))
    print("\nwritten:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
