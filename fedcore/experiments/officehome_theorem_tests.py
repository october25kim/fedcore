"""Traffic-derived Lambda_hat theorem suite (Office-Home, §6.3).

Runs the required validity studies for the traffic-derived deployment-mixture
set and emits the frozen artifacts:

* ``results/officehome/theorem_tests/traffic_set_coverage.csv`` -- Monte-Carlo
  coverage of ``lambda_star in Lambda_hat`` across ``m`` and mixture shapes
  (including boundary mixtures).
* ``results/officehome/theorem_tests/combined_validity.csv`` -- one row per
  theorem property with PASS/FAIL.
* ``docs/agent/data_derived_lambda_theorem.md`` -- the human-readable statement.

Properties checked (§6.3):
1. MC coverage >= 1 - delta_lambda (incl. boundary mixtures).
2. Exact coordinate-tail accounting (union bound == delta_lambda).
3. Label/score-free construction.
4. Deterministic replay.
5. Aggregate width decreases with m.
6. Fail-closed on identity overlap.
7. Full-simplex fallback on empty traffic.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Sequence

import numpy as np
from scipy.stats import beta as _beta

from fedcore.officehome_traffic_lambda import (
    DELTA_LAMBDA,
    M_GRID,
    M_PRIMARY,
    N_DOMAIN_CLIENTS,
    TrafficLambdaError,
    assert_traffic_identity_disjoint,
    box_total_width,
    build_traffic_lambda,
    coordinate_tail_report,
    covers,
    draw_traffic_client_counts,
    traffic_lambda_box,
)


MIXTURES = {
    "uniform": np.array([0.25, 0.25, 0.25, 0.25]),
    "skewed": np.array([0.55, 0.25, 0.15, 0.05]),
    "boundary_zero": np.array([0.0, 0.34, 0.33, 0.33]),
    "boundary_extreme": np.array([0.90, 0.06, 0.03, 0.01]),
}


def _cp_box_bounds(counts: np.ndarray, total: int, tail: float):
    """Vectorized two-sided Clopper-Pearson box bounds over trials x coordinates.

    Same formula as :func:`fedcore.mixture.traffic_mixture_confidence_box`
    (verified equal in the theorem tests); vectorized so the MC study over
    thousands of trials is fast.
    """
    lo = np.zeros_like(counts, dtype=float)
    hi = np.ones_like(counts, dtype=float)
    pos = counts > 0
    lt = counts < total
    lo[pos] = _beta.ppf(tail, counts[pos], total - counts[pos] + 1)
    hi[lt] = _beta.ppf(1.0 - tail, counts[lt] + 1, total - counts[lt])
    return lo, hi


def _mc_coverage(lambda_star: np.ndarray, m: int, n_trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(m, lambda_star, size=n_trials).astype(np.int64)
    tail = DELTA_LAMBDA / (2.0 * lambda_star.size)
    lo, hi = _cp_box_bounds(counts, m, tail)
    inside = np.all(
        (lambda_star[None, :] >= lo - 1e-12) & (lambda_star[None, :] <= hi + 1e-12),
        axis=1,
    )
    return float(inside.mean())


def _mean_width(lambda_star: np.ndarray, m: int, n_trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(m, lambda_star, size=n_trials).astype(np.int64)
    tail = DELTA_LAMBDA / (2.0 * lambda_star.size)
    lo, hi = _cp_box_bounds(counts, m, tail)
    return float(np.mean(np.sum(hi - lo, axis=1)))


def run_theorem_suite(
    *,
    n_trials: int = 12000,
    ms: Sequence[int] = M_GRID,
    seed: int = 20260718,
    out_dir: str | None = None,
    docs_path: str | None = None,
    write: bool = True,
) -> dict:
    """Run all properties; optionally write the frozen artifacts. Returns a dict.

    Coverage guarantee ``P(lambda in Lambda_hat) >= 1 - delta_lambda`` is EXACT by
    the union bound over conservative Clopper-Pearson tails. The Monte-Carlo study
    only confirms it; a cell PASSES when the empirical coverage is not
    significantly below target, i.e. ``coverage >= target - 3 * se`` with
    ``se = sqrt(target (1 - target) / n_trials)`` (3-sigma MC allowance).
    """

    target = 1.0 - DELTA_LAMBDA
    mc_tol = 3.0 * math.sqrt(target * (1.0 - target) / n_trials)
    coverage_rows: list[dict] = []
    coverage_pass = True
    for label, lam in MIXTURES.items():
        for m in ms:
            cov = _mc_coverage(lam, m, n_trials, seed + hash(label) % 9973 + m)
            row = {
                "mixture": label,
                "lambda_star": ";".join(f"{x:.4f}" for x in lam),
                "m": int(m),
                "n_trials": int(n_trials),
                "coverage": round(cov, 5),
                "target": target,
                "mc_tolerance": round(mc_tol, 5),
                "pass": bool(cov >= target - mc_tol),
            }
            coverage_rows.append(row)
            coverage_pass = coverage_pass and row["pass"]

    # Module/formula consistency: the vectorized CP bounds used above must equal
    # the fedcore.mixture box the certificate actually consumes.
    consistency_pass = True
    for counts in ([100, 200, 300, 400], [0, 5, 500, 10], [1, 1, 1, 1]):
        cnt = np.asarray(counts, dtype=np.int64)
        lo, hi = _cp_box_bounds(cnt[None, :], int(cnt.sum()), DELTA_LAMBDA / (2 * cnt.size))
        box = traffic_lambda_box(cnt, DELTA_LAMBDA)
        consistency_pass = consistency_pass and bool(
            np.allclose(lo[0], box.raw_lower) and np.allclose(hi[0], box.raw_upper)
        )

    # (2) exact coordinate-tail accounting.
    box = traffic_lambda_box([100, 200, 300, 400], DELTA_LAMBDA)
    tail = coordinate_tail_report(box)
    tail_pass = (
        tail["n_tails"] == 2 * N_DOMAIN_CLIENTS
        and abs(tail["union_bound"] - DELTA_LAMBDA) < 1e-12
        and abs(tail["per_side_tail"] - DELTA_LAMBDA / (2 * N_DOMAIN_CLIENTS)) < 1e-12
    )

    # (3) label/score-free construction: counts depend only on client ids + seed,
    #     never on any label/score array (the function accepts none).
    client_ids = [0, 1, 2, 3] * 250
    c_a = draw_traffic_client_counts(client_ids, M_PRIMARY, seed=1)
    c_b = draw_traffic_client_counts(client_ids, M_PRIMARY, seed=1)
    label_free_pass = np.array_equal(c_a, c_b) and int(c_a.sum()) == M_PRIMARY

    # (4) deterministic replay: same seed -> identical box.
    box_a = traffic_lambda_box(c_a, DELTA_LAMBDA)
    box_b = traffic_lambda_box(
        draw_traffic_client_counts(client_ids, M_PRIMARY, seed=1), DELTA_LAMBDA
    )
    replay_pass = np.array_equal(box_a.raw_lower, box_b.raw_lower) and np.array_equal(
        box_a.raw_upper, box_b.raw_upper
    )

    # (5) aggregate width decreases with m.
    lam = MIXTURES["uniform"]
    widths = {m: _mean_width(lam, m, max(400, n_trials // 4), seed + m) for m in ms}
    ordered = sorted(ms)
    width_pass = all(
        widths[ordered[i]] > widths[ordered[i + 1]] for i in range(len(ordered) - 1)
    )

    # (6) fail-closed on identity overlap.
    overlap_pass = False
    try:
        assert_traffic_identity_disjoint(["a", "b"], ["b", "c"])
    except TrafficLambdaError:
        overlap_pass = True

    # (7) full-simplex fallback on empty traffic.
    fallback = build_traffic_lambda([0, 0, 0, 0], DELTA_LAMBDA)
    fallback_pass = (
        fallback.fell_back_to_simplex
        and np.array_equal(fallback.box.raw_lower, np.zeros(N_DOMAIN_CLIENTS))
        and np.array_equal(fallback.box.raw_upper, np.ones(N_DOMAIN_CLIENTS))
    )

    properties = {
        "mc_coverage_ge_target": coverage_pass,
        "cp_box_matches_certificate_module": bool(consistency_pass),
        "exact_coordinate_tail_accounting": bool(tail_pass),
        "label_score_free_construction": bool(label_free_pass),
        "deterministic_replay": bool(replay_pass),
        "aggregate_width_decreases_with_m": bool(width_pass),
        "fail_closed_on_identity_overlap": bool(overlap_pass),
        "full_simplex_fallback": bool(fallback_pass),
    }
    all_pass = all(properties.values())

    result = {
        "delta_lambda": DELTA_LAMBDA,
        "target_coverage": target,
        "n_trials": n_trials,
        "ms": list(ms),
        "coverage_rows": coverage_rows,
        "widths_by_m": {int(k): v for k, v in widths.items()},
        "properties": properties,
        "all_pass": all_pass,
    }

    if write:
        out_dir = out_dir or "results/officehome/theorem_tests"
        docs_path = docs_path or "docs/agent/data_derived_lambda_theorem.md"
        os.makedirs(out_dir, exist_ok=True)
        _write_coverage_csv(os.path.join(out_dir, "traffic_set_coverage.csv"), coverage_rows)
        _write_validity_csv(
            os.path.join(out_dir, "combined_validity.csv"), properties, widths, ordered
        )
        _write_theorem_md(docs_path, result)
        result["out_dir"] = out_dir
        result["docs_path"] = docs_path
    return result


def _write_coverage_csv(path: str, rows: list[dict]) -> None:
    fields = ["mixture", "lambda_star", "m", "n_trials", "coverage", "target", "mc_tolerance", "pass"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_validity_csv(path: str, properties: dict, widths: dict, ordered: list) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["property", "pass", "detail"])
        for name, ok in properties.items():
            writer.writerow([name, bool(ok), ""])
        for m in ordered:
            writer.writerow([f"mean_total_width_m={m}", "", round(widths[m], 6)])


def _write_theorem_md(path: str, result: dict) -> None:
    props = result["properties"]
    lines = [
        "# Data-derived deployment mixture Lambda_hat (Office-Home traffic fold)",
        "",
        "## Statement",
        "",
        "Let the traffic fold supply domain (client) identities only. Draw `m` traffic",
        "observations; let `N_j` be the count from client `j` and `N = sum_j N_j`.",
        "Marginally `N_j ~ Binomial(N, lambda_j)` for the true deployment mixture",
        "`lambda`. Each coordinate receives a two-sided Clopper-Pearson interval at",
        f"per-side tail `delta_lambda/(2J)` with `delta_lambda = {result['delta_lambda']}`,",
        "`J = 4`. A union bound over the `2J` tails gives",
        "",
        "    P( lambda in Lambda_hat ) >= 1 - delta_lambda .",
        "",
        "The raw coordinate box is intersected with the simplex (exact coordinate-hull",
        "tightening) for use by the certificate's box target. Construction reads no",
        "label, score, prediction, or model output.",
        "",
        f"Budgets: delta_lambda / delta_r / delta_c = 0.02 / 0.04 / 0.04; m primary = {M_PRIMARY};",
        f"sensitivity grid m in {list(M_GRID)}.",
        "",
        "## Verified properties (see results/officehome/theorem_tests/)",
        "",
    ]
    for name, ok in props.items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} -- {name}")
    lines.append("")
    lines.append(f"Overall: {'PASS' if result['all_pass'] else 'FAIL'} "
                 f"({result['n_trials']} MC trials per (mixture, m) cell).")
    lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def main() -> int:
    result = run_theorem_suite(write=True)
    import json

    view = {k: v for k, v in result.items() if k != "coverage_rows"}
    view["n_coverage_rows"] = len(result["coverage_rows"])
    print(json.dumps(view, indent=2))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
