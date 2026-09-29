"""Independent count-to-decision replay for the sealed v3 H/S/B outputs.

This module intentionally does not import ``v3_hsb`` or the production Holm and
Clopper--Pearson helpers.  It re-implements the registered formulae from SciPy
primitives and compares every primary candidate decision.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta, binom

from fedcore.experiments.v3_contract import (
    ALPHA,
    CLIENTS,
    DELTA_C,
    DELTA_R,
    J,
    M,
    PROCEDURES,
    PROTOCOL_ID,
    ContractError,
    canonical_json_sha256,
    sha256_file,
)
from fedcore.experiments.v3_orchestrator import read_csv, write_json
from fedcore.experiments.v3_synthetic import SYNTHETIC_MARKERS


ATOL = 1e-12


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() == "true":
        return True
    if str(value).lower() == "false":
        return False
    raise ContractError(f"not a serialized boolean: {value!r}")


def _float(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _int(value: Any) -> int | None:
    return None if value in (None, "") else int(value)


def _uplus(k: int, n: int, eta: float) -> float:
    if n <= 0 or k >= n or eta <= 0.0:
        return 1.0
    return float(beta.isf(eta, k + 1, n - k))


def _lminus(x: int, n: int, eta: float) -> float:
    if n <= 0 or x <= 0:
        return 0.0
    return float(beta.ppf(eta, x, n - x + 1))


def _holm(A: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, ...]:
    raw = np.asarray(
        [
            max(1.0 if int(A[m, j]) == 0 else float(binom.cdf(int(K[m, j]), int(A[m, j]), ALPHA)) for j in range(J))
            for m in range(M)
        ],
        dtype=np.float64,
    )
    order = sorted(range(M), key=lambda index: (float(raw[index]), index))
    rejected = np.zeros(M, dtype=bool)
    ranks = np.zeros(M, dtype=np.int64)
    adjusted = np.zeros(M, dtype=np.float64)
    running = 0.0
    active = True
    for zero_rank, index in enumerate(order):
        rank = zero_rank + 1
        ranks[index] = rank
        running = max(running, (M - zero_rank) * float(raw[index]))
        adjusted[index] = min(1.0, running)
        threshold = DELTA_R / (M - zero_rank)
        if active and float(raw[index]) <= threshold:
            rejected[index] = True
        else:
            active = False
    critical = np.asarray([DELTA_R / (M - int(rank) + 1) for rank in ranks])
    return raw, adjusted, ranks, rejected, critical


def _tensor_hash(rows: Sequence[Mapping[str, str]]) -> str:
    ordered = sorted(
        rows,
        key=lambda row: (int(row["candidate_index"]), CLIENTS.index(row["client"])),
    )
    payload = [
        {
            "candidate_index": int(row["candidate_index"]),
            "client": row["client"],
            "status": row["status"],
            "n": _int(row["n"]),
            "A": _int(row["A"]),
            "K": _int(row["K"]),
        }
        for row in ordered
    ]
    return canonical_json_sha256(payload)


def _select(rows: Sequence[dict[str, Any]]) -> int | None:
    eligible = [row for row in rows if row["candidate_certified"]]
    if not eligible:
        return None
    return int(
        min(
            eligible,
            key=lambda row: (
                -float(row["coverage_lcb"]),
                float(row["gamma"]),
                str(row["score"]),
                int(row["candidate_index"]),
            ),
        )["candidate_index"]
    )


def _replay_model(
    proposal_rows: Sequence[Mapping[str, str]],
    count_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    proposal = {int(row["candidate_index"]): row for row in proposal_rows}
    if len(proposal) != M or len(count_rows) != M * J:
        raise ContractError("independent replay denominator drift")
    model_failure = all(row["status"] == "model_failure" for row in count_rows)
    if model_failure:
        return [
            {
                "candidate_index": m,
                "procedure": procedure,
                "model_evaluable": False,
                "proposal_feasible": False,
                "count_tensor_sha256": None,
                "risk_tail": None,
                "coverage_tail": None,
                "raw_max_client_p": None,
                "holm_rank": None,
                "holm_critical_value": None,
                "holm_adjusted_p": None,
                "risk_ucb": None,
                "coverage_lcb": None,
                "risk_pass": None,
                "coverage_positive": None,
                "candidate_certified": False,
                "selected": False,
            }
            for procedure in PROCEDURES
            for m in range(M)
        ]
    if any(row["status"] != "ok" for row in count_rows):
        raise ContractError("mixed count status in independent replay")
    A = np.zeros((M, J), dtype=np.int64)
    K = np.zeros((M, J), dtype=np.int64)
    n = np.zeros(J, dtype=np.int64)
    for row in count_rows:
        m = int(row["candidate_index"])
        j = CLIENTS.index(row["client"])
        n[j] = int(row["n"])
        A[m, j] = int(row["A"])
        K[m, j] = int(row["K"])
    if np.any(K < 0) or np.any(K > A) or np.any(A > n[None, :]):
        raise ContractError("invalid count ordering in independent replay")
    tensor_hash = _tensor_hash(count_rows)
    raw, adjusted, ranks, rejected, critical = _holm(A, K)
    coverage_hs = np.asarray(
        [min(_lminus(int(A[m, j]), int(n[j]), DELTA_C / M) for j in range(J)) for m in range(M)]
    )
    risk_s = np.asarray(
        [max(_uplus(int(K[m, j]), int(A[m, j]), DELTA_R / M) for j in range(J)) for m in range(M)]
    )
    risk_b = np.asarray(
        [max(_uplus(int(K[m, j]), int(A[m, j]), DELTA_R / (M * J)) for j in range(J)) for m in range(M)]
    )
    coverage_b = np.asarray(
        [min(_lminus(int(A[m, j]), int(n[j]), DELTA_C / (M * J)) for j in range(J)) for m in range(M)]
    )
    rows: list[dict[str, Any]] = []
    for procedure in PROCEDURES:
        for m in range(M):
            feasible = _bool(proposal[m]["proposal_feasible"])
            if procedure == "H":
                row = {
                    "risk_tail": None,
                    "coverage_tail": DELTA_C / M,
                    "raw_max_client_p": float(raw[m]),
                    "holm_rank": int(ranks[m]),
                    "holm_critical_value": float(critical[m]),
                    "holm_adjusted_p": float(adjusted[m]),
                    "risk_ucb": None,
                    "coverage_lcb": float(coverage_hs[m]),
                    "risk_pass": bool(rejected[m]),
                }
            elif procedure == "S":
                row = {
                    "risk_tail": DELTA_R / M,
                    "coverage_tail": DELTA_C / M,
                    "raw_max_client_p": None,
                    "holm_rank": None,
                    "holm_critical_value": None,
                    "holm_adjusted_p": None,
                    "risk_ucb": float(risk_s[m]),
                    "coverage_lcb": float(coverage_hs[m]),
                    "risk_pass": bool(risk_s[m] <= ALPHA),
                }
            else:
                row = {
                    "risk_tail": DELTA_R / (M * J),
                    "coverage_tail": DELTA_C / (M * J),
                    "raw_max_client_p": None,
                    "holm_rank": None,
                    "holm_critical_value": None,
                    "holm_adjusted_p": None,
                    "risk_ucb": float(risk_b[m]),
                    "coverage_lcb": float(coverage_b[m]),
                    "risk_pass": bool(risk_b[m] <= ALPHA),
                }
            positive = float(row["coverage_lcb"]) > 0.0
            rows.append(
                {
                    "candidate_index": m,
                    "procedure": procedure,
                    "model_evaluable": True,
                    "proposal_feasible": feasible,
                    "count_tensor_sha256": tensor_hash,
                    **row,
                    "coverage_positive": positive,
                    "candidate_certified": bool(feasible and row["risk_pass"] and positive),
                    "selected": False,
                    "score": proposal[m]["score"],
                    "gamma": float(proposal[m]["gamma"]),
                }
            )
    for procedure in PROCEDURES:
        selected = _select([row for row in rows if row["procedure"] == procedure])
        for row in rows:
            if row["procedure"] == procedure:
                row["selected"] = selected == row["candidate_index"]
    return rows


def _compare(actual: Mapping[str, str], expected: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    bool_fields = (
        "model_evaluable",
        "proposal_feasible",
        "risk_pass",
        "coverage_positive",
        "candidate_certified",
        "selected",
    )
    int_fields = ("candidate_index", "holm_rank")
    float_fields = (
        "risk_tail",
        "coverage_tail",
        "raw_max_client_p",
        "holm_critical_value",
        "holm_adjusted_p",
        "risk_ucb",
        "coverage_lcb",
    )
    for field in bool_fields:
        observed = None if actual.get(field, "") == "" else _bool(actual[field])
        if observed != expected.get(field):
            failures.append(f"{field}: {observed!r} != {expected.get(field)!r}")
    for field in int_fields:
        observed = _int(actual.get(field))
        if observed != expected.get(field):
            failures.append(f"{field}: {observed!r} != {expected.get(field)!r}")
    for field in float_fields:
        observed = _float(actual.get(field))
        wanted = expected.get(field)
        if observed is None or wanted is None:
            if observed is not None or wanted is not None:
                failures.append(f"{field}: {observed!r} != {wanted!r}")
        elif not np.isclose(observed, float(wanted), atol=ATOL, rtol=0.0):
            failures.append(f"{field}: {observed!r} != {wanted!r}")
    for field in ("procedure", "count_tensor_sha256"):
        observed = actual.get(field) or None
        if observed != expected.get(field):
            failures.append(f"{field}: {observed!r} != {expected.get(field)!r}")
    return failures


def replay(output_dir: Path, *, write_report: bool = True) -> dict[str, Any]:
    output_dir = Path(output_dir)
    proposals = read_csv(output_dir / "PROPOSAL_FAMILY_MANIFEST.csv.gz")
    counts = read_csv(output_dir / "PRIMARY_COUNTS.csv.gz")
    actual = read_csv(output_dir / "PRIMARY_CANDIDATE_DECISIONS.csv.gz")
    by_model_proposal: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_model_counts: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_key_actual: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in proposals:
        by_model_proposal[row["model_id"]].append(row)
    for row in counts:
        by_model_counts[row["model_id"]].append(row)
    for row in actual:
        key = (row["model_id"], int(row["candidate_index"]), row["procedure"])
        if key in by_key_actual:
            raise ContractError(f"duplicate production candidate key: {key!r}")
        by_key_actual[key] = row
    failures: list[dict[str, Any]] = []
    replayed = 0
    for model_id in sorted(by_model_proposal):
        expected_rows = _replay_model(by_model_proposal[model_id], by_model_counts[model_id])
        for expected in expected_rows:
            key = (model_id, int(expected["candidate_index"]), str(expected["procedure"]))
            observed = by_key_actual.get(key)
            if observed is None:
                failures.append({"key": key, "errors": ["missing production row"]})
                continue
            errors = _compare(observed, expected)
            if errors:
                failures.append({"key": key, "errors": errors})
            replayed += 1
    if replayed != 1080 or len(by_key_actual) != 1080:
        failures.append(
            {
                "key": "denominator",
                "errors": [f"replayed={replayed}, production={len(by_key_actual)}, expected=1080"],
            }
        )
    report = {
        **SYNTHETIC_MARKERS,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_INDEPENDENT_REPLAY" if not failures else "INVALID_REPLAY_MISMATCH",
        "production_module_imported": False,
        "candidate_rows_replayed": replayed,
        "mismatch_count": len(failures),
        "mismatches": failures[:100],
        "input_sha256": {
            "proposal": sha256_file(output_dir / "PROPOSAL_FAMILY_MANIFEST.csv.gz"),
            "counts": sha256_file(output_dir / "PRIMARY_COUNTS.csv.gz"),
            "candidate_decisions": sha256_file(
                output_dir / "PRIMARY_CANDIDATE_DECISIONS.csv.gz"
            ),
        },
        "atol": ATOL,
        "rtol": 0.0,
    }
    if write_report:
        write_json(output_dir / "INDEPENDENT_REPLAY.json", report)
    return report


__all__ = ["ATOL", "replay"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.output_dir)
    print(report["status"])
    return 0 if report["status"] == "PASS_INDEPENDENT_REPLAY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
