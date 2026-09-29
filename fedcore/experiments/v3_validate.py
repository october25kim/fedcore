"""Validation and fail-closed execution gates for the FedCORE v3 binding."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fedcore.experiments.v3_contract import (
    EXPECTED_PRIMARY_CANDIDATE_ROWS,
    EXPECTED_PRIMARY_COUNT_ROWS,
    EXPECTED_PRIMARY_PROCEDURE_ROWS,
    EXPECTED_PROPOSAL_ROWS,
    M,
    PROCEDURES,
    PROTOCOL_ID,
    ContractError,
    read_json,
    sha256_file,
)
from fedcore.experiments.v3_counts import count_tensor_sha256
from fedcore.experiments.v3_hsb import CandidateDecision, validate_hsb_invariants
from fedcore.experiments.v3_orchestrator import read_csv, write_json
from fedcore.experiments.v3_replay_independent import replay
from fedcore.experiments.v3_synthetic import SYNTHETIC_MARKERS


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() == "true":
        return True
    if str(value).lower() == "false":
        return False
    raise ContractError(f"not a boolean: {value!r}")


def _none(value: Any) -> Any:
    return None if value in (None, "") else value


def _candidate(row: Mapping[str, str]) -> CandidateDecision:
    return CandidateDecision(
        model_id=row["model_id"],
        candidate_index=int(row["candidate_index"]),
        procedure=row["procedure"],
        model_evaluable=_bool(row["model_evaluable"]),
        proposal_feasible=_bool(row["proposal_feasible"]),
        checkpoint_sha256=_none(row["checkpoint_sha256"]),
        proposal_seal_sha256=_none(row["proposal_seal_sha256"]),
        audit_sequence_sha256=_none(row["audit_sequence_sha256"]),
        count_tensor_sha256=_none(row["count_tensor_sha256"]),
        budget_total=int(row["budget_total"]),
        replicate_id=int(row["replicate_id"]),
        score=row["score"],
        gamma=float(row["gamma"]),
        threshold=None if row["threshold"] == "" else float(row["threshold"]),
        risk_tail=None if row["risk_tail"] == "" else float(row["risk_tail"]),
        coverage_tail=(
            None if row["coverage_tail"] == "" else float(row["coverage_tail"])
        ),
        raw_max_client_p=(
            None if row["raw_max_client_p"] == "" else float(row["raw_max_client_p"])
        ),
        holm_rank=None if row["holm_rank"] == "" else int(row["holm_rank"]),
        holm_critical_value=(
            None
            if row["holm_critical_value"] == ""
            else float(row["holm_critical_value"])
        ),
        holm_adjusted_p=(
            None if row["holm_adjusted_p"] == "" else float(row["holm_adjusted_p"])
        ),
        risk_ucb=None if row["risk_ucb"] == "" else float(row["risk_ucb"]),
        coverage_lcb=(
            None if row["coverage_lcb"] == "" else float(row["coverage_lcb"])
        ),
        risk_pass=None if row["risk_pass"] == "" else _bool(row["risk_pass"]),
        coverage_positive=(
            None if row["coverage_positive"] == "" else _bool(row["coverage_positive"])
        ),
        candidate_certified=_bool(row["candidate_certified"]),
        selected=_bool(row["selected"]),
    )


def _verify_sha256s(output_dir: Path) -> dict[str, str]:
    path = output_dir / "SHA256SUMS.txt"
    expected: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            digest, name = line.rstrip("\n").split("  ", 1)
            expected[name] = digest
    for name, digest in expected.items():
        observed = sha256_file(output_dir / name)
        if observed != digest:
            raise ContractError(f"artifact hash mismatch for {name}")
    return expected


def validate_synthetic_output(output_dir: Path, *, write_report: bool = True) -> dict[str, Any]:
    output_dir = Path(output_dir)
    state = read_json(output_dir / "RUN_STATE.json")
    if state.get("status") != "PASS_SYNTHETIC_COMPLETE":
        raise ContractError(f"synthetic run is not complete: {state.get('status')}")
    for key, value in SYNTHETIC_MARKERS.items():
        if state.get(key) != value:
            raise ContractError(f"synthetic marker drift: {key}")
    hashes = _verify_sha256s(output_dir)
    proposals = read_csv(output_dir / "PROPOSAL_FAMILY_MANIFEST.csv.gz")
    counts = read_csv(output_dir / "PRIMARY_COUNTS.csv.gz")
    candidate_text = read_csv(output_dir / "PRIMARY_CANDIDATE_DECISIONS.csv.gz")
    primary = read_csv(output_dir / "PRIMARY_512_REP000.csv")
    truth = read_csv(output_dir / "POST_DECISION_TRUTH.csv")
    observed_lengths = {
        "proposal": len(proposals),
        "counts": len(counts),
        "candidates": len(candidate_text),
        "primary": len(primary),
        "truth": len(truth),
    }
    expected_lengths = {
        "proposal": EXPECTED_PROPOSAL_ROWS,
        "counts": EXPECTED_PRIMARY_COUNT_ROWS,
        "candidates": EXPECTED_PRIMARY_CANDIDATE_ROWS,
        "primary": EXPECTED_PRIMARY_PROCEDURE_ROWS,
        "truth": EXPECTED_PRIMARY_PROCEDURE_ROWS,
    }
    if observed_lengths != expected_lengths:
        raise ContractError(f"synthetic row denominator mismatch: {observed_lengths!r}")
    by_model_candidates: dict[str, list[CandidateDecision]] = defaultdict(list)
    for row in candidate_text:
        by_model_candidates[row["model_id"]].append(_candidate(row))
    if len(by_model_candidates) != 30:
        raise ContractError("candidate model denominator drift")
    for rows in by_model_candidates.values():
        validate_hsb_invariants(rows)
    counts_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in counts:
        counts_by_model[row["model_id"]].append(row)
    for model_id, rows in by_model_candidates.items():
        if all(not row.model_evaluable for row in rows):
            continue
        expected_tensor_hash = count_tensor_sha256(counts_by_model[model_id])
        observed = {row.count_tensor_sha256 for row in rows}
        if observed != {expected_tensor_hash}:
            raise ContractError(f"count tensor hash mismatch for {model_id}")
    primary_hash = sha256_file(output_dir / "PRIMARY_512_REP000.csv")
    candidate_hash = sha256_file(output_dir / "PRIMARY_CANDIDATE_DECISIONS.csv.gz")
    if any(row["primary_decision_file_sha256"] != primary_hash for row in truth):
        raise ContractError("truth file does not reference the sealed primary hash")
    if any(
        row["primary_candidate_decision_file_sha256"] != candidate_hash for row in truth
    ):
        raise ContractError("truth file does not reference the candidate-decision hash")
    ordering = read_json(output_dir / "ORDERING_LEDGER.json")
    if ordering.get("primary_seal_created_before_truth") is not True:
        raise ContractError("primary-before-truth ordering evidence is absent")
    independent = replay(output_dir, write_report=write_report)
    if independent["status"] != "PASS_INDEPENDENT_REPLAY":
        raise ContractError("independent replay mismatch")
    cert_counts: dict[str, int] = {}
    means: dict[str, float] = {}
    for procedure in PROCEDURES:
        rows = [row for row in primary if row["procedure"] == procedure]
        cert_counts[procedure] = sum(_bool(row["certified"]) for row in rows)
        means[procedure] = float(np.mean([float(row["ECA"]) for row in rows]))
    if cert_counts != {"H": 27, "S": 27, "B": 27}:
        raise ContractError(f"unexpected synthetic certification counts: {cert_counts!r}")
    expected_means = {
        "H": 0.8622776096169696,
        "S": 0.6038421887274016,
        "B": 0.32669409029251906,
    }
    if any(not np.isclose(means[key], value, atol=1e-12, rtol=0.0) for key, value in expected_means.items()):
        raise ContractError(f"unexpected synthetic ECA means: {means!r}")
    refusal_counts = Counter(row["refusal_reason"] for row in primary)
    if refusal_counts != {
        "none": 81,
        "no_candidate_certified": 3,
        "model_failure": 3,
        "proposal_family_all_infeasible": 3,
    }:
        raise ContractError(f"unexpected refusal ledger: {refusal_counts!r}")
    report = {
        **SYNTHETIC_MARKERS,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_SYNTHETIC_BINDING_TESTS",
        "row_counts": observed_lengths,
        "certified_cells": cert_counts,
        "mean_ECA": means,
        "S_minus_B": means["S"] - means["B"],
        "H_minus_S": means["H"] - means["S"],
        "H_minus_B": means["H"] - means["B"],
        "refusal_rows": dict(refusal_counts),
        "independent_replay_status": independent["status"],
        "verified_artifact_hashes": hashes,
    }
    if write_report:
        write_json(output_dir / "FINAL_VALIDATION.json", report)
    return report


def validate_mount_scope(
    observed: Iterable[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    forbidden_roots: Sequence[Path],
) -> None:
    """Validate a normalized Docker-mount receipt against an exact allowlist."""

    def normalize(row: Mapping[str, Any]) -> tuple[str, str, str, bool]:
        mount_type = str(row["type"])
        if mount_type not in ("bind", "tmpfs"):
            raise ContractError(f"INVALID_MOUNT_SCOPE unsupported mount type {mount_type!r}")
        source = "" if mount_type == "tmpfs" else str(Path(str(row["source"])).resolve())
        destination = str(Path(str(row["destination"])).resolve())
        read_only = _bool(row["read_only"])
        return mount_type, source, destination, read_only

    observed_rows = [normalize(row) for row in observed]
    expected_rows = [normalize(row) for row in expected]
    if len(observed_rows) != len(set(observed_rows)):
        raise ContractError("INVALID_MOUNT_SCOPE duplicate observed mount")
    if len(expected_rows) != len(set(expected_rows)):
        raise ContractError("INVALID_MOUNT_SCOPE duplicate expected mount")
    observed_destinations = [row[2] for row in observed_rows]
    expected_destinations = [row[2] for row in expected_rows]
    if len(observed_destinations) != len(set(observed_destinations)):
        raise ContractError("INVALID_MOUNT_SCOPE duplicate observed destination")
    if len(expected_destinations) != len(set(expected_destinations)):
        raise ContractError("INVALID_MOUNT_SCOPE duplicate expected destination")
    observed_sources = [row[1] for row in observed_rows if row[0] != "tmpfs"]
    expected_sources = [row[1] for row in expected_rows if row[0] != "tmpfs"]
    if len(observed_sources) != len(set(observed_sources)):
        raise ContractError("INVALID_MOUNT_SCOPE duplicate observed source")
    if len(expected_sources) != len(set(expected_sources)):
        raise ContractError("INVALID_MOUNT_SCOPE duplicate expected source")
    observed_set = set(observed_rows)
    expected_set = set(expected_rows)
    if observed_set != expected_set:
        extra = sorted(observed_set - expected_set)
        missing = sorted(expected_set - observed_set)
        raise ContractError(f"INVALID_MOUNT_SCOPE extra={extra!r} missing={missing!r}")
    forbidden = [Path(path).resolve() for path in forbidden_roots]
    for kind, source, _, _ in observed_set:
        if kind == "tmpfs":
            continue
        path = Path(source).resolve()
        for root in forbidden:
            if path == root or root in path.parents:
                raise ContractError(f"INVALID_MOUNT_SCOPE forbidden source {path}")


def require_scientific_execution_authorization(execution_gate: Path) -> dict[str, Any]:
    gate = read_json(Path(execution_gate))
    if gate.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("execution gate protocol mismatch")
    if gate.get("status") != "PASS_EXECUTION_READY":
        raise ContractError("scientific profile refused: PASS_EXECUTION_READY is absent")
    if gate.get("training_authorized") is not True:
        raise ContractError("scientific profile refused: training_authorized is not true")
    return gate


__all__ = [
    "require_scientific_execution_authorization",
    "validate_mount_scope",
    "validate_synthetic_output",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_synthetic_output(args.synthetic_output)
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
