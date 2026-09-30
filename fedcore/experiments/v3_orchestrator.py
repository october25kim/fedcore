"""Deterministic staged orchestration for PACS-free v3 synthetic integration."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from fedcore.experiments.v3_contract import (
    EXPECTED_PRIMARY_CANDIDATE_ROWS,
    EXPECTED_PRIMARY_COUNT_ROWS,
    EXPECTED_PRIMARY_PROCEDURE_ROWS,
    EXPECTED_PROPOSAL_ROWS,
    PROCEDURES,
    PROTOCOL_ID,
    ContractError,
    ModelCell,
    canonical_json_bytes,
    canonical_json_sha256,
    load_model_cells,
    sha256_file,
    verify_scientific_contract,
)
from fedcore.experiments.v3_hsb import (
    CandidateDecision,
    candidate_decisions,
    procedure_rows,
    validate_hsb_invariants,
)
from fedcore.experiments.v3_proposal import members_from_rows
from fedcore.experiments.v3_synthetic import SYNTHETIC_MARKERS, synthetic_cell_payload


PROPOSAL_FIELDS = (
    "model_id",
    "candidate_index",
    "score",
    "gamma",
    "proposal_feasible",
    "threshold",
    "proposal_n",
    "proposal_A",
    "proposal_K",
    "proposal_risk",
)
COUNT_FIELDS = ("model_id", "candidate_index", "client", "status", "n", "A", "K")
CANDIDATE_FIELDS = tuple(CandidateDecision.__dataclass_fields__.keys())
PRIMARY_FIELDS = (
    "protocol_id",
    "model_id",
    "architecture",
    "split",
    "nominal_training_seed",
    "training_status",
    "model_evaluable",
    "failure_stage",
    "checkpoint_sha256",
    "proposal_seal_sha256",
    "audit_sequence_sha256",
    "count_tensor_sha256",
    "candidate_decision_file_sha256",
    "procedure",
    "alpha",
    "delta_r",
    "delta_c",
    "mixture_set",
    "budget_total",
    "replicate_id",
    "proposal_feasible_members",
    "certified",
    "selected_candidate",
    "selected_score",
    "selected_gamma",
    "selected_threshold",
    "risk_decision",
    "risk_ucb",
    "raw_p_value",
    "holm_adjusted_p_value",
    "coverage_lcb",
    "ECA",
    "refusal_reason",
    "diagnostic_failure_subtype",
)
TRUTH_FIELDS = (
    "model_id",
    "procedure",
    "primary_decision_file_sha256",
    "primary_candidate_decision_file_sha256",
    "truth_opened_after_primary_hash",
    "selected_candidate",
    "true_worst_client_risk",
    "true_min_client_acceptance",
    "validity_failure",
    "truth_unavailable_reason",
)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, canonical_json_bytes(value))


def _csv_bytes(
    rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str], *, compressed: bool
) -> bytes:
    text_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        text_buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for source in rows:
        row = {field: source.get(field) for field in fieldnames}
        for field, value in tuple(row.items()):
            if value is None:
                row[field] = ""
            elif isinstance(value, bool):
                row[field] = "true" if value else "false"
        writer.writerow(row)
    raw = text_buffer.getvalue().encode("utf-8")
    if not compressed:
        return raw
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as handle:
        handle.write(raw)
    return output.getvalue()


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
    *,
    compressed: bool = False,
) -> None:
    _atomic_bytes(path, _csv_bytes(rows, fieldnames, compressed=compressed))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _model_from_payload(value: Mapping[str, Any]) -> ModelCell:
    return ModelCell(
        model_id=str(value["model_id"]),
        architecture=str(value["architecture"]),
        split=int(value["split"]),
        nominal_training_seed=int(value["nominal_training_seed"]),
        model_init_seed_uint64=int(value["model_init_seed_uint64"]),
        known_classes=tuple(value["known_classes"]),
        unknown_classes=tuple(value["unknown_classes"]),
    )


def _generate_fragment(model: ModelCell, ordinal: int) -> dict[str, Any]:
    payload = synthetic_cell_payload(model, ordinal)
    members = members_from_rows(payload["members"])
    candidates = candidate_decisions(
        model,
        members,
        payload["counts"],
        checkpoint_sha256=payload["checkpoint_sha256"] or "",
        proposal_seal_sha256=payload["proposal_seal_sha256"] or "",
        audit_sequence_sha256=payload["audit_sequence_sha256"] or "",
    )
    validate_hsb_invariants(candidates)
    payload["candidate_decisions"] = [row.as_row() for row in candidates]
    payload["fragment_sha256"] = canonical_json_sha256(
        {
            "model": payload["model"],
            "members": payload["members"],
            "counts": payload["counts"],
            "candidate_decisions": payload["candidate_decisions"],
        }
    )
    return payload


def _validate_fragment(payload: Mapping[str, Any], expected: ModelCell, ordinal: int) -> None:
    if any(payload.get(key) != value for key, value in SYNTHETIC_MARKERS.items()):
        raise ContractError("synthetic provenance marker mismatch")
    if payload.get("ordinal") != ordinal:
        raise ContractError("synthetic fragment ordinal mismatch")
    model = _model_from_payload(payload["model"])
    if model != expected:
        raise ContractError("synthetic fragment model binding mismatch")
    rows = [CandidateDecision(**row) for row in payload["candidate_decisions"]]
    validate_hsb_invariants(rows)
    observed = canonical_json_sha256(
        {
            "model": payload["model"],
            "members": payload["members"],
            "counts": payload["counts"],
            "candidate_decisions": payload["candidate_decisions"],
        }
    )
    if payload.get("fragment_sha256") != observed:
        raise ContractError("synthetic fragment hash mismatch")


def _truth_rows(
    primary_rows: Sequence[Mapping[str, Any]],
    count_rows: Sequence[Mapping[str, Any]],
    primary_hash: str,
    candidate_hash: str,
) -> list[dict[str, Any]]:
    counts_by_model: dict[str, list[Mapping[str, Any]]] = {}
    for row in count_rows:
        counts_by_model.setdefault(str(row["model_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for row in primary_rows:
        evaluable = bool(row["model_evaluable"])
        certified = bool(row["certified"])
        selected = int(row["selected_candidate"])
        if not evaluable:
            reason = "model_failure"
            true_risk = None
            true_acceptance = None
            validity_failure = None
        elif row["refusal_reason"] == "proposal_family_all_infeasible":
            reason = "proposal_infeasible"
            true_risk = None
            true_acceptance = None
            validity_failure = None
        elif not certified:
            reason = "certificate_refusal"
            true_risk = None
            true_acceptance = None
            validity_failure = None
        else:
            reason = "none"
            selected_counts = [
                item
                for item in counts_by_model[str(row["model_id"])]
                if int(item["candidate_index"]) == selected
            ]
            risks = [
                (float(item["K"]) / float(item["A"])) if int(item["A"]) else 1.0
                for item in selected_counts
            ]
            acceptances = [float(item["A"]) / float(item["n"]) for item in selected_counts]
            true_risk = max(risks)
            true_acceptance = min(acceptances)
            validity_failure = bool(true_risk > 0.20)
        output.append(
            {
                "model_id": row["model_id"],
                "procedure": row["procedure"],
                "primary_decision_file_sha256": primary_hash,
                "primary_candidate_decision_file_sha256": candidate_hash,
                "truth_opened_after_primary_hash": evaluable,
                "selected_candidate": selected,
                "true_worst_client_risk": true_risk,
                "true_min_client_acceptance": true_acceptance,
                "validity_failure": validity_failure,
                "truth_unavailable_reason": reason,
            }
        )
    return output


def _write_sha256s(output_dir: Path, names: Sequence[str]) -> dict[str, str]:
    hashes = {name: sha256_file(output_dir / name) for name in sorted(names)}
    text = "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    _atomic_bytes(output_dir / "SHA256SUMS.txt", text.encode("utf-8"))
    return hashes


def materialize_synthetic(output_dir: Path, fragments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    proposal_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    candidates_by_model: dict[str, list[CandidateDecision]] = {}
    models_by_id: dict[str, ModelCell] = {}
    for payload in fragments:
        model = _model_from_payload(payload["model"])
        models_by_id[model.model_id] = model
        proposal_rows.extend({"model_id": model.model_id, **row} for row in payload["members"])
        count_rows.extend(dict(row) for row in payload["counts"])
        decisions = [CandidateDecision(**row) for row in payload["candidate_decisions"]]
        candidates_by_model[model.model_id] = decisions
        candidate_rows.extend(row.as_row() for row in decisions)
    model_order = {cell.model_id: i for i, cell in enumerate(load_model_cells())}
    procedure_order = {name: i for i, name in enumerate(PROCEDURES)}
    proposal_rows.sort(key=lambda row: (model_order[str(row["model_id"])], int(row["candidate_index"])))
    count_rows.sort(
        key=lambda row: (
            model_order[str(row["model_id"])],
            int(row["candidate_index"]),
            str(row["client"]),
        )
    )
    candidate_rows.sort(
        key=lambda row: (
            model_order[str(row["model_id"])],
            int(row["candidate_index"]),
            procedure_order[str(row["procedure"])],
        )
    )
    if len(proposal_rows) != EXPECTED_PROPOSAL_ROWS:
        raise ContractError("proposal denominator drift")
    if len(count_rows) != EXPECTED_PRIMARY_COUNT_ROWS:
        raise ContractError("primary count denominator drift")
    if len(candidate_rows) != EXPECTED_PRIMARY_CANDIDATE_ROWS:
        raise ContractError("primary candidate denominator drift")
    write_csv(
        output_dir / "PROPOSAL_FAMILY_MANIFEST.csv.gz",
        proposal_rows,
        PROPOSAL_FIELDS,
        compressed=True,
    )
    write_csv(output_dir / "PRIMARY_COUNTS.csv.gz", count_rows, COUNT_FIELDS, compressed=True)
    write_csv(
        output_dir / "PRIMARY_CANDIDATE_DECISIONS.csv.gz",
        candidate_rows,
        CANDIDATE_FIELDS,
        compressed=True,
    )
    candidate_hash = sha256_file(output_dir / "PRIMARY_CANDIDATE_DECISIONS.csv.gz")
    primary_rows: list[dict[str, Any]] = []
    for model_id in sorted(candidates_by_model, key=model_order.__getitem__):
        primary_rows.extend(
            procedure_rows(
                models_by_id[model_id],
                candidates_by_model[model_id],
                candidate_decision_file_sha256=candidate_hash,
            )
        )
    if len(primary_rows) != EXPECTED_PRIMARY_PROCEDURE_ROWS:
        raise ContractError("primary procedure denominator drift")
    write_csv(output_dir / "PRIMARY_512_REP000.csv", primary_rows, PRIMARY_FIELDS)
    primary_hash = sha256_file(output_dir / "PRIMARY_512_REP000.csv")
    primary_seal = {
        **SYNTHETIC_MARKERS,
        "protocol_id": PROTOCOL_ID,
        "state": "PRIMARY_HASH_SEALED_BEFORE_TRUTH",
        "primary_decision_file_sha256": primary_hash,
        "primary_candidate_decision_file_sha256": candidate_hash,
        "primary_count_file_sha256": sha256_file(output_dir / "PRIMARY_COUNTS.csv.gz"),
        "primary_procedure_rows": len(primary_rows),
        "primary_candidate_rows": len(candidate_rows),
        "primary_count_rows": len(count_rows),
    }
    write_json(output_dir / "PRIMARY_SEAL.json", primary_seal)
    truth_rows = _truth_rows(primary_rows, count_rows, primary_hash, candidate_hash)
    write_csv(output_dir / "POST_DECISION_TRUTH.csv", truth_rows, TRUTH_FIELDS)
    failures = [row for row in primary_rows if row["refusal_reason"] == "model_failure"]
    write_csv(
        output_dir / "FAILURE_LEDGER.csv",
        failures,
        PRIMARY_FIELDS,
    )
    denominator = {
        **SYNTHETIC_MARKERS,
        "protocol_id": PROTOCOL_ID,
        "planned_models": 30,
        "accounted_models": 30,
        "proposal_rows": len(proposal_rows),
        "primary_count_rows": len(count_rows),
        "primary_candidate_rows": len(candidate_rows),
        "primary_procedure_rows": len(primary_rows),
        "terminal_model_failures": len({row["model_id"] for row in failures}),
        "proposal_infeasible_models": len(
            {
                row["model_id"]
                for row in primary_rows
                if row["refusal_reason"] == "proposal_family_all_infeasible"
            }
        ),
    }
    write_json(output_dir / "DENOMINATOR_LEDGER.json", denominator)
    ordering = {
        **SYNTHETIC_MARKERS,
        "protocol_id": PROTOCOL_ID,
        "primary_seal_created_before_truth": True,
        "primary_seal_sha256": sha256_file(output_dir / "PRIMARY_SEAL.json"),
        "post_decision_truth_sha256": sha256_file(output_dir / "POST_DECISION_TRUTH.csv"),
        "truth_references_primary_sha256": True,
    }
    write_json(output_dir / "ORDERING_LEDGER.json", ordering)
    negative_text = (
        "# Synthetic negative controls\n\n"
        "These are wiring controls, not PACS results. One frozen model cell uses "
        "terminal model-failure placeholders, one evaluable cell retains a "
        "12-member proposal-infeasible family, and one proposal-feasible cell "
        "fails every registered risk decision.\n"
    )
    _atomic_bytes(output_dir / "NEGATIVE_RESULTS.md", negative_text.encode("utf-8"))
    names = (
        "PROPOSAL_FAMILY_MANIFEST.csv.gz",
        "PRIMARY_COUNTS.csv.gz",
        "PRIMARY_CANDIDATE_DECISIONS.csv.gz",
        "PRIMARY_512_REP000.csv",
        "PRIMARY_SEAL.json",
        "POST_DECISION_TRUTH.csv",
        "FAILURE_LEDGER.csv",
        "DENOMINATOR_LEDGER.json",
        "ORDERING_LEDGER.json",
        "NEGATIVE_RESULTS.md",
    )
    hashes = _write_sha256s(output_dir, names)
    return {
        **SYNTHETIC_MARKERS,
        "status": "PASS_SYNTHETIC_MATERIALIZATION",
        "artifact_hashes": hashes,
        "sha256s_sha256": sha256_file(output_dir / "SHA256SUMS.txt"),
        "artifact_root_sha256": canonical_json_sha256(hashes),
    }


def run_synthetic(
    output_dir: Path,
    *,
    max_new_cells: int | None = None,
) -> dict[str, Any]:
    """Run or resume the deterministic 30-cell synthetic integration."""

    verify_scientific_contract()
    output_dir = Path(output_dir)
    fragments_dir = output_dir / "cells"
    fragments_dir.mkdir(parents=True, exist_ok=True)
    cells = load_model_cells()
    generated = 0
    for ordinal, model in enumerate(cells):
        fragment_path = fragments_dir / f"{ordinal:02d}_{model.model_id}.json"
        if fragment_path.exists():
            payload = read_json(fragment_path)
            _validate_fragment(payload, model, ordinal)
            continue
        if max_new_cells is not None and generated >= max_new_cells:
            break
        payload = _generate_fragment(model, ordinal)
        write_json(fragment_path, payload)
        generated += 1
    fragments: list[dict[str, Any]] = []
    missing: list[str] = []
    for ordinal, model in enumerate(cells):
        fragment_path = fragments_dir / f"{ordinal:02d}_{model.model_id}.json"
        if not fragment_path.exists():
            missing.append(model.model_id)
            continue
        payload = read_json(fragment_path)
        _validate_fragment(payload, model, ordinal)
        fragments.append(payload)
    if missing:
        state = {
            **SYNTHETIC_MARKERS,
            "protocol_id": PROTOCOL_ID,
            "status": "HOLD_INCOMPLETE_SYNTHETIC",
            "completed_cells": len(fragments),
            "planned_cells": len(cells),
            "missing_model_ids": missing,
        }
        write_json(output_dir / "RUN_STATE.json", state)
        return state
    materialized = materialize_synthetic(output_dir, fragments)
    state = {
        **SYNTHETIC_MARKERS,
        "protocol_id": PROTOCOL_ID,
        "completed_cells": len(fragments),
        "planned_cells": len(cells),
        **materialized,
        "status": "PASS_SYNTHETIC_COMPLETE",
    }
    write_json(output_dir / "RUN_STATE.json", state)
    return state


__all__ = [
    "CANDIDATE_FIELDS",
    "COUNT_FIELDS",
    "PRIMARY_FIELDS",
    "PROPOSAL_FIELDS",
    "TRUTH_FIELDS",
    "materialize_synthetic",
    "read_csv",
    "read_json",
    "run_synthetic",
    "write_csv",
    "write_json",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("synthetic",), default="synthetic")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-cells", type=int)
    args = parser.parse_args()
    state = run_synthetic(args.output_dir, max_new_cells=args.max_new_cells)
    print(state["status"])
    return 0 if state["status"] == "PASS_SYNTHETIC_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
