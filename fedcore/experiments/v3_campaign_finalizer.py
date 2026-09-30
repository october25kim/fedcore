"""Campaign-global single writer for the sealed v3 H/S/B protocol.

This module is deliberately PACS agnostic.  It accepts only already reduced
proposal/count/decision/truth fragments and never imports torch, opens an image,
or selects a checkpoint.  The primary phase is a separate transaction: all 30
fragments are validated, the 1,080 candidate rows and 90 procedure rows are
atomically materialized, and ``PRIMARY_SEAL.json`` is written last.  The seal is
the sole authority consumed by the truth and secondary aggregation phase.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fedcore.experiments.v3_contract import (
    ALPHA,
    CLIENTS,
    EXPECTED_MODELS,
    EXPECTED_PRIMARY_CANDIDATE_ROWS,
    EXPECTED_PRIMARY_COUNT_ROWS,
    EXPECTED_PRIMARY_PROCEDURE_ROWS,
    EXPECTED_PROPOSAL_ROWS,
    M,
    PRIMARY_BUDGET_TOTAL,
    PRIMARY_REPLICATE_ID,
    PROCEDURES,
    PROTOCOL_ID,
    ContractError,
    ModelCell,
    assert_unique_keys,
    canonical_json_sha256,
    default_protocol_dir,
    load_model_cells,
    read_json,
    sha256_file,
    verify_scientific_contract,
)
from fedcore.experiments.v3_counts import count_tensor_sha256, validate_count_rows
from fedcore.experiments.v3_hsb import (
    CandidateDecision,
    procedure_rows,
    validate_hsb_invariants,
)
from fedcore.experiments.v3_orchestrator import (
    CANDIDATE_FIELDS,
    COUNT_FIELDS,
    PRIMARY_FIELDS,
    PROPOSAL_FIELDS,
    TRUTH_FIELDS,
    _atomic_bytes,
    read_csv,
    write_csv,
    write_json,
)
from fedcore.experiments.v3_proposal import members_from_rows, proposal_manifest_sha256
from fedcore.experiments.v3_replay_independent import replay


OUTCOME_SCHEMA_SHA256 = "26691f27e73585193edcf9bfc0b4ce20a4749bec2ebf2d6ba2c77aa6332d75b1"
SECONDARY_CONFIGURATIONS_PER_MODEL = 299
EXPECTED_SECONDARY_COUNT_ROWS = EXPECTED_MODELS * SECONDARY_CONFIGURATIONS_PER_MODEL * M * len(CLIENTS)
EXPECTED_SECONDARY_CANDIDATE_ROWS = EXPECTED_MODELS * SECONDARY_CONFIGURATIONS_PER_MODEL * M * len(PROCEDURES)
EXPECTED_SECONDARY_PROCEDURE_ROWS = EXPECTED_MODELS * SECONDARY_CONFIGURATIONS_PER_MODEL * len(PROCEDURES)

MODEL_STATUS_FIELDS = (
    "protocol_id",
    "ordinal",
    "model_id",
    "architecture",
    "split",
    "nominal_training_seed",
    "training_status",
    "model_evaluable",
    "failure_stage",
    "checkpoint_sha256",
    "primary_fragment_sha256",
)
SECONDARY_COUNT_FIELDS = (
    "model_id",
    "budget_total",
    "replicate_id",
    "candidate_index",
    "client",
    "status",
    "n",
    "A",
    "K",
)
FAILURE_FIELDS = MODEL_STATUS_FIELDS + ("failure_reason",)

PRIMARY_ARTIFACTS = (
    "MODEL_STATUS.csv",
    "PROPOSAL_FAMILY_MANIFEST.csv.gz",
    "PRIMARY_COUNTS.csv.gz",
    "PRIMARY_CANDIDATE_DECISIONS.csv.gz",
    "PRIMARY_512_REP000.csv",
)
FINAL_ARTIFACTS = PRIMARY_ARTIFACTS + (
    "CAMPAIGN_PRIMARY_STATE.json",
    "PRIMARY_SEAL.json",
    "POST_DECISION_TRUTH.csv",
    "DENOMINATOR_LEDGER.json",
    "FAILURE_LEDGER.csv",
    "SECONDARY_COUNTS.csv.gz",
    "SECONDARY_CANDIDATE_DECISIONS.csv.gz",
    "SECONDARY_AUDIT_REPLICATES.csv.gz",
    "INDEPENDENT_REPLAY.json",
    "FINAL_VALIDATION.json",
    "NEGATIVE_RESULTS.md",
)


def _is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class VerifiedPrimarySeal:
    """In-memory authority obtained only by verifying the on-disk seal."""

    output_dir: Path
    primary_decision_file_sha256: str
    primary_candidate_decision_file_sha256: str
    primary_seal_sha256: str
    artifact_hashes: Mapping[str, str]


def _audit_sequence_from_manifest(
    value: Mapping[str, Any], *, expected_split: int
) -> str | None:
    split = value.get("split")
    if isinstance(split, bool) or not isinstance(split, int) or split != expected_split:
        raise ContractError("audit draw manifest split/model mismatch")
    if value.get("replicate_id") != PRIMARY_REPLICATE_ID:
        raise ContractError("primary audit draw manifest replicate drift")
    if value.get("status") == "model_failure":
        if value.get("clients") not in ([], ()):
            raise ContractError("model-failure audit manifest must not contain draws")
        return None
    if value.get("status") != "ok":
        raise ContractError("primary audit draw manifest status drift")
    clients = value.get("clients")
    if not isinstance(clients, list) or [row.get("client") for row in clients] != list(CLIENTS):
        raise ContractError("audit draw manifest clients must be unique and in frozen order")
    for row in clients:
        reservoir_size = row.get("reservoir_size")
        indices = row.get("indices")
        if (
            isinstance(reservoir_size, bool)
            or not isinstance(reservoir_size, int)
            or reservoir_size <= 0
            or not isinstance(indices, list)
            or len(indices) != 256
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= reservoir_size
                for index in indices
            )
        ):
            raise ContractError(
                "audit draw manifest must retain four in-range 256-index prefixes"
            )
        # Draws are registered with replacement.  Duplicate indices are valid;
        # uniqueness applies to the four client rows, whose exact order is
        # checked above, not to the sampled positions within a client.
    return canonical_json_sha256(
        {
            "split": split,
            "replicate_id": PRIMARY_REPLICATE_ID,
            "clients": [
                {"client": row["client"], "indices": row["indices"]} for row in clients
            ],
        }
    )


def _proposal_payload(fragment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "ordinal": fragment["ordinal"],
        "model": fragment["model"],
        "training_status": fragment["training_status"],
        "checkpoint_sha256": fragment.get("checkpoint_sha256"),
        "members": fragment["members"],
        "truth_opened": False,
    }


def _validate_proposal_payload(
    value: Mapping[str, Any], model: ModelCell, ordinal: int
) -> tuple[Any, ...]:
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("proposal fragment protocol mismatch")
    if value.get("ordinal") != ordinal or value.get("model") != _model_dict(model):
        raise ContractError(f"proposal fragment model/order mismatch for {model.model_id}")
    if value.get("truth_opened") is not False:
        raise ContractError("proposal fragment cannot contain opened truth")
    members = members_from_rows(value.get("members", ()))
    training_status = value.get("training_status")
    checkpoint = value.get("checkpoint_sha256")
    if training_status == "terminal_success":
        if not _is_hex64(checkpoint):
            raise ContractError("successful proposal fragment lacks its checkpoint hash")
    elif training_status == "terminal_model_failure":
        if checkpoint not in (None, ""):
            raise ContractError("failed proposal fragment cannot retain a checkpoint hash")
        if any(
            member.proposal_feasible
            or member.threshold is not None
            or member.proposal_A != 0
            or member.proposal_K != 0
            or member.proposal_risk is not None
            for member in members
        ):
            raise ContractError("failed proposal fragment must use reject-all placeholders")
    else:
        raise ContractError("unregistered terminal training status")
    return members


def _fsync_directory(path: Path) -> None:
    """Make newly renamed stage files durable before the next stage can begin."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_proposal_fragment_bundle(
    cell_dir: Path,
    fragment: Mapping[str, Any],
    *,
    protocol_dir: Path | None = None,
) -> Path:
    """Write and durably seal proposal-only state before any primary audit work."""

    models = load_model_cells(protocol_dir)
    ordinal = int(fragment.get("ordinal", -1))
    if ordinal not in range(len(models)):
        raise ContractError("proposal fragment ordinal is outside the frozen model grid")
    model = models[ordinal]
    proposal_payload = _proposal_payload(fragment)
    members = _validate_proposal_payload(proposal_payload, model, ordinal)
    root = Path(cell_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ContractError("proposal fragment bundle directory must be empty")
    proposal_path = root / "PROPOSAL_FRAGMENT.json"
    write_json(proposal_path, proposal_payload)
    proposal_seal = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": model.model_id,
        "status": "PROPOSAL_SEALED",
        "proposal_fragment_sha256": sha256_file(proposal_path),
        "proposal_manifest_sha256": proposal_manifest_sha256(members),
        "checkpoint_sha256": fragment.get("checkpoint_sha256"),
        "truth_opened": False,
    }
    proposal_seal_path = root / "PROPOSAL_SEAL.json"
    write_json(proposal_seal_path, proposal_seal)
    _fsync_directory(root)
    return proposal_seal_path


def append_primary_fragment_bundle(
    cell_dir: Path,
    fragment: Mapping[str, Any],
    audit_draw_manifest: Mapping[str, Any],
    *,
    protocol_dir: Path | None = None,
) -> Path:
    """Append primary-audit state without rewriting either proposal file."""

    models = load_model_cells(protocol_dir)
    ordinal = int(fragment.get("ordinal", -1))
    if ordinal not in range(len(models)):
        raise ContractError("primary fragment ordinal is outside the frozen model grid")
    model = models[ordinal]
    root = Path(cell_dir)
    if root.is_symlink() or not root.is_dir():
        raise ContractError("proposal fragment bundle directory is absent or unsafe")
    proposal_path = root / "PROPOSAL_FRAGMENT.json"
    proposal_seal_path = root / "PROPOSAL_SEAL.json"
    if {path.name for path in root.iterdir()} != {
        proposal_path.name,
        proposal_seal_path.name,
    }:
        raise ContractError("primary append requires exactly two sealed proposal files")
    proposal_bytes = proposal_path.read_bytes()
    proposal_seal_bytes = proposal_seal_path.read_bytes()
    proposal = read_json(proposal_path)
    proposal_seal = read_json(proposal_seal_path)
    members = _validate_proposal_payload(proposal, model, ordinal)
    if proposal != _proposal_payload(fragment):
        raise ContractError("primary fragment differs from its sealed proposal")
    if (
        proposal_seal.get("schema_version") != 1
        or proposal_seal.get("protocol_id") != PROTOCOL_ID
        or proposal_seal.get("model_id") != model.model_id
        or proposal_seal.get("status") != "PROPOSAL_SEALED"
        or proposal_seal.get("proposal_fragment_sha256") != sha256_file(proposal_path)
        or proposal_seal.get("proposal_manifest_sha256")
        != proposal_manifest_sha256(members)
        or proposal_seal.get("checkpoint_sha256") != fragment.get("checkpoint_sha256")
        or proposal_seal.get("truth_opened") is not False
    ):
        raise ContractError("proposal seal is invalid or does not bind the proposal fragment")
    if audit_draw_manifest.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("audit draw manifest protocol mismatch")
    if audit_draw_manifest.get("model_id") != model.model_id:
        raise ContractError("audit draw manifest model mismatch")
    sequence_hash = _audit_sequence_from_manifest(
        audit_draw_manifest, expected_split=model.split
    )
    decisions = [_candidate(row) for row in fragment.get("candidate_decisions", ())]
    evaluable = all(row.model_evaluable for row in decisions)
    if evaluable and (
        sequence_hash is None
        or any(row.audit_sequence_sha256 != sequence_hash for row in decisions)
    ):
        raise ContractError("candidate rows are not bound to the actual audit draw manifest")
    if not evaluable and sequence_hash is not None:
        raise ContractError("model failure cannot bind an audit sequence")
    _validate_primary_fragment(fragment, model, ordinal)
    audit_path = root / "AUDIT_DRAW_INDICES.json"
    write_json(audit_path, dict(audit_draw_manifest))
    primary_payload = {
        "protocol_id": PROTOCOL_ID,
        "model_id": model.model_id,
        "proposal_seal_sha256": sha256_file(proposal_seal_path),
        "audit_draw_manifest_sha256": sha256_file(audit_path),
        "audit_sequence_sha256": sequence_hash,
        "counts": fragment["counts"],
        "candidate_decisions": fragment["candidate_decisions"],
        "fragment_sha256": fragment["fragment_sha256"],
        "truth_opened": False,
    }
    primary_path = root / "PRIMARY_FRAGMENT.json"
    write_json(primary_path, primary_payload)
    stage = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": model.model_id,
        "stage": "PRIMARY_FRAGMENT_SEALED",
        "proposal_seal_sha256": sha256_file(proposal_seal_path),
        "audit_draw_manifest_sha256": sha256_file(audit_path),
        "primary_fragment_sha256": sha256_file(primary_path),
        "truth_opened": False,
        "secondary_started": False,
    }
    stage_path = root / "CELL_STAGE.json"
    write_json(stage_path, stage)
    final_seal = {
        **stage,
        "status": "PRIMARY_FRAGMENT_SEALED",
        "cell_stage_sha256": sha256_file(stage_path),
    }
    seal_path = root / "PRIMARY_FRAGMENT_SEAL.json"
    write_json(seal_path, final_seal)
    _fsync_directory(root)
    if (
        proposal_path.read_bytes() != proposal_bytes
        or proposal_seal_path.read_bytes() != proposal_seal_bytes
    ):
        raise ContractError("primary append rewrote immutable proposal files")
    return seal_path


def write_primary_fragment_bundle(
    cell_dir: Path,
    fragment: Mapping[str, Any],
    audit_draw_manifest: Mapping[str, Any],
    *,
    protocol_dir: Path | None = None,
) -> Path:
    """Compatibility wrapper for pre-reduced, PACS-free bundle producers.

    Scientific execution must call :func:`write_proposal_fragment_bundle`
    before constructing audit indices or audit-derived values, then call
    :func:`append_primary_fragment_bundle`.
    """

    write_proposal_fragment_bundle(cell_dir, fragment, protocol_dir=protocol_dir)
    return append_primary_fragment_bundle(
        cell_dir,
        fragment,
        audit_draw_manifest,
        protocol_dir=protocol_dir,
    )


def load_primary_fragment_bundle(cell_dir: Path) -> dict[str, Any]:
    """Verify actual per-cell files and return the reduced fragment payload."""

    root = Path(cell_dir)
    proposal_path = root / "PROPOSAL_FRAGMENT.json"
    proposal_seal_path = root / "PROPOSAL_SEAL.json"
    audit_path = root / "AUDIT_DRAW_INDICES.json"
    primary_path = root / "PRIMARY_FRAGMENT.json"
    stage_path = root / "CELL_STAGE.json"
    final_seal_path = root / "PRIMARY_FRAGMENT_SEAL.json"
    proposal = read_json(proposal_path)
    proposal_seal = read_json(proposal_seal_path)
    audit = read_json(audit_path)
    primary = read_json(primary_path)
    stage = read_json(stage_path)
    final_seal = read_json(final_seal_path)
    models = load_model_cells()
    ordinal = int(proposal.get("ordinal", -1))
    if ordinal not in range(len(models)):
        raise ContractError("sealed proposal ordinal is outside the frozen model grid")
    model = models[ordinal]
    members = _validate_proposal_payload(proposal, model, ordinal)
    if (
        proposal_seal.get("proposal_fragment_sha256") != sha256_file(proposal_path)
        or primary.get("proposal_seal_sha256") != sha256_file(proposal_seal_path)
        or primary.get("audit_draw_manifest_sha256") != sha256_file(audit_path)
        or stage.get("primary_fragment_sha256") != sha256_file(primary_path)
        or final_seal.get("cell_stage_sha256") != sha256_file(stage_path)
    ):
        raise ContractError("per-cell primary fragment file-hash chain mismatch")
    if (
        proposal_seal.get("schema_version") != 1
        or proposal_seal.get("protocol_id") != PROTOCOL_ID
        or proposal_seal.get("model_id") != model.model_id
        or proposal_seal.get("status") != "PROPOSAL_SEALED"
        or proposal_seal.get("proposal_manifest_sha256")
        != proposal_manifest_sha256(members)
        or proposal_seal.get("checkpoint_sha256") != proposal.get("checkpoint_sha256")
        or stage.get("stage") != "PRIMARY_FRAGMENT_SEALED"
        or final_seal.get("status") != "PRIMARY_FRAGMENT_SEALED"
        or any(value.get("truth_opened") is not False for value in (proposal, proposal_seal, primary, stage, final_seal))
        or stage.get("secondary_started") is not False
    ):
        raise ContractError("per-cell stage state permits premature truth/secondary access")
    if (
        audit.get("protocol_id") != PROTOCOL_ID
        or audit.get("model_id") != model.model_id
    ):
        raise ContractError("audit draw manifest protocol/model mismatch")
    sequence_hash = _audit_sequence_from_manifest(audit, expected_split=model.split)
    if primary.get("audit_sequence_sha256") != sequence_hash:
        raise ContractError("audit draw manifest serialization/hash mismatch")
    fragment = {
        "protocol_id": proposal["protocol_id"],
        "ordinal": proposal["ordinal"],
        "model": proposal["model"],
        "training_status": proposal["training_status"],
        "checkpoint_sha256": proposal.get("checkpoint_sha256"),
        "members": proposal["members"],
        "counts": primary["counts"],
        "candidate_decisions": primary["candidate_decisions"],
        "fragment_sha256": primary["fragment_sha256"],
        "_bundle_receipt": {
            "root": str(root.resolve()),
            "proposal_seal_file_sha256": sha256_file(proposal_seal_path),
            "primary_fragment_seal_sha256": sha256_file(final_seal_path),
        },
    }
    if fragment["fragment_sha256"] != primary_fragment_sha256(fragment):
        raise ContractError("reconstructed primary fragment hash mismatch")
    return fragment


def _model_dict(model: ModelCell) -> dict[str, Any]:
    value = asdict(model)
    value["known_classes"] = list(model.known_classes)
    value["unknown_classes"] = list(model.unknown_classes)
    return value


def primary_fragment_sha256(fragment: Mapping[str, Any]) -> str:
    """Hash the implementation-level primary-fragment payload."""

    return canonical_json_sha256(
        {
            "protocol_id": fragment.get("protocol_id"),
            "ordinal": fragment.get("ordinal"),
            "model": fragment.get("model"),
            "training_status": fragment.get("training_status"),
            "checkpoint_sha256": fragment.get("checkpoint_sha256"),
            "members": fragment.get("members"),
            "counts": fragment.get("counts"),
            "candidate_decisions": fragment.get("candidate_decisions"),
        }
    )


def truth_fragment_sha256(fragment: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "protocol_id": fragment.get("protocol_id"),
            "model_id": fragment.get("model_id"),
            "primary_seal_sha256": fragment.get("primary_seal_sha256"),
            "rows": fragment.get("rows"),
        }
    )


def secondary_fragment_sha256(fragment: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "protocol_id": fragment.get("protocol_id"),
            "model_id": fragment.get("model_id"),
            "primary_seal_sha256": fragment.get("primary_seal_sha256"),
            "counts": fragment.get("counts"),
            "candidate_decisions": fragment.get("candidate_decisions"),
        }
    )


def _candidate(value: Mapping[str, Any]) -> CandidateDecision:
    try:
        return CandidateDecision(**value)
    except (TypeError, ValueError) as exc:
        raise ContractError("candidate fragment does not match CandidateDecision schema") from exc


def _validate_outcome_schema(protocol_dir: Path | None) -> Path:
    root = default_protocol_dir() if protocol_dir is None else Path(protocol_dir)
    path = root / "OUTCOME_SCHEMA.json"
    if sha256_file(path) != OUTCOME_SCHEMA_SHA256:
        raise ContractError("sealed OUTCOME_SCHEMA.json hash drift")
    schema = read_json(path)
    observed = (
        schema.get("protocol_id"),
        schema.get("expected_primary_rows"),
        schema.get("primary_candidate_decision_file", {}).get("expected_rows"),
        schema.get("secondary_procedure_file", {}).get("expected_rows"),
        schema.get("secondary_candidate_decision_file", {}).get("expected_rows"),
        schema.get("secondary_count_file", {}).get("expected_rows"),
    )
    expected = (
        PROTOCOL_ID,
        EXPECTED_PRIMARY_PROCEDURE_ROWS,
        EXPECTED_PRIMARY_CANDIDATE_ROWS,
        EXPECTED_SECONDARY_PROCEDURE_ROWS,
        EXPECTED_SECONDARY_CANDIDATE_ROWS,
        EXPECTED_SECONDARY_COUNT_ROWS,
    )
    if observed != expected:
        raise ContractError("sealed outcome denominator drift")
    return path


def _validate_primary_fragment(
    fragment: Mapping[str, Any], model: ModelCell, ordinal: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[CandidateDecision], dict[str, Any]]:
    if fragment.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("primary fragment protocol mismatch")
    if fragment.get("ordinal") != ordinal or fragment.get("model") != _model_dict(model):
        raise ContractError(f"primary fragment model/order mismatch for {model.model_id}")
    if fragment.get("fragment_sha256") != primary_fragment_sha256(fragment):
        raise ContractError(f"primary fragment hash mismatch for {model.model_id}")
    members = members_from_rows(fragment.get("members", ()))
    proposal_rows = [{"model_id": model.model_id, **member.as_row()} for member in members]
    counts = validate_count_rows(
        fragment.get("counts", ()), expected_model_id=model.model_id
    )
    decisions = [_candidate(row) for row in fragment.get("candidate_decisions", ())]
    if len(decisions) != M * len(PROCEDURES):
        raise ContractError(f"primary candidate denominator drift for {model.model_id}")
    validate_hsb_invariants(decisions)
    if any(
        row.model_id != model.model_id
        or row.budget_total != PRIMARY_BUDGET_TOTAL
        or row.replicate_id != PRIMARY_REPLICATE_ID
        for row in decisions
    ):
        raise ContractError(f"primary candidate binding mismatch for {model.model_id}")
    training_status = fragment.get("training_status")
    model_evaluable = all(row.model_evaluable for row in decisions)
    if training_status not in ("terminal_success", "terminal_model_failure"):
        raise ContractError("unregistered terminal training status")
    if model_evaluable != (training_status == "terminal_success"):
        raise ContractError("training status and candidate evaluability disagree")
    if model_evaluable:
        validate_count_rows(counts, expected_model_id=model.model_id, expected_n=128)
        checkpoint = str(fragment.get("checkpoint_sha256") or "")
        if not _is_hex64(checkpoint) or any(row.checkpoint_sha256 != checkpoint for row in decisions):
            raise ContractError("checkpoint hash mismatch within primary fragment")
        proposal_manifest_hash = proposal_manifest_sha256(members)
        # The public v3 schema retains the historical field name
        # ``proposal_seal_sha256``.  Its frozen meaning is the canonical
        # proposal-family manifest hash.  The actual PROPOSAL_SEAL.json file
        # hash is bound separately in the per-cell receipt and campaign state.
        if any(
            row.proposal_seal_sha256 != proposal_manifest_hash for row in decisions
        ):
            raise ContractError("proposal-manifest hash mismatch within primary fragment")
        tensor_hash = count_tensor_sha256(counts)
        if any(row.count_tensor_sha256 != tensor_hash for row in decisions):
            raise ContractError("count tensor hash mismatch within primary fragment")
        audit_hashes = {row.audit_sequence_sha256 for row in decisions}
        if len(audit_hashes) != 1 or not _is_hex64(next(iter(audit_hashes))):
            raise ContractError("audit sequence hash mismatch within primary fragment")
    else:
        if fragment.get("checkpoint_sha256") not in (None, ""):
            raise ContractError("terminal model failure cannot retain a checkpoint hash")
        if any(str(row["status"]) != "model_failure" for row in counts):
            raise ContractError("terminal model failure lacks count placeholders")
    status_row = {
        "protocol_id": PROTOCOL_ID,
        "ordinal": ordinal,
        "model_id": model.model_id,
        "architecture": model.architecture,
        "split": model.split,
        "nominal_training_seed": model.nominal_training_seed,
        "training_status": training_status,
        "model_evaluable": model_evaluable,
        "failure_stage": "none" if model_evaluable else "training",
        "checkpoint_sha256": fragment.get("checkpoint_sha256"),
        "primary_fragment_sha256": fragment["fragment_sha256"],
    }
    return proposal_rows, counts, decisions, status_row


def _sort_primary(
    rows: Iterable[Mapping[str, Any]], models: Sequence[ModelCell], kind: str
) -> list[dict[str, Any]]:
    order = {model.model_id: index for index, model in enumerate(models)}
    procedure = {name: index for index, name in enumerate(PROCEDURES)}
    if kind == "proposal":
        key = lambda row: (order[str(row["model_id"])], int(row["candidate_index"]))
    elif kind == "count":
        key = lambda row: (
            order[str(row["model_id"])], int(row["candidate_index"]), CLIENTS.index(str(row["client"]))
        )
    elif kind == "candidate":
        key = lambda row: (
            order[str(row["model_id"])], int(row["candidate_index"]), procedure[str(row["procedure"])]
        )
    else:
        raise ValueError(kind)
    return [dict(row) for row in sorted(rows, key=key)]


def seal_primary_campaign(
    output_dir: Path,
    fragments: Sequence[Mapping[str, Any]],
    *,
    protocol_dir: Path | None = None,
) -> VerifiedPrimarySeal:
    """Validate all 30 fragments and write the global primary seal last."""

    verify_scientific_contract(protocol_dir)
    schema_path = _validate_outcome_schema(protocol_dir)
    output = Path(output_dir)
    existing = output / "PRIMARY_SEAL.json"
    if existing.exists():
        return verify_primary_seal(output, protocol_dir=protocol_dir)
    models = load_model_cells(protocol_dir)
    if len(fragments) != EXPECTED_MODELS:
        raise ContractError(
            f"all {EXPECTED_MODELS} primary fragments are required before sealing"
        )
    by_model: dict[str, Mapping[str, Any]] = {}
    for fragment in fragments:
        receipt = fragment.get("_bundle_receipt")
        if not isinstance(receipt, Mapping):
            raise ContractError("unsealed in-memory primary fragment is forbidden")
        bundle_root = Path(str(receipt.get("root", "")))
        seal_path = bundle_root / "PRIMARY_FRAGMENT_SEAL.json"
        if (
            not seal_path.is_file()
            or receipt.get("primary_fragment_seal_sha256") != sha256_file(seal_path)
            or receipt.get("proposal_seal_file_sha256")
            != sha256_file(bundle_root / "PROPOSAL_SEAL.json")
        ):
            raise ContractError("primary fragment bundle receipt/hash mismatch")
        reloaded = load_primary_fragment_bundle(bundle_root)
        if (
            reloaded.get("fragment_sha256") != fragment.get("fragment_sha256")
            or reloaded.get("_bundle_receipt") != receipt
        ):
            raise ContractError("primary fragment differs from its sealed bundle")
        model_id = str(fragment.get("model", {}).get("model_id", ""))
        if model_id in by_model:
            raise ContractError(f"duplicate primary fragment for {model_id}")
        by_model[model_id] = fragment
    if set(by_model) != {model.model_id for model in models}:
        raise ContractError("primary fragment model set does not equal the frozen 30-cell set")

    proposal_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    candidate_objects: dict[str, list[CandidateDecision]] = {}
    status_rows: list[dict[str, Any]] = []
    for ordinal, model in enumerate(models):
        proposals, counts, candidates, status = _validate_primary_fragment(
            by_model[model.model_id], model, ordinal
        )
        proposal_rows.extend(proposals)
        count_rows.extend(counts)
        candidate_objects[model.model_id] = candidates
        status_rows.append(status)
    candidate_rows = [
        row.as_row() for model in models for row in candidate_objects[model.model_id]
    ]
    proposal_rows = _sort_primary(proposal_rows, models, "proposal")
    count_rows = _sort_primary(count_rows, models, "count")
    candidate_rows = _sort_primary(candidate_rows, models, "candidate")
    if (len(proposal_rows), len(count_rows), len(candidate_rows)) != (
        EXPECTED_PROPOSAL_ROWS,
        EXPECTED_PRIMARY_COUNT_ROWS,
        EXPECTED_PRIMARY_CANDIDATE_ROWS,
    ):
        raise ContractError("campaign primary denominator drift")
    assert_unique_keys(proposal_rows, ("model_id", "candidate_index"))
    assert_unique_keys(count_rows, ("model_id", "candidate_index", "client"))
    assert_unique_keys(candidate_rows, ("model_id", "candidate_index", "procedure"))

    write_csv(output / "MODEL_STATUS.csv", status_rows, MODEL_STATUS_FIELDS)
    write_csv(output / "PROPOSAL_FAMILY_MANIFEST.csv.gz", proposal_rows, PROPOSAL_FIELDS, compressed=True)
    write_csv(output / "PRIMARY_COUNTS.csv.gz", count_rows, COUNT_FIELDS, compressed=True)
    write_csv(
        output / "PRIMARY_CANDIDATE_DECISIONS.csv.gz",
        candidate_rows,
        CANDIDATE_FIELDS,
        compressed=True,
    )
    candidate_hash = sha256_file(output / "PRIMARY_CANDIDATE_DECISIONS.csv.gz")
    primary_rows = [
        row
        for model in models
        for row in procedure_rows(
            model,
            candidate_objects[model.model_id],
            candidate_decision_file_sha256=candidate_hash,
        )
    ]
    if len(primary_rows) != EXPECTED_PRIMARY_PROCEDURE_ROWS:
        raise ContractError("campaign primary procedure denominator drift")
    assert_unique_keys(primary_rows, ("model_id", "procedure"))
    write_csv(output / "PRIMARY_512_REP000.csv", primary_rows, PRIMARY_FIELDS)
    artifact_hashes = {name: sha256_file(output / name) for name in PRIMARY_ARTIFACTS}
    primary_hash = artifact_hashes["PRIMARY_512_REP000.csv"]
    campaign_state = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage": "PRIMARY_FRAGMENTS_VALIDATED",
        "truth_opened": False,
        "secondary_started": False,
        "primary_artifact_hashes": artifact_hashes,
        "cell_fragment_seal_sha256": {
            model.model_id: by_model[model.model_id]["_bundle_receipt"][
                "primary_fragment_seal_sha256"
            ]
            for model in models
        },
        "cell_proposal_seal_file_sha256": {
            model.model_id: by_model[model.model_id]["_bundle_receipt"][
                "proposal_seal_file_sha256"
            ]
            for model in models
        },
    }
    campaign_state_path = output / "CAMPAIGN_PRIMARY_STATE.json"
    write_json(campaign_state_path, campaign_state)
    seal = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PRIMARY_SEALED",
        "truth_and_secondary_authorized": True,
        "outcome_schema_sha256": sha256_file(schema_path),
        "primary_artifact_hashes": artifact_hashes,
        "campaign_primary_state_sha256": sha256_file(campaign_state_path),
        "primary_decision_file_sha256": primary_hash,
        "primary_candidate_decision_file_sha256": candidate_hash,
        "primary_fragment_sha256": {
            model.model_id: by_model[model.model_id]["fragment_sha256"] for model in models
        },
        "row_counts": {
            "models": len(status_rows),
            "proposal": len(proposal_rows),
            "counts": len(count_rows),
            "candidates": len(candidate_rows),
            "procedures": len(primary_rows),
        },
        "primary_written_and_fsynced_before_seal": True,
        "truth_opened": False,
    }
    write_json(existing, seal)
    return verify_primary_seal(output, protocol_dir=protocol_dir)


def verify_primary_seal(
    output_dir: Path, *, protocol_dir: Path | None = None
) -> VerifiedPrimarySeal:
    """Re-hash every immutable primary artifact before granting authority."""

    _validate_outcome_schema(protocol_dir)
    output = Path(output_dir)
    path = output / "PRIMARY_SEAL.json"
    seal = read_json(path)
    expected_seal_keys = {
        "schema_version", "protocol_id", "status", "truth_and_secondary_authorized",
        "outcome_schema_sha256", "primary_artifact_hashes",
        "campaign_primary_state_sha256", "primary_decision_file_sha256",
        "primary_candidate_decision_file_sha256", "primary_fragment_sha256",
        "row_counts", "primary_written_and_fsynced_before_seal", "truth_opened",
    }
    if set(seal) != expected_seal_keys or seal.get("schema_version") != 1:
        raise ContractError("primary seal schema/key drift")
    if (
        seal.get("protocol_id") != PROTOCOL_ID
        or seal.get("status") != "PRIMARY_SEALED"
        or seal.get("truth_and_secondary_authorized") is not True
        or seal.get("truth_opened") is not False
        or seal.get("outcome_schema_sha256") != OUTCOME_SCHEMA_SHA256
    ):
        raise ContractError("primary seal does not authorize post-primary work")
    expected_row_counts = {
        "models": EXPECTED_MODELS,
        "proposal": EXPECTED_PROPOSAL_ROWS,
        "counts": EXPECTED_PRIMARY_COUNT_ROWS,
        "candidates": EXPECTED_PRIMARY_CANDIDATE_ROWS,
        "procedures": EXPECTED_PRIMARY_PROCEDURE_ROWS,
    }
    if seal.get("row_counts") != expected_row_counts:
        raise ContractError("primary seal row-count denominator drift")
    if seal.get("primary_written_and_fsynced_before_seal") is not True:
        raise ContractError("primary seal lacks the write-before-seal assertion")
    models = load_model_cells(protocol_dir)
    fragment_hashes = seal.get("primary_fragment_sha256")
    if (
        not isinstance(fragment_hashes, Mapping)
        or set(fragment_hashes) != {model.model_id for model in models}
        or any(not _is_hex64(value) for value in fragment_hashes.values())
    ):
        raise ContractError("primary seal fragment inventory drift")
    expected = seal.get("primary_artifact_hashes")
    if not isinstance(expected, Mapping) or set(expected) != set(PRIMARY_ARTIFACTS):
        raise ContractError("primary seal artifact inventory drift")
    for name in PRIMARY_ARTIFACTS:
        if sha256_file(output / name) != expected[name]:
            raise ContractError(f"immutable primary artifact changed after sealing: {name}")
    state_path = output / "CAMPAIGN_PRIMARY_STATE.json"
    state = read_json(state_path)
    expected_state_keys = {
        "schema_version", "protocol_id", "stage", "truth_opened",
        "secondary_started", "primary_artifact_hashes", "cell_fragment_seal_sha256",
        "cell_proposal_seal_file_sha256",
    }
    cell_seals = state.get("cell_fragment_seal_sha256")
    proposal_file_seals = state.get("cell_proposal_seal_file_sha256")
    if (
        set(state) != expected_state_keys
        or state.get("schema_version") != 1
        or state.get("protocol_id") != PROTOCOL_ID
        or seal.get("campaign_primary_state_sha256") != sha256_file(state_path)
        or state.get("stage") != "PRIMARY_FRAGMENTS_VALIDATED"
        or state.get("truth_opened") is not False
        or state.get("secondary_started") is not False
        or state.get("primary_artifact_hashes") != expected
        or not isinstance(cell_seals, Mapping)
        or set(cell_seals) != {model.model_id for model in models}
        or any(not _is_hex64(value) for value in cell_seals.values())
        or not isinstance(proposal_file_seals, Mapping)
        or set(proposal_file_seals) != {model.model_id for model in models}
        or any(not _is_hex64(value) for value in proposal_file_seals.values())
    ):
        raise ContractError("campaign-global primary stage/hash state mismatch")
    primary_hash = str(expected["PRIMARY_512_REP000.csv"])
    candidate_hash = str(expected["PRIMARY_CANDIDATE_DECISIONS.csv.gz"])
    if (
        seal.get("primary_decision_file_sha256") != primary_hash
        or seal.get("primary_candidate_decision_file_sha256") != candidate_hash
    ):
        raise ContractError("primary seal hash aliases disagree")
    return VerifiedPrimarySeal(
        output.resolve(), primary_hash, candidate_hash, sha256_file(path), dict(expected)
    )


def _registered_secondary_configs() -> set[tuple[int, int]]:
    return {
        (budget, replicate)
        for budget in (256, 512, 1024)
        for replicate in range(100)
        if (budget, replicate) != (PRIMARY_BUDGET_TOTAL, PRIMARY_REPLICATE_ID)
    }


def validate_secondary_fragment(
    fragment: Mapping[str, Any],
    model: ModelCell,
    authority: VerifiedPrimarySeal,
) -> tuple[list[dict[str, Any]], list[CandidateDecision]]:
    if fragment.get("protocol_id") != PROTOCOL_ID or fragment.get("model_id") != model.model_id:
        raise ContractError("secondary fragment protocol/model mismatch")
    if fragment.get("primary_seal_sha256") != authority.primary_seal_sha256:
        raise ContractError("secondary fragment does not bind the verified primary seal")
    if fragment.get("fragment_sha256") != secondary_fragment_sha256(fragment):
        raise ContractError("secondary fragment hash mismatch")
    counts = [dict(row) for row in fragment.get("counts", ())]
    candidates = [_candidate(row) for row in fragment.get("candidate_decisions", ())]
    if len(counts) != SECONDARY_CONFIGURATIONS_PER_MODEL * M * len(CLIENTS):
        raise ContractError("secondary count denominator drift")
    if len(candidates) != SECONDARY_CONFIGURATIONS_PER_MODEL * M * len(PROCEDURES):
        raise ContractError("secondary candidate denominator drift")
    expected_configs = _registered_secondary_configs()
    count_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    candidate_groups: dict[tuple[int, int], list[CandidateDecision]] = defaultdict(list)
    for row in counts:
        if row.get("model_id") != model.model_id:
            raise ContractError("secondary count model mismatch")
        config = (int(row["budget_total"]), int(row["replicate_id"]))
        count_groups[config].append(row)
    for row in candidates:
        if row.model_id != model.model_id:
            raise ContractError("secondary candidate model mismatch")
        candidate_groups[(row.budget_total, row.replicate_id)].append(row)
    if set(count_groups) != expected_configs or set(candidate_groups) != expected_configs:
        raise ContractError("secondary registered configuration grid drift")
    for config in sorted(expected_configs):
        budget, _ = config
        normalized = validate_count_rows(
            count_groups[config],
            expected_model_id=model.model_id,
            expected_n=budget // len(CLIENTS) if count_groups[config][0]["status"] == "ok" else None,
        )
        count_groups[config] = [
            {**row, "budget_total": budget, "replicate_id": config[1]} for row in normalized
        ]
        validate_hsb_invariants(candidate_groups[config])
        if any(
            row.budget_total != budget or row.replicate_id != config[1]
            for row in candidate_groups[config]
        ):
            raise ContractError("secondary candidate configuration mismatch")
        evaluable = all(row.model_evaluable for row in candidate_groups[config])
        if evaluable:
            tensor = count_tensor_sha256(count_groups[config])
            if any(row.count_tensor_sha256 != tensor for row in candidate_groups[config]):
                raise ContractError("secondary count tensor hash mismatch")
            if any(
                not _is_hex64(row.checkpoint_sha256)
                or not _is_hex64(row.proposal_seal_sha256)
                or not _is_hex64(row.audit_sequence_sha256)
                for row in candidate_groups[config]
            ):
                raise ContractError("secondary provenance hash is not lowercase hex64")
    evaluable_rows = [row for row in candidates if row.model_evaluable]
    if evaluable_rows:
        primary_candidates = [
            row
            for row in read_csv(
                authority.output_dir / "PRIMARY_CANDIDATE_DECISIONS.csv.gz"
            )
            if row["model_id"] == model.model_id
        ]
        if len(primary_candidates) != M * len(PROCEDURES):
            raise ContractError("secondary fragment cannot resolve primary model provenance")
        expected_checkpoint = {row["checkpoint_sha256"] for row in primary_candidates}
        expected_proposal = {row["proposal_seal_sha256"] for row in primary_candidates}
        if (
            len(expected_checkpoint) != 1
            or len(expected_proposal) != 1
            or {row.checkpoint_sha256 for row in evaluable_rows} != expected_checkpoint
            or {row.proposal_seal_sha256 for row in evaluable_rows} != expected_proposal
        ):
            raise ContractError("secondary fragment provenance differs from sealed primary")
    flattened_counts = [row for key in sorted(expected_configs) for row in count_groups[key]]
    flattened_candidates = [row for key in sorted(expected_configs) for row in candidate_groups[key]]
    return flattened_counts, flattened_candidates


def _validate_truth_fragments(
    fragments: Sequence[Mapping[str, Any]],
    models: Sequence[ModelCell],
    authority: VerifiedPrimarySeal,
    primary_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    if len(fragments) != EXPECTED_MODELS:
        raise ContractError("all 30 truth fragments are required")
    by_model = {str(fragment.get("model_id")): fragment for fragment in fragments}
    if len(by_model) != EXPECTED_MODELS or set(by_model) != {model.model_id for model in models}:
        raise ContractError("truth fragment model set drift")
    primary_by_key = {(row["model_id"], row["procedure"]): row for row in primary_rows}
    output: list[dict[str, Any]] = []
    for model in models:
        fragment = by_model[model.model_id]
        if (
            fragment.get("protocol_id") != PROTOCOL_ID
            or fragment.get("primary_seal_sha256") != authority.primary_seal_sha256
            or fragment.get("fragment_sha256") != truth_fragment_sha256(fragment)
        ):
            raise ContractError("truth fragment seal/hash mismatch")
        rows = [dict(row) for row in fragment.get("rows", ())]
        if len(rows) != len(PROCEDURES) or tuple(row.get("procedure") for row in rows) != PROCEDURES:
            raise ContractError("truth fragment must contain ordered H/S/B rows")
        for row in rows:
            key = (model.model_id, str(row.get("procedure")))
            primary = primary_by_key[key]
            if row.get("model_id") != model.model_id:
                raise ContractError("truth row model mismatch")
            if row.get("primary_decision_file_sha256") != authority.primary_decision_file_sha256:
                raise ContractError("truth row does not reference sealed primary decisions")
            if row.get("primary_candidate_decision_file_sha256") != authority.primary_candidate_decision_file_sha256:
                raise ContractError("truth row does not reference sealed candidate decisions")
            if int(row.get("selected_candidate", -2)) != int(primary["selected_candidate"]):
                raise ContractError("truth row selected candidate differs from primary")
            evaluable = primary["model_evaluable"].lower() == "true"
            if evaluable and row.get("truth_opened_after_primary_hash") is not True:
                raise ContractError("evaluable truth row lacks post-seal authorization marker")
            certified = primary["certified"].lower() == "true"
            refusal = primary["refusal_reason"]
            if not evaluable:
                expected_reason = "model_failure"
            elif refusal == "proposal_family_all_infeasible":
                expected_reason = "proposal_infeasible"
            elif not certified:
                expected_reason = "certificate_refusal"
            else:
                expected_reason = "none"
            if row.get("truth_unavailable_reason") != expected_reason:
                raise ContractError("truth-unavailable reason disagrees with primary decision")
            unavailable = expected_reason != "none"
            observed_values = (
                row.get("true_worst_client_risk"),
                row.get("true_min_client_acceptance"),
                row.get("validity_failure"),
            )
            if unavailable and any(value not in (None, "") for value in observed_values):
                raise ContractError("unavailable truth row contains opened truth values")
            if not unavailable:
                risk, acceptance, invalid = observed_values
                if (
                    risk in (None, "")
                    or acceptance in (None, "")
                    or not 0.0 <= float(risk) <= 1.0
                    or not 0.0 <= float(acceptance) <= 1.0
                    or not isinstance(invalid, bool)
                ):
                    raise ContractError("certified truth row has invalid truth values")
                expected_invalid = bool(
                    float(risk) > ALPHA
                    or float(primary["coverage_lcb"]) > float(acceptance)
                )
                if invalid is not expected_invalid:
                    raise ContractError("truth validity flag disagrees with registered rule")
        output.extend(rows)
    assert_unique_keys(output, ("model_id", "procedure"))
    return output


def _scientific_verdict(primary_rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    by_procedure = {
        procedure: [row for row in primary_rows if str(row["procedure"]) == procedure]
        for procedure in PROCEDURES
    }
    if any(len(rows) != EXPECTED_MODELS for rows in by_procedure.values()):
        raise ContractError("verdict requires 30 rows for each procedure")
    truthy = lambda value: value is True or str(value).lower() == "true"
    h = by_procedure["H"]
    certified_h = [row for row in h if truthy(row["certified"])]
    h_absolute = {
        "certified_at_least_6": len(certified_h) >= 6,
        "both_architectures": len({row["architecture"] for row in certified_h}) == 2,
        "at_least_3_splits": len({int(row["split"]) for row in certified_h}) >= 3,
        "mean_H_ECA_at_least_0_05": float(np.mean([float(row["ECA"]) for row in h])) >= 0.05,
    }
    keyed = {
        (str(row["model_id"]), str(row["procedure"])): row for row in primary_rows
    }
    differences = {
        model_id: float(keyed[(model_id, "S")]["ECA"]) - float(keyed[(model_id, "B")]["ECA"])
        for model_id in {str(row["model_id"]) for row in h}
    }
    architecture = {
        name: [differences[str(row["model_id"])] for row in h if row["architecture"] == name]
        for name in ("resnet18", "convnext_tiny")
    }
    gain = {
        "mean_S_minus_B_at_least_0_01": float(np.mean(list(differences.values()))) >= 0.01,
        "architecture_means_nonnegative": all(float(np.mean(values)) >= 0.0 for values in architecture.values()),
        "strict_positive_cell_each_architecture": all(any(value > 0.0 for value in values) for values in architecture.values()),
    }
    conditional = {}
    for procedure, rows in by_procedure.items():
        certified_rows = [row for row in rows if truthy(row["certified"])]
        conditional[procedure] = {
            "denominator": len(certified_rows),
            "CondCertCov": (
                None
                if not certified_rows
                else float(np.mean([float(row["coverage_lcb"]) for row in certified_rows]))
            ),
        }
    if not all(h_absolute.values()):
        verdict = "VALID_NEGATIVE"
    elif all(gain.values()):
        verdict = "PASS_FRESH_SEED_REPLICATION"
    else:
        verdict = "PASS_CORE_ONLY"
    return verdict, {
        "H_absolute_gates": h_absolute,
        "gain_gates": gain,
        "H_certified": len(certified_h),
        "mean_ECA": {
            procedure: float(np.mean([float(row["ECA"]) for row in rows]))
            for procedure, rows in by_procedure.items()
        },
        "conditional_certified_coverage": conditional,
        "mean_S_minus_B": float(np.mean(list(differences.values()))),
        "architecture_mean_S_minus_B": {
            name: float(np.mean(values)) for name, values in architecture.items()
        },
    }


def _negative_results_text(verdict: str, metrics: Mapping[str, Any]) -> str:
    failed_h = [key for key, value in metrics["H_absolute_gates"].items() if not value]
    failed_gain = [key for key, value in metrics["gain_gates"].items() if not value]
    return (
        "# Registered negative and mixed outcomes\n\n"
        f"Final registered verdict: `{verdict}`.\n\n"
        f"Failed absolute H gates: {', '.join(failed_h) if failed_h else 'none'}.\n\n"
        f"Failed S-minus-B gain gates: {', '.join(failed_gain) if failed_gain else 'none'}.\n\n"
        "Terminal model failures, proposal-infeasible cells, certificate refusals, and "
        "zero certified denominators remain in the machine-readable ledgers; none are "
        "removed from the planned denominator of 30 models.\n"
    )


def _write_sha256_manifest(output: Path, names: Sequence[str]) -> dict[str, str]:
    hashes = {name: sha256_file(output / name) for name in sorted(names)}
    _atomic_bytes(
        output / "SHA256SUMS.txt",
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()).encode("utf-8"),
    )
    return hashes


def finalize_campaign(
    output_dir: Path,
    truth_fragments: Sequence[Mapping[str, Any]],
    secondary_fragments: Sequence[Mapping[str, Any]],
    *,
    protocol_dir: Path | None = None,
) -> dict[str, Any]:
    """Aggregate post-seal truth and all 299 secondary configurations/model."""

    authority = verify_primary_seal(output_dir, protocol_dir=protocol_dir)
    output = authority.output_dir
    if (output / "FINAL_VALIDATION.json").exists() or (output / "SHA256SUMS.txt").exists():
        raise ContractError("campaign is already finalized and immutable")
    models = load_model_cells(protocol_dir)
    primary_rows = read_csv(output / "PRIMARY_512_REP000.csv")
    truth_rows = _validate_truth_fragments(truth_fragments, models, authority, primary_rows)
    if len(secondary_fragments) != EXPECTED_MODELS:
        raise ContractError("all 30 secondary fragments are required")
    by_model = {str(fragment.get("model_id")): fragment for fragment in secondary_fragments}
    if len(by_model) != EXPECTED_MODELS or set(by_model) != {model.model_id for model in models}:
        raise ContractError("secondary fragment model set drift")
    all_counts: list[dict[str, Any]] = []
    all_candidates: dict[str, list[CandidateDecision]] = {}
    for model in models:
        counts, candidates = validate_secondary_fragment(by_model[model.model_id], model, authority)
        all_counts.extend(counts)
        all_candidates[model.model_id] = candidates
    model_order = {model.model_id: index for index, model in enumerate(models)}
    procedure_order = {name: index for index, name in enumerate(PROCEDURES)}
    all_counts.sort(
        key=lambda row: (
            model_order[str(row["model_id"])], int(row["budget_total"]),
            int(row["replicate_id"]), int(row["candidate_index"]),
            CLIENTS.index(str(row["client"])),
        )
    )
    candidate_rows = [row.as_row() for model in models for row in all_candidates[model.model_id]]
    candidate_rows.sort(
        key=lambda row: (
            model_order[str(row["model_id"])], int(row["budget_total"]),
            int(row["replicate_id"]), int(row["candidate_index"]),
            procedure_order[str(row["procedure"])],
        )
    )
    if len(all_counts) != EXPECTED_SECONDARY_COUNT_ROWS or len(candidate_rows) != EXPECTED_SECONDARY_CANDIDATE_ROWS:
        raise ContractError("campaign secondary denominator drift")
    assert_unique_keys(all_counts, ("model_id", "budget_total", "replicate_id", "candidate_index", "client"))
    assert_unique_keys(candidate_rows, ("model_id", "budget_total", "replicate_id", "candidate_index", "procedure"))

    write_csv(output / "POST_DECISION_TRUTH.csv", truth_rows, TRUTH_FIELDS)
    write_csv(output / "SECONDARY_COUNTS.csv.gz", all_counts, SECONDARY_COUNT_FIELDS, compressed=True)
    write_csv(
        output / "SECONDARY_CANDIDATE_DECISIONS.csv.gz",
        candidate_rows,
        CANDIDATE_FIELDS,
        compressed=True,
    )
    secondary_candidate_hash = sha256_file(output / "SECONDARY_CANDIDATE_DECISIONS.csv.gz")
    secondary_rows: list[dict[str, Any]] = []
    for model in models:
        groups: dict[tuple[int, int], list[CandidateDecision]] = defaultdict(list)
        for row in all_candidates[model.model_id]:
            groups[(row.budget_total, row.replicate_id)].append(row)
        for config in sorted(_registered_secondary_configs()):
            rows = groups[config]
            rows.sort(key=lambda row: (PROCEDURES.index(row.procedure), row.candidate_index))
            secondary_rows.extend(
                procedure_rows(
                    model, rows, candidate_decision_file_sha256=secondary_candidate_hash
                )
            )
    if len(secondary_rows) != EXPECTED_SECONDARY_PROCEDURE_ROWS:
        raise ContractError("secondary procedure denominator drift")
    write_csv(
        output / "SECONDARY_AUDIT_REPLICATES.csv.gz",
        secondary_rows,
        PRIMARY_FIELDS,
        compressed=True,
    )

    status_rows = read_csv(output / "MODEL_STATUS.csv")
    failures = [
        {**row, "failure_reason": "terminal_model_failure"}
        for row in status_rows
        if row["training_status"] == "terminal_model_failure"
    ]
    write_csv(output / "FAILURE_LEDGER.csv", failures, FAILURE_FIELDS)
    proposal_infeasible = {
        row["model_id"] for row in primary_rows if row["refusal_reason"] == "proposal_family_all_infeasible"
    }
    truth_unavailable: dict[str, int] = defaultdict(int)
    for row in truth_rows:
        truth_unavailable[str(row["truth_unavailable_reason"])] += 1
    denominator = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "planned_models": EXPECTED_MODELS,
        "accounted_models": len(status_rows),
        "evaluable_models": sum(row["model_evaluable"] == "true" for row in status_rows),
        "terminal_model_failures": len(failures),
        "proposal_infeasible_models": len(proposal_infeasible),
        "primary_rows": len(primary_rows),
        "truth_rows": len(truth_rows),
        "secondary_count_rows": len(all_counts),
        "secondary_candidate_rows": len(candidate_rows),
        "secondary_procedure_rows": len(secondary_rows),
        "certified_by_procedure": {
            procedure: sum(
                row["procedure"] == procedure and row["certified"] == "true" for row in primary_rows
            )
            for procedure in PROCEDURES
        },
        "conditional_certified_coverage": {
            procedure: {
                "denominator": len(
                    certified_rows := [
                        row
                        for row in primary_rows
                        if row["procedure"] == procedure and row["certified"] == "true"
                    ]
                ),
                "value": (
                    None
                    if not certified_rows
                    else float(np.mean([float(row["coverage_lcb"]) for row in certified_rows]))
                ),
            }
            for procedure in PROCEDURES
        },
        "truth_unavailable_reason_rows": dict(sorted(truth_unavailable.items())),
    }
    write_json(output / "DENOMINATOR_LEDGER.json", denominator)

    replay_report = replay(output, write_report=False)
    replay_report = {
        key: value for key, value in replay_report.items()
        if key not in ("synthetic", "PACS_images_used", "PACS_inference")
    }
    write_json(output / "INDEPENDENT_REPLAY.json", replay_report)
    if replay_report["status"] != "PASS_INDEPENDENT_REPLAY":
        raise ContractError("independent primary replay failed")
    verdict, metrics = _scientific_verdict(primary_rows)
    _atomic_bytes(
        output / "NEGATIVE_RESULTS.md", _negative_results_text(verdict, metrics).encode("utf-8")
    )
    verify_primary_seal(output, protocol_dir=protocol_dir)
    validation = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": verdict,
        "primary_seal_sha256": authority.primary_seal_sha256,
        "primary_immutable_after_truth_and_secondary": True,
        "row_counts": {
            "primary": len(primary_rows),
            "truth": len(truth_rows),
            "secondary_counts": len(all_counts),
            "secondary_candidates": len(candidate_rows),
            "secondary_procedures": len(secondary_rows),
        },
        "denominators": denominator,
        "registered_metrics_and_gates": metrics,
        "independent_replay_status": replay_report["status"],
        "PACS_access_performed_by_finalizer": False,
        "GPU_used_by_finalizer": False,
    }
    write_json(output / "FINAL_VALIDATION.json", validation)
    hashes = _write_sha256_manifest(output, FINAL_ARTIFACTS)
    return {
        **validation,
        "artifact_hashes": hashes,
        "sha256s_sha256": sha256_file(output / "SHA256SUMS.txt"),
    }


def _load_fragment_directory(path: Path) -> list[dict[str, Any]]:
    root = Path(path)
    if not root.is_dir():
        raise ContractError(f"fragment directory is absent: {root}")
    files = sorted(root.glob("*.json"))
    if len(files) != EXPECTED_MODELS:
        raise ContractError(
            f"fragment directory must contain exactly {EXPECTED_MODELS} JSON files"
        )
    fragments: list[dict[str, Any]] = []
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ContractError(f"fragment must be a JSON object: {file}")
        fragments.append(value)
    return fragments


def _load_primary_bundle_directory(path: Path) -> list[dict[str, Any]]:
    root = Path(path)
    if not root.is_dir():
        raise ContractError(f"primary bundle directory is absent: {root}")
    cell_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if len(cell_dirs) != EXPECTED_MODELS:
        raise ContractError(
            f"primary bundle directory must contain exactly {EXPECTED_MODELS} cell directories"
        )
    return [load_primary_fragment_bundle(path) for path in cell_dirs]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    primary = subparsers.add_parser("seal-primary")
    primary.add_argument("--output-dir", type=Path, required=True)
    primary.add_argument("--fragments-dir", type=Path, required=True)
    final = subparsers.add_parser("finalize")
    final.add_argument("--output-dir", type=Path, required=True)
    final.add_argument("--truth-fragments-dir", type=Path, required=True)
    final.add_argument("--secondary-fragments-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "seal-primary":
        authority = seal_primary_campaign(
            args.output_dir, _load_primary_bundle_directory(args.fragments_dir)
        )
        print(authority.primary_seal_sha256)
        return 0
    # This verification is intentionally before either post-decision directory
    # is listed or opened.  The CLI is the fail-closed lazy truth loader.
    verify_primary_seal(args.output_dir)
    truth = _load_fragment_directory(args.truth_fragments_dir)
    secondary = _load_fragment_directory(args.secondary_fragments_dir)
    result = finalize_campaign(args.output_dir, truth, secondary)
    print(result["status"])
    return 0


__all__ = [
    "EXPECTED_SECONDARY_CANDIDATE_ROWS",
    "EXPECTED_SECONDARY_COUNT_ROWS",
    "EXPECTED_SECONDARY_PROCEDURE_ROWS",
    "VerifiedPrimarySeal",
    "append_primary_fragment_bundle",
    "finalize_campaign",
    "primary_fragment_sha256",
    "load_primary_fragment_bundle",
    "seal_primary_campaign",
    "secondary_fragment_sha256",
    "truth_fragment_sha256",
    "validate_secondary_fragment",
    "verify_primary_seal",
    "write_primary_fragment_bundle",
    "write_proposal_fragment_bundle",
]


if __name__ == "__main__":
    raise SystemExit(main())
