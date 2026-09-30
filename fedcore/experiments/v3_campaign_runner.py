"""Authorized two-phase scientific runner for the FedCORE PACS v3 campaign.

The runner is intentionally a single process.  It validates the seven mounted
control artifacts and re-hashes all five scientific inputs before importing
torch, decoding PACS, or touching CUDA.  It then collects all 30 primary
fragments, writes the campaign-global primary seal, and only afterwards opens
full audit truth and computes the registered secondary audits.

The mounted fold manifest physically contains all registered labels.  The
primary-before-truth rule is therefore a sealed-code procedural restriction,
not a claim that full-frame truth is physically inaccessible to the host.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from fedcore.experiments.v3_campaign_finalizer import (
    append_primary_fragment_bundle,
    finalize_campaign,
    load_primary_fragment_bundle,
    primary_fragment_sha256,
    seal_primary_campaign,
    secondary_fragment_sha256,
    truth_fragment_sha256,
    verify_primary_seal,
    write_proposal_fragment_bundle,
)
from fedcore.experiments.v3_campaign_state import (
    initial_campaign_state,
    next_state,
    validate_campaign_state,
    verify_state_history,
    write_state_history,
)
from fedcore.experiments.v3_contract import (
    CLIENTS,
    PRIMARY_BUDGET_TOTAL,
    PRIMARY_REPLICATE_ID,
    PROCEDURES,
    PROTOCOL_ID,
    ContractError,
    ModelCell,
    canonical_json_bytes,
    canonical_json_sha256,
    candidate_family,
    load_model_cells,
    sha256_file,
    verify_scientific_contract,
)
from fedcore.experiments.v3_counts import model_failure_count_rows
from fedcore.experiments.v3_execution_binding import (
    VerifiedExecutionAuthorization,
    acquire_exclusive_lease,
    validate_mounted_authorization_bundle,
)
from fedcore.experiments.v3_hsb import model_failure_candidate_decisions
from fedcore.experiments.v3_orchestrator import read_csv, write_json
from fedcore.experiments.v3_pacs_data import (
    PACSAuditFrameRecord,
    PACSZipDataset,
    PACSRecord,
    audit_frame_by_domain,
    build_registered_transforms,
    load_pacs_audit_frame,
    load_pacs_full_audit_records,
    load_pacs_proposal_records,
    load_pacs_selected_audit_records,
    load_pacs_train_records,
    records_by_use_and_domain,
)
from fedcore.experiments.v3_posttrain import (
    LogitFrame,
    audit_index_plan,
    audit_sequence_sha256,
    draw_frames,
    infer_logit_frame,
    load_audit_seeds,
    post_decision_truth,
    primary_cell,
    proposal_members,
    secondary_cell,
)
from fedcore.experiments.v3_proposal import ProposalMember, members_from_rows
from fedcore.experiments.v3_train import (
    WEIGHT_FILENAMES,
    WEIGHT_SHA256,
    build_registered_model,
    load_round_checkpoint,
    train_registered_model,
    verify_bound_input_hashes,
)


EXPECTED_INPUT_PATHS = {
    "PACS_mirror.zip": Path("/inputs/PACS_mirror.zip"),
    "IMAGE_MANIFEST.csv": Path("/inputs/IMAGE_MANIFEST.csv"),
    "FOLD_MANIFEST.csv": Path("/inputs/FOLD_MANIFEST.csv"),
    "resnet18-f37072fd.pth": Path("/inputs/resnet18-f37072fd.pth"),
    "convnext_tiny-983f1562.pth": Path("/inputs/convnext_tiny-983f1562.pth"),
}
EXPECTED_OUTPUT_DIR = Path("/output")
EXPECTED_AUTHORIZATION_DIR = Path("/authorization")
MAX_PREPROPOSAL_TECHNICAL_RESTARTS = 1


def _fsync_directory(path: Path) -> None:
    """Durably persist a newly created receipt name on POSIX filesystems."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(Path(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> Path:
    """Create one immutable receipt without an overwrite race."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, 0o444)
    except FileExistsError as exc:
        raise ContractError(f"immutable receipt already exists: {destination}") from exc
    payload = canonical_json_bytes(dict(value))
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ContractError("short write while sealing an immutable receipt")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(destination.parent)
    return destination


def _regular_file_inventory(root: Path, *, allow_missing: bool = False) -> list[dict[str, str]]:
    """Return a stable path/hash inventory and reject symlink substitution."""

    directory = Path(root)
    if not directory.exists() and allow_missing:
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise ContractError(f"inventory root is not a real directory: {directory}")
    rows: list[dict[str, str]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"inventory contains a symlink: {path}")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    return rows


def _partial_science_inventory(output: Path) -> list[dict[str, str]]:
    """Bind every extant non-checkpoint science fragment at a terminal failure."""

    rows: list[dict[str, str]] = []
    root = Path(output)
    for directory_name in (
        "state",
        "primary_fragments",
        "truth_fragments",
        "secondary_fragments",
    ):
        for row in _regular_file_inventory(
            root / directory_name, allow_missing=True
        ):
            rows.append(
                {
                    "path": f"{directory_name}/{row['path']}",
                    "sha256": row["sha256"],
                }
            )
    for filename in (
        "CAMPAIGN_LEASE.json",
        "CURRENT_STATE.json",
        "RUN_PROVENANCE.json",
        "CAMPAIGN_PRIMARY_STATE.json",
        "PRIMARY_SEAL.json",
        "PRIMARY_512_REP000.csv",
        "PRIMARY_CANDIDATE_DECISIONS.csv.gz",
        "FINAL_VALIDATION.json",
        "SHA256SUMS.txt",
    ):
        path = root / filename
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise ContractError(f"failure inventory contains an unsafe path: {path}")
            rows.append({"path": filename, "sha256": sha256_file(path)})
    return sorted(rows, key=lambda row: row["path"])


def _directory_tree_sha256(root: Path) -> str:
    """Hash the relative path and bytes of every regular file in one tree."""

    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise ContractError(f"completion tree is not a real directory: {directory}")
    rows: list[dict[str, str]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"completion tree contains a symlink: {path}")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    if not rows:
        raise ContractError(f"completion tree is empty: {directory}")
    return canonical_json_sha256(rows)


def _write_checkpoint_inventory(output: Path) -> tuple[Path, dict[str, Any]]:
    """Seal every successful or partial checkpoint before completion is declared."""

    files = _regular_file_inventory(Path(output) / "checkpoints", allow_missing=True)
    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "CHECKPOINT_INVENTORY_SEALED",
        "files": files,
        "checkpoint_tree_sha256": canonical_json_sha256(files),
    }
    payload = dict(body)
    payload["inventory_sha256"] = canonical_json_sha256(body)
    payload["inventory_sha256_definition"] = (
        "canonical JSON of this inventory before inventory_sha256 and "
        "inventory_sha256_definition are inserted"
    )
    path = Path(output) / "CHECKPOINT_INVENTORY.json"
    _write_exclusive_json(path, payload)
    return path, payload


def _failure_campaign_phase(exc: BaseException) -> str:
    """Separate technical incompleteness from deterministic implementation drift."""

    if isinstance(exc, ContractError):
        return "INVALID"
    if isinstance(
        exc,
        (OSError, MemoryError, FloatingPointError, KeyboardInterrupt, SystemExit),
    ):
        return "HOLD_INCOMPLETE"
    if isinstance(exc, RuntimeError) and any(
        marker in str(exc).lower()
        for marker in (
            "cuda",
            "cudnn",
            "cublas",
            "nccl",
            "out of memory",
            "driver",
            "resource temporarily unavailable",
        )
    ):
        return "HOLD_INCOMPLETE"
    return "INVALID"


def _write_campaign_failure_receipt(
    output: Path,
    *,
    exc: BaseException,
    authority: VerifiedExecutionAuthorization,
    state: Mapping[str, Any],
) -> Path:
    """Preserve one terminal failure and every extant checkpoint without overwrite."""

    checkpoint_files = _regular_file_inventory(
        Path(output) / "checkpoints", allow_missing=True
    )
    restart_receipts = _regular_file_inventory(
        Path(output) / "restart_receipts", allow_missing=True
    )
    science_files = _partial_science_inventory(output)
    cells = state.get("cells")
    outcomes_accessed = state.get("campaign_phase") != "PRIMARY_COLLECTION"
    if isinstance(cells, list):
        outcomes_accessed = outcomes_accessed or any(
            isinstance(row, Mapping)
            and row.get("primary_stage")
            in {"PROPOSAL_SEALED", "PRIMARY_FRAGMENT_SEALED", "MODEL_FAILURE"}
            for row in cells
        )
    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": str(state["campaign_phase"]),
        "failure_classification": (
            "DETERMINISTIC_CONTRACT_BREACH"
            if _failure_campaign_phase(exc) == "INVALID"
            else "INCOMPLETE_TECHNICAL_FAILURE"
        ),
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "run_authorization_sha256": authority.run_authorization_sha256,
        "state_sha256": state["state_sha256"],
        "state_sequence_number": int(state["sequence_number"]),
        "active_model_id": state.get("active_model_id"),
        "proposal_or_audit_outcome_accessed": outcomes_accessed,
        "same_protocol_retry_permitted": False,
        "checkpoint_files": checkpoint_files,
        "checkpoint_tree_sha256": canonical_json_sha256(checkpoint_files),
        "restart_receipt_files": restart_receipts,
        "restart_receipt_tree_sha256": canonical_json_sha256(restart_receipts),
        "science_files": science_files,
        "science_tree_sha256": canonical_json_sha256(science_files),
    }
    directory = Path(output) / "failure_receipts"
    path = directory / f"FAILURE_{int(state['sequence_number']):06d}.json"
    return _write_exclusive_json(path, _receipt_payload(body))


def _write_unverified_state_failure_receipt(
    output: Path,
    *,
    exc: BaseException,
    state_exception: BaseException,
    authority: VerifiedExecutionAuthorization,
) -> Path:
    """Retain both failures when even the append-only state chain cannot be read."""

    checkpoint_files = _regular_file_inventory(
        Path(output) / "checkpoints", allow_missing=True
    )
    restart_receipts = _regular_file_inventory(
        Path(output) / "restart_receipts", allow_missing=True
    )
    science_files = _partial_science_inventory(output)
    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "INVALID",
        "failure_classification": "UNVERIFIED_CAMPAIGN_STATE",
        "original_exception_type": type(exc).__name__,
        "original_message": str(exc),
        "state_exception_type": type(state_exception).__name__,
        "state_exception_message": str(state_exception),
        "run_authorization_sha256": authority.run_authorization_sha256,
        "same_protocol_retry_permitted": False,
        "checkpoint_files": checkpoint_files,
        "checkpoint_tree_sha256": canonical_json_sha256(checkpoint_files),
        "restart_receipt_files": restart_receipts,
        "restart_receipt_tree_sha256": canonical_json_sha256(restart_receipts),
        "science_files": science_files,
        "science_tree_sha256": canonical_json_sha256(science_files),
    }
    return _write_exclusive_json(
        Path(output) / "failure_receipts" / "FAILURE_STATE_UNVERIFIED.json",
        _receipt_payload(body),
    )


def _model_dict(model: ModelCell) -> dict[str, Any]:
    value = asdict(model)
    value["known_classes"] = list(model.known_classes)
    value["unknown_classes"] = list(model.unknown_classes)
    return value


def _input_mapping(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "PACS_mirror.zip": Path(args.pacs_archive),
        "IMAGE_MANIFEST.csv": Path(args.image_manifest),
        "FOLD_MANIFEST.csv": Path(args.fold_manifest),
        "resnet18-f37072fd.pth": Path(args.resnet18_weights),
        "convnext_tiny-983f1562.pth": Path(args.convnext_tiny_weights),
    }


def _validate_fixed_cli_paths(args: argparse.Namespace) -> None:
    observed = _input_mapping(args)
    if observed != EXPECTED_INPUT_PATHS:
        raise ContractError("scientific runner input path contract drift")
    if Path(args.output_dir) != EXPECTED_OUTPUT_DIR:
        raise ContractError("scientific runner output path contract drift")
    if Path(args.authorization_dir) != EXPECTED_AUTHORIZATION_DIR:
        raise ContractError("scientific runner authorization path contract drift")
    if args.device != "cuda:0":
        raise ContractError("scientific runner requires the registered cuda:0 device")


def _require_fresh_output(output: Path) -> None:
    try:
        metadata = output.lstat()
    except FileNotFoundError as exc:
        raise ContractError("registered output mount is absent") from exc
    if output.is_symlink() or not output.is_dir():
        raise ContractError("registered output must be a real directory")
    if next(output.iterdir(), None) is not None:
        raise ContractError("fresh run authorization requires an empty output directory")


def _retryable_preproposal_failure(exc: BaseException) -> bool:
    """Recognize only hardware/I/O failures that the sealed protocol permits retrying."""

    if isinstance(exc, (ContractError, FloatingPointError, KeyboardInterrupt, SystemExit)):
        return False
    if isinstance(exc, OSError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "cuda driver",
            "cuda error: unknown error",
            "cudnn_status",
            "cublas_status",
            "nccl error",
            "input/output error",
            "resource temporarily unavailable",
        )
    )


def _assert_preproposal_restart_allowed(
    output: Path,
    model: ModelCell,
    ordinal: int,
    *,
    primary_stage: str,
) -> None:
    """Fail closed if a retry is attempted after proposal or audit access."""

    if primary_stage not in {"TRAINING", "CHECKPOINT_SEALED"}:
        raise ContractError("same-cell restart is prohibited after proposal access")
    bundle = Path(output) / "primary_fragments" / f"{ordinal:02d}_{model.model_id}"
    forbidden = (
        "PROPOSAL_FRAGMENT.json",
        "PROPOSAL_SEAL.json",
        "AUDIT_DRAW_INDICES.json",
        "PRIMARY_FRAGMENT.json",
        "PRIMARY_FRAGMENT_SEAL.json",
    )
    if bundle.exists():
        if bundle.is_symlink() or not bundle.is_dir():
            raise ContractError("primary fragment path is unsafe before restart")
        if any((bundle / name).exists() for name in forbidden):
            raise ContractError("same-cell restart is prohibited after proposal access")


def _restart_checkpoint(
    output: Path,
    model: ModelCell,
    authorization_sha256: Mapping[str, str],
) -> tuple[bool, str | None, int]:
    """Validate an exact round checkpoint, or authorize a same-seed round-zero restart."""

    checkpoint = Path(output) / "checkpoints" / model.model_id / "latest.pt"
    if not checkpoint.exists():
        return False, None, 0
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ContractError("pre-proposal restart checkpoint is unsafe")
    payload = load_round_checkpoint(
        checkpoint,
        cell=model,
        weight_sha256=WEIGHT_SHA256[model.architecture],
        authorization_sha256=authorization_sha256,
    )
    return True, sha256_file(checkpoint), int(payload["completed_rounds"])


def _receipt_payload(body: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(body)
    sealed["receipt_sha256"] = canonical_json_sha256(body)
    sealed["receipt_sha256_definition"] = (
        "canonical JSON of this receipt before receipt_sha256 and "
        "receipt_sha256_definition are inserted"
    )
    return sealed


def _write_restart_receipt(
    output: Path,
    model: ModelCell,
    ordinal: int,
    *,
    state_sha256: str,
    run_authorization_sha256: str,
    first_exception: BaseException,
    resume_from_checkpoint: bool,
    checkpoint_sha256: str | None,
    completed_rounds: int,
) -> Path:
    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PREPROPOSAL_SAME_CELL_RESTART_AUTHORIZED",
        "model_id": model.model_id,
        "ordinal": ordinal,
        "failed_attempt": 1,
        "restart_attempt": 1,
        "maximum_restarts": MAX_PREPROPOSAL_TECHNICAL_RESTARTS,
        "same_model_cell": True,
        "same_training_seed": True,
        "primary_stage": "TRAINING",
        "proposal_accessed": False,
        "audit_accessed": False,
        "failure_type": type(first_exception).__name__,
        "failure_message": str(first_exception),
        "resume_from_checkpoint": resume_from_checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "completed_rounds": completed_rounds,
        "state_sha256_before_restart": state_sha256,
        "run_authorization_sha256": run_authorization_sha256,
    }
    path = (
        Path(output)
        / "restart_receipts"
        / f"{ordinal:02d}_{model.model_id}"
        / "RESTART_001.json"
    )
    return _write_exclusive_json(path, _receipt_payload(body))


def _write_restart_outcome(
    output: Path,
    model: ModelCell,
    ordinal: int,
    *,
    restart_receipt: Path,
    result: Any | None,
    exception: BaseException | None,
) -> Path:
    if (result is None) == (exception is None):
        raise ContractError("restart outcome requires exactly one result or exception")
    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": (
            "PREPROPOSAL_SAME_CELL_RESTART_COMPLETED"
            if exception is None
            else "PREPROPOSAL_SAME_CELL_RESTART_FAILED"
        ),
        "model_id": model.model_id,
        "ordinal": ordinal,
        "restart_attempt": 1,
        "further_restart_permitted": False,
        "proposal_accessed": False,
        "audit_accessed": False,
        "restart_receipt_sha256": sha256_file(restart_receipt),
        "completed_rounds": (
            int(result.completed_rounds) if result is not None else None
        ),
        "failure_type": type(exception).__name__ if exception is not None else None,
        "failure_message": str(exception) if exception is not None else None,
    }
    path = restart_receipt.with_name("RESTART_OUTCOME_001.json")
    return _write_exclusive_json(path, _receipt_payload(body))


def _train_with_preproposal_restart(
    model: ModelCell,
    datasets: Mapping[str, Any],
    *,
    ordinal: int,
    output: Path,
    weights_path: Path,
    authorization_sha256: Mapping[str, str],
    run_authorization_sha256: str,
    state_sha256: str,
    device: str,
) -> Any:
    """Run one cell and permit exactly one evidence-preserving technical restart."""

    kwargs = {
        "weights_path": weights_path,
        "output_dir": output,
        "authorization_sha256": authorization_sha256,
        "device": device,
    }
    try:
        return train_registered_model(model, datasets, resume=False, **kwargs)
    except Exception as first_exception:
        if not _retryable_preproposal_failure(first_exception):
            raise
        _assert_preproposal_restart_allowed(
            output, model, ordinal, primary_stage="TRAINING"
        )
        resume, checkpoint_hash, completed_rounds = _restart_checkpoint(
            output, model, authorization_sha256
        )
        restart_receipt = _write_restart_receipt(
            output,
            model,
            ordinal,
            state_sha256=state_sha256,
            run_authorization_sha256=run_authorization_sha256,
            first_exception=first_exception,
            resume_from_checkpoint=resume,
            checkpoint_sha256=checkpoint_hash,
            completed_rounds=completed_rounds,
        )
        try:
            result = train_registered_model(model, datasets, resume=resume, **kwargs)
        except Exception as restart_exception:
            _write_restart_outcome(
                output,
                model,
                ordinal,
                restart_receipt=restart_receipt,
                result=None,
                exception=restart_exception,
            )
            raise
        if result.stopped or int(result.completed_rounds) != 30:
            restart_exception = ContractError(
                "same-cell restart returned before registered round 30"
            )
            _write_restart_outcome(
                output,
                model,
                ordinal,
                restart_receipt=restart_receipt,
                result=None,
                exception=restart_exception,
            )
            raise restart_exception
        _write_restart_outcome(
            output,
            model,
            ordinal,
            restart_receipt=restart_receipt,
            result=result,
            exception=None,
        )
        return result


def _write_and_advance(
    output: Path,
    state: Mapping[str, Any],
    cells: Sequence[ModelCell],
    authorization_hash: str,
    **transition: Any,
) -> dict[str, Any]:
    updated = next_state(
        state,
        cells=cells,
        authorization_sha256=authorization_hash,
        **transition,
    )
    write_state_history(output, updated)
    return updated


def _infer_by_client(
    model: Any,
    archive: Path,
    records: Mapping[str, Sequence[PACSRecord]],
    *,
    transform: Any,
    device: str,
) -> dict[str, LogitFrame]:
    output: dict[str, LogitFrame] = {}
    for client in CLIENTS:
        rows = tuple(records[client])
        dataset = PACSZipDataset(
            archive,
            rows,
            transform=transform,
            allow_unknown=True,
        )
        try:
            output[client] = infer_logit_frame(
                model,
                dataset,
                client=client,
                identities=tuple(row.identity for row in rows),
                device=device,
            )
        finally:
            dataset.close()
    return output


def _infer_registered_draws_by_client(
    model: Any,
    archive: Path,
    reservoirs: Mapping[str, Sequence[PACSAuditFrameRecord]],
    selected_truth: Mapping[str, Sequence[PACSRecord]],
    indices: Mapping[str, Any],
    *,
    draws_per_client: int,
    transform: Any,
    device: str,
) -> dict[str, LogitFrame]:
    """Infer each distinct sampled atom once, then restore draw order.

    Registered audits sample with replacement.  Duplicate source identities are
    therefore valid draws, although the ordinary PACS dataset and full-frame
    ``LogitFrame`` validators intentionally require unique identities.  This
    helper keeps the decoded dataset unique and reconstructs the registered
    with-replacement prefix only after inference.
    """

    if draws_per_client not in (64, 128, 256):
        raise ContractError("registered audit draw count must be 64, 128, or 256")
    output: dict[str, LogitFrame] = {}
    for client in CLIENTS:
        reservoir = tuple(reservoirs[client])
        draw = tuple(int(value) for value in indices[client][:draws_per_client])
        if len(draw) != draws_per_client or any(
            value < 0 or value >= len(reservoir) for value in draw
        ):
            raise ContractError("registered audit prefix is outside its reservoir")
        selected_identities = tuple(reservoir[value].identity for value in draw)
        truth_by_identity = {
            record.identity: record for record in selected_truth[client]
        }
        if set(truth_by_identity) != set(selected_identities):
            raise ContractError("selected audit truth differs from registered identities")
        unique_records = tuple(
            truth_by_identity[identity] for identity in sorted(truth_by_identity)
        )
        dataset = PACSZipDataset(
            archive,
            unique_records,
            transform=transform,
            allow_unknown=True,
        )
        try:
            unique = infer_logit_frame(
                model,
                dataset,
                client=client,
                identities=tuple(record.identity for record in unique_records),
                device=device,
            )
        finally:
            dataset.close()
        position = {identity: offset for offset, identity in enumerate(unique.identities)}
        restored = [position[identity] for identity in selected_identities]
        labels = unique.labels[restored]
        output[client] = LogitFrame(
            client=client,
            identities=selected_identities,
            logits=unique.logits[restored],
            labels=labels,
            is_unknown=labels < 0,
        )
    return output


def _training_datasets(
    archive: Path,
    records: Mapping[str, Sequence[PACSRecord]],
    transform: Any,
) -> dict[str, PACSZipDataset]:
    return {
        client: PACSZipDataset(
            archive,
            tuple(records[client]),
            transform=transform,
            allow_unknown=False,
        )
        for client in CLIENTS
    }


def _close_datasets(values: Mapping[str, PACSZipDataset]) -> None:
    for dataset in values.values():
        dataset.close()


def _proposal_placeholder_members() -> tuple[ProposalMember, ...]:
    return tuple(
        ProposalMember(
            candidate_index=item.candidate_index,
            score=item.score,
            gamma=item.gamma,
            proposal_feasible=False,
            threshold=None,
            proposal_n=0,
            proposal_A=0,
            proposal_K=0,
            proposal_risk=None,
        )
        for item in candidate_family()
    )


def _failure_primary_fragment(model: ModelCell, ordinal: int) -> dict[str, Any]:
    members = _proposal_placeholder_members()
    decisions = model_failure_candidate_decisions(model, members)
    fragment: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "ordinal": ordinal,
        "model": _model_dict(model),
        "training_status": "terminal_model_failure",
        "checkpoint_sha256": None,
        "members": [member.as_row() for member in members],
        "counts": model_failure_count_rows(model.model_id),
        "candidate_decisions": [row.as_row() for row in decisions],
    }
    fragment["fragment_sha256"] = primary_fragment_sha256(fragment)
    return fragment


def _failure_audit_manifest(model: ModelCell) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "model_id": model.model_id,
        "status": "model_failure",
        "split": model.split,
        "replicate_id": PRIMARY_REPLICATE_ID,
        "clients": [],
    }


def _primary_success_fragment(
    model: ModelCell,
    ordinal: int,
    *,
    checkpoint_sha256: str,
    proposal_frames: Mapping[str, LogitFrame],
    sealed_members: Sequence[ProposalMember],
    primary_frames: Sequence[LogitFrame],
    audit_sequence_hash: str,
) -> dict[str, Any]:
    primary = primary_cell(
        model,
        tuple(proposal_frames[client] for client in CLIENTS),
        primary_frames,
        checkpoint_sha256=checkpoint_sha256,
        audit_sequence_hash=audit_sequence_hash,
    )
    if primary.members != tuple(sealed_members):
        raise ContractError("primary computation changed the sealed proposal family")
    fragment: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "ordinal": ordinal,
        "model": _model_dict(model),
        "training_status": "terminal_success",
        "checkpoint_sha256": checkpoint_sha256,
        "members": [member.as_row() for member in primary.members],
        "counts": list(primary.count_rows),
        "candidate_decisions": [row.as_row() for row in primary.candidate_rows],
    }
    fragment["fragment_sha256"] = primary_fragment_sha256(fragment)
    return fragment


def _audit_manifest(
    model: ModelCell,
    indices: Mapping[str, Any],
    reservoir_sizes: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "model_id": model.model_id,
        "status": "ok",
        "split": model.split,
        "replicate_id": PRIMARY_REPLICATE_ID,
        "clients": [
            {
                "client": client,
                "reservoir_size": int(reservoir_sizes[client]),
                "indices": [int(value) for value in indices[client]],
            }
            for client in CLIENTS
        ],
    }


def _checkpoint_seal(
    output: Path,
    model: ModelCell,
    checkpoint: Path,
    authority: VerifiedExecutionAuthorization,
) -> Path:
    path = output / "checkpoints" / model.model_id / "CHECKPOINT_SEAL.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "model_id": model.model_id,
            "checkpoint_path": str(checkpoint.relative_to(output)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "readiness_sha256": authority.readiness_sha256,
            "run_authorization_sha256": authority.run_authorization_sha256,
            "input_binding_sha256": authority.input_binding_sha256,
            "completed_rounds": 30,
        },
    )
    return path


def _load_sealed_model(
    output: Path,
    model: ModelCell,
    inputs: Mapping[str, Path],
    authority: VerifiedExecutionAuthorization,
    *,
    expected_checkpoint_sha256: str,
    expected_checkpoint_seal_sha256: str,
    device: str,
) -> Any:
    seal_path = output / "checkpoints" / model.model_id / "CHECKPOINT_SEAL.json"
    from fedcore.experiments.v3_contract import read_json

    if seal_path.is_symlink() or not seal_path.is_file():
        raise ContractError("checkpoint seal is absent or unsafe")
    if sha256_file(seal_path) != expected_checkpoint_seal_sha256:
        raise ContractError("checkpoint seal changed after the primary barrier")
    seal = read_json(seal_path)
    expected_checkpoint = output / "checkpoints" / model.model_id / "latest.pt"
    expected_relative = expected_checkpoint.relative_to(output).as_posix()
    if seal.get("checkpoint_path") != expected_relative:
        raise ContractError("checkpoint seal path drift")
    checkpoint = expected_checkpoint
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ContractError("checkpoint is absent or unsafe")
    observed_checkpoint_sha256 = sha256_file(checkpoint)
    if (
        seal.get("checkpoint_sha256") != observed_checkpoint_sha256
        or expected_checkpoint_sha256 != observed_checkpoint_sha256
    ):
        raise ContractError("checkpoint changed after its seal")
    if seal.get("run_authorization_sha256") != authority.run_authorization_sha256:
        raise ContractError("checkpoint seal authorization mismatch")
    weights = inputs[WEIGHT_FILENAMES[model.architecture]]
    network = build_registered_model(
        model.architecture,
        weights,
        model_init_seed_uint64=model.model_init_seed_uint64,
    )
    payload = load_round_checkpoint(
        checkpoint,
        cell=model,
        weight_sha256=WEIGHT_SHA256[model.architecture],
        authorization_sha256=authority.checkpoint_fields(),
    )
    network.load_state_dict(payload["model_state_dict"], strict=True)
    return network.to(device)


def _primary_rows_for_model(output: Path, model_id: str) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv(output / "PRIMARY_512_REP000.csv")
        if row["model_id"] == model_id
    ]
    if len(rows) != len(PROCEDURES) or tuple(row["procedure"] for row in rows) != PROCEDURES:
        raise ContractError("sealed primary rows are missing or out of order")
    return rows


def _failure_truth_fragment(
    model: ModelCell,
    primary_rows: Sequence[Mapping[str, Any]],
    *,
    primary_hash: str,
    candidate_hash: str,
    primary_seal_hash: str,
) -> dict[str, Any]:
    rows = [
        {
            "model_id": model.model_id,
            "procedure": row["procedure"],
            "primary_decision_file_sha256": primary_hash,
            "primary_candidate_decision_file_sha256": candidate_hash,
            "truth_opened_after_primary_hash": False,
            "selected_candidate": int(row["selected_candidate"]),
            "true_worst_client_risk": None,
            "true_min_client_acceptance": None,
            "validity_failure": None,
            "truth_unavailable_reason": "model_failure",
        }
        for row in primary_rows
    ]
    fragment = {
        "protocol_id": PROTOCOL_ID,
        "model_id": model.model_id,
        "primary_seal_sha256": primary_seal_hash,
        "rows": rows,
    }
    fragment["fragment_sha256"] = truth_fragment_sha256(fragment)
    return fragment


def _failure_secondary_fragment(
    model: ModelCell,
    *,
    primary_seal_hash: str,
) -> dict[str, Any]:
    members = _proposal_placeholder_members()
    counts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for budget in (256, 512, 1024):
        for replicate in range(100):
            if (budget, replicate) == (PRIMARY_BUDGET_TOTAL, PRIMARY_REPLICATE_ID):
                continue
            counts.extend(
                {
                    **row,
                    "budget_total": budget,
                    "replicate_id": replicate,
                }
                for row in model_failure_count_rows(model.model_id)
            )
            candidates.extend(
                row.as_row()
                for row in model_failure_candidate_decisions(
                    model, members, budget_total=budget, replicate_id=replicate
                )
            )
    fragment = {
        "protocol_id": PROTOCOL_ID,
        "model_id": model.model_id,
        "primary_seal_sha256": primary_seal_hash,
        "counts": counts,
        "candidate_decisions": candidates,
    }
    fragment["fragment_sha256"] = secondary_fragment_sha256(fragment)
    return fragment


def _run_primary_phase(
    *,
    output: Path,
    cells: Sequence[ModelCell],
    state: Mapping[str, Any],
    authority: VerifiedExecutionAuthorization,
    inputs: Mapping[str, Path],
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    archive = inputs["PACS_mirror.zip"]
    train_transform, inference_transform = build_registered_transforms()
    fragments: list[dict[str, Any]] = []
    for ordinal, model in enumerate(cells):
        bundle_dir = output / "primary_fragments" / f"{ordinal:02d}_{model.model_id}"
        training_attempted = False
        training_finished = False
        state = _write_and_advance(
            output,
            state,
            cells,
            authority.run_authorization_sha256,
            cell_index=ordinal,
            primary_stage="TRAINING",
        )
        try:
            train_stage_records = load_pacs_train_records(
                inputs["IMAGE_MANIFEST.csv"],
                inputs["FOLD_MANIFEST.csv"],
                split=model.split,
                known_classes=model.known_classes,
                expected_image_count=9991,
            )
            train_records = records_by_use_and_domain(train_stage_records, "train")
            datasets = _training_datasets(archive, train_records, train_transform)
            try:
                training_attempted = True
                result = _train_with_preproposal_restart(
                    model,
                    datasets,
                    ordinal=ordinal,
                    weights_path=inputs[WEIGHT_FILENAMES[model.architecture]],
                    output=output,
                    authorization_sha256=authority.checkpoint_fields(),
                    run_authorization_sha256=authority.run_authorization_sha256,
                    state_sha256=str(state["state_sha256"]),
                    device=device,
                )
                training_finished = True
            finally:
                _close_datasets(datasets)
            if result.stopped or result.completed_rounds != 30:
                raise ContractError("fresh campaign cell did not reach the registered round 30")
            checkpoint_seal = _checkpoint_seal(output, model, result.checkpoint_path, authority)
            checkpoint_hash = sha256_file(result.checkpoint_path)
            checkpoint_seal_hash = sha256_file(checkpoint_seal)
            state = _write_and_advance(
                output,
                state,
                cells,
                authority.run_authorization_sha256,
                cell_index=ordinal,
                primary_stage="CHECKPOINT_SEALED",
                checkpoint_seal_sha256=checkpoint_seal_hash,
            )

            inference_model = _load_sealed_model(
                output,
                model,
                inputs,
                authority,
                expected_checkpoint_sha256=checkpoint_hash,
                expected_checkpoint_seal_sha256=checkpoint_seal_hash,
                device=device,
            )

            proposal_stage_records = load_pacs_proposal_records(
                inputs["IMAGE_MANIFEST.csv"],
                inputs["FOLD_MANIFEST.csv"],
                split=model.split,
                known_classes=model.known_classes,
                expected_image_count=9991,
            )
            proposal_records = records_by_use_and_domain(
                proposal_stage_records, "proposal"
            )
            proposal_frames = _infer_by_client(
                inference_model,
                archive,
                proposal_records,
                transform=inference_transform,
                device=device,
            )
            sealed_members = proposal_members(
                tuple(proposal_frames[client] for client in CLIENTS)
            )
            proposal_fragment = {
                "protocol_id": PROTOCOL_ID,
                "ordinal": ordinal,
                "model": _model_dict(model),
                "training_status": "terminal_success",
                "checkpoint_sha256": checkpoint_hash,
                "members": [member.as_row() for member in sealed_members],
            }
            proposal_seal_path = write_proposal_fragment_bundle(
                bundle_dir, proposal_fragment
            )
            proposal_seal_sha256 = sha256_file(proposal_seal_path)
            state = _write_and_advance(
                output,
                state,
                cells,
                authority.run_authorization_sha256,
                cell_index=ordinal,
                primary_stage="PROPOSAL_SEALED",
                proposal_seal_sha256=proposal_seal_sha256,
            )
        except FloatingPointError:
            if not training_attempted or training_finished:
                raise
            fragment = _failure_primary_fragment(model, ordinal)
            proposal_seal_path = write_proposal_fragment_bundle(bundle_dir, fragment)
            audit_manifest = _failure_audit_manifest(model)
            seal_path = append_primary_fragment_bundle(
                bundle_dir, fragment, audit_manifest
            )
            state = _write_and_advance(
                output,
                state,
                cells,
                authority.run_authorization_sha256,
                cell_index=ordinal,
                primary_stage="MODEL_FAILURE",
                failure_stage="training_nonfinite",
                proposal_seal_sha256=sha256_file(proposal_seal_path),
                primary_fragment_seal_sha256=sha256_file(seal_path),
            )
            fragments.append(load_primary_fragment_bundle(bundle_dir))
            continue

        # The proposal files and PROPOSAL_SEAL.json are durable before this
        # point.  Only now may the runner expose primary-audit strata, draw
        # indices, labels, or inference outputs.
        audit_frame_records = load_pacs_audit_frame(
            inputs["IMAGE_MANIFEST.csv"],
            inputs["FOLD_MANIFEST.csv"],
            split=model.split,
            known_classes=model.known_classes,
            proposal_seal_path=proposal_seal_path,
            expected_proposal_seal_sha256=proposal_seal_sha256,
            expected_image_count=9991,
        )
        audit_frame = audit_frame_by_domain(audit_frame_records)
        sizes = {client: len(audit_frame[client]) for client in CLIENTS}
        # Audit stream seeds are a stage-three object and are not loaded until
        # the exact proposal artifact is durable and hash-verified above.
        seeds = load_audit_seeds()
        indices = audit_index_plan(model.split, PRIMARY_REPLICATE_ID, sizes, seeds=seeds)
        selected_audit_records = load_pacs_selected_audit_records(
            inputs["IMAGE_MANIFEST.csv"],
            inputs["FOLD_MANIFEST.csv"],
            split=model.split,
            known_classes=model.known_classes,
            audit_frame=audit_frame_records,
            indices=indices,
            draws_per_client=128,
            proposal_seal_path=proposal_seal_path,
            expected_proposal_seal_sha256=proposal_seal_sha256,
            expected_image_count=9991,
        )
        selected_audit_truth = records_by_use_and_domain(
            selected_audit_records, "audit"
        )
        primary_frames = _infer_registered_draws_by_client(
            inference_model,
            archive,
            audit_frame,
            selected_audit_truth,
            indices,
            draws_per_client=128,
            transform=inference_transform,
            device=device,
        )
        sequence_hash = audit_sequence_sha256(model.split, PRIMARY_REPLICATE_ID, indices)
        fragment = _primary_success_fragment(
            model,
            ordinal,
            checkpoint_sha256=checkpoint_hash,
            proposal_frames=proposal_frames,
            sealed_members=sealed_members,
            primary_frames=tuple(primary_frames[client] for client in CLIENTS),
            audit_sequence_hash=sequence_hash,
        )
        audit_manifest = _audit_manifest(model, indices, sizes)
        seal_path = append_primary_fragment_bundle(bundle_dir, fragment, audit_manifest)
        state = _write_and_advance(
            output,
            state,
            cells,
            authority.run_authorization_sha256,
            cell_index=ordinal,
            primary_stage="PRIMARY_FRAGMENT_SEALED",
            primary_fragment_seal_sha256=sha256_file(seal_path),
        )
        fragments.append(load_primary_fragment_bundle(bundle_dir))
    primary_authority = seal_primary_campaign(output, fragments)
    state = _write_and_advance(
        output,
        state,
        cells,
        authority.run_authorization_sha256,
        campaign_phase="PRIMARY_SEALED",
        primary_seal_sha256=primary_authority.primary_seal_sha256,
    )
    state = _write_and_advance(
        output,
        state,
        cells,
        authority.run_authorization_sha256,
        campaign_phase="POST_PRIMARY_COLLECTION",
    )
    return state, fragments


def _run_post_primary_phase(
    *,
    output: Path,
    cells: Sequence[ModelCell],
    state: Mapping[str, Any],
    authority: VerifiedExecutionAuthorization,
    inputs: Mapping[str, Path],
    primary_fragments: Sequence[Mapping[str, Any]],
    device: str,
) -> dict[str, Any]:
    primary_authority = verify_primary_seal(output)
    if primary_authority.primary_seal_sha256 != state.get("primary_seal_sha256"):
        raise ContractError("on-disk primary seal differs from the campaign-state barrier")
    archive = inputs["PACS_mirror.zip"]
    _, inference_transform = build_registered_transforms()
    seeds = load_audit_seeds()
    truth_fragments: list[dict[str, Any]] = []
    secondary_fragments: list[dict[str, Any]] = []
    fragments_by_model = {
        str(fragment["model"]["model_id"]): fragment for fragment in primary_fragments
    }
    for ordinal, model in enumerate(cells):
        primary_rows = _primary_rows_for_model(output, model.model_id)
        fragment = fragments_by_model[model.model_id]
        if fragment["training_status"] == "terminal_model_failure":
            truth = _failure_truth_fragment(
                model,
                primary_rows,
                primary_hash=primary_authority.primary_decision_file_sha256,
                candidate_hash=primary_authority.primary_candidate_decision_file_sha256,
                primary_seal_hash=primary_authority.primary_seal_sha256,
            )
            secondary = _failure_secondary_fragment(
                model, primary_seal_hash=primary_authority.primary_seal_sha256
            )
            truth_path = output / "truth_fragments" / f"{ordinal:02d}_{model.model_id}.json"
            secondary_path = (
                output / "secondary_fragments" / f"{ordinal:02d}_{model.model_id}.json"
            )
            write_json(truth_path, truth)
            write_json(secondary_path, secondary)
            state = _write_and_advance(
                output,
                state,
                cells,
                authority.run_authorization_sha256,
                cell_index=ordinal,
                post_primary_stage="MODEL_FAILURE",
                failure_stage="training_nonfinite",
                truth_fragment_sha256=truth["fragment_sha256"],
                secondary_fragment_sha256=secondary["fragment_sha256"],
            )
        else:
            network = _load_sealed_model(
                output,
                model,
                inputs,
                authority,
                expected_checkpoint_sha256=str(fragment["checkpoint_sha256"]),
                expected_checkpoint_seal_sha256=str(
                    state["cells"][ordinal]["checkpoint_seal_sha256"]
                ),
                device=device,
            )
            audit_stage_records = load_pacs_full_audit_records(
                inputs["IMAGE_MANIFEST.csv"],
                inputs["FOLD_MANIFEST.csv"],
                split=model.split,
                known_classes=model.known_classes,
                primary_seal_path=output / "PRIMARY_SEAL.json",
                expected_primary_seal_sha256=(
                    primary_authority.primary_seal_sha256
                ),
                expected_image_count=9991,
            )
            audit_records = records_by_use_and_domain(audit_stage_records, "audit")
            frames = _infer_by_client(
                network,
                archive,
                audit_records,
                transform=inference_transform,
                device=device,
            )
            members = members_from_rows(fragment["members"])
            truth_rows = post_decision_truth(
                primary_rows,
                members,
                tuple(frames[client] for client in CLIENTS),
                primary_decision_file_sha256=primary_authority.primary_decision_file_sha256,
                primary_candidate_decision_file_sha256=(
                    primary_authority.primary_candidate_decision_file_sha256
                ),
                primary_seal_verified=True,
            )
            truth = {
                "protocol_id": PROTOCOL_ID,
                "model_id": model.model_id,
                "primary_seal_sha256": primary_authority.primary_seal_sha256,
                "rows": truth_rows,
            }
            truth["fragment_sha256"] = truth_fragment_sha256(truth)
            truth_path = output / "truth_fragments" / f"{ordinal:02d}_{model.model_id}.json"
            write_json(truth_path, truth)
            state = _write_and_advance(
                output,
                state,
                cells,
                authority.run_authorization_sha256,
                cell_index=ordinal,
                post_primary_stage="TRUTH_COMPLETE",
                truth_fragment_sha256=truth["fragment_sha256"],
            )
            counts, decisions = secondary_cell(
                model,
                members,
                frames,
                checkpoint_sha256=str(fragment["checkpoint_sha256"]),
                proposal_seal_sha256=str(
                    fragment["candidate_decisions"][0]["proposal_seal_sha256"]
                ),
                primary_seal_verified=True,
                seeds=seeds,
            )
            secondary = {
                "protocol_id": PROTOCOL_ID,
                "model_id": model.model_id,
                "primary_seal_sha256": primary_authority.primary_seal_sha256,
                "counts": counts,
                "candidate_decisions": [row.as_row() for row in decisions],
            }
            secondary["fragment_sha256"] = secondary_fragment_sha256(secondary)
            secondary_path = (
                output / "secondary_fragments" / f"{ordinal:02d}_{model.model_id}.json"
            )
            write_json(secondary_path, secondary)
            state = _write_and_advance(
                output,
                state,
                cells,
                authority.run_authorization_sha256,
                cell_index=ordinal,
                post_primary_stage="SECONDARY_COMPLETE",
                secondary_fragment_sha256=secondary["fragment_sha256"],
            )
            state = _write_and_advance(
                output,
                state,
                cells,
                authority.run_authorization_sha256,
                cell_index=ordinal,
                post_primary_stage="COMPLETE",
            )
        truth_fragments.append(truth)
        secondary_fragments.append(secondary)
    final = finalize_campaign(output, truth_fragments, secondary_fragments)
    state = _write_and_advance(
        output,
        state,
        cells,
        authority.run_authorization_sha256,
        campaign_phase="FINALIZED",
        final_validation_sha256=sha256_file(output / "FINAL_VALIDATION.json"),
    )
    if final["status"] not in {
        "PASS_FRESH_SEED_REPLICATION",
        "PASS_CORE_ONLY",
        "VALID_NEGATIVE",
    }:
        raise ContractError("unregistered final scientific verdict")
    return state


def _write_execution_completion(
    output: Path,
    state: Mapping[str, Any],
    authority: VerifiedExecutionAuthorization,
    observed_input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Bind final science, state history, fragments, inputs, and authorization."""

    cells = load_model_cells()
    verified_state = verify_state_history(
        output,
        cells,
        authorization_sha256=authority.run_authorization_sha256,
    )
    if verified_state.get("state_sha256") != state.get("state_sha256"):
        raise ContractError("in-memory final state differs from immutable state history")
    if state.get("campaign_phase") != "FINALIZED":
        raise ContractError("execution completion requires a FINALIZED campaign state")

    current_state = output / "CURRENT_STATE.json"
    provenance = output / "RUN_PROVENANCE.json"
    lease = output / "CAMPAIGN_LEASE.json"
    final_validation = output / "FINAL_VALIDATION.json"
    scientific_manifest = output / "SHA256SUMS.txt"
    for path in (current_state, provenance, lease, final_validation, scientific_manifest):
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"completion artifact is absent or unsafe: {path.name}")
    checkpoint_inventory_path, checkpoint_inventory = _write_checkpoint_inventory(output)
    restart_receipts = _regular_file_inventory(
        output / "restart_receipts", allow_missing=True
    )
    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_AUTHORIZED_CAMPAIGN_COMPLETION_BINDING",
        "readiness_sha256": authority.readiness_sha256,
        "run_authorization_sha256": authority.run_authorization_sha256,
        "authorized_plan_sha256": authority.authorized_plan_sha256,
        "runtime_receipt_sha256": authority.runtime_receipt_sha256,
        "input_binding_sha256": authority.input_binding_sha256,
        "source_commit": authority.source_commit,
        "image_id": authority.image_id,
        "authorization_id": authority.authorization_id,
        "input_sha256": dict(sorted(observed_input_hashes.items())),
        "run_provenance_file_sha256": sha256_file(provenance),
        "campaign_lease_file_sha256": sha256_file(lease),
        "current_state_file_sha256": sha256_file(current_state),
        "final_state_sha256": state["state_sha256"],
        "final_validation_file_sha256": sha256_file(final_validation),
        "scientific_manifest_file_sha256": sha256_file(scientific_manifest),
        "checkpoint_inventory_file_sha256": sha256_file(checkpoint_inventory_path),
        "checkpoint_tree_sha256": checkpoint_inventory["checkpoint_tree_sha256"],
        "restart_receipt_files": restart_receipts,
        "restart_receipt_tree_sha256": canonical_json_sha256(restart_receipts),
        "state_tree_sha256": _directory_tree_sha256(output / "state"),
        "primary_fragment_tree_sha256": _directory_tree_sha256(
            output / "primary_fragments"
        ),
        "truth_fragment_tree_sha256": _directory_tree_sha256(
            output / "truth_fragments"
        ),
        "secondary_fragment_tree_sha256": _directory_tree_sha256(
            output / "secondary_fragments"
        ),
        "planned_cells": 30,
        "campaign_finalized": True,
    }
    seal = dict(body)
    seal["completion_sha256"] = canonical_json_sha256(body)
    seal["completion_sha256_definition"] = (
        "canonical JSON of this object before completion_sha256 and "
        "completion_sha256_definition are inserted"
    )
    completion_path = output / "EXECUTION_COMPLETION_SEAL.json"
    write_json(completion_path, seal)
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_EXECUTION_MANIFEST",
        "completion_seal_file_sha256": sha256_file(completion_path),
        "run_provenance_file_sha256": sha256_file(provenance),
        "current_state_file_sha256": sha256_file(current_state),
        "final_validation_file_sha256": sha256_file(final_validation),
        "scientific_manifest_file_sha256": sha256_file(scientific_manifest),
        "checkpoint_inventory_file_sha256": sha256_file(checkpoint_inventory_path),
        "checkpoint_tree_sha256": checkpoint_inventory["checkpoint_tree_sha256"],
        "restart_receipt_tree_sha256": canonical_json_sha256(restart_receipts),
    }
    write_json(output / "EXECUTION_MANIFEST.json", manifest)
    return seal


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    """Run one fresh campaign after the complete mounted guard passes."""

    _validate_fixed_cli_paths(args)
    source_commit = os.environ.get("FEDCORE_SOURCE_COMMIT")
    if not source_commit:
        raise ContractError("running image lacks FEDCORE_SOURCE_COMMIT")
    authority = validate_mounted_authorization_bundle(
        Path(args.authorization_dir), expected_source_commit=source_commit
    )
    inputs = _input_mapping(args)
    observed_hashes = verify_bound_input_hashes(inputs)
    verify_scientific_contract()
    output = Path(args.output_dir)
    _require_fresh_output(output)
    acquire_exclusive_lease(
        output / "CAMPAIGN_LEASE.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "authorization_id": authority.authorization_id,
            "run_authorization_sha256": authority.run_authorization_sha256,
            "authorized_plan_sha256": authority.authorized_plan_sha256,
            "runtime_receipt_sha256": authority.runtime_receipt_sha256,
            "fresh_only": True,
        },
    )
    cells = load_model_cells()
    state = initial_campaign_state(
        cells, authorization_sha256=authority.run_authorization_sha256
    )
    validate_campaign_state(
        state, cells, authorization_sha256=authority.run_authorization_sha256
    )
    write_state_history(output, state)
    write_json(
        output / "RUN_PROVENANCE.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "source_commit": authority.source_commit,
            "image_id": authority.image_id,
            "authorization_id": authority.authorization_id,
            "readiness_sha256": authority.readiness_sha256,
            "run_authorization_sha256": authority.run_authorization_sha256,
            "authorized_plan_sha256": authority.authorized_plan_sha256,
            "runtime_receipt_sha256": authority.runtime_receipt_sha256,
            "input_sha256": observed_hashes,
            "planned_cells": 30,
            "primary_before_truth": True,
            "truth_is_procedurally_gated_not_physically_absent": True,
        },
    )
    try:
        state, fragments = _run_primary_phase(
            output=output,
            cells=cells,
            state=state,
            authority=authority,
            inputs=inputs,
            device=args.device,
        )
        state = _run_post_primary_phase(
            output=output,
            cells=cells,
            state=state,
            authority=authority,
            inputs=inputs,
            primary_fragments=fragments,
            device=args.device,
        )
        completion = _write_execution_completion(
            output, state, authority, observed_hashes
        )
    except BaseException as exc:
        try:
            current = verify_state_history(
                output,
                cells,
                authorization_sha256=authority.run_authorization_sha256,
            )
        except BaseException as state_exception:
            receipt = _write_unverified_state_failure_receipt(
                output,
                exc=exc,
                state_exception=state_exception,
                authority=authority,
            )
            _write_exclusive_json(
                output / "RUN_FAILURE.json",
                {
                    "schema_version": 1,
                    "protocol_id": PROTOCOL_ID,
                    "status": "INVALID",
                    "failure_receipt_path": receipt.relative_to(output).as_posix(),
                    "failure_receipt_sha256": sha256_file(receipt),
                    "state_sha256": None,
                    "run_authorization_sha256": authority.run_authorization_sha256,
                },
            )
            raise ContractError(
                "campaign failure occurred with an unverifiable state history"
            ) from state_exception
        try:
            target_phase = _failure_campaign_phase(exc)
            if current["campaign_phase"] not in {"HOLD_INCOMPLETE", "INVALID"}:
                current = next_state(
                    current,
                    cells=cells,
                    authorization_sha256=authority.run_authorization_sha256,
                    campaign_phase=target_phase,
                )
                write_state_history(output, current)
        except BaseException as terminal_exception:
            receipt = _write_unverified_state_failure_receipt(
                output,
                exc=exc,
                state_exception=terminal_exception,
                authority=authority,
            )
            _write_exclusive_json(
                output / "RUN_FAILURE.json",
                {
                    "schema_version": 1,
                    "protocol_id": PROTOCOL_ID,
                    "status": "INVALID",
                    "failure_receipt_path": receipt.relative_to(output).as_posix(),
                    "failure_receipt_sha256": sha256_file(receipt),
                    "state_sha256": None,
                    "run_authorization_sha256": authority.run_authorization_sha256,
                },
            )
            raise ContractError(
                "campaign terminal failure state could not be sealed"
            ) from terminal_exception
        receipt = _write_campaign_failure_receipt(
            output,
            exc=exc,
            authority=authority,
            state=current,
        )
        _write_exclusive_json(
            output / "RUN_FAILURE.json",
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "status": current["campaign_phase"],
                "failure_receipt_path": receipt.relative_to(output).as_posix(),
                "failure_receipt_sha256": sha256_file(receipt),
                "state_sha256": current["state_sha256"],
                "run_authorization_sha256": authority.run_authorization_sha256,
            },
        )
        raise
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "CAMPAIGN_FINALIZED",
        "state_sha256": state["state_sha256"],
        "final_validation_sha256": sha256_file(output / "FINAL_VALIDATION.json"),
        "completion_sha256": completion["completion_sha256"],
        "execution_manifest_sha256": sha256_file(output / "EXECUTION_MANIFEST.json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--authorization-dir", required=True)
    run.add_argument("--pacs-archive", required=True)
    run.add_argument("--image-manifest", required=True)
    run.add_argument("--fold-manifest", required=True)
    run.add_argument("--resnet18-weights", required=True)
    run.add_argument("--convnext-tiny-weights", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--device", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_campaign(args)
    import json

    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_campaign"]
