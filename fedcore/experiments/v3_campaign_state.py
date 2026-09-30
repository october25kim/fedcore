"""Hash-chained two-phase state machine for the FedCORE PACS v3 campaign."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from fedcore.experiments.v3_contract import (
    PROTOCOL_ID,
    ContractError,
    ModelCell,
    canonical_json_sha256,
)
from fedcore.experiments.v3_orchestrator import write_json


CAMPAIGN_PHASES = (
    "PRIMARY_COLLECTION",
    "PRIMARY_SEALED",
    "POST_PRIMARY_COLLECTION",
    "FINALIZED",
    "HOLD_INCOMPLETE",
    "INVALID",
)
PRIMARY_STAGES = (
    "NOT_STARTED",
    "TRAINING",
    "CHECKPOINT_SEALED",
    "PROPOSAL_SEALED",
    "PRIMARY_FRAGMENT_SEALED",
    "MODEL_FAILURE",
)
POST_PRIMARY_STAGES = (
    "LOCKED",
    "NOT_STARTED",
    "TRUTH_COMPLETE",
    "SECONDARY_COMPLETE",
    "COMPLETE",
    "MODEL_FAILURE",
)
PRIMARY_TERMINAL = frozenset({"PRIMARY_FRAGMENT_SEALED", "MODEL_FAILURE"})
POST_TERMINAL = frozenset({"COMPLETE", "MODEL_FAILURE"})
PRIMARY_ACTIVE = frozenset({"TRAINING", "CHECKPOINT_SEALED", "PROPOSAL_SEALED"})
POST_ACTIVE = frozenset({"TRUTH_COMPLETE", "SECONDARY_COMPLETE"})
PRIMARY_TRANSITIONS = {
    "NOT_STARTED": frozenset({"TRAINING"}),
    "TRAINING": frozenset({"CHECKPOINT_SEALED", "MODEL_FAILURE"}),
    "CHECKPOINT_SEALED": frozenset({"PROPOSAL_SEALED", "MODEL_FAILURE"}),
    "PROPOSAL_SEALED": frozenset({"PRIMARY_FRAGMENT_SEALED", "MODEL_FAILURE"}),
    "PRIMARY_FRAGMENT_SEALED": frozenset(),
    "MODEL_FAILURE": frozenset(),
}
POST_TRANSITIONS = {
    "LOCKED": frozenset(),
    "NOT_STARTED": frozenset({"TRUTH_COMPLETE", "MODEL_FAILURE"}),
    "TRUTH_COMPLETE": frozenset({"SECONDARY_COMPLETE", "MODEL_FAILURE"}),
    "SECONDARY_COMPLETE": frozenset({"COMPLETE", "MODEL_FAILURE"}),
    "COMPLETE": frozenset(),
    "MODEL_FAILURE": frozenset(),
}


def _is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _state_body(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"state_sha256", "state_sha256_definition"}
    }


def seal_state(value: Mapping[str, Any]) -> dict[str, Any]:
    body = _state_body(value)
    output = deepcopy(body)
    output["state_sha256"] = canonical_json_sha256(body)
    output["state_sha256_definition"] = (
        "canonical JSON of this state before state_sha256 and "
        "state_sha256_definition are inserted"
    )
    return output


def initial_campaign_state(
    cells: Sequence[ModelCell], *, authorization_sha256: str
) -> dict[str, Any]:
    if len(cells) != 30 or not _is_hex64(authorization_sha256):
        raise ContractError("initial state requires 30 cells and a hex64 authorization")
    return seal_state(
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "campaign_phase": "PRIMARY_COLLECTION",
            "sequence_number": 0,
            "previous_state_sha256": None,
            "run_authorization_sha256": authorization_sha256,
            "active_model_id": None,
            "primary_seal_sha256": None,
            "final_validation_sha256": None,
            "cells": [
                {
                    "ordinal": ordinal,
                    "model_id": cell.model_id,
                    "primary_stage": "NOT_STARTED",
                    "post_primary_stage": "LOCKED",
                    "checkpoint_seal_sha256": None,
                    "proposal_seal_sha256": None,
                    "primary_fragment_seal_sha256": None,
                    "truth_fragment_sha256": None,
                    "secondary_fragment_sha256": None,
                    "failure_stage": None,
                }
                for ordinal, cell in enumerate(cells)
            ],
        }
    )


def validate_campaign_state(
    value: Mapping[str, Any],
    cells: Sequence[ModelCell],
    *,
    authorization_sha256: str,
) -> str:
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("campaign-state protocol mismatch")
    if value.get("campaign_phase") not in CAMPAIGN_PHASES:
        raise ContractError("unknown campaign phase")
    if value.get("run_authorization_sha256") != authorization_sha256:
        raise ContractError("campaign state authorization mismatch")
    if not isinstance(value.get("sequence_number"), int) or value["sequence_number"] < 0:
        raise ContractError("campaign-state sequence number is invalid")
    observed_hash = value.get("state_sha256")
    if not _is_hex64(observed_hash) or canonical_json_sha256(_state_body(value)) != observed_hash:
        raise ContractError("campaign-state self-hash mismatch")
    if value.get("state_sha256_definition") != (
        "canonical JSON of this state before state_sha256 and "
        "state_sha256_definition are inserted"
    ):
        raise ContractError("campaign-state hash definition drift")
    rows = value.get("cells")
    if not isinstance(rows, list) or len(rows) != len(cells) or len(cells) != 30:
        raise ContractError("campaign state must contain the frozen 30-cell set")
    phase = str(value["campaign_phase"])
    if phase not in {"FINALIZED", "HOLD_INCOMPLETE", "INVALID"} and value.get(
        "final_validation_sha256"
    ) is not None:
        raise ContractError("final validation appeared before campaign finalization")
    primary_open = False
    post_open = False
    primary_active: list[str] = []
    post_active: list[str] = []
    for ordinal, (cell, row) in enumerate(zip(cells, rows, strict=True)):
        if not isinstance(row, Mapping):
            raise ContractError("campaign cell state must be an object")
        if row.get("ordinal") != ordinal or row.get("model_id") != cell.model_id:
            raise ContractError("campaign cell order/model drift")
        primary = row.get("primary_stage")
        post = row.get("post_primary_stage")
        if primary not in PRIMARY_STAGES or post not in POST_PRIMARY_STAGES:
            raise ContractError("unknown per-cell stage")
        if primary in PRIMARY_ACTIVE:
            primary_active.append(cell.model_id)
        if post in POST_ACTIVE:
            post_active.append(cell.model_id)

        if primary == "NOT_STARTED":
            primary_open = True
        elif primary in PRIMARY_ACTIVE:
            if primary_open:
                raise ContractError("active primary cell appears after an unfinished cell")
            primary_open = True
        elif primary in PRIMARY_TERMINAL and primary_open:
            raise ContractError("terminal primary cell appears after an unfinished cell")

        if post == "NOT_STARTED":
            post_open = True
        elif post in POST_ACTIVE:
            if post_open:
                raise ContractError("active post-primary cell appears after an unfinished cell")
            post_open = True
        elif post in POST_TERMINAL and post_open:
            raise ContractError("terminal post-primary cell appears after an unfinished cell")

        checkpoint = row.get("checkpoint_seal_sha256")
        proposal_seal = row.get("proposal_seal_sha256")
        fragment = row.get("primary_fragment_seal_sha256")
        truth = row.get("truth_fragment_sha256")
        secondary = row.get("secondary_fragment_sha256")
        failure_stage = row.get("failure_stage")
        if primary in {"NOT_STARTED", "TRAINING"} and checkpoint is not None:
            raise ContractError("checkpoint seal appeared before checkpoint sealing")
        if primary in {"CHECKPOINT_SEALED", "PROPOSAL_SEALED", "PRIMARY_FRAGMENT_SEALED"}:
            if not _is_hex64(checkpoint):
                raise ContractError("primary stage lacks its checkpoint seal")
        if primary in {"NOT_STARTED", "TRAINING", "CHECKPOINT_SEALED"}:
            if proposal_seal is not None:
                raise ContractError("proposal seal appeared before proposal sealing")
        elif primary in {"PROPOSAL_SEALED", "PRIMARY_FRAGMENT_SEALED", "MODEL_FAILURE"}:
            if not _is_hex64(proposal_seal):
                raise ContractError("primary stage lacks its actual proposal-seal file hash")
        if primary in PRIMARY_TERMINAL and not _is_hex64(fragment):
            raise ContractError("primary terminal stage lacks its fragment seal")
        if primary not in PRIMARY_TERMINAL and fragment is not None:
            raise ContractError("primary fragment seal appeared before a terminal primary stage")
        if post in POST_TERMINAL | {"TRUTH_COMPLETE", "SECONDARY_COMPLETE"} and not _is_hex64(truth):
            raise ContractError("post-primary stage lacks its truth fragment hash")
        if post in POST_TERMINAL | {"SECONDARY_COMPLETE"} and not _is_hex64(secondary):
            raise ContractError("post-primary stage lacks its secondary fragment hash")
        if post in {"LOCKED", "NOT_STARTED"} and (truth is not None or secondary is not None):
            raise ContractError("post-primary fragment hash appeared before truth opening")
        if post == "TRUTH_COMPLETE" and secondary is not None:
            raise ContractError("secondary fragment hash appeared before secondary completion")
        failed = primary == "MODEL_FAILURE" or post == "MODEL_FAILURE"
        if failed and not isinstance(failure_stage, str):
            raise ContractError("model-failure state lacks its failure stage")
        if not failed and failure_stage is not None:
            raise ContractError("nonfailure state contains a failure stage")
    if len(primary_active) > 1 or len(post_active) > 1:
        raise ContractError("more than one scientific cell is active")
    active = primary_active + post_active
    expected_active = None if not active else active[0]
    if value.get("active_model_id") != expected_active:
        raise ContractError("campaign active_model_id disagrees with cell stages")

    all_primary = all(row["primary_stage"] in PRIMARY_TERMINAL for row in rows)
    all_post = all(row["post_primary_stage"] in POST_TERMINAL for row in rows)
    if phase == "PRIMARY_COLLECTION":
        if any(row["post_primary_stage"] != "LOCKED" for row in rows):
            raise ContractError("post-primary work opened before the global primary seal")
        if value.get("primary_seal_sha256") is not None:
            raise ContractError("primary seal recorded before leaving primary collection")
    elif phase == "PRIMARY_SEALED":
        if not all_primary or not _is_hex64(value.get("primary_seal_sha256")):
            raise ContractError("PRIMARY_SEALED lacks all 30 primary terminals or seal")
        if any(row["post_primary_stage"] != "LOCKED" for row in rows):
            raise ContractError("PRIMARY_SEALED must retain the post-primary lock")
    elif phase == "POST_PRIMARY_COLLECTION":
        if not all_primary or not _is_hex64(value.get("primary_seal_sha256")):
            raise ContractError("post-primary phase lacks a valid global primary barrier")
        if any(row["post_primary_stage"] == "LOCKED" for row in rows):
            raise ContractError("post-primary collection contains a locked cell")
    elif phase == "FINALIZED":
        if (
            not all_primary
            or not all_post
            or not _is_hex64(value.get("primary_seal_sha256"))
            or not _is_hex64(value.get("final_validation_sha256"))
        ):
            raise ContractError("FINALIZED lacks complete cells or final validation")
        if value.get("active_model_id") is not None:
            raise ContractError("FINALIZED cannot retain an active model")
    return str(observed_hash)


def validate_state_transition(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    cells: Sequence[ModelCell],
    *,
    authorization_sha256: str,
) -> None:
    previous_hash = validate_campaign_state(
        previous, cells, authorization_sha256=authorization_sha256
    )
    validate_campaign_state(current, cells, authorization_sha256=authorization_sha256)
    if current.get("sequence_number") != previous.get("sequence_number") + 1:
        raise ContractError("campaign-state sequence must increase by exactly one")
    if current.get("previous_state_sha256") != previous_hash:
        raise ContractError("campaign-state previous hash mismatch")
    previous_phase = str(previous["campaign_phase"])
    current_phase = str(current["campaign_phase"])
    for field in ("schema_version", "protocol_id", "run_authorization_sha256"):
        if current.get(field) != previous.get(field):
            raise ContractError(f"immutable campaign-state field changed: {field}")
    changed = [
        index
        for index, (left, right) in enumerate(
            zip(previous["cells"], current["cells"], strict=True)
        )
        if left != right
    ]
    if previous_phase == current_phase:
        if current.get("primary_seal_sha256") != previous.get("primary_seal_sha256"):
            raise ContractError("primary seal changed during a per-cell transition")
        if current.get("final_validation_sha256") != previous.get("final_validation_sha256"):
            raise ContractError("final validation changed during a per-cell transition")
        if len(changed) != 1:
            raise ContractError("one state transaction must advance exactly one cell")
        before = previous["cells"][changed[0]]
        after = current["cells"][changed[0]]
        primary_change = before["primary_stage"] != after["primary_stage"]
        post_change = before["post_primary_stage"] != after["post_primary_stage"]
        if primary_change == post_change:
            raise ContractError("one transaction must advance one phase of one cell")
        if primary_change and after["primary_stage"] not in PRIMARY_TRANSITIONS[before["primary_stage"]]:
            raise ContractError("invalid primary-stage transition")
        if post_change and after["post_primary_stage"] not in POST_TRANSITIONS[before["post_primary_stage"]]:
            raise ContractError("invalid post-primary-stage transition")
        if (
            before.get("proposal_seal_sha256") is not None
            and after.get("proposal_seal_sha256")
            != before.get("proposal_seal_sha256")
        ):
            raise ContractError("actual proposal-seal file hash changed after sealing")
    else:
        allowed = {
            "PRIMARY_COLLECTION": {"PRIMARY_SEALED", "HOLD_INCOMPLETE", "INVALID"},
            "PRIMARY_SEALED": {
                "POST_PRIMARY_COLLECTION",
                "HOLD_INCOMPLETE",
                "INVALID",
            },
            "POST_PRIMARY_COLLECTION": {"FINALIZED", "HOLD_INCOMPLETE", "INVALID"},
            "HOLD_INCOMPLETE": set(),
            "INVALID": set(),
            "FINALIZED": {"HOLD_INCOMPLETE", "INVALID"},
        }
        if current_phase not in allowed[previous_phase]:
            raise ContractError(f"invalid campaign-phase transition {previous_phase} -> {current_phase}")
        if previous_phase == "PRIMARY_COLLECTION" and current_phase == "PRIMARY_SEALED":
            if changed:
                raise ContractError("global primary seal transition cannot rewrite cell rows")
            if previous.get("primary_seal_sha256") is not None:
                raise ContractError("primary collection already contained a primary seal")
        elif previous_phase == "PRIMARY_SEALED" and current_phase == "POST_PRIMARY_COLLECTION":
            if len(changed) != len(cells) or any(
                current["cells"][index]["post_primary_stage"] != "NOT_STARTED"
                for index in changed
            ):
                raise ContractError("post-primary unlock must release all 30 cells together")
            if current.get("primary_seal_sha256") != previous.get("primary_seal_sha256"):
                raise ContractError("post-primary unlock changed the primary seal")
        elif current_phase in {"HOLD_INCOMPLETE", "INVALID"}:
            if changed:
                raise ContractError("terminal hold/invalid transition cannot rewrite cell rows")
        elif previous_phase == "POST_PRIMARY_COLLECTION" and current_phase == "FINALIZED":
            if changed:
                raise ContractError("finalization transition cannot rewrite cell rows")
            if current.get("primary_seal_sha256") != previous.get("primary_seal_sha256"):
                raise ContractError("finalization changed the primary seal")
            if previous.get("final_validation_sha256") is not None:
                raise ContractError("post-primary state already contained final validation")


def next_state(
    previous: Mapping[str, Any],
    *,
    cells: Sequence[ModelCell],
    authorization_sha256: str,
    campaign_phase: str | None = None,
    cell_index: int | None = None,
    primary_stage: str | None = None,
    post_primary_stage: str | None = None,
    **updates: Any,
) -> dict[str, Any]:
    output = deepcopy(dict(previous))
    output.pop("state_sha256", None)
    output.pop("state_sha256_definition", None)
    output["sequence_number"] = int(previous["sequence_number"]) + 1
    output["previous_state_sha256"] = previous["state_sha256"]
    if campaign_phase is not None:
        output["campaign_phase"] = campaign_phase
        if campaign_phase == "POST_PRIMARY_COLLECTION":
            for row in output["cells"]:
                row["post_primary_stage"] = "NOT_STARTED"
    if cell_index is not None:
        if cell_index not in range(len(cells)):
            raise ContractError("cell_index is outside the frozen campaign")
        row = output["cells"][cell_index]
        if primary_stage is not None:
            row["primary_stage"] = primary_stage
        if post_primary_stage is not None:
            row["post_primary_stage"] = post_primary_stage
        for key, value in updates.items():
            if key not in row:
                raise ContractError(f"unknown cell-state field {key!r}")
            row[key] = value
    else:
        for key, value in updates.items():
            if key not in output:
                raise ContractError(f"unknown campaign-state field {key!r}")
            output[key] = value
    active = [
        row["model_id"]
        for row in output["cells"]
        if row["primary_stage"] in PRIMARY_ACTIVE or row["post_primary_stage"] in POST_ACTIVE
    ]
    output["active_model_id"] = None if not active else active[0]
    sealed = seal_state(output)
    validate_state_transition(
        previous, sealed, cells, authorization_sha256=authorization_sha256
    )
    return sealed


def write_state_history(root: Path, state: Mapping[str, Any]) -> Path:
    """Write one immutable sequence file and atomically update CURRENT_STATE."""

    directory = Path(root) / "state"
    directory.mkdir(parents=True, exist_ok=True)
    sequence = int(state["sequence_number"])
    path = directory / f"{sequence:06d}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise ContractError(f"campaign-state sequence already exists: {sequence}") from exc
    payload = (
        json.dumps(dict(state), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ContractError("short write while sealing campaign state")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    write_json(
        Path(root) / "CURRENT_STATE.json",
        {
            "protocol_id": PROTOCOL_ID,
            "sequence_number": sequence,
            "state_path": str(path.relative_to(root)),
            "state_sha256": state["state_sha256"],
        },
    )
    return path


def verify_state_history(
    root: Path,
    cells: Sequence[ModelCell],
    *,
    authorization_sha256: str,
) -> dict[str, Any]:
    """Re-read the complete immutable sequence and verify every transition."""

    directory = Path(root) / "state"
    files = sorted(directory.glob("*.json"))
    if not files or [path.name for path in files] != [
        f"{index:06d}.json" for index in range(len(files))
    ]:
        raise ContractError("campaign-state history is absent or noncontiguous")
    states: list[dict[str, Any]] = []
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ContractError("campaign-state history contains an unsafe path")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("campaign-state history contains invalid JSON") from exc
        if not isinstance(value, dict):
            raise ContractError("campaign-state history entry is not an object")
        states.append(value)
    validate_campaign_state(states[0], cells, authorization_sha256=authorization_sha256)
    if states[0].get("sequence_number") != 0:
        raise ContractError("campaign-state history does not begin at sequence zero")
    for previous, current in zip(states, states[1:]):
        validate_state_transition(
            previous,
            current,
            cells,
            authorization_sha256=authorization_sha256,
        )
    pointer_path = Path(root) / "CURRENT_STATE.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("current-state pointer is invalid") from exc
    final = states[-1]
    if pointer != {
        "protocol_id": PROTOCOL_ID,
        "sequence_number": final["sequence_number"],
        "state_path": f"state/{int(final['sequence_number']):06d}.json",
        "state_sha256": final["state_sha256"],
    }:
        raise ContractError("current-state pointer disagrees with immutable history")
    return final


__all__ = [
    "CAMPAIGN_PHASES",
    "POST_PRIMARY_STAGES",
    "PRIMARY_STAGES",
    "initial_campaign_state",
    "next_state",
    "seal_state",
    "validate_campaign_state",
    "validate_state_transition",
    "verify_state_history",
    "write_state_history",
]
