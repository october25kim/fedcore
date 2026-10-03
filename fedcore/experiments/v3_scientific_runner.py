"""Host-side scientific launch staging for the sealed FedCORE PACS v3 run.

This module deliberately does not import torch, inspect PACS records, start a
container, or touch CUDA.  It has two responsibilities only:

* validate the preceding synthetic-binding HOLD without mistaking it for an
  execution-readiness seal; and
* emit the control-schema-2 pre-authorization execution template
  (``HOST_LAUNCH_PLAN.json``): the fixed 30-cell order, the exact five
  read-only inputs, the read-only ``/authorization`` control mount, one empty
  read-write output directory, and the registered single-GPU
  ``docker create`` -> inspect -> start lifecycle of
  ``fedcore.experiments.v3_campaign_runner run``, all bound to one final image
  ID read from the actual ``IMAGE_INSPECT.json`` bytes.

The template contains no authorization-dependent value.  The container name is
a rule, ``fedcore-v3-{authorization_id}``, rendered only after authorization.
There is no execute subcommand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from fedcore.experiments.v3_contract import (
    EXPECTED_MODELS,
    PROTOCOL_ID,
    ContractError,
    ModelCell,
    canonical_json_sha256,
    default_protocol_dir,
    load_model_cells,
    read_csv,
    read_json,
    sha256_file,
    verify_scientific_contract,
)
from fedcore.experiments.v3_input_binding import (
    ISOLATED_INPUT_DIRECTORY,
    REGISTERED_SCIENTIFIC_INPUTS,
    ScientificInputSpec,
)
from fedcore.experiments.v3_execution_binding import (
    CONTAINER_USER,
    CONTAINER_WORKDIR,
    CONTROL_BINDING_MODE,
    CONTROL_SCHEMA_VERSION,
    EXPECTED_HOST_CONTROL_DIR,
    EXPECTED_HOST_LOG_DIR,
    LAUNCH_PLAN_HASH_DEFINITION,
    LAUNCH_PLAN_KEYS,
    LAUNCH_TEMPLATE_STATUS,
    PLAN_KIND,
    RENDERER_VERSION,
    build_execution_template,
    read_control_artifact,
)


PREBINDING_STATUS = (
    "PASS_SYNTHETIC_IMPLEMENTATION_BINDING_"
    "HOLD_SCIENTIFIC_RUNNER_AND_INPUT_BINDING"
)
READINESS_STATUS = "PASS_EXECUTION_READY"
AUTHORIZATION_STATUS = "RUN_AUTHORIZED"
EXPECTED_TRAINING_MANIFEST_SHA256 = (
    "7c55b885e6c245b749ca8ba9d9462e077c85e60fa7a6b9f7cbe2ee0dc1e08bfc"
)
EXPECTED_IMAGE_TAG = "fedcore-v3-hsb:binding-r2"
EXPECTED_GPU_INDEX = 0
EXPECTED_GPU_UUID = "GPU-ee5a0082-32e6-ede4-27d5-441bee0ca7c6"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 4070 Ti SUPER"
MAX_CONCURRENT_SCIENTIFIC_PROCESSES = 1
CONTAINER_OUTPUT_ROOT = Path("/output")
CONTAINER_TMP_ROOT = Path("/tmp")
REGISTERED_OUTPUT_DIR = Path(
    "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_local_v1/runs/scientific/"
    "FEDCORE-IJAR-PACS-AUTHOR-DETERMINISTIC-RERUN-HSB-local-v1"
)


SCIENTIFIC_INPUTS = REGISTERED_SCIENTIFIC_INPUTS

CELL_STAGES = (
    "NOT_STARTED",
    "TRAINING",
    "CHECKPOINT_SEALED",
    "PROPOSAL_SEALED",
    "PRIMARY_COUNTS_SEALED",
    "PRIMARY_DECISIONS_SEALED",
    "TRUTH_COMPLETE",
    "SECONDARY_COMPLETE",
    "COMPLETE",
    "MODEL_FAILURE",
)
TERMINAL_CELL_STAGES = frozenset({"COMPLETE", "MODEL_FAILURE"})
ACTIVE_CELL_STAGES = frozenset(CELL_STAGES) - TERMINAL_CELL_STAGES - {"NOT_STARTED"}
ALLOWED_CELL_TRANSITIONS = {
    "NOT_STARTED": frozenset({"TRAINING"}),
    "TRAINING": frozenset({"CHECKPOINT_SEALED", "MODEL_FAILURE"}),
    "CHECKPOINT_SEALED": frozenset({"PROPOSAL_SEALED"}),
    "PROPOSAL_SEALED": frozenset({"PRIMARY_COUNTS_SEALED"}),
    "PRIMARY_COUNTS_SEALED": frozenset({"PRIMARY_DECISIONS_SEALED"}),
    "PRIMARY_DECISIONS_SEALED": frozenset({"TRUTH_COMPLETE"}),
    "TRUTH_COMPLETE": frozenset({"SECONDARY_COMPLETE"}),
    "SECONDARY_COMPLETE": frozenset({"COMPLETE"}),
    "COMPLETE": frozenset(),
    "MODEL_FAILURE": frozenset(),
}


def _require_bool(value: Any, expected: bool, label: str) -> None:
    if value is not expected:
        raise ContractError(f"{label} must be {expected!r}")


def _fixed_cell_pattern() -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (architecture, split, seed)
        for split in range(5)
        for seed in range(3)
        for architecture in ("resnet18", "convnext_tiny")
    )


def fixed_model_cells(protocol_dir: Path | None = None) -> tuple[ModelCell, ...]:
    """Load and verify the immutable split, seed, and architecture order."""

    root = default_protocol_dir() if protocol_dir is None else Path(protocol_dir)
    verify_scientific_contract(root)
    cells = load_model_cells(root)
    pattern = _fixed_cell_pattern()
    observed = tuple(
        (cell.architecture, cell.split, cell.nominal_training_seed) for cell in cells
    )
    if observed != pattern:
        raise ContractError(
            "30-cell order drift: expected split asc, seed asc, then "
            "resnet18 before convnext_tiny"
        )
    for cell, (architecture, split, seed) in zip(cells, pattern, strict=True):
        expected_id = f"pacs__{architecture}__split{split:02d}__seed{seed}"
        if cell.model_id != expected_id:
            raise ContractError(
                f"model_id/order mismatch: expected {expected_id}, observed {cell.model_id}"
            )
    training_manifest = root / "TRAINING_SEED_MANIFEST.csv"
    if sha256_file(training_manifest) != EXPECTED_TRAINING_MANIFEST_SHA256:
        raise ContractError("training seed manifest hash drift")
    seed_rows = read_csv(training_manifest)
    if len(seed_rows) != EXPECTED_MODELS:
        raise ContractError("training seed denominator drift")
    for cell, row in zip(cells, seed_rows, strict=True):
        expected = (
            cell.model_id,
            cell.architecture,
            cell.split,
            cell.nominal_training_seed,
            cell.model_init_seed_uint64,
        )
        observed_row = (
            row["model_id"],
            row["architecture"],
            int(row["split"]),
            int(row["nominal_seed"]),
            int(row["model_init_seed_uint64"]),
        )
        if observed_row != expected:
            raise ContractError(f"training seed binding drift for {cell.model_id}")
    return cells


def cell_order_rows(cells: Sequence[ModelCell]) -> list[dict[str, Any]]:
    if len(cells) != EXPECTED_MODELS:
        raise ContractError("cell order must contain exactly 30 models")
    return [
        {
            "ordinal": ordinal,
            "model_id": cell.model_id,
            "architecture": cell.architecture,
            "split": cell.split,
            "nominal_training_seed": cell.nominal_training_seed,
            "model_init_seed_uint64": cell.model_init_seed_uint64,
        }
        for ordinal, cell in enumerate(cells)
    ]


def cell_order_sha256(cells: Sequence[ModelCell]) -> str:
    return canonical_json_sha256(cell_order_rows(cells))


def initial_cell_state(cells: Sequence[ModelCell]) -> dict[str, Any]:
    order = cell_order_rows(cells)
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "cell_order_sha256": canonical_json_sha256(order),
        "max_concurrent_scientific_processes": MAX_CONCURRENT_SCIENTIFIC_PROCESSES,
        "cells": [
            {**row, "stage": "NOT_STARTED"}
            for row in order
        ],
    }


def validate_cell_state(state: Mapping[str, Any], cells: Sequence[ModelCell]) -> None:
    """Fail closed on reordering, skipped cells, or concurrent active cells."""

    expected_rows = cell_order_rows(cells)
    if state.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("cell-state protocol mismatch")
    if state.get("cell_order_sha256") != canonical_json_sha256(expected_rows):
        raise ContractError("cell-state order hash mismatch")
    if state.get("max_concurrent_scientific_processes") != 1:
        raise ContractError("scientific execution must be single-process")
    rows = state.get("cells")
    if not isinstance(rows, list) or len(rows) != EXPECTED_MODELS:
        raise ContractError("cell-state denominator must equal 30")
    seen_open = False
    active = 0
    for expected, observed in zip(expected_rows, rows, strict=True):
        if not isinstance(observed, Mapping):
            raise ContractError("cell-state row must be an object")
        for field, value in expected.items():
            if observed.get(field) != value:
                raise ContractError(f"cell-state order drift at {expected['model_id']}")
        stage = observed.get("stage")
        if stage not in CELL_STAGES:
            raise ContractError(f"unknown cell stage {stage!r}")
        if stage in ACTIVE_CELL_STAGES:
            active += 1
            if active > 1:
                raise ContractError("more than one active scientific cell")
            if seen_open:
                raise ContractError("active cell appears after an unfinished prefix")
            seen_open = True
        elif stage == "NOT_STARTED":
            seen_open = True
        elif stage in TERMINAL_CELL_STAGES and seen_open:
            raise ContractError("terminal cell appears after an unfinished cell")


def validate_cell_state_transition(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    cells: Sequence[ModelCell],
) -> None:
    """Allow one registered stage advance and reject skips or rewrites."""

    validate_cell_state(previous, cells)
    validate_cell_state(current, cells)
    previous_rows = previous["cells"]
    current_rows = current["cells"]
    changed = [
        index
        for index, (left, right) in enumerate(
            zip(previous_rows, current_rows, strict=True)
        )
        if left["stage"] != right["stage"]
    ]
    if len(changed) > 1:
        raise ContractError("one state transaction may advance only one model cell")
    if not changed:
        return
    index = changed[0]
    before = str(previous_rows[index]["stage"])
    after = str(current_rows[index]["stage"])
    if after not in ALLOWED_CELL_TRANSITIONS[before]:
        raise ContractError(f"invalid cell-stage transition {before} -> {after}")


def validate_prebinding_gate(path: Path) -> dict[str, Any]:
    """Validate the earlier public HOLD used as the binding prerequisite."""

    gate = read_json(Path(path))
    if gate.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("prebinding gate protocol mismatch")
    if gate.get("status") != PREBINDING_STATUS:
        raise ContractError("synthetic prebinding HOLD is absent")
    _require_bool(gate.get("training_authorized"), False, "training_authorized")
    _require_bool(gate.get("scientific_runner_bound"), False, "scientific_runner_bound")
    _require_bool(gate.get("scientific_inputs_bound"), False, "scientific_inputs_bound")
    _require_bool(gate.get("runs_scientific_empty"), True, "runs_scientific_empty")
    for field in ("pacs_training_started", "pacs_inference_started"):
        _require_bool(gate.get(field), False, field)
    return gate


def validate_readiness_gate(
    path: Path,
    *,
    launch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Refuse readiness until the end-to-end artifact chain is implemented.

    This release binds inputs and a no-GPU mount probe only.  It deliberately
    does not implement an executable campaign orchestrator, a mount-receipt
    hash in the readiness artifact, or in-container verification of the
    authorization files.  Consequently no JSON document can promote this
    static binding to ``PASS_EXECUTION_READY``.
    """

    del path, launch_plan
    raise ContractError(
        "PASS_EXECUTION_READY refused: end-to-end artifact-bound runner is not implemented"
    )


def _regular_non_symlink(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ContractError(f"scientific input is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ContractError(f"scientific input must not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"scientific input is not a regular file: {path}")


def validate_input_binding(
    path: Path,
    *,
    specs: Sequence[ScientificInputSpec] = SCIENTIFIC_INPUTS,
    expected_input_dir: Path = ISOLATED_INPUT_DIRECTORY,
) -> list[dict[str, Any]]:
    """Verify the five staged files and their in-container destinations."""

    binding = read_json(Path(path))
    if binding.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("input binding protocol mismatch")
    if binding.get("status") != "PASS_FIVE_INPUT_BINDING":
        raise ContractError("PASS_FIVE_INPUT_BINDING is absent")
    if binding.get("isolated_input_file_count") != len(specs):
        raise ContractError("isolated input denominator drift")
    for field in (
        "exact_directory_contents",
        "all_regular_nonsymlink",
        "all_read_only",
    ):
        _require_bool(binding.get(field), True, field)
    input_dir = Path(str(binding.get("isolated_input_directory")))
    if input_dir.resolve() != Path(expected_input_dir).resolve():
        raise ContractError("isolated input directory drift")
    records = binding.get("files")
    if not isinstance(records, list) or len(records) != len(specs):
        raise ContractError("input binding must contain exactly five records")
    normalized: list[dict[str, Any]] = []
    for spec, record in zip(specs, records, strict=True):
        if not isinstance(record, Mapping):
            raise ContractError("input binding record must be an object")
        provenance_source = Path(str(record.get("source_path")))
        source = Path(str(record.get("staged_path")))
        destination = Path(str(record.get("container_path")))
        if record.get("name") != spec.name:
            raise ContractError(f"input order/name drift: expected {spec.name}")
        if provenance_source != Path(spec.source_path):
            raise ContractError(f"input provenance-source drift for {spec.name}")
        expected_staged = Path(expected_input_dir) / spec.name
        if source.resolve() != expected_staged.resolve():
            raise ContractError(f"staged input source drift for {spec.name}")
        if destination != Path(spec.container_path):
            raise ContractError(f"input destination drift for {spec.name}")
        if record.get("read_only_mount_required") is not True:
            raise ContractError(f"input is not declared read-only: {spec.name}")
        if record.get("symlink") is not False:
            raise ContractError(f"input symlink declaration invalid: {spec.name}")
        if record.get("regular_file") is not True:
            raise ContractError(f"input regular-file declaration invalid: {spec.name}")
        _regular_non_symlink(source)
        observed_bytes = source.stat().st_size
        if (
            observed_bytes != spec.size_bytes
            or int(record.get("size_bytes", -1)) != spec.size_bytes
        ):
            raise ContractError(f"input byte-size drift for {spec.name}")
        observed_hash = sha256_file(source)
        if observed_hash != spec.sha256 or record.get("sha256") != spec.sha256:
            raise ContractError(f"input SHA-256 drift for {spec.name}")
        if stat.S_IMODE(source.lstat().st_mode) & 0o222:
            raise ContractError(f"staged input remains writable: {spec.name}")
        normalized.append(
            {
                "name": spec.name,
                "source": str(source),
                "provenance_source": str(provenance_source),
                "realpath": str(source.resolve()),
                "destination": str(destination),
                "read_only": True,
                "sha256": observed_hash,
                "bytes": observed_bytes,
                "symlink": False,
            }
        )
    return normalized


def validate_empty_output_dir(
    output_dir: Path,
    *,
    expected_output_dir: Path = REGISTERED_OUTPUT_DIR,
) -> Path:
    output_dir = Path(output_dir)
    try:
        metadata = output_dir.lstat()
    except FileNotFoundError as error:
        raise ContractError(f"scientific output directory is missing: {output_dir}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ContractError("scientific output must be a real directory, not a symlink")
    if output_dir.resolve() != Path(expected_output_dir).resolve():
        raise ContractError("scientific output directory drift")
    if next(output_dir.iterdir(), None) is not None:
        raise ContractError("scientific output directory must be empty at launch seal")
    return output_dir.resolve()


def _image_identity(image_inspect: Path) -> tuple[str, str, list[str], str]:
    """Return (image ID, source revision, environment, inspect file hash) from actual bytes."""

    artifact = read_control_artifact(Path(image_inspect), require_object=False)
    value = artifact.value
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
        raise ContractError("image inspect must contain exactly one image")
    image = value[0]
    image_id = image.get("Id")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise ContractError("container image ID must use an immutable sha256 reference")
    digest = image_id.removeprefix("sha256:")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ContractError("container image ID is not a valid SHA-256 digest")
    config = image.get("Config")
    if not isinstance(config, Mapping):
        raise ContractError("image inspect lacks Config")
    labels = config.get("Labels") or {}
    revision = labels.get("org.opencontainers.image.revision") if isinstance(labels, Mapping) else None
    if not isinstance(revision, str):
        raise ContractError("image inspect lacks a source revision label")
    environment = config.get("Env")
    if not isinstance(environment, list) or not all(isinstance(item, str) for item in environment):
        raise ContractError("image inspect environment must be a list of strings")
    if (
        config.get("Entrypoint") != ["python"]
        or config.get("User") != CONTAINER_USER
        or config.get("WorkingDir") != CONTAINER_WORKDIR
    ):
        raise ContractError("image inspect entrypoint/user/workdir mismatch")
    return image_id, revision, list(environment), artifact.sha256


def build_launch_plan(
    *,
    prebinding_gate: Path,
    input_binding: Path,
    output_dir: Path,
    image_inspect: Path,
    protocol_dir: Path | None = None,
    specs: Sequence[ScientificInputSpec] = SCIENTIFIC_INPUTS,
    expected_input_dir: Path = ISOLATED_INPUT_DIRECTORY,
    expected_output_dir: Path = REGISTERED_OUTPUT_DIR,
    control_dir: Path | None = None,
    log_dir: Path | None = None,
    image_tag: str = EXPECTED_IMAGE_TAG,
) -> dict[str, Any]:
    """Build the unauthorized, non-executing pre-authorization execution template.

    The image ID and source revision are read from the actual image-inspect
    bytes; no caller-supplied image ID is accepted.
    """

    validate_prebinding_gate(prebinding_gate)
    image_id, source_commit, environment, image_inspect_sha256 = _image_identity(image_inspect)
    cells = fixed_model_cells(protocol_dir)
    inputs = validate_input_binding(
        input_binding,
        specs=specs,
        expected_input_dir=expected_input_dir,
    )
    output = validate_empty_output_dir(
        output_dir, expected_output_dir=expected_output_dir
    )
    return assemble_launch_plan(
        prebinding_gate_path=str(Path(prebinding_gate).resolve()),
        prebinding_gate_sha256=sha256_file(Path(prebinding_gate)),
        input_binding_path=str(Path(input_binding).resolve()),
        input_binding_sha256=sha256_file(Path(input_binding)),
        inputs=inputs,
        image_id=image_id,
        image_tag=image_tag,
        image_inspect_sha256=image_inspect_sha256,
        source_commit=source_commit,
        environment=environment,
        cells=cells,
        output_dir=output,
        input_dir=Path(expected_input_dir),
        control_dir=EXPECTED_HOST_CONTROL_DIR if control_dir is None else Path(control_dir),
        log_dir=EXPECTED_HOST_LOG_DIR if log_dir is None else Path(log_dir),
    )


def assemble_launch_plan(
    *,
    prebinding_gate_path: str,
    prebinding_gate_sha256: str,
    input_binding_path: str,
    input_binding_sha256: str,
    inputs: Sequence[Mapping[str, Any]],
    image_id: str,
    image_tag: str,
    image_inspect_sha256: str,
    source_commit: str,
    environment: Sequence[str],
    cells: Sequence[ModelCell],
    output_dir: Path,
    input_dir: Path,
    control_dir: Path,
    log_dir: Path,
) -> dict[str, Any]:
    """Assemble the template from already-validated values; performs no file I/O."""

    state = initial_cell_state(cells)
    validate_cell_state(state, cells)
    output = Path(output_dir)
    template = build_execution_template(
        image_id=image_id,
        source_commit=source_commit,
        environment=environment,
        inputs=inputs,
        cell_order_sha256=cell_order_sha256(cells),
        control_dir=Path(control_dir),
        input_dir=Path(input_dir),
        output_dir=output,
        log_dir=Path(log_dir),
    )
    body: dict[str, Any] = {
        "schema_version": 2,
        "control_schema_version": CONTROL_SCHEMA_VERSION,
        "control_binding_mode": CONTROL_BINDING_MODE,
        "plan_kind": PLAN_KIND,
        "renderer_version": RENDERER_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": LAUNCH_TEMPLATE_STATUS,
        "execution_allowed": False,
        "execution_readiness_required": True,
        "run_authorization_required": True,
        "prebinding_status": PREBINDING_STATUS,
        "prebinding_gate_path": prebinding_gate_path,
        "prebinding_gate_sha256": prebinding_gate_sha256,
        "input_binding_path": input_binding_path,
        "input_binding_sha256": input_binding_sha256,
        "input_inventory": [dict(row) for row in inputs],
        "image": {"tag": image_tag, "id": image_id},
        "image_inspect_sha256": image_inspect_sha256,
        "source_commit": source_commit,
        "runtime": {
            "network": "none",
            "root_filesystem_read_only": True,
            "user": CONTAINER_USER,
            "gpu_index": EXPECTED_GPU_INDEX,
            "gpu_uuid": EXPECTED_GPU_UUID,
            "gpu_name": EXPECTED_GPU_NAME,
            "max_concurrent_scientific_processes": 1,
            "single_process": True,
        },
        "output_directory": str(output),
        "output_empty_at_plan": True,
        "cell_order": cell_order_rows(cells),
        "cell_order_sha256": cell_order_sha256(cells),
        "initial_cell_state": state,
        "stage_order": [
            "training",
            "checkpoint_seal",
            "proposal_seal",
            "primary_audit",
            "primary_decision_seal",
            "post_decision_truth",
            "secondary_audits",
        ],
        "execution_template": template,
        "execution_template_sha256": canonical_json_sha256(template),
        "PACS_archive_hashed_for_binding": True,
        "PACS_records_decoded": False,
        "PACS_labels_inspected": False,
        "PACS_scientific_content_opened": False,
        "torch_imported": False,
        "CUDA_touched": False,
    }
    body["launch_plan_sha256"] = canonical_json_sha256(body)
    body["launch_plan_hash_definition"] = LAUNCH_PLAN_HASH_DEFINITION
    if set(body) != set(LAUNCH_PLAN_KEYS):
        raise ContractError("launch-plan builder key drift")
    return body


def validate_launch_plan_integrity(launch_plan: Mapping[str, Any]) -> None:
    """Recompute the self-hash and recheck mutable host files before declaration."""

    if launch_plan.get("schema_version") != 2 or launch_plan.get(
        "control_schema_version"
    ) != CONTROL_SCHEMA_VERSION:
        raise ContractError("launch plan is not control schema 2")
    if set(launch_plan) != set(LAUNCH_PLAN_KEYS):
        raise ContractError("launch-plan key drift")
    if launch_plan.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("launch-plan protocol mismatch")
    if launch_plan.get("status") != LAUNCH_TEMPLATE_STATUS or launch_plan.get(
        "plan_kind"
    ) != PLAN_KIND:
        raise ContractError("launch-plan HOLD status mismatch")
    _require_bool(launch_plan.get("execution_allowed"), False, "execution_allowed")
    body = {
        key: value
        for key, value in launch_plan.items()
        if key not in {"launch_plan_sha256", "launch_plan_hash_definition"}
    }
    if canonical_json_sha256(body) != launch_plan.get("launch_plan_sha256"):
        raise ContractError("launch-plan self-hash mismatch")
    prebinding_path = Path(str(launch_plan.get("prebinding_gate_path")))
    input_binding_path = Path(str(launch_plan.get("input_binding_path")))
    if sha256_file(prebinding_path) != launch_plan.get("prebinding_gate_sha256"):
        raise ContractError("prebinding gate changed after launch planning")
    if sha256_file(input_binding_path) != launch_plan.get("input_binding_sha256"):
        raise ContractError("input binding changed after launch planning")
    validate_prebinding_gate(prebinding_path)
    binding = read_json(input_binding_path)
    if binding.get("status") != "PASS_FIVE_INPUT_BINDING":
        raise ContractError("input binding no longer has PASS_FIVE_INPUT_BINDING")
    inventory = launch_plan.get("input_inventory")
    if not isinstance(inventory, list) or len(inventory) != 5:
        raise ContractError("launch-plan input inventory drift")
    binding_rows = binding.get("files")
    if not isinstance(binding_rows, list) or len(binding_rows) != 5:
        raise ContractError("input-binding receipt denominator drift")
    for record, binding_record in zip(inventory, binding_rows, strict=True):
        if not isinstance(record, Mapping) or not isinstance(binding_record, Mapping):
            raise ContractError("input inventory row must be an object")
        expected_projection = {
            "name": binding_record.get("name"),
            "source": binding_record.get("staged_path"),
            "provenance_source": binding_record.get("source_path"),
            "destination": binding_record.get("container_path"),
            "read_only": binding_record.get("read_only_mount_required"),
            "sha256": binding_record.get("sha256"),
            "bytes": binding_record.get("size_bytes"),
            "symlink": binding_record.get("symlink"),
        }
        for key, value in expected_projection.items():
            if record.get(key) != value:
                raise ContractError(f"launch-plan/input-binding drift for {key}")
        source = Path(str(record["source"]))
        _regular_non_symlink(source)
        if source.resolve() != Path(str(record["realpath"])).resolve():
            raise ContractError(f"input realpath changed after planning: {record['name']}")
        if source.stat().st_size != int(record["bytes"]):
            raise ContractError(f"input size changed after planning: {record['name']}")
        if stat.S_IMODE(source.lstat().st_mode) & 0o222:
            raise ContractError(f"input became writable after planning: {record['name']}")
        if sha256_file(source) != record["sha256"]:
            raise ContractError(f"input hash changed after planning: {record['name']}")
    output = Path(str(launch_plan.get("output_directory")))
    validate_empty_output_dir(output, expected_output_dir=output)
    template = launch_plan.get("execution_template")
    if not isinstance(template, Mapping) or not isinstance(template.get("paths"), Mapping):
        raise ContractError("launch-plan execution template is absent")
    paths = template["paths"]
    if any(Path(str(row["source"])).parent != Path(str(paths.get("input"))) for row in inventory):
        raise ContractError("launch-plan staged inputs are outside the template input directory")
    rebuilt = build_execution_template(
        image_id=str(launch_plan["image"]["id"]),
        source_commit=str(launch_plan.get("source_commit")),
        environment=template.get("environment"),
        inputs=inventory,
        cell_order_sha256=str(launch_plan.get("cell_order_sha256")),
        control_dir=Path(str(paths.get("control"))),
        input_dir=Path(str(paths.get("input"))),
        output_dir=output,
        log_dir=Path(str(paths.get("log"))),
    )
    if dict(template) != rebuilt:
        raise ContractError("launch-plan execution template drift")
    if canonical_json_sha256(rebuilt) != launch_plan.get("execution_template_sha256"):
        raise ContractError("launch-plan execution-template hash mismatch")


def validate_run_authorization(
    path: Path,
    launch_plan: Mapping[str, Any],
    readiness_gate: Path,
) -> dict[str, Any]:
    """Validate explicit authorization separately from technical readiness."""

    validate_readiness_gate(readiness_gate, launch_plan=launch_plan)
    readiness_hash = sha256_file(Path(readiness_gate))
    authorization = read_json(Path(path))
    if authorization.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("run authorization protocol mismatch")
    if authorization.get("status") != AUTHORIZATION_STATUS:
        raise ContractError("RUN_AUTHORIZATION status is absent")
    _require_bool(authorization.get("training_authorized"), True, "training_authorized")
    if authorization.get("launch_plan_sha256") != launch_plan.get("launch_plan_sha256"):
        raise ContractError("run authorization launch-plan hash mismatch")
    if authorization.get("readiness_gate_sha256") != readiness_hash:
        raise ContractError("run authorization readiness hash mismatch")
    if authorization.get("input_binding_sha256") != launch_plan.get(
        "input_binding_sha256"
    ):
        raise ContractError("run authorization input-binding hash mismatch")
    if authorization.get("gpu_uuid") != EXPECTED_GPU_UUID:
        raise ContractError("run authorization GPU UUID mismatch")
    if authorization.get("max_concurrent_scientific_processes") != 1:
        raise ContractError("run authorization must retain single-process execution")
    if not str(authorization.get("authorized_by", "")).strip():
        raise ContractError("run authorization lacks an authorizing identity")
    return authorization


def bind_authorized_launch(
    launch_plan: Mapping[str, Any], readiness_gate: Path, authorization_path: Path
) -> dict[str, Any]:
    """Legacy entry point; it can never emit an argument vector.

    Control schema 2 renders the authorized Docker create plan only through
    ``v3_execution_binding.build_authorized_campaign_plan`` from a validated
    template and authorization.  This function keeps its refusal semantics.
    """

    validate_run_authorization(authorization_path, launch_plan, readiness_gate)
    raise ContractError("legacy launch binding is superseded by the control-schema-2 renderer")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-only", "plan"):
        command = subparsers.add_parser(name)
        command.add_argument("--prebinding-gate", type=Path, required=True)
        command.add_argument("--input-binding", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--protocol-dir", type=Path)
        command.add_argument("--image-tag", default=EXPECTED_IMAGE_TAG)
        command.add_argument("--image-inspect", type=Path, required=True)
    authorize = subparsers.add_parser("authorization-check")
    authorize.add_argument("--launch-plan", type=Path, required=True)
    authorize.add_argument("--readiness-gate", type=Path, required=True)
    authorize.add_argument("--run-authorization", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "authorization-check":
        plan = read_json(args.launch_plan)
        result = bind_authorized_launch(
            plan, args.readiness_gate, args.run_authorization
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    plan = build_launch_plan(
        prebinding_gate=args.prebinding_gate,
        input_binding=args.input_binding,
        output_dir=args.output_dir,
        protocol_dir=args.protocol_dir,
        image_tag=args.image_tag,
        image_inspect=args.image_inspect,
    )
    if args.command == "validate-only":
        result = {
            "protocol_id": PROTOCOL_ID,
            "status": "PASS_VALIDATE_ONLY_HOLD_EXECUTION_READINESS",
            "launch_plan_sha256": plan["launch_plan_sha256"],
            "execution_allowed": False,
            "PACS_archive_hashed_for_binding": True,
            "PACS_records_decoded": False,
            "PACS_labels_inspected": False,
            "PACS_scientific_content_opened": False,
            "torch_imported": False,
            "CUDA_touched": False,
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVE_CELL_STAGES",
    "AUTHORIZATION_STATUS",
    "CELL_STAGES",
    "EXPECTED_GPU_UUID",
    "MAX_CONCURRENT_SCIENTIFIC_PROCESSES",
    "PREBINDING_STATUS",
    "READINESS_STATUS",
    "REGISTERED_OUTPUT_DIR",
    "SCIENTIFIC_INPUTS",
    "ScientificInputSpec",
    "assemble_launch_plan",
    "bind_authorized_launch",
    "build_launch_plan",
    "cell_order_rows",
    "cell_order_sha256",
    "fixed_model_cells",
    "initial_cell_state",
    "main",
    "validate_cell_state",
    "validate_cell_state_transition",
    "validate_empty_output_dir",
    "validate_input_binding",
    "validate_launch_plan_integrity",
    "validate_prebinding_gate",
    "validate_readiness_gate",
    "validate_run_authorization",
]
