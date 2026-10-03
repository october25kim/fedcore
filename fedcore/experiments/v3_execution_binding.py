"""Artifact-bound execution contract for the FedCORE PACS v3 campaign.

The earlier static-binding release deliberately stops before scientific
execution.  This module adds the next contract layer without weakening that
boundary.  Technical readiness, human run authorization, the authorized
Docker create plan, and the post-create inspect receipt are distinct files.
The scientific container must re-read and re-hash their actual mounted bytes
before it may import torch, open PACS, or touch CUDA.

This is an evidence contract, not a cryptographic identity protocol.  The host
operator remains trusted to issue ``RUN_AUTHORIZATION.json``.  A self-hash
detects accidental or post-issuance mutation; it does not prove who wrote the
file.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat
import subprocess
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ElementTree

from fedcore.experiments.v3_contract import (
    PROTOCOL_ID,
    ContractError,
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_file,
)
from fedcore.experiments.v3_input_binding import (
    ISOLATED_INPUT_DIRECTORY,
    REGISTERED_SCIENTIFIC_INPUTS,
)


READINESS_STATUS = "PASS_EXECUTION_READY_HOLD_RUN_AUTHORIZATION"
AUTHORIZATION_STATUS = "RUN_AUTHORIZED"
AUTHORIZED_PLAN_STATUS = "PASS_AUTHORIZED_CAMPAIGN_PLAN_HOLD_CONTAINER_START"
RUNTIME_RECEIPT_STATUS = "PASS_CREATED_CONTAINER_INSPECTED_HOLD_START"
CONTAINER_AUTHORIZATION_DIR = Path("/authorization")
CONTAINER_OUTPUT_DIR = Path("/output")
CANONICAL_CONTAINER_WORKSPACE_DIR = Path("/workspace")
CONTAINER_WORKSPACE_DIR = CANONICAL_CONTAINER_WORKSPACE_DIR
EXPECTED_GPU_UUID = "GPU-ee5a0082-32e6-ede4-27d5-441bee0ca7c6"
AUTHORIZATION_METHOD = "trusted_host_manual_artifact"
AUTHORIZED_BY = "Sanghoon Kim"
SOURCE_MANIFEST_STATUS = "PASS_EXACT_SCIENTIFIC_SOURCE_MANIFEST"
SOURCE_PROBE_STATUS = "PASS_CPU_ONLY_IMAGE_SOURCE_PROBE_NO_PACS_NO_GPU"
TEST_REPORT_STATUS = "PASS_PACS_FREE_EXECUTION_GATE_SUITE"
SOURCE_MANIFEST_LABEL = "org.opencontainers.image.source-manifest-sha256"
RUNTIME_INVENTORY_LABEL = "org.opencontainers.image.runtime-inventory-sha256"
BUILD_INVENTORY_LABEL = "org.opencontainers.image.build-inventory-sha256"
DOCKERFILE_LABEL = "org.opencontainers.image.dockerfile-sha256"
REQUIREMENTS_LABEL = "org.opencontainers.image.requirements-lock-sha256"
# The original 48-hour scientific cutoff is superseded pre-execution by the
# author-approved resource amendments.  This 30-day watchdog is an orphan-process
# safety bound, not a scientific stopping or outcome-selection rule.
TOTAL_GPU_HOUR_CAP = 722.0
SEALED_CAPACITY_GPU_HOURS = 2.0
POST_TIMEOUT_CLEANUP_COMMAND_BOUND_SECONDS = 90
SCIENTIFIC_CLEANUP_RESERVE_SECONDS = 300
SCIENTIFIC_CLEANUP_SAFETY_MARGIN_SECONDS = (
    SCIENTIFIC_CLEANUP_RESERVE_SECONDS - POST_TIMEOUT_CLEANUP_COMMAND_BOUND_SECONDS
)
SCIENTIFIC_EXECUTION_TIMEOUT_SECONDS = (
    int((TOTAL_GPU_HOUR_CAP - SEALED_CAPACITY_GPU_HOURS) * 60 * 60)
    - SCIENTIFIC_CLEANUP_RESERVE_SECONDS
)
REMAINING_SCIENTIFIC_WALLTIME_HOURS = (
    SCIENTIFIC_EXECUTION_TIMEOUT_SECONDS / (60 * 60)
)
CPU_LIMIT_CORES = 0.5
MAXIMUM_CPU_CORE_HOURS = CPU_LIMIT_CORES * REMAINING_SCIENTIFIC_WALLTIME_HOURS
DOCKER_NANO_CPUS = 500_000_000


def _validate_gpu_resource_cap_invariant() -> None:
    """Fail closed unless execution plus bounded cleanup fits the 48 h cap."""

    total_seconds = int(TOTAL_GPU_HOUR_CAP * 60 * 60)
    sealed_seconds = int(SEALED_CAPACITY_GPU_HOURS * 60 * 60)
    if (
        SCIENTIFIC_CLEANUP_RESERVE_SECONDS
        < POST_TIMEOUT_CLEANUP_COMMAND_BOUND_SECONDS
    ):
        raise ContractError("scientific cleanup reserve is smaller than its command bound")
    if SCIENTIFIC_CLEANUP_SAFETY_MARGIN_SECONDS != (
        SCIENTIFIC_CLEANUP_RESERVE_SECONDS
        - POST_TIMEOUT_CLEANUP_COMMAND_BOUND_SECONDS
    ):
        raise ContractError("scientific cleanup safety margin drift")
    if (
        sealed_seconds
        + SCIENTIFIC_EXECUTION_TIMEOUT_SECONDS
        + SCIENTIFIC_CLEANUP_RESERVE_SECONDS
        > total_seconds
    ):
        raise ContractError("scientific timeout plus cleanup reserve exceeds the GPU cap")
    if (
        REMAINING_SCIENTIFIC_WALLTIME_HOURS * 60 * 60
        != SCIENTIFIC_EXECUTION_TIMEOUT_SECONDS
    ):
        raise ContractError("scientific timeout hour/second representation drift")
    if MAXIMUM_CPU_CORE_HOURS != (
        CPU_LIMIT_CORES * SCIENTIFIC_EXECUTION_TIMEOUT_SECONDS / (60 * 60)
    ):
        raise ContractError("scientific CPU-hour cap drift")


SCIENTIFIC_RUNTIME_SINGLETONS = ("pyproject.toml",)
SCIENTIFIC_RUNTIME_ROOTS = (
    "fedcore",
    "paper/ijar-v34-pacs-fresh-seed-hsb-v3",
)
TEST_RUNTIME_PYTEST_VERSION = "9.1.1"
TEST_RUNTIME_ROOT = "/opt/fedcore-test-runtime"
TEST_RUNTIME_WHEEL_FILES = (
    "docker/pytest-wheelhouse/exceptiongroup-1.3.1-py3-none-any.whl",
    "docker/pytest-wheelhouse/iniconfig-2.3.0-py3-none-any.whl",
    "docker/pytest-wheelhouse/packaging-26.3-py3-none-any.whl",
    "docker/pytest-wheelhouse/pluggy-1.6.0-py3-none-any.whl",
    "docker/pytest-wheelhouse/pygments-2.21.0-py3-none-any.whl",
    "docker/pytest-wheelhouse/pytest-9.1.1-py3-none-any.whl",
    "docker/pytest-wheelhouse/tomli-2.4.1-py3-none-any.whl",
    "docker/pytest-wheelhouse/typing_extensions-4.16.0-py3-none-any.whl",
)
SCIENTIFIC_BUILD_FILES = (
    "docker/Dockerfile.v3_hsb_scientific",
    "docker/Dockerfile.v3_hsb_scientific.dockerignore",
    *TEST_RUNTIME_WHEEL_FILES,
    "requirements.lock",
)
REQUIRED_RUNTIME_SOURCE_FILES = (
    "fedcore/certificate/cp.py",
    "fedcore/officehome_rescue.py",
    "fedcore/experiments/v3_campaign_finalizer.py",
    "fedcore/experiments/v3_campaign_runner.py",
    "fedcore/experiments/v3_campaign_state.py",
    "fedcore/experiments/v3_contract.py",
    "fedcore/experiments/v3_counts.py",
    "fedcore/experiments/v3_execution_binding.py",
    "fedcore/experiments/v3_hsb.py",
    "fedcore/experiments/v3_local_control_issuer.py",
    "fedcore/experiments/v3_orchestrator.py",
    "fedcore/experiments/v3_pacs_data.py",
    "fedcore/experiments/v3_posttrain.py",
    "fedcore/experiments/v3_proposal.py",
    "fedcore/experiments/v3_train.py",
    "paper/ijar-v34-pacs-fresh-seed-hsb-v3/PACS_MODEL_MATRIX.csv",
    "paper/ijar-v34-pacs-fresh-seed-hsb-v3/PROSPECTIVE_CONFIRMATION_PROTOCOL.json",
    "paper/ijar-v34-pacs-fresh-seed-hsb-v3/OUTCOME_SCHEMA.json",
    "paper/ijar-v34-pacs-fresh-seed-hsb-v3/H_S_B_ANALYSIS_PLAN.json",
)
REQUIRED_EXECUTION_TEST_FILES = (
    "tests/test_v3_campaign_finalizer.py",
    "tests/test_v3_campaign_runner.py",
    "tests/test_v3_campaign_runner_safety.py",
    "tests/test_v3_campaign_state.py",
    "tests/test_v3_execution_binding.py",
    "tests/test_v3_host_executor.py",
    "tests/test_v3_hsb_binding.py",
    "tests/test_v3_input_binding.py",
    "tests/test_v3_mount_receipt.py",
    "tests/test_v3_pacs_stage_isolation.py",
    "tests/test_v3_posttrain.py",
    "tests/test_v3_posttrain_serialized_bool.py",
    "tests/test_v3_scientific_runner.py",
    "tests/test_v3_training_primitives.py",
)
REQUIRED_EXECUTION_TEST_GATES = (
    "campaign_runner_synthetic_e2e",
    "host_executor_synthetic_e2e",
    "resume_semantics",
    "failure_ledger",
    "resource_cap",
    "source_closure",
    "output_schema",
    "input_parity",
    "hsb_parity",
    "independent_replay",
    "authorization_replay_rejection",
    "pacs_stage_isolation",
    "test_runtime_provenance",
)
REQUIRED_EXECUTION_GATE_NODEIDS = {
    "campaign_runner_synthetic_e2e": (
        "tests/test_v3_campaign_runner_safety.py::"
        "test_run_campaign_pacs_free_calls_full_fresh_orchestration_in_order"
    ),
    "host_executor_synthetic_e2e": (
        "tests/test_v3_host_executor.py::"
        "test_prepare_then_start_reinspects_exact_stopped_container"
    ),
    "resume_semantics": (
        "tests/test_v3_hsb_binding.py::test_resume_matches_uninterrupted_artifact_hashes"
    ),
    "failure_ledger": (
        "tests/test_v3_campaign_runner_safety.py::"
        "test_run_campaign_preserves_original_and_state_verification_failures"
    ),
    "resource_cap": (
        "tests/test_v3_host_executor.py::"
        "test_attached_watchdog_stops_container_and_records_cap_exhaustion"
    ),
    "source_closure": (
        "tests/test_v3_execution_binding.py::"
        "test_source_probe_rejects_dirty_image_despite_matching_labels"
    ),
    "output_schema": (
        "tests/test_v3_campaign_finalizer.py::"
        "test_primary_single_writer_materializes_global_hashes_then_authorizes"
    ),
    "input_parity": (
        "tests/test_v3_input_binding.py::"
        "test_exact_five_input_binding_is_read_only_canonical_and_idempotent"
    ),
    "hsb_parity": (
        "tests/test_v3_hsb_binding.py::test_strict_h_greater_s_greater_b_fixture"
    ),
    "independent_replay": (
        "tests/test_v3_hsb_binding.py::"
        "test_full_synthetic_denominators_and_independent_replay"
    ),
    "authorization_replay_rejection": (
        "tests/test_v3_execution_binding.py::"
        "test_exclusive_lease_allows_exactly_one_writer"
    ),
    "pacs_stage_isolation": (
        "tests/test_v3_pacs_stage_isolation.py::"
        "test_only_selected_unique_primary_truth_is_decoded"
    ),
    "test_runtime_provenance": (
        "tests/test_v3_execution_binding.py::test_registered_test_runtime_version"
    ),
}
# Critical dynamic image-bound control-chain tests.  Readiness requires each
# node to appear as a passed test case in the registered test-container JUnit.
REQUIRED_EXECUTION_TEST_NODEIDS = (
    "tests/test_v3_execution_binding.py::test_dynamic_chain_binds_one_final_image_end_to_end",
    "tests/test_v3_execution_binding.py::test_template_declares_docker_create_campaign_runner_not_legacy_run",
    "tests/test_v3_execution_binding.py::test_authorized_plan_is_deterministic_render_of_template_and_authorization",
    "tests/test_v3_execution_binding.py::test_wrong_image_rejected_at_every_stage",
    "tests/test_v3_execution_binding.py::test_execution_configuration_drift_rejected",
    "tests/test_v3_execution_binding.py::test_resealed_upstream_artifacts_rejected",
    "tests/test_v3_execution_binding.py::test_role_swapped_labeled_control_root_rejected",
    "tests/test_v3_execution_binding.py::test_old_v1_r2_r3_bundles_rejected",
    "tests/test_v3_execution_binding.py::test_cross_run_artifact_mixing_rejected",
    "tests/test_v3_execution_binding.py::test_legacy_and_mixed_schema_rejected",
    "tests/test_v3_execution_binding.py::test_authorization_missing_expired_unissued_or_challenge_mismatch_rejected",
    "tests/test_v3_execution_binding.py::test_mounted_bundle_rejects_unsafe_or_incomplete_control_files",
    "tests/test_v3_execution_binding.py::test_issuer_derives_identity_and_ignores_caller_hashes",
    "tests/test_v3_execution_binding.py::test_issuer_rejects_artifact_replacement_before_authorization",
    "tests/test_v3_execution_binding.py::test_historical_r3_bundle_unchanged",
    "tests/test_v3_execution_binding.py::test_scientific_contract_invariants_unchanged",
    "tests/test_v3_host_executor.py::test_host_executes_only_rerendered_authorized_plan_bytes",
    "tests/test_v3_host_executor.py::test_host_rejects_control_replacement_before_docker_create",
    "tests/test_v3_host_executor.py::test_host_load_rejects_unsafe_or_incomplete_control_files",
    "tests/test_v3_scientific_runner.py::test_template_command_parses_under_campaign_runner_cli",
    "tests/test_v3_mount_receipt.py::test_mount_declaration_includes_read_only_authorization_mount",
)
REQUIRED_EXECUTION_CONTAINER_TEST_FILES = tuple(
    f"/testrepo/{path}" for path in REQUIRED_EXECUTION_TEST_FILES
)
REQUIRED_EXECUTION_PYTEST_ARGV = (
    "python",
    "-m",
    "pytest",
    "--rootdir=/testrepo",
    "-c",
    "/dev/null",
    "--import-mode=importlib",
    "-p",
    "no:cacheprovider",
    "-q",
    *REQUIRED_EXECUTION_CONTAINER_TEST_FILES,
    "--junitxml=/evidence/TEST_JUNIT.xml",
)
EXPECTED_HOST_CONTROL_DIR = Path(
    "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_local_v1/control/execution-local-v1"
)
EXPECTED_HOST_OUTPUT_DIR = Path(
    "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_local_v1/runs/scientific/"
    "FEDCORE-IJAR-PACS-AUTHOR-DETERMINISTIC-RERUN-HSB-local-v1"
)
SCIENTIFIC_COMMAND = (
    "-m",
    "fedcore.experiments.v3_campaign_runner",
    "run",
    "--authorization-dir",
    "/authorization",
    "--pacs-archive",
    "/inputs/PACS_mirror.zip",
    "--image-manifest",
    "/inputs/IMAGE_MANIFEST.csv",
    "--fold-manifest",
    "/inputs/FOLD_MANIFEST.csv",
    "--resnet18-weights",
    "/inputs/resnet18-f37072fd.pth",
    "--convnext-tiny-weights",
    "/inputs/convnext_tiny-983f1562.pth",
    "--output-dir",
    "/output",
    "--device",
    "cuda:0",
)

EXPECTED_HOST_LOG_DIR = Path(
    "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_local_v1/logs/execution-local-v1"
)

# The only compile-time exact-file anchor.  INPUT_BINDING.json names the five
# staged inputs and contains no image identity, so anchoring it in source does
# not create an image self-reference.
STATIC_INPUT_BINDING_FILE_SHA256 = (
    "dc330ceb35752ffece621bd1e2c1006b0edd2f3f340c08b80b574fd42defc4ad"
)

# Dynamic image-bound control chain (control schema 2).  The launch template,
# mount declaration, readiness, authorization, authorized plan, and runtime
# receipt are bound to one final image ID at run time and are never copied
# back into source, the Docker build context, or the image.
CONTROL_SCHEMA_VERSION = 2
CONTROL_BINDING_MODE = "DYNAMIC_IMAGE_BOUND_V1"
PLAN_KIND = "PREAUTH_EXECUTION_TEMPLATE"
RENDERER_VERSION = "FEDCORE_LOCAL_EXECUTION_RENDERER_V1"
CONTAINER_NAME_RULE = "fedcore-v3-{authorization_id}"
EXECUTION_TEMPLATE_DOMAIN = "FEDCORE_LOCAL_EXECUTION_TEMPLATE_V1"
INSTANTIATED_EXECUTION_DOMAIN = "FEDCORE_LOCAL_INSTANTIATED_EXECUTION_V1"
PRE_READINESS_CONTROL_ROOT_DOMAIN = "FEDCORE_LOCAL_PRE_READINESS_CONTROL_ROOT_V1"
LAUNCH_TEMPLATE_STATUS = "PASS_PREAUTH_EXECUTION_TEMPLATE_HOLD_READINESS"
MOUNT_DECLARATION_STATUS = "PASS_PREEXECUTION_MOUNT_DECLARATION_NO_GPU"
MOUNT_DECLARATION_ROLE = "PREEXECUTION_MOUNT_DECLARATION_NOT_RUNTIME_PROOF"
CONTAINER_WORKDIR = "/workspace"
CONTAINER_USER = "1000:1000"
CONTAINER_TMPFS_OPTIONS = "rw,noexec,nosuid,size=1g"
CONTAINER_LIFECYCLE = ("docker create", "docker inspect", "docker start --attach")
MAX_CONTROL_ARTIFACT_BYTES = 16 * 1024 * 1024
TRUSTED_COMPUTING_BASE = (
    "host root, the Docker daemon, and the kernel are trusted; this contract does "
    "not defend against a malicious root user or Docker daemon"
)

PRE_READINESS_FILENAMES = (
    "HOST_LAUNCH_PLAN.json",
    "INPUT_BINDING.json",
    "MOUNT_RECEIPT.json",
    "SOURCE_MANIFEST.json",
    "SOURCE_PROBE_RECEIPT.json",
    "TEST_REPORT.json",
)
CONTROL_FILENAMES = (
    "HOST_LAUNCH_PLAN.json",
    "INPUT_BINDING.json",
    "MOUNT_RECEIPT.json",
    "SOURCE_MANIFEST.json",
    "SOURCE_PROBE_RECEIPT.json",
    "TEST_REPORT.json",
    "EXECUTION_READINESS.json",
    "RUN_AUTHORIZATION.json",
    "AUTHORIZED_CAMPAIGN_PLAN.json",
    "RUNTIME_INSPECT.json",
    "RUNTIME_INSPECT_RECEIPT.json",
)


@dataclass(frozen=True)
class VerifiedExecutionAuthorization:
    """Hash-bound authority returned only after all mounted files validate."""

    readiness_sha256: str
    run_authorization_sha256: str
    authorized_plan_sha256: str
    runtime_receipt_sha256: str
    input_binding_sha256: str
    source_commit: str
    image_id: str
    authorization_id: str

    def checkpoint_fields(self) -> dict[str, str]:
        return {
            "readiness_sha256": self.readiness_sha256,
            "input_binding_sha256": self.input_binding_sha256,
            "run_authorization_sha256": self.run_authorization_sha256,
        }


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_hex(value: Any, length: int, label: str) -> str:
    if not _is_hex(value, length):
        raise ContractError(f"{label} must be lowercase hex{length}")
    return str(value)


def _require_bool(value: Any, expected: bool, label: str) -> None:
    if value is not expected:
        raise ContractError(f"{label} must be {expected!r}")


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ContractError(f"{label} key drift: missing={missing!r}, extra={extra!r}")


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body or f"{field}_definition" in body:
        raise ContractError(f"self-hash fields already present for {field}")
    result = dict(body)
    definition_field = f"{field}_definition"
    result[definition_field] = (
        f"canonical JSON of this object after {definition_field} is inserted "
        f"and before {field} is inserted"
    )
    result[field] = canonical_json_sha256(result)
    return result


def _validate_self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    digest = _require_hex(value.get(field), 64, field)
    definition_field = f"{field}_definition"
    expected_definition = (
        f"canonical JSON of this object after {definition_field} is inserted "
        f"and before {field} is inserted"
    )
    if value.get(definition_field) != expected_definition:
        raise ContractError(f"{label} self-hash definition drift")
    body = {
        key: item
        for key, item in value.items()
        if key != field
    }
    if canonical_json_sha256(body) != digest:
        raise ContractError(f"{label} self-hash mismatch")
    return digest


def _read_object(path: Path) -> dict[str, Any]:
    try:
        metadata = Path(path).lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"required control artifact is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"control artifact must be a regular nonsymlink file: {path}")
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=lambda pairs: _json_object_without_duplicate_keys(
                pairs, Path(path)
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid control artifact JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"control artifact must contain a JSON object: {path}")
    return value


@dataclass(frozen=True)
class ControlArtifact:
    """One control file read exactly once: parsed value, byte hash, identity."""

    path: Path
    value: Any
    sha256: str
    payload: bytes
    identity: tuple[int, int, int, int]


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def read_control_artifact(
    path: Path,
    *,
    require_object: bool = True,
    require_read_only: bool = False,
    max_bytes: int = MAX_CONTROL_ARTIFACT_BYTES,
) -> ControlArtifact:
    """Hash and parse one descriptor-backed byte buffer.

    Reading once through an ``O_NOFOLLOW`` descriptor prevents a path swap
    between a hash pass and a later JSON parse pass.  The descriptor's
    ``fstat`` must match the path's ``lstat`` so a replacement between the two
    calls is refused.  The returned identity lets callers re-check device,
    inode, size, and mtime immediately before a later side effect.
    """

    path = Path(path)
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"required control artifact is missing: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ContractError(f"control artifact must be a regular nonsymlink file: {path}")
    if require_read_only and stat.S_IMODE(before.st_mode) & 0o222:
        raise ContractError(f"control artifact must be read-only: {path.name}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractError(f"cannot safely open control artifact: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ContractError(f"control artifact changed type while opening: {path}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ContractError(f"control artifact was replaced while opening: {path}")
        if opened.st_size > max_bytes:
            raise ContractError(f"control artifact exceeds the size limit: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ContractError(f"control artifact exceeds the size limit: {path}")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) != opened.st_size:
        raise ContractError(f"control artifact changed size while reading: {path}")
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=lambda pairs: _json_object_without_duplicate_keys(
                pairs, path
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid control artifact JSON: {path}") from exc
    if require_object and not isinstance(value, dict):
        raise ContractError(f"control artifact must contain a JSON object: {path}")
    return ControlArtifact(
        path=path,
        value=value,
        sha256=hashlib.sha256(payload).hexdigest(),
        payload=payload,
        identity=_file_identity(opened),
    )


def recheck_control_artifact(artifact: ControlArtifact) -> None:
    """Fail closed unless the path still names the exact bytes that were validated."""

    try:
        metadata = artifact.path.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"control artifact disappeared after validation: {artifact.path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"control artifact changed type after validation: {artifact.path}")
    if _file_identity(metadata) != artifact.identity:
        raise ContractError(f"control artifact changed after validation: {artifact.path.name}")
    again = read_control_artifact(
        artifact.path, require_object=isinstance(artifact.value, dict)
    )
    if again.sha256 != artifact.sha256 or again.identity != artifact.identity:
        raise ContractError(f"control artifact changed after validation: {artifact.path.name}")


def _read_control_bytes_once(
    path: Path, *, require_object: bool = True
) -> tuple[Any, str]:
    artifact = read_control_artifact(path, require_object=require_object)
    return artifact.value, artifact.sha256


def _json_object_without_duplicate_keys(
    pairs: Sequence[tuple[str, Any]], path: Path
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ContractError(f"duplicate JSON key {key!r} in {path}")
        output[key] = value
    return output


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or "T" not in value or not value.endswith("Z"):
        raise ContractError(f"{label} must be a full RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise ContractError(f"{label} must use UTC")
    return parsed


def _authorization_challenge(readiness_sha256: str) -> str:
    return canonical_json_sha256(
        {
            "purpose": "AUTHORIZE_ONE_FRESH_30_CELL_FEDCORE_PACS_V3_CAMPAIGN",
            "readiness_sha256": readiness_sha256,
        }
    )


def _require_real_directory(path: Path, label: str, *, empty: bool = False) -> Path:
    value = Path(path)
    try:
        metadata = value.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"{label} is absent: {value}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ContractError(f"{label} must be a real nonsymlink directory: {value}")
    if not value.is_absolute():
        raise ContractError(f"{label} must be absolute: {value}")
    if empty and next(value.iterdir(), None) is not None:
        raise ContractError(f"{label} must be empty when readiness is issued")
    return value.resolve()


def _hash_regular_file(path: Path, label: str) -> str:
    value = Path(path)
    try:
        metadata = value.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"{label} is absent: {value}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"{label} must be a regular nonsymlink file: {value}")
    return sha256_file(value)


def _manifest_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a nonempty repository-relative path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or value != parsed.as_posix() or ".." in parsed.parts:
        raise ContractError(f"{label} is not a canonical repository-relative path")
    return value


def _scientific_runtime_paths(root: Path) -> tuple[str, ...]:
    """Return the exact source inventory copied into the scientific image.

    Interpreter caches and operating-system metadata are excluded because they
    are not scientific source and may be created after image construction.
    Every other regular file under the two Docker COPY roots is part of the
    closure, including helper modules and sealed protocol artifacts.
    """

    repository = Path(root)
    paths: set[str] = set()
    for relative in SCIENTIFIC_RUNTIME_SINGLETONS:
        candidate = repository / relative
        _hash_regular_file(candidate, f"runtime source {relative}")
        paths.add(relative)
    for relative_root in SCIENTIFIC_RUNTIME_ROOTS:
        directory = repository / relative_root
        try:
            metadata = directory.lstat()
        except FileNotFoundError as exc:
            raise ContractError(f"runtime source root is absent: {relative_root}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ContractError(f"runtime source root is unsafe: {relative_root}")
        for candidate in directory.rglob("*"):
            relative = candidate.relative_to(repository)
            if "__pycache__" in relative.parts or candidate.name in {".DS_Store"}:
                continue
            if candidate.suffix in {".pyc", ".pyo"}:
                continue
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ContractError(f"runtime source contains a symlink: {relative.as_posix()}")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ContractError(
                    f"runtime source contains a nonregular entry: {relative.as_posix()}"
                )
            paths.add(relative.as_posix())
    missing = sorted(set(REQUIRED_RUNTIME_SOURCE_FILES) - paths)
    if missing:
        raise ContractError(f"runtime source inventory lacks required files: {missing!r}")
    return tuple(sorted(paths))


def _excluded_runtime_artifacts(root: Path) -> tuple[str, ...]:
    """Return build-only/cache artifacts that may never survive in the image.

    The source checkout may contain interpreter or test caches.  They are
    intentionally absent from the scientific manifest and the Docker build
    context through the Dockerfile-specific ignore file.  A stopped-image
    probe, however, must fail if any such unbound bytes remain under a COPY
    root rather than silently skipping them.
    """

    repository = Path(root)
    excluded: set[str] = set()
    excluded_directories = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
    for relative_root in SCIENTIFIC_RUNTIME_ROOTS:
        directory = repository / relative_root
        if not directory.exists():
            continue
        for candidate in directory.rglob("*"):
            relative = candidate.relative_to(repository)
            if (
                any(part in excluded_directories for part in relative.parts)
                or candidate.name == ".DS_Store"
                or candidate.suffix in {".pyc", ".pyo"}
            ):
                excluded.add(relative.as_posix())
    return tuple(sorted(excluded))


def _source_record(root: Path, relative: str, *, container_path: str | None) -> dict[str, Any]:
    relative = _manifest_path(relative, "source path")
    path = Path(root) / relative
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"manifest source is absent: {relative}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"manifest source must be a regular nonsymlink file: {relative}")
    return {
        "path": relative,
        "container_path": container_path,
        "sha256": sha256_file(path),
        "size_bytes": int(metadata.st_size),
    }


def _validate_test_runtime_wheelhouse(repository_root: Path) -> None:
    """Require one exact, flat, nonsymlink test-runtime wheel set.

    The Dockerfile names every wheel explicitly.  This independent directory
    check prevents an unbound extra, nested directory, or symlink from entering
    a future directory-level COPY or influencing an offline resolver.
    """

    repository = Path(repository_root).resolve()
    wheelhouse = _require_real_directory(
        repository / "docker/pytest-wheelhouse", "test-runtime wheelhouse"
    )
    observed: list[str] = []
    for candidate in wheelhouse.iterdir():
        metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ContractError(
                f"test-runtime wheelhouse contains an unsafe entry: {candidate.name}"
            )
        observed.append(candidate.relative_to(repository).as_posix())
    expected = tuple(sorted(TEST_RUNTIME_WHEEL_FILES))
    actual = tuple(sorted(observed))
    if actual != expected:
        raise ContractError(
            "test-runtime wheelhouse file-set drift: "
            f"missing={sorted(set(expected) - set(actual))!r}, "
            f"extra={sorted(set(actual) - set(expected))!r}"
        )


def build_scientific_source_manifest(
    repository_root: Path,
    *,
    source_commit: str,
    source_tree: str,
) -> dict[str, Any]:
    """Build the exact clean-source closure consumed by the scientific image."""

    source_commit = _require_hex(source_commit, 40, "source_commit")
    source_tree = _require_hex(source_tree, 40, "source_tree")
    repository = Path(repository_root).resolve()
    _validate_test_runtime_wheelhouse(repository)
    runtime = [
        _source_record(
            repository,
            relative,
            container_path=(CANONICAL_CONTAINER_WORKSPACE_DIR / relative).as_posix(),
        )
        for relative in _scientific_runtime_paths(repository)
    ]
    build = [
        _source_record(repository, relative, container_path=None)
        for relative in SCIENTIFIC_BUILD_FILES
    ]
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": SOURCE_MANIFEST_STATUS,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "workspace_root": str(CANONICAL_CONTAINER_WORKSPACE_DIR),
        "runtime_files": runtime,
        "build_files": build,
        "runtime_inventory_sha256": canonical_json_sha256(runtime),
        "build_inventory_sha256": canonical_json_sha256(build),
    }


def _validate_source_records(
    rows: Any,
    *,
    label: str,
    expected_container: bool,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"{label} must be a nonempty list")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ContractError(f"{label}[{index}] must be an object")
        _require_exact_keys(
            row, {"path", "container_path", "sha256", "size_bytes"}, f"{label}[{index}]"
        )
        path = _manifest_path(row.get("path"), f"{label}[{index}].path")
        if path in seen:
            raise ContractError(f"duplicate source-manifest path: {path}")
        seen.add(path)
        digest = _require_hex(row.get("sha256"), 64, f"{label}[{index}].sha256")
        size = row.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ContractError(f"{label}[{index}].size_bytes must be a nonnegative integer")
        container_path = row.get("container_path")
        if expected_container:
            expected = (CANONICAL_CONTAINER_WORKSPACE_DIR / path).as_posix()
            if container_path != expected:
                raise ContractError(f"runtime container path drift for {path}")
        elif container_path is not None:
            raise ContractError(f"build-only source unexpectedly has a container path: {path}")
        validated.append(
            {
                "path": path,
                "container_path": container_path,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    if [row["path"] for row in validated] != sorted(seen):
        raise ContractError(f"{label} must be ordered by canonical path")
    return tuple(validated)


def validate_scientific_source_manifest(
    value: Mapping[str, Any],
    *,
    expected_source_commit: str | None = None,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    expected_keys = {
        "schema_version", "protocol_id", "status", "source_commit", "source_tree",
        "workspace_root", "runtime_files", "build_files", "runtime_inventory_sha256",
        "build_inventory_sha256",
    }
    _require_exact_keys(value, expected_keys, "scientific source manifest")
    if value.get("schema_version") != 1:
        raise ContractError("scientific source-manifest schema version mismatch")
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != SOURCE_MANIFEST_STATUS:
        raise ContractError("scientific source-manifest status/protocol mismatch")
    source_commit = _require_hex(value.get("source_commit"), 40, "source_commit")
    _require_hex(value.get("source_tree"), 40, "source_tree")
    if expected_source_commit is not None and source_commit != expected_source_commit:
        raise ContractError("scientific source-manifest commit mismatch")
    if value.get("workspace_root") != str(CANONICAL_CONTAINER_WORKSPACE_DIR):
        raise ContractError("scientific source-manifest workspace root drift")
    runtime = _validate_source_records(
        value.get("runtime_files"), label="runtime_files", expected_container=True
    )
    build = _validate_source_records(
        value.get("build_files"), label="build_files", expected_container=False
    )
    runtime_paths = {row["path"] for row in runtime}
    if not set(REQUIRED_RUNTIME_SOURCE_FILES).issubset(runtime_paths):
        raise ContractError("scientific source manifest omits a required runtime source")
    if {row["path"] for row in build} != set(SCIENTIFIC_BUILD_FILES):
        raise ContractError("scientific source manifest build-file inventory drift")
    if value.get("runtime_inventory_sha256") != canonical_json_sha256(list(runtime)):
        raise ContractError("scientific runtime inventory hash mismatch")
    if value.get("build_inventory_sha256") != canonical_json_sha256(list(build)):
        raise ContractError("scientific build inventory hash mismatch")
    return runtime, build


def _validate_manifest_files(
    value: Mapping[str, Any],
    root: Path,
    *,
    include_runtime: bool,
    include_build: bool,
) -> None:
    runtime, build = validate_scientific_source_manifest(value)
    expected_rows: tuple[dict[str, Any], ...] = ()
    if include_runtime:
        expected_rows += runtime
    if include_build:
        _validate_test_runtime_wheelhouse(Path(root))
        expected_rows += build
    for row in expected_rows:
        path = Path(root) / str(row["path"])
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise ContractError(f"bound scientific source is absent: {row['path']}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ContractError(f"bound scientific source is unsafe: {row['path']}")
        if metadata.st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise ContractError(f"bound scientific source bytes differ: {row['path']}")
    if include_runtime:
        observed = set(_scientific_runtime_paths(Path(root)))
        expected = {row["path"] for row in runtime}
        if observed != expected:
            raise ContractError(
                f"scientific runtime source-set drift: missing={sorted(expected - observed)!r}, "
                f"extra={sorted(observed - expected)!r}"
            )


def _git_bytes(repository: Path, arguments: Sequence[str], label: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError(f"cannot inspect Git object for {label}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ContractError(f"Git object inspection failed for {label}: {detail}")
    return result.stdout


def _git_runtime_paths(repository: Path, source_commit: str) -> tuple[str, ...]:
    payload = _git_bytes(
        repository,
        [
            "ls-tree",
            "-r",
            "-z",
            source_commit,
            "--",
            *SCIENTIFIC_RUNTIME_SINGLETONS,
            *SCIENTIFIC_RUNTIME_ROOTS,
        ],
        "scientific runtime tree",
    )
    paths: set[str] = set()
    for encoded in payload.split(b"\0"):
        if not encoded:
            continue
        try:
            metadata, encoded_path = encoded.split(b"\t", 1)
            mode, object_type, _object_id = metadata.decode("ascii").split(" ", 2)
            path = encoded_path.decode("utf-8")
        except (ValueError, UnicodeError) as exc:
            raise ContractError("Git runtime-tree record is malformed") from exc
        parsed = PurePosixPath(path)
        if "__pycache__" in parsed.parts or parsed.name == ".DS_Store":
            continue
        if parsed.suffix in {".pyc", ".pyo"}:
            continue
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise ContractError(f"Git runtime source is not a regular blob: {path}")
        paths.add(path)
    return tuple(sorted(paths))


def _validate_manifest_against_git_commit(
    value: Mapping[str, Any], repository_root: Path
) -> None:
    """Independently compare every bound byte with the claimed commit object."""

    runtime, build = validate_scientific_source_manifest(value)
    repository = Path(repository_root).resolve()
    _validate_test_runtime_wheelhouse(repository)
    source_commit = str(value["source_commit"])
    resolved_commit = _git_bytes(
        repository,
        ["rev-parse", "--verify", f"{source_commit}^{{commit}}"],
        "source commit",
    ).decode("ascii", errors="strict").strip()
    if resolved_commit != source_commit:
        raise ContractError("source_commit does not name the exact Git commit object")
    resolved_tree = _git_bytes(
        repository,
        ["rev-parse", "--verify", f"{source_commit}^{{tree}}"],
        "source tree",
    ).decode("ascii", errors="strict").strip()
    if resolved_tree != value.get("source_tree"):
        raise ContractError("scientific source-manifest Git tree mismatch")

    manifest_runtime_paths = tuple(row["path"] for row in runtime)
    committed_runtime_paths = _git_runtime_paths(repository, source_commit)
    if committed_runtime_paths != manifest_runtime_paths:
        raise ContractError(
            "scientific source-manifest differs from the exact Git runtime tree"
        )
    for row in (*runtime, *build):
        payload = _git_bytes(
            repository,
            ["cat-file", "blob", f"{source_commit}:{row['path']}"],
            f"committed source {row['path']}",
        )
        if len(payload) != row["size_bytes"] or hashlib.sha256(payload).hexdigest() != row[
            "sha256"
        ]:
            raise ContractError(
                f"scientific source-manifest bytes differ from Git commit: {row['path']}"
            )


def build_source_probe_receipt(
    *,
    source_manifest_path: Path,
    extracted_workspace_root: Path,
    image_id: str,
    container_inspect_path: Path,
) -> dict[str, Any]:
    """Verify a stopped image filesystem exported without PACS or GPU mounts.

    The host creates a stopped, network-disabled container from ``image_id``
    without mounts or a GPU request, copies ``/workspace`` to a temporary host
    directory, and passes that directory here.  No image process is started.
    """

    source_manifest = _read_object(source_manifest_path)
    runtime, _build = validate_scientific_source_manifest(source_manifest)
    source_manifest_sha256 = _hash_regular_file(
        source_manifest_path, "source-probe manifest"
    )
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise ContractError("source-probe image ID must be immutable")
    _require_hex(image_id.removeprefix("sha256:"), 64, "source_probe.image_id")

    inspect_value, inspect_file_sha256 = _read_control_bytes_once(
        container_inspect_path, require_object=False
    )
    if (
        not isinstance(inspect_value, list)
        or len(inspect_value) != 1
        or not isinstance(inspect_value[0], Mapping)
    ):
        raise ContractError("source-probe inspect must contain exactly one container")
    inspect = inspect_value[0]
    host = inspect.get("HostConfig")
    state = inspect.get("State")
    mounts = inspect.get("Mounts")
    if not isinstance(host, Mapping) or not isinstance(state, Mapping):
        raise ContractError("source-probe inspect lacks HostConfig/State")
    if inspect.get("Image") != image_id:
        raise ContractError("source-probe container image ID mismatch")
    container_id = _require_hex(inspect.get("Id"), 64, "source_probe.container_id")
    if state.get("Status") != "created" or state.get("Running") is not False:
        raise ContractError("source-probe container must remain stopped")
    if host.get("NetworkMode") != "none":
        raise ContractError("source-probe container network must be disabled")
    if host.get("DeviceRequests") not in (None, []):
        raise ContractError("source-probe container must not request a GPU")
    if host.get("Binds") not in (None, []):
        raise ContractError("source-probe container must not have host binds")
    if mounts not in (None, []):
        raise ContractError("source-probe container must not have any mounts")

    workspace = _require_real_directory(
        extracted_workspace_root, "extracted image workspace"
    )
    excluded_artifacts = _excluded_runtime_artifacts(workspace)
    if excluded_artifacts:
        raise ContractError(
            "source-probe image contains excluded unbound runtime artifacts: "
            f"{list(excluded_artifacts)!r}"
        )
    _validate_manifest_files(
        source_manifest,
        workspace,
        include_runtime=True,
        include_build=False,
    )
    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": SOURCE_PROBE_STATUS,
        "probe_method": "stopped_container_docker_cp_host_rehash",
        "image_id": image_id,
        "source_commit": source_manifest["source_commit"],
        "source_tree": source_manifest["source_tree"],
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_inventory_sha256": source_manifest["runtime_inventory_sha256"],
        "runtime_file_count": len(runtime),
        "workspace_root": str(CANONICAL_CONTAINER_WORKSPACE_DIR),
        "container_id": container_id,
        "container_inspect_sha256": inspect_file_sha256,
        "exact_runtime_source_set": True,
        "container_started": False,
        "network_none": True,
        "mount_count": 0,
        "PACS_mounted": False,
        "PACS_opened": False,
        "GPU_requested": False,
        "GPU_used": False,
    }
    return _self_hashed(body, "source_probe_sha256")


def validate_source_probe_receipt(
    value: Mapping[str, Any],
    *,
    source_manifest: Mapping[str, Any],
    expected_source_manifest_sha256: str,
    expected_image_id: str,
    expected_source_commit: str,
) -> str:
    digest = _validate_self_hash(value, "source_probe_sha256", "source-probe receipt")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "probe_method",
        "image_id",
        "source_commit",
        "source_tree",
        "source_manifest_sha256",
        "runtime_inventory_sha256",
        "runtime_file_count",
        "workspace_root",
        "container_id",
        "container_inspect_sha256",
        "exact_runtime_source_set",
        "container_started",
        "network_none",
        "mount_count",
        "PACS_mounted",
        "PACS_opened",
        "GPU_requested",
        "GPU_used",
        "source_probe_sha256",
        "source_probe_sha256_definition",
    }
    _require_exact_keys(value, expected_keys, "source-probe receipt")
    runtime, _build = validate_scientific_source_manifest(
        source_manifest, expected_source_commit=expected_source_commit
    )
    if value.get("schema_version") != 1:
        raise ContractError("source-probe schema version mismatch")
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != SOURCE_PROBE_STATUS:
        raise ContractError("source-probe status/protocol mismatch")
    if value.get("probe_method") != "stopped_container_docker_cp_host_rehash":
        raise ContractError("source-probe method mismatch")
    if value.get("image_id") != expected_image_id:
        raise ContractError("source-probe image ID mismatch")
    if value.get("source_commit") != expected_source_commit:
        raise ContractError("source-probe source commit mismatch")
    if value.get("source_tree") != source_manifest.get("source_tree"):
        raise ContractError("source-probe source tree mismatch")
    if value.get("source_manifest_sha256") != expected_source_manifest_sha256:
        raise ContractError("source-probe manifest hash mismatch")
    if value.get("runtime_inventory_sha256") != source_manifest.get(
        "runtime_inventory_sha256"
    ):
        raise ContractError("source-probe runtime inventory mismatch")
    if value.get("runtime_file_count") != len(runtime):
        raise ContractError("source-probe runtime file count mismatch")
    if value.get("workspace_root") != str(CANONICAL_CONTAINER_WORKSPACE_DIR):
        raise ContractError("source-probe workspace root mismatch")
    _require_hex(value.get("container_id"), 64, "source_probe.container_id")
    _require_hex(
        value.get("container_inspect_sha256"),
        64,
        "source_probe.container_inspect_sha256",
    )
    if value.get("mount_count") != 0:
        raise ContractError("source-probe mount count must be zero")
    for field in ("exact_runtime_source_set", "network_none"):
        _require_bool(value.get(field), True, f"source_probe.{field}")
    for field in (
        "container_started",
        "PACS_mounted",
        "PACS_opened",
        "GPU_requested",
        "GPU_used",
    ):
        _require_bool(value.get(field), False, f"source_probe.{field}")
    return digest


def validate_static_input_binding(value: Mapping[str, Any]) -> None:
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != "PASS_FIVE_INPUT_BINDING":
        raise ContractError("static input-binding status/protocol mismatch")
    _require_bool(value.get("exact_directory_contents"), True, "exact_directory_contents")
    _require_bool(value.get("all_regular_nonsymlink"), True, "all_regular_nonsymlink")
    _require_bool(value.get("all_read_only"), True, "all_read_only")
    if value.get("isolated_input_directory") != str(ISOLATED_INPUT_DIRECTORY):
        raise ContractError("static input-binding isolated input directory drift")
    rows = value.get("files")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ContractError("static input binding must contain exactly five records")
    for spec, row in zip(REGISTERED_SCIENTIFIC_INPUTS, rows, strict=True):
        expected = {
            "name": spec.name,
            "source_path": spec.source_path,
            "container_path": spec.container_path,
            "staged_path": str(ISOLATED_INPUT_DIRECTORY / spec.name),
            "sha256": spec.sha256,
            "size_bytes": spec.size_bytes,
            "read_only_mount_required": True,
            "regular_file": True,
            "symlink": False,
        }
        if not isinstance(row, Mapping) or any(row.get(k) != v for k, v in expected.items()):
            raise ContractError(f"static input-binding registry drift for {spec.name}")


# --------------------------------------------------------------------------
# control schema 2: pre-authorization execution template and its renderer
# --------------------------------------------------------------------------

LAUNCH_PLAN_KEYS = frozenset(
    {
        "schema_version", "control_schema_version", "control_binding_mode", "plan_kind",
        "renderer_version", "protocol_id", "status", "execution_allowed",
        "execution_readiness_required", "run_authorization_required",
        "prebinding_status", "prebinding_gate_path", "prebinding_gate_sha256",
        "input_binding_path", "input_binding_sha256", "input_inventory", "image",
        "image_inspect_sha256", "source_commit", "runtime", "output_directory",
        "output_empty_at_plan", "cell_order", "cell_order_sha256",
        "initial_cell_state", "stage_order", "execution_template",
        "execution_template_sha256", "PACS_archive_hashed_for_binding",
        "PACS_records_decoded", "PACS_labels_inspected",
        "PACS_scientific_content_opened", "torch_imported", "CUDA_touched",
        "launch_plan_sha256", "launch_plan_hash_definition",
    }
)
LAUNCH_PLAN_HASH_DEFINITION = (
    "canonical JSON of the plan before launch_plan_sha256 and "
    "launch_plan_hash_definition are inserted"
)
EXECUTION_TEMPLATE_KEYS = frozenset(
    {
        "domain", "renderer_version", "control_schema_version", "protocol_id",
        "lifecycle", "image_id", "source_commit", "entrypoint", "container_command",
        "module", "subcommand", "authorization_directory", "device", "environment",
        "working_dir", "user", "network_mode", "read_only_root_filesystem", "tmpfs",
        "cap_drop", "security_opt", "restart_policy", "cpu_limit_cores", "nano_cpus",
        "gpu_device_ids", "gpu_capabilities", "mounts", "paths",
        "cell_order_sha256", "container_name_rule",
    }
)
RUNTIME_EXECUTION_KEYS = (
    "image_id", "container_name", "entrypoint", "cmd", "environment", "working_dir",
    "user", "network_mode", "read_only_root_filesystem", "tmpfs", "cap_drop",
    "security_opt", "restart_policy", "nano_cpus", "gpu_device_ids",
    "gpu_capabilities", "mounts",
)
MOUNT_RECEIPT_KEYS = frozenset(
    {
        "schema_version", "control_schema_version", "control_binding_mode",
        "protocol_id", "status", "evidence_role", "launch_plan_sha256",
        "execution_template_sha256", "input_binding_sha256", "image_id",
        "container_id", "container_name", "network_mode", "read_only_root_filesystem",
        "user", "cap_drop", "security_opt", "device_requests", "mounts",
        "exact_five_read_only_inputs", "authorization_mount_read_only_per_inspect",
        "authorization_write_rejection_tested", "one_read_write_output",
        "tmpfs_only_at_tmp", "legacy_workspace_mounted", "runtime_proof",
        "PACS_records_decoded", "PACS_labels_inspected", "GPU_used",
        "training_started",
    }
)


def _require_image_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ContractError(f"{label} must be an immutable sha256 image ID")
    _require_hex(value.removeprefix("sha256:"), 64, label)
    return value


def _require_absolute_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or not PurePosixPath(value).is_absolute():
        raise ContractError(f"{label} must be a nonempty absolute path")
    if PurePosixPath(value).as_posix() != value or ".." in PurePosixPath(value).parts:
        raise ContractError(f"{label} is not a canonical absolute path")
    return value


def canonical_mounts(rows: Any) -> list[dict[str, Any]]:
    """Canonicalize mounts by destination and refuse duplicate destinations."""

    if not isinstance(rows, (list, tuple)):
        raise ContractError("mount list must be a list")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ContractError("mount row must be an object")
        kind = row.get("type")
        if kind == "bind":
            _require_exact_keys(row, {"type", "source", "destination", "read_only"}, "bind mount")
            if not isinstance(row.get("read_only"), bool):
                raise ContractError("bind mount read_only must be a boolean")
            normalized.append(
                {
                    "type": "bind",
                    "source": _require_absolute_path(row.get("source"), "bind mount source"),
                    "destination": _require_absolute_path(
                        row.get("destination"), "bind mount destination"
                    ),
                    "read_only": row["read_only"],
                }
            )
        elif kind == "tmpfs":
            _require_exact_keys(
                row, {"type", "source", "destination", "read_only", "options"}, "tmpfs mount"
            )
            if (
                row.get("source") != ""
                or row.get("destination") != "/tmp"
                or row.get("read_only") is not False
                or row.get("options") != CONTAINER_TMPFS_OPTIONS
            ):
                raise ContractError("tmpfs mount drift")
            normalized.append(dict(row))
        else:
            raise ContractError(f"unsupported mount type {kind!r}")
    destinations = [row["destination"] for row in normalized]
    if len(destinations) != len(set(destinations)):
        raise ContractError("duplicate mount destination")
    return sorted(normalized, key=lambda row: row["destination"])


def _execution_mounts(
    inputs: Sequence[Mapping[str, Any]], *, control_dir: Path, output_dir: Path
) -> list[dict[str, Any]]:
    if len(inputs) != 5:
        raise ContractError("execution template requires exactly five input mounts")
    rows: list[dict[str, Any]] = []
    for row in inputs:
        destination = str(row["destination"])
        if PurePosixPath(destination).parent != PurePosixPath("/inputs"):
            raise ContractError(f"input mount destination drift: {destination}")
        rows.append(
            {
                "type": "bind",
                "source": str(row["source"]),
                "destination": destination,
                "read_only": True,
            }
        )
    rows.append(
        {
            "type": "bind",
            "source": str(control_dir),
            "destination": str(CONTAINER_AUTHORIZATION_DIR),
            "read_only": True,
        }
    )
    rows.append(
        {
            "type": "bind",
            "source": str(output_dir),
            "destination": str(CONTAINER_OUTPUT_DIR),
            "read_only": False,
        }
    )
    rows.append(
        {
            "type": "tmpfs",
            "source": "",
            "destination": "/tmp",
            "read_only": False,
            "options": CONTAINER_TMPFS_OPTIONS,
        }
    )
    return canonical_mounts(rows)


def _command_option(command: Sequence[str], option: str) -> str:
    if list(command).count(option) != 1:
        raise ContractError(f"container command must contain {option} exactly once")
    index = list(command).index(option)
    if index + 1 >= len(command):
        raise ContractError(f"container command lacks a value for {option}")
    return str(command[index + 1])


def build_execution_template(
    *,
    image_id: str,
    source_commit: str,
    environment: Sequence[str],
    inputs: Sequence[Mapping[str, Any]],
    cell_order_sha256: str,
    control_dir: Path | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the pre-authorization semantic execution object.

    It contains everything the executor will run except the authorization ID
    and the container name derived from it.
    """

    image_id = _require_image_id(image_id, "template image_id")
    source_commit = _require_hex(source_commit, 40, "template source_commit")
    if not isinstance(environment, (list, tuple)) or not all(
        isinstance(item, str) for item in environment
    ):
        raise ContractError("template environment must be a list of strings")
    commits = [item for item in environment if item.startswith("FEDCORE_SOURCE_COMMIT=")]
    if commits != [f"FEDCORE_SOURCE_COMMIT={source_commit}"]:
        raise ContractError("template environment source revision mismatch")
    _require_hex(cell_order_sha256, 64, "template cell_order_sha256")
    control = Path(EXPECTED_HOST_CONTROL_DIR if control_dir is None else control_dir)
    staged = Path(ISOLATED_INPUT_DIRECTORY if input_dir is None else input_dir)
    output = Path(EXPECTED_HOST_OUTPUT_DIR if output_dir is None else output_dir)
    logs = Path(EXPECTED_HOST_LOG_DIR if log_dir is None else log_dir)
    command = list(SCIENTIFIC_COMMAND)
    return {
        "domain": EXECUTION_TEMPLATE_DOMAIN,
        "renderer_version": RENDERER_VERSION,
        "control_schema_version": CONTROL_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "lifecycle": list(CONTAINER_LIFECYCLE),
        "image_id": image_id,
        "source_commit": source_commit,
        "entrypoint": ["python"],
        "container_command": command,
        "module": command[1],
        "subcommand": command[2],
        "authorization_directory": _command_option(command, "--authorization-dir"),
        "device": _command_option(command, "--device"),
        "environment": list(environment),
        "working_dir": CONTAINER_WORKDIR,
        "user": CONTAINER_USER,
        "network_mode": "none",
        "read_only_root_filesystem": True,
        "tmpfs": {"/tmp": CONTAINER_TMPFS_OPTIONS},
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
        "restart_policy": "no",
        "cpu_limit_cores": CPU_LIMIT_CORES,
        "nano_cpus": DOCKER_NANO_CPUS,
        "gpu_device_ids": [EXPECTED_GPU_UUID],
        "gpu_capabilities": [["gpu"]],
        "mounts": _execution_mounts(inputs, control_dir=control, output_dir=output),
        "paths": {
            "control": str(control),
            "input": str(staged),
            "output": str(output),
            "log": str(logs),
        },
        "cell_order_sha256": cell_order_sha256,
        "container_name_rule": CONTAINER_NAME_RULE,
    }


def _template_shape(template: Any) -> Mapping[str, Any]:
    if not isinstance(template, Mapping):
        raise ContractError("execution template must be an object")
    _require_exact_keys(template, set(EXECUTION_TEMPLATE_KEYS), "execution template")
    if template.get("domain") != EXECUTION_TEMPLATE_DOMAIN:
        raise ContractError("execution template domain drift")
    if template.get("renderer_version") != RENDERER_VERSION:
        raise ContractError("execution template renderer version drift")
    if template.get("control_schema_version") != CONTROL_SCHEMA_VERSION:
        raise ContractError("execution template control schema drift")
    if template.get("container_name_rule") != CONTAINER_NAME_RULE:
        raise ContractError("execution template container-name rule drift")
    if template.get("lifecycle") != list(CONTAINER_LIFECYCLE):
        raise ContractError("execution template lifecycle drift")
    return template


def validate_execution_template(
    template: Any,
    *,
    expected_image_id: str,
    expected_source_commit: str,
    inputs: Sequence[Mapping[str, Any]],
    expected_environment: Sequence[str] | None = None,
    expected_cell_order_sha256: str | None = None,
) -> str:
    """Rebuild the template from trusted inputs and require exact equality."""

    template = _template_shape(template)
    environment = template.get("environment") if expected_environment is None else expected_environment
    cell_order = (
        template.get("cell_order_sha256")
        if expected_cell_order_sha256 is None
        else expected_cell_order_sha256
    )
    rebuilt = build_execution_template(
        image_id=expected_image_id,
        source_commit=expected_source_commit,
        environment=environment,
        inputs=inputs,
        cell_order_sha256=cell_order,
    )
    if dict(template) != rebuilt:
        drift = sorted(key for key in rebuilt if template.get(key) != rebuilt[key])
        raise ContractError(f"execution template drift: {drift!r}")
    return canonical_json_sha256(rebuilt)


def execution_template_sha256(template: Mapping[str, Any]) -> str:
    return canonical_json_sha256(_template_shape(template))


def render_authorized_execution(
    template: Mapping[str, Any], authorization_id: str
) -> dict[str, Any]:
    """Render the exact Docker create plan for one authorization ID."""

    template = _template_shape(template)
    authorization_id = _require_hex(authorization_id, 64, "authorization_id")
    if template.get("container_command") != list(SCIENTIFIC_COMMAND):
        raise ContractError("execution template command drift")
    name = CONTAINER_NAME_RULE.format(authorization_id=authorization_id)
    gpu_ids = template["gpu_device_ids"]
    if gpu_ids != [EXPECTED_GPU_UUID]:
        raise ContractError("execution template GPU device drift")
    argv = [
        "docker",
        "create",
        "--name",
        name,
        "--network",
        str(template["network_mode"]),
        "--read-only",
        "--tmpfs",
        f"/tmp:{template['tmpfs']['/tmp']}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--restart",
        str(template["restart_policy"]),
        "--user",
        str(template["user"]),
        "--workdir",
        str(template["working_dir"]),
        "--cpus",
        str(template["cpu_limit_cores"]),
        "--gpus",
        f"device={gpu_ids[0]}",
    ]
    for mount in canonical_mounts(template["mounts"]):
        if mount["type"] == "tmpfs":
            continue
        argv.extend(
            (
                "--mount",
                f"type=bind,src={mount['source']},dst={mount['destination']}"
                + (",readonly" if mount["read_only"] else ""),
            )
        )
    argv.append(str(template["image_id"]))
    argv.extend(str(item) for item in template["container_command"])
    runtime = {
        "image_id": template["image_id"],
        "container_name": name,
        "entrypoint": list(template["entrypoint"]),
        "cmd": list(template["container_command"]),
        "environment": list(template["environment"]),
        "working_dir": template["working_dir"],
        "user": template["user"],
        "network_mode": template["network_mode"],
        "read_only_root_filesystem": template["read_only_root_filesystem"],
        "tmpfs": dict(template["tmpfs"]),
        "cap_drop": list(template["cap_drop"]),
        "security_opt": list(template["security_opt"]),
        "restart_policy": template["restart_policy"],
        "nano_cpus": template["nano_cpus"],
        "gpu_device_ids": list(gpu_ids),
        "gpu_capabilities": [list(item) for item in template["gpu_capabilities"]],
        "mounts": canonical_mounts(template["mounts"]),
    }
    return {
        "domain": INSTANTIATED_EXECUTION_DOMAIN,
        "renderer_version": RENDERER_VERSION,
        "execution_template_sha256": canonical_json_sha256(template),
        "authorization_id": authorization_id,
        "container_name": name,
        "docker_create_argv": argv,
        "docker_start_argv_template": ["docker", "start", "--attach", "__CONTAINER_ID__"],
        "runtime": runtime,
    }


def instantiated_execution_configuration_sha256(instantiated: Mapping[str, Any]) -> str:
    return canonical_json_sha256(instantiated)


def _normalize_inspect_mounts(inspect: Mapping[str, Any]) -> list[dict[str, Any]]:
    mounts = inspect.get("Mounts")
    if not isinstance(mounts, list):
        raise ContractError("docker inspect mount list is absent")
    rows: list[dict[str, Any]] = []
    for row in mounts:
        if not isinstance(row, Mapping) or row.get("Type") != "bind":
            raise ContractError("docker inspect has an unregistered non-bind mount")
        if str(row.get("Propagation", "rprivate")) not in ("", "rprivate"):
            raise ContractError("bind mount propagation must be rprivate")
        if not isinstance(row.get("RW"), bool):
            raise ContractError("docker inspect mount RW flag must be a boolean")
        rows.append(
            {
                "type": "bind",
                "source": str(row.get("Source", "")),
                "destination": str(row.get("Destination", "")),
                "read_only": not row["RW"],
            }
        )
    host = inspect.get("HostConfig")
    tmpfs = host.get("Tmpfs") if isinstance(host, Mapping) else None
    if not isinstance(tmpfs, Mapping) or set(tmpfs) != {"/tmp"}:
        raise ContractError("docker inspect requires exactly one /tmp tmpfs")
    options = set(str(tmpfs["/tmp"]).split(","))
    size = {item for item in options if item.startswith("size=")}
    if options - size != {"rw", "noexec", "nosuid"} or size not in (
        {"size=1g"},
        {"size=1073741824"},
    ):
        raise ContractError("/tmp tmpfs option drift")
    rows.append(
        {
            "type": "tmpfs",
            "source": "",
            "destination": "/tmp",
            "read_only": False,
            "options": CONTAINER_TMPFS_OPTIONS,
        }
    )
    return canonical_mounts(rows)


def _one_container(inspect_value: Any, label: str) -> Mapping[str, Any]:
    if (
        not isinstance(inspect_value, list)
        or len(inspect_value) != 1
        or not isinstance(inspect_value[0], Mapping)
    ):
        raise ContractError(f"{label} docker inspect must contain exactly one container")
    inspect = inspect_value[0]
    for field in ("HostConfig", "Config"):
        if not isinstance(inspect.get(field), Mapping):
            raise ContractError(f"{label} docker inspect lacks {field}")
    return inspect


def normalize_runtime_inspect_execution(inspect_value: Any) -> dict[str, Any]:
    """Project raw ``docker inspect`` onto the rendered runtime execution object."""

    inspect = _one_container(inspect_value, "scientific")
    host = inspect["HostConfig"]
    config = inspect["Config"]
    requests = host.get("DeviceRequests")
    if not isinstance(requests, list) or len(requests) != 1 or not isinstance(requests[0], Mapping):
        raise ContractError("scientific container must request exactly one GPU device")
    request = requests[0]
    if request.get("Count") not in (0, -1):
        raise ContractError("scientific container GPU request count drift")
    if request.get("Driver") not in (None, "", "nvidia"):
        raise ContractError("scientific container GPU driver request drift")
    if request.get("Options") not in (None, {}):
        raise ContractError("scientific container GPU request options drift")
    restart = host.get("RestartPolicy")
    if not isinstance(restart, Mapping) or restart.get("Name") not in ("", "no"):
        raise ContractError("scientific container restart policy drift")
    if int(restart.get("MaximumRetryCount", 0)) != 0:
        raise ContractError("scientific container retry count drift")
    security = []
    for item in host.get("SecurityOpt") or []:
        text = str(item)
        security.append("no-new-privileges" if text == "no-new-privileges:true" else text)
    mounts = _normalize_inspect_mounts(inspect)
    return {
        "image_id": str(inspect.get("Image", "")),
        "container_name": str(inspect.get("Name", "")).lstrip("/"),
        "entrypoint": config.get("Entrypoint"),
        "cmd": config.get("Cmd"),
        "environment": config.get("Env"),
        "working_dir": config.get("WorkingDir"),
        "user": config.get("User"),
        "network_mode": host.get("NetworkMode"),
        "read_only_root_filesystem": host.get("ReadonlyRootfs"),
        "tmpfs": {"/tmp": CONTAINER_TMPFS_OPTIONS},
        "cap_drop": sorted(str(item) for item in (host.get("CapDrop") or [])),
        "security_opt": security,
        "restart_policy": "no",
        "nano_cpus": host.get("NanoCpus"),
        "gpu_device_ids": request.get("DeviceIDs"),
        "gpu_capabilities": request.get("Capabilities"),
        "mounts": mounts,
    }


def _launch_plan_inputs(input_binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_static_input_binding(input_binding)
    return [
        {"source": row["staged_path"], "destination": row["container_path"]}
        for row in input_binding["files"]
    ]


def validate_launch_template_plan(
    plan: Any,
    *,
    input_binding: Mapping[str, Any],
    expected_image_id: str,
    expected_source_commit: str,
    expected_environment: Sequence[str] | None = None,
) -> tuple[str, str]:
    """Validate a schema-2 HOST_LAUNCH_PLAN.json; return (internal hash, template hash)."""

    if not isinstance(plan, Mapping):
        raise ContractError("launch plan must be an object")
    if plan.get("schema_version") != 2 or plan.get("control_schema_version") != CONTROL_SCHEMA_VERSION:
        raise ContractError("launch plan is not control schema 2")
    _require_exact_keys(plan, set(LAUNCH_PLAN_KEYS), "launch plan")
    if plan.get("control_binding_mode") != CONTROL_BINDING_MODE or plan.get("plan_kind") != PLAN_KIND:
        raise ContractError("launch plan binding mode/kind drift")
    if plan.get("renderer_version") != RENDERER_VERSION:
        raise ContractError("launch plan renderer version drift")
    if plan.get("protocol_id") != PROTOCOL_ID or plan.get("status") != LAUNCH_TEMPLATE_STATUS:
        raise ContractError("launch plan status/protocol mismatch")
    _require_bool(plan.get("execution_allowed"), False, "execution_allowed")
    _require_bool(plan.get("execution_readiness_required"), True, "execution_readiness_required")
    _require_bool(plan.get("run_authorization_required"), True, "run_authorization_required")
    if plan.get("launch_plan_hash_definition") != LAUNCH_PLAN_HASH_DEFINITION:
        raise ContractError("launch plan hash definition drift")
    internal = _require_hex(plan.get("launch_plan_sha256"), 64, "launch_plan_sha256")
    body = {
        key: value
        for key, value in plan.items()
        if key not in {"launch_plan_sha256", "launch_plan_hash_definition"}
    }
    if canonical_json_sha256(body) != internal:
        raise ContractError("launch plan self-hash mismatch")
    image = plan.get("image")
    if not isinstance(image, Mapping) or set(image) != {"tag", "id"}:
        raise ContractError("launch plan image record drift")
    if image.get("id") != expected_image_id:
        raise ContractError("launch plan image ID differs from the bound image")
    if plan.get("source_commit") != expected_source_commit:
        raise ContractError("launch plan source revision differs from the bound source")
    _require_hex(plan.get("image_inspect_sha256"), 64, "launch plan image_inspect_sha256")
    if plan.get("input_binding_sha256") != STATIC_INPUT_BINDING_FILE_SHA256:
        raise ContractError("launch plan input-binding hash differs from the static anchor")
    if plan.get("output_directory") != str(EXPECTED_HOST_OUTPUT_DIR):
        raise ContractError("launch plan output directory drift")
    _require_bool(plan.get("output_empty_at_plan"), True, "output_empty_at_plan")
    inputs = _launch_plan_inputs(input_binding)
    inventory = plan.get("input_inventory")
    if not isinstance(inventory, list) or len(inventory) != 5:
        raise ContractError("launch plan must contain exactly five inputs")
    for spec, row, binding_row in zip(
        REGISTERED_SCIENTIFIC_INPUTS, inventory, input_binding["files"], strict=True
    ):
        expected = {
            "name": spec.name,
            "source": binding_row["staged_path"],
            "provenance_source": spec.source_path,
            "destination": spec.container_path,
            "sha256": spec.sha256,
            "bytes": spec.size_bytes,
            "read_only": True,
            "symlink": False,
        }
        if not isinstance(row, Mapping) or any(row.get(k) != v for k, v in expected.items()):
            raise ContractError(f"launch plan input registry drift for {spec.name}")
    cell_order = plan.get("cell_order")
    if not isinstance(cell_order, list) or len(cell_order) != 30:
        raise ContractError("launch plan must bind exactly 30 cells")
    if canonical_json_sha256(cell_order) != plan.get("cell_order_sha256"):
        raise ContractError("launch plan cell-order hash mismatch")
    if plan.get("runtime") != {
        "gpu_index": 0,
        "gpu_name": "NVIDIA GeForce RTX 4070 Ti SUPER",
        "gpu_uuid": EXPECTED_GPU_UUID,
        "max_concurrent_scientific_processes": 1,
        "network": "none",
        "root_filesystem_read_only": True,
        "single_process": True,
        "user": CONTAINER_USER,
    }:
        raise ContractError("launch plan runtime registry drift")
    for field in (
        "PACS_records_decoded",
        "PACS_labels_inspected",
        "PACS_scientific_content_opened",
        "torch_imported",
        "CUDA_touched",
    ):
        _require_bool(plan.get(field), False, field)
    template_hash = validate_execution_template(
        plan.get("execution_template"),
        expected_image_id=expected_image_id,
        expected_source_commit=expected_source_commit,
        inputs=inputs,
        expected_environment=expected_environment,
        expected_cell_order_sha256=plan.get("cell_order_sha256"),
    )
    if plan.get("execution_template_sha256") != template_hash:
        raise ContractError("launch plan execution-template hash mismatch")
    return internal, template_hash


def validate_mount_declaration(
    value: Any,
    *,
    launch_plan: Mapping[str, Any],
    expected_image_id: str,
) -> None:
    """Validate MOUNT_RECEIPT.json as a pre-execution declaration only."""

    if not isinstance(value, Mapping):
        raise ContractError("mount declaration must be an object")
    if value.get("schema_version") != 2 or value.get("control_schema_version") != CONTROL_SCHEMA_VERSION:
        raise ContractError("mount declaration is not control schema 2")
    _require_exact_keys(value, set(MOUNT_RECEIPT_KEYS), "mount declaration")
    if value.get("control_binding_mode") != CONTROL_BINDING_MODE:
        raise ContractError("mount declaration binding mode drift")
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != MOUNT_DECLARATION_STATUS:
        raise ContractError("mount declaration status/protocol mismatch")
    if value.get("evidence_role") != MOUNT_DECLARATION_ROLE:
        raise ContractError("mount declaration evidence role drift")
    if value.get("launch_plan_sha256") != launch_plan.get("launch_plan_sha256"):
        raise ContractError("mount declaration launch-plan mismatch")
    if value.get("execution_template_sha256") != launch_plan.get("execution_template_sha256"):
        raise ContractError("mount declaration execution-template mismatch")
    if value.get("input_binding_sha256") != STATIC_INPUT_BINDING_FILE_SHA256:
        raise ContractError("mount declaration input-binding mismatch")
    if value.get("image_id") != expected_image_id:
        raise ContractError("mount declaration image ID differs from the bound image")
    _require_hex(value.get("container_id"), 64, "mount declaration container_id")
    if value.get("mounts") != launch_plan["execution_template"]["mounts"]:
        raise ContractError("mount declaration mount scope differs from the template")
    if (
        value.get("network_mode") != "none"
        or value.get("user") != CONTAINER_USER
        or value.get("cap_drop") != ["ALL"]
        or value.get("security_opt") != ["no-new-privileges"]
        or value.get("device_requests") is not None
    ):
        raise ContractError("mount declaration isolation drift")
    for field in (
        "read_only_root_filesystem",
        "exact_five_read_only_inputs",
        "authorization_mount_read_only_per_inspect",
        "one_read_write_output",
        "tmpfs_only_at_tmp",
    ):
        _require_bool(value.get(field), True, field)
    for field in (
        "authorization_write_rejection_tested",
        "legacy_workspace_mounted",
        "runtime_proof",
        "PACS_records_decoded",
        "PACS_labels_inspected",
        "GPU_used",
        "training_started",
    ):
        _require_bool(value.get(field), False, field)


def pre_readiness_control_root_sha256(
    *,
    source_commit: str,
    image_id: str,
    execution_template_sha256_value: str,
    file_sha256: Mapping[str, str],
) -> str:
    """Domain-separated canonical root over the six pre-readiness control files."""

    if not isinstance(file_sha256, Mapping):
        raise ContractError("control root file hashes must be an object")
    _require_exact_keys(file_sha256, set(PRE_READINESS_FILENAMES), "control root files")
    for name in PRE_READINESS_FILENAMES:
        _require_hex(file_sha256.get(name), 64, f"control root {name}")
    return canonical_json_sha256(
        {
            "domain": PRE_READINESS_CONTROL_ROOT_DOMAIN,
            "control_schema_version": CONTROL_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "source_commit": _require_hex(source_commit, 40, "control root source_commit"),
            "image_id": _require_image_id(image_id, "control root image_id"),
            "execution_template_sha256": _require_hex(
                execution_template_sha256_value, 64, "control root template hash"
            ),
            "files": {name: file_sha256[name] for name in PRE_READINESS_FILENAMES},
        }
    )


def validate_execution_test_report(
    value: Mapping[str, Any],
    *,
    expected_image_id: str,
    expected_source_commit: str,
    expected_source_manifest_sha256: str,
) -> None:
    """Reject a status-only report and require every PACS-free launch gate."""

    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "image_id",
        "source_commit",
        "source_manifest_sha256",
        "test_container_inspect_sha256",
        "test_container_id",
        "junit_xml_sha256",
        "pytest_stdout_sha256",
        "pytest_argv",
        "required_test_files",
        "test_file_sha256",
        "pytest_counts",
        "exit_code",
        "environment",
        "PACS_opened",
        "GPU_used",
        "test_gates",
        "gate_test_nodeids",
    }
    _require_exact_keys(value, expected_keys, "execution test report")
    if value.get("schema_version") != 1:
        raise ContractError("execution test-report schema version mismatch")
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != TEST_REPORT_STATUS:
        raise ContractError("execution test-report status/protocol mismatch")

    image_id = str(value.get("image_id", ""))
    if image_id != expected_image_id or not image_id.startswith("sha256:"):
        raise ContractError("execution test-report image ID mismatch")
    _require_hex(image_id.removeprefix("sha256:"), 64, "test_report.image_id")
    source_commit = _require_hex(
        value.get("source_commit"), 40, "test_report.source_commit"
    )
    if source_commit != expected_source_commit:
        raise ContractError("execution test-report source commit mismatch")
    manifest_sha256 = _require_hex(
        value.get("source_manifest_sha256"),
        64,
        "test_report.source_manifest_sha256",
    )
    if manifest_sha256 != expected_source_manifest_sha256:
        raise ContractError("execution test-report source-manifest mismatch")
    _require_hex(
        value.get("test_container_inspect_sha256"),
        64,
        "test_report.test_container_inspect_sha256",
    )
    _require_hex(value.get("test_container_id"), 64, "test_report.test_container_id")
    _require_hex(value.get("junit_xml_sha256"), 64, "test_report.junit_xml_sha256")
    _require_hex(
        value.get("pytest_stdout_sha256"), 64, "test_report.pytest_stdout_sha256"
    )
    if value.get("pytest_argv") != list(REQUIRED_EXECUTION_PYTEST_ARGV):
        raise ContractError("execution test-report pytest argv/order mismatch")

    if value.get("required_test_files") != list(REQUIRED_EXECUTION_TEST_FILES):
        raise ContractError("execution test-report required-file set/order mismatch")
    test_file_sha256 = value.get("test_file_sha256")
    if not isinstance(test_file_sha256, Mapping):
        raise ContractError("execution test-report file hashes are absent")
    _require_exact_keys(
        test_file_sha256,
        set(REQUIRED_EXECUTION_TEST_FILES),
        "execution test-report file hashes",
    )
    for path in REQUIRED_EXECUTION_TEST_FILES:
        _require_hex(
            test_file_sha256.get(path), 64, f"test_report.test_file_sha256[{path!r}]"
        )
    counts = value.get("pytest_counts")
    if not isinstance(counts, Mapping):
        raise ContractError("execution test-report pytest counts are absent")
    _require_exact_keys(
        counts,
        {"collected", "passed", "failed", "skipped"},
        "execution test-report pytest counts",
    )
    for field in ("collected", "passed", "failed", "skipped"):
        count = counts.get(field)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ContractError(f"execution test-report {field} count is invalid")
    if (
        counts["collected"] < len(REQUIRED_EXECUTION_TEST_FILES)
        or counts["passed"] != counts["collected"]
        or counts["failed"] != 0
        or counts["skipped"] != 0
    ):
        raise ContractError("execution test-report does not record a complete clean pass")
    if value.get("exit_code") != 0:
        raise ContractError("execution test-report exit code is not zero")
    if value.get("environment") != {"CUDA_VISIBLE_DEVICES": ""}:
        raise ContractError("execution test-report CUDA isolation mismatch")
    _require_bool(value.get("PACS_opened"), False, "test_report.PACS_opened")
    _require_bool(value.get("GPU_used"), False, "test_report.GPU_used")

    gates = value.get("test_gates")
    if not isinstance(gates, Mapping):
        raise ContractError("execution test-report gates are absent")
    _require_exact_keys(
        gates, set(REQUIRED_EXECUTION_TEST_GATES), "execution test-report gates"
    )
    for gate in REQUIRED_EXECUTION_TEST_GATES:
        _require_bool(gates.get(gate), True, f"test_report.test_gates.{gate}")
    if value.get("gate_test_nodeids") != REQUIRED_EXECUTION_GATE_NODEIDS:
        raise ContractError("execution test-report gate node-ID evidence mismatch")


def validate_test_container_inspect(
    value: Any,
    *,
    expected_image_id: str,
    expected_repository_root: Path,
    expected_evidence_root: Path,
) -> str:
    """Validate the stopped evidence-only container used for the test suite.

    This validator proves the environmental part of the PACS-free test claim:
    the exact immutable image ran with no network or GPU request, the checkout
    was mounted read-only at ``/testrepo``, and only ``/evidence`` was
    writable.  Image code remains at ``/workspace`` and is not shadowed by a
    host bind.  JUnit/stdout validation remains separate, but their exact host
    directory is passed here so that ``/evidence`` cannot point at a different
    execution's evidence directory.
    """

    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], Mapping)
    ):
        raise ContractError("test-container inspect must contain exactly one container")
    inspect = value[0]
    container_id = _require_hex(
        inspect.get("Id"), 64, "test_container_inspect.container_id"
    )
    if inspect.get("Image") != expected_image_id:
        raise ContractError("test-container image ID mismatch")
    state = inspect.get("State")
    config = inspect.get("Config")
    host = inspect.get("HostConfig")
    mounts = inspect.get("Mounts")
    if not all(isinstance(item, Mapping) for item in (state, config, host)):
        raise ContractError("test-container inspect lacks State/Config/HostConfig")
    if (
        state.get("Status") != "exited"
        or state.get("Running") is not False
        or state.get("ExitCode") != 0
    ):
        raise ContractError("test-container must be a cleanly exited container")
    if host.get("NetworkMode") != "none":
        raise ContractError("test-container network must be disabled")
    if host.get("DeviceRequests") not in (None, []):
        raise ContractError("test-container must not request a GPU")
    if host.get("Devices") not in (None, []):
        raise ContractError("test-container must not expose host devices")
    if host.get("ReadonlyRootfs") is not True:
        raise ContractError("test-container root filesystem must be read-only")
    if host.get("Privileged") is not False:
        raise ContractError("test-container must not be privileged")
    if host.get("CapDrop") != ["ALL"]:
        raise ContractError("test-container must drop all Linux capabilities")
    if host.get("SecurityOpt") != ["no-new-privileges"]:
        raise ContractError("test-container no-new-privileges setting mismatch")
    if host.get("Tmpfs") != {"/tmp": "rw,noexec,nosuid,size=1g"}:
        raise ContractError("test-container must have only the registered /tmp tmpfs")
    environment = config.get("Env")
    if (
        not isinstance(environment, list)
        or "CUDA_VISIBLE_DEVICES=" not in environment
        or f"PYTHONPATH=/workspace:{TEST_RUNTIME_ROOT}" not in environment
        or "PYTHONSAFEPATH=1" not in environment
    ):
        raise ContractError("test-container CUDA isolation is absent")
    if (
        config.get("Entrypoint") != ["python"]
        or config.get("WorkingDir") != "/tmp"
        or config.get("User") != "1000:1000"
    ):
        raise ContractError("test-container entrypoint/workdir/user mismatch")
    if config.get("Cmd") != list(REQUIRED_EXECUTION_PYTEST_ARGV[1:]):
        raise ContractError("test-container pytest command/file order mismatch")
    if not isinstance(mounts, list):
        raise ContractError("test-container mounts are absent")
    by_destination: dict[str, Mapping[str, Any]] = {}
    tmpfs_mounts: list[Mapping[str, Any]] = []
    for mount in mounts:
        if not isinstance(mount, Mapping):
            raise ContractError("test-container mount record is malformed")
        mount_type = mount.get("Type")
        if mount_type == "tmpfs":
            tmpfs_mounts.append(mount)
            continue
        if mount_type != "bind":
            raise ContractError("test-container has an unregistered mount type")
        destination = mount.get("Destination")
        if not isinstance(destination, str) or destination in by_destination:
            raise ContractError("test-container mount destination is invalid")
        by_destination[destination] = mount
    if set(by_destination) != {"/testrepo", "/evidence"}:
        raise ContractError("test-container has an unregistered or PACS/input mount")
    workspace_mount = by_destination["/testrepo"]
    evidence_mount = by_destination["/evidence"]
    if workspace_mount.get("Type") != "bind" or workspace_mount.get("RW") is not False:
        raise ContractError("test-container repository mount must be read-only")
    if workspace_mount.get("Source") != str(Path(expected_repository_root).resolve()):
        raise ContractError("test-container repository source differs from the exact checkout")
    if evidence_mount.get("Type") != "bind" or evidence_mount.get("RW") is not True:
        raise ContractError("test-container evidence mount must be the only writable bind")
    evidence_root = _require_real_directory(
        expected_evidence_root, "test-container evidence directory"
    )
    if evidence_mount.get("Source") != str(evidence_root):
        raise ContractError(
            "test-container evidence source differs from the exact raw-evidence directory"
        )
    if tmpfs_mounts and (
        len(tmpfs_mounts) != 1
        or tmpfs_mounts[0].get("Destination") != "/tmp"
        or tmpfs_mounts[0].get("RW") is not True
    ):
        raise ContractError("test-container tmpfs mount differs from the registered /tmp")
    return container_id


def validate_test_evidence_bundle_paths(
    *,
    test_container_inspect_path: Path,
    test_junit_path: Path,
    test_stdout_path: Path,
) -> Path:
    """Return the one exact host directory containing all raw test evidence.

    The container writes JUnit evidence to ``/evidence/TEST_JUNIT.xml``.  The
    stdout transcript and post-exit raw inspect must be placed beside it under
    the same directory mounted at ``/evidence``.  Canonical direct-child paths
    prevent bytes copied from another execution directory from being bound to
    an otherwise valid inspect record.
    """

    artifacts = (
        (
            Path(test_container_inspect_path),
            "TEST_CONTAINER_INSPECT.json",
            "test-container inspect",
        ),
        (Path(test_junit_path), "TEST_JUNIT.xml", "test JUnit XML"),
        (Path(test_stdout_path), "TEST_STDOUT.log", "test stdout transcript"),
    )
    evidence_root: Path | None = None
    for path, expected_name, label in artifacts:
        if not path.is_absolute():
            raise ContractError(f"{label} path must be absolute: {path}")
        _hash_regular_file(path, label)
        parent = _require_real_directory(path.parent, f"{label} directory")
        if path.name != expected_name or path.resolve() != parent / expected_name:
            raise ContractError(f"{label} does not use its exact evidence path")
        if evidence_root is None:
            evidence_root = parent
        elif parent != evidence_root:
            raise ContractError(
                "raw test evidence files do not share one exact evidence directory"
            )
    assert evidence_root is not None
    return evidence_root


def _validate_test_report_against_git_commit(
    value: Mapping[str, Any], repository_root: Path
) -> None:
    """Bind every required test file to both local and committed exact bytes."""

    repository = Path(repository_root).resolve()
    source_commit = str(value["source_commit"])
    hashes = value["test_file_sha256"]
    for path in REQUIRED_EXECUTION_TEST_FILES:
        committed = _git_bytes(
            repository,
            ["cat-file", "blob", f"{source_commit}:{path}"],
            f"committed execution test {path}",
        )
        expected = str(hashes[path])
        if hashlib.sha256(committed).hexdigest() != expected:
            raise ContractError(
                f"execution test-report bytes differ from Git commit: {path}"
            )
        if _hash_regular_file(repository / path, f"execution test {path}") != expected:
            raise ContractError(f"execution test working-tree bytes are dirty: {path}")


def validate_execution_test_report_against_git_commit(
    value: Mapping[str, Any], repository_root: Path
) -> None:
    """Public wrapper for exact test-source versus Git-object closure."""

    _validate_test_report_against_git_commit(value, repository_root)


def _read_regular_bytes(path: Path, label: str, *, max_bytes: int = MAX_CONTROL_ARTIFACT_BYTES) -> bytes:
    path = Path(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"{label} is absent: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"{label} must be a regular nonsymlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ContractError(f"{label} was replaced while opening: {path}")
        if opened.st_size > max_bytes:
            raise ContractError(f"{label} exceeds the size limit: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _junit_passed_nodeids(payload: bytes) -> set[str]:
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise ContractError("registered test JUnit XML is not well formed") from exc
    passed: set[str] = set()
    for case in root.iter("testcase"):
        if any(child.tag in ("failure", "error", "skipped") for child in case):
            continue
        classname = str(case.get("classname", ""))
        name = str(case.get("name", ""))
        passed.add(f"{classname.replace('.', '/')}.py::{name}")
    return passed


def validate_critical_test_nodes(test_junit_path: Path) -> None:
    """Require every registered control-chain node to have passed in the JUnit XML."""

    passed = _junit_passed_nodeids(_read_regular_bytes(test_junit_path, "test JUnit XML"))
    missing = [node for node in REQUIRED_EXECUTION_TEST_NODEIDS if node not in passed]
    if missing:
        raise ContractError(f"critical control-chain test nodes did not pass: {missing!r}")


READINESS_KEYS = frozenset(
    {
        "schema_version", "control_schema_version", "control_binding_mode",
        "renderer_version", "protocol_id", "status", "technical_execution_ready",
        "run_authorization_required", "run_authorization_issued", "training_authorized",
        "fresh_output_required", "resume_authorized",
        "pre_readiness_control_root_sha256", "launch_plan_file_sha256",
        "launch_plan_sha256", "execution_template_sha256", "input_binding_sha256",
        "mount_receipt_sha256", "source_commit", "image_id", "image_source_revision",
        "image_inspect_sha256", "source_manifest_sha256", "runtime_inventory_sha256",
        "build_inventory_sha256", "dockerfile_sha256", "requirements_lock_sha256",
        "orchestrator_sha256", "state_machine_sha256", "execution_binding_sha256",
        "finalizer_sha256", "posttrain_sha256", "test_report_sha256",
        "source_probe_receipt_sha256", "cell_order_sha256", "critical_test_nodeids",
        "planned_cells", "max_concurrent_scientific_processes", "total_gpu_hour_cap",
        "sealed_capacity_gpu_hours", "remaining_scientific_walltime_hours",
        "scientific_execution_timeout_seconds",
        "post_timeout_cleanup_command_bound_seconds",
        "scientific_cleanup_reserve_seconds", "scientific_cleanup_safety_margin_seconds",
        "cpu_limit_cores", "maximum_cpu_core_hours", "host_control_dir",
        "host_input_dir", "host_output_dir", "host_log_dir",
        "container_authorization_dir", "container_output_dir", "container_entry_module",
        "container_command", "authorization_method_required",
        "authorization_threat_model", "trusted_computing_base",
        "scientific_output_empty_when_readiness_issued", "PACS_opened",
        "PACS_inference_started", "PACS_training_started", "GPU_used",
        "readiness_sha256", "readiness_sha256_definition",
    }
)
READINESS_FILE_FIELDS = {
    "HOST_LAUNCH_PLAN.json": "launch_plan_file_sha256",
    "INPUT_BINDING.json": "input_binding_sha256",
    "MOUNT_RECEIPT.json": "mount_receipt_sha256",
    "SOURCE_MANIFEST.json": "source_manifest_sha256",
    "SOURCE_PROBE_RECEIPT.json": "source_probe_receipt_sha256",
    "TEST_REPORT.json": "test_report_sha256",
}
AUTHORIZATION_THREAT_MODEL = (
    "trusted-host procedural authorization; no cryptographic signer identity claim"
)


def _image_record(image_inspect_value: Any, image_id: str, source_commit: str) -> Mapping[str, Any]:
    if (
        not isinstance(image_inspect_value, list)
        or len(image_inspect_value) != 1
        or not isinstance(image_inspect_value[0], Mapping)
    ):
        raise ContractError("image inspect must contain exactly one image")
    image = image_inspect_value[0]
    config = image.get("Config")
    if not isinstance(config, Mapping) or image.get("Id") != image_id:
        raise ContractError("image inspect immutable ID mismatch")
    labels = config.get("Labels") or {}
    if not isinstance(labels, Mapping) or labels.get(
        "org.opencontainers.image.revision"
    ) != source_commit:
        raise ContractError("image inspect source-revision label mismatch")
    environment = config.get("Env")
    if not isinstance(environment, list) or f"FEDCORE_SOURCE_COMMIT={source_commit}" not in environment:
        raise ContractError("image inspect lacks the immutable source environment")
    if config.get("Entrypoint") != ["python"] or config.get("User") != CONTAINER_USER:
        raise ContractError("image inspect entrypoint/user mismatch")
    if config.get("WorkingDir") != CONTAINER_WORKDIR:
        raise ContractError("image inspect working directory mismatch")
    return image


def build_execution_readiness(
    *,
    static_launch_plan_path: Path,
    input_binding_path: Path,
    mount_receipt_path: Path,
    source_commit: str,
    image_id: str,
    image_inspect_path: Path,
    source_manifest_path: Path,
    source_probe_receipt_path: Path,
    orchestrator_path: Path,
    state_machine_path: Path,
    execution_binding_path: Path,
    finalizer_path: Path,
    posttrain_path: Path,
    test_report_path: Path,
    test_container_inspect_path: Path,
    test_junit_path: Path,
    test_stdout_path: Path,
    cell_order_path: Path,
    host_control_dir: Path,
    host_output_dir: Path,
) -> dict[str, Any]:
    """Create technical readiness without creating run authorization.

    Every control artifact is read once from its actual bytes; no caller-supplied
    artifact hash is accepted.
    """

    _validate_gpu_resource_cap_invariant()
    control = {
        "HOST_LAUNCH_PLAN.json": read_control_artifact(static_launch_plan_path, require_read_only=True),
        "INPUT_BINDING.json": read_control_artifact(input_binding_path, require_read_only=True),
        "MOUNT_RECEIPT.json": read_control_artifact(mount_receipt_path, require_read_only=True),
        "SOURCE_MANIFEST.json": read_control_artifact(source_manifest_path, require_read_only=True),
        "SOURCE_PROBE_RECEIPT.json": read_control_artifact(
            source_probe_receipt_path, require_read_only=True
        ),
        "TEST_REPORT.json": read_control_artifact(test_report_path, require_read_only=True),
    }
    image_artifact = read_control_artifact(image_inspect_path, require_object=False)
    plan = control["HOST_LAUNCH_PLAN.json"].value
    binding = control["INPUT_BINDING.json"].value
    mount = control["MOUNT_RECEIPT.json"].value
    if control["INPUT_BINDING.json"].sha256 != STATIC_INPUT_BINDING_FILE_SHA256:
        raise ContractError("local input-binding file hash mismatch")
    validate_static_input_binding(binding)
    source_commit = _require_hex(source_commit, 40, "source_commit")
    image_id = _require_image_id(image_id, "image_id")
    image = _image_record(image_artifact.value, image_id, source_commit)
    config = image["Config"]
    labels = config.get("Labels") or {}
    plan_internal, template_hash = validate_launch_template_plan(
        plan,
        input_binding=binding,
        expected_image_id=image_id,
        expected_source_commit=source_commit,
        expected_environment=list(config["Env"]),
    )
    if plan.get("image_inspect_sha256") != image_artifact.sha256:
        raise ContractError("launch plan was rendered from a different image inspect")
    validate_mount_declaration(mount, launch_plan=plan, expected_image_id=image_id)

    source_manifest = control["SOURCE_MANIFEST.json"].value
    runtime_sources, build_sources = validate_scientific_source_manifest(
        source_manifest, expected_source_commit=source_commit
    )
    del runtime_sources
    repository_root = Path(execution_binding_path).resolve().parents[2]
    _validate_manifest_files(
        source_manifest,
        repository_root,
        include_runtime=True,
        include_build=True,
    )
    _validate_manifest_against_git_commit(source_manifest, repository_root)
    source_manifest_file_sha256 = control["SOURCE_MANIFEST.json"].sha256
    expected_labels = {
        SOURCE_MANIFEST_LABEL: source_manifest_file_sha256,
        RUNTIME_INVENTORY_LABEL: source_manifest["runtime_inventory_sha256"],
        BUILD_INVENTORY_LABEL: source_manifest["build_inventory_sha256"],
        DOCKERFILE_LABEL: next(
            row["sha256"]
            for row in build_sources
            if row["path"] == "docker/Dockerfile.v3_hsb_scientific"
        ),
        REQUIREMENTS_LABEL: next(
            row["sha256"]
            for row in build_sources
            if row["path"] == "requirements.lock"
        ),
    }
    for label, expected in expected_labels.items():
        if labels.get(label) != expected:
            raise ContractError(f"image inspect scientific source label mismatch: {label}")
    test_report = control["TEST_REPORT.json"].value
    validate_execution_test_report(
        test_report,
        expected_image_id=image_id,
        expected_source_commit=source_commit,
        expected_source_manifest_sha256=source_manifest_file_sha256,
    )
    _validate_test_report_against_git_commit(test_report, repository_root)
    try:
        test_container_inspect = json.loads(
            Path(test_container_inspect_path).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("test-container inspect is not valid JSON") from exc
    test_evidence_root = validate_test_evidence_bundle_paths(
        test_container_inspect_path=test_container_inspect_path,
        test_junit_path=test_junit_path,
        test_stdout_path=test_stdout_path,
    )
    test_container_id = validate_test_container_inspect(
        test_container_inspect,
        expected_image_id=image_id,
        expected_repository_root=repository_root,
        expected_evidence_root=test_evidence_root,
    )
    if test_container_id != test_report["test_container_id"]:
        raise ContractError("test-container ID differs from the execution test report")
    if (
        _hash_regular_file(test_container_inspect_path, "test-container inspect")
        != test_report["test_container_inspect_sha256"]
    ):
        raise ContractError("raw test-container inspect differs from the test report")
    if (
        _hash_regular_file(test_junit_path, "test JUnit XML")
        != test_report["junit_xml_sha256"]
    ):
        raise ContractError("raw JUnit XML differs from the test report")
    if (
        _hash_regular_file(test_stdout_path, "test stdout transcript")
        != test_report["pytest_stdout_sha256"]
    ):
        raise ContractError("raw pytest transcript differs from the test report")
    validate_critical_test_nodes(test_junit_path)
    source_probe = control["SOURCE_PROBE_RECEIPT.json"].value
    validate_source_probe_receipt(
        source_probe,
        source_manifest=source_manifest,
        expected_source_manifest_sha256=source_manifest_file_sha256,
        expected_image_id=image_id,
        expected_source_commit=source_commit,
    )
    source_hashes = {
        "source_manifest_sha256": source_manifest_file_sha256,
        "runtime_inventory_sha256": source_manifest["runtime_inventory_sha256"],
        "build_inventory_sha256": source_manifest["build_inventory_sha256"],
        "dockerfile_sha256": expected_labels[DOCKERFILE_LABEL],
        "requirements_lock_sha256": expected_labels[REQUIREMENTS_LABEL],
        "orchestrator_sha256": _hash_regular_file(
            orchestrator_path, "campaign orchestrator"
        ),
        "state_machine_sha256": _hash_regular_file(
            state_machine_path, "campaign state machine"
        ),
        "execution_binding_sha256": _hash_regular_file(
            execution_binding_path, "execution binding"
        ),
        "finalizer_sha256": _hash_regular_file(finalizer_path, "campaign finalizer"),
        "posttrain_sha256": _hash_regular_file(posttrain_path, "post-training module"),
        "test_report_sha256": control["TEST_REPORT.json"].sha256,
        "source_probe_receipt_sha256": control["SOURCE_PROBE_RECEIPT.json"].sha256,
        "cell_order_sha256": _hash_regular_file(cell_order_path, "cell-order manifest"),
    }
    control_dir = _require_real_directory(host_control_dir, "host control directory")
    output_dir = _require_real_directory(
        host_output_dir, "host scientific output directory", empty=True
    )
    if control_dir != EXPECTED_HOST_CONTROL_DIR or output_dir != EXPECTED_HOST_OUTPUT_DIR:
        raise ContractError("host control/output path differs from the fixed execution root")
    if control_dir == output_dir or control_dir in output_dir.parents or output_dir in control_dir.parents:
        raise ContractError("control and scientific output directories must be disjoint")
    root = pre_readiness_control_root_sha256(
        source_commit=source_commit,
        image_id=image_id,
        execution_template_sha256_value=template_hash,
        file_sha256={name: artifact.sha256 for name, artifact in control.items()},
    )
    for artifact in (*control.values(), image_artifact):
        recheck_control_artifact(artifact)
    body = {
        "schema_version": 2,
        "control_schema_version": CONTROL_SCHEMA_VERSION,
        "control_binding_mode": CONTROL_BINDING_MODE,
        "renderer_version": RENDERER_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": READINESS_STATUS,
        "technical_execution_ready": True,
        "run_authorization_required": True,
        "run_authorization_issued": False,
        "training_authorized": False,
        "fresh_output_required": True,
        "resume_authorized": False,
        "pre_readiness_control_root_sha256": root,
        "launch_plan_file_sha256": control["HOST_LAUNCH_PLAN.json"].sha256,
        "launch_plan_sha256": plan_internal,
        "execution_template_sha256": template_hash,
        "input_binding_sha256": control["INPUT_BINDING.json"].sha256,
        "mount_receipt_sha256": control["MOUNT_RECEIPT.json"].sha256,
        "source_commit": source_commit,
        "image_id": image_id,
        "image_source_revision": source_commit,
        "image_inspect_sha256": image_artifact.sha256,
        **source_hashes,
        "critical_test_nodeids": list(REQUIRED_EXECUTION_TEST_NODEIDS),
        "planned_cells": 30,
        "max_concurrent_scientific_processes": 1,
        "total_gpu_hour_cap": TOTAL_GPU_HOUR_CAP,
        "sealed_capacity_gpu_hours": SEALED_CAPACITY_GPU_HOURS,
        "remaining_scientific_walltime_hours": REMAINING_SCIENTIFIC_WALLTIME_HOURS,
        "scientific_execution_timeout_seconds": SCIENTIFIC_EXECUTION_TIMEOUT_SECONDS,
        "post_timeout_cleanup_command_bound_seconds": (
            POST_TIMEOUT_CLEANUP_COMMAND_BOUND_SECONDS
        ),
        "scientific_cleanup_reserve_seconds": SCIENTIFIC_CLEANUP_RESERVE_SECONDS,
        "scientific_cleanup_safety_margin_seconds": (
            SCIENTIFIC_CLEANUP_SAFETY_MARGIN_SECONDS
        ),
        "cpu_limit_cores": CPU_LIMIT_CORES,
        "maximum_cpu_core_hours": MAXIMUM_CPU_CORE_HOURS,
        "host_control_dir": str(control_dir),
        "host_input_dir": str(ISOLATED_INPUT_DIRECTORY),
        "host_output_dir": str(output_dir),
        "host_log_dir": str(EXPECTED_HOST_LOG_DIR),
        "container_authorization_dir": str(CONTAINER_AUTHORIZATION_DIR),
        "container_output_dir": str(CONTAINER_OUTPUT_DIR),
        "container_entry_module": "fedcore.experiments.v3_campaign_runner",
        "container_command": list(SCIENTIFIC_COMMAND),
        "authorization_method_required": AUTHORIZATION_METHOD,
        "authorization_threat_model": AUTHORIZATION_THREAT_MODEL,
        "trusted_computing_base": TRUSTED_COMPUTING_BASE,
        "scientific_output_empty_when_readiness_issued": True,
        "PACS_opened": False,
        "PACS_inference_started": False,
        "PACS_training_started": False,
        "GPU_used": False,
    }
    return _self_hashed(body, "readiness_sha256")


def _launch_plan_self_hash(plan: Any) -> str:
    if not isinstance(plan, Mapping):
        raise ContractError("launch plan must be an object")
    digest = _require_hex(plan.get("launch_plan_sha256"), 64, "launch_plan_sha256")
    body = {
        key: value
        for key, value in plan.items()
        if key not in {"launch_plan_sha256", "launch_plan_hash_definition"}
    }
    if canonical_json_sha256(body) != digest:
        raise ContractError("launch plan self-hash mismatch")
    return digest


def validate_execution_readiness(
    value: Mapping[str, Any],
    *,
    static_launch_plan: Mapping[str, Any],
    input_binding: Mapping[str, Any],
    mount_receipt: Mapping[str, Any],
    artifact_sha256: Mapping[str, str],
    expected_source_commit: str | None = None,
) -> str:
    """Validate readiness against the actual bytes of the six pre-readiness files.

    ``artifact_sha256`` must be computed by the caller from the exact bytes it
    read; it is compared with readiness and folded into the control root.
    """

    if not isinstance(value, Mapping):
        raise ContractError("execution readiness must be an object")
    if value.get("schema_version") != 2 or value.get("control_schema_version") != CONTROL_SCHEMA_VERSION:
        raise ContractError("execution readiness is not control schema 2")
    digest = _validate_self_hash(value, "readiness_sha256", "execution readiness")
    _require_exact_keys(value, set(READINESS_KEYS), "execution readiness")
    if value.get("control_binding_mode") != CONTROL_BINDING_MODE:
        raise ContractError("execution-readiness binding mode drift")
    if value.get("renderer_version") != RENDERER_VERSION:
        raise ContractError("execution-readiness renderer version drift")
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != READINESS_STATUS:
        raise ContractError("execution-readiness status/protocol mismatch")
    _require_bool(value.get("technical_execution_ready"), True, "technical_execution_ready")
    _require_bool(value.get("run_authorization_required"), True, "run_authorization_required")
    _require_bool(value.get("run_authorization_issued"), False, "run_authorization_issued")
    _require_bool(value.get("training_authorized"), False, "training_authorized")
    _require_bool(value.get("fresh_output_required"), True, "fresh_output_required")
    _require_bool(value.get("resume_authorized"), False, "resume_authorized")
    if not isinstance(artifact_sha256, Mapping):
        raise ContractError("actual artifact hashes are absent")
    _require_exact_keys(artifact_sha256, set(PRE_READINESS_FILENAMES), "actual artifact hashes")
    for name, field in READINESS_FILE_FIELDS.items():
        _require_hex(artifact_sha256.get(name), 64, f"actual hash of {name}")
        if value.get(field) != artifact_sha256[name]:
            raise ContractError(f"readiness {field} differs from the actual {name} bytes")
    # Each supplied object must be the canonical encoding of the file whose
    # actual-byte hash was supplied for that role; this also refuses
    # non-canonical byte drift and role-swapped hashes.
    for name, supplied in (
        ("HOST_LAUNCH_PLAN.json", static_launch_plan),
        ("INPUT_BINDING.json", input_binding),
        ("MOUNT_RECEIPT.json", mount_receipt),
    ):
        if canonical_json_sha256(supplied) != artifact_sha256[name]:
            raise ContractError(f"{name} is not the canonical encoding of its actual bytes")
    if value.get("input_binding_sha256") != STATIC_INPUT_BINDING_FILE_SHA256:
        raise ContractError("readiness input-binding hash differs from the static anchor")
    source_commit = _require_hex(value.get("source_commit"), 40, "source_commit")
    if expected_source_commit is not None and source_commit != expected_source_commit:
        raise ContractError("readiness source commit differs from the running image")
    if value.get("image_source_revision") != source_commit:
        raise ContractError("readiness image/source label mismatch")
    image_id = _require_image_id(value.get("image_id"), "readiness image_id")
    plan_internal, template_hash = validate_launch_template_plan(
        static_launch_plan,
        input_binding=input_binding,
        expected_image_id=image_id,
        expected_source_commit=source_commit,
    )
    if value.get("launch_plan_sha256") != plan_internal:
        raise ContractError("readiness launch-plan internal hash mismatch")
    if value.get("execution_template_sha256") != template_hash:
        raise ContractError("readiness execution-template hash mismatch")
    if static_launch_plan.get("image_inspect_sha256") != value.get("image_inspect_sha256"):
        raise ContractError("readiness image inspect differs from the launch plan")
    validate_mount_declaration(mount_receipt, launch_plan=static_launch_plan, expected_image_id=image_id)
    root = pre_readiness_control_root_sha256(
        source_commit=source_commit,
        image_id=image_id,
        execution_template_sha256_value=template_hash,
        file_sha256=artifact_sha256,
    )
    if value.get("pre_readiness_control_root_sha256") != root:
        raise ContractError("readiness pre-readiness control root mismatch")
    for field in (
        "image_inspect_sha256",
        "source_manifest_sha256",
        "runtime_inventory_sha256",
        "build_inventory_sha256",
        "dockerfile_sha256",
        "requirements_lock_sha256",
        "orchestrator_sha256",
        "state_machine_sha256",
        "execution_binding_sha256",
        "finalizer_sha256",
        "posttrain_sha256",
        "test_report_sha256",
        "source_probe_receipt_sha256",
        "cell_order_sha256",
    ):
        _require_hex(value.get(field), 64, field)
    if value.get("critical_test_nodeids") != list(REQUIRED_EXECUTION_TEST_NODEIDS):
        raise ContractError("readiness critical test-node registry drift")
    if value.get("planned_cells") != 30 or value.get("max_concurrent_scientific_processes") != 1:
        raise ContractError("readiness cell/process denominator drift")
    _validate_gpu_resource_cap_invariant()
    expected_resources = {
        "total_gpu_hour_cap": TOTAL_GPU_HOUR_CAP,
        "sealed_capacity_gpu_hours": SEALED_CAPACITY_GPU_HOURS,
        "remaining_scientific_walltime_hours": REMAINING_SCIENTIFIC_WALLTIME_HOURS,
        "scientific_execution_timeout_seconds": SCIENTIFIC_EXECUTION_TIMEOUT_SECONDS,
        "post_timeout_cleanup_command_bound_seconds": (
            POST_TIMEOUT_CLEANUP_COMMAND_BOUND_SECONDS
        ),
        "scientific_cleanup_reserve_seconds": SCIENTIFIC_CLEANUP_RESERVE_SECONDS,
        "scientific_cleanup_safety_margin_seconds": (
            SCIENTIFIC_CLEANUP_SAFETY_MARGIN_SECONDS
        ),
        "cpu_limit_cores": CPU_LIMIT_CORES,
        "maximum_cpu_core_hours": MAXIMUM_CPU_CORE_HOURS,
    }
    if any(value.get(field) != expected for field, expected in expected_resources.items()):
        raise ContractError("readiness resource-cap drift")
    if value.get("container_authorization_dir") != str(CONTAINER_AUTHORIZATION_DIR):
        raise ContractError("readiness authorization mount destination drift")
    if value.get("container_output_dir") != str(CONTAINER_OUTPUT_DIR):
        raise ContractError("readiness output destination drift")
    if value.get("container_entry_module") != "fedcore.experiments.v3_campaign_runner":
        raise ContractError("readiness campaign entry module drift")
    if value.get("container_command") != list(SCIENTIFIC_COMMAND):
        raise ContractError("readiness scientific command drift")
    if value.get("host_control_dir") != str(EXPECTED_HOST_CONTROL_DIR):
        raise ContractError("readiness host control directory drift")
    if value.get("host_input_dir") != str(ISOLATED_INPUT_DIRECTORY):
        raise ContractError("readiness host input directory drift")
    if value.get("host_output_dir") != str(EXPECTED_HOST_OUTPUT_DIR):
        raise ContractError("readiness host output directory drift")
    if value.get("host_log_dir") != str(EXPECTED_HOST_LOG_DIR):
        raise ContractError("readiness host log directory drift")
    template_paths = static_launch_plan["execution_template"]["paths"]
    if template_paths != {
        "control": value["host_control_dir"],
        "input": value["host_input_dir"],
        "output": value["host_output_dir"],
        "log": value["host_log_dir"],
    }:
        raise ContractError("readiness host paths differ from the execution template")
    if value.get("authorization_method_required") != AUTHORIZATION_METHOD:
        raise ContractError("readiness authorization method drift")
    if value.get("authorization_threat_model") != AUTHORIZATION_THREAT_MODEL:
        raise ContractError("readiness authorization threat model drift")
    if value.get("trusted_computing_base") != TRUSTED_COMPUTING_BASE:
        raise ContractError("readiness trusted computing base drift")
    _require_bool(
        value.get("scientific_output_empty_when_readiness_issued"),
        True,
        "scientific_output_empty_when_readiness_issued",
    )
    for field in ("PACS_opened", "PACS_inference_started", "PACS_training_started", "GPU_used"):
        _require_bool(value.get(field), False, field)
    return digest


AUTHORIZATION_KEYS = frozenset(
    {
        "schema_version", "control_schema_version", "control_binding_mode",
        "protocol_id", "status", "training_authorized", "scope", "readiness_sha256",
        "authorization_challenge_sha256", "authorization_method", "source_commit",
        "image_id", "input_binding_sha256", "launch_plan_file_sha256",
        "launch_plan_sha256", "mount_receipt_sha256",
        "pre_readiness_control_root_sha256", "execution_template_sha256",
        "renderer_version", "instantiated_execution_configuration_sha256",
        "cell_order_sha256", "host_output_dir", "planned_cells",
        "max_concurrent_scientific_processes", "fresh_output_required",
        "remaining_scientific_walltime_hours", "cpu_limit_cores",
        "maximum_cpu_core_hours", "resume_authorized", "authorization_nonce",
        "authorization_id", "authorized_by", "issued_at_utc", "valid_until_utc",
        "authorization_sha256", "authorization_sha256_definition",
    }
)
AUTHORIZATION_ECHO_FIELDS = (
    "source_commit",
    "image_id",
    "input_binding_sha256",
    "launch_plan_file_sha256",
    "launch_plan_sha256",
    "mount_receipt_sha256",
    "pre_readiness_control_root_sha256",
    "execution_template_sha256",
    "renderer_version",
    "cell_order_sha256",
    "host_output_dir",
    "remaining_scientific_walltime_hours",
    "cpu_limit_cores",
    "maximum_cpu_core_hours",
)


def authorization_id_for(
    *,
    readiness_sha256: str,
    authorization_nonce: str,
    issued_at_utc: str,
    valid_until_utc: str,
) -> str:
    """Derive the authorization ID from the readiness challenge, never from a self-hash."""

    return canonical_json_sha256(
        {
            "authorization_challenge_sha256": _authorization_challenge(readiness_sha256),
            "authorization_nonce": _require_hex(authorization_nonce, 64, "authorization_nonce"),
            "authorized_by": AUTHORIZED_BY,
            "issued_at_utc": issued_at_utc,
            "valid_until_utc": valid_until_utc,
        }
    )


def build_run_authorization_template(readiness: Mapping[str, Any]) -> dict[str, Any]:
    """Return a non-authorizing template.  It can never pass validation."""

    readiness_hash = _validate_self_hash(readiness, "readiness_sha256", "execution readiness")
    value: dict[str, Any] = {
        "schema_version": 2,
        "control_schema_version": CONTROL_SCHEMA_VERSION,
        "control_binding_mode": CONTROL_BINDING_MODE,
        "protocol_id": PROTOCOL_ID,
        "status": "TEMPLATE_NOT_AUTHORIZED",
        "training_authorized": False,
        "scope": "ONE_FRESH_30_CELL_CAMPAIGN",
        "readiness_sha256": readiness_hash,
        "authorization_challenge_sha256": _authorization_challenge(readiness_hash),
        "authorization_method": AUTHORIZATION_METHOD,
        "instantiated_execution_configuration_sha256": None,
        "planned_cells": 30,
        "max_concurrent_scientific_processes": 1,
        "fresh_output_required": True,
        "resume_authorized": False,
        "authorization_nonce": None,
        "authorization_id": None,
        "authorized_by": None,
        "issued_at_utc": None,
        "valid_until_utc": None,
        "authorization_sha256": None,
        "authorization_sha256_definition": (
            "canonical JSON before authorization_sha256 and its definition are inserted"
        ),
    }
    for field in AUTHORIZATION_ECHO_FIELDS:
        value[field] = readiness.get(field)
    return value


def validate_run_authorization(
    value: Mapping[str, Any],
    readiness: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
) -> str:
    """Validate an issued authorization; ``launch_plan`` supplies the template to render."""

    if not isinstance(value, Mapping):
        raise ContractError("run authorization must be an object")
    if value.get("schema_version") != 2 or value.get("control_schema_version") != CONTROL_SCHEMA_VERSION:
        raise ContractError("run authorization is not control schema 2")
    readiness_hash = _validate_self_hash(readiness, "readiness_sha256", "execution readiness")
    digest = _validate_self_hash(value, "authorization_sha256", "run authorization")
    _require_exact_keys(value, set(AUTHORIZATION_KEYS), "run authorization")
    if value.get("control_binding_mode") != CONTROL_BINDING_MODE:
        raise ContractError("run authorization binding mode drift")
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != AUTHORIZATION_STATUS:
        raise ContractError("RUN_AUTHORIZATION status/protocol mismatch")
    _require_bool(value.get("training_authorized"), True, "training_authorized")
    _require_bool(value.get("fresh_output_required"), True, "fresh_output_required")
    _require_bool(value.get("resume_authorized"), False, "resume_authorized")
    if value.get("scope") != "ONE_FRESH_30_CELL_CAMPAIGN":
        raise ContractError("run authorization scope mismatch")
    if value.get("readiness_sha256") != readiness_hash:
        raise ContractError("run authorization readiness hash mismatch")
    challenge = _authorization_challenge(readiness_hash)
    if value.get("authorization_challenge_sha256") != challenge:
        raise ContractError("run authorization challenge mismatch")
    if value.get("authorization_method") != AUTHORIZATION_METHOD:
        raise ContractError("run authorization method mismatch")
    for field in AUTHORIZATION_ECHO_FIELDS:
        if value.get(field) != readiness.get(field):
            raise ContractError(f"run authorization {field} mismatch")
    nonce = _require_hex(value.get("authorization_nonce"), 64, "authorization_nonce")
    authorization_id = _require_hex(value.get("authorization_id"), 64, "authorization_id")
    if value.get("authorized_by") != AUTHORIZED_BY:
        raise ContractError("run authorization authorizing identity mismatch")
    issued = _parse_timestamp(value.get("issued_at_utc"), "issued_at_utc")
    valid_until = _parse_timestamp(value.get("valid_until_utc"), "valid_until_utc")
    now = datetime.now(timezone.utc)
    if issued > now or valid_until <= issued or now > valid_until:
        raise ContractError("run authorization is future-dated, expired, or has an invalid window")
    expected_authorization_id = authorization_id_for(
        readiness_sha256=readiness_hash,
        authorization_nonce=nonce,
        issued_at_utc=value["issued_at_utc"],
        valid_until_utc=value["valid_until_utc"],
    )
    if authorization_id != expected_authorization_id:
        raise ContractError("run authorization ID is not derived from the challenge")
    if value.get("planned_cells") != 30 or value.get("max_concurrent_scientific_processes") != 1:
        raise ContractError("run authorization denominator/process drift")
    if authorization_id == "0" * 64:
        raise ContractError("run authorization ID must not be the all-zero placeholder")
    if _launch_plan_self_hash(launch_plan) != readiness.get("launch_plan_sha256"):
        raise ContractError("run authorization launch plan differs from readiness")
    template = launch_plan.get("execution_template")
    if execution_template_sha256(template) != value.get("execution_template_sha256"):
        raise ContractError("run authorization execution template mismatch")
    instantiated = render_authorized_execution(template, authorization_id)
    if value.get("instantiated_execution_configuration_sha256") != (
        instantiated_execution_configuration_sha256(instantiated)
    ):
        raise ContractError("run authorization instantiated execution configuration mismatch")
    return digest


AUTHORIZED_PLAN_ECHO_FIELDS = (
    "source_commit",
    "image_id",
    "input_binding_sha256",
    "launch_plan_file_sha256",
    "launch_plan_sha256",
    "mount_receipt_sha256",
    "pre_readiness_control_root_sha256",
    "execution_template_sha256",
    "cell_order_sha256",
    "remaining_scientific_walltime_hours",
    "cpu_limit_cores",
    "maximum_cpu_core_hours",
)


def build_authorized_campaign_plan(
    readiness: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    input_binding: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically render the Docker-create plan.  It never starts a container.

    The plan may reference the authorization hash; the authorization never
    references the plan hash, so there is no bidirectional cycle.
    """

    readiness_hash = _validate_self_hash(readiness, "readiness_sha256", "execution readiness")
    authorization_hash = validate_run_authorization(authorization, readiness, launch_plan)
    validate_static_input_binding(input_binding)
    instantiated = render_authorized_execution(
        launch_plan["execution_template"], str(authorization["authorization_id"])
    )
    body: dict[str, Any] = {
        "schema_version": 2,
        "control_schema_version": CONTROL_SCHEMA_VERSION,
        "control_binding_mode": CONTROL_BINDING_MODE,
        "renderer_version": RENDERER_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": AUTHORIZED_PLAN_STATUS,
        "execution_allowed": True,
        "container_start_allowed_only_after_inspect_receipt": True,
        "readiness_sha256": readiness_hash,
        "run_authorization_sha256": authorization_hash,
        "authorization_id": authorization["authorization_id"],
        "instantiated_execution_configuration_sha256": authorization[
            "instantiated_execution_configuration_sha256"
        ],
    }
    for field in AUTHORIZED_PLAN_ECHO_FIELDS:
        body[field] = readiness[field]
    body.update(
        {
            "mounts": instantiated["runtime"]["mounts"],
            "container_name": instantiated["container_name"],
            "docker_create_argv": instantiated["docker_create_argv"],
            "docker_start_argv_template": instantiated["docker_start_argv_template"],
            "runtime_inspect_receipt_required": True,
            "execution_started": False,
        }
    )
    return _self_hashed(body, "authorized_plan_sha256")


def validate_authorized_campaign_plan(
    value: Mapping[str, Any],
    readiness: Mapping[str, Any],
    authorization: Mapping[str, Any],
    input_binding: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
) -> str:
    if not isinstance(value, Mapping):
        raise ContractError("authorized campaign plan must be an object")
    if value.get("schema_version") != 2 or value.get("control_schema_version") != CONTROL_SCHEMA_VERSION:
        raise ContractError("authorized campaign plan is not control schema 2")
    digest = _validate_self_hash(value, "authorized_plan_sha256", "authorized plan")
    rebuilt = build_authorized_campaign_plan(
        readiness, authorization, input_binding=input_binding, launch_plan=launch_plan
    )
    if dict(value) != rebuilt:
        drift = sorted(
            key for key in set(value) | set(rebuilt) if value.get(key) != rebuilt.get(key)
        )
        raise ContractError(f"authorized plan differs from the deterministic render: {drift!r}")
    return digest


RUNTIME_RECEIPT_KEYS = frozenset(
    {
        "schema_version", "control_schema_version", "control_binding_mode",
        "protocol_id", "status", "readiness_sha256", "run_authorization_sha256",
        "authorized_plan_sha256", "authorization_id", "image_id",
        "execution_template_sha256", "instantiated_execution_configuration_sha256",
        "normalized_runtime_execution_sha256", "cell_order_sha256", "container_id",
        "container_name", "remaining_scientific_walltime_hours", "cpu_limit_cores",
        "maximum_cpu_core_hours", "runtime_inspect_sha256", "container_created",
        "container_inspected_before_start", "exact_mounts_verified",
        "exact_gpu_uuid_verified", "network_none", "root_filesystem_read_only",
        "capabilities_dropped", "no_new_privileges", "exact_cpu_limit_verified",
        "execution_started", "mounts", "runtime_receipt_sha256",
        "runtime_receipt_sha256_definition",
    }
)


def validate_runtime_receipt(
    value: Mapping[str, Any],
    *,
    readiness: Mapping[str, Any],
    authorization: Mapping[str, Any],
    authorized_plan: Mapping[str, Any],
    inspect_value: Any,
    input_binding: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
) -> str:
    if not isinstance(value, Mapping):
        raise ContractError("runtime inspect receipt must be an object")
    if value.get("schema_version") != 2 or value.get("control_schema_version") != CONTROL_SCHEMA_VERSION:
        raise ContractError("runtime inspect receipt is not control schema 2")
    digest = _validate_self_hash(value, "runtime_receipt_sha256", "runtime receipt")
    _require_exact_keys(value, set(RUNTIME_RECEIPT_KEYS), "runtime inspect receipt")
    rebuilt = build_runtime_receipt_from_inspect(
        inspect_value,
        readiness=readiness,
        authorization=authorization,
        authorized_plan=authorized_plan,
        input_binding=input_binding,
        launch_plan=launch_plan,
    )
    if dict(value) != rebuilt:
        raise ContractError("runtime receipt differs from the exact raw Docker inspect")
    return digest


def build_runtime_receipt_from_inspect(
    inspect_value: Any,
    *,
    readiness: Mapping[str, Any],
    authorization: Mapping[str, Any],
    authorized_plan: Mapping[str, Any],
    input_binding: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a stopped scientific container before any ``docker start``.

    The caller must obtain ``inspect_value`` from the Docker daemon after
    ``docker create``.  Raw inspect is the authority for the actual mounts and
    configuration; MOUNT_RECEIPT.json is only a pre-execution declaration.
    This function has no Docker side effects.
    """

    plan_hash = validate_authorized_campaign_plan(
        authorized_plan, readiness, authorization, input_binding, launch_plan
    )
    inspect = _one_container(inspect_value, "scientific")
    host = inspect["HostConfig"]
    config = inspect["Config"]
    state = inspect.get("State")
    if not isinstance(state, Mapping):
        raise ContractError("scientific docker inspect lacks State")
    if str(inspect.get("Image", "")) != readiness.get("image_id"):
        raise ContractError("scientific container image ID mismatch")
    _require_hex(inspect.get("Id"), 64, "container_id")
    if state.get("Status") != "created" or state.get("Running") is not False:
        raise ContractError("scientific container must be inspected in stopped created state")
    if state.get("Dead") not in (None, False) or state.get("OOMKilled") not in (None, False):
        raise ContractError("scientific container state is not a clean created state")
    if host.get("Privileged") is not False or host.get("Devices") not in (None, []):
        raise ContractError("scientific container exposes unregistered host privilege/device")
    if host.get("AutoRemove") is not False:
        raise ContractError("scientific container must not auto-remove its inspect evidence")
    if host.get("CapAdd") not in (None, []):
        raise ContractError("scientific container adds capabilities")
    for field in ("Links", "VolumesFrom"):
        if host.get(field) not in (None, []):
            raise ContractError(f"scientific container has unregistered {field}")
    if host.get("NanoCpus") != DOCKER_NANO_CPUS:
        raise ContractError("scientific container CPU limit drift")
    labels = config.get("Labels") or {}
    if not isinstance(labels, Mapping) or labels.get(
        "org.opencontainers.image.revision"
    ) != readiness.get("source_commit"):
        raise ContractError("scientific container image revision label mismatch")
    expected_source_labels = {
        SOURCE_MANIFEST_LABEL: readiness.get("source_manifest_sha256"),
        RUNTIME_INVENTORY_LABEL: readiness.get("runtime_inventory_sha256"),
        BUILD_INVENTORY_LABEL: readiness.get("build_inventory_sha256"),
        DOCKERFILE_LABEL: readiness.get("dockerfile_sha256"),
        REQUIREMENTS_LABEL: readiness.get("requirements_lock_sha256"),
    }
    for label, expected in expected_source_labels.items():
        if labels.get(label) != expected:
            raise ContractError(f"scientific container source label mismatch: {label}")
    normalized = normalize_runtime_inspect_execution(inspect_value)
    instantiated = render_authorized_execution(
        launch_plan["execution_template"], str(authorization["authorization_id"])
    )
    expected_runtime = instantiated["runtime"]
    for key in RUNTIME_EXECUTION_KEYS:
        if normalized.get(key) != expected_runtime[key]:
            raise ContractError(f"runtime execution configuration drift: {key}")
    if instantiated_execution_configuration_sha256(instantiated) != authorization.get(
        "instantiated_execution_configuration_sha256"
    ):
        raise ContractError("runtime instantiated execution configuration mismatch")
    readiness_hash = _validate_self_hash(readiness, "readiness_sha256", "execution readiness")
    authorization_hash = validate_run_authorization(authorization, readiness, launch_plan)
    body = {
        "schema_version": 2,
        "control_schema_version": CONTROL_SCHEMA_VERSION,
        "control_binding_mode": CONTROL_BINDING_MODE,
        "protocol_id": PROTOCOL_ID,
        "status": RUNTIME_RECEIPT_STATUS,
        "readiness_sha256": readiness_hash,
        "run_authorization_sha256": authorization_hash,
        "authorized_plan_sha256": plan_hash,
        "authorization_id": authorization["authorization_id"],
        "image_id": readiness["image_id"],
        "execution_template_sha256": readiness["execution_template_sha256"],
        "instantiated_execution_configuration_sha256": authorization[
            "instantiated_execution_configuration_sha256"
        ],
        "normalized_runtime_execution_sha256": canonical_json_sha256(normalized),
        "cell_order_sha256": readiness["cell_order_sha256"],
        "remaining_scientific_walltime_hours": readiness[
            "remaining_scientific_walltime_hours"
        ],
        "cpu_limit_cores": readiness["cpu_limit_cores"],
        "maximum_cpu_core_hours": readiness["maximum_cpu_core_hours"],
        "container_id": str(inspect.get("Id", "")),
        "container_name": str(inspect.get("Name", "")).lstrip("/"),
        "container_created": True,
        "container_inspected_before_start": True,
        "exact_mounts_verified": True,
        "exact_gpu_uuid_verified": True,
        "network_none": True,
        "root_filesystem_read_only": True,
        "capabilities_dropped": True,
        "no_new_privileges": True,
        "exact_cpu_limit_verified": True,
        "execution_started": False,
        "runtime_inspect_sha256": canonical_json_sha256(inspect_value),
        "mounts": expected_runtime["mounts"],
    }
    if body["container_name"] != authorized_plan["container_name"]:
        raise ContractError("scientific container ID/name mismatch")
    return _self_hashed(body, "runtime_receipt_sha256")


def canonical_runtime_inspect_comparison_bytes(inspect_value: Any) -> bytes:
    """Canonicalize only Docker's order-unstable top-level mount list.

    The sealed raw inspect and its receipt retain their exact original bytes and
    hash.  This representation is used only when comparing a later fresh
    ``docker inspect`` with that sealed object.  Every field is retained, every
    list other than ``Mounts`` remains order-sensitive, and duplicate mount rows
    remain present.
    """

    if (
        not isinstance(inspect_value, list)
        or len(inspect_value) != 1
        or not isinstance(inspect_value[0], Mapping)
    ):
        raise ContractError(
            "scientific docker inspect must contain exactly one container"
        )
    mounts = inspect_value[0].get("Mounts")
    if not isinstance(mounts, list) or any(
        not isinstance(row, Mapping) for row in mounts
    ):
        raise ContractError("scientific container mount list is absent or malformed")

    normalized = deepcopy(inspect_value)
    normalized[0]["Mounts"] = sorted(
        normalized[0]["Mounts"], key=canonical_json_bytes
    )
    return canonical_json_bytes(normalized)


def validate_mounted_authorization_bundle(
    authorization_dir: Path = CONTAINER_AUTHORIZATION_DIR,
    *,
    expected_source_commit: str | None = None,
) -> VerifiedExecutionAuthorization:
    """Validate the exact mounted control directory inside the container."""

    root = Path(authorization_dir)
    try:
        metadata = root.lstat()
    except FileNotFoundError as exc:
        raise ContractError("authorization directory is not mounted") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ContractError("authorization mount must be a real directory")
    names = tuple(sorted(item.name for item in root.iterdir()))
    if names != tuple(sorted(CONTROL_FILENAMES)):
        raise ContractError(f"authorization directory content drift: {names!r}")
    artifacts = {
        name: read_control_artifact(
            root / name,
            require_object=name != "RUNTIME_INSPECT.json",
            require_read_only=True,
        )
        for name in CONTROL_FILENAMES
    }
    static_plan = artifacts["HOST_LAUNCH_PLAN.json"].value
    input_binding = artifacts["INPUT_BINDING.json"].value
    mount_receipt = artifacts["MOUNT_RECEIPT.json"].value
    source_manifest = artifacts["SOURCE_MANIFEST.json"].value
    source_manifest_file_hash = artifacts["SOURCE_MANIFEST.json"].sha256
    source_probe = artifacts["SOURCE_PROBE_RECEIPT.json"].value
    test_report = artifacts["TEST_REPORT.json"].value
    readiness = artifacts["EXECUTION_READINESS.json"].value
    authorization = artifacts["RUN_AUTHORIZATION.json"].value
    authorized_plan = artifacts["AUTHORIZED_CAMPAIGN_PLAN.json"].value
    runtime_receipt = artifacts["RUNTIME_INSPECT_RECEIPT.json"].value
    runtime_inspect = artifacts["RUNTIME_INSPECT.json"].value

    if artifacts["INPUT_BINDING.json"].sha256 != STATIC_INPUT_BINDING_FILE_SHA256:
        raise ContractError("mounted input-binding bytes differ from the static anchor")
    readiness_hash = validate_execution_readiness(
        readiness,
        static_launch_plan=static_plan,
        input_binding=input_binding,
        mount_receipt=mount_receipt,
        artifact_sha256={name: artifacts[name].sha256 for name in PRE_READINESS_FILENAMES},
        expected_source_commit=expected_source_commit,
    )
    validate_scientific_source_manifest(
        source_manifest, expected_source_commit=str(readiness["source_commit"])
    )
    if source_manifest.get("runtime_inventory_sha256") != readiness.get(
        "runtime_inventory_sha256"
    ) or source_manifest.get("build_inventory_sha256") != readiness.get(
        "build_inventory_sha256"
    ):
        raise ContractError("mounted source-manifest inventory differs from readiness")
    validate_source_probe_receipt(
        source_probe,
        source_manifest=source_manifest,
        expected_source_manifest_sha256=source_manifest_file_hash,
        expected_image_id=str(readiness["image_id"]),
        expected_source_commit=str(readiness["source_commit"]),
    )
    _validate_manifest_files(
        source_manifest,
        CONTAINER_WORKSPACE_DIR,
        include_runtime=True,
        include_build=False,
    )
    validate_execution_test_report(
        test_report,
        expected_image_id=str(readiness["image_id"]),
        expected_source_commit=str(readiness["source_commit"]),
        expected_source_manifest_sha256=source_manifest_file_hash,
    )
    workspace_sources = {
        "orchestrator_sha256": CONTAINER_WORKSPACE_DIR
        / "fedcore/experiments/v3_campaign_runner.py",
        "state_machine_sha256": CONTAINER_WORKSPACE_DIR
        / "fedcore/experiments/v3_campaign_state.py",
        "execution_binding_sha256": CONTAINER_WORKSPACE_DIR
        / "fedcore/experiments/v3_execution_binding.py",
        "finalizer_sha256": CONTAINER_WORKSPACE_DIR
        / "fedcore/experiments/v3_campaign_finalizer.py",
        "posttrain_sha256": CONTAINER_WORKSPACE_DIR
        / "fedcore/experiments/v3_posttrain.py",
        "cell_order_sha256": CONTAINER_WORKSPACE_DIR
        / "paper/ijar-v34-pacs-fresh-seed-hsb-v3/PACS_MODEL_MATRIX.csv",
    }
    for field, path in workspace_sources.items():
        if _hash_regular_file(path, field) != readiness.get(field):
            raise ContractError(f"running image source bytes differ for {field}")
    authorization_hash = validate_run_authorization(authorization, readiness, static_plan)
    authorized_plan_hash = validate_authorized_campaign_plan(
        authorized_plan, readiness, authorization, input_binding, static_plan
    )
    rendered_plan_bytes = canonical_json_bytes(
        build_authorized_campaign_plan(
            readiness, authorization, input_binding=input_binding, launch_plan=static_plan
        )
    )
    if artifacts["AUTHORIZED_CAMPAIGN_PLAN.json"].payload != rendered_plan_bytes:
        raise ContractError("mounted authorized-plan bytes differ from the deterministic render")
    runtime_hash = validate_runtime_receipt(
        runtime_receipt,
        readiness=readiness,
        authorization=authorization,
        authorized_plan=authorized_plan,
        inspect_value=runtime_inspect,
        input_binding=input_binding,
        launch_plan=static_plan,
    )
    if artifacts["RUNTIME_INSPECT.json"].sha256 != runtime_receipt.get("runtime_inspect_sha256"):
        raise ContractError("mounted raw inspect bytes differ from runtime receipt")
    return VerifiedExecutionAuthorization(
        readiness_sha256=readiness_hash,
        run_authorization_sha256=authorization_hash,
        authorized_plan_sha256=authorized_plan_hash,
        runtime_receipt_sha256=runtime_hash,
        input_binding_sha256=STATIC_INPUT_BINDING_FILE_SHA256,
        source_commit=str(readiness["source_commit"]),
        image_id=str(readiness["image_id"]),
        authorization_id=str(authorization["authorization_id"]),
    )


def acquire_exclusive_lease(path: Path, payload: Mapping[str, Any]) -> None:
    """Create an O_EXCL lease to prevent concurrent or replayed fresh starts."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, 0o444)
    except FileExistsError as exc:
        raise ContractError(f"exclusive campaign lease already exists: {destination}") from exc
    try:
        data = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n"
        ).encode("utf-8")
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise ContractError(f"short write while sealing lease: {destination}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


__all__ = [
    "AUTHORIZATION_STATUS",
    "AUTHORIZED_PLAN_STATUS",
    "BUILD_INVENTORY_LABEL",
    "CANONICAL_CONTAINER_WORKSPACE_DIR",
    "CONTAINER_AUTHORIZATION_DIR",
    "CONTROL_FILENAMES",
    "CPU_LIMIT_CORES",
    "DOCKERFILE_LABEL",
    "DOCKER_NANO_CPUS",
    "MAXIMUM_CPU_CORE_HOURS",
    "POST_TIMEOUT_CLEANUP_COMMAND_BOUND_SECONDS",
    "READINESS_STATUS",
    "REMAINING_SCIENTIFIC_WALLTIME_HOURS",
    "SCIENTIFIC_CLEANUP_RESERVE_SECONDS",
    "SCIENTIFIC_CLEANUP_SAFETY_MARGIN_SECONDS",
    "SCIENTIFIC_EXECUTION_TIMEOUT_SECONDS",
    "REQUIRED_EXECUTION_TEST_FILES",
    "REQUIRED_EXECUTION_GATE_NODEIDS",
    "REQUIRED_EXECUTION_TEST_GATES",
    "REQUIRED_EXECUTION_PYTEST_ARGV",
    "REQUIREMENTS_LABEL",
    "RUNTIME_RECEIPT_STATUS",
    "RUNTIME_INVENTORY_LABEL",
    "SOURCE_MANIFEST_LABEL",
    "SOURCE_MANIFEST_STATUS",
    "SOURCE_PROBE_STATUS",
    "TEST_REPORT_STATUS",
    "TEST_RUNTIME_PYTEST_VERSION",
    "TEST_RUNTIME_ROOT",
    "TEST_RUNTIME_WHEEL_FILES",
    "STATIC_INPUT_BINDING_FILE_SHA256",
    "CONTAINER_NAME_RULE",
    "CONTROL_BINDING_MODE",
    "CONTROL_SCHEMA_VERSION",
    "ControlArtifact",
    "EXPECTED_HOST_LOG_DIR",
    "LAUNCH_PLAN_KEYS",
    "LAUNCH_TEMPLATE_STATUS",
    "MOUNT_DECLARATION_ROLE",
    "MOUNT_DECLARATION_STATUS",
    "MOUNT_RECEIPT_KEYS",
    "PLAN_KIND",
    "PRE_READINESS_FILENAMES",
    "RENDERER_VERSION",
    "REQUIRED_EXECUTION_TEST_NODEIDS",
    "VerifiedExecutionAuthorization",
    "acquire_exclusive_lease",
    "authorization_id_for",
    "build_authorized_campaign_plan",
    "build_execution_readiness",
    "build_execution_template",
    "build_run_authorization_template",
    "build_runtime_receipt_from_inspect",
    "canonical_mounts",
    "canonical_runtime_inspect_comparison_bytes",
    "build_scientific_source_manifest",
    "build_source_probe_receipt",
    "execution_template_sha256",
    "instantiated_execution_configuration_sha256",
    "normalize_runtime_inspect_execution",
    "pre_readiness_control_root_sha256",
    "read_control_artifact",
    "recheck_control_artifact",
    "render_authorized_execution",
    "validate_authorized_campaign_plan",
    "validate_critical_test_nodes",
    "validate_execution_template",
    "validate_launch_template_plan",
    "validate_mount_declaration",
    "validate_execution_readiness",
    "validate_execution_test_report",
    "validate_execution_test_report_against_git_commit",
    "validate_test_evidence_bundle_paths",
    "validate_test_container_inspect",
    "validate_mounted_authorization_bundle",
    "validate_run_authorization",
    "validate_runtime_receipt",
    "validate_scientific_source_manifest",
    "validate_source_probe_receipt",
    "validate_static_input_binding",
]
