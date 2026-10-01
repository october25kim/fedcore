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

from fedcore.experiments.v3_contract import (
    PROTOCOL_ID,
    ContractError,
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_file,
)
from fedcore.experiments.v3_input_binding import REGISTERED_SCIENTIFIC_INPUTS


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
TOTAL_GPU_HOUR_CAP = 48.0
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
    "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_v3_exec/control/execution-r3"
)
EXPECTED_HOST_OUTPUT_DIR = Path(
    "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_v3_exec/runs/scientific/"
    "FEDCORE-IJAR-V34-PACS-FRESH-SEED-REPLICATION-HSB-v3"
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

# Public r2 static-binding anchors.  These constants prevent a self-consistent
# replacement of the preceding launch plan, input receipt, or mount receipt.
STATIC_BINDING_ROOT_SHA256 = (
    "362f4e9dc19b9276db54fdef2748887296bc0c01f8e40af56b90a2613a5db752"
)
STATIC_LAUNCH_PLAN_FILE_SHA256 = (
    "8eb4fda2228c569a556386dfd7f91a95a9c3400f1c7f706dec2e7325a9f98280"
)
STATIC_LAUNCH_PLAN_SHA256 = (
    "0eb4a8b50adbe552c7506475eec379774a776c73f5601fa8e670a26100c37364"
)
STATIC_INPUT_BINDING_FILE_SHA256 = (
    "e12f305858bf793dac02a1e7b38575b4f12fd5ab66872efb8bb40894f00928d9"
)
STATIC_MOUNT_RECEIPT_FILE_SHA256 = (
    "cc4bbbfc68faacfe4e9f6c6f6ebf52c26d44540afac82b84619b19a24b50ef8b"
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


def _read_control_bytes_once(
    path: Path, *, require_object: bool = True
) -> tuple[Any, str]:
    """Hash and parse one descriptor-backed byte buffer.

    Reading once through an ``O_NOFOLLOW`` descriptor prevents a path swap
    between a hash pass and a later JSON parse pass.  This is used by the
    in-container guard, where the mounted bytes are the actual authority.
    """

    path = Path(path)
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"required control artifact is missing: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ContractError(f"control artifact must be a regular nonsymlink file: {path}")
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
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > 16 * 1024 * 1024:
                raise ContractError(f"control artifact exceeds the 16 MiB limit: {path}")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
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
    import hashlib

    return value, hashlib.sha256(payload).hexdigest()


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


def validate_static_launch_plan(value: Mapping[str, Any]) -> str:
    """Validate the public r2 plan against compiled registry values.

    This check is intentionally structural.  Host file re-hashing belongs to
    the r2 builder; the execution layer consumes its immutable public bytes.
    """

    if value.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("static launch-plan protocol mismatch")
    if value.get("status") != (
        "PASS_CANDIDATE_HOST_LAUNCH_PLAN_HOLD_EXECUTION_READINESS"
    ):
        raise ContractError("static launch-plan status mismatch")
    _require_bool(value.get("execution_allowed"), False, "execution_allowed")
    if value.get("launch_plan_sha256") != STATIC_LAUNCH_PLAN_SHA256:
        raise ContractError("static launch-plan registry hash mismatch")
    body = {
        key: item
        for key, item in value.items()
        if key not in {"launch_plan_sha256", "launch_plan_hash_definition"}
    }
    if canonical_json_sha256(body) != STATIC_LAUNCH_PLAN_SHA256:
        raise ContractError("static launch-plan canonical hash mismatch")
    inventory = value.get("input_inventory")
    if not isinstance(inventory, list) or len(inventory) != 5:
        raise ContractError("static launch plan must contain exactly five inputs")
    for spec, row in zip(REGISTERED_SCIENTIFIC_INPUTS, inventory, strict=True):
        expected = {
            "name": spec.name,
            "provenance_source": spec.source_path,
            "destination": spec.container_path,
            "sha256": spec.sha256,
            "bytes": spec.size_bytes,
            "read_only": True,
            "symlink": False,
        }
        if not isinstance(row, Mapping) or any(row.get(k) != v for k, v in expected.items()):
            raise ContractError(f"static launch-plan input registry drift for {spec.name}")
    if value.get("runtime") != {
        "gpu_index": 0,
        "gpu_name": "NVIDIA GeForce RTX 4070 Ti SUPER",
        "gpu_uuid": "GPU-ee5a0082-32e6-ede4-27d5-441bee0ca7c6",
        "max_concurrent_scientific_processes": 1,
        "network": "none",
        "root_filesystem_read_only": True,
        "single_process": True,
        "user": "1000:1000",
    }:
        raise ContractError("static launch-plan runtime registry drift")
    return STATIC_LAUNCH_PLAN_SHA256


def validate_static_input_binding(value: Mapping[str, Any]) -> None:
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != "PASS_FIVE_INPUT_BINDING":
        raise ContractError("static input-binding status/protocol mismatch")
    _require_bool(value.get("exact_directory_contents"), True, "exact_directory_contents")
    _require_bool(value.get("all_regular_nonsymlink"), True, "all_regular_nonsymlink")
    _require_bool(value.get("all_read_only"), True, "all_read_only")
    rows = value.get("files")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ContractError("static input binding must contain exactly five records")
    for spec, row in zip(REGISTERED_SCIENTIFIC_INPUTS, rows, strict=True):
        expected = {
            "name": spec.name,
            "source_path": spec.source_path,
            "container_path": spec.container_path,
            "sha256": spec.sha256,
            "size_bytes": spec.size_bytes,
            "read_only_mount_required": True,
            "regular_file": True,
            "symlink": False,
        }
        if not isinstance(row, Mapping) or any(row.get(k) != v for k, v in expected.items()):
            raise ContractError(f"static input-binding registry drift for {spec.name}")


def validate_static_mount_receipt(value: Mapping[str, Any]) -> None:
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("static mount-receipt protocol mismatch")
    if value.get("status") != "PASS_EXACT_FIVE_RO_MOUNT_PROBE_NO_GPU":
        raise ContractError("static mount-receipt status mismatch")
    if value.get("launch_plan_sha256") != STATIC_LAUNCH_PLAN_SHA256:
        raise ContractError("static mount-receipt launch-plan mismatch")
    if value.get("input_binding_sha256") != STATIC_INPUT_BINDING_FILE_SHA256:
        raise ContractError("static mount-receipt input-binding mismatch")
    for field in (
        "exact_five_read_only_inputs",
        "one_read_write_output",
        "tmpfs_only_at_tmp",
    ):
        _require_bool(value.get(field), True, field)
    _require_bool(value.get("GPU_used"), False, "GPU_used")
    _require_bool(value.get("training_started"), False, "training_started")


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
    """Create technical readiness without creating run authorization."""

    _validate_gpu_resource_cap_invariant()
    plan = _read_object(static_launch_plan_path)
    binding = _read_object(input_binding_path)
    mount = _read_object(mount_receipt_path)
    validate_static_launch_plan(plan)
    validate_static_input_binding(binding)
    validate_static_mount_receipt(mount)
    if sha256_file(static_launch_plan_path) != STATIC_LAUNCH_PLAN_FILE_SHA256:
        raise ContractError("public static launch-plan file hash mismatch")
    if sha256_file(input_binding_path) != STATIC_INPUT_BINDING_FILE_SHA256:
        raise ContractError("public input-binding file hash mismatch")
    if sha256_file(mount_receipt_path) != STATIC_MOUNT_RECEIPT_FILE_SHA256:
        raise ContractError("public mount-receipt file hash mismatch")
    source_commit = _require_hex(source_commit, 40, "source_commit")
    source_manifest = _read_object(source_manifest_path)
    runtime_sources, build_sources = validate_scientific_source_manifest(
        source_manifest, expected_source_commit=source_commit
    )
    repository_root = Path(execution_binding_path).resolve().parents[2]
    _validate_manifest_files(
        source_manifest,
        repository_root,
        include_runtime=True,
        include_build=True,
    )
    _validate_manifest_against_git_commit(source_manifest, repository_root)
    source_manifest_file_sha256 = _hash_regular_file(
        source_manifest_path, "source manifest"
    )
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise ContractError("image_id must be an immutable sha256 reference")
    _require_hex(image_id.removeprefix("sha256:"), 64, "image_id")
    image_inspect_sha256 = _hash_regular_file(image_inspect_path, "image inspect")
    try:
        image_inspect = json.loads(Path(image_inspect_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("image inspect is not valid JSON") from exc
    if (
        not isinstance(image_inspect, list)
        or len(image_inspect) != 1
        or not isinstance(image_inspect[0], Mapping)
    ):
        raise ContractError("image inspect must contain exactly one image")
    image = image_inspect[0]
    config = image.get("Config")
    if not isinstance(config, Mapping) or image.get("Id") != image_id:
        raise ContractError("image inspect immutable ID mismatch")
    labels = config.get("Labels") or {}
    if not isinstance(labels, Mapping) or labels.get(
        "org.opencontainers.image.revision"
    ) != source_commit:
        raise ContractError("image inspect source-revision label mismatch")
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
    environment = {str(item) for item in (config.get("Env") or [])}
    if f"FEDCORE_SOURCE_COMMIT={source_commit}" not in environment:
        raise ContractError("image inspect lacks the immutable source environment")
    if config.get("Entrypoint") != ["python"] or config.get("User") != "1000:1000":
        raise ContractError("image inspect entrypoint/user mismatch")
    test_report = _read_object(test_report_path)
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
    source_probe = _read_object(source_probe_receipt_path)
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
        "test_report_sha256": _hash_regular_file(test_report_path, "test report"),
        "source_probe_receipt_sha256": _hash_regular_file(
            source_probe_receipt_path, "source-probe receipt"
        ),
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
    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": READINESS_STATUS,
        "technical_execution_ready": True,
        "run_authorization_required": True,
        "run_authorization_issued": False,
        "training_authorized": False,
        "fresh_output_required": True,
        "resume_authorized": False,
        "static_binding_root_sha256": STATIC_BINDING_ROOT_SHA256,
        "static_launch_plan_file_sha256": STATIC_LAUNCH_PLAN_FILE_SHA256,
        "static_launch_plan_sha256": STATIC_LAUNCH_PLAN_SHA256,
        "input_binding_sha256": STATIC_INPUT_BINDING_FILE_SHA256,
        "mount_receipt_sha256": STATIC_MOUNT_RECEIPT_FILE_SHA256,
        "source_commit": source_commit,
        "image_id": image_id,
        "image_source_revision": source_commit,
        "image_inspect_sha256": image_inspect_sha256,
        **source_hashes,
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
        "host_output_dir": str(output_dir),
        "container_authorization_dir": str(CONTAINER_AUTHORIZATION_DIR),
        "container_output_dir": str(CONTAINER_OUTPUT_DIR),
        "container_entry_module": "fedcore.experiments.v3_campaign_runner",
        "container_command": list(SCIENTIFIC_COMMAND),
        "authorization_method_required": AUTHORIZATION_METHOD,
        "authorization_threat_model": (
            "trusted-host procedural authorization; no cryptographic signer identity claim"
        ),
        "scientific_output_empty_when_readiness_issued": True,
        "PACS_opened": False,
        "PACS_inference_started": False,
        "PACS_training_started": False,
        "GPU_used": False,
    }
    return _self_hashed(body, "readiness_sha256")


def validate_execution_readiness(
    value: Mapping[str, Any],
    *,
    static_launch_plan: Mapping[str, Any],
    input_binding: Mapping[str, Any],
    mount_receipt: Mapping[str, Any],
    expected_source_commit: str | None = None,
) -> str:
    digest = _validate_self_hash(value, "readiness_sha256", "execution readiness")
    expected_keys = {
        "schema_version", "protocol_id", "status", "technical_execution_ready",
        "run_authorization_required", "run_authorization_issued", "training_authorized",
        "fresh_output_required", "resume_authorized", "static_binding_root_sha256",
        "static_launch_plan_file_sha256", "static_launch_plan_sha256",
        "input_binding_sha256", "mount_receipt_sha256", "source_commit", "image_id",
        "image_source_revision", "image_inspect_sha256", "source_manifest_sha256",
        "runtime_inventory_sha256", "build_inventory_sha256", "dockerfile_sha256",
        "requirements_lock_sha256",
        "orchestrator_sha256", "state_machine_sha256", "execution_binding_sha256",
        "finalizer_sha256", "posttrain_sha256", "test_report_sha256",
        "source_probe_receipt_sha256",
        "cell_order_sha256", "planned_cells", "max_concurrent_scientific_processes",
        "total_gpu_hour_cap", "sealed_capacity_gpu_hours",
        "remaining_scientific_walltime_hours", "scientific_execution_timeout_seconds",
        "post_timeout_cleanup_command_bound_seconds",
        "scientific_cleanup_reserve_seconds",
        "scientific_cleanup_safety_margin_seconds", "cpu_limit_cores",
        "maximum_cpu_core_hours",
        "host_control_dir", "host_output_dir", "container_authorization_dir",
        "container_output_dir", "container_entry_module", "container_command",
        "authorization_method_required", "authorization_threat_model",
        "scientific_output_empty_when_readiness_issued", "PACS_opened",
        "PACS_inference_started", "PACS_training_started", "GPU_used",
        "readiness_sha256", "readiness_sha256_definition",
    }
    _require_exact_keys(value, expected_keys, "execution readiness")
    if value.get("schema_version") != 1:
        raise ContractError("execution-readiness schema version mismatch")
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != READINESS_STATUS:
        raise ContractError("execution-readiness status/protocol mismatch")
    _require_bool(value.get("technical_execution_ready"), True, "technical_execution_ready")
    _require_bool(value.get("run_authorization_required"), True, "run_authorization_required")
    _require_bool(value.get("run_authorization_issued"), False, "run_authorization_issued")
    _require_bool(value.get("training_authorized"), False, "training_authorized")
    _require_bool(value.get("fresh_output_required"), True, "fresh_output_required")
    _require_bool(value.get("resume_authorized"), False, "resume_authorized")
    if value.get("static_binding_root_sha256") != STATIC_BINDING_ROOT_SHA256:
        raise ContractError("readiness static-binding root mismatch")
    if value.get("static_launch_plan_file_sha256") != STATIC_LAUNCH_PLAN_FILE_SHA256:
        raise ContractError("readiness static-plan file mismatch")
    if value.get("static_launch_plan_sha256") != STATIC_LAUNCH_PLAN_SHA256:
        raise ContractError("readiness static-plan hash mismatch")
    if value.get("input_binding_sha256") != STATIC_INPUT_BINDING_FILE_SHA256:
        raise ContractError("readiness input-binding hash mismatch")
    if value.get("mount_receipt_sha256") != STATIC_MOUNT_RECEIPT_FILE_SHA256:
        raise ContractError("readiness mount-receipt hash mismatch")
    validate_static_launch_plan(static_launch_plan)
    validate_static_input_binding(input_binding)
    validate_static_mount_receipt(mount_receipt)
    source_commit = _require_hex(value.get("source_commit"), 40, "source_commit")
    if expected_source_commit is not None and source_commit != expected_source_commit:
        raise ContractError("readiness source commit differs from the running image")
    if value.get("image_source_revision") != source_commit:
        raise ContractError("readiness image/source label mismatch")
    image_id = str(value.get("image_id", ""))
    if not image_id.startswith("sha256:"):
        raise ContractError("readiness image ID is mutable or absent")
    _require_hex(image_id.removeprefix("sha256:"), 64, "image_id")
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
    if value.get("host_output_dir") != str(EXPECTED_HOST_OUTPUT_DIR):
        raise ContractError("readiness host output directory drift")
    if value.get("authorization_method_required") != AUTHORIZATION_METHOD:
        raise ContractError("readiness authorization method drift")
    if value.get("authorization_threat_model") != (
        "trusted-host procedural authorization; no cryptographic signer identity claim"
    ):
        raise ContractError("readiness authorization threat model drift")
    for field in (
        "scientific_output_empty_when_readiness_issued",
    ):
        _require_bool(value.get(field), True, field)
    for field in ("PACS_opened", "PACS_inference_started", "PACS_training_started", "GPU_used"):
        _require_bool(value.get(field), False, field)
    return digest


def build_run_authorization_template(readiness: Mapping[str, Any]) -> dict[str, Any]:
    """Return a non-authorizing template.  It can never pass validation."""

    readiness_hash = _validate_self_hash(readiness, "readiness_sha256", "execution readiness")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "TEMPLATE_NOT_AUTHORIZED",
        "training_authorized": False,
        "scope": "ONE_FRESH_30_CELL_CAMPAIGN",
        "readiness_sha256": readiness_hash,
        "authorization_challenge_sha256": _authorization_challenge(readiness_hash),
        "authorization_method": AUTHORIZATION_METHOD,
        "source_commit": readiness.get("source_commit"),
        "image_id": readiness.get("image_id"),
        "input_binding_sha256": readiness.get("input_binding_sha256"),
        "mount_receipt_sha256": readiness.get("mount_receipt_sha256"),
        "cell_order_sha256": readiness.get("cell_order_sha256"),
        "host_output_dir": readiness.get("host_output_dir"),
        "planned_cells": 30,
        "max_concurrent_scientific_processes": 1,
        "remaining_scientific_walltime_hours": readiness.get(
            "remaining_scientific_walltime_hours"
        ),
        "cpu_limit_cores": readiness.get("cpu_limit_cores"),
        "maximum_cpu_core_hours": readiness.get("maximum_cpu_core_hours"),
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


def validate_run_authorization(
    value: Mapping[str, Any], readiness: Mapping[str, Any]
) -> str:
    readiness_hash = _validate_self_hash(readiness, "readiness_sha256", "execution readiness")
    digest = _validate_self_hash(value, "authorization_sha256", "run authorization")
    expected_keys = {
        "schema_version", "protocol_id", "status", "training_authorized", "scope",
        "readiness_sha256", "authorization_challenge_sha256", "authorization_method",
        "source_commit", "image_id", "input_binding_sha256", "mount_receipt_sha256",
        "cell_order_sha256", "host_output_dir", "planned_cells",
        "max_concurrent_scientific_processes", "fresh_output_required",
        "remaining_scientific_walltime_hours", "cpu_limit_cores",
        "maximum_cpu_core_hours",
        "resume_authorized", "authorization_nonce", "authorization_id", "authorized_by",
        "issued_at_utc", "valid_until_utc", "authorization_sha256",
        "authorization_sha256_definition",
    }
    _require_exact_keys(value, expected_keys, "run authorization")
    if value.get("schema_version") != 1:
        raise ContractError("run-authorization schema version mismatch")
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
    for field in (
        "source_commit",
        "image_id",
        "input_binding_sha256",
        "mount_receipt_sha256",
        "cell_order_sha256",
        "host_output_dir",
        "remaining_scientific_walltime_hours",
        "cpu_limit_cores",
        "maximum_cpu_core_hours",
    ):
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
    expected_authorization_id = canonical_json_sha256(
        {
            "authorization_challenge_sha256": challenge,
            "authorization_nonce": nonce,
            "authorized_by": AUTHORIZED_BY,
            "issued_at_utc": value["issued_at_utc"],
            "valid_until_utc": value["valid_until_utc"],
        }
    )
    if authorization_id != expected_authorization_id:
        raise ContractError("run authorization ID is not derived from the challenge")
    if value.get("planned_cells") != 30 or value.get("max_concurrent_scientific_processes") != 1:
        raise ContractError("run authorization denominator/process drift")
    if authorization_id == "0" * 64:
        raise ContractError("run authorization ID must not be the all-zero placeholder")
    return digest


def expected_scientific_mounts(
    readiness: Mapping[str, Any], input_binding: Mapping[str, Any]
) -> list[dict[str, Any]]:
    validate_static_input_binding(input_binding)
    rows = input_binding["files"]
    mounts = [
        {
            "type": "bind",
            "source": str(row["staged_path"]),
            "destination": str(row["container_path"]),
            "read_only": True,
        }
        for row in rows
    ]
    mounts.extend(
        (
            {
                "type": "bind",
                "source": str(readiness["host_control_dir"]),
                "destination": str(CONTAINER_AUTHORIZATION_DIR),
                "read_only": True,
            },
            {
                "type": "bind",
                "source": str(readiness["host_output_dir"]),
                "destination": str(CONTAINER_OUTPUT_DIR),
                "read_only": False,
            },
            {
                "type": "tmpfs",
                "source": "",
                "destination": "/tmp",
                "read_only": False,
                "options": "rw,noexec,nosuid,size=1g",
            },
        )
    )
    return mounts


def build_authorized_campaign_plan(
    readiness: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a Docker-create plan.  It never starts the container."""

    readiness_hash = _validate_self_hash(readiness, "readiness_sha256", "execution readiness")
    authorization_hash = validate_run_authorization(authorization, readiness)
    mounts = expected_scientific_mounts(readiness, input_binding)
    name = f"fedcore-v3-{str(authorization['authorization_id'])[:12]}"
    argv = [
        "docker",
        "create",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=1g",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--restart",
        "no",
        "--user",
        "1000:1000",
        "--cpus",
        str(CPU_LIMIT_CORES),
        "--gpus",
        f"device={EXPECTED_GPU_UUID}",
    ]
    for mount in mounts:
        if mount["type"] == "tmpfs":
            continue
        value = (
            f"type=bind,src={mount['source']},dst={mount['destination']}"
            + (",readonly" if mount["read_only"] else "")
        )
        argv.extend(("--mount", value))
    argv.append(str(readiness["image_id"]))
    argv.extend(str(item) for item in readiness["container_command"])
    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": AUTHORIZED_PLAN_STATUS,
        "execution_allowed": True,
        "container_start_allowed_only_after_inspect_receipt": True,
        "readiness_sha256": readiness_hash,
        "run_authorization_sha256": authorization_hash,
        "authorization_id": authorization["authorization_id"],
        "source_commit": readiness["source_commit"],
        "image_id": readiness["image_id"],
        "input_binding_sha256": readiness["input_binding_sha256"],
        "mount_receipt_sha256": readiness["mount_receipt_sha256"],
        "cell_order_sha256": readiness["cell_order_sha256"],
        "remaining_scientific_walltime_hours": readiness[
            "remaining_scientific_walltime_hours"
        ],
        "cpu_limit_cores": readiness["cpu_limit_cores"],
        "maximum_cpu_core_hours": readiness["maximum_cpu_core_hours"],
        "mounts": mounts,
        "container_name": name,
        "docker_create_argv": argv,
        "docker_start_argv_template": ["docker", "start", "--attach", "__CONTAINER_ID__"],
        "runtime_inspect_receipt_required": True,
        "execution_started": False,
    }
    return _self_hashed(body, "authorized_plan_sha256")


def validate_authorized_campaign_plan(
    value: Mapping[str, Any],
    readiness: Mapping[str, Any],
    authorization: Mapping[str, Any],
    input_binding: Mapping[str, Any],
) -> str:
    digest = _validate_self_hash(value, "authorized_plan_sha256", "authorized plan")
    expected_keys = {
        "schema_version", "protocol_id", "status", "execution_allowed",
        "container_start_allowed_only_after_inspect_receipt", "readiness_sha256",
        "run_authorization_sha256", "authorization_id", "source_commit", "image_id",
        "input_binding_sha256", "mount_receipt_sha256", "cell_order_sha256", "mounts",
        "remaining_scientific_walltime_hours", "cpu_limit_cores",
        "maximum_cpu_core_hours",
        "container_name", "docker_create_argv", "docker_start_argv_template",
        "runtime_inspect_receipt_required", "execution_started", "authorized_plan_sha256",
        "authorized_plan_sha256_definition",
    }
    _require_exact_keys(value, expected_keys, "authorized campaign plan")
    if value.get("schema_version") != 1:
        raise ContractError("authorized-plan schema version mismatch")
    readiness_hash = _validate_self_hash(readiness, "readiness_sha256", "execution readiness")
    authorization_hash = validate_run_authorization(authorization, readiness)
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != AUTHORIZED_PLAN_STATUS:
        raise ContractError("authorized campaign-plan status/protocol mismatch")
    _require_bool(value.get("execution_allowed"), True, "execution_allowed")
    _require_bool(
        value.get("container_start_allowed_only_after_inspect_receipt"),
        True,
        "container_start_allowed_only_after_inspect_receipt",
    )
    _require_bool(value.get("runtime_inspect_receipt_required"), True, "runtime_inspect_receipt_required")
    _require_bool(value.get("execution_started"), False, "execution_started")
    if value.get("readiness_sha256") != readiness_hash:
        raise ContractError("authorized plan readiness mismatch")
    if value.get("run_authorization_sha256") != authorization_hash:
        raise ContractError("authorized plan run-authorization mismatch")
    for field in (
        "authorization_id",
        "source_commit",
        "image_id",
        "input_binding_sha256",
        "mount_receipt_sha256",
        "cell_order_sha256",
        "remaining_scientific_walltime_hours",
        "cpu_limit_cores",
        "maximum_cpu_core_hours",
    ):
        expected = authorization.get(field, readiness.get(field))
        if value.get(field) != expected:
            raise ContractError(f"authorized plan {field} mismatch")
    if value.get("mounts") != expected_scientific_mounts(readiness, input_binding):
        raise ContractError("authorized plan mount scope mismatch")
    rebuilt = build_authorized_campaign_plan(
        readiness, authorization, input_binding=input_binding
    )
    for field in (
        "container_name",
        "docker_create_argv",
        "docker_start_argv_template",
    ):
        if value.get(field) != rebuilt.get(field):
            raise ContractError(f"authorized plan {field} drift")
    return digest


def validate_runtime_receipt(
    value: Mapping[str, Any],
    *,
    readiness: Mapping[str, Any],
    authorization: Mapping[str, Any],
    authorized_plan: Mapping[str, Any],
    inspect_value: Any,
    input_binding: Mapping[str, Any],
) -> str:
    digest = _validate_self_hash(value, "runtime_receipt_sha256", "runtime receipt")
    expected_keys = {
        "schema_version", "protocol_id", "status", "readiness_sha256",
        "run_authorization_sha256", "authorized_plan_sha256", "authorization_id",
        "image_id", "cell_order_sha256", "container_id", "container_name",
        "remaining_scientific_walltime_hours", "cpu_limit_cores",
        "maximum_cpu_core_hours",
        "runtime_inspect_sha256", "container_created", "container_inspected_before_start",
        "exact_mounts_verified", "exact_gpu_uuid_verified", "network_none",
        "root_filesystem_read_only", "capabilities_dropped", "no_new_privileges",
        "exact_cpu_limit_verified",
        "execution_started", "mounts", "runtime_receipt_sha256",
        "runtime_receipt_sha256_definition",
    }
    _require_exact_keys(value, expected_keys, "runtime inspect receipt")
    if value.get("schema_version") != 1:
        raise ContractError("runtime-receipt schema version mismatch")
    readiness_hash = _validate_self_hash(readiness, "readiness_sha256", "execution readiness")
    authorization_hash = validate_run_authorization(authorization, readiness)
    plan_hash = _validate_self_hash(
        authorized_plan, "authorized_plan_sha256", "authorized plan"
    )
    if value.get("protocol_id") != PROTOCOL_ID or value.get("status") != RUNTIME_RECEIPT_STATUS:
        raise ContractError("runtime receipt status/protocol mismatch")
    expected = {
        "readiness_sha256": readiness_hash,
        "run_authorization_sha256": authorization_hash,
        "authorized_plan_sha256": plan_hash,
        "authorization_id": authorization["authorization_id"],
        "image_id": readiness["image_id"],
        "cell_order_sha256": readiness["cell_order_sha256"],
        "remaining_scientific_walltime_hours": readiness[
            "remaining_scientific_walltime_hours"
        ],
        "cpu_limit_cores": readiness["cpu_limit_cores"],
        "maximum_cpu_core_hours": readiness["maximum_cpu_core_hours"],
    }
    if any(value.get(field) != item for field, item in expected.items()):
        raise ContractError("runtime receipt artifact chain mismatch")
    _require_hex(value.get("container_id"), 64, "container_id")
    _require_hex(value.get("runtime_inspect_sha256"), 64, "runtime_inspect_sha256")
    if value.get("container_name") != authorized_plan.get("container_name"):
        raise ContractError("runtime receipt container name mismatch")
    if value.get("mounts") != authorized_plan.get("mounts"):
        raise ContractError("runtime receipt mount scope mismatch")
    for field in (
        "container_created",
        "container_inspected_before_start",
        "exact_mounts_verified",
        "exact_gpu_uuid_verified",
        "network_none",
        "root_filesystem_read_only",
        "capabilities_dropped",
        "no_new_privileges",
        "exact_cpu_limit_verified",
    ):
        _require_bool(value.get(field), True, field)
    _require_bool(value.get("execution_started"), False, "execution_started")
    rebuilt = build_runtime_receipt_from_inspect(
        inspect_value,
        readiness=readiness,
        authorization=authorization,
        authorized_plan=authorized_plan,
        input_binding=input_binding,
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
) -> dict[str, Any]:
    """Validate a stopped scientific container before any ``docker start``.

    The caller must obtain ``inspect_value`` from the Docker daemon after
    ``docker create``.  This function has no Docker side effects.
    """

    validate_authorized_campaign_plan(
        authorized_plan, readiness, authorization, input_binding
    )
    if (
        not isinstance(inspect_value, list)
        or len(inspect_value) != 1
        or not isinstance(inspect_value[0], Mapping)
    ):
        raise ContractError("scientific docker inspect must contain exactly one container")
    inspect = inspect_value[0]
    host = inspect.get("HostConfig")
    config = inspect.get("Config")
    state = inspect.get("State")
    if not isinstance(host, Mapping) or not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ContractError("scientific docker inspect lacks Config/HostConfig/State")
    if str(inspect.get("Image", "")) != readiness.get("image_id"):
        raise ContractError("scientific container image ID mismatch")
    _require_hex(inspect.get("Id"), 64, "container_id")
    if state.get("Status") != "created" or state.get("Running") is not False:
        raise ContractError("scientific container must be inspected in stopped created state")
    if state.get("Dead") not in (None, False) or state.get("OOMKilled") not in (None, False):
        raise ContractError("scientific container state is not a clean created state")
    if host.get("NetworkMode") != "none" or host.get("ReadonlyRootfs") is not True:
        raise ContractError("scientific container network/root-filesystem drift")
    if host.get("Privileged") is not False or host.get("Devices") not in (None, []):
        raise ContractError("scientific container exposes unregistered host privilege/device")
    if host.get("AutoRemove") is not False:
        raise ContractError("scientific container must not auto-remove its inspect evidence")
    restart = host.get("RestartPolicy")
    if not isinstance(restart, Mapping) or restart.get("Name") not in ("", "no"):
        raise ContractError("scientific container restart policy drift")
    if int(restart.get("MaximumRetryCount", 0)) != 0:
        raise ContractError("scientific container retry count drift")
    for field in ("Links", "VolumesFrom"):
        if host.get(field) not in (None, []):
            raise ContractError(f"scientific container has unregistered {field}")
    if set(str(item) for item in (host.get("CapDrop") or [])) != {"ALL"}:
        raise ContractError("scientific container must drop all capabilities")
    security = [str(item) for item in (host.get("SecurityOpt") or [])]
    if not any(item.split(":", 1)[0] == "no-new-privileges" for item in security):
        raise ContractError("scientific container lacks no-new-privileges")
    if str(config.get("User", "")) != "1000:1000":
        raise ContractError("scientific container user mismatch")
    if config.get("Entrypoint") != ["python"]:
        raise ContractError("scientific container entrypoint mismatch")
    if config.get("Cmd") != readiness.get("container_command"):
        raise ContractError("scientific container command mismatch")
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
    environment = {str(item) for item in (config.get("Env") or [])}
    expected_source_env = f"FEDCORE_SOURCE_COMMIT={readiness['source_commit']}"
    if expected_source_env not in environment:
        raise ContractError("scientific container source-revision environment mismatch")

    requests = host.get("DeviceRequests")
    if not isinstance(requests, list) or len(requests) != 1 or not isinstance(requests[0], Mapping):
        raise ContractError("scientific container must request exactly one GPU device")
    request = requests[0]
    if request.get("DeviceIDs") != [EXPECTED_GPU_UUID]:
        raise ContractError("scientific container GPU UUID mismatch")
    capabilities = request.get("Capabilities")
    if capabilities != [["gpu"]]:
        raise ContractError("scientific container GPU capability request mismatch")
    if request.get("Count") not in (0, -1):
        raise ContractError("scientific container GPU request count drift")
    if request.get("Driver") not in (None, "", "nvidia"):
        raise ContractError("scientific container GPU driver request drift")
    if request.get("Options") not in (None, {}):
        raise ContractError("scientific container GPU request options drift")

    if host.get("NanoCpus") != DOCKER_NANO_CPUS:
        raise ContractError("scientific container CPU limit drift")

    tmpfs = host.get("Tmpfs")
    if not isinstance(tmpfs, Mapping) or set(tmpfs) != {"/tmp"}:
        raise ContractError("scientific container requires exactly one /tmp tmpfs")
    options = set(str(tmpfs["/tmp"]).split(","))
    size = {item for item in options if item.startswith("size=")}
    if options - size != {"rw", "noexec", "nosuid"} or size not in (
        {"size=1g"},
        {"size=1073741824"},
    ):
        raise ContractError("scientific /tmp tmpfs option drift")

    observed_mounts: list[dict[str, Any]] = []
    mounts = inspect.get("Mounts")
    if not isinstance(mounts, list):
        raise ContractError("scientific container mount list is absent")
    for row in mounts:
        if not isinstance(row, Mapping) or row.get("Type") != "bind":
            raise ContractError("scientific container has an unregistered non-bind mount")
        if str(row.get("Propagation", "rprivate")) not in ("", "rprivate"):
            raise ContractError("scientific bind mount propagation must be rprivate")
        observed_mounts.append(
            {
                "type": "bind",
                "source": str(row.get("Source", "")),
                "destination": str(row.get("Destination", "")),
                "read_only": not bool(row.get("RW")),
            }
        )
    observed_mounts.append(
        {
            "type": "tmpfs",
            "source": "",
            "destination": "/tmp",
            "read_only": False,
            "options": "rw,noexec,nosuid,size=1g",
        }
    )
    expected_mounts = expected_scientific_mounts(readiness, input_binding)
    observed_by_destination = {
        str(row["destination"]): row for row in observed_mounts
    }
    expected_by_destination = {
        str(row["destination"]): row for row in expected_mounts
    }
    if (
        len(observed_by_destination) != len(observed_mounts)
        or observed_by_destination != expected_by_destination
    ):
        raise ContractError("scientific container exact mount scope mismatch")

    readiness_hash = _validate_self_hash(readiness, "readiness_sha256", "execution readiness")
    authorization_hash = validate_run_authorization(authorization, readiness)
    plan_hash = _validate_self_hash(
        authorized_plan, "authorized_plan_sha256", "authorized plan"
    )
    body = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": RUNTIME_RECEIPT_STATUS,
        "readiness_sha256": readiness_hash,
        "run_authorization_sha256": authorization_hash,
        "authorized_plan_sha256": plan_hash,
        "authorization_id": authorization["authorization_id"],
        "image_id": readiness["image_id"],
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
        "mounts": expected_mounts,
    }
    if not body["container_id"] or body["container_name"] != authorized_plan["container_name"]:
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
    paths = {name: root / name for name in CONTROL_FILENAMES}
    for path in paths.values():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ContractError(f"authorization artifact is not a regular file: {path.name}")
        if stat.S_IMODE(metadata.st_mode) & 0o222:
            raise ContractError(f"authorization artifact is writable in container: {path.name}")

    loaded = {
        name: _read_control_bytes_once(
            path, require_object=name != "RUNTIME_INSPECT.json"
        )
        for name, path in paths.items()
    }
    static_plan, static_plan_file_hash = loaded["HOST_LAUNCH_PLAN.json"]
    input_binding, input_binding_file_hash = loaded["INPUT_BINDING.json"]
    mount_receipt, mount_receipt_file_hash = loaded["MOUNT_RECEIPT.json"]
    source_manifest = loaded["SOURCE_MANIFEST.json"][0]
    source_manifest_file_hash = loaded["SOURCE_MANIFEST.json"][1]
    source_probe, source_probe_file_hash = loaded["SOURCE_PROBE_RECEIPT.json"]
    test_report, test_report_file_hash = loaded["TEST_REPORT.json"]
    readiness = loaded["EXECUTION_READINESS.json"][0]
    authorization = loaded["RUN_AUTHORIZATION.json"][0]
    authorized_plan = loaded["AUTHORIZED_CAMPAIGN_PLAN.json"][0]
    runtime_receipt = loaded["RUNTIME_INSPECT_RECEIPT.json"][0]
    runtime_inspect, runtime_inspect_file_hash = loaded["RUNTIME_INSPECT.json"]

    if static_plan_file_hash != STATIC_LAUNCH_PLAN_FILE_SHA256:
        raise ContractError("mounted static launch-plan bytes differ from the public release")
    if input_binding_file_hash != STATIC_INPUT_BINDING_FILE_SHA256:
        raise ContractError("mounted input-binding bytes differ from the public release")
    if mount_receipt_file_hash != STATIC_MOUNT_RECEIPT_FILE_SHA256:
        raise ContractError("mounted mount-receipt bytes differ from the public release")
    readiness_hash = validate_execution_readiness(
        readiness,
        static_launch_plan=static_plan,
        input_binding=input_binding,
        mount_receipt=mount_receipt,
        expected_source_commit=expected_source_commit,
    )
    if source_manifest_file_hash != readiness.get("source_manifest_sha256"):
        raise ContractError("mounted source-manifest bytes differ from readiness")
    validate_scientific_source_manifest(
        source_manifest, expected_source_commit=str(readiness["source_commit"])
    )
    if source_manifest.get("runtime_inventory_sha256") != readiness.get(
        "runtime_inventory_sha256"
    ) or source_manifest.get("build_inventory_sha256") != readiness.get(
        "build_inventory_sha256"
    ):
        raise ContractError("mounted source-manifest inventory differs from readiness")
    if source_probe_file_hash != readiness.get("source_probe_receipt_sha256"):
        raise ContractError("mounted source-probe receipt bytes differ from readiness")
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
    if test_report_file_hash != readiness.get("test_report_sha256"):
        raise ContractError("mounted test-report bytes differ from readiness")
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
    authorization_hash = validate_run_authorization(authorization, readiness)
    authorized_plan_hash = validate_authorized_campaign_plan(
        authorized_plan, readiness, authorization, input_binding
    )
    runtime_hash = validate_runtime_receipt(
        runtime_receipt,
        readiness=readiness,
        authorization=authorization,
        authorized_plan=authorized_plan,
        inspect_value=runtime_inspect,
        input_binding=input_binding,
    )
    if runtime_inspect_file_hash != runtime_receipt.get("runtime_inspect_sha256"):
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
    "STATIC_BINDING_ROOT_SHA256",
    "STATIC_INPUT_BINDING_FILE_SHA256",
    "STATIC_LAUNCH_PLAN_FILE_SHA256",
    "STATIC_LAUNCH_PLAN_SHA256",
    "STATIC_MOUNT_RECEIPT_FILE_SHA256",
    "VerifiedExecutionAuthorization",
    "acquire_exclusive_lease",
    "build_authorized_campaign_plan",
    "build_execution_readiness",
    "build_run_authorization_template",
    "build_runtime_receipt_from_inspect",
    "canonical_runtime_inspect_comparison_bytes",
    "build_scientific_source_manifest",
    "build_source_probe_receipt",
    "expected_scientific_mounts",
    "validate_authorized_campaign_plan",
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
    "validate_static_launch_plan",
    "validate_static_mount_receipt",
]
