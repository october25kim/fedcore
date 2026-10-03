"""Trusted-host executor for one artifact-authorized PACS v3 campaign.

``prepare`` and ``start`` are deliberately separate operations.  Preparation
creates a stopped container, validates its exact Docker inspect, and writes the
raw inspect plus its derived receipt with exclusive creation.  Starting later
re-inspects the same stopped container and refuses any drift before invoking
``docker start --attach``.  This module never issues RUN_AUTHORIZATION.json.

The only ``docker create`` argument vector executed is the authorized plan's,
after it has been re-rendered from the validated template and authorization
and compared byte for byte.  Host root and the Docker daemon are the trusted
computing base; this module does not defend against either.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from fedcore.experiments.v3_contract import (
    PROTOCOL_ID,
    ContractError,
    canonical_json_bytes,
    sha256_file,
)
from fedcore.experiments.v3_execution_binding import (
    CPU_LIMIT_CORES,
    CONTROL_FILENAMES,
    EXPECTED_HOST_CONTROL_DIR,
    EXPECTED_HOST_LOG_DIR,
    EXPECTED_HOST_OUTPUT_DIR,
    POST_TIMEOUT_CLEANUP_COMMAND_BOUND_SECONDS,
    PRE_READINESS_FILENAMES,
    REMAINING_SCIENTIFIC_WALLTIME_HOURS,
    SEALED_CAPACITY_GPU_HOURS,
    SCIENTIFIC_CLEANUP_RESERVE_SECONDS,
    SCIENTIFIC_CLEANUP_SAFETY_MARGIN_SECONDS,
    SCIENTIFIC_EXECUTION_TIMEOUT_SECONDS,
    STATIC_INPUT_BINDING_FILE_SHA256,
    TOTAL_GPU_HOUR_CAP,
    ControlArtifact,
    acquire_exclusive_lease,
    build_authorized_campaign_plan,
    build_runtime_receipt_from_inspect,
    canonical_runtime_inspect_comparison_bytes,
    read_control_artifact,
    recheck_control_artifact,
    validate_authorized_campaign_plan,
    validate_execution_readiness,
    validate_run_authorization,
    validate_runtime_receipt,
)


PREPARE_FILENAMES = tuple(
    name
    for name in CONTROL_FILENAMES
    if name not in {"RUNTIME_INSPECT.json", "RUNTIME_INSPECT_RECEIPT.json"}
)
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
SCIENTIFIC_WALLTIME_SECONDS = SCIENTIFIC_EXECUTION_TIMEOUT_SECONDS
SCIENTIFIC_CPU_LIMIT = CPU_LIMIT_CORES
DOCKER_STOP_GRACE_SECONDS = 30
DOCKER_STOP_COMMAND_TIMEOUT_SECONDS = 45
DOCKER_KILL_COMMAND_TIMEOUT_SECONDS = 15
DOCKER_INSPECT_COMMAND_TIMEOUT_SECONDS = 15


def _validate_watchdog_gpu_cap(timeout_seconds: int) -> None:
    """Reject a watchdog budget that exceeds the amended safety envelope."""

    cleanup_command_bound = (
        DOCKER_STOP_COMMAND_TIMEOUT_SECONDS
        + DOCKER_INSPECT_COMMAND_TIMEOUT_SECONDS
        + DOCKER_KILL_COMMAND_TIMEOUT_SECONDS
        + DOCKER_INSPECT_COMMAND_TIMEOUT_SECONDS
    )
    if cleanup_command_bound != POST_TIMEOUT_CLEANUP_COMMAND_BOUND_SECONDS:
        raise ContractError("host cleanup command bound differs from the sealed contract")
    if SCIENTIFIC_CLEANUP_RESERVE_SECONDS < cleanup_command_bound:
        raise ContractError("host cleanup reserve is smaller than its command bound")
    if SCIENTIFIC_CLEANUP_SAFETY_MARGIN_SECONDS != (
        SCIENTIFIC_CLEANUP_RESERVE_SECONDS - cleanup_command_bound
    ):
        raise ContractError("host cleanup safety margin drift")
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool):
        raise ContractError(
            "scientific watchdog timeout must be an integer number of seconds"
        )
    if timeout_seconds <= 0 or timeout_seconds > SCIENTIFIC_EXECUTION_TIMEOUT_SECONDS:
        raise ContractError("scientific watchdog timeout exceeds the cleanup-reserved cap")
    maximum_accounted_seconds = (
        int(SEALED_CAPACITY_GPU_HOURS * 60 * 60)
        + timeout_seconds
        + SCIENTIFIC_CLEANUP_RESERVE_SECONDS
    )
    if maximum_accounted_seconds > int(TOTAL_GPU_HOUR_CAP * 60 * 60):
        raise ContractError("watchdog plus cleanup reserve exceeds the total GPU cap")


@dataclass(frozen=True)
class HostArtifacts:
    static_plan: dict[str, Any]
    input_binding: dict[str, Any]
    mount_receipt: dict[str, Any]
    readiness: dict[str, Any]
    authorization: dict[str, Any]
    authorized_plan: dict[str, Any]
    files: Mapping[str, ControlArtifact]


def _require_directory(path: Path, expected: Path, label: str, *, empty: bool = False) -> Path:
    value = Path(path)
    try:
        metadata = value.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"{label} is absent: {value}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ContractError(f"{label} must be a real nonsymlink directory")
    resolved = value.resolve()
    if resolved != expected:
        raise ContractError(f"{label} differs from the registered path")
    if empty and next(resolved.iterdir(), None) is not None:
        raise ContractError(f"{label} must be empty")
    return resolved


def _load_host_artifacts(control_dir: Path, *, prepared: bool) -> HostArtifacts:
    control = _require_directory(
        control_dir, EXPECTED_HOST_CONTROL_DIR, "host control directory"
    )
    expected_names = CONTROL_FILENAMES if prepared else PREPARE_FILENAMES
    names = tuple(sorted(item.name for item in control.iterdir()))
    if names != tuple(sorted(expected_names)):
        raise ContractError(f"host control directory content drift: {names!r}")
    files = {
        name: read_control_artifact(control / name, require_read_only=True)
        for name in PREPARE_FILENAMES
    }
    values = {name: artifact.value for name, artifact in files.items()}
    if files["INPUT_BINDING.json"].sha256 != STATIC_INPUT_BINDING_FILE_SHA256:
        raise ContractError("host input-binding bytes differ from the static anchor")
    readiness = values["EXECUTION_READINESS.json"]
    if not isinstance(readiness, dict):
        raise ContractError("host readiness must be an object")
    for name, field, label in (
        ("SOURCE_MANIFEST.json", "source_manifest_sha256", "source-manifest"),
        ("SOURCE_PROBE_RECEIPT.json", "source_probe_receipt_sha256", "source-probe receipt"),
        ("TEST_REPORT.json", "test_report_sha256", "test-report"),
    ):
        if files[name].sha256 != readiness.get(field):
            raise ContractError(f"host {label} bytes differ from readiness")
    validate_execution_readiness(
        readiness,
        static_launch_plan=values["HOST_LAUNCH_PLAN.json"],
        input_binding=values["INPUT_BINDING.json"],
        mount_receipt=values["MOUNT_RECEIPT.json"],
        artifact_sha256={name: files[name].sha256 for name in PRE_READINESS_FILENAMES},
    )
    validate_run_authorization(
        values["RUN_AUTHORIZATION.json"], readiness, values["HOST_LAUNCH_PLAN.json"]
    )
    validate_authorized_campaign_plan(
        values["AUTHORIZED_CAMPAIGN_PLAN.json"],
        readiness,
        values["RUN_AUTHORIZATION.json"],
        values["INPUT_BINDING.json"],
        values["HOST_LAUNCH_PLAN.json"],
    )
    rendered = canonical_json_bytes(
        build_authorized_campaign_plan(
            readiness,
            values["RUN_AUTHORIZATION.json"],
            input_binding=values["INPUT_BINDING.json"],
            launch_plan=values["HOST_LAUNCH_PLAN.json"],
        )
    )
    if files["AUTHORIZED_CAMPAIGN_PLAN.json"].payload != rendered:
        raise ContractError("host authorized-plan bytes differ from the deterministic render")
    return HostArtifacts(
        static_plan=values["HOST_LAUNCH_PLAN.json"],
        input_binding=values["INPUT_BINDING.json"],
        mount_receipt=values["MOUNT_RECEIPT.json"],
        readiness=readiness,
        authorization=values["RUN_AUTHORIZATION.json"],
        authorized_plan=values["AUTHORIZED_CAMPAIGN_PLAN.json"],
        files=files,
    )


def _recheck_control_files(artifacts: HostArtifacts) -> None:
    """Refuse any device/inode/size/mtime or byte drift since validation."""

    if set(artifacts.files) != set(PREPARE_FILENAMES):
        raise ContractError("host control identity record is incomplete")
    for artifact in artifacts.files.values():
        recheck_control_artifact(artifact)


def _run(runner: CommandRunner, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(
            list(argv),
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"host command failed: {list(argv)!r}") from exc
    return result


def _write_exclusive_json(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise ContractError(f"host evidence already exists: {path.name}") from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ContractError(f"short write while sealing {path.name}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _bounded_termination_attempt(
    runner: CommandRunner,
    argv: Sequence[str],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Attempt one bounded cleanup command and return evidence, never raise."""

    outcome: dict[str, Any] = {
        "attempted": True,
        "argv": list(argv),
        "timeout_seconds": timeout_seconds,
        "returncode": None,
        "failure_type": None,
        "failure_message": None,
    }
    try:
        result = runner(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_seconds,
        )
        outcome["returncode"] = int(result.returncode)
    except Exception as exc:
        outcome["failure_type"] = type(exc).__name__
        outcome["failure_message"] = str(exc)
    return outcome


def _bounded_running_state_inspect(
    runner: CommandRunner,
    container_id: str,
) -> dict[str, Any]:
    """Return a bounded post-cleanup running-state observation, never raise."""

    argv = ["docker", "inspect", "--format", "{{.State.Running}}", container_id]
    outcome: dict[str, Any] = {
        "attempted": True,
        "argv": argv,
        "timeout_seconds": DOCKER_INSPECT_COMMAND_TIMEOUT_SECONDS,
        "returncode": None,
        "running": None,
        "failure_type": None,
        "failure_message": None,
    }
    try:
        result = runner(
            argv,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=DOCKER_INSPECT_COMMAND_TIMEOUT_SECONDS,
        )
        outcome["returncode"] = int(result.returncode)
        if result.returncode == 0:
            rendered = str(result.stdout).strip().lower()
            if rendered == "true":
                outcome["running"] = True
            elif rendered == "false":
                outcome["running"] = False
            else:
                outcome["failure_type"] = "UnexpectedInspectOutput"
                outcome["failure_message"] = rendered
    except Exception as exc:
        outcome["failure_type"] = type(exc).__name__
        outcome["failure_message"] = str(exc)
    return outcome


def _run_attached_with_watchdog(
    runner: CommandRunner,
    argv: Sequence[str],
    *,
    container_id: str,
    log_dir: Path,
    timeout_seconds: int = SCIENTIFIC_WALLTIME_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Stream a multi-day run, seal every terminal path, and enforce the cap.

    A process-launch I/O failure is re-raised as ``ContractError`` only after
    both log files and a ``HOLD_INCOMPLETE`` terminal receipt are durable.
    """

    _validate_watchdog_gpu_cap(timeout_seconds)
    destination = Path(log_dir)
    if destination.exists():
        raise ContractError("scientific host log directory already exists")
    destination.mkdir(parents=True, mode=0o755)
    stdout_path = destination / "container.stdout.log"
    stderr_path = destination / "container.stderr.log"
    receipt_path = destination / "HOST_TERMINAL_RECEIPT.json"
    started_monotonic = time.monotonic()
    status = "HOLD_INCOMPLETE"
    returncode: int | None = None
    timed_out = False
    launch_exception: BaseException | None = None
    stop_outcome: dict[str, Any] = {
        "attempted": False,
        "argv": None,
        "timeout_seconds": None,
        "returncode": None,
        "failure_type": None,
        "failure_message": None,
    }
    kill_outcome = dict(stop_outcome)
    post_stop_inspect_outcome: dict[str, Any] = {
        "attempted": False,
        "argv": None,
        "timeout_seconds": None,
        "returncode": None,
        "running": None,
        "failure_type": None,
        "failure_message": None,
    }
    post_kill_inspect_outcome = dict(post_stop_inspect_outcome)
    termination_command_succeeded = False
    container_confirmed_stopped = False
    with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
        try:
            result = runner(
                list(argv),
                check=False,
                stdout=stdout_handle,
                stderr=stderr_handle,
                shell=False,
                timeout=timeout_seconds,
            )
            returncode = int(result.returncode)
            status = "PROCESS_EXITED_ZERO" if returncode == 0 else "HOLD_INCOMPLETE"
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_outcome = _bounded_termination_attempt(
                runner,
                [
                    "docker",
                    "stop",
                    "--time",
                    str(DOCKER_STOP_GRACE_SECONDS),
                    container_id,
                ],
                timeout_seconds=DOCKER_STOP_COMMAND_TIMEOUT_SECONDS,
            )
            termination_command_succeeded = stop_outcome["returncode"] == 0
            post_stop_inspect_outcome = _bounded_running_state_inspect(
                runner, container_id
            )
            container_confirmed_stopped = (
                post_stop_inspect_outcome["returncode"] == 0
                and post_stop_inspect_outcome["running"] is False
            )
            if not container_confirmed_stopped:
                kill_outcome = _bounded_termination_attempt(
                    runner,
                    ["docker", "kill", container_id],
                    timeout_seconds=DOCKER_KILL_COMMAND_TIMEOUT_SECONDS,
                )
                termination_command_succeeded = (
                    termination_command_succeeded or kill_outcome["returncode"] == 0
                )
                post_kill_inspect_outcome = _bounded_running_state_inspect(
                    runner, container_id
                )
                container_confirmed_stopped = (
                    post_kill_inspect_outcome["returncode"] == 0
                    and post_kill_inspect_outcome["running"] is False
                )
            result = subprocess.CompletedProcess(list(argv), 124, "", "")
            returncode = 124
            status = (
                "HOLD_INCOMPLETE_RESOURCE_CAP"
                if container_confirmed_stopped
                else "HOLD_INCOMPLETE_RESOURCE_CAP_CLEANUP_UNCONFIRMED"
            )
        except (OSError, subprocess.SubprocessError) as exc:
            launch_exception = exc
            status = "HOLD_INCOMPLETE"
            returncode = None
        finally:
            for handle in (stdout_handle, stderr_handle):
                handle.flush()
                os.fsync(handle.fileno())
    elapsed_seconds = time.monotonic() - started_monotonic
    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "container_id": container_id,
        "returncode": returncode,
        "timed_out": timed_out,
        "failure_type": (
            type(launch_exception).__name__ if launch_exception is not None else None
        ),
        "failure_message": (
            str(launch_exception) if launch_exception is not None else None
        ),
        "stop_outcome": stop_outcome,
        "post_stop_inspect_outcome": post_stop_inspect_outcome,
        "kill_outcome": kill_outcome,
        "post_kill_inspect_outcome": post_kill_inspect_outcome,
        "termination_command_succeeded": termination_command_succeeded,
        "container_confirmed_stopped": container_confirmed_stopped,
        "walltime_seconds": elapsed_seconds,
        "walltime_cap_seconds": timeout_seconds,
        "gpu_hour_cap_total": TOTAL_GPU_HOUR_CAP,
        "capacity_stage_gpu_hours_already_used": SEALED_CAPACITY_GPU_HOURS,
        "scientific_gpu_hour_cap": REMAINING_SCIENTIFIC_WALLTIME_HOURS,
        "scientific_execution_timeout_seconds": timeout_seconds,
        "post_timeout_cleanup_command_bound_seconds": (
            POST_TIMEOUT_CLEANUP_COMMAND_BOUND_SECONDS
        ),
        "scientific_cleanup_reserve_seconds": SCIENTIFIC_CLEANUP_RESERVE_SECONDS,
        "scientific_cleanup_safety_margin_seconds": (
            SCIENTIFIC_CLEANUP_SAFETY_MARGIN_SECONDS
        ),
        "maximum_accounted_gpu_seconds": (
            int(SEALED_CAPACITY_GPU_HOURS * 60 * 60)
            + timeout_seconds
            + SCIENTIFIC_CLEANUP_RESERVE_SECONDS
        ),
        "gpu_cap_invariant_verified": True,
        "cpu_core_limit": SCIENTIFIC_CPU_LIMIT,
        "maximum_scientific_cpu_core_hours": SCIENTIFIC_CPU_LIMIT
        * timeout_seconds
        / 3600,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    _write_exclusive_json(receipt_path, receipt)
    if launch_exception is not None:
        raise ContractError(
            "scientific container process launch failed after terminal receipt sealing"
        ) from launch_exception
    if timed_out:
        raise ContractError("scientific execution exhausted the registered resource cap")
    if returncode != 0:
        raise ContractError(f"scientific container exited with status {returncode}")
    return result


def prepare_scientific_container(
    *,
    control_dir: Path = EXPECTED_HOST_CONTROL_DIR,
    output_dir: Path = EXPECTED_HOST_OUTPUT_DIR,
    lease_path: Path | None = None,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Create and inspect a stopped container; never start it."""

    artifacts = _load_host_artifacts(control_dir, prepared=False)
    output = _require_directory(
        output_dir, EXPECTED_HOST_OUTPUT_DIR, "scientific output directory", empty=True
    )
    control = Path(control_dir).resolve()
    lease = (
        control.parent / "HOST_PREPARE_LEASE.json"
        if lease_path is None
        else Path(lease_path)
    )
    if lease.parent.resolve() == output or output in lease.parents:
        raise ContractError("host prepare lease must live outside scientific output")
    acquire_exclusive_lease(
        lease,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "stage": "docker_create_and_inspect",
            "run_authorization_sha256": artifacts.authorization["authorization_sha256"],
            "authorized_plan_sha256": artifacts.authorized_plan["authorized_plan_sha256"],
        },
    )
    # The executor runs only the validated, re-rendered authorized plan argv.
    _recheck_control_files(artifacts)
    created = _run(runner, artifacts.authorized_plan["docker_create_argv"])
    container_id = created.stdout.strip()
    if len(container_id) != 64 or any(c not in "0123456789abcdef" for c in container_id):
        raise ContractError("docker create did not return one immutable container ID")
    inspected = _run(runner, ["docker", "inspect", container_id])
    try:
        inspect_value = json.loads(inspected.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError("docker inspect did not return valid JSON") from exc
    receipt = build_runtime_receipt_from_inspect(
        inspect_value,
        readiness=artifacts.readiness,
        authorization=artifacts.authorization,
        authorized_plan=artifacts.authorized_plan,
        input_binding=artifacts.input_binding,
        launch_plan=artifacts.static_plan,
    )
    if receipt["container_id"] != container_id:
        raise ContractError("docker create and inspect container IDs disagree")
    _write_exclusive_json(control / "RUNTIME_INSPECT.json", inspect_value)
    _write_exclusive_json(control / "RUNTIME_INSPECT_RECEIPT.json", receipt)
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_CREATED_CONTAINER_INSPECTED_HOLD_START",
        "container_id": container_id,
        "runtime_receipt_sha256": receipt["runtime_receipt_sha256"],
        "execution_started": False,
    }


def start_scientific_container(
    *,
    control_dir: Path = EXPECTED_HOST_CONTROL_DIR,
    output_dir: Path = EXPECTED_HOST_OUTPUT_DIR,
    lease_path: Path | None = None,
    log_dir: Path = EXPECTED_HOST_LOG_DIR,
    runner: CommandRunner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    """Revalidate the prepared container, then start that exact container ID."""

    artifacts = _load_host_artifacts(control_dir, prepared=True)
    _require_directory(
        output_dir, EXPECTED_HOST_OUTPUT_DIR, "scientific output directory", empty=True
    )
    control = Path(control_dir).resolve()
    raw_inspect_artifact = read_control_artifact(
        control / "RUNTIME_INSPECT.json", require_object=False, require_read_only=True
    )
    receipt_artifact = read_control_artifact(
        control / "RUNTIME_INSPECT_RECEIPT.json", require_read_only=True
    )
    raw_inspect = raw_inspect_artifact.value
    receipt = receipt_artifact.value
    validate_runtime_receipt(
        receipt,
        readiness=artifacts.readiness,
        authorization=artifacts.authorization,
        authorized_plan=artifacts.authorized_plan,
        inspect_value=raw_inspect,
        input_binding=artifacts.input_binding,
        launch_plan=artifacts.static_plan,
    )
    if raw_inspect_artifact.sha256 != receipt.get("runtime_inspect_sha256"):
        raise ContractError("host raw inspect bytes differ from the runtime receipt")
    container_id = str(receipt["container_id"])
    fresh_inspect_result = _run(runner, ["docker", "inspect", container_id])
    try:
        fresh_inspect = json.loads(fresh_inspect_result.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError("fresh docker inspect did not return valid JSON") from exc
    build_runtime_receipt_from_inspect(
        fresh_inspect,
        readiness=artifacts.readiness,
        authorization=artifacts.authorization,
        authorized_plan=artifacts.authorized_plan,
        input_binding=artifacts.input_binding,
        launch_plan=artifacts.static_plan,
    )
    _validate_stopped_container_unchanged(raw_inspect, fresh_inspect)
    _recheck_control_files(artifacts)
    for artifact in (raw_inspect_artifact, receipt_artifact):
        recheck_control_artifact(artifact)
    lease = (
        control.parent / "HOST_START_LEASE.json" if lease_path is None else Path(lease_path)
    )
    acquire_exclusive_lease(
        lease,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "stage": "docker_start_attach",
            "container_id": container_id,
            "runtime_receipt_sha256": receipt["runtime_receipt_sha256"],
        },
    )
    start_argv = [
        container_id if item == "__CONTAINER_ID__" else str(item)
        for item in artifacts.authorized_plan["docker_start_argv_template"]
    ]
    return _run_attached_with_watchdog(
        runner,
        start_argv,
        container_id=container_id,
        log_dir=log_dir,
    )


def _validate_stopped_container_unchanged(
    prepared_inspect: Any, fresh_inspect: Any
) -> None:
    """Reject all post-prepare drift except top-level mount-row ordering."""

    if canonical_runtime_inspect_comparison_bytes(
        fresh_inspect
    ) != canonical_runtime_inspect_comparison_bytes(prepared_inspect):
        raise ContractError("stopped container drifted after preparation")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "start"))
    parser.add_argument("--control-dir", type=Path, default=EXPECTED_HOST_CONTROL_DIR)
    parser.add_argument("--output-dir", type=Path, default=EXPECTED_HOST_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result: Any = prepare_scientific_container(
            control_dir=args.control_dir, output_dir=args.output_dir
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        start_scientific_container(
            control_dir=args.control_dir, output_dir=args.output_dir
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_HOST_LOG_DIR",
    "DOCKER_KILL_COMMAND_TIMEOUT_SECONDS",
    "DOCKER_INSPECT_COMMAND_TIMEOUT_SECONDS",
    "DOCKER_STOP_COMMAND_TIMEOUT_SECONDS",
    "DOCKER_STOP_GRACE_SECONDS",
    "HostArtifacts",
    "PREPARE_FILENAMES",
    "SCIENTIFIC_CPU_LIMIT",
    "SCIENTIFIC_WALLTIME_SECONDS",
    "main",
    "prepare_scientific_container",
    "start_scientific_container",
]
