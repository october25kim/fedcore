"""Normalize and validate a no-GPU Docker mount probe for FedCORE v3.

The probe is deliberately pre-scientific.  It verifies container isolation and
the exact five read-only input mounts from ``docker inspect``.  It neither
requests a GPU nor opens any mounted file.  A later execution authorization
must separately bind the registered GPU device request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fedcore.experiments.v3_contract import PROTOCOL_ID, ContractError, read_json
from fedcore.experiments.v3_input_binding import (
    REGISTERED_SCIENTIFIC_INPUTS,
    ScientificInputSpec,
)
from fedcore.experiments.v3_scientific_runner import (
    REGISTERED_OUTPUT_DIR,
    exact_mount_plan,
    validate_launch_plan_integrity,
)
from fedcore.experiments.v3_validate import validate_mount_scope


FORBIDDEN_RUNTIME_ROOTS = (
    "/home/sanghoon/Desktop/Workspace/Fedcore",
    "/home/sanghoon/Desktop/Workspace/Fedcore/outputs",
    "/home/sanghoon/Desktop/Workspace/Fedcore/work",
)

MOUNT_PROBE_ENTRYPOINT = ["python"]
MOUNT_PROBE_COMMAND = ["-m", "fedcore.experiments.v3_mount_probe"]
MOUNT_PROBE_LOG = (
    "PASS_INPUTS_READ_ONLY\n"
    "PASS_ROOT_READ_ONLY\n"
    "PASS_TMP_RW\n"
    "PASS_OUTPUT_RW\n"
)


def _one_inspect(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
        raise ContractError("docker inspect receipt must contain exactly one container")
    return value[0]


def _observed_mounts(inspect: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mounts = inspect.get("Mounts")
    if not isinstance(mounts, list):
        raise ContractError("docker inspect Mounts is absent")
    for row in mounts:
        if not isinstance(row, Mapping) or row.get("Type") != "bind":
            raise ContractError("only bind mounts are permitted outside /tmp")
        rows.append(
            {
                "type": "bind",
                "source": str(row.get("Source", "")),
                "destination": str(row.get("Destination", "")),
                "read_only": not bool(row.get("RW")),
            }
        )
    tmpfs = inspect.get("HostConfig", {}).get("Tmpfs", {})
    if not isinstance(tmpfs, Mapping) or set(tmpfs) != {"/tmp"}:
        raise ContractError("exactly one /tmp tmpfs mount is required")
    rows.append(
        {
            "type": "tmpfs",
            "source": "",
            "destination": "/tmp",
            "read_only": False,
            "options": str(tmpfs["/tmp"]),
        }
    )
    return rows


def validate_no_gpu_mount_probe(
    inspect_value: Any,
    launch_plan: Mapping[str, Any],
    *,
    expected_image_id: str,
    probe_log: str,
    registered_specs: Sequence[ScientificInputSpec] = REGISTERED_SCIENTIFIC_INPUTS,
    expected_output_dir: Path = REGISTERED_OUTPUT_DIR,
) -> dict[str, Any]:
    """Validate Docker isolation and emit a deterministic receipt body."""

    validate_launch_plan_integrity(launch_plan)
    inspect = _one_inspect(inspect_value)
    if launch_plan.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("launch-plan protocol mismatch")
    if launch_plan.get("image", {}).get("id") != expected_image_id:
        raise ContractError("launch-plan image ID mismatch")
    observed_image = str(inspect.get("Image", ""))
    if observed_image != expected_image_id:
        raise ContractError("probe container image ID mismatch")
    host = inspect.get("HostConfig", {})
    config = inspect.get("Config", {})
    state = inspect.get("State", {})
    if launch_plan.get("status") != (
        "PASS_CANDIDATE_HOST_LAUNCH_PLAN_HOLD_EXECUTION_READINESS"
    ) or launch_plan.get("execution_allowed") is not False:
        raise ContractError("mount probe requires the non-executing candidate plan")
    inventory = launch_plan.get("input_inventory")
    if not isinstance(inventory, list) or len(inventory) != 5:
        raise ContractError("mount probe requires exactly five registered inputs")
    if len(registered_specs) != 5:
        raise ContractError("registered scientific input denominator drift")
    for spec, record in zip(registered_specs, inventory, strict=True):
        expected = {
            "name": spec.name,
            "provenance_source": spec.source_path,
            "destination": spec.container_path,
            "sha256": spec.sha256,
            "bytes": spec.size_bytes,
            "read_only": True,
            "symlink": False,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ContractError(f"registered input drift for {spec.name}: {key}")
    if Path(str(launch_plan.get("output_directory"))).resolve() != Path(
        expected_output_dir
    ).resolve():
        raise ContractError("registered scientific output path drift")
    if host.get("NetworkMode") != "none":
        raise ContractError("probe network mode must be none")
    if host.get("ReadonlyRootfs") is not True:
        raise ContractError("probe root filesystem must be read-only")
    if host.get("Privileged") is not False:
        raise ContractError("probe container must not be privileged")
    if host.get("Devices") not in (None, []):
        raise ContractError("probe must not expose host devices")
    if str(host.get("Runtime", "")) not in ("", "runc"):
        raise ContractError("probe must use the ordinary non-GPU runtime")
    if str(config.get("User", "")) != "1000:1000":
        raise ContractError("probe container user mismatch")
    if set(host.get("CapDrop") or []) != {"ALL"}:
        raise ContractError("probe must drop all capabilities")
    security_opt = [str(item) for item in (host.get("SecurityOpt") or [])]
    if not any(item.split(":", 1)[0] == "no-new-privileges" for item in security_opt):
        raise ContractError("probe lacks no-new-privileges")
    if host.get("DeviceRequests") not in (None, []):
        raise ContractError("pre-scientific mount probe must not request a GPU")
    environment = {str(item) for item in (config.get("Env") or [])}
    if "NVIDIA_VISIBLE_DEVICES=void" not in environment:
        raise ContractError("probe must explicitly disable NVIDIA device visibility")
    if config.get("Entrypoint") != MOUNT_PROBE_ENTRYPOINT:
        raise ContractError("probe entrypoint drift")
    if config.get("Cmd") != MOUNT_PROBE_COMMAND:
        raise ContractError("probe command drift")
    if state.get("Status") != "exited" or state.get("ExitCode") != 0:
        raise ContractError("mount probe did not exit successfully")
    if state.get("Running") is not False or state.get("OOMKilled") is not False:
        raise ContractError("mount probe state is not a clean completed state")
    if probe_log != MOUNT_PROBE_LOG:
        raise ContractError("mount probe output drift")

    observed = _observed_mounts(inspect)
    tmpfs_options = set(str(host["Tmpfs"]["/tmp"]).split(","))
    size_tokens = {item for item in tmpfs_options if item.startswith("size=")}
    if tmpfs_options - size_tokens != {"rw", "noexec", "nosuid"}:
        raise ContractError("/tmp tmpfs option drift")
    if size_tokens not in ({"size=1g"}, {"size=1073741824"}):
        raise ContractError("/tmp tmpfs size drift")
    expected = exact_mount_plan(
        launch_plan["input_inventory"], Path(str(launch_plan["output_directory"]))
    )
    validate_mount_scope(observed, expected, forbidden_roots=FORBIDDEN_RUNTIME_ROOTS)
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_EXACT_FIVE_RO_MOUNT_PROBE_NO_GPU",
        "launch_plan_sha256": launch_plan["launch_plan_sha256"],
        "input_binding_sha256": launch_plan["input_binding_sha256"],
        "container_id": str(inspect.get("Id", "")),
        "container_name": str(inspect.get("Name", "")).lstrip("/"),
        "image_id": observed_image,
        "network_mode": "none",
        "read_only_root_filesystem": True,
        "user": "1000:1000",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
        "device_requests": None,
        "mounts": observed,
        "exact_five_read_only_inputs": True,
        "one_read_write_output": True,
        "tmpfs_only_at_tmp": True,
        "legacy_workspace_mounted": False,
        "PACS_records_decoded": False,
        "PACS_labels_inspected": False,
        "GPU_used": False,
        "training_started": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-inspect", type=Path, required=True)
    parser.add_argument("--launch-plan", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--probe-log", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with args.docker_inspect.open("r", encoding="utf-8") as handle:
        inspect_value = json.load(handle)
    receipt = validate_no_gpu_mount_probe(
        inspect_value,
        read_json(args.launch_plan),
        expected_image_id=args.image_id,
        probe_log=args.probe_log.read_text(encoding="utf-8"),
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORBIDDEN_RUNTIME_ROOTS",
    "MOUNT_PROBE_COMMAND",
    "MOUNT_PROBE_ENTRYPOINT",
    "MOUNT_PROBE_LOG",
    "main",
    "validate_no_gpu_mount_probe",
]
