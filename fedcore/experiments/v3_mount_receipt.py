"""Normalize and validate a no-GPU Docker mount probe for FedCORE v3.

The probe is deliberately pre-scientific.  Its receipt is a pre-execution
mount declaration (``PREEXECUTION_MOUNT_DECLARATION_NOT_RUNTIME_PROOF``), not
evidence of the scientific run: the raw runtime ``docker inspect`` of the
created scientific container is the only authority for actual mounts.

The declaration covers the five read-only inputs, the read-only
``/authorization`` control mount, the read-write output, and ``/tmp``.  For
``/authorization`` it claims only that ``docker inspect`` reported ``RW=false``;
the probe module does not attempt a write there.  The probe neither requests a
GPU nor opens any mounted file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fedcore.experiments.v3_contract import PROTOCOL_ID, ContractError, read_json
from fedcore.experiments.v3_execution_binding import (
    CONTAINER_AUTHORIZATION_DIR,
    CONTROL_BINDING_MODE,
    CONTROL_SCHEMA_VERSION,
    EXPECTED_HOST_CONTROL_DIR,
    MOUNT_DECLARATION_ROLE,
    MOUNT_DECLARATION_STATUS,
    MOUNT_RECEIPT_KEYS,
    canonical_mounts,
)
from fedcore.experiments.v3_input_binding import (
    REGISTERED_SCIENTIFIC_INPUTS,
    ScientificInputSpec,
)
from fedcore.experiments.v3_scientific_runner import (
    REGISTERED_OUTPUT_DIR,
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
    expected_control_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate Docker isolation and emit the deterministic mount declaration."""

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
    if launch_plan.get("execution_allowed") is not False:
        raise ContractError("mount probe requires the non-executing pre-authorization template")
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
    template = launch_plan["execution_template"]
    control = Path(EXPECTED_HOST_CONTROL_DIR if expected_control_dir is None else expected_control_dir)
    if template["paths"]["control"] != str(control):
        raise ContractError("registered control path drift")
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
    declared = canonical_mounts(template["mounts"])
    validate_mount_scope(observed, declared, forbidden_roots=FORBIDDEN_RUNTIME_ROOTS)
    authorization_rows = [
        row for row in declared if row["destination"] == str(CONTAINER_AUTHORIZATION_DIR)
    ]
    if len(authorization_rows) != 1 or authorization_rows[0]["read_only"] is not True:
        raise ContractError("declaration requires one read-only /authorization mount")
    container_id = str(inspect.get("Id", ""))
    if len(container_id) != 64 or any(ch not in "0123456789abcdef" for ch in container_id):
        raise ContractError("probe container ID must be one immutable 64-hex ID")
    receipt = {
        "schema_version": 2,
        "control_schema_version": CONTROL_SCHEMA_VERSION,
        "control_binding_mode": CONTROL_BINDING_MODE,
        "protocol_id": PROTOCOL_ID,
        "status": MOUNT_DECLARATION_STATUS,
        "evidence_role": MOUNT_DECLARATION_ROLE,
        "launch_plan_sha256": launch_plan["launch_plan_sha256"],
        "execution_template_sha256": launch_plan["execution_template_sha256"],
        "input_binding_sha256": launch_plan["input_binding_sha256"],
        "container_id": container_id,
        "container_name": str(inspect.get("Name", "")).lstrip("/"),
        "image_id": observed_image,
        "network_mode": "none",
        "read_only_root_filesystem": True,
        "user": "1000:1000",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
        "device_requests": None,
        "mounts": declared,
        "exact_five_read_only_inputs": True,
        "authorization_mount_read_only_per_inspect": True,
        "authorization_write_rejection_tested": False,
        "one_read_write_output": True,
        "tmpfs_only_at_tmp": True,
        "legacy_workspace_mounted": False,
        "runtime_proof": False,
        "PACS_records_decoded": False,
        "PACS_labels_inspected": False,
        "GPU_used": False,
        "training_started": False,
    }
    if set(receipt) != set(MOUNT_RECEIPT_KEYS):
        raise ContractError("mount declaration builder key drift")
    return receipt


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
