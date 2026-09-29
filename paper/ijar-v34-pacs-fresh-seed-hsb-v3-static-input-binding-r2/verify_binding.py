#!/usr/bin/env python3
"""Verify the static PACS v3 input/runtime binding without third-party packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
EXPECTED_STATUS = (
    "PASS_FIVE_INPUT_AND_STATIC_RUNTIME_BINDING_"
    "HOLD_END_TO_END_SCIENTIFIC_ORCHESTRATOR_AND_AUTHORIZATION"
)
EXPECTED_INPUTS = (
    ("PACS_mirror.zip", 184417365, "42bf567f1ed8a01d522e47e4a677e2a3149577bbd6fcbb38bedfdd73cb59e147"),
    ("IMAGE_MANIFEST.csv", 1537910, "283d47bd8cfb7017d6e7fde6d65e07ff6479ab25127afcba135da3a06775afae"),
    ("FOLD_MANIFEST.csv", 7853382, "fe5b64d237f47f503b8895aa5e566727bc818e05ee597f84c440df70e91e5344"),
    ("resnet18-f37072fd.pth", 46830571, "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
    ("convnext_tiny-983f1562.pth", 114419221, "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"),
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main() -> int:
    manifest_rows: list[tuple[str, str]] = []
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        observed = digest(ROOT / name)
        if observed != expected:
            raise SystemExit(f"INVALID_BUNDLE_HASH {name} {observed} != {expected}")
        if name != "SEAL.json":
            manifest_rows.append((name, f"{expected}  {name}\n"))
    seal = load("SEAL.json")
    root_hash = hashlib.sha256(
        "".join(row for _, row in sorted(manifest_rows)).encode("utf-8")
    ).hexdigest()
    if root_hash != seal["bundle_root_sha256"]:
        raise SystemExit("INVALID_BUNDLE_ROOT")

    gate = load("EXECUTION_GATE.json")
    if gate["status"] != EXPECTED_STATUS:
        raise SystemExit("INVALID_GATE_STATUS")
    for field in (
        "training_authorized",
        "pacs_training_started",
        "pacs_inference_started",
        "submission_authorized",
        "end_to_end_scientific_runner_bound",
        "artifact_bound_run_authorization_implemented",
    ):
        if gate[field] is not False:
            raise SystemExit(f"INVALID_GATE_FIELD {field}")

    for line in (ROOT / "SOURCE_MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        observed = digest(REPOSITORY_ROOT / name)
        if observed != expected:
            raise SystemExit(f"INVALID_SOURCE_HASH {name} {observed} != {expected}")

    binding = load("INPUT_BINDING.json")
    if binding["status"] != "PASS_FIVE_INPUT_BINDING" or len(binding["files"]) != 5:
        raise SystemExit("INVALID_INPUT_BINDING")
    for record, (name, size, sha256) in zip(binding["files"], EXPECTED_INPUTS, strict=True):
        if (
            record["name"] != name
            or record["size_bytes"] != size
            or record["sha256"] != sha256
            or record["staged_mode_octal"] != "0444"
            or record["read_only_mount_required"] is not True
            or record["symlink"] is not False
        ):
            raise SystemExit(f"INVALID_INPUT_RECORD {name}")

    plan = load("HOST_LAUNCH_PLAN.json")
    if (
        plan["status"] != "PASS_CANDIDATE_HOST_LAUNCH_PLAN_HOLD_EXECUTION_READINESS"
        or plan["execution_allowed"] is not False
        or len(plan["input_inventory"]) != 5
    ):
        raise SystemExit("INVALID_STATIC_LAUNCH_PLAN")
    receipt = load("MOUNT_RECEIPT.json")
    if receipt["status"] != "PASS_EXACT_FIVE_RO_MOUNT_PROBE_NO_GPU":
        raise SystemExit("INVALID_MOUNT_RECEIPT")
    if receipt["launch_plan_sha256"] != plan["launch_plan_sha256"]:
        raise SystemExit("INVALID_LAUNCH_PLAN_BINDING")
    if receipt["input_binding_sha256"] != digest(ROOT / "INPUT_BINDING.json"):
        raise SystemExit("INVALID_INPUT_RECEIPT_BINDING")
    if receipt["GPU_used"] or receipt["training_started"]:
        raise SystemExit("INVALID_PROBE_EXECUTION_CLAIM")

    inspect = load("MOUNT_PROBE.inspect.json")[0]
    host = inspect["HostConfig"]
    config = inspect["Config"]
    state = inspect["State"]
    if host["DeviceRequests"] not in (None, []) or host["Devices"] not in (None, []):
        raise SystemExit("INVALID_GPU_DEVICE_SCOPE")
    if host["NetworkMode"] != "none" or host["ReadonlyRootfs"] is not True:
        raise SystemExit("INVALID_CONTAINER_ISOLATION")
    if config["Entrypoint"] != ["python"] or config["Cmd"] != ["-m", "fedcore.experiments.v3_mount_probe"]:
        raise SystemExit("INVALID_PROBE_COMMAND")
    if state["Status"] != "exited" or state["ExitCode"] != 0:
        raise SystemExit("INVALID_PROBE_STATE")
    if (ROOT / "MOUNT_PROBE.stdout.txt").read_text(encoding="utf-8") != (
        "PASS_INPUTS_READ_ONLY\nPASS_ROOT_READ_ONLY\nPASS_TMP_RW\nPASS_OUTPUT_RW\n"
    ):
        raise SystemExit("INVALID_PROBE_OUTPUT")

    weights = load("CPU_WEIGHT_BINDING.json")
    if (
        weights["status"] != "PASS_CPU_STRICT_WEIGHT_BINDING"
        or weights["GPU_used"]
        or weights["training_started"]
        or weights["PACS_records_decoded"]
        or weights["PACS_labels_inspected"]
    ):
        raise SystemExit("INVALID_CPU_WEIGHT_BINDING")

    image = load("IMAGE_INSPECT.json")[0]
    container = load("CONTAINER_BINDING.json")
    if image["Id"] != container["image_id"] or plan["image"]["id"] != container["image_id"]:
        raise SystemExit("INVALID_IMAGE_BINDING")
    if image["Config"]["Labels"]["org.opencontainers.image.revision"] != container["source_revision_label"]:
        raise SystemExit("INVALID_SOURCE_REVISION_LABEL")

    print("PASS_FIVE_INPUT_AND_STATIC_RUNTIME_BINDING_HOLD_EXECUTION")
    print(f"bundle_root_sha256={root_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
