#!/usr/bin/env python3
"""Verify the workstation4 GPU 3 PACS v3 static binding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, name = line.split("  ", 1)
        if digest(ROOT / name) != expected:
            raise SystemExit(f"INVALID_BUNDLE_HASH {name}")
        rows.append(f"{expected}  {name}\n")
    seal = load("SEAL.json")
    root_hash = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
    if root_hash != seal["bundle_root_sha256"]:
        raise SystemExit("INVALID_BUNDLE_ROOT")
    plan = load("HOST_LAUNCH_PLAN.json")
    if len(plan["cell_launches"]) != 30 or plan["execution_allowed"] is not False:
        raise SystemExit("INVALID_PLAN")
    runtime = plan["runtime"]
    if runtime["gpu_index"] != 3 or runtime["gpu_uuid"] != seal["gpu_uuid"]:
        raise SystemExit("INVALID_GPU3_BINDING")
    if plan["input_binding_sha256"] != digest(ROOT / "INPUT_BINDING.json"):
        raise SystemExit("INVALID_INPUT_BINDING")
    receipt = load("MOUNT_RECEIPT.json")
    if receipt["launch_plan_sha256"] != plan["launch_plan_sha256"]:
        raise SystemExit("INVALID_MOUNT_PLAN_BINDING")
    if receipt["GPU_used"] or receipt["training_started"]:
        raise SystemExit("INVALID_PREEXECUTION_CLAIM")
    image = load("IMAGE_INSPECT.json")[0]
    if image["Id"] != seal["candidate_image_id"]:
        raise SystemExit("INVALID_IMAGE_BINDING")
    if (ROOT / "MOUNT_PROBE.stdout.txt").read_text(encoding="utf-8") != (
        "PASS_INPUTS_READ_ONLY\nPASS_ROOT_READ_ONLY\nPASS_TMP_RW\nPASS_OUTPUT_RW\n"
    ):
        raise SystemExit("INVALID_MOUNT_PROBE_LOG")
    print("PASS_WORKSTATION4_GPU3_FIVE_INPUT_STATIC_BINDING_HOLD_EXECUTION")
    print(f"bundle_root_sha256={root_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

