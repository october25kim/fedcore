"""PACS-free Docker-inspect tests for the v3 mount receipt."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from fedcore.experiments.v3_contract import (
    PROTOCOL_ID,
    ContractError,
    canonical_json_bytes,
)
from fedcore.experiments.v3_input_binding import (
    ScientificInputSpec,
    bind_scientific_inputs,
)
from fedcore.experiments.v3_mount_receipt import (
    MOUNT_PROBE_LOG,
    MOUNT_PROBE_SCRIPT,
    validate_no_gpu_mount_probe,
)
from fedcore.experiments.v3_scientific_runner import (
    PREBINDING_STATUS,
    build_launch_plan,
)


IMAGE_ID = "sha256:" + "a" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _plan(
    tmp_path: Path,
) -> tuple[dict[str, object], tuple[ScientificInputSpec, ...], Path]:
    names = (
        "PACS_mirror.zip",
        "IMAGE_MANIFEST.csv",
        "FOLD_MANIFEST.csv",
        "resnet18-f37072fd.pth",
        "convnext_tiny-983f1562.pth",
    )
    sources = tmp_path / "sources"
    sources.mkdir(parents=True)
    specs: list[ScientificInputSpec] = []
    for index, name in enumerate(names):
        source = sources / name
        source.write_bytes(f"mount-probe-{index}-{name}".encode())
        specs.append(
            ScientificInputSpec(
                name=name,
                source_path=str(source),
                container_path=f"/inputs/{name}",
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                size_bytes=source.stat().st_size,
            )
        )
    staged = tmp_path / "isolated-inputs"
    binding = tmp_path / "INPUT_BINDING.json"
    bind_scientific_inputs(staged, specs=specs, receipt_path=binding)
    prebinding = tmp_path / "PREBINDING_GATE.json"
    _write_json(
        prebinding,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": PREBINDING_STATUS,
            "scientific_runner_bound": False,
            "scientific_inputs_bound": False,
            "runs_scientific_empty": True,
            "training_authorized": False,
            "pacs_training_started": False,
            "pacs_inference_started": False,
        },
    )
    output = tmp_path / "scientific-output"
    output.mkdir()
    plan = build_launch_plan(
        prebinding_gate=prebinding,
        input_binding=binding,
        output_dir=output,
        specs=specs,
        expected_input_dir=staged,
        expected_output_dir=output,
        image_tag="synthetic:mount-probe",
        image_id=IMAGE_ID,
    )
    return plan, tuple(specs), output


def _inspect(plan: dict[str, object]) -> list[dict[str, object]]:
    mounts = [
        {
            "Type": "bind",
            "Source": row["source"],
            "Destination": row["destination"],
            "RW": False,
        }
        for row in plan["input_inventory"]
    ]
    mounts.append(
        {
            "Type": "bind",
            "Source": plan["output_directory"],
            "Destination": "/output",
            "RW": True,
        }
    )
    return [
        {
            "Id": "container-id",
            "Name": "/fedcore-v3-mount-probe",
            "Image": IMAGE_ID,
            "State": {
                "Status": "exited",
                "Running": False,
                "OOMKilled": False,
                "ExitCode": 0,
            },
            "Config": {
                "User": "1000:1000",
                "Entrypoint": ["/bin/sh"],
                "Cmd": ["-c", MOUNT_PROBE_SCRIPT],
                "Env": ["NVIDIA_VISIBLE_DEVICES=void"],
            },
            "HostConfig": {
                "NetworkMode": "none",
                "ReadonlyRootfs": True,
                "Privileged": False,
                "Devices": [],
                "Runtime": "runc",
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges:true"],
                "DeviceRequests": None,
                "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=1g"},
            },
            "Mounts": mounts,
        }
    ]


def _validate(
    inspect: list[dict[str, object]],
    plan: dict[str, object],
    specs: tuple[ScientificInputSpec, ...],
    output: Path,
) -> dict[str, object]:
    return validate_no_gpu_mount_probe(
        inspect,
        plan,
        expected_image_id=IMAGE_ID,
        probe_log=MOUNT_PROBE_LOG,
        registered_specs=specs,
        expected_output_dir=output,
    )


def test_exact_five_ro_mount_probe_passes_without_gpu(tmp_path: Path) -> None:
    plan, specs, output = _plan(tmp_path)
    receipt = _validate(_inspect(plan), plan, specs, output)
    assert receipt["status"] == "PASS_EXACT_FIVE_RO_MOUNT_PROBE_NO_GPU"
    assert receipt["launch_plan_sha256"] == plan["launch_plan_sha256"]
    assert receipt["input_binding_sha256"] == plan["input_binding_sha256"]
    assert receipt["exact_five_read_only_inputs"] is True
    assert receipt["GPU_used"] is False
    assert len(receipt["mounts"]) == 7


@pytest.mark.parametrize(
    "failure",
    (
        "writable_input",
        "extra_mount",
        "gpu",
        "network",
        "tmpfs",
        "privileged",
        "device",
        "runtime",
        "command",
        "state",
        "probe_log",
    ),
)
def test_mount_probe_fails_closed(tmp_path: Path, failure: str) -> None:
    plan, specs, output = _plan(tmp_path)
    inspect = _inspect(plan)
    probe_log = MOUNT_PROBE_LOG
    if failure == "writable_input":
        inspect[0]["Mounts"][0]["RW"] = True
    elif failure == "extra_mount":
        inspect[0]["Mounts"].append(
            {
                "Type": "bind",
                "Source": "/home/sanghoon/Desktop/Workspace/Fedcore/outputs",
                "Destination": "/legacy",
                "RW": False,
            }
        )
    elif failure == "gpu":
        inspect[0]["HostConfig"]["DeviceRequests"] = [{"DeviceIDs": ["GPU-x"]}]
    elif failure == "network":
        inspect[0]["HostConfig"]["NetworkMode"] = "bridge"
    elif failure == "tmpfs":
        inspect[0]["HostConfig"]["Tmpfs"] = {"/tmp": "rw"}
    elif failure == "privileged":
        inspect[0]["HostConfig"]["Privileged"] = True
    elif failure == "device":
        inspect[0]["HostConfig"]["Devices"] = [{"PathOnHost": "/dev/nvidia0"}]
    elif failure == "runtime":
        inspect[0]["HostConfig"]["Runtime"] = "nvidia"
    elif failure == "command":
        inspect[0]["Config"]["Cmd"] = ["-c", "python -m fedcore.experiments.v3_train"]
    elif failure == "state":
        inspect[0]["State"]["ExitCode"] = 1
    else:
        probe_log = "PASS_INPUTS_READ_ONLY\n"
    with pytest.raises(ContractError):
        validate_no_gpu_mount_probe(
            inspect,
            plan,
            expected_image_id=IMAGE_ID,
            probe_log=probe_log,
            registered_specs=specs,
            expected_output_dir=output,
        )


def test_forged_or_tampered_launch_plan_is_rejected(tmp_path: Path) -> None:
    plan, specs, output = _plan(tmp_path)
    forged = copy.deepcopy(plan)
    forged["input_inventory"][0]["source"] = "/tmp/not-registered"
    with pytest.raises(ContractError, match="self-hash mismatch"):
        _validate(_inspect(plan), forged, specs, output)
