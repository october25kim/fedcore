"""PACS-free Docker-inspect tests for the v3 pre-execution mount declaration."""

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
from fedcore.experiments.v3_execution_binding import (
    MOUNT_DECLARATION_ROLE,
    MOUNT_DECLARATION_STATUS,
)
from fedcore.experiments.v3_input_binding import (
    ScientificInputSpec,
    bind_scientific_inputs,
)
from fedcore.experiments.v3_mount_receipt import (
    MOUNT_PROBE_COMMAND,
    MOUNT_PROBE_ENTRYPOINT,
    MOUNT_PROBE_LOG,
    validate_no_gpu_mount_probe,
)
from fedcore.experiments.v3_scientific_runner import (
    PREBINDING_STATUS,
    build_launch_plan,
)


IMAGE_ID = "sha256:" + "a" * 64
PROBE_CONTAINER_ID = "f" * 64
SOURCE_REVISION = "c" * 40


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _plan(
    tmp_path: Path,
) -> tuple[dict[str, object], tuple[ScientificInputSpec, ...], Path, Path]:
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
    image_inspect = tmp_path / "IMAGE_INSPECT.json"
    _write_json(
        image_inspect,
        [
            {
                "Id": IMAGE_ID,
                "Config": {
                    "Labels": {"org.opencontainers.image.revision": SOURCE_REVISION},
                    "Env": [f"FEDCORE_SOURCE_COMMIT={SOURCE_REVISION}"],
                    "Entrypoint": ["python"],
                    "User": "1000:1000",
                    "WorkingDir": "/workspace",
                },
            }
        ],
    )
    output = tmp_path / "scientific-output"
    output.mkdir()
    control = tmp_path / "control"
    control.mkdir()
    plan = build_launch_plan(
        prebinding_gate=prebinding,
        input_binding=binding,
        output_dir=output,
        image_inspect=image_inspect,
        specs=specs,
        expected_input_dir=staged,
        expected_output_dir=output,
        control_dir=control,
        log_dir=tmp_path / "logs",
        image_tag="synthetic:mount-probe",
    )
    return plan, tuple(specs), output, control


def _inspect(plan: dict[str, object]) -> list[dict[str, object]]:
    mounts = [
        {
            "Type": "bind",
            "Source": row["source"],
            "Destination": row["destination"],
            "RW": not row["read_only"],
        }
        for row in plan["execution_template"]["mounts"]
        if row["type"] == "bind"
    ]
    return [
        {
            "Id": PROBE_CONTAINER_ID,
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
                "Entrypoint": MOUNT_PROBE_ENTRYPOINT,
                "Cmd": MOUNT_PROBE_COMMAND,
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
    control: Path,
    probe_log: str = MOUNT_PROBE_LOG,
) -> dict[str, object]:
    return validate_no_gpu_mount_probe(
        inspect,
        plan,
        expected_image_id=IMAGE_ID,
        probe_log=probe_log,
        registered_specs=specs,
        expected_output_dir=output,
        expected_control_dir=control,
    )


def test_exact_five_ro_mount_probe_passes_without_gpu(tmp_path: Path) -> None:
    plan, specs, output, control = _plan(tmp_path)
    receipt = _validate(_inspect(plan), plan, specs, output, control)
    assert receipt["status"] == MOUNT_DECLARATION_STATUS
    assert receipt["launch_plan_sha256"] == plan["launch_plan_sha256"]
    assert receipt["execution_template_sha256"] == plan["execution_template_sha256"]
    assert receipt["input_binding_sha256"] == plan["input_binding_sha256"]
    assert receipt["exact_five_read_only_inputs"] is True
    assert receipt["GPU_used"] is False
    assert receipt["image_id"] == IMAGE_ID
    assert len(receipt["mounts"]) == 8


def test_mount_declaration_includes_read_only_authorization_mount(tmp_path: Path) -> None:
    plan, specs, output, control = _plan(tmp_path)
    receipt = _validate(_inspect(plan), plan, specs, output, control)
    assert receipt["evidence_role"] == MOUNT_DECLARATION_ROLE
    assert receipt["runtime_proof"] is False
    assert receipt["authorization_mount_read_only_per_inspect"] is True
    assert receipt["authorization_write_rejection_tested"] is False
    rows = [row for row in receipt["mounts"] if row["destination"] == "/authorization"]
    assert rows == [
        {"type": "bind", "source": str(control), "destination": "/authorization", "read_only": True}
    ]
    assert receipt["mounts"] == plan["execution_template"]["mounts"]


@pytest.mark.parametrize(
    "failure",
    (
        "writable_input",
        "writable_authorization",
        "missing_authorization",
        "wrong_control_source",
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
        "wrong_image",
        "mutable_container_id",
    ),
)
def test_mount_probe_fails_closed(tmp_path: Path, failure: str) -> None:
    plan, specs, output, control = _plan(tmp_path)
    inspect = _inspect(plan)
    probe_log = MOUNT_PROBE_LOG
    mounts = inspect[0]["Mounts"]
    authorization = next(row for row in mounts if row["Destination"] == "/authorization")
    if failure == "writable_input":
        next(row for row in mounts if row["Destination"].startswith("/inputs/"))["RW"] = True
    elif failure == "writable_authorization":
        authorization["RW"] = True
    elif failure == "missing_authorization":
        mounts.remove(authorization)
    elif failure == "wrong_control_source":
        authorization["Source"] = str(tmp_path / "other-control")
    elif failure == "extra_mount":
        mounts.append(
            {
                "Type": "bind",
                "Source": "/data/workspace/sanghoon/fedcore2/outputs",
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
    elif failure == "wrong_image":
        inspect[0]["Image"] = "sha256:" + "b" * 64
    elif failure == "mutable_container_id":
        inspect[0]["Id"] = "container-id"
    else:
        probe_log = "PASS_INPUTS_READ_ONLY\n"
    with pytest.raises(ContractError):
        _validate(inspect, plan, specs, output, control, probe_log)


def test_forged_or_tampered_launch_plan_is_rejected(tmp_path: Path) -> None:
    plan, specs, output, control = _plan(tmp_path)
    forged = copy.deepcopy(plan)
    forged["input_inventory"][0]["source"] = "/tmp/not-registered"
    with pytest.raises(ContractError, match="self-hash mismatch"):
        _validate(_inspect(plan), forged, specs, output, control)
    legacy = copy.deepcopy(plan)
    legacy["schema_version"] = 1
    with pytest.raises(ContractError, match="not control schema 2"):
        _validate(_inspect(plan), legacy, specs, output, control)
