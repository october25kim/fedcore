from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil

import pytest

from fedcore.experiments.v3_contract import ContractError, canonical_json_sha256
import fedcore.experiments.v3_execution_binding as binding
from fedcore.experiments.v3_orchestrator import write_json


REPOSITORY = Path(__file__).resolve().parents[1]
STATIC = REPOSITORY / "paper" / "ijar-v34-pacs-fresh-seed-hsb-v3-static-input-binding-r2"
SOURCE_COMMIT = "a" * 40
SOURCE_TREE = "f" * 40
IMAGE_ID = "sha256:" + "b" * 64
CONTAINER_ID = "c" * 64
PROBE_CONTAINER_ID = "e" * 64
TEST_CONTAINER_ID = "9" * 64


def _reseal(value: dict, field: str) -> dict:
    body = {
        key: item
        for key, item in value.items()
        if key not in {field, f"{field}_definition"}
    }
    return binding._self_hashed(body, field)


def _fake_git_object_reader(
    *,
    source_commit: str,
    source_tree: str,
    committed_blobs: dict[str, bytes],
    runtime_paths: tuple[str, ...],
):
    """Return deterministic Git-object bytes without requiring a Git executable.

    Production closure still invokes the real Git object database in
    ``build_binding.py``.  These unit tests isolate the fail-closed comparison
    so the same-image execution gate does not acquire an unregistered system
    package dependency.
    """

    def reader(_repository: Path, arguments, _label: str) -> bytes:
        argv = tuple(arguments)
        if argv == ("rev-parse", "--verify", f"{source_commit}^{{commit}}"):
            return f"{source_commit}\n".encode("ascii")
        if argv == ("rev-parse", "--verify", f"{source_commit}^{{tree}}"):
            return f"{source_tree}\n".encode("ascii")
        if argv[:4] == ("ls-tree", "-r", "-z", source_commit):
            object_id = "0" * 40
            return b"".join(
                f"100644 blob {object_id}\t{path}\0".encode("utf-8")
                for path in sorted(runtime_paths)
            )
        if argv[:2] == ("cat-file", "blob") and len(argv) == 3:
            prefix = f"{source_commit}:"
            if argv[2].startswith(prefix):
                return committed_blobs[argv[2][len(prefix) :]]
        raise AssertionError(f"unexpected Git-object query: {argv!r}")

    return reader


def _build_source_manifest(path: Path) -> dict:
    value = binding.build_scientific_source_manifest(
        REPOSITORY,
        source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE,
    )
    write_json(path, value)
    return value


def _source_labels(source_manifest_path: Path, manifest: dict) -> dict[str, str]:
    build = {row["path"]: row["sha256"] for row in manifest["build_files"]}
    return {
        "org.opencontainers.image.revision": SOURCE_COMMIT,
        binding.SOURCE_MANIFEST_LABEL: binding._hash_regular_file(
            source_manifest_path, "test source manifest"
        ),
        binding.RUNTIME_INVENTORY_LABEL: manifest["runtime_inventory_sha256"],
        binding.BUILD_INVENTORY_LABEL: manifest["build_inventory_sha256"],
        binding.DOCKERFILE_LABEL: build["docker/Dockerfile.v3_hsb_scientific"],
        binding.REQUIREMENTS_LABEL: build["requirements.lock"],
    }


def _write_image_inspect(
    path: Path, source_manifest_path: Path, manifest: dict
) -> None:
    write_json(
        path,
        [
            {
                "Id": IMAGE_ID,
                "Config": {
                    "Labels": _source_labels(source_manifest_path, manifest),
                    "Env": [f"FEDCORE_SOURCE_COMMIT={SOURCE_COMMIT}"],
                    "Entrypoint": ["python"],
                    "User": "1000:1000",
                },
            }
        ],
    )


def _write_source_probe_inspect(path: Path) -> None:
    write_json(
        path,
        [
            {
                "Id": PROBE_CONTAINER_ID,
                "Image": IMAGE_ID,
                "State": {"Status": "created", "Running": False},
                "HostConfig": {
                    "NetworkMode": "none",
                    "DeviceRequests": [],
                    "Binds": None,
                },
                "Mounts": [],
            }
        ],
    )


def _write_test_container_inspect(path: Path) -> None:
    write_json(
        path,
        [
            {
                "Id": TEST_CONTAINER_ID,
                "Image": IMAGE_ID,
                "State": {"Status": "exited", "Running": False, "ExitCode": 0},
                "Config": {
                    "Entrypoint": ["python"],
                    "Cmd": list(binding.REQUIRED_EXECUTION_PYTEST_ARGV[1:]),
                    "Env": [
                        "CUDA_VISIBLE_DEVICES=",
                        f"PYTHONPATH=/workspace:{binding.TEST_RUNTIME_ROOT}",
                        "PYTHONSAFEPATH=1",
                    ],
                    "WorkingDir": "/tmp",
                    "User": "1000:1000",
                },
                "HostConfig": {
                    "NetworkMode": "none",
                    "DeviceRequests": [],
                    "Devices": [],
                    "ReadonlyRootfs": True,
                    "Privileged": False,
                    "CapDrop": ["ALL"],
                    "SecurityOpt": ["no-new-privileges"],
                    "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=1g"},
                },
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": str(REPOSITORY),
                        "Destination": "/testrepo",
                        "RW": False,
                    },
                    {
                        "Type": "bind",
                        "Source": str(path.parent.resolve()),
                        "Destination": "/evidence",
                        "RW": True,
                    },
                ],
            }
        ],
    )


def _write_test_report(
    path: Path,
    source_manifest_path: Path,
    test_container_inspect_path: Path,
    junit_xml_path: Path,
    pytest_stdout_path: Path,
) -> dict:
    value = {
        "schema_version": 1,
        "protocol_id": binding.PROTOCOL_ID,
        "status": binding.TEST_REPORT_STATUS,
        "image_id": IMAGE_ID,
        "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": binding._hash_regular_file(
            source_manifest_path, "test source manifest"
        ),
        "test_container_inspect_sha256": binding._hash_regular_file(
            test_container_inspect_path, "test-container inspect"
        ),
        "test_container_id": TEST_CONTAINER_ID,
        "junit_xml_sha256": binding._hash_regular_file(
            junit_xml_path, "test JUnit XML"
        ),
        "pytest_stdout_sha256": binding._hash_regular_file(
            pytest_stdout_path, "test stdout"
        ),
        "pytest_argv": list(binding.REQUIRED_EXECUTION_PYTEST_ARGV),
        "required_test_files": list(binding.REQUIRED_EXECUTION_TEST_FILES),
        "test_file_sha256": {
            test_path: binding._hash_regular_file(
                REPOSITORY / test_path, f"test source {test_path}"
            )
            for test_path in binding.REQUIRED_EXECUTION_TEST_FILES
        },
        "pytest_counts": {
            "collected": len(binding.REQUIRED_EXECUTION_TEST_FILES),
            "passed": len(binding.REQUIRED_EXECUTION_TEST_FILES),
            "failed": 0,
            "skipped": 0,
        },
        "exit_code": 0,
        "environment": {"CUDA_VISIBLE_DEVICES": ""},
        "PACS_opened": False,
        "GPU_used": False,
        "test_gates": {
            gate: True for gate in binding.REQUIRED_EXECUTION_TEST_GATES
        },
        "gate_test_nodeids": dict(binding.REQUIRED_EXECUTION_GATE_NODEIDS),
    }
    write_json(path, value)
    return value


def _authorization(readiness: dict) -> dict:
    issued = datetime.now(timezone.utc) - timedelta(minutes=1)
    valid_until = issued + timedelta(hours=2)
    nonce = "d" * 64
    challenge = binding._authorization_challenge(readiness["readiness_sha256"])
    issued_text = issued.isoformat(timespec="seconds").replace("+00:00", "Z")
    valid_text = valid_until.isoformat(timespec="seconds").replace("+00:00", "Z")
    authorization_id = canonical_json_sha256(
        {
            "authorization_challenge_sha256": challenge,
            "authorization_nonce": nonce,
            "authorized_by": binding.AUTHORIZED_BY,
            "issued_at_utc": issued_text,
            "valid_until_utc": valid_text,
        }
    )
    body = {
        "schema_version": 1,
        "protocol_id": readiness["protocol_id"],
        "status": binding.AUTHORIZATION_STATUS,
        "training_authorized": True,
        "scope": "ONE_FRESH_30_CELL_CAMPAIGN",
        "readiness_sha256": readiness["readiness_sha256"],
        "authorization_challenge_sha256": challenge,
        "authorization_method": binding.AUTHORIZATION_METHOD,
        "source_commit": readiness["source_commit"],
        "image_id": readiness["image_id"],
        "input_binding_sha256": readiness["input_binding_sha256"],
        "mount_receipt_sha256": readiness["mount_receipt_sha256"],
        "cell_order_sha256": readiness["cell_order_sha256"],
        "host_output_dir": readiness["host_output_dir"],
        "planned_cells": 30,
        "max_concurrent_scientific_processes": 1,
        "remaining_scientific_walltime_hours": readiness[
            "remaining_scientific_walltime_hours"
        ],
        "cpu_limit_cores": readiness["cpu_limit_cores"],
        "maximum_cpu_core_hours": readiness["maximum_cpu_core_hours"],
        "fresh_output_required": True,
        "resume_authorized": False,
        "authorization_nonce": nonce,
        "authorization_id": authorization_id,
        "authorized_by": binding.AUTHORIZED_BY,
        "issued_at_utc": issued_text,
        "valid_until_utc": valid_text,
    }
    return binding._self_hashed(body, "authorization_sha256")


def _docker_inspect(readiness: dict, plan: dict) -> list[dict]:
    mounts = []
    for row in reversed(plan["mounts"]):
        if row["type"] == "tmpfs":
            continue
        mounts.append(
            {
                "Type": "bind",
                "Source": row["source"],
                "Destination": row["destination"],
                "RW": not row["read_only"],
                "Propagation": "rprivate",
            }
        )
    return [
        {
            "Id": CONTAINER_ID,
            "Image": readiness["image_id"],
            "Name": "/" + plan["container_name"],
            "State": {
                "Status": "created",
                "Running": False,
                "Dead": False,
                "OOMKilled": False,
            },
            "Config": {
                "User": "1000:1000",
                "Entrypoint": ["python"],
                "Cmd": readiness["container_command"],
                "Env": [f"FEDCORE_SOURCE_COMMIT={readiness['source_commit']}"],
                "Labels": {
                    "org.opencontainers.image.revision": readiness["source_commit"],
                    binding.SOURCE_MANIFEST_LABEL: readiness["source_manifest_sha256"],
                    binding.RUNTIME_INVENTORY_LABEL: readiness["runtime_inventory_sha256"],
                    binding.BUILD_INVENTORY_LABEL: readiness["build_inventory_sha256"],
                    binding.DOCKERFILE_LABEL: readiness["dockerfile_sha256"],
                    binding.REQUIREMENTS_LABEL: readiness["requirements_lock_sha256"],
                },
            },
            "HostConfig": {
                "NetworkMode": "none",
                "ReadonlyRootfs": True,
                "Privileged": False,
                "Devices": [],
                "AutoRemove": False,
                "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                "Links": None,
                "VolumesFrom": None,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges"],
                "DeviceRequests": [
                    {
                        "Driver": "",
                        "Count": -1,
                        "DeviceIDs": [binding.EXPECTED_GPU_UUID],
                        "Capabilities": [["gpu"]],
                        "Options": {},
                    }
                ],
                "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=1g"},
                "NanoCpus": binding.DOCKER_NANO_CPUS,
            },
            "Mounts": mounts,
        }
    ]


@pytest.fixture
def chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    control = tmp_path / "control"
    output = tmp_path / "output"
    control.mkdir()
    output.mkdir()
    monkeypatch.setattr(binding, "EXPECTED_HOST_CONTROL_DIR", control.resolve())
    monkeypatch.setattr(binding, "EXPECTED_HOST_OUTPUT_DIR", output.resolve())
    monkeypatch.setattr(binding, "CONTAINER_WORKSPACE_DIR", REPOSITORY)
    monkeypatch.setattr(
        binding, "_validate_manifest_against_git_commit", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        binding, "_validate_test_report_against_git_commit", lambda *args, **kwargs: None
    )
    image_inspect = tmp_path / "IMAGE_INSPECT.json"
    source_manifest = tmp_path / "SOURCE_MANIFEST.json"
    source_probe_inspect = tmp_path / "SOURCE_PROBE_INSPECT.json"
    source_probe_receipt = tmp_path / "SOURCE_PROBE_RECEIPT.json"
    test_container_inspect = tmp_path / "TEST_CONTAINER_INSPECT.json"
    test_junit = tmp_path / "TEST_JUNIT.xml"
    test_stdout = tmp_path / "TEST_STDOUT.log"
    test_report = tmp_path / "TEST_REPORT.json"
    manifest = _build_source_manifest(source_manifest)
    _write_image_inspect(image_inspect, source_manifest, manifest)
    _write_source_probe_inspect(source_probe_inspect)
    probe_workspace = tmp_path / "probe-workspace"
    _copy_manifest_runtime(source_manifest, probe_workspace)
    write_json(
        source_probe_receipt,
        binding.build_source_probe_receipt(
            source_manifest_path=source_manifest,
            extracted_workspace_root=probe_workspace,
            image_id=IMAGE_ID,
            container_inspect_path=source_probe_inspect,
        ),
    )
    _write_test_container_inspect(test_container_inspect)
    test_junit.write_text("<testsuites/>\n", encoding="utf-8")
    test_stdout.write_text("all registered tests passed\n", encoding="utf-8")
    _write_test_report(
        test_report,
        source_manifest,
        test_container_inspect,
        test_junit,
        test_stdout,
    )
    readiness = binding.build_execution_readiness(
        static_launch_plan_path=STATIC / "HOST_LAUNCH_PLAN.json",
        input_binding_path=STATIC / "INPUT_BINDING.json",
        mount_receipt_path=STATIC / "MOUNT_RECEIPT.json",
        source_commit=SOURCE_COMMIT,
        image_id=IMAGE_ID,
        image_inspect_path=image_inspect,
        source_manifest_path=source_manifest,
        source_probe_receipt_path=source_probe_receipt,
        orchestrator_path=REPOSITORY / "fedcore/experiments/v3_campaign_runner.py",
        state_machine_path=REPOSITORY / "fedcore/experiments/v3_campaign_state.py",
        execution_binding_path=REPOSITORY / "fedcore/experiments/v3_execution_binding.py",
        finalizer_path=REPOSITORY / "fedcore/experiments/v3_campaign_finalizer.py",
        posttrain_path=REPOSITORY / "fedcore/experiments/v3_posttrain.py",
        test_report_path=test_report,
        test_container_inspect_path=test_container_inspect,
        test_junit_path=test_junit,
        test_stdout_path=test_stdout,
        cell_order_path=REPOSITORY
        / "paper/ijar-v34-pacs-fresh-seed-hsb-v3/PACS_MODEL_MATRIX.csv",
        host_control_dir=control,
        host_output_dir=output,
    )
    static_plan = json.loads((STATIC / "HOST_LAUNCH_PLAN.json").read_text())
    input_binding = json.loads((STATIC / "INPUT_BINDING.json").read_text())
    mount_receipt = json.loads((STATIC / "MOUNT_RECEIPT.json").read_text())
    authorization = _authorization(readiness)
    plan = binding.build_authorized_campaign_plan(
        readiness, authorization, input_binding=input_binding
    )
    inspect = _docker_inspect(readiness, plan)
    receipt = binding.build_runtime_receipt_from_inspect(
        inspect,
        readiness=readiness,
        authorization=authorization,
        authorized_plan=plan,
        input_binding=input_binding,
    )
    return {
        "control": control,
        "output": output,
        "image_inspect": image_inspect,
        "source_manifest": source_manifest,
        "source_probe_inspect": source_probe_inspect,
        "source_probe_receipt": source_probe_receipt,
        "test_container_inspect": test_container_inspect,
        "test_junit": test_junit,
        "test_stdout": test_stdout,
        "test_report": test_report,
        "readiness": readiness,
        "static_plan": static_plan,
        "input_binding": input_binding,
        "mount_receipt": mount_receipt,
        "authorization": authorization,
        "plan": plan,
        "inspect": inspect,
        "receipt": receipt,
    }


def _write_control_bundle(chain: dict) -> None:
    files = {
        "HOST_LAUNCH_PLAN.json": STATIC / "HOST_LAUNCH_PLAN.json",
        "INPUT_BINDING.json": STATIC / "INPUT_BINDING.json",
        "MOUNT_RECEIPT.json": STATIC / "MOUNT_RECEIPT.json",
        "SOURCE_MANIFEST.json": chain["source_manifest"],
        "SOURCE_PROBE_RECEIPT.json": chain["source_probe_receipt"],
        "TEST_REPORT.json": chain["test_report"],
    }
    for name, source in files.items():
        shutil.copyfile(source, chain["control"] / name)
    for name, value in (
        ("EXECUTION_READINESS.json", chain["readiness"]),
        ("RUN_AUTHORIZATION.json", chain["authorization"]),
        ("AUTHORIZED_CAMPAIGN_PLAN.json", chain["plan"]),
        ("RUNTIME_INSPECT.json", chain["inspect"]),
        ("RUNTIME_INSPECT_RECEIPT.json", chain["receipt"]),
    ):
        write_json(chain["control"] / name, value)
    for path in chain["control"].iterdir():
        path.chmod(0o444)


def _copy_manifest_runtime(source_manifest: Path, destination: Path) -> None:
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    for row in manifest["runtime_files"]:
        source = REPOSITORY / row["path"]
        target = destination / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def test_full_artifact_chain_and_mount_order(chain):
    assert binding.validate_runtime_receipt(
        chain["receipt"],
        readiness=chain["readiness"],
        authorization=chain["authorization"],
        authorized_plan=chain["plan"],
        inspect_value=chain["inspect"],
        input_binding=chain["input_binding"],
    ) == chain["receipt"]["runtime_receipt_sha256"]
    assert "--cpus" in chain["plan"]["docker_create_argv"]
    cpu_index = chain["plan"]["docker_create_argv"].index("--cpus")
    assert chain["plan"]["docker_create_argv"][cpu_index + 1] == "0.5"
    assert chain["receipt"]["exact_cpu_limit_verified"] is True


def test_readiness_precharges_bounded_cleanup_inside_total_gpu_cap(chain):
    readiness = chain["readiness"]
    assert (
        readiness["scientific_execution_timeout_seconds"]
        == (46 * 60 * 60) - 300
    )
    assert readiness["post_timeout_cleanup_command_bound_seconds"] == 90
    assert readiness["scientific_cleanup_reserve_seconds"] == 300
    assert readiness["scientific_cleanup_safety_margin_seconds"] == 210
    assert (
        int(readiness["sealed_capacity_gpu_hours"] * 60 * 60)
        + readiness["scientific_execution_timeout_seconds"]
        + readiness["scientific_cleanup_reserve_seconds"]
    ) == int(readiness["total_gpu_hour_cap"] * 60 * 60)
    assert (
        readiness["scientific_cleanup_reserve_seconds"]
        >= readiness["post_timeout_cleanup_command_bound_seconds"]
    )


def test_resource_cap_invariant_rejects_old_46_hour_timeout(monkeypatch):
    monkeypatch.setattr(
        binding, "SCIENTIFIC_EXECUTION_TIMEOUT_SECONDS", 46 * 60 * 60
    )
    monkeypatch.setattr(binding, "REMAINING_SCIENTIFIC_WALLTIME_HOURS", 46.0)
    monkeypatch.setattr(binding, "MAXIMUM_CPU_CORE_HOURS", 23.0)
    with pytest.raises(ContractError, match="exceeds the GPU cap"):
        binding._validate_gpu_resource_cap_invariant()


def test_readiness_rejects_resealed_cleanup_reserve_drift(chain):
    forged = deepcopy(chain["readiness"])
    forged["scientific_cleanup_reserve_seconds"] = 90
    forged["scientific_cleanup_safety_margin_seconds"] = 0
    forged = _reseal(forged, "readiness_sha256")
    with pytest.raises(ContractError, match="resource-cap drift"):
        binding.validate_execution_readiness(
            forged,
            static_launch_plan=chain["static_plan"],
            input_binding=chain["input_binding"],
            mount_receipt=chain["mount_receipt"],
        )


def test_readiness_rejects_self_consistent_command_and_path_forgery(chain):
    for field, replacement in (
        ("container_command", ["-c", "print('not campaign')"]),
        ("host_output_dir", "/tmp/evil-output"),
        ("host_control_dir", "/tmp/evil-control"),
    ):
        forged = deepcopy(chain["readiness"])
        forged[field] = replacement
        forged = _reseal(forged, "readiness_sha256")
        with pytest.raises(ContractError):
            binding.validate_execution_readiness(
                forged,
                static_launch_plan=chain["static_plan"],
                input_binding=chain["input_binding"],
                mount_receipt=chain["mount_receipt"],
            )


def test_readiness_builder_rejects_nonempty_output(chain, tmp_path: Path):
    (chain["output"] / "unexpected").write_text("x", encoding="utf-8")
    image_inspect = tmp_path / "image.json"
    manifest = json.loads(chain["source_manifest"].read_text(encoding="utf-8"))
    _write_image_inspect(image_inspect, chain["source_manifest"], manifest)
    with pytest.raises(ContractError, match="must be empty"):
        binding.build_execution_readiness(
            static_launch_plan_path=STATIC / "HOST_LAUNCH_PLAN.json",
            input_binding_path=STATIC / "INPUT_BINDING.json",
            mount_receipt_path=STATIC / "MOUNT_RECEIPT.json",
            source_commit=SOURCE_COMMIT,
            image_id=IMAGE_ID,
            image_inspect_path=image_inspect,
            source_manifest_path=chain["source_manifest"],
            source_probe_receipt_path=chain["source_probe_receipt"],
            orchestrator_path=REPOSITORY / "fedcore/experiments/v3_campaign_runner.py",
            state_machine_path=REPOSITORY / "fedcore/experiments/v3_campaign_state.py",
            execution_binding_path=REPOSITORY / "fedcore/experiments/v3_execution_binding.py",
            finalizer_path=REPOSITORY / "fedcore/experiments/v3_campaign_finalizer.py",
            posttrain_path=REPOSITORY / "fedcore/experiments/v3_posttrain.py",
            test_report_path=chain["test_report"],
            test_container_inspect_path=chain["test_container_inspect"],
            test_junit_path=chain["test_junit"],
            test_stdout_path=chain["test_stdout"],
            cell_order_path=REPOSITORY
            / "paper/ijar-v34-pacs-fresh-seed-hsb-v3/PACS_MODEL_MATRIX.csv",
            host_control_dir=chain["control"],
            host_output_dir=chain["output"],
        )


def test_placeholder_test_report_is_rejected(chain):
    with pytest.raises(ContractError, match="test report key drift"):
        binding.validate_execution_test_report(
            {"schema_version": 1, "status": "PASS_TEST_REPORT"},
            expected_image_id=IMAGE_ID,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_manifest_sha256=binding._hash_regular_file(
                chain["source_manifest"], "test source manifest"
            ),
        )


def test_test_report_rejects_unpassed_campaign_runner_e2e_gate(chain):
    report = json.loads(chain["test_report"].read_text(encoding="utf-8"))
    report["test_gates"]["campaign_runner_synthetic_e2e"] = False
    with pytest.raises(ContractError, match="campaign_runner_synthetic_e2e"):
        binding.validate_execution_test_report(
            report,
            expected_image_id=IMAGE_ID,
            expected_source_commit=SOURCE_COMMIT,
            expected_source_manifest_sha256=binding._hash_regular_file(
                chain["source_manifest"], "test source manifest"
            ),
        )


def test_test_container_inspect_rejects_gpu_or_pacs_mount(chain):
    baseline = json.loads(
        chain["test_container_inspect"].read_text(encoding="utf-8")
    )
    gpu = deepcopy(baseline)
    gpu[0]["HostConfig"]["DeviceRequests"] = [{"Capabilities": [["gpu"]]}]
    with pytest.raises(ContractError, match="must not request a GPU"):
        binding.validate_test_container_inspect(
            gpu,
            expected_image_id=IMAGE_ID,
            expected_repository_root=REPOSITORY,
            expected_evidence_root=chain["test_junit"].parent,
        )

    pacs = deepcopy(baseline)
    pacs[0]["Mounts"].append(
        {
            "Type": "bind",
            "Source": "/private/PACS_mirror.zip",
            "Destination": "/inputs/PACS_mirror.zip",
            "RW": False,
        }
    )
    with pytest.raises(ContractError, match="unregistered or PACS/input mount"):
        binding.validate_test_container_inspect(
            pacs,
            expected_image_id=IMAGE_ID,
            expected_repository_root=REPOSITORY,
            expected_evidence_root=chain["test_junit"].parent,
        )


def test_test_container_rejects_evidence_from_another_execution(chain, tmp_path):
    inspect = json.loads(
        chain["test_container_inspect"].read_text(encoding="utf-8")
    )
    other_execution = tmp_path / "other-execution-evidence"
    other_execution.mkdir()
    inspect[0]["Mounts"][1]["Source"] = str(other_execution.resolve())

    with pytest.raises(ContractError, match="exact raw-evidence directory"):
        binding.validate_test_container_inspect(
            inspect,
            expected_image_id=IMAGE_ID,
            expected_repository_root=REPOSITORY,
            expected_evidence_root=chain["test_junit"].parent,
        )


def test_test_container_rejects_repo_workdir_or_relative_test_paths(chain):
    baseline = json.loads(
        chain["test_container_inspect"].read_text(encoding="utf-8")
    )

    repo_workdir = deepcopy(baseline)
    repo_workdir[0]["Config"]["WorkingDir"] = "/testrepo"
    with pytest.raises(ContractError, match="entrypoint/workdir/user mismatch"):
        binding.validate_test_container_inspect(
            repo_workdir,
            expected_image_id=IMAGE_ID,
            expected_repository_root=REPOSITORY,
            expected_evidence_root=chain["test_junit"].parent,
        )

    relative_test = deepcopy(baseline)
    absolute_test = binding.REQUIRED_EXECUTION_CONTAINER_TEST_FILES[0]
    command = relative_test[0]["Config"]["Cmd"]
    command[command.index(absolute_test)] = binding.REQUIRED_EXECUTION_TEST_FILES[0]
    with pytest.raises(ContractError, match="pytest command/file order mismatch"):
        binding.validate_test_container_inspect(
            relative_test,
            expected_image_id=IMAGE_ID,
            expected_repository_root=REPOSITORY,
            expected_evidence_root=chain["test_junit"].parent,
        )


def test_registered_test_runtime_version():
    assert pytest.__version__ == binding.TEST_RUNTIME_PYTEST_VERSION
    plugin_index = binding.REQUIRED_EXECUTION_PYTEST_ARGV.index("-p")
    assert binding.REQUIRED_EXECUTION_PYTEST_ARGV[plugin_index : plugin_index + 2] == (
        "-p",
        "no:cacheprovider",
    )


def test_registered_test_runtime_wheelhouse_is_exact():
    binding._validate_test_runtime_wheelhouse(REPOSITORY)


def test_test_runtime_wheelhouse_rejects_extra_unbound_file(tmp_path):
    wheelhouse = tmp_path / "docker/pytest-wheelhouse"
    wheelhouse.mkdir(parents=True)
    for relative in binding.TEST_RUNTIME_WHEEL_FILES:
        (tmp_path / relative).write_bytes(b"registered-wheel")
    (wheelhouse / "unregistered-extra.whl").write_bytes(b"extra")
    with pytest.raises(ContractError, match="wheelhouse file-set drift"):
        binding._validate_test_runtime_wheelhouse(tmp_path)


def test_raw_test_evidence_must_share_exact_registered_paths(chain, tmp_path):
    copied_stdout = tmp_path / "different-execution" / "TEST_STDOUT.log"
    copied_stdout.parent.mkdir()
    copied_stdout.write_bytes(chain["test_stdout"].read_bytes())

    with pytest.raises(ContractError, match="share one exact evidence directory"):
        binding.validate_test_evidence_bundle_paths(
            test_container_inspect_path=chain["test_container_inspect"],
            test_junit_path=chain["test_junit"],
            test_stdout_path=copied_stdout,
        )


def test_test_report_rejects_raw_evidence_hash_drift(chain):
    report = json.loads(chain["test_report"].read_text(encoding="utf-8"))
    report["test_container_inspect_sha256"] = "0" * 64
    write_json(chain["test_report"], report)
    with pytest.raises(ContractError, match="raw test-container inspect differs"):
        binding.build_execution_readiness(
            static_launch_plan_path=STATIC / "HOST_LAUNCH_PLAN.json",
            input_binding_path=STATIC / "INPUT_BINDING.json",
            mount_receipt_path=STATIC / "MOUNT_RECEIPT.json",
            source_commit=SOURCE_COMMIT,
            image_id=IMAGE_ID,
            image_inspect_path=chain["image_inspect"],
            source_manifest_path=chain["source_manifest"],
            source_probe_receipt_path=chain["source_probe_receipt"],
            orchestrator_path=REPOSITORY / "fedcore/experiments/v3_campaign_runner.py",
            state_machine_path=REPOSITORY / "fedcore/experiments/v3_campaign_state.py",
            execution_binding_path=REPOSITORY
            / "fedcore/experiments/v3_execution_binding.py",
            finalizer_path=REPOSITORY / "fedcore/experiments/v3_campaign_finalizer.py",
            posttrain_path=REPOSITORY / "fedcore/experiments/v3_posttrain.py",
            test_report_path=chain["test_report"],
            test_container_inspect_path=chain["test_container_inspect"],
            test_junit_path=chain["test_junit"],
            test_stdout_path=chain["test_stdout"],
            cell_order_path=REPOSITORY
            / "paper/ijar-v34-pacs-fresh-seed-hsb-v3/PACS_MODEL_MATRIX.csv",
            host_control_dir=chain["control"],
            host_output_dir=chain["output"],
        )


def test_runtime_receipt_requires_exact_raw_inspect(chain):
    forged = deepcopy(chain["receipt"])
    forged["container_id"] = "e" * 64
    forged = _reseal(forged, "runtime_receipt_sha256")
    with pytest.raises(ContractError):
        binding.validate_runtime_receipt(
            forged,
            readiness=chain["readiness"],
            authorization=chain["authorization"],
            authorized_plan=chain["plan"],
            inspect_value=chain["inspect"],
            input_binding=chain["input_binding"],
        )


def test_authorization_rejects_expired_or_date_only_timestamp(chain):
    for issued in ("2020-01-01Z", "2020-01-01T00:00:00Z"):
        forged = deepcopy(chain["authorization"])
        forged["issued_at_utc"] = issued
        forged = _reseal(forged, "authorization_sha256")
        with pytest.raises(ContractError):
            binding.validate_run_authorization(forged, chain["readiness"])


def test_mounted_bundle_rehashes_source_and_rejects_duplicate_json(chain):
    _write_control_bundle(chain)
    authority = binding.validate_mounted_authorization_bundle(
        chain["control"], expected_source_commit=SOURCE_COMMIT
    )
    assert authority.run_authorization_sha256 == chain["authorization"]["authorization_sha256"]
    target = chain["control"] / "TEST_REPORT.json"
    target.chmod(0o644)
    target.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    target.chmod(0o444)
    with pytest.raises(ContractError):
        binding.validate_mounted_authorization_bundle(
            chain["control"], expected_source_commit=SOURCE_COMMIT
        )


def test_manifest_closes_runtime_and_rejects_dirty_label_only_image(
    chain, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_control_bundle(chain)
    workspace = tmp_path / "workspace"
    _copy_manifest_runtime(chain["source_manifest"], workspace)
    monkeypatch.setattr(binding, "CONTAINER_WORKSPACE_DIR", workspace)
    binding.validate_mounted_authorization_bundle(
        chain["control"], expected_source_commit=SOURCE_COMMIT
    )
    dirty = workspace / "fedcore/experiments/v3_train.py"
    dirty.write_bytes(dirty.read_bytes() + b"\n# dirty image bytes\n")
    with pytest.raises(ContractError, match="bound scientific source bytes differ"):
        binding.validate_mounted_authorization_bundle(
            chain["control"], expected_source_commit=SOURCE_COMMIT
        )


def test_source_probe_rejects_dirty_image_despite_matching_labels(
    chain, tmp_path: Path
):
    workspace = tmp_path / "extracted-workspace"
    _copy_manifest_runtime(chain["source_manifest"], workspace)
    dirty = workspace / "fedcore/experiments/v3_train.py"
    dirty.write_bytes(dirty.read_bytes() + b"\n# dirty built image\n")
    with pytest.raises(ContractError, match="bound scientific source bytes differ"):
        binding.build_source_probe_receipt(
            source_manifest_path=chain["source_manifest"],
            extracted_workspace_root=workspace,
            image_id=IMAGE_ID,
            container_inspect_path=chain["source_probe_inspect"],
        )


def test_source_probe_rejects_gpu_request_or_pacs_mount(chain, tmp_path: Path):
    workspace = tmp_path / "extracted-workspace"
    _copy_manifest_runtime(chain["source_manifest"], workspace)
    for field in ("gpu", "mount"):
        inspect = json.loads(chain["source_probe_inspect"].read_text(encoding="utf-8"))
        if field == "gpu":
            inspect[0]["HostConfig"]["DeviceRequests"] = [{"Capabilities": [["gpu"]]}]
        else:
            inspect[0]["Mounts"] = [
                {
                    "Type": "bind",
                    "Source": "/inputs/PACS_mirror.zip",
                    "Destination": "/inputs/PACS_mirror.zip",
                    "RW": False,
                }
            ]
        inspect_path = tmp_path / f"probe-{field}.json"
        write_json(inspect_path, inspect)
        with pytest.raises(ContractError, match="must not"):
            binding.build_source_probe_receipt(
                source_manifest_path=chain["source_manifest"],
                extracted_workspace_root=workspace,
                image_id=IMAGE_ID,
                container_inspect_path=inspect_path,
            )


def test_source_probe_rejects_excluded_unbound_runtime_artifact(chain, tmp_path: Path):
    workspace = tmp_path / "extracted-workspace"
    _copy_manifest_runtime(chain["source_manifest"], workspace)
    cache = workspace / "fedcore/experiments/__pycache__/unbound.cpython-312.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"unbound bytecode")
    with pytest.raises(ContractError, match="excluded unbound runtime artifacts"):
        binding.build_source_probe_receipt(
            source_manifest_path=chain["source_manifest"],
            extracted_workspace_root=workspace,
            image_id=IMAGE_ID,
            container_inspect_path=chain["source_probe_inspect"],
        )


def test_source_manifest_rejects_dirty_worktree_claimed_as_clean_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository = tmp_path / "repository"
    runtime = repository / "fedcore/experiments/v3_train.py"
    dockerfile = repository / "docker/Dockerfile.v3_hsb_scientific"
    requirements = repository / "requirements.lock"
    runtime.parent.mkdir(parents=True)
    dockerfile.parent.mkdir(parents=True)
    runtime.write_text("SEALED = 1\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text("[project]\nname='sealed'\n", encoding="utf-8")
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    requirements.write_text("numpy==1.0\n", encoding="utf-8")
    source_commit = "1" * 40
    source_tree = "2" * 40
    committed_blobs = {
        path.relative_to(repository).as_posix(): path.read_bytes()
        for path in (runtime, repository / "pyproject.toml", dockerfile, requirements)
    }
    runtime.write_text("SEALED = 2\n", encoding="utf-8")

    monkeypatch.setattr(binding, "SCIENTIFIC_RUNTIME_SINGLETONS", ("pyproject.toml",))
    monkeypatch.setattr(binding, "SCIENTIFIC_RUNTIME_ROOTS", ("fedcore",))
    monkeypatch.setattr(binding, "_validate_test_runtime_wheelhouse", lambda *_: None)
    monkeypatch.setattr(
        binding,
        "SCIENTIFIC_BUILD_FILES",
        ("docker/Dockerfile.v3_hsb_scientific", "requirements.lock"),
    )
    monkeypatch.setattr(
        binding,
        "REQUIRED_RUNTIME_SOURCE_FILES",
        ("fedcore/experiments/v3_train.py",),
    )
    monkeypatch.setattr(
        binding,
        "_git_bytes",
        _fake_git_object_reader(
            source_commit=source_commit,
            source_tree=source_tree,
            committed_blobs=committed_blobs,
            runtime_paths=("fedcore/experiments/v3_train.py", "pyproject.toml"),
        ),
    )
    manifest = binding.build_scientific_source_manifest(
        repository,
        source_commit=source_commit,
        source_tree=source_tree,
    )
    with pytest.raises(ContractError, match="bytes differ from Git commit"):
        binding._validate_manifest_against_git_commit(manifest, repository)


def test_test_report_rejects_dirty_test_claimed_as_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository = tmp_path / "repository"
    test_path = "tests/test_gate.py"
    test_file = repository / test_path
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_gate():\n    assert True\n", encoding="utf-8")
    source_commit = "3" * 40
    committed_test = test_file.read_bytes()
    test_file.write_text("def test_gate():\n    assert False\n", encoding="utf-8")
    monkeypatch.setattr(binding, "REQUIRED_EXECUTION_TEST_FILES", (test_path,))
    monkeypatch.setattr(
        binding,
        "_git_bytes",
        _fake_git_object_reader(
            source_commit=source_commit,
            source_tree="4" * 40,
            committed_blobs={test_path: committed_test},
            runtime_paths=(),
        ),
    )
    report = {
        "source_commit": source_commit,
        "test_file_sha256": {
            test_path: binding._hash_regular_file(test_file, "dirty test fixture")
        },
    }
    with pytest.raises(ContractError, match="bytes differ from Git commit"):
        binding._validate_test_report_against_git_commit(report, repository)


def test_source_manifest_rejects_required_file_omission(chain):
    manifest = json.loads(chain["source_manifest"].read_text(encoding="utf-8"))
    manifest["runtime_files"] = [
        row
        for row in manifest["runtime_files"]
        if row["path"] != "fedcore/experiments/v3_train.py"
    ]
    manifest["runtime_inventory_sha256"] = canonical_json_sha256(
        manifest["runtime_files"]
    )
    with pytest.raises(ContractError, match="omits a required runtime source"):
        binding.validate_scientific_source_manifest(
            manifest, expected_source_commit=SOURCE_COMMIT
        )


def test_runtime_receipt_rejects_cpu_limit_drift(chain):
    forged = deepcopy(chain["inspect"])
    forged[0]["HostConfig"]["NanoCpus"] = 1_000_000_000
    with pytest.raises(ContractError, match="CPU limit drift"):
        binding.build_runtime_receipt_from_inspect(
            forged,
            readiness=chain["readiness"],
            authorization=chain["authorization"],
            authorized_plan=chain["plan"],
            input_binding=chain["input_binding"],
        )


def test_exclusive_lease_allows_exactly_one_writer(tmp_path: Path):
    lease = tmp_path / "lease.json"
    binding.acquire_exclusive_lease(lease, {"attempt": 1})
    with pytest.raises(ContractError, match="already exists"):
        binding.acquire_exclusive_lease(lease, {"attempt": 2})


def test_build_binding_has_no_duplicate_literal_dict_keys():
    source = (
        REPOSITORY
        / "paper/ijar-v34-pacs-fresh-seed-hsb-v3-execution-binding-r3/build_binding.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    duplicates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        seen: dict[str, int] = {}
        for key in node.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            if key.value in seen:
                duplicates.append((key.value, seen[key.value], key.lineno))
            seen[key.value] = key.lineno
    assert duplicates == []
