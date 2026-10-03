from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from fedcore.experiments.v3_contract import (
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    ContractError,
    canonical_json_bytes,
    canonical_json_sha256,
    verify_scientific_contract,
)
import fedcore.experiments.v3_execution_binding as binding
from fedcore.experiments import v3_host_executor as host
from fedcore.experiments import v3_local_control_issuer as issuer
import fedcore.experiments.v3_mount_receipt as mount_receipt_module
import fedcore.experiments.v3_scientific_runner as runner
from fedcore.experiments.v3_orchestrator import write_json


REPOSITORY = Path(__file__).resolve().parents[1]
# Historical bundles.  None of them may satisfy the control-schema-2 chain.
LOCAL_V1_STATIC = (
    REPOSITORY
    / "paper"
    / "ijar-v34-pacs-author-deterministic-rerun-local-v1-static-binding-r1"
)
R2_STATIC = REPOSITORY / "paper" / "ijar-v34-pacs-fresh-seed-hsb-v3-static-input-binding-r2"
R3_BUNDLE = REPOSITORY / "paper" / "ijar-v34-pacs-fresh-seed-hsb-v3-execution-binding-r3"
PREBINDING_GATE = (
    REPOSITORY / "paper/ijar-v34-pacs-fresh-seed-hsb-v3-execution-binding-r1/EXECUTION_GATE.json"
)
LEGACY_V3_ROOT = "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_v3_exec"
LEGACY_V4_ROOT = "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_v4_exec"
SOURCE_COMMIT = "a" * 40
SOURCE_TREE = "f" * 40
IMAGE_ID = "sha256:" + "b" * 64
OTHER_IMAGE_ID = "sha256:" + "7" * 64
OLD_PROBE_IMAGE_ID = "sha256:943f4ca6d930c839489a8de50b0acef434c0d5d9b5775280c6d757584aaaef94"
CONTAINER_ID = "c" * 64
PROBE_CONTAINER_ID = "e" * 64
MOUNT_PROBE_CONTAINER_ID = "8" * 64
TEST_CONTAINER_ID = "9" * 64
IMAGE_ENV = [
    "PATH=/opt/conda/bin:/usr/bin:/bin",
    f"FEDCORE_SOURCE_COMMIT={SOURCE_COMMIT}",
    "PYTHONPATH=/workspace",
]
NONCE = "d" * 64


def _reseal(value: dict, field: str) -> dict:
    body = {
        key: item
        for key, item in value.items()
        if key not in {field, f"{field}_definition"}
    }
    return binding._self_hashed(body, field)


def _reseal_plan(plan: dict) -> dict:
    body = {
        key: value
        for key, value in plan.items()
        if key not in {"launch_plan_sha256", "launch_plan_hash_definition"}
    }
    body["execution_template_sha256"] = canonical_json_sha256(body["execution_template"])
    body["launch_plan_sha256"] = canonical_json_sha256(body)
    body["launch_plan_hash_definition"] = binding.LAUNCH_PLAN_HASH_DEFINITION
    return body


def _fake_git_object_reader(
    *,
    source_commit: str,
    source_tree: str,
    committed_blobs: dict[str, bytes],
    runtime_paths: tuple[str, ...],
    runtime_pathspecs: tuple[str, ...],
):
    """Return deterministic Git-object bytes without requiring a Git executable.

    These unit tests isolate the fail-closed comparison so the same-image
    execution gate does not acquire an unregistered system package dependency.
    """

    def reader(_repository: Path, arguments, _label: str) -> bytes:
        argv = tuple(arguments)
        if argv == ("rev-parse", "--verify", f"{source_commit}^{{commit}}"):
            return f"{source_commit}\n".encode("ascii")
        if argv == ("rev-parse", "--verify", f"{source_commit}^{{tree}}"):
            return f"{source_tree}\n".encode("ascii")
        if argv == (
            "ls-tree",
            "-r",
            "-z",
            source_commit,
            "--",
            *runtime_pathspecs,
        ):
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


def _fake_issuer_git(commit: str = SOURCE_COMMIT, tree: str = SOURCE_TREE, status: str = ""):
    def reader(_repository: Path, arguments) -> str:
        argv = tuple(arguments)
        if argv == ("rev-parse", "HEAD"):
            return commit + "\n"
        if argv == ("rev-parse", "HEAD^{tree}"):
            return tree + "\n"
        if argv[:1] == ("status",):
            return status
        raise AssertionError(f"unexpected issuer git query: {argv!r}")

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
    path: Path, source_manifest_path: Path, manifest: dict, *, image_id: str = IMAGE_ID
) -> None:
    write_json(
        path,
        [
            {
                "Id": image_id,
                "Config": {
                    "Labels": _source_labels(source_manifest_path, manifest),
                    "Env": list(IMAGE_ENV),
                    "Entrypoint": ["python"],
                    "User": "1000:1000",
                    "WorkingDir": "/workspace",
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


def _write_junit(path: Path, nodeids=None) -> None:
    nodes = binding.REQUIRED_EXECUTION_TEST_NODEIDS if nodeids is None else nodeids
    cases = "".join(
        '<testcase classname="{0}" name="{1}"/>'.format(
            node.split("::", 1)[0].removesuffix(".py").replace("/", "."),
            node.split("::", 1)[1],
        )
        for node in nodes
    )
    path.write_text(
        f'<testsuites><testsuite name="pytest" tests="{len(nodes)}">{cases}</testsuite></testsuites>\n',
        encoding="utf-8",
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


def _registered_inventory(input_binding: dict) -> list[dict]:
    return [
        {
            "name": row["name"],
            "source": row["staged_path"],
            "provenance_source": row["source_path"],
            "realpath": row["staged_path"],
            "destination": row["container_path"],
            "read_only": True,
            "sha256": row["sha256"],
            "bytes": row["size_bytes"],
            "symlink": False,
        }
        for row in input_binding["files"]
    ]


def _template_plan(image_inspect: Path, control: Path, output: Path, *, image_id: str = IMAGE_ID) -> dict:
    input_binding = json.loads((LOCAL_V1_STATIC / "INPUT_BINDING.json").read_text(encoding="utf-8"))
    return runner.assemble_launch_plan(
        prebinding_gate_path=str(PREBINDING_GATE),
        prebinding_gate_sha256=binding.sha256_file(PREBINDING_GATE),
        input_binding_path=str(LOCAL_V1_STATIC / "INPUT_BINDING.json"),
        input_binding_sha256=binding.sha256_file(LOCAL_V1_STATIC / "INPUT_BINDING.json"),
        inputs=_registered_inventory(input_binding),
        image_id=image_id,
        image_tag="fedcore-v3-hsb:synthetic-final",
        image_inspect_sha256=binding.sha256_file(image_inspect),
        source_commit=SOURCE_COMMIT,
        environment=list(IMAGE_ENV),
        cells=runner.fixed_model_cells(),
        output_dir=output,
        input_dir=binding.ISOLATED_INPUT_DIRECTORY,
        control_dir=control,
        log_dir=binding.EXPECTED_HOST_LOG_DIR,
    )


def _probe_inspect(plan: dict, *, image_id: str = IMAGE_ID) -> list[dict]:
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
            "Id": MOUNT_PROBE_CONTAINER_ID,
            "Name": "/fedcore-local-v1-mount-probe-synthetic",
            "Image": image_id,
            "State": {"Status": "exited", "Running": False, "OOMKilled": False, "ExitCode": 0},
            "Config": {
                "User": "1000:1000",
                "Entrypoint": mount_receipt_module.MOUNT_PROBE_ENTRYPOINT,
                "Cmd": mount_receipt_module.MOUNT_PROBE_COMMAND,
                "Env": ["NVIDIA_VISIBLE_DEVICES=void"],
            },
            "HostConfig": {
                "NetworkMode": "none",
                "ReadonlyRootfs": True,
                "Privileged": False,
                "Devices": [],
                "Runtime": "runc",
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges"],
                "DeviceRequests": None,
                "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=1g"},
            },
            "Mounts": mounts,
        }
    ]


def _mount_declaration(plan: dict, control: Path, output: Path, monkeypatch, *, image_id: str = IMAGE_ID) -> dict:
    # Plan integrity re-hashes the five real staged inputs, which are absent
    # from the PACS-free test container; template integrity is checked by the
    # readiness validators below.
    monkeypatch.setattr(mount_receipt_module, "validate_launch_plan_integrity", lambda plan: None)
    return mount_receipt_module.validate_no_gpu_mount_probe(
        _probe_inspect(plan, image_id=image_id),
        plan,
        expected_image_id=image_id,
        probe_log=mount_receipt_module.MOUNT_PROBE_LOG,
        expected_output_dir=output,
        expected_control_dir=control,
    )


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
                "Env": list(IMAGE_ENV),
                "WorkingDir": "/workspace",
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
                "CapAdd": None,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges"],
                "DeviceRequests": [
                    {
                        "Driver": "",
                        "Count": 0,
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


def _copy_manifest_runtime(source_manifest: Path, destination: Path) -> None:
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    for row in manifest["runtime_files"]:
        source = REPOSITORY / row["path"]
        target = destination / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _readonly(path: Path) -> None:
    path.chmod(0o444)


def _replace_bytes(path: Path, payload: bytes) -> None:
    path.chmod(0o644)
    path.write_bytes(payload)
    path.chmod(0o444)


def _actual_hashes(control: Path) -> dict[str, str]:
    return {
        name: binding.sha256_file(control / name) for name in binding.PRE_READINESS_FILENAMES
    }


@pytest.fixture
def ready_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Six pre-readiness artifacts plus readiness issued by the real local issuer."""

    control = tmp_path / "control"
    output = tmp_path / "output"
    evidence = tmp_path / "evidence"
    for directory in (control, output, evidence):
        directory.mkdir()
    control = control.resolve()
    output = output.resolve()
    monkeypatch.setattr(binding, "EXPECTED_HOST_CONTROL_DIR", control)
    monkeypatch.setattr(binding, "EXPECTED_HOST_OUTPUT_DIR", output)
    monkeypatch.setattr(binding, "CONTAINER_WORKSPACE_DIR", REPOSITORY)
    monkeypatch.setattr(
        binding, "_validate_manifest_against_git_commit", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        binding, "_validate_test_report_against_git_commit", lambda *args, **kwargs: None
    )
    source_manifest = control / "SOURCE_MANIFEST.json"
    manifest = _build_source_manifest(source_manifest)
    image_inspect = tmp_path / "IMAGE_INSPECT.json"
    _write_image_inspect(image_inspect, source_manifest, manifest)
    source_probe_inspect = tmp_path / "SOURCE_PROBE_INSPECT.json"
    _write_source_probe_inspect(source_probe_inspect)
    probe_workspace = tmp_path / "probe-workspace"
    _copy_manifest_runtime(source_manifest, probe_workspace)
    write_json(
        control / "SOURCE_PROBE_RECEIPT.json",
        binding.build_source_probe_receipt(
            source_manifest_path=source_manifest,
            extracted_workspace_root=probe_workspace,
            image_id=IMAGE_ID,
            container_inspect_path=source_probe_inspect,
        ),
    )
    test_container_inspect = evidence / "TEST_CONTAINER_INSPECT.json"
    test_junit = evidence / "TEST_JUNIT.xml"
    test_stdout = evidence / "TEST_STDOUT.log"
    _write_test_container_inspect(test_container_inspect)
    _write_junit(test_junit)
    test_stdout.write_text("all registered tests passed\n", encoding="utf-8")
    _write_test_report(
        control / "TEST_REPORT.json",
        source_manifest,
        test_container_inspect,
        test_junit,
        test_stdout,
    )
    shutil.copyfile(LOCAL_V1_STATIC / "INPUT_BINDING.json", control / "INPUT_BINDING.json")
    plan = _template_plan(image_inspect, control, output)
    write_json(control / "HOST_LAUNCH_PLAN.json", plan)
    declaration = _mount_declaration(plan, control, output, monkeypatch)
    write_json(control / "MOUNT_RECEIPT.json", declaration)
    for path in (*control.iterdir(), image_inspect):
        _readonly(path)
    readiness = issuer.issue_readiness(
        repository=REPOSITORY,
        image_inspect_path=image_inspect,
        test_container_inspect_path=test_container_inspect,
        test_junit_path=test_junit,
        test_stdout_path=test_stdout,
        control_dir=control,
        output_dir=output,
        reader=_fake_issuer_git(),
    )
    return {
        "tmp_path": tmp_path,
        "control": control,
        "output": output,
        "evidence": evidence,
        "image_inspect": image_inspect,
        "source_manifest": source_manifest,
        "source_probe_inspect": source_probe_inspect,
        "source_probe_receipt": control / "SOURCE_PROBE_RECEIPT.json",
        "test_container_inspect": test_container_inspect,
        "test_junit": test_junit,
        "test_stdout": test_stdout,
        "test_report": control / "TEST_REPORT.json",
        "readiness": readiness,
        "static_plan": plan,
        "input_binding": json.loads((control / "INPUT_BINDING.json").read_text(encoding="utf-8")),
        "mount_receipt": declaration,
    }


@pytest.fixture
def chain(ready_chain):
    control = ready_chain["control"]
    issued = datetime.now(timezone.utc) - timedelta(minutes=1)
    authorization, plan = issuer.issue_authorization(
        confirmation=issuer.AUTHORIZATION_CONFIRMATION,
        control_dir=control,
        issued_at=issued,
        valid_for=timedelta(hours=2),
        nonce=NONCE,
    )
    inspect = _docker_inspect(ready_chain["readiness"], plan)
    receipt = binding.build_runtime_receipt_from_inspect(
        inspect,
        readiness=ready_chain["readiness"],
        authorization=authorization,
        authorized_plan=plan,
        input_binding=ready_chain["input_binding"],
        launch_plan=ready_chain["static_plan"],
    )
    return {**ready_chain, "authorization": authorization, "plan": plan, "inspect": inspect, "receipt": receipt}


def _write_control_bundle(chain: dict) -> None:
    for name, value in (
        ("RUNTIME_INSPECT.json", chain["inspect"]),
        ("RUNTIME_INSPECT_RECEIPT.json", chain["receipt"]),
    ):
        path = chain["control"] / name
        if not path.exists():
            write_json(path, value)
            _readonly(path)


def _validate_readiness(chain: dict, readiness: dict | None = None, **overrides) -> str:
    arguments = {
        "static_launch_plan": chain["static_plan"],
        "input_binding": chain["input_binding"],
        "mount_receipt": chain["mount_receipt"],
        "artifact_sha256": _actual_hashes(chain["control"]),
    }
    arguments.update(overrides)
    return binding.validate_execution_readiness(
        chain["readiness"] if readiness is None else readiness, **arguments
    )


def _validate_forged(chain: dict, *, plan=None, receipt=None, binding_record=None) -> str:
    """Validate a fully self-consistent forgery.

    The replaced objects are treated as the actual canonical file bytes, and
    readiness is rewritten so its file hashes, internal hashes, and control
    root all agree with them before it is resealed.  Only the semantic checks
    can then reject the forgery.
    """

    plan = chain["static_plan"] if plan is None else plan
    receipt = chain["mount_receipt"] if receipt is None else receipt
    binding_record = chain["input_binding"] if binding_record is None else binding_record
    hashes = _actual_hashes(chain["control"])
    hashes["HOST_LAUNCH_PLAN.json"] = canonical_json_sha256(plan)
    hashes["MOUNT_RECEIPT.json"] = canonical_json_sha256(receipt)
    hashes["INPUT_BINDING.json"] = canonical_json_sha256(binding_record)
    readiness = deepcopy(chain["readiness"])
    readiness["launch_plan_file_sha256"] = hashes["HOST_LAUNCH_PLAN.json"]
    readiness["mount_receipt_sha256"] = hashes["MOUNT_RECEIPT.json"]
    readiness["input_binding_sha256"] = hashes["INPUT_BINDING.json"]
    readiness["launch_plan_sha256"] = plan.get("launch_plan_sha256")
    readiness["execution_template_sha256"] = plan.get("execution_template_sha256")
    try:
        readiness["pre_readiness_control_root_sha256"] = binding.pre_readiness_control_root_sha256(
            source_commit=readiness["source_commit"],
            image_id=readiness["image_id"],
            execution_template_sha256_value=plan.get("execution_template_sha256"),
            file_sha256=hashes,
        )
    except ContractError:
        readiness["pre_readiness_control_root_sha256"] = "0" * 64
    return binding.validate_execution_readiness(
        _reseal(readiness, "readiness_sha256"),
        static_launch_plan=plan,
        input_binding=binding_record,
        mount_receipt=receipt,
        artifact_sha256=hashes,
    )


def _receipt_from(chain: dict, inspect) -> dict:
    return binding.build_runtime_receipt_from_inspect(
        inspect,
        readiness=chain["readiness"],
        authorization=chain["authorization"],
        authorized_plan=chain["plan"],
        input_binding=chain["input_binding"],
        launch_plan=chain["static_plan"],
    )


# --------------------------------------------------------------------------
# end-to-end chain and the template/renderer
# --------------------------------------------------------------------------


def test_dynamic_chain_binds_one_final_image_end_to_end(chain, monkeypatch):
    plan = chain["static_plan"]
    readiness = chain["readiness"]
    authorization = chain["authorization"]
    images = {
        plan["image"]["id"],
        plan["execution_template"]["image_id"],
        chain["mount_receipt"]["image_id"],
        readiness["image_id"],
        authorization["image_id"],
        chain["plan"]["image_id"],
        chain["inspect"][0]["Image"],
        chain["receipt"]["image_id"],
    }
    assert images == {IMAGE_ID}
    assert {
        plan["input_binding_sha256"],
        chain["mount_receipt"]["input_binding_sha256"],
        readiness["input_binding_sha256"],
        authorization["input_binding_sha256"],
    } == {binding.STATIC_INPUT_BINDING_FILE_SHA256}
    assert chain["mount_receipt"]["launch_plan_sha256"] == plan["launch_plan_sha256"]
    actual = _actual_hashes(chain["control"])
    assert readiness["launch_plan_file_sha256"] == actual["HOST_LAUNCH_PLAN.json"]
    assert readiness["mount_receipt_sha256"] == actual["MOUNT_RECEIPT.json"]
    assert readiness["pre_readiness_control_root_sha256"] == binding.pre_readiness_control_root_sha256(
        source_commit=SOURCE_COMMIT,
        image_id=IMAGE_ID,
        execution_template_sha256_value=plan["execution_template_sha256"],
        file_sha256=actual,
    )
    for field in (
        "readiness_sha256",
        "execution_template_sha256",
        "renderer_version",
        "pre_readiness_control_root_sha256",
        "launch_plan_file_sha256",
        "launch_plan_sha256",
        "mount_receipt_sha256",
        "source_commit",
    ):
        assert authorization[field] == readiness[field]
    assert binding.validate_runtime_receipt(
        chain["receipt"],
        readiness=readiness,
        authorization=authorization,
        authorized_plan=chain["plan"],
        inspect_value=chain["inspect"],
        input_binding=chain["input_binding"],
        launch_plan=plan,
    ) == chain["receipt"]["runtime_receipt_sha256"]
    _write_control_bundle(chain)
    authority = binding.validate_mounted_authorization_bundle(
        chain["control"], expected_source_commit=SOURCE_COMMIT
    )
    assert authority.image_id == IMAGE_ID
    assert authority.run_authorization_sha256 == authorization["authorization_sha256"]

    # The host executor runs exactly the validated authorized-plan argv.
    for path in (chain["control"] / "RUNTIME_INSPECT.json", chain["control"] / "RUNTIME_INSPECT_RECEIPT.json"):
        path.chmod(0o644)
        path.unlink()
    monkeypatch.setattr(host, "EXPECTED_HOST_CONTROL_DIR", chain["control"])
    monkeypatch.setattr(host, "EXPECTED_HOST_OUTPUT_DIR", chain["output"])
    commands = []

    def runner_stub(argv, **kwargs):
        commands.append(list(argv))
        if argv[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(list(argv), 0, CONTAINER_ID + "\n", "")
        if argv[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(list(argv), 0, json.dumps(chain["inspect"]), "")
        raise AssertionError(argv)

    prepared = host.prepare_scientific_container(
        control_dir=chain["control"],
        output_dir=chain["output"],
        lease_path=chain["tmp_path"] / "prepare.lease",
        runner=runner_stub,
    )
    assert prepared["execution_started"] is False
    assert commands[0] == chain["plan"]["docker_create_argv"]


def test_template_declares_docker_create_campaign_runner_not_legacy_run(chain):
    template = chain["static_plan"]["execution_template"]
    assert template["lifecycle"] == ["docker create", "docker inspect", "docker start --attach"]
    assert template["module"] == "fedcore.experiments.v3_campaign_runner"
    assert template["subcommand"] == "run"
    assert template["authorization_directory"] == "/authorization"
    assert template["device"] == "cuda:0"
    assert template["container_name_rule"] == "fedcore-v3-{authorization_id}"
    assert template["restart_policy"] == "no"
    assert template["nano_cpus"] == binding.DOCKER_NANO_CPUS
    assert template["gpu_device_ids"] == [binding.EXPECTED_GPU_UUID]
    mounts = {row["destination"]: row for row in template["mounts"]}
    assert mounts["/authorization"] == {
        "type": "bind",
        "source": str(chain["control"]),
        "destination": "/authorization",
        "read_only": True,
    }
    assert mounts["/output"]["read_only"] is False
    assert sum(1 for row in template["mounts"] if row["destination"].startswith("/inputs/")) == 5
    assert all(mounts[f"/inputs/{spec.name}"]["read_only"] for spec in binding.REGISTERED_SCIENTIFIC_INPUTS)
    assert "cell_launches" not in chain["static_plan"]
    argv = chain["plan"]["docker_create_argv"]
    assert argv[:2] == ["docker", "create"]
    assert "run" not in argv[:2] and "--rm" not in argv
    assert "fedcore.experiments.v3_train" not in argv
    assert argv[argv.index(IMAGE_ID) + 1 :] == list(binding.SCIENTIFIC_COMMAND)
    for flag, value in (("--cpus", "0.5"), ("--restart", "no"), ("--user", "1000:1000")):
        assert argv[argv.index(flag) + 1] == value
    assert f"type=bind,src={chain['control']},dst=/authorization,readonly" in argv
    template_text = json.dumps(chain["static_plan"])
    for authorization_value in (
        chain["authorization"]["authorization_sha256"],
        chain["plan"]["authorized_plan_sha256"],
        chain["plan"]["container_name"],
    ):
        assert authorization_value not in template_text


def test_template_rejects_legacy_docker_run_rm_and_per_cell_train(chain):
    template = chain["static_plan"]["execution_template"]
    legacy = deepcopy(template)
    legacy["container_command"] = ["-m", "fedcore.experiments.v3_train", "--model-id", "x"]
    with pytest.raises(ContractError):
        binding.render_authorized_execution(legacy, "1" * 64)
    lifecycle = deepcopy(template)
    lifecycle["lifecycle"] = ["docker run --rm"]
    with pytest.raises(ContractError, match="lifecycle drift"):
        binding.render_authorized_execution(lifecycle, "1" * 64)
    legacy_plan = json.loads((R2_STATIC / "HOST_LAUNCH_PLAN.json").read_text(encoding="utf-8"))
    assert legacy_plan["cell_launches"][0]["docker_argv_template"][:3] == ["docker", "run", "--rm"]
    with pytest.raises(ContractError, match="not control schema 2"):
        binding.validate_launch_template_plan(
            legacy_plan,
            input_binding=chain["input_binding"],
            expected_image_id=IMAGE_ID,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_authorized_plan_is_deterministic_render_of_template_and_authorization(chain):
    authorization = chain["authorization"]
    expected_id = binding.authorization_id_for(
        readiness_sha256=chain["readiness"]["readiness_sha256"],
        authorization_nonce=NONCE,
        issued_at_utc=authorization["issued_at_utc"],
        valid_until_utc=authorization["valid_until_utc"],
    )
    assert authorization["authorization_id"] == expected_id
    assert authorization["authorization_id"] != authorization["authorization_sha256"]
    assert chain["plan"]["container_name"] == f"fedcore-v3-{expected_id}"
    template = chain["static_plan"]["execution_template"]
    first = binding.render_authorized_execution(template, expected_id)
    second = binding.render_authorized_execution(deepcopy(template), expected_id)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert authorization["instantiated_execution_configuration_sha256"] == (
        binding.instantiated_execution_configuration_sha256(first)
    )
    rebuilt = binding.build_authorized_campaign_plan(
        chain["readiness"],
        authorization,
        input_binding=chain["input_binding"],
        launch_plan=chain["static_plan"],
    )
    assert canonical_json_bytes(rebuilt) == (chain["control"] / "AUTHORIZED_CAMPAIGN_PLAN.json").read_bytes()
    # One direction only: the plan references the authorization hash, never the reverse.
    assert chain["plan"]["run_authorization_sha256"] == authorization["authorization_sha256"]
    assert "authorized_plan_sha256" not in authorization
    assert chain["plan"]["authorized_plan_sha256"] not in json.dumps(authorization)
    other = binding.render_authorized_execution(template, "2" * 64)
    assert other["container_name"] != first["container_name"]


def test_wrong_image_rejected_at_every_stage(chain, monkeypatch):
    plan = deepcopy(chain["static_plan"])
    plan["image"]["id"] = OTHER_IMAGE_ID
    plan["execution_template"]["image_id"] = OTHER_IMAGE_ID
    with pytest.raises(ContractError, match="image ID differs"):
        _validate_forged(chain, plan=_reseal_plan(plan))

    receipt = deepcopy(chain["mount_receipt"])
    receipt["image_id"] = OTHER_IMAGE_ID
    with pytest.raises(ContractError, match="image ID differs"):
        _validate_forged(chain, receipt=receipt)

    readiness = deepcopy(chain["readiness"])
    readiness["image_id"] = OTHER_IMAGE_ID
    with pytest.raises(ContractError):
        _validate_readiness(chain, _reseal(readiness, "readiness_sha256"))

    authorization = deepcopy(chain["authorization"])
    authorization["image_id"] = OTHER_IMAGE_ID
    with pytest.raises(ContractError, match="image_id mismatch"):
        binding.validate_run_authorization(
            _reseal(authorization, "authorization_sha256"), chain["readiness"], chain["static_plan"]
        )

    inspect = deepcopy(chain["inspect"])
    inspect[0]["Image"] = OTHER_IMAGE_ID
    with pytest.raises(ContractError, match="image ID mismatch"):
        _receipt_from(chain, inspect)


def test_execution_configuration_drift_rejected(chain):
    mutations = {
        "cmd": lambda c: c["Config"].__setitem__("Cmd", c["Config"]["Cmd"] + ["--resume"]),
        "environment": lambda c: c["Config"]["Env"].append("EXTRA=1"),
        "working_dir": lambda c: c["Config"].__setitem__("WorkingDir", "/tmp"),
        "user": lambda c: c["Config"].__setitem__("User", "0:0"),
        "cpu": lambda c: c["HostConfig"].__setitem__("NanoCpus", 1_000_000_000),
        "gpu": lambda c: c["HostConfig"]["DeviceRequests"][0].__setitem__("DeviceIDs", ["GPU-other"]),
        "mount": lambda c: c["Mounts"][0].__setitem__("Source", "/tmp/elsewhere"),
        "authorization_rw": lambda c: next(
            row for row in c["Mounts"] if row["Destination"] == "/authorization"
        ).__setitem__("RW", True),
        "security": lambda c: c["HostConfig"].__setitem__("SecurityOpt", []),
        "restart": lambda c: c["HostConfig"].__setitem__("RestartPolicy", {"Name": "always"}),
        "name": lambda c: c.__setitem__("Name", "/fedcore-v3-other"),
        "cap_add": lambda c: c["HostConfig"].__setitem__("CapAdd", ["SYS_ADMIN"]),
    }
    rejected = []
    for label, mutate in mutations.items():
        inspect = deepcopy(chain["inspect"])
        mutate(inspect[0])
        try:
            _receipt_from(chain, inspect)
        except ContractError:
            rejected.append(label)
    assert rejected == list(mutations)


def test_resealed_upstream_artifacts_rejected(chain):
    plan = deepcopy(chain["static_plan"])
    plan["execution_template"]["restart_policy"] = "always"
    with pytest.raises(ContractError, match="execution template drift"):
        _validate_forged(chain, plan=_reseal_plan(plan))

    plan = deepcopy(chain["static_plan"])
    plan["execution_template"]["container_command"] = list(binding.SCIENTIFIC_COMMAND[:-1]) + ["cpu"]
    with pytest.raises(ContractError, match="execution template drift"):
        _validate_forged(chain, plan=_reseal_plan(plan))

    receipt = deepcopy(chain["mount_receipt"])
    receipt["mounts"] = receipt["mounts"][1:]
    with pytest.raises(ContractError, match="mount scope"):
        _validate_forged(chain, receipt=receipt)

    readiness = deepcopy(chain["readiness"])
    readiness["launch_plan_file_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="differs from the actual"):
        _validate_readiness(chain, _reseal(readiness, "readiness_sha256"))

    binding_record = deepcopy(chain["input_binding"])
    binding_record["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ContractError, match="static anchor"):
        _validate_forged(chain, binding_record=binding_record)
    with pytest.raises(ContractError, match="registry drift"):
        binding.validate_static_input_binding(binding_record)

    authorization = deepcopy(chain["authorization"])
    authorization["instantiated_execution_configuration_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="instantiated execution configuration"):
        binding.validate_run_authorization(
            _reseal(authorization, "authorization_sha256"), chain["readiness"], chain["static_plan"]
        )

    plan = deepcopy(chain["plan"])
    plan["docker_create_argv"] = plan["docker_create_argv"] + ["--privileged"]
    with pytest.raises(ContractError, match="deterministic render"):
        binding.validate_authorized_campaign_plan(
            _reseal(plan, "authorized_plan_sha256"),
            chain["readiness"],
            chain["authorization"],
            chain["input_binding"],
            chain["static_plan"],
        )


def test_role_swapped_labeled_control_root_rejected(chain):
    actual = _actual_hashes(chain["control"])
    swapped = dict(actual)
    swapped["HOST_LAUNCH_PLAN.json"], swapped["MOUNT_RECEIPT.json"] = (
        actual["MOUNT_RECEIPT.json"],
        actual["HOST_LAUNCH_PLAN.json"],
    )
    kwargs = {
        "source_commit": SOURCE_COMMIT,
        "image_id": IMAGE_ID,
        "execution_template_sha256_value": chain["readiness"]["execution_template_sha256"],
    }
    assert binding.pre_readiness_control_root_sha256(file_sha256=swapped, **kwargs) != (
        binding.pre_readiness_control_root_sha256(file_sha256=actual, **kwargs)
    )
    with pytest.raises(ContractError):
        _validate_readiness(chain, artifact_sha256=swapped)
    readiness = deepcopy(chain["readiness"])
    readiness["launch_plan_file_sha256"], readiness["mount_receipt_sha256"] = (
        readiness["mount_receipt_sha256"],
        readiness["launch_plan_file_sha256"],
    )
    readiness["pre_readiness_control_root_sha256"] = binding.pre_readiness_control_root_sha256(
        file_sha256=swapped, **kwargs
    )
    with pytest.raises(ContractError):
        _validate_readiness(chain, _reseal(readiness, "readiness_sha256"), artifact_sha256=swapped)


def test_byte_only_drift_rejected(chain):
    for name in ("HOST_LAUNCH_PLAN.json", "MOUNT_RECEIPT.json", "INPUT_BINDING.json"):
        path = chain["control"] / name
        original = path.read_bytes()
        value = json.loads(original)
        reordered = json.dumps(value, sort_keys=False, indent=1).encode("utf-8") + b"\n"
        for drifted in (original[:-1] + b" \n", reordered):
            _replace_bytes(path, drifted)
            with pytest.raises(ContractError):
                _validate_readiness(chain)
        _replace_bytes(path, original)
    _write_control_bundle(chain)
    plan_path = chain["control"] / "AUTHORIZED_CAMPAIGN_PLAN.json"
    original = plan_path.read_bytes()
    _replace_bytes(plan_path, json.dumps(json.loads(original), indent=2).encode("utf-8"))
    with pytest.raises(ContractError, match="authorized-plan bytes"):
        binding.validate_mounted_authorization_bundle(
            chain["control"], expected_source_commit=SOURCE_COMMIT
        )


def test_old_v1_r2_r3_bundles_rejected(chain):
    for bundle in (LOCAL_V1_STATIC, R2_STATIC):
        old_plan = json.loads((bundle / "HOST_LAUNCH_PLAN.json").read_text(encoding="utf-8"))
        old_receipt = json.loads((bundle / "MOUNT_RECEIPT.json").read_text(encoding="utf-8"))
        for image_id in (IMAGE_ID, old_plan["image"]["id"]):
            with pytest.raises(ContractError, match="not control schema 2"):
                binding.validate_launch_template_plan(
                    old_plan,
                    input_binding=chain["input_binding"],
                    expected_image_id=image_id,
                    expected_source_commit=SOURCE_COMMIT,
                )
        with pytest.raises(ContractError, match="not control schema 2"):
            binding.validate_mount_declaration(
                old_receipt, launch_plan=chain["static_plan"], expected_image_id=IMAGE_ID
            )
        with pytest.raises(ContractError):
            _validate_forged(chain, plan=old_plan, receipt=old_receipt)
    # The specific old probe-image trio (local v1, image 943f4ca6...) is refused.
    v1_plan = json.loads((LOCAL_V1_STATIC / "HOST_LAUNCH_PLAN.json").read_text(encoding="utf-8"))
    assert v1_plan["image"]["id"] == OLD_PROBE_IMAGE_ID
    upgraded = _reseal_plan(
        {**deepcopy(chain["static_plan"]), "image": {"tag": "probe", "id": OLD_PROBE_IMAGE_ID}}
    )
    with pytest.raises(ContractError, match="image ID differs"):
        _validate_forged(chain, plan=upgraded)
    r2_binding = json.loads((R2_STATIC / "INPUT_BINDING.json").read_text(encoding="utf-8"))
    with pytest.raises(ContractError):
        _validate_forged(chain, binding_record=r2_binding)
    r3_readiness = json.loads((R3_BUNDLE / "EXECUTION_READINESS.json").read_text(encoding="utf-8"))
    with pytest.raises(ContractError, match="not control schema 2"):
        _validate_readiness(chain, r3_readiness)
    r3_template = json.loads((R3_BUNDLE / "RUN_AUTHORIZATION_TEMPLATE.json").read_text(encoding="utf-8"))
    with pytest.raises(ContractError):
        binding.validate_run_authorization(r3_template, chain["readiness"], chain["static_plan"])


def test_cross_run_artifact_mixing_rejected(chain, tmp_path):
    other_authorization = issuer.build_run_authorization(
        chain["readiness"],
        chain["static_plan"],
        issued_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        valid_for=timedelta(hours=2),
        nonce="5" * 64,
    )
    assert other_authorization["authorization_id"] != chain["authorization"]["authorization_id"]
    with pytest.raises(ContractError):
        binding.validate_authorized_campaign_plan(
            chain["plan"],
            chain["readiness"],
            other_authorization,
            chain["input_binding"],
            chain["static_plan"],
        )
    with pytest.raises(ContractError):
        binding.validate_runtime_receipt(
            chain["receipt"],
            readiness=chain["readiness"],
            authorization=other_authorization,
            authorized_plan=chain["plan"],
            inspect_value=chain["inspect"],
            input_binding=chain["input_binding"],
            launch_plan=chain["static_plan"],
        )
    # A template and declaration rendered for another final image cannot be mixed in.
    other_inspect = tmp_path / "OTHER_IMAGE_INSPECT.json"
    _write_image_inspect(other_inspect, chain["source_manifest"], json.loads(chain["source_manifest"].read_text()), image_id=OTHER_IMAGE_ID)
    other_plan = _template_plan(other_inspect, chain["control"], chain["output"], image_id=OTHER_IMAGE_ID)
    with pytest.raises(ContractError):
        _validate_forged(chain, plan=other_plan)
    with pytest.raises(ContractError):
        binding.validate_run_authorization(chain["authorization"], chain["readiness"], other_plan)


def test_legacy_and_mixed_schema_rejected(chain):
    readiness = chain["readiness"]
    for field, value in (
        ("schema_version", 1),
        ("control_schema_version", 1),
        ("control_schema_version", None),
        ("control_schema_version", "2"),
    ):
        forged = deepcopy(readiness)
        forged[field] = value
        with pytest.raises(ContractError, match="not control schema 2"):
            _validate_readiness(chain, _reseal(forged, "readiness_sha256"))
    for field, value in (("image_id", ""), ("image_id", None), ("source_commit", ""), ("critical_test_nodeids", [])):
        forged = deepcopy(readiness)
        forged[field] = value
        with pytest.raises(ContractError):
            _validate_readiness(chain, _reseal(forged, "readiness_sha256"))
    missing = deepcopy(readiness)
    del missing["pre_readiness_control_root_sha256"]
    with pytest.raises(ContractError, match="key drift"):
        _validate_readiness(chain, _reseal(missing, "readiness_sha256"))
    legacy_extra = deepcopy(readiness)
    legacy_extra["static_binding_root_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="key drift"):
        _validate_readiness(chain, _reseal(legacy_extra, "readiness_sha256"))
    mixed_plan = deepcopy(chain["static_plan"])
    mixed_plan["schema_version"] = 1
    with pytest.raises(ContractError, match="not control schema 2"):
        _validate_forged(chain, plan=_reseal_plan(mixed_plan))
    for artifact, field in (
        (chain["authorization"], "authorization_sha256"),
        (chain["plan"], "authorized_plan_sha256"),
        (chain["receipt"], "runtime_receipt_sha256"),
    ):
        forged = deepcopy(artifact)
        forged["schema_version"] = 1
        forged = _reseal(forged, field)
        with pytest.raises(ContractError, match="not control schema 2"):
            if field == "authorization_sha256":
                binding.validate_run_authorization(forged, readiness, chain["static_plan"])
            elif field == "authorized_plan_sha256":
                binding.validate_authorized_campaign_plan(
                    forged, readiness, chain["authorization"], chain["input_binding"], chain["static_plan"]
                )
            else:
                binding.validate_runtime_receipt(
                    forged,
                    readiness=readiness,
                    authorization=chain["authorization"],
                    authorized_plan=chain["plan"],
                    inspect_value=chain["inspect"],
                    input_binding=chain["input_binding"],
                    launch_plan=chain["static_plan"],
                )


def test_authorization_missing_expired_unissued_or_challenge_mismatch_rejected(chain):
    for issued in ("2020-01-01Z", "2020-01-01T00:00:00Z", "2999-01-01T00:00:00Z"):
        forged = deepcopy(chain["authorization"])
        forged["issued_at_utc"] = issued
        with pytest.raises(ContractError):
            binding.validate_run_authorization(
                _reseal(forged, "authorization_sha256"), chain["readiness"], chain["static_plan"]
            )
    unissued = binding.build_run_authorization_template(chain["readiness"])
    with pytest.raises(ContractError):
        binding.validate_run_authorization(unissued, chain["readiness"], chain["static_plan"])
    challenge = deepcopy(chain["authorization"])
    challenge["authorization_challenge_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="challenge mismatch"):
        binding.validate_run_authorization(
            _reseal(challenge, "authorization_sha256"), chain["readiness"], chain["static_plan"]
        )
    self_hash_id = deepcopy(chain["authorization"])
    self_hash_id["authorization_id"] = chain["authorization"]["authorization_sha256"]
    with pytest.raises(ContractError, match="not derived from the challenge"):
        binding.validate_run_authorization(
            _reseal(self_hash_id, "authorization_sha256"), chain["readiness"], chain["static_plan"]
        )
    _write_control_bundle(chain)
    path = chain["control"] / "RUN_AUTHORIZATION.json"
    saved = path.read_bytes()
    path.chmod(0o644)
    path.unlink()
    with pytest.raises(ContractError, match="content drift"):
        binding.validate_mounted_authorization_bundle(
            chain["control"], expected_source_commit=SOURCE_COMMIT
        )
    path.write_bytes(saved)
    _readonly(path)
    with pytest.raises(ContractError, match="explicit author confirmation"):
        issuer.issue_authorization(confirmation="yes", control_dir=chain["control"])


def test_mounted_bundle_rejects_unsafe_or_incomplete_control_files(chain):
    _write_control_bundle(chain)
    binding.validate_mounted_authorization_bundle(
        chain["control"], expected_source_commit=SOURCE_COMMIT
    )
    target = chain["control"] / "TEST_REPORT.json"
    original = target.read_bytes()

    _replace_bytes(target, b'{"schema_version":1,"schema_version":1}\n')
    with pytest.raises(ContractError, match="duplicate JSON key"):
        binding.validate_mounted_authorization_bundle(chain["control"])
    _replace_bytes(target, original)

    target.chmod(0o644)
    with pytest.raises(ContractError, match="read-only"):
        binding.validate_mounted_authorization_bundle(chain["control"])
    _readonly(target)

    moved = chain["tmp_path"] / "moved-report.json"
    target.rename(moved)
    target.symlink_to(moved)
    with pytest.raises(ContractError, match="regular nonsymlink"):
        binding.validate_mounted_authorization_bundle(chain["control"])
    target.unlink()
    moved.rename(target)

    extra = chain["control"] / "EXTRA.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ContractError, match="content drift"):
        binding.validate_mounted_authorization_bundle(chain["control"])
    extra.unlink()
    binding.validate_mounted_authorization_bundle(
        chain["control"], expected_source_commit=SOURCE_COMMIT
    )


def test_issuer_derives_identity_and_ignores_caller_hashes(ready_chain, tmp_path):
    control = ready_chain["control"]
    readiness_path = control / "EXECUTION_READINESS.json"
    saved = readiness_path.read_bytes()
    readiness_path.chmod(0o644)
    readiness_path.unlink()

    def issue(**overrides):
        arguments = {
            "repository": REPOSITORY,
            "image_inspect_path": ready_chain["image_inspect"],
            "test_container_inspect_path": ready_chain["test_container_inspect"],
            "test_junit_path": ready_chain["test_junit"],
            "test_stdout_path": ready_chain["test_stdout"],
            "control_dir": control,
            "output_dir": ready_chain["output"],
            "reader": _fake_issuer_git(),
        }
        arguments.update(overrides)
        return issuer.issue_readiness(**arguments)

    forged_inspect = tmp_path / "FORGED_IMAGE_INSPECT.json"
    _write_image_inspect(
        forged_inspect,
        ready_chain["source_manifest"],
        json.loads(ready_chain["source_manifest"].read_text()),
        image_id=OTHER_IMAGE_ID,
    )
    _readonly(forged_inspect)
    with pytest.raises(ContractError, match="image ID differs"):
        issue(image_inspect_path=forged_inspect)
    with pytest.raises(ContractError, match="label differs from the Git commit"):
        issue(reader=_fake_issuer_git(commit="1" * 40))
    with pytest.raises(ContractError, match="not clean"):
        issue(reader=_fake_issuer_git(status=" M fedcore/x.py\n"))
    with pytest.raises(TypeError):
        issue(image_id=IMAGE_ID)
    assert not readiness_path.exists()

    readiness_path.write_bytes(saved)
    _readonly(readiness_path)
    forged = json.loads(saved)
    forged["mount_receipt_sha256"] = "1" * 64
    _replace_bytes(readiness_path, canonical_json_bytes(_reseal(forged, "readiness_sha256")))
    with pytest.raises(ContractError, match="differs from the actual"):
        issuer.issue_authorization(
            confirmation=issuer.AUTHORIZATION_CONFIRMATION, control_dir=control
        )
    assert not (control / "RUN_AUTHORIZATION.json").exists()


def test_issuer_rejects_artifact_replacement_before_authorization(ready_chain, monkeypatch):
    control = ready_chain["control"]
    original_build = binding.build_authorized_campaign_plan

    def replace_then_build(*args, **kwargs):
        target = control / "MOUNT_RECEIPT.json"
        payload = target.read_bytes()
        target.chmod(0o644)
        target.unlink()
        target.write_bytes(payload)
        target.chmod(0o444)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(binding, "build_authorized_campaign_plan", replace_then_build)
    with pytest.raises(ContractError, match="changed after validation"):
        issuer.issue_authorization(
            confirmation=issuer.AUTHORIZATION_CONFIRMATION,
            control_dir=control,
            issued_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            nonce=NONCE,
        )
    assert not (control / "RUN_AUTHORIZATION.json").exists()
    assert not (control / "AUTHORIZED_CAMPAIGN_PLAN.json").exists()


def test_issuer_refuses_overwrite_and_unexpected_phase(chain):
    with pytest.raises(ContractError, match="unexpected control phase files"):
        issuer.issue_authorization(
            confirmation=issuer.AUTHORIZATION_CONFIRMATION, control_dir=chain["control"]
        )
    with pytest.raises(ContractError, match="already exists"):
        issuer._write_exclusive(chain["control"] / "RUN_AUTHORIZATION.json", b"{}\n")


def test_wrong_control_input_output_log_paths_rejected(chain):
    for field in ("host_control_dir", "host_input_dir", "host_output_dir", "host_log_dir"):
        for root in (LEGACY_V3_ROOT, LEGACY_V4_ROOT, "/tmp"):
            forged = deepcopy(chain["readiness"])
            forged[field] = root + "/wrong"
            with pytest.raises(ContractError):
                _validate_readiness(chain, _reseal(forged, "readiness_sha256"))
    plan = deepcopy(chain["static_plan"])
    plan["execution_template"]["paths"]["log"] = LEGACY_V3_ROOT + "/logs/execution-r6"
    with pytest.raises(ContractError, match="execution template drift"):
        _validate_forged(chain, plan=_reseal_plan(plan))
    staged = deepcopy(chain["input_binding"])
    staged["files"][0]["staged_path"] = LEGACY_V3_ROOT + "/inputs/PACS_mirror.zip"
    with pytest.raises(ContractError, match="registry drift"):
        binding.validate_static_input_binding(staged)


def test_historical_r3_bundle_unchanged():
    expected = {
        "build_binding.py": "f6302efcb9ec9d84220118ffadadcf91dc6b552486221a4fd9da2ff7e5049af0",
        "EXECUTION_READINESS.json": "f03b75515edc49b77d8d6d320af5c3effe42fb20c3013e7fc5edbed709645375",
        "IMAGE_INSPECT.json": "8fc2b53982e786ceff5ffb43699f8cc22339b7cbf7475caf7da0d41e1e148946",
        "PUBLIC_BUNDLE_MANIFEST.json": "5b04564ec1c305fe22243190c0648895df98841c3c0e12a986b6623c2c9abef4",
        "RUN_AUTHORIZATION_TEMPLATE.json": "82a89e8c8982c20302d16a794baf1144a60ccb356fdc38a4523c556840a862ab",
        "SOURCE_MANIFEST.json": "eb141011fdc2cc6d7fe0c393dcd8212105b2fb114f4090a957b5f9cd0fc3d7da",
        "SOURCE_PROBE_INSPECT.json": "e92cb3cb71470e7c4b3971a00ee895d7f33c6385432f89ac1c7fa9608121a31f",
        "SOURCE_PROBE_RECEIPT.json": "ec0cc8591dc458899e3e1b06c05be4e2a91a259131333325ed0cc6c2cedab593",
        "TEST_CONTAINER_INSPECT.json": "849f9500f9015f2fd03eeb951ddd9469c4e33581dab016032cce7ccacd69f4a2",
        "TEST_JUNIT.xml": "ebc00df689bb4861df45bd3e9d9fb83ef5d49c492e6da24bf8ad379c56c55e24",
        "TEST_REPORT.json": "f2c05ef63e2e9b5d93f3e85e2fb8b62cbaa56e821a3a83e1e313d890ae80c940",
        "TEST_STDOUT.log": "bf0630e9bd74b559464fa7beb7163e3a3734ba66b4c0635e4a0aa32641b9e02e",
    }
    observed = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in R3_BUNDLE.iterdir()
        if path.is_file()
    }
    assert observed == expected
    issuer_source = (REPOSITORY / "fedcore/experiments/v3_local_control_issuer.py").read_text(encoding="utf-8")
    tree = ast.parse(issuer_source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any("build_binding" in name for name in imported)
    assert "execution-binding-r3" not in issuer_source


def test_scientific_contract_invariants_unchanged():
    assert PROTOCOL_ID == "FEDCORE-IJAR-V34-PACS-FRESH-SEED-REPLICATION-HSB-v3"
    assert PROTOCOL_SHA256 == "d7fba4c748d449994b11912acd6004dfd690e5053825169e482eb7f64b2ee728"
    verify_scientific_contract()
    cells = runner.fixed_model_cells()
    assert len(cells) == 30
    assert runner.cell_order_sha256(cells) == (
        "9d5472d2346c8be7d716951cc85d52e2fa282e7baab7593cbb31743cbbf9f614"
    )
    assert [(spec.name, spec.size_bytes, spec.sha256) for spec in binding.REGISTERED_SCIENTIFIC_INPUTS] == [
        ("PACS_mirror.zip", 184417365, "42bf567f1ed8a01d522e47e4a677e2a3149577bbd6fcbb38bedfdd73cb59e147"),
        ("IMAGE_MANIFEST.csv", 1537910, "283d47bd8cfb7017d6e7fde6d65e07ff6479ab25127afcba135da3a06775afae"),
        ("FOLD_MANIFEST.csv", 7853382, "fe5b64d237f47f503b8895aa5e566727bc818e05ee597f84c440df70e91e5344"),
        ("resnet18-f37072fd.pth", 46830571, "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
        ("convnext_tiny-983f1562.pth", 114419221, "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"),
    ]
    assert binding.SCIENTIFIC_COMMAND == (
        "-m", "fedcore.experiments.v3_campaign_runner", "run",
        "--authorization-dir", "/authorization",
        "--pacs-archive", "/inputs/PACS_mirror.zip",
        "--image-manifest", "/inputs/IMAGE_MANIFEST.csv",
        "--fold-manifest", "/inputs/FOLD_MANIFEST.csv",
        "--resnet18-weights", "/inputs/resnet18-f37072fd.pth",
        "--convnext-tiny-weights", "/inputs/convnext_tiny-983f1562.pth",
        "--output-dir", "/output",
        "--device", "cuda:0",
    )
    unchanged = {
        "fedcore/experiments/v3_contract.py": "ba77b92d939e99f4002c0afbf259fe7a1f89f11638be6399be9c836b4fee3f76",
        "fedcore/experiments/v3_hsb.py": "03e4c7065b291f48dcd630a86c92cd9927e2140165dd1ee33433a7e63da50710",
        "fedcore/experiments/v3_proposal.py": "57ebd998a940b177100cee149a1f59edce54e7a110c41c9d95423e675bbf3a54",
        "fedcore/experiments/v3_counts.py": "e6a7b588ba1b0e81d45f5bc2c663c623be96058e894179ee663ec83ecf716258",
        "fedcore/experiments/v3_train.py": "4d133ad7b313057a3c6c13380a1f3cbd6c550c9cd2e6dfe9c1c43e125812ee89",
        "fedcore/experiments/v3_posttrain.py": "26475e9ee2df4023897b791102b94c87e76913e88697883a44d74b9e588624bd",
        "fedcore/experiments/v3_campaign_runner.py": "eec8531937ab2a62ed62119677495bc572ce666af0e252efeb03bdb80dae9149",
        "fedcore/experiments/v3_campaign_state.py": "7686c6e282b9e785259c85fff2fa08bc56d3448cd1d642f9af0f8e45fbfe9757",
        "fedcore/experiments/v3_campaign_finalizer.py": "d29f7617bcc1977a0483c1604f2a491af1533d183288dd948ba2fbc421792844",
        "fedcore/experiments/v3_input_binding.py": "4d47b00aa0f3f11a37893a909ac644af5ae153e52bbf131b5c4cde1b8739985c",
        "fedcore/experiments/v3_pacs_data.py": "d0735ee2ee09a77cbf6f89a887c236c1b940057e749446275423b7265fde2046",
        "fedcore/experiments/v3_orchestrator.py": "1fd6e03402b41c6f5d9b28231413141f06438608978416272c7e767a686afa52",
        "fedcore/experiments/v3_mount_probe.py": "354ff4cb9bd2e2b987b4a3732ff077aae1c29e3050e9c10d2a972233a1ec4594",
        "paper/ijar-v34-pacs-fresh-seed-hsb-v3/SHA256SUMS.txt": "6184584dd51da45fd65c180bd62b62dc451b7f8df4c92d11206e4450cfaf44b1",
    }
    for relative, digest in unchanged.items():
        assert hashlib.sha256((REPOSITORY / relative).read_bytes()).hexdigest() == digest, relative
    protocol = REPOSITORY / "paper/ijar-v34-pacs-fresh-seed-hsb-v3"
    for line in (protocol / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((protocol / name).read_bytes()).hexdigest() == digest, name


# --------------------------------------------------------------------------
# preserved execution-binding tests, adapted to control schema 2
# --------------------------------------------------------------------------


def test_full_artifact_chain_and_mount_order(chain):
    assert binding.validate_runtime_receipt(
        chain["receipt"],
        readiness=chain["readiness"],
        authorization=chain["authorization"],
        authorized_plan=chain["plan"],
        inspect_value=chain["inspect"],
        input_binding=chain["input_binding"],
        launch_plan=chain["static_plan"],
    ) == chain["receipt"]["runtime_receipt_sha256"]
    assert "--cpus" in chain["plan"]["docker_create_argv"]
    cpu_index = chain["plan"]["docker_create_argv"].index("--cpus")
    assert chain["plan"]["docker_create_argv"][cpu_index + 1] == "0.5"
    assert chain["receipt"]["exact_cpu_limit_verified"] is True
    assert [row["destination"] for row in chain["receipt"]["mounts"]] == sorted(
        row["destination"] for row in chain["receipt"]["mounts"]
    )


def test_readiness_precharges_bounded_cleanup_inside_total_gpu_cap(chain):
    readiness = chain["readiness"]
    assert (
        readiness["scientific_execution_timeout_seconds"]
        == (720 * 60 * 60) - 300
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


def test_resource_cap_invariant_rejects_unreserved_720_hour_timeout(monkeypatch):
    monkeypatch.setattr(
        binding, "SCIENTIFIC_EXECUTION_TIMEOUT_SECONDS", 720 * 60 * 60
    )
    monkeypatch.setattr(binding, "REMAINING_SCIENTIFIC_WALLTIME_HOURS", 720.0)
    monkeypatch.setattr(binding, "MAXIMUM_CPU_CORE_HOURS", 360.0)
    with pytest.raises(ContractError, match="exceeds the GPU cap"):
        binding._validate_gpu_resource_cap_invariant()


def test_readiness_rejects_resealed_cleanup_reserve_drift(chain):
    forged = deepcopy(chain["readiness"])
    forged["scientific_cleanup_reserve_seconds"] = 90
    forged["scientific_cleanup_safety_margin_seconds"] = 0
    with pytest.raises(ContractError, match="resource-cap drift"):
        _validate_readiness(chain, _reseal(forged, "readiness_sha256"))


def test_readiness_rejects_self_consistent_command_and_path_forgery(chain):
    for field, replacement in (
        ("container_command", ["-c", "print('not campaign')"]),
        ("host_output_dir", "/tmp/evil-output"),
        ("host_control_dir", "/tmp/evil-control"),
    ):
        forged = deepcopy(chain["readiness"])
        forged[field] = replacement
        with pytest.raises(ContractError):
            _validate_readiness(chain, _reseal(forged, "readiness_sha256"))


def _build_readiness_with(chain, tmp_path: Path, **static_paths: Path) -> dict:
    paths = {
        "static_launch_plan_path": chain["control"] / "HOST_LAUNCH_PLAN.json",
        "input_binding_path": chain["control"] / "INPUT_BINDING.json",
        "mount_receipt_path": chain["control"] / "MOUNT_RECEIPT.json",
        **static_paths,
    }
    return binding.build_execution_readiness(
        **paths,
        source_commit=SOURCE_COMMIT,
        image_id=IMAGE_ID,
        image_inspect_path=chain["image_inspect"],
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


def _readonly_copy(source: Path, destination: Path, payload: bytes | None = None) -> Path:
    destination.write_bytes(source.read_bytes() if payload is None else payload)
    _readonly(destination)
    return destination


def test_readiness_builder_rejects_nonempty_output(ready_chain, tmp_path: Path):
    assert _build_readiness_with(ready_chain, tmp_path)["readiness_sha256"] == (
        ready_chain["readiness"]["readiness_sha256"]
    )
    (ready_chain["output"] / "unexpected").write_text("x", encoding="utf-8")
    with pytest.raises(ContractError, match="must be empty"):
        _build_readiness_with(ready_chain, tmp_path)


def test_readiness_builder_requires_passed_critical_test_nodes(ready_chain, tmp_path: Path):
    _write_junit(ready_chain["test_junit"], binding.REQUIRED_EXECUTION_TEST_NODEIDS[1:])
    report_path = tmp_path / "TEST_REPORT.json"
    report = json.loads(ready_chain["test_report"].read_text(encoding="utf-8"))
    report["junit_xml_sha256"] = binding.sha256_file(ready_chain["test_junit"])
    _readonly_copy(ready_chain["test_report"], report_path, canonical_json_bytes(report))
    with pytest.raises(ContractError, match="critical control-chain test nodes"):
        binding.build_execution_readiness(
            static_launch_plan_path=ready_chain["control"] / "HOST_LAUNCH_PLAN.json",
            input_binding_path=ready_chain["control"] / "INPUT_BINDING.json",
            mount_receipt_path=ready_chain["control"] / "MOUNT_RECEIPT.json",
            source_commit=SOURCE_COMMIT,
            image_id=IMAGE_ID,
            image_inspect_path=ready_chain["image_inspect"],
            source_manifest_path=ready_chain["source_manifest"],
            source_probe_receipt_path=ready_chain["source_probe_receipt"],
            orchestrator_path=REPOSITORY / "fedcore/experiments/v3_campaign_runner.py",
            state_machine_path=REPOSITORY / "fedcore/experiments/v3_campaign_state.py",
            execution_binding_path=REPOSITORY / "fedcore/experiments/v3_execution_binding.py",
            finalizer_path=REPOSITORY / "fedcore/experiments/v3_campaign_finalizer.py",
            posttrain_path=REPOSITORY / "fedcore/experiments/v3_posttrain.py",
            test_report_path=report_path,
            test_container_inspect_path=ready_chain["test_container_inspect"],
            test_junit_path=ready_chain["test_junit"],
            test_stdout_path=ready_chain["test_stdout"],
            cell_order_path=REPOSITORY / "paper/ijar-v34-pacs-fresh-seed-hsb-v3/PACS_MODEL_MATRIX.csv",
            host_control_dir=ready_chain["control"],
            host_output_dir=ready_chain["output"],
        )


def test_only_input_binding_keeps_a_compile_time_exact_file_anchor():
    for removed in (
        "STATIC_BINDING_ROOT_SHA256",
        "STATIC_LAUNCH_PLAN_FILE_SHA256",
        "STATIC_LAUNCH_PLAN_SHA256",
        "STATIC_MOUNT_RECEIPT_FILE_SHA256",
    ):
        assert not hasattr(binding, removed)
    assert binding.STATIC_INPUT_BINDING_FILE_SHA256 == binding.sha256_file(
        LOCAL_V1_STATIC / "INPUT_BINDING.json"
    )
    assert "Image" not in (LOCAL_V1_STATIC / "INPUT_BINDING.json").read_text(encoding="utf-8")
    assert "sha256:" not in (LOCAL_V1_STATIC / "INPUT_BINDING.json").read_text(encoding="utf-8")


def test_readiness_rejects_old_r2_input_binding(ready_chain, tmp_path):
    old = _readonly_copy(R2_STATIC / "INPUT_BINDING.json", tmp_path / "INPUT_BINDING.json")
    with pytest.raises(ContractError, match="input-binding"):
        _build_readiness_with(ready_chain, tmp_path, input_binding_path=old)


def test_readiness_rejects_old_r2_launch_plan(ready_chain, tmp_path):
    old = _readonly_copy(R2_STATIC / "HOST_LAUNCH_PLAN.json", tmp_path / "HOST_LAUNCH_PLAN.json")
    with pytest.raises(ContractError, match="not control schema 2"):
        _build_readiness_with(ready_chain, tmp_path, static_launch_plan_path=old)


def test_readiness_rejects_one_byte_input_binding_mutation(ready_chain, tmp_path):
    data = (ready_chain["control"] / "INPUT_BINDING.json").read_bytes()
    mutated = _readonly_copy(
        ready_chain["control"] / "INPUT_BINDING.json",
        tmp_path / "INPUT_BINDING.json",
        data[:-1] + b" " + data[-1:],
    )
    assert json.loads(mutated.read_text(encoding="utf-8")) == json.loads(data)
    with pytest.raises(ContractError, match="input-binding file hash mismatch"):
        _build_readiness_with(ready_chain, tmp_path, input_binding_path=mutated)


def test_static_launch_plan_rejects_internal_hash_mutation(chain):
    plan = chain["static_plan"]
    binding.validate_launch_template_plan(
        plan,
        input_binding=chain["input_binding"],
        expected_image_id=IMAGE_ID,
        expected_source_commit=SOURCE_COMMIT,
    )
    relabeled = deepcopy(plan)
    relabeled["launch_plan_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="self-hash mismatch"):
        _validate_forged(chain, plan=relabeled)
    edited = deepcopy(plan)
    edited["output_directory"] = LEGACY_V3_ROOT + "/runs/scientific/x"
    with pytest.raises(ContractError, match="self-hash mismatch"):
        _validate_forged(chain, plan=edited)


def test_readiness_rejects_mount_receipt_mismatch(ready_chain, tmp_path):
    old = _readonly_copy(R2_STATIC / "MOUNT_RECEIPT.json", tmp_path / "MOUNT_RECEIPT.json")
    with pytest.raises(ContractError, match="not control schema 2"):
        _build_readiness_with(ready_chain, tmp_path, mount_receipt_path=old)
    receipt = json.loads((ready_chain["control"] / "MOUNT_RECEIPT.json").read_text(encoding="utf-8"))
    receipt["input_binding_sha256"] = binding.sha256_file(R2_STATIC / "INPUT_BINDING.json")
    forged = _readonly_copy(
        ready_chain["control"] / "MOUNT_RECEIPT.json",
        tmp_path / "FORGED_MOUNT_RECEIPT.json",
        canonical_json_bytes(receipt),
    )
    with pytest.raises(ContractError, match="mount declaration input-binding mismatch"):
        _build_readiness_with(ready_chain, tmp_path, mount_receipt_path=forged)


def test_input_binding_rejects_non_local_staged_path(tmp_path: Path):
    local = json.loads((LOCAL_V1_STATIC / "INPUT_BINDING.json").read_text(encoding="utf-8"))
    assert local["isolated_input_directory"] == str(runner.ISOLATED_INPUT_DIRECTORY)
    moved_directory = deepcopy(local)
    moved_directory["isolated_input_directory"] = LEGACY_V3_ROOT + "/inputs"
    moved_record = deepcopy(local)
    moved_record["files"][0]["staged_path"] = LEGACY_V3_ROOT + "/inputs/PACS_mirror.zip"
    for forged, message in (
        (moved_directory, "isolated input directory drift"),
        (moved_record, "staged input source drift"),
    ):
        path = tmp_path / "forged-input-binding.json"
        path.write_bytes(canonical_json_bytes(forged))
        with pytest.raises(ContractError, match=message):
            runner.validate_input_binding(path)


def test_registered_host_paths_are_local_not_v3_or_v4():
    assert binding.EXPECTED_HOST_OUTPUT_DIR == runner.REGISTERED_OUTPUT_DIR
    assert host.EXPECTED_HOST_LOG_DIR == binding.EXPECTED_HOST_LOG_DIR
    for legacy_root in (LEGACY_V3_ROOT, LEGACY_V4_ROOT):
        for path in (
            binding.EXPECTED_HOST_OUTPUT_DIR,
            binding.EXPECTED_HOST_CONTROL_DIR,
            binding.EXPECTED_HOST_LOG_DIR,
            runner.REGISTERED_OUTPUT_DIR,
            runner.ISOLATED_INPUT_DIRECTORY,
        ):
            assert legacy_root not in str(path)


def test_v3_and_v4_output_paths_are_rejected(chain, tmp_path: Path):
    for legacy_root in (LEGACY_V3_ROOT, LEGACY_V4_ROOT):
        forged = deepcopy(chain["readiness"])
        forged["host_output_dir"] = legacy_root + "/runs/scientific/x"
        with pytest.raises(ContractError):
            _validate_readiness(chain, _reseal(forged, "readiness_sha256"))
    elsewhere = tmp_path / "Fedcore_HSB_v3_exec-output"
    elsewhere.mkdir()
    with pytest.raises(ContractError, match="output directory drift"):
        runner.validate_empty_output_dir(elsewhere)


def test_local_mount_probe_rejects_gpu_request(chain, monkeypatch):
    monkeypatch.setattr(mount_receipt_module, "validate_launch_plan_integrity", lambda plan: None)
    gpu = _probe_inspect(chain["static_plan"])
    gpu[0]["HostConfig"]["DeviceRequests"] = [
        {"Driver": "nvidia", "Count": 0, "DeviceIDs": ["0"], "Capabilities": [["gpu"]], "Options": {}}
    ]
    with pytest.raises(ContractError, match="must not request a GPU"):
        mount_receipt_module.validate_no_gpu_mount_probe(
            gpu,
            chain["static_plan"],
            expected_image_id=IMAGE_ID,
            probe_log=mount_receipt_module.MOUNT_PROBE_LOG,
            expected_output_dir=chain["output"],
            expected_control_dir=chain["control"],
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


def test_test_report_rejects_raw_evidence_hash_drift(ready_chain, tmp_path):
    report = json.loads(ready_chain["test_report"].read_text(encoding="utf-8"))
    report["test_container_inspect_sha256"] = "0" * 64
    drifted = _readonly_copy(
        ready_chain["test_report"], tmp_path / "TEST_REPORT.json", canonical_json_bytes(report)
    )
    with pytest.raises(ContractError, match="raw test-container inspect differs"):
        binding.build_execution_readiness(
            static_launch_plan_path=ready_chain["control"] / "HOST_LAUNCH_PLAN.json",
            input_binding_path=ready_chain["control"] / "INPUT_BINDING.json",
            mount_receipt_path=ready_chain["control"] / "MOUNT_RECEIPT.json",
            source_commit=SOURCE_COMMIT,
            image_id=IMAGE_ID,
            image_inspect_path=ready_chain["image_inspect"],
            source_manifest_path=ready_chain["source_manifest"],
            source_probe_receipt_path=ready_chain["source_probe_receipt"],
            orchestrator_path=REPOSITORY / "fedcore/experiments/v3_campaign_runner.py",
            state_machine_path=REPOSITORY / "fedcore/experiments/v3_campaign_state.py",
            execution_binding_path=REPOSITORY
            / "fedcore/experiments/v3_execution_binding.py",
            finalizer_path=REPOSITORY / "fedcore/experiments/v3_campaign_finalizer.py",
            posttrain_path=REPOSITORY / "fedcore/experiments/v3_posttrain.py",
            test_report_path=drifted,
            test_container_inspect_path=ready_chain["test_container_inspect"],
            test_junit_path=ready_chain["test_junit"],
            test_stdout_path=ready_chain["test_stdout"],
            cell_order_path=REPOSITORY
            / "paper/ijar-v34-pacs-fresh-seed-hsb-v3/PACS_MODEL_MATRIX.csv",
            host_control_dir=ready_chain["control"],
            host_output_dir=ready_chain["output"],
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
            launch_plan=chain["static_plan"],
        )


def test_authorization_rejects_expired_or_date_only_timestamp(chain):
    for issued in ("2020-01-01Z", "2020-01-01T00:00:00Z"):
        forged = deepcopy(chain["authorization"])
        forged["issued_at_utc"] = issued
        forged = _reseal(forged, "authorization_sha256")
        with pytest.raises(ContractError):
            binding.validate_run_authorization(forged, chain["readiness"], chain["static_plan"])


def test_mounted_bundle_rehashes_source_and_rejects_duplicate_json(chain):
    _write_control_bundle(chain)
    authority = binding.validate_mounted_authorization_bundle(
        chain["control"], expected_source_commit=SOURCE_COMMIT
    )
    assert authority.run_authorization_sha256 == chain["authorization"]["authorization_sha256"]
    _replace_bytes(chain["control"] / "TEST_REPORT.json", b'{"schema_version":1,"schema_version":1}\n')
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
            runtime_pathspecs=("pyproject.toml", "fedcore"),
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
            runtime_pathspecs=(),
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


def test_source_manifest_closes_the_local_control_issuer(chain):
    manifest = json.loads(chain["source_manifest"].read_text(encoding="utf-8"))
    paths = {row["path"] for row in manifest["runtime_files"]}
    assert "fedcore/experiments/v3_local_control_issuer.py" in binding.REQUIRED_RUNTIME_SOURCE_FILES
    assert "fedcore/experiments/v3_local_control_issuer.py" in paths
    for generated in (
        "HOST_LAUNCH_PLAN.json",
        "MOUNT_RECEIPT.json",
        "EXECUTION_READINESS.json",
        "RUN_AUTHORIZATION.json",
        "AUTHORIZED_CAMPAIGN_PLAN.json",
        "RUNTIME_INSPECT_RECEIPT.json",
    ):
        assert not any(path.endswith(generated) for path in paths)


def test_runtime_receipt_rejects_cpu_limit_drift(chain):
    forged = deepcopy(chain["inspect"])
    forged[0]["HostConfig"]["NanoCpus"] = 1_000_000_000
    with pytest.raises(ContractError, match="CPU limit drift"):
        _receipt_from(chain, forged)


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


def test_critical_test_nodes_are_registered_in_required_files():
    assert len(binding.REQUIRED_EXECUTION_TEST_NODEIDS) == len(
        set(binding.REQUIRED_EXECUTION_TEST_NODEIDS)
    )
    for node in binding.REQUIRED_EXECUTION_TEST_NODEIDS:
        path, name = node.split("::", 1)
        assert path in binding.REQUIRED_EXECUTION_TEST_FILES
        tree = ast.parse((REPOSITORY / path).read_text(encoding="utf-8"))
        assert name in {
            item.name for item in tree.body if isinstance(item, ast.FunctionDef)
        }, node
    passed = binding._junit_passed_nodeids(
        b'<testsuites><testsuite><testcase classname="tests.test_x" name="test_a"/>'
        b'<testcase classname="tests.test_x" name="test_b"><failure/></testcase>'
        b'<testcase classname="tests.test_x" name="test_c"><skipped/></testcase>'
        b"</testsuite></testsuites>"
    )
    assert passed == {"tests/test_x.py::test_a"}
