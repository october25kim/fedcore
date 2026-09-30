#!/usr/bin/env python3
"""Build the non-executing PACS v3 execution-binding artifacts.

This utility may create technical readiness, one user-authorized launch plan,
and a stopped-container preparation bundle.  It never starts a container and
never opens a PACS input.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Sequence
import xml.etree.ElementTree as ET

from fedcore.experiments.v3_contract import (
    PROTOCOL_ID,
    ContractError,
    canonical_json_sha256,
    sha256_file,
)
from fedcore.experiments.v3_execution_binding import (
    AUTHORIZATION_METHOD,
    AUTHORIZATION_STATUS,
    AUTHORIZED_BY,
    EXPECTED_HOST_CONTROL_DIR,
    EXPECTED_HOST_OUTPUT_DIR,
    REQUIRED_EXECUTION_GATE_NODEIDS,
    REQUIRED_EXECUTION_PYTEST_ARGV,
    REQUIRED_EXECUTION_TEST_FILES,
    REQUIRED_EXECUTION_TEST_GATES,
    TEST_REPORT_STATUS,
    build_authorized_campaign_plan,
    build_execution_readiness,
    build_run_authorization_template,
    build_scientific_source_manifest,
    build_source_probe_receipt,
    validate_authorized_campaign_plan,
    validate_execution_readiness,
    validate_execution_test_report,
    validate_execution_test_report_against_git_commit,
    validate_run_authorization,
    validate_test_container_inspect,
    validate_test_evidence_bundle_paths,
)
from fedcore.experiments.v3_orchestrator import write_json


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(__file__).resolve().parent
STATIC = ROOT / "paper/ijar-v34-pacs-fresh-seed-hsb-v3-static-input-binding-r2"
PUBLIC_BUNDLE_FILES = (
    "build_binding.py",
    "SOURCE_MANIFEST.json",
    "IMAGE_INSPECT.json",
    "SOURCE_PROBE_INSPECT.json",
    "SOURCE_PROBE_RECEIPT.json",
    "TEST_CONTAINER_INSPECT.json",
    "TEST_JUNIT.xml",
    "TEST_STDOUT.log",
    "TEST_REPORT.json",
    "EXECUTION_READINESS.json",
    "RUN_AUTHORIZATION_TEMPLATE.json",
)


def _git(repository: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"Git query failed: {args!r}") from exc
    return result.stdout.strip()


def _command(*args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(args),
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"host command failed: {args!r}") from exc


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object: {path}")
    return value


def _preserve_public_artifact(source: Path, name: str) -> Path:
    destination = BUNDLE / name
    source = Path(source).resolve()
    if destination.resolve() == source:
        return destination
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise ContractError(f"public bundle artifact already differs: {name}")
        return destination
    shutil.copyfile(source, destination)
    return destination


def build_public_bundle_manifest() -> dict[str, Any]:
    rows = []
    for name in PUBLIC_BUNDLE_FILES:
        path = BUNDLE / name
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ContractError(f"public bundle artifact is unsafe: {name}")
        rows.append(
            {
                "path": name,
                "sha256": sha256_file(path),
                "size_bytes": int(metadata.st_size),
            }
        )
    value = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_EXECUTION_READINESS_PUBLIC_BUNDLE_INVENTORY",
        "files": rows,
        "source_probe_inspect_preserved": True,
    }
    write_json(BUNDLE / "PUBLIC_BUNDLE_MANIFEST.json", value)
    return value


def _parse_utc(value: str) -> datetime:
    if not value.endswith("Z") or "T" not in value:
        raise ContractError("authorization timestamps must be full RFC3339 UTC")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo != timezone.utc:
        raise ContractError("authorization timestamp must use UTC")
    return parsed


def build_source_manifest(repository: Path, destination: Path) -> dict[str, Any]:
    commit = _git(repository, "rev-parse", "HEAD")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    value = build_scientific_source_manifest(
        repository,
        source_commit=commit,
        source_tree=tree,
    )
    write_json(destination, value)
    return value


def run_source_probe(
    *,
    image_id: str,
    source_manifest: Path,
    inspect_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Export and hash a stopped image workspace without PACS or GPU access."""

    for path in (inspect_output, receipt_output):
        if path.exists():
            raise ContractError(f"source-probe artifact already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    name = f"fedcore-v3-source-probe-{image_id.removeprefix('sha256:')[:12]}"
    created = _command(
        "docker",
        "create",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--entrypoint",
        "/bin/true",
        image_id,
    )
    container_id = created.stdout.strip()
    if len(container_id) != 64 or any(
        character not in "0123456789abcdef" for character in container_id
    ):
        raise ContractError("source-probe docker create returned an invalid container ID")
    try:
        inspected = _command("docker", "inspect", container_id)
        inspect_output.write_text(inspected.stdout, encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="fedcore-v3-source-probe-") as temporary:
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            _command("docker", "cp", f"{container_id}:/workspace/.", str(workspace))
            receipt = build_source_probe_receipt(
                source_manifest_path=source_manifest,
                extracted_workspace_root=workspace,
                image_id=image_id,
                container_inspect_path=inspect_output,
            )
            write_json(receipt_output, receipt)
    finally:
        _command("docker", "rm", container_id)
    return receipt


def build_test_report(
    *,
    repository: Path,
    image_id: str,
    source_manifest: Path,
    test_container_inspect: Path,
    junit_xml: Path,
    pytest_stdout: Path,
    output: Path,
) -> dict[str, Any]:
    """Derive the test report from raw JUnit evidence, never manual counts."""

    manifest = _read_object(source_manifest)
    source_commit = str(manifest.get("source_commit", ""))
    if _git(repository, "rev-parse", "HEAD") != source_commit:
        raise ContractError("test report repository HEAD differs from source manifest")
    try:
        test_inspect = json.loads(test_container_inspect.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("invalid test-container inspect JSON") from exc
    test_evidence_root = validate_test_evidence_bundle_paths(
        test_container_inspect_path=test_container_inspect,
        test_junit_path=junit_xml,
        test_stdout_path=pytest_stdout,
    )
    test_container_id = validate_test_container_inspect(
        test_inspect,
        expected_image_id=image_id,
        expected_repository_root=repository,
        expected_evidence_root=test_evidence_root,
    )
    try:
        root = ET.parse(junit_xml).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ContractError("invalid JUnit XML") from exc
    cases = list(root.iter("testcase"))
    if not cases:
        raise ContractError("JUnit XML contains no test cases")
    failures = sum(any(child.tag == "failure" for child in case) for case in cases)
    errors = sum(any(child.tag == "error" for child in case) for case in cases)
    skipped = sum(any(child.tag == "skipped" for child in case) for case in cases)
    collected = len(cases)
    passed = collected - failures - errors - skipped
    if failures != 0 or errors != 0 or skipped != 0 or passed != collected:
        raise ContractError("JUnit XML does not record a complete clean pass")

    observed_files: set[str] = set()
    observed_nodeids: set[str] = set()
    for case in cases:
        file_attribute = str(case.get("file", ""))
        classname = str(case.get("classname", ""))
        matches = [
            path
            for path in REQUIRED_EXECUTION_TEST_FILES
            if file_attribute == path
            or file_attribute.endswith("/" + path)
            or Path(path).stem in classname.split(".")
        ]
        if len(matches) != 1:
            raise ContractError(
                f"JUnit testcase cannot be assigned to one required file: {classname!r}"
            )
        path = matches[0]
        observed_files.add(path)
        name = str(case.get("name", "")).split("[", 1)[0]
        if not name:
            raise ContractError("JUnit testcase lacks a name")
        observed_nodeids.add(f"{path}::{name}")
    if observed_files != set(REQUIRED_EXECUTION_TEST_FILES):
        raise ContractError("JUnit XML does not cover the exact required test-file set")
    missing_gates = sorted(
        set(REQUIRED_EXECUTION_GATE_NODEIDS.values()) - observed_nodeids
    )
    if missing_gates:
        raise ContractError(f"JUnit XML lacks registered gate tests: {missing_gates!r}")

    test_hashes = {
        path: sha256_file(repository / path) for path in REQUIRED_EXECUTION_TEST_FILES
    }
    value = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": TEST_REPORT_STATUS,
        "image_id": image_id,
        "source_commit": source_commit,
        "source_manifest_sha256": sha256_file(source_manifest),
        "test_container_inspect_sha256": sha256_file(test_container_inspect),
        "test_container_id": test_container_id,
        "junit_xml_sha256": sha256_file(junit_xml),
        "pytest_stdout_sha256": sha256_file(pytest_stdout),
        "pytest_argv": list(REQUIRED_EXECUTION_PYTEST_ARGV),
        "required_test_files": list(REQUIRED_EXECUTION_TEST_FILES),
        "test_file_sha256": test_hashes,
        "pytest_counts": {
            "collected": collected,
            "passed": passed,
            "failed": failures + errors,
            "skipped": skipped,
        },
        "exit_code": 0,
        "environment": {"CUDA_VISIBLE_DEVICES": ""},
        "PACS_opened": False,
        "GPU_used": False,
        "test_gates": {gate: True for gate in REQUIRED_EXECUTION_TEST_GATES},
        "gate_test_nodeids": dict(REQUIRED_EXECUTION_GATE_NODEIDS),
    }
    validate_execution_test_report(
        value,
        expected_image_id=image_id,
        expected_source_commit=source_commit,
        expected_source_manifest_sha256=sha256_file(source_manifest),
    )
    validate_execution_test_report_against_git_commit(value, repository)
    write_json(output, value)
    for name, source in {
        "TEST_CONTAINER_INSPECT.json": test_container_inspect,
        "TEST_JUNIT.xml": junit_xml,
        "TEST_STDOUT.log": pytest_stdout,
        "TEST_REPORT.json": output,
    }.items():
        _preserve_public_artifact(source, name)
    return value


def stage_readiness(
    *,
    repository: Path,
    control_dir: Path,
    image_id: str,
    image_inspect: Path,
    source_manifest: Path,
    source_probe_inspect: Path,
    source_probe_receipt: Path,
    test_container_inspect: Path,
    test_junit: Path,
    test_stdout: Path,
    test_report: Path,
) -> dict[str, Any]:
    if control_dir != EXPECTED_HOST_CONTROL_DIR:
        raise ContractError("control directory differs from the sealed host path")
    control_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in control_dir.iterdir()]
    if unexpected:
        raise ContractError("execution-r3 control directory must begin empty")
    EXPECTED_HOST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if next(EXPECTED_HOST_OUTPUT_DIR.iterdir(), None) is not None:
        raise ContractError("scientific output directory must be empty")
    probe_receipt = _read_object(source_probe_receipt)
    if sha256_file(source_probe_inspect) != probe_receipt.get(
        "container_inspect_sha256"
    ):
        raise ContractError("raw source-probe inspect differs from its receipt")
    test_report_value = _read_object(test_report)
    if sha256_file(test_container_inspect) != test_report_value.get(
        "test_container_inspect_sha256"
    ):
        raise ContractError("raw test-container inspect differs from its report")
    if sha256_file(test_junit) != test_report_value.get("junit_xml_sha256"):
        raise ContractError("raw JUnit XML differs from its report")
    if sha256_file(test_stdout) != test_report_value.get("pytest_stdout_sha256"):
        raise ContractError("raw pytest transcript differs from its report")

    copies = {
        "HOST_LAUNCH_PLAN.json": STATIC / "HOST_LAUNCH_PLAN.json",
        "INPUT_BINDING.json": STATIC / "INPUT_BINDING.json",
        "MOUNT_RECEIPT.json": STATIC / "MOUNT_RECEIPT.json",
        "SOURCE_MANIFEST.json": source_manifest,
        "SOURCE_PROBE_RECEIPT.json": source_probe_receipt,
        "TEST_REPORT.json": test_report,
    }
    for name, source in copies.items():
        shutil.copyfile(source, control_dir / name)

    commit = _git(repository, "rev-parse", "HEAD")
    value = build_execution_readiness(
        static_launch_plan_path=control_dir / "HOST_LAUNCH_PLAN.json",
        input_binding_path=control_dir / "INPUT_BINDING.json",
        mount_receipt_path=control_dir / "MOUNT_RECEIPT.json",
        source_commit=commit,
        image_id=image_id,
        image_inspect_path=image_inspect,
        source_manifest_path=control_dir / "SOURCE_MANIFEST.json",
        source_probe_receipt_path=control_dir / "SOURCE_PROBE_RECEIPT.json",
        orchestrator_path=repository / "fedcore/experiments/v3_campaign_runner.py",
        state_machine_path=repository / "fedcore/experiments/v3_campaign_state.py",
        execution_binding_path=repository / "fedcore/experiments/v3_execution_binding.py",
        finalizer_path=repository / "fedcore/experiments/v3_campaign_finalizer.py",
        posttrain_path=repository / "fedcore/experiments/v3_posttrain.py",
        test_report_path=control_dir / "TEST_REPORT.json",
        test_container_inspect_path=test_container_inspect,
        test_junit_path=test_junit,
        test_stdout_path=test_stdout,
        cell_order_path=repository
        / "paper/ijar-v34-pacs-fresh-seed-hsb-v3/PACS_MODEL_MATRIX.csv",
        host_control_dir=control_dir,
        host_output_dir=EXPECTED_HOST_OUTPUT_DIR,
    )
    validate_execution_readiness(
        value,
        static_launch_plan=_read_object(control_dir / "HOST_LAUNCH_PLAN.json"),
        input_binding=_read_object(control_dir / "INPUT_BINDING.json"),
        mount_receipt=_read_object(control_dir / "MOUNT_RECEIPT.json"),
        expected_source_commit=commit,
    )
    write_json(control_dir / "EXECUTION_READINESS.json", value)
    for name, source in {
        "SOURCE_MANIFEST.json": source_manifest,
        "IMAGE_INSPECT.json": image_inspect,
        "SOURCE_PROBE_INSPECT.json": source_probe_inspect,
        "SOURCE_PROBE_RECEIPT.json": source_probe_receipt,
        "TEST_CONTAINER_INSPECT.json": test_container_inspect,
        "TEST_JUNIT.xml": test_junit,
        "TEST_STDOUT.log": test_stdout,
        "TEST_REPORT.json": test_report,
        "EXECUTION_READINESS.json": control_dir / "EXECUTION_READINESS.json",
    }.items():
        _preserve_public_artifact(source, name)
    write_json(
        BUNDLE / "RUN_AUTHORIZATION_TEMPLATE.json",
        build_run_authorization_template(value),
    )
    build_public_bundle_manifest()
    return value


def issue_authorization(
    *,
    control_dir: Path,
    nonce: str,
    issued_at_utc: str,
    valid_until_utc: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(nonce) != 64 or any(character not in "0123456789abcdef" for character in nonce):
        raise ContractError("authorization nonce must be lowercase hex64")
    issued = _parse_utc(issued_at_utc)
    valid_until = _parse_utc(valid_until_utc)
    if valid_until <= issued:
        raise ContractError("authorization expiry must follow issuance")
    readiness = _read_object(control_dir / "EXECUTION_READINESS.json")
    input_binding = _read_object(control_dir / "INPUT_BINDING.json")
    challenge = canonical_json_sha256(
        {
            "purpose": "AUTHORIZE_ONE_FRESH_30_CELL_FEDCORE_PACS_V3_CAMPAIGN",
            "readiness_sha256": readiness["readiness_sha256"],
        }
    )
    authorization_id = canonical_json_sha256(
        {
            "authorization_challenge_sha256": challenge,
            "authorization_nonce": nonce,
            "authorized_by": AUTHORIZED_BY,
            "issued_at_utc": issued_at_utc,
            "valid_until_utc": valid_until_utc,
        }
    )
    body = {
        "schema_version": 1,
        "protocol_id": readiness["protocol_id"],
        "status": AUTHORIZATION_STATUS,
        "training_authorized": True,
        "scope": "ONE_FRESH_30_CELL_CAMPAIGN",
        "readiness_sha256": readiness["readiness_sha256"],
        "authorization_challenge_sha256": challenge,
        "authorization_method": AUTHORIZATION_METHOD,
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
        "authorized_by": AUTHORIZED_BY,
        "issued_at_utc": issued_at_utc,
        "valid_until_utc": valid_until_utc,
    }
    authorization = dict(body)
    definition = (
        "canonical JSON of this object after authorization_sha256_definition "
        "is inserted and before authorization_sha256 is inserted"
    )
    authorization["authorization_sha256_definition"] = definition
    authorization["authorization_sha256"] = canonical_json_sha256(authorization)
    validate_run_authorization(authorization, readiness)
    plan = build_authorized_campaign_plan(
        readiness,
        authorization,
        input_binding=input_binding,
    )
    validate_authorized_campaign_plan(plan, readiness, authorization, input_binding)
    write_json(control_dir / "RUN_AUTHORIZATION.json", authorization)
    write_json(control_dir / "AUTHORIZED_CAMPAIGN_PLAN.json", plan)
    return authorization, plan


def seal_control(control_dir: Path) -> None:
    expected = {
        "HOST_LAUNCH_PLAN.json",
        "INPUT_BINDING.json",
        "MOUNT_RECEIPT.json",
        "SOURCE_MANIFEST.json",
        "SOURCE_PROBE_RECEIPT.json",
        "TEST_REPORT.json",
        "EXECUTION_READINESS.json",
        "RUN_AUTHORIZATION.json",
        "AUTHORIZED_CAMPAIGN_PLAN.json",
    }
    observed = {path.name for path in control_dir.iterdir()}
    if observed != expected:
        raise ContractError(
            f"pre-prepare control file set drift: missing={sorted(expected-observed)!r}, "
            f"extra={sorted(observed-expected)!r}"
        )
    for path in control_dir.iterdir():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ContractError(f"unsafe control artifact: {path}")
        path.chmod(0o444)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = subparsers.add_parser("source-manifest")
    source.add_argument("--repository", type=Path, default=ROOT)
    source.add_argument("--output", type=Path, default=BUNDLE / "SOURCE_MANIFEST.json")

    probe = subparsers.add_parser("source-probe")
    probe.add_argument("--image-id", required=True)
    probe.add_argument(
        "--source-manifest", type=Path, default=BUNDLE / "SOURCE_MANIFEST.json"
    )
    probe.add_argument(
        "--inspect-output", type=Path, default=BUNDLE / "SOURCE_PROBE_INSPECT.json"
    )
    probe.add_argument(
        "--receipt-output", type=Path, default=BUNDLE / "SOURCE_PROBE_RECEIPT.json"
    )

    test_report_parser = subparsers.add_parser("test-report")
    test_report_parser.add_argument("--repository", type=Path, default=ROOT)
    test_report_parser.add_argument("--image-id", required=True)
    test_report_parser.add_argument(
        "--source-manifest", type=Path, default=BUNDLE / "SOURCE_MANIFEST.json"
    )
    test_report_parser.add_argument(
        "--test-container-inspect",
        type=Path,
        default=BUNDLE / "TEST_CONTAINER_INSPECT.json",
    )
    test_report_parser.add_argument(
        "--junit-xml", type=Path, default=BUNDLE / "TEST_JUNIT.xml"
    )
    test_report_parser.add_argument(
        "--pytest-stdout", type=Path, default=BUNDLE / "TEST_STDOUT.log"
    )
    test_report_parser.add_argument(
        "--output", type=Path, default=BUNDLE / "TEST_REPORT.json"
    )

    readiness = subparsers.add_parser("readiness")
    readiness.add_argument("--repository", type=Path, default=ROOT)
    readiness.add_argument("--control-dir", type=Path, default=EXPECTED_HOST_CONTROL_DIR)
    readiness.add_argument("--image-id", required=True)
    readiness.add_argument("--image-inspect", type=Path, required=True)
    readiness.add_argument("--source-manifest", type=Path, required=True)
    readiness.add_argument("--source-probe-inspect", type=Path, required=True)
    readiness.add_argument("--source-probe-receipt", type=Path, required=True)
    readiness.add_argument("--test-container-inspect", type=Path, required=True)
    readiness.add_argument("--test-junit", type=Path, required=True)
    readiness.add_argument("--test-stdout", type=Path, required=True)
    readiness.add_argument("--test-report", type=Path, required=True)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--control-dir", type=Path, default=EXPECTED_HOST_CONTROL_DIR)
    authorize.add_argument("--nonce", required=True)
    authorize.add_argument("--issued-at-utc", required=True)
    authorize.add_argument("--valid-until-utc", required=True)

    seal = subparsers.add_parser("seal-control")
    seal.add_argument("--control-dir", type=Path, default=EXPECTED_HOST_CONTROL_DIR)

    args = parser.parse_args(argv)
    if args.command == "source-manifest":
        result = build_source_manifest(args.repository, args.output)
    elif args.command == "source-probe":
        result = run_source_probe(
            image_id=args.image_id,
            source_manifest=args.source_manifest,
            inspect_output=args.inspect_output,
            receipt_output=args.receipt_output,
        )
    elif args.command == "test-report":
        result = build_test_report(
            repository=args.repository,
            image_id=args.image_id,
            source_manifest=args.source_manifest,
            test_container_inspect=args.test_container_inspect,
            junit_xml=args.junit_xml,
            pytest_stdout=args.pytest_stdout,
            output=args.output,
        )
    elif args.command == "readiness":
        result = stage_readiness(
            repository=args.repository,
            control_dir=args.control_dir,
            image_id=args.image_id,
            image_inspect=args.image_inspect,
            source_manifest=args.source_manifest,
            source_probe_inspect=args.source_probe_inspect,
            source_probe_receipt=args.source_probe_receipt,
            test_container_inspect=args.test_container_inspect,
            test_junit=args.test_junit,
            test_stdout=args.test_stdout,
            test_report=args.test_report,
        )
    elif args.command == "authorize":
        authorization, plan = issue_authorization(
            control_dir=args.control_dir,
            nonce=args.nonce,
            issued_at_utc=args.issued_at_utc,
            valid_until_utc=args.valid_until_utc,
        )
        result = {
            "authorization_sha256": authorization["authorization_sha256"],
            "authorized_plan_sha256": plan["authorized_plan_sha256"],
        }
    else:
        seal_control(args.control_dir)
        result = {"status": "PASS_CONTROL_DIRECTORY_SEALED_HOLD_CONTAINER_START"}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
