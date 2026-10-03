"""Local issuer of control-schema-2 readiness and run authorization.

This replaces the historical r3 ``build_binding.py`` for the local
author-executed rerun; it neither imports nor writes that historical bundle.

The issuer trusts no caller-supplied identity or hash:

* the image ID and its source-revision label are read from the actual
  ``IMAGE_INSPECT.json`` bytes;
* the source commit and tree are read from Git and must equal both the image
  label and ``SOURCE_MANIFEST.json``;
* every control artifact is a read-only, nonsymlink regular file read once,
  hashed and parsed from the same bytes, with duplicate JSON keys and
  oversized files refused;
* the control directory must contain exactly the files of the current phase;
* every artifact is re-validated, and its device, inode, size, mtime, and
  bytes re-checked immediately before each exclusive (``O_EXCL``) write.

``RUN_AUTHORIZATION.json`` and ``AUTHORIZED_CAMPAIGN_PLAN.json`` are both
built and validated in memory before either is written.  This module never
starts Docker, touches a GPU, or opens PACS.  Host root and the Docker daemon
are the trusted computing base.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
from typing import Any, Callable, Mapping, Sequence

from fedcore.experiments import v3_execution_binding as binding
from fedcore.experiments.v3_contract import PROTOCOL_ID, ContractError, canonical_json_bytes


READINESS_FILENAME = "EXECUTION_READINESS.json"
AUTHORIZATION_FILENAME = "RUN_AUTHORIZATION.json"
AUTHORIZED_PLAN_FILENAME = "AUTHORIZED_CAMPAIGN_PLAN.json"
AUTHORIZATION_PHASE_FILENAMES = (*binding.PRE_READINESS_FILENAMES, READINESS_FILENAME)
AUTHORIZATION_CONFIRMATION = "I_AUTHORIZE_ONE_FRESH_30_CELL_CAMPAIGN"
GitReader = Callable[[Path, Sequence[str]], str]


def git_reader(repository: Path, arguments: Sequence[str]) -> str:
    """Read-only Git query that does not rewrite the index."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"git query failed: {list(arguments)!r}") from exc
    return result.stdout


def _control_dir(control_dir: Path | None) -> Path:
    path = Path(binding.EXPECTED_HOST_CONTROL_DIR if control_dir is None else control_dir)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"control directory is absent: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ContractError("control directory must be a real nonsymlink directory")
    resolved = path.resolve()
    if resolved != binding.EXPECTED_HOST_CONTROL_DIR:
        raise ContractError("control directory differs from the registered control path")
    return resolved


def _require_phase(control: Path, expected: Sequence[str]) -> None:
    names = tuple(sorted(item.name for item in control.iterdir()))
    if names != tuple(sorted(expected)):
        raise ContractError(f"unexpected control phase files: {names!r}")


def _read_phase(control: Path, names: Sequence[str]) -> dict[str, binding.ControlArtifact]:
    return {
        name: binding.read_control_artifact(control / name, require_read_only=True)
        for name in names
    }


def _image_identity(image_inspect_path: Path) -> tuple[str, str, binding.ControlArtifact]:
    artifact = binding.read_control_artifact(
        image_inspect_path, require_object=False, require_read_only=True
    )
    value = artifact.value
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
        raise ContractError("image inspect must contain exactly one image")
    image_id = binding._require_image_id(value[0].get("Id"), "image inspect Id")
    config = value[0].get("Config")
    labels = config.get("Labels") if isinstance(config, Mapping) else None
    revision = labels.get("org.opencontainers.image.revision") if isinstance(labels, Mapping) else None
    revision = binding._require_hex(revision, 40, "image source-revision label")
    return image_id, revision, artifact


def _source_identity(
    repository: Path,
    *,
    manifest: Mapping[str, Any],
    image_revision: str,
    reader: GitReader,
) -> tuple[str, str]:
    commit = binding._require_hex(reader(repository, ["rev-parse", "HEAD"]).strip(), 40, "git HEAD")
    tree = binding._require_hex(
        reader(repository, ["rev-parse", "HEAD^{tree}"]).strip(), 40, "git tree"
    )
    if reader(repository, ["status", "--porcelain", "--untracked-files=all"]).strip():
        raise ContractError("source repository is not clean")
    if image_revision != commit:
        raise ContractError("image source-revision label differs from the Git commit")
    if manifest.get("source_commit") != commit or manifest.get("source_tree") != tree:
        raise ContractError("source manifest commit/tree differs from Git")
    return commit, tree


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise ContractError(f"control artifact already exists: {path.name}") from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ContractError(f"short write while issuing {path.name}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _artifact_hashes(files: Mapping[str, binding.ControlArtifact]) -> dict[str, str]:
    return {name: files[name].sha256 for name in binding.PRE_READINESS_FILENAMES}


def _validate_readiness_phase(files: Mapping[str, binding.ControlArtifact]) -> str:
    return binding.validate_execution_readiness(
        files[READINESS_FILENAME].value,
        static_launch_plan=files["HOST_LAUNCH_PLAN.json"].value,
        input_binding=files["INPUT_BINDING.json"].value,
        mount_receipt=files["MOUNT_RECEIPT.json"].value,
        artifact_sha256=_artifact_hashes(files),
    )


def issue_readiness(
    *,
    repository: Path,
    image_inspect_path: Path,
    test_container_inspect_path: Path,
    test_junit_path: Path,
    test_stdout_path: Path,
    control_dir: Path | None = None,
    output_dir: Path | None = None,
    reader: GitReader = git_reader,
) -> dict[str, Any]:
    """Validate the six pre-readiness artifacts and write EXECUTION_READINESS.json."""

    control = _control_dir(control_dir)
    _require_phase(control, binding.PRE_READINESS_FILENAMES)
    files = _read_phase(control, binding.PRE_READINESS_FILENAMES)
    image_id, image_revision, image_artifact = _image_identity(image_inspect_path)
    repository = Path(repository).resolve()
    source_commit, _ = _source_identity(
        repository,
        manifest=files["SOURCE_MANIFEST.json"].value,
        image_revision=image_revision,
        reader=reader,
    )
    readiness = binding.build_execution_readiness(
        static_launch_plan_path=control / "HOST_LAUNCH_PLAN.json",
        input_binding_path=control / "INPUT_BINDING.json",
        mount_receipt_path=control / "MOUNT_RECEIPT.json",
        source_commit=source_commit,
        image_id=image_id,
        image_inspect_path=Path(image_inspect_path),
        source_manifest_path=control / "SOURCE_MANIFEST.json",
        source_probe_receipt_path=control / "SOURCE_PROBE_RECEIPT.json",
        orchestrator_path=repository / "fedcore/experiments/v3_campaign_runner.py",
        state_machine_path=repository / "fedcore/experiments/v3_campaign_state.py",
        execution_binding_path=repository / "fedcore/experiments/v3_execution_binding.py",
        finalizer_path=repository / "fedcore/experiments/v3_campaign_finalizer.py",
        posttrain_path=repository / "fedcore/experiments/v3_posttrain.py",
        test_report_path=control / "TEST_REPORT.json",
        test_container_inspect_path=Path(test_container_inspect_path),
        test_junit_path=Path(test_junit_path),
        test_stdout_path=Path(test_stdout_path),
        cell_order_path=repository / "paper/ijar-v34-pacs-fresh-seed-hsb-v3/PACS_MODEL_MATRIX.csv",
        host_control_dir=control,
        host_output_dir=Path(
            binding.EXPECTED_HOST_OUTPUT_DIR if output_dir is None else output_dir
        ),
    )
    binding.validate_execution_readiness(
        readiness,
        static_launch_plan=files["HOST_LAUNCH_PLAN.json"].value,
        input_binding=files["INPUT_BINDING.json"].value,
        mount_receipt=files["MOUNT_RECEIPT.json"].value,
        artifact_sha256=_artifact_hashes(files),
        expected_source_commit=source_commit,
    )
    for artifact in (*files.values(), image_artifact):
        binding.recheck_control_artifact(artifact)
    _require_phase(control, binding.PRE_READINESS_FILENAMES)
    _write_exclusive(control / READINESS_FILENAME, canonical_json_bytes(readiness))
    return readiness


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ContractError("authorization timestamps must be timezone-aware UTC")
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def issue_authorization(
    *,
    confirmation: str,
    control_dir: Path | None = None,
    issued_at: datetime | None = None,
    valid_for: timedelta = timedelta(hours=24),
    nonce: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Issue RUN_AUTHORIZATION.json and AUTHORIZED_CAMPAIGN_PLAN.json for one fresh run."""

    if confirmation != AUTHORIZATION_CONFIRMATION:
        raise ContractError("explicit author confirmation is required to issue authorization")
    control = _control_dir(control_dir)
    _require_phase(control, AUTHORIZATION_PHASE_FILENAMES)
    files = _read_phase(control, AUTHORIZATION_PHASE_FILENAMES)
    _validate_readiness_phase(files)
    readiness = files[READINESS_FILENAME].value
    launch_plan = files["HOST_LAUNCH_PLAN.json"].value
    input_binding = files["INPUT_BINDING.json"].value
    authorization = build_run_authorization(
        readiness,
        launch_plan,
        issued_at=issued_at if issued_at is not None else datetime.now(timezone.utc),
        valid_for=valid_for,
        nonce=secrets.token_hex(32) if nonce is None else nonce,
    )
    authorized_plan = binding.build_authorized_campaign_plan(
        readiness, authorization, input_binding=input_binding, launch_plan=launch_plan
    )
    binding.validate_authorized_campaign_plan(
        authorized_plan, readiness, authorization, input_binding, launch_plan
    )
    # Immediately before issuance every artifact must still be the validated bytes.
    for artifact in files.values():
        binding.recheck_control_artifact(artifact)
    _require_phase(control, AUTHORIZATION_PHASE_FILENAMES)
    authorization_path = control / AUTHORIZATION_FILENAME
    _write_exclusive(authorization_path, canonical_json_bytes(authorization))
    try:
        _write_exclusive(control / AUTHORIZED_PLAN_FILENAME, canonical_json_bytes(authorized_plan))
    except BaseException:
        authorization_path.unlink(missing_ok=True)
        raise
    return authorization, authorized_plan


def build_run_authorization(
    readiness: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
    *,
    issued_at: datetime,
    valid_for: timedelta,
    nonce: str,
) -> dict[str, Any]:
    """Build and validate one authorization in memory; every echoed value comes from readiness."""

    readiness_hash = binding._validate_self_hash(
        readiness, "readiness_sha256", "execution readiness"
    )
    issued_text = _utc_text(issued_at)
    valid_text = _utc_text(issued_at + valid_for)
    nonce = binding._require_hex(nonce, 64, "nonce")
    authorization_id = binding.authorization_id_for(
        readiness_sha256=readiness_hash,
        authorization_nonce=nonce,
        issued_at_utc=issued_text,
        valid_until_utc=valid_text,
    )
    instantiated = binding.render_authorized_execution(
        launch_plan["execution_template"], authorization_id
    )
    body: dict[str, Any] = {
        "schema_version": 2,
        "control_schema_version": binding.CONTROL_SCHEMA_VERSION,
        "control_binding_mode": binding.CONTROL_BINDING_MODE,
        "protocol_id": PROTOCOL_ID,
        "status": binding.AUTHORIZATION_STATUS,
        "training_authorized": True,
        "scope": "ONE_FRESH_30_CELL_CAMPAIGN",
        "readiness_sha256": readiness_hash,
        "authorization_challenge_sha256": binding._authorization_challenge(readiness_hash),
        "authorization_method": binding.AUTHORIZATION_METHOD,
        "instantiated_execution_configuration_sha256": (
            binding.instantiated_execution_configuration_sha256(instantiated)
        ),
        "planned_cells": 30,
        "max_concurrent_scientific_processes": 1,
        "fresh_output_required": True,
        "resume_authorized": False,
        "authorization_nonce": nonce,
        "authorization_id": authorization_id,
        "authorized_by": binding.AUTHORIZED_BY,
        "issued_at_utc": issued_text,
        "valid_until_utc": valid_text,
    }
    for field in binding.AUTHORIZATION_ECHO_FIELDS:
        body[field] = readiness[field]
    authorization = binding._self_hashed(body, "authorization_sha256")
    binding.validate_run_authorization(authorization, readiness, launch_plan)
    return authorization


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    readiness = subparsers.add_parser("readiness")
    readiness.add_argument("--repository", type=Path, required=True)
    readiness.add_argument("--image-inspect", type=Path, required=True)
    readiness.add_argument("--test-container-inspect", type=Path, required=True)
    readiness.add_argument("--test-junit", type=Path, required=True)
    readiness.add_argument("--test-stdout", type=Path, required=True)
    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--confirm", required=True)
    authorize.add_argument("--valid-hours", type=int, default=24)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "readiness":
        readiness = issue_readiness(
            repository=args.repository,
            image_inspect_path=args.image_inspect,
            test_container_inspect_path=args.test_container_inspect,
            test_junit_path=args.test_junit,
            test_stdout_path=args.test_stdout,
        )
        result = {"status": readiness["status"], "readiness_sha256": readiness["readiness_sha256"]}
    else:
        if not 0 < args.valid_hours <= 168:
            raise ContractError("authorization validity must be between 1 and 168 hours")
        authorization, plan = issue_authorization(
            confirmation=args.confirm, valid_for=timedelta(hours=args.valid_hours)
        )
        result = {
            "status": authorization["status"],
            "authorization_id": authorization["authorization_id"],
            "authorized_plan_sha256": plan["authorized_plan_sha256"],
            "execution_started": False,
        }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_CONFIRMATION",
    "AUTHORIZATION_PHASE_FILENAMES",
    "build_run_authorization",
    "git_reader",
    "issue_authorization",
    "issue_readiness",
    "main",
]
