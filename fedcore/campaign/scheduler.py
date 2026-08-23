"""A durable, restart-safe local experiment scheduler.

The scheduler has deliberately few policy assumptions: it executes one local
job at a time, persists every status transition, and validates an artifact
before declaring success.  A completed artifact is reusable only when its
sidecar manifest binds its SHA-256 digest to the exact immutable job spec.

No command is ever passed through a shell.  This module has no dependencies
outside the Python standard library and can also be used as a small CLI::

    python -m fedcore.campaign.scheduler --root results/queue run job.json
    python -m fedcore.campaign.scheduler --root results/queue run-pending
    python -m fedcore.campaign.scheduler --root results/queue status

``job.json`` is the JSON representation accepted by :meth:`JobSpec.from_dict`.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


STATE_SCHEMA_VERSION = 1
ARTIFACT_MANIFEST_SCHEMA_VERSION = 1
STATE_STATUSES = frozenset({"pending", "running", "succeeded", "failed"})
TERMINAL_STATUSES = frozenset({"succeeded", "failed"})


class SchedulerError(RuntimeError):
    """Base class for scheduler failures."""


class SpecMismatchError(SchedulerError):
    """The same semantic experiment ID was submitted with a different spec."""


class StateCorruptionError(SchedulerError):
    """A durable state file is malformed or its immutable spec was changed."""


class JobLockedError(SchedulerError):
    """Another scheduler owns the per-job lock."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _parse_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value).rstrip(b"\n")).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync after replace/unlink."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_json(
    path: str | os.PathLike[str], value: Any, mode: int = 0o600
) -> None:
    """Durably replace ``path`` with canonical JSON on the same filesystem."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        os.fchmod(fd, mode)
        payload = _json_bytes(value)
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a regular file."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _freeze_json(value: Any) -> Any:
    """Recursively freeze a JSON-compatible value against caller mutation."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_json(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        # This also rejects NaN/Infinity early.
        json.dumps(value, allow_nan=False)
        return value
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw_json(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(v) for v in value]
    return value


def _absolute_path(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve(strict=False))


@dataclass(frozen=True)
class JobSpec:
    """Immutable execution and provenance contract for one semantic experiment.

    ``retry_limit`` is the number of retries *after* the initial attempt.  Seeds
    are a namespace-to-integer mapping; the scheduler intentionally does not
    prescribe namespace names, so every campaign-specific seed can be recorded.

    ``artifact_schema`` supports these standard-library validators:

    - ``format``: ``auto``, ``json``, ``jsonl``, ``csv``, ``zip``, ``text``, or
      ``binary``;
    - ``required_keys`` / ``expected_values`` / ``field_types`` for JSON;
    - ``required_columns`` / ``min_rows`` for CSV;
    - ``required_members`` for zip/npz;
    - ``min_size_bytes`` and optional ``magic_hex`` for any format.

    Defaults are deterministic absolute paths adjacent to ``expected_artifact``.
    """

    experiment_id: str
    argv: Sequence[str]
    expected_artifact: str
    config_hash: str
    code_commit: str
    dataset_fold_hash: str
    seeds: Mapping[str, int]
    retry_limit: int = 0
    checkpoint_path: str = ""
    stdout_path: str = ""
    stderr_path: str = ""
    manifest_path: str = ""
    heartbeat_path: str = ""
    artifact_schema: Mapping[str, Any] = field(default_factory=dict)
    expected_artifact_sha256: str | None = None
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty string")
        if "\x00" in self.experiment_id:
            raise ValueError("experiment_id contains NUL")
        if isinstance(self.argv, (str, bytes)) or not isinstance(self.argv, Sequence):
            raise TypeError("argv must be a sequence of strings, never a shell string")
        argv = tuple(self.argv)
        if not argv or any(not isinstance(arg, str) or "\x00" in arg for arg in argv):
            raise ValueError("argv must contain at least one NUL-free string")
        if (
            not isinstance(self.retry_limit, int)
            or isinstance(self.retry_limit, bool)
            or self.retry_limit < 0
        ):
            raise ValueError("retry_limit must be a non-negative integer")
        if (
            not isinstance(self.expected_artifact, str)
            or not self.expected_artifact.strip()
        ):
            raise ValueError("expected_artifact must be a non-empty path")
        for name, value in (
            ("config_hash", self.config_hash),
            ("code_commit", self.code_commit),
            ("dataset_fold_hash", self.dataset_fold_hash),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        normalized_seeds: dict[str, int] = {}
        if not isinstance(self.seeds, Mapping):
            raise TypeError("seeds must be a namespace-to-integer mapping")
        for namespace, value in self.seeds.items():
            if not isinstance(namespace, str) or not namespace or "\x00" in namespace:
                raise ValueError("seed namespaces must be non-empty NUL-free strings")
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"seed {namespace!r} must be an integer")
            normalized_seeds[namespace] = value

        normalized_env: dict[str, str] = {}
        if not isinstance(self.env, Mapping):
            raise TypeError("env must be a string mapping")
        for key, value in self.env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("env keys and values must be strings")
            if "\x00" in key or "=" in key or "\x00" in value:
                raise ValueError("invalid environment key or value")
            normalized_env[key] = value

        timeout = self.timeout_seconds
        if timeout is not None and (
            not isinstance(timeout, (int, float)) or timeout <= 0
        ):
            raise ValueError("timeout_seconds must be positive or None")

        base = Path(self.cwd or os.getcwd()).expanduser().resolve(strict=False)
        artifact = _absolute_path(self.expected_artifact, base)
        defaults = {
            "checkpoint_path": artifact + ".checkpoint",
            "stdout_path": artifact + ".stdout.log",
            "stderr_path": artifact + ".stderr.log",
            "manifest_path": artifact + ".manifest.json",
            "heartbeat_path": artifact + ".heartbeat.json",
        }
        for field_name, default in defaults.items():
            raw = getattr(self, field_name) or default
            object.__setattr__(self, field_name, _absolute_path(raw, base))

        schema: Mapping[str, Any] = self.artifact_schema or {
            "format": "auto",
            "min_size_bytes": 1,
        }
        frozen_schema = _freeze_json(schema)
        # Canonical serialization here catches unsupported values before a job is queued.
        _json_bytes(_thaw_json(frozen_schema))

        expected_digest = self.expected_artifact_sha256
        if expected_digest is not None:
            expected_digest = expected_digest.lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
                raise ValueError(
                    "expected_artifact_sha256 must be a 64-character hex digest"
                )

        object.__setattr__(self, "argv", argv)
        object.__setattr__(
            self, "seeds", MappingProxyType(dict(sorted(normalized_seeds.items())))
        )
        object.__setattr__(
            self, "env", MappingProxyType(dict(sorted(normalized_env.items())))
        )
        object.__setattr__(self, "artifact_schema", frozen_schema)
        object.__setattr__(self, "expected_artifact_sha256", expected_digest)
        object.__setattr__(self, "cwd", str(base))
        object.__setattr__(self, "expected_artifact", artifact)
        object.__setattr__(
            self, "timeout_seconds", float(timeout) if timeout is not None else None
        )

        owned_paths = [
            self.expected_artifact,
            self.checkpoint_path,
            self.stdout_path,
            self.stderr_path,
            self.manifest_path,
            self.heartbeat_path,
        ]
        if len(set(owned_paths)) != len(owned_paths):
            raise ValueError(
                "artifact, checkpoint, stdout, stderr, manifest, and heartbeat paths must be distinct"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "argv": list(self.argv),
            "expected_artifact": self.expected_artifact,
            "config_hash": self.config_hash,
            "code_commit": self.code_commit,
            "dataset_fold_hash": self.dataset_fold_hash,
            "seeds": dict(self.seeds),
            "retry_limit": self.retry_limit,
            "checkpoint_path": self.checkpoint_path,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "manifest_path": self.manifest_path,
            "heartbeat_path": self.heartbeat_path,
            "artifact_schema": _thaw_json(self.artifact_schema),
            "expected_artifact_sha256": self.expected_artifact_sha256,
            "cwd": self.cwd,
            "env": dict(self.env),
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JobSpec":
        allowed = {
            "experiment_id",
            "argv",
            "expected_artifact",
            "config_hash",
            "code_commit",
            "dataset_fold_hash",
            "seeds",
            "retry_limit",
            "checkpoint_path",
            "stdout_path",
            "stderr_path",
            "manifest_path",
            "heartbeat_path",
            "artifact_schema",
            "expected_artifact_sha256",
            "cwd",
            "env",
            "timeout_seconds",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown JobSpec fields: {sorted(unknown)}")
        return cls(**dict(value))

    @property
    def spec_hash(self) -> str:
        return _canonical_hash(self.to_dict())

    @property
    def max_attempts(self) -> int:
        return self.retry_limit + 1


@dataclass(frozen=True)
class ArtifactValidation:
    valid: bool
    reason: str
    path: str
    sha256: str | None = None
    size_bytes: int | None = None
    format: str | None = None
    schema_checked: bool = False
    manifest_checked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "format": self.format,
            "schema_checked": self.schema_checked,
            "manifest_checked": self.manifest_checked,
        }


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    if suffix in {".zip", ".npz"}:
        return "zip"
    if suffix in {".txt", ".md", ".log"}:
        return "text"
    return "binary"


def _json_path_get(value: Any, dotted_path: str) -> tuple[bool, Any]:
    current = value
    for part in dotted_path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return False, None
    return True, current


def _matches_declared_type(value: Any, declared: str) -> bool:
    if declared == "null":
        return value is None
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if declared == "string":
        return isinstance(value, str)
    if declared == "array":
        return isinstance(value, list)
    if declared == "object":
        return isinstance(value, dict)
    return False


def _validate_schema(
    path: Path, schema: Mapping[str, Any], detected_format: str
) -> str | None:
    min_size = schema.get("min_size_bytes", 1)
    if not isinstance(min_size, int) or isinstance(min_size, bool) or min_size < 0:
        return "artifact_schema.min_size_bytes must be a non-negative integer"
    if path.stat().st_size < min_size:
        return f"artifact is smaller than min_size_bytes={min_size}"

    expected_suffix = schema.get("suffix")
    if expected_suffix is not None:
        suffixes = (
            [expected_suffix]
            if isinstance(expected_suffix, str)
            else list(expected_suffix)
        )
        if not any(str(path).endswith(str(suffix)) for suffix in suffixes):
            return f"artifact suffix does not match {suffixes!r}"

    magic_hex = schema.get("magic_hex")
    if magic_hex is not None:
        try:
            magic = bytes.fromhex(str(magic_hex))
        except ValueError:
            return "artifact_schema.magic_hex is not valid hex"
        with path.open("rb") as handle:
            if handle.read(len(magic)) != magic:
                return "artifact magic bytes do not match"

    if detected_format == "binary":
        return None
    if detected_format == "text":
        try:
            path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return f"artifact is not valid UTF-8 text: {exc}"
        return None
    if detected_format == "zip":
        try:
            with zipfile.ZipFile(path, "r") as archive:
                corrupt = archive.testzip()
                if corrupt is not None:
                    return f"zip member failed CRC validation: {corrupt}"
                names = set(archive.namelist())
        except (OSError, zipfile.BadZipFile) as exc:
            return f"artifact is not a valid zip/npz file: {exc}"
        missing = set(schema.get("required_members", ())) - names
        if missing:
            return f"zip artifact is missing members: {sorted(missing)}"
        return None
    if detected_format == "csv":
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames
                if not columns:
                    return "CSV artifact has no header"
                if len(columns) != len(set(columns)):
                    return "CSV artifact has duplicate columns"
                required = set(schema.get("required_columns", ()))
                missing = required - set(columns)
                if missing:
                    return f"CSV artifact is missing columns: {sorted(missing)}"
                row_count = sum(1 for _ in reader)
        except (OSError, UnicodeError, csv.Error) as exc:
            return f"CSV artifact could not be parsed: {exc}"
        min_rows = schema.get("min_rows", 0)
        if not isinstance(min_rows, int) or isinstance(min_rows, bool) or min_rows < 0:
            return "artifact_schema.min_rows must be a non-negative integer"
        if row_count < min_rows:
            return f"CSV artifact has {row_count} rows, expected at least {min_rows}"
        return None

    def validate_json_object(
        document: Any, row_label: str = "JSON artifact"
    ) -> str | None:
        required_keys = schema.get("required_keys", ())
        if required_keys and not isinstance(document, dict):
            return f"{row_label} must be an object"
        missing = [
            key for key in required_keys if not _json_path_get(document, str(key))[0]
        ]
        if missing:
            return f"{row_label} is missing keys: {sorted(missing)}"
        for dotted_path, expected in dict(schema.get("expected_values", {})).items():
            present, actual = _json_path_get(document, str(dotted_path))
            if not present or actual != _thaw_json(expected):
                return f"{row_label} value mismatch at {dotted_path!r}"
        for dotted_path, declared in dict(schema.get("field_types", {})).items():
            present, actual = _json_path_get(document, str(dotted_path))
            if not present or not _matches_declared_type(actual, str(declared)):
                return f"{row_label} type mismatch at {dotted_path!r}: expected {declared!r}"
        return None

    if detected_format == "json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return f"artifact is not valid JSON: {exc}"
        return validate_json_object(document)
    if detected_format == "jsonl":
        row_count = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    document = json.loads(line)
                    row_count += 1
                    problem = validate_json_object(document, f"JSONL row {line_number}")
                    if problem:
                        return problem
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return f"artifact is not valid JSONL: {exc}"
        min_rows = schema.get("min_rows", 1)
        if row_count < min_rows:
            return f"JSONL artifact has {row_count} rows, expected at least {min_rows}"
        return None
    return f"unsupported artifact format: {detected_format!r}"


def _manifest_matches_job(
    manifest: Any, job: JobSpec, result: ArtifactValidation
) -> str | None:
    if not isinstance(manifest, dict):
        return "artifact manifest is not a JSON object"
    if manifest.get("manifest_type") != "fedcore.campaign.artifact":
        return "artifact manifest_type is invalid"
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        return "artifact manifest schema_version is incompatible"
    exact = {
        "experiment_id": job.experiment_id,
        "spec_hash": job.spec_hash,
        "config_hash": job.config_hash,
        "code_commit": job.code_commit,
        "dataset_fold_hash": job.dataset_fold_hash,
        "seeds": dict(job.seeds),
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            return f"artifact manifest {key!r} does not match the job spec"
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        return "artifact manifest has no artifact record"
    if artifact.get("path") != job.expected_artifact:
        return "artifact manifest path does not match"
    if artifact.get("sha256") != result.sha256:
        return "artifact SHA-256 differs from its manifest"
    if artifact.get("size_bytes") != result.size_bytes:
        return "artifact size differs from its manifest"
    return None


def validate_artifact(
    job: JobSpec, *, require_manifest: bool = True
) -> ArtifactValidation:
    """Validate file stability, digest, declared schema, and optional manifest.

    ``require_manifest=False`` is used immediately after a successful subprocess
    exits; the scheduler then writes a provenance-binding manifest and validates
    again with ``require_manifest=True``.  Reuse always requires the manifest.
    """

    path = Path(job.expected_artifact)
    if not path.exists():
        return ArtifactValidation(False, "expected artifact does not exist", str(path))
    if not path.is_file():
        return ArtifactValidation(
            False, "expected artifact is not a regular file", str(path)
        )
    try:
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
    except OSError as exc:
        return ArtifactValidation(
            False, f"artifact could not be read: {exc}", str(path)
        )
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        return ArtifactValidation(
            False, "artifact changed while it was being hashed", str(path)
        )
    if job.expected_artifact_sha256 and digest != job.expected_artifact_sha256:
        return ArtifactValidation(
            False,
            "artifact SHA-256 does not match expected_artifact_sha256",
            str(path),
            digest,
            after.st_size,
        )

    schema = job.artifact_schema
    requested_format = str(schema.get("format", "auto")).lower()
    detected_format = (
        _infer_format(path) if requested_format == "auto" else requested_format
    )
    if detected_format not in {"binary", "text", "zip", "csv", "json", "jsonl"}:
        return ArtifactValidation(
            False,
            f"unsupported artifact format: {detected_format!r}",
            str(path),
            digest,
            after.st_size,
            detected_format,
        )
    problem = _validate_schema(path, schema, detected_format)
    if problem:
        return ArtifactValidation(
            False,
            problem,
            str(path),
            digest,
            after.st_size,
            detected_format,
            True,
            False,
        )
    result = ArtifactValidation(
        True,
        "artifact content and schema are valid",
        str(path),
        digest,
        after.st_size,
        detected_format,
        True,
        False,
    )
    if not require_manifest:
        return result

    manifest_path = Path(job.manifest_path)
    if not manifest_path.is_file():
        return replace(result, valid=False, reason="artifact manifest does not exist")
    try:
        manifest = _read_json(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return replace(
            result, valid=False, reason=f"artifact manifest is invalid JSON: {exc}"
        )
    problem = _manifest_matches_job(manifest, job, result)
    if problem:
        return replace(result, valid=False, reason=problem, manifest_checked=True)
    return replace(
        result,
        reason="artifact schema, SHA-256, and provenance manifest are valid",
        manifest_checked=True,
    )


class ExclusiveFileLock:
    """A small ownership-token lock created with ``O_CREAT | O_EXCL``."""

    def __init__(
        self, path: str | os.PathLike[str], metadata: Mapping[str, Any] | None = None
    ):
        self.path = Path(path)
        self.token = uuid.uuid4().hex
        self.metadata = dict(metadata or {})
        self.acquired = False

    def acquire(self) -> "ExclusiveFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        payload = {
            **self.metadata,
            # Ownership fields cannot be overridden by caller-supplied metadata.
            "token": self.token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at": _utc_now(),
        }
        try:
            fd = os.open(str(self.path), flags, 0o600)
        except FileExistsError as exc:
            raise JobLockedError(f"job lock already exists: {self.path}") from exc
        try:
            data = _json_bytes(payload)
            offset = 0
            while offset < len(data):
                offset += os.write(fd, data[offset:])
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            raise
        else:
            os.close(fd)
        _fsync_directory(self.path.parent)
        self.acquired = True
        return self

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            current = _read_json(self.path)
        except (OSError, ValueError):
            current = None
        # Never unlink a lock that was replaced by another owner.
        if isinstance(current, dict) and current.get("token") == self.token:
            try:
                self.path.unlink()
                _fsync_directory(self.path.parent)
            except FileNotFoundError:
                pass
        self.acquired = False

    def __enter__(self) -> "ExclusiveFileLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


@dataclass(frozen=True)
class RunOutcome:
    experiment_id: str
    status: str
    attempt: int
    returncode: int | None = None
    message: str = ""
    artifact_validation: ArtifactValidation | None = None


class PersistentLocalScheduler:
    """Persistent sequential local queue with restart and stale-job recovery."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        heartbeat_interval: float = 5.0,
        stale_after: float = 300.0,
        poll_interval: float = 0.1,
    ):
        if heartbeat_interval <= 0 or stale_after < 0 or poll_interval <= 0:
            raise ValueError(
                "heartbeat_interval/poll_interval must be positive; stale_after non-negative"
            )
        self.root = Path(root).expanduser().resolve(strict=False)
        self.state_dir = self.root / "states"
        self.lock_dir = self.root / "locks"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_interval = float(heartbeat_interval)
        self.stale_after = float(stale_after)
        self.poll_interval = float(poll_interval)
        self.hostname = socket.gethostname()

    @staticmethod
    def _job_key(experiment_id: str) -> str:
        readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", experiment_id).strip("-.")[:48]
        readable = readable or "job"
        digest = hashlib.sha256(experiment_id.encode("utf-8")).hexdigest()[:16]
        return f"{readable}-{digest}"

    def state_path(self, experiment_id: str) -> Path:
        return self.state_dir / f"{self._job_key(experiment_id)}.json"

    def lock_path(self, experiment_id: str) -> Path:
        return self.lock_dir / f"{self._job_key(experiment_id)}.lock"

    def _new_state(self, job: JobSpec) -> dict[str, Any]:
        now = _utc_now()
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "experiment_id": job.experiment_id,
            "spec_hash": job.spec_hash,
            "spec": job.to_dict(),
            "status": "pending",
            "attempt": 0,
            "retry_limit": job.retry_limit,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "ended_at": None,
            "heartbeat_at": None,
            "child_pid": None,
            "returncode": None,
            "last_error": None,
            "artifact_validation": None,
            "attempts": [],
            "history": [
                {
                    "from": None,
                    "to": "pending",
                    "at": now,
                    "reason": "job submitted",
                    "attempt": 0,
                }
            ],
        }

    @staticmethod
    def _owned_paths(job: JobSpec) -> dict[str, str]:
        return {
            name: getattr(job, name)
            for name in (
                "expected_artifact",
                "checkpoint_path",
                "stdout_path",
                "stderr_path",
                "manifest_path",
                "heartbeat_path",
            )
        }

    def _assert_no_path_collision(self, job: JobSpec) -> None:
        """Prevent two semantic jobs from racing on any scheduler-owned file."""

        proposed = self._owned_paths(job)
        proposed_reverse = {path: name for name, path in proposed.items()}
        for state_path in sorted(self.state_dir.glob("*.json")):
            state = self._validate_state(_read_json(state_path))
            if state["experiment_id"] == job.experiment_id:
                continue
            existing_job = JobSpec.from_dict(state["spec"])
            for existing_name, existing_path in self._owned_paths(existing_job).items():
                if existing_path in proposed_reverse:
                    proposed_name = proposed_reverse[existing_path]
                    raise SpecMismatchError(
                        f"job {job.experiment_id!r} {proposed_name} collides with "
                        f"job {existing_job.experiment_id!r} {existing_name}: {existing_path}"
                    )

    def _validate_state(
        self, state: Any, expected_id: str | None = None
    ) -> dict[str, Any]:
        if not isinstance(state, dict):
            raise StateCorruptionError("state is not a JSON object")
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise StateCorruptionError("unsupported state schema_version")
        if state.get("status") not in STATE_STATUSES:
            raise StateCorruptionError(f"invalid state status: {state.get('status')!r}")
        if expected_id is not None and state.get("experiment_id") != expected_id:
            raise StateCorruptionError(
                "state experiment_id does not match its requested ID"
            )
        spec = state.get("spec")
        if not isinstance(spec, dict):
            raise StateCorruptionError("state has no immutable job spec")
        try:
            reconstructed = JobSpec.from_dict(spec)
        except (TypeError, ValueError) as exc:
            raise StateCorruptionError(f"state job spec is invalid: {exc}") from exc
        if reconstructed.experiment_id != state.get("experiment_id"):
            raise StateCorruptionError("state and spec experiment IDs differ")
        if reconstructed.spec_hash != state.get("spec_hash"):
            raise StateCorruptionError("immutable job spec hash mismatch")
        if state.get("retry_limit") != reconstructed.retry_limit:
            raise StateCorruptionError("retry_limit differs from immutable spec")
        return state

    def load_state(self, experiment_id: str) -> dict[str, Any]:
        path = self.state_path(experiment_id)
        try:
            state = _read_json(path)
        except FileNotFoundError as exc:
            raise KeyError(f"unknown experiment_id: {experiment_id}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise StateCorruptionError(f"could not read state {path}: {exc}") from exc
        return self._validate_state(state, experiment_id)

    def _write_state(self, state: dict[str, Any]) -> None:
        self._validate_state(state, state.get("experiment_id"))
        atomic_write_json(self.state_path(state["experiment_id"]), state)

    def submit(self, job: JobSpec) -> dict[str, Any]:
        """Create a pending state, or verify exact identity of an existing job."""

        path = self.state_path(job.experiment_id)
        lock = ExclusiveFileLock(
            self.lock_path(job.experiment_id),
            {"experiment_id": job.experiment_id, "purpose": "submit"},
        )
        try:
            lock.acquire()
        except JobLockedError:
            # A concurrent runner may own it; reading its already-created state is safe.
            if not path.exists():
                raise
            state = self.load_state(job.experiment_id)
        else:
            try:
                if path.exists():
                    state = self.load_state(job.experiment_id)
                else:
                    self._assert_no_path_collision(job)
                    state = self._new_state(job)
                    self._write_state(state)
            finally:
                lock.release()
        if state["spec_hash"] != job.spec_hash:
            raise SpecMismatchError(
                f"experiment_id {job.experiment_id!r} is already bound to a different "
                "configuration, command, seed namespace, or provenance hash"
            )
        return state

    enqueue = submit

    def _transition(
        self,
        state: dict[str, Any],
        status: str,
        reason: str,
        **updates: Any,
    ) -> dict[str, Any]:
        if status not in STATE_STATUSES:
            raise ValueError(f"invalid transition target: {status}")
        old = state["status"]
        now = _utc_now()
        state.update(updates)
        state["status"] = status
        state["updated_at"] = now
        state.setdefault("history", []).append(
            {
                "from": old,
                "to": status,
                "at": now,
                "reason": reason,
                "attempt": state.get("attempt", 0),
            }
        )
        self._write_state(state)
        return state

    def _start_attempt(self, state: dict[str, Any]) -> dict[str, Any]:
        now = _utc_now()
        attempt = int(state["attempt"]) + 1
        state["attempt"] = attempt
        if state.get("started_at") is None:
            state["started_at"] = now
        state["ended_at"] = None
        state["heartbeat_at"] = now
        state["child_pid"] = None
        state["returncode"] = None
        state["last_error"] = None
        state.setdefault("attempts", []).append(
            {
                "attempt": attempt,
                "started_at": now,
                "heartbeat_at": now,
                "ended_at": None,
                "status": "running",
                "child_pid": None,
                "returncode": None,
                "error": None,
            }
        )
        return self._transition(state, "running", f"attempt {attempt} started")

    @staticmethod
    def _current_attempt(state: dict[str, Any]) -> dict[str, Any] | None:
        attempts = state.get("attempts") or []
        return attempts[-1] if attempts else None

    def _write_heartbeat(
        self, state: dict[str, Any], *, terminal: bool = False
    ) -> None:
        now = _utc_now()
        state["heartbeat_at"] = now
        state["updated_at"] = now
        attempt = self._current_attempt(state)
        if attempt is not None:
            attempt["heartbeat_at"] = now
        self._write_state(state)
        job = JobSpec.from_dict(state["spec"])
        atomic_write_json(
            job.heartbeat_path,
            {
                "schema_version": 1,
                "experiment_id": job.experiment_id,
                "spec_hash": job.spec_hash,
                "attempt": state["attempt"],
                "status": state["status"],
                "heartbeat_at": now,
                "terminal": terminal,
                "scheduler_pid": os.getpid(),
                "child_pid": state.get("child_pid"),
                "hostname": self.hostname,
            },
        )

    def _write_manifest(self, job: JobSpec, validation: ArtifactValidation) -> None:
        if (
            not validation.valid
            or not validation.sha256
            or validation.size_bytes is None
        ):
            raise ValueError("cannot manifest an invalid artifact")
        atomic_write_json(
            job.manifest_path,
            {
                "manifest_type": "fedcore.campaign.artifact",
                "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
                "created_at": _utc_now(),
                "experiment_id": job.experiment_id,
                "spec_hash": job.spec_hash,
                "config_hash": job.config_hash,
                "code_commit": job.code_commit,
                "dataset_fold_hash": job.dataset_fold_hash,
                "seeds": dict(job.seeds),
                "artifact": {
                    "path": job.expected_artifact,
                    "sha256": validation.sha256,
                    "size_bytes": validation.size_bytes,
                    "format": validation.format,
                    "schema_hash": _canonical_hash(_thaw_json(job.artifact_schema)),
                },
            },
        )

    def _finish_attempt_record(
        self,
        state: dict[str, Any],
        *,
        status: str,
        returncode: int | None,
        error: str | None,
    ) -> None:
        attempt = self._current_attempt(state)
        if attempt is None:
            return
        attempt.update(
            {
                "ended_at": _utc_now(),
                "status": status,
                "returncode": returncode,
                "error": error,
            }
        )

    def _technical_failure(
        self,
        state: dict[str, Any],
        reason: str,
        returncode: int | None,
        validation: ArtifactValidation | None = None,
    ) -> RunOutcome:
        job = JobSpec.from_dict(state["spec"])
        self._finish_attempt_record(
            state, status="failed", returncode=returncode, error=reason
        )
        state["returncode"] = returncode
        state["last_error"] = reason
        state["child_pid"] = None
        state["artifact_validation"] = validation.to_dict() if validation else None
        if int(state["attempt"]) < job.max_attempts:
            self._transition(
                state,
                "pending",
                f"technical failure; retry with identical spec: {reason}",
                ended_at=None,
            )
            self._write_heartbeat(state, terminal=False)
            return RunOutcome(
                job.experiment_id,
                "pending",
                state["attempt"],
                returncode,
                f"retry pending: {reason}",
                validation,
            )
        ended = _utc_now()
        self._transition(
            state, "failed", f"retry limit exhausted: {reason}", ended_at=ended
        )
        self._write_heartbeat(state, terminal=True)
        return RunOutcome(
            job.experiment_id,
            "failed",
            state["attempt"],
            returncode,
            reason,
            validation,
        )

    def _validated_success(
        self,
        state: dict[str, Any],
        validation: ArtifactValidation,
        reason: str,
    ) -> RunOutcome:
        job = JobSpec.from_dict(state["spec"])
        self._finish_attempt_record(state, status="succeeded", returncode=0, error=None)
        ended = _utc_now()
        self._transition(
            state,
            "succeeded",
            reason,
            ended_at=ended,
            child_pid=None,
            returncode=0,
            last_error=None,
            artifact_validation=validation.to_dict(),
        )
        self._write_heartbeat(state, terminal=True)
        return RunOutcome(
            job.experiment_id,
            "succeeded",
            state["attempt"],
            0,
            reason,
            validation,
        )

    @staticmethod
    def _process_is_alive(pid: Any) -> bool:
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError as exc:
            return exc.errno == errno.EPERM
        return True

    def _heartbeat_timestamp(self, state: dict[str, Any]) -> float | None:
        latest = _parse_timestamp(state.get("heartbeat_at"))
        try:
            job = JobSpec.from_dict(state["spec"])
            heartbeat = _read_json(job.heartbeat_path)
            if (
                isinstance(heartbeat, dict)
                and heartbeat.get("spec_hash") == state.get("spec_hash")
                and heartbeat.get("attempt") == state.get("attempt")
            ):
                external = _parse_timestamp(heartbeat.get("heartbeat_at"))
                if external is not None and (latest is None or external > latest):
                    latest = external
        except (OSError, ValueError, TypeError):
            pass
        return latest

    def _remove_lock(self, experiment_id: str) -> None:
        try:
            self.lock_path(experiment_id).unlink()
            _fsync_directory(self.lock_dir)
        except FileNotFoundError:
            pass

    def _break_orphan_lock_if_safe(
        self, experiment_id: str, *, now: float | None = None
    ) -> bool:
        """Remove a dead-owner lock; return false when a live owner may hold it."""

        path = self.lock_path(experiment_id)
        if not path.exists():
            return True
        current_time = time.time() if now is None else float(now)
        try:
            owner = _read_json(path)
        except (OSError, ValueError, UnicodeError):
            owner = None
        if isinstance(owner, dict):
            owner_host = owner.get("hostname")
            owner_pid = owner.get("pid")
            if owner_host == self.hostname and self._process_is_alive(owner_pid):
                return False
            created = _parse_timestamp(owner.get("created_at"))
            # A remote-host lock cannot be tested with os.kill.  Respect it until
            # the same stale timeout used for heartbeats has elapsed.
            if owner_host != self.hostname and created is not None:
                if current_time - created <= self.stale_after:
                    return False
        else:
            try:
                age = current_time - path.stat().st_mtime
            except FileNotFoundError:
                return True
            if age <= self.stale_after:
                return False
        self._remove_lock(experiment_id)
        return True

    def recover_stale_jobs(self, *, now: float | None = None) -> list[RunOutcome]:
        """Recover stale ``running`` jobs without duplicating a live child.

        A same-host child that still answers ``kill(pid, 0)`` remains running,
        even when its scheduler heartbeat is old.  Once it is gone, a valid
        manifested artifact is accepted; otherwise the exact frozen spec is
        retried subject to its original retry limit.
        """

        current_time = time.time() if now is None else float(now)
        outcomes: list[RunOutcome] = []
        for path in sorted(self.state_dir.glob("*.json")):
            try:
                state = self._validate_state(_read_json(path))
            except (OSError, ValueError, StateCorruptionError):
                continue
            if state["status"] != "running":
                continue
            heartbeat = self._heartbeat_timestamp(state)
            if heartbeat is not None and current_time - heartbeat <= self.stale_after:
                continue
            child_pid = state.get("child_pid")
            # This queue is local.  A live local process must never be launched twice.
            if self._process_is_alive(child_pid):
                outcomes.append(
                    RunOutcome(
                        state["experiment_id"],
                        "running",
                        state["attempt"],
                        None,
                        f"stale scheduler heartbeat but child PID {child_pid} is still alive",
                    )
                )
                continue

            if not self._break_orphan_lock_if_safe(
                state["experiment_id"], now=current_time
            ):
                outcomes.append(
                    RunOutcome(
                        state["experiment_id"],
                        "running",
                        state["attempt"],
                        None,
                        "stale heartbeat but a live scheduler still owns the O_EXCL lock",
                    )
                )
                continue
            job = JobSpec.from_dict(state["spec"])
            validation = validate_artifact(job, require_manifest=True)
            if validation.valid:
                outcomes.append(
                    self._validated_success(
                        state, validation, "stale run recovered from validated artifact"
                    )
                )
            else:
                outcomes.append(
                    self._technical_failure(
                        state,
                        f"stale running job had no valid artifact: {validation.reason}",
                        None,
                        validation,
                    )
                )
        return outcomes

    def _reconcile_completed_state(self, state: dict[str, Any]) -> RunOutcome | None:
        if state["status"] != "succeeded":
            return None
        job = JobSpec.from_dict(state["spec"])
        validation = validate_artifact(job, require_manifest=True)
        if validation.valid:
            return RunOutcome(
                job.experiment_id,
                "succeeded",
                state["attempt"],
                0,
                "validated completed artifact; execution skipped",
                validation,
            )
        # Corruption is a technical failure, not grounds to silently keep a success.
        if int(state["attempt"]) >= job.max_attempts:
            self._finish_attempt_record(
                state, status="failed", returncode=None, error=validation.reason
            )
            self._transition(
                state,
                "failed",
                f"completed artifact invalidated and retry limit exhausted: {validation.reason}",
                ended_at=_utc_now(),
                child_pid=None,
                returncode=None,
                artifact_validation=validation.to_dict(),
                last_error=validation.reason,
            )
            self._write_heartbeat(state, terminal=True)
            return RunOutcome(
                job.experiment_id,
                "failed",
                state["attempt"],
                None,
                f"completed artifact invalidated: {validation.reason}",
                validation,
            )
        self._transition(
            state,
            "pending",
            f"completed artifact invalidated: {validation.reason}",
            ended_at=None,
            artifact_validation=validation.to_dict(),
            last_error=validation.reason,
        )
        return None

    def run_job(self, job_or_id: JobSpec | str, *, dry_run: bool = False) -> RunOutcome:
        """Run at most one attempt of a job.

        A dry-run performs state/spec/artifact reconciliation but never acquires a
        run lock, creates logs, or starts a subprocess; pending work stays pending.
        """

        if isinstance(job_or_id, JobSpec):
            state = self.submit(job_or_id)
            job = job_or_id
        else:
            state = self.load_state(job_or_id)
            job = JobSpec.from_dict(state["spec"])

        completed = self._reconcile_completed_state(state)
        if completed is not None:
            return completed
        state = self.load_state(job.experiment_id)
        if state["status"] == "failed":
            return RunOutcome(
                job.experiment_id,
                "failed",
                state["attempt"],
                state.get("returncode"),
                "job has exhausted its immutable retry limit",
            )
        if state["status"] == "running":
            self.recover_stale_jobs()
            state = self.load_state(job.experiment_id)
            if state["status"] == "running":
                return RunOutcome(
                    job.experiment_id,
                    "running",
                    state["attempt"],
                    None,
                    "job is already running",
                )
            if state["status"] == "succeeded":
                return self._reconcile_completed_state(state) or RunOutcome(
                    job.experiment_id,
                    "pending",
                    state["attempt"],
                    message="artifact invalidated",
                )
            if state["status"] == "failed":
                return RunOutcome(
                    job.experiment_id,
                    "failed",
                    state["attempt"],
                    state.get("returncode"),
                    state.get("last_error") or "stale job failed",
                )

        # A crash can leave an artifact+manifest durable just before the final
        # state transition.  It is safe to reuse only when both bind exactly to
        # this pending immutable spec.
        existing = validate_artifact(job, require_manifest=True)
        if existing.valid:
            if state.get("started_at") is None:
                state["started_at"] = state.get("created_at") or _utc_now()
            return self._validated_success(
                state, existing, "validated pre-existing artifact; execution skipped"
            )

        if int(state["attempt"]) >= job.max_attempts:
            self._transition(
                state,
                "failed",
                "pending state has no remaining attempts",
                ended_at=_utc_now(),
                last_error="retry limit exhausted",
            )
            self._write_heartbeat(state, terminal=True)
            return RunOutcome(
                job.experiment_id,
                "failed",
                state["attempt"],
                state.get("returncode"),
                "job has no remaining attempts",
            )

        if dry_run:
            return RunOutcome(
                job.experiment_id,
                "dry-run",
                state["attempt"],
                message="would execute argv without a shell: " + repr(list(job.argv)),
            )

        # Recover the narrow crash window between O_EXCL acquisition and the
        # pending->running state write.  A live owner's lock is never broken.
        if not self._break_orphan_lock_if_safe(job.experiment_id):
            return RunOutcome(
                job.experiment_id,
                "locked",
                state["attempt"],
                None,
                "another live scheduler owns the O_EXCL job lock",
            )
        lock = ExclusiveFileLock(
            self.lock_path(job.experiment_id),
            {
                "experiment_id": job.experiment_id,
                "purpose": "run",
                "spec_hash": job.spec_hash,
            },
        )
        try:
            lock.acquire()
        except JobLockedError:
            return RunOutcome(
                job.experiment_id,
                "locked",
                state["attempt"],
                None,
                "another scheduler owns the O_EXCL job lock",
            )

        process: subprocess.Popen[bytes] | None = None
        stdout_handle = None
        stderr_handle = None
        try:
            # The state may have changed while this process was waiting for the lock.
            state = self.load_state(job.experiment_id)
            if state["spec_hash"] != job.spec_hash:
                raise SpecMismatchError("durable job spec changed before execution")
            if state["status"] != "pending":
                return RunOutcome(
                    job.experiment_id,
                    state["status"],
                    state["attempt"],
                    state.get("returncode"),
                    "job is no longer pending",
                )
            state = self._start_attempt(state)
            for output_path in (
                job.stdout_path,
                job.stderr_path,
                job.checkpoint_path,
                job.manifest_path,
            ):
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = open(job.stdout_path, "ab", buffering=0)
            stderr_handle = open(job.stderr_path, "ab", buffering=0)
            environment = os.environ.copy()
            environment.update(job.env)
            try:
                process = subprocess.Popen(
                    list(job.argv),
                    cwd=job.cwd,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    shell=False,
                )
            except OSError as exc:
                return self._technical_failure(
                    state, f"subprocess could not start: {exc}", None
                )
            state["child_pid"] = process.pid
            current_attempt = self._current_attempt(state)
            if current_attempt is not None:
                current_attempt["child_pid"] = process.pid
            self._write_heartbeat(state)

            started_monotonic = time.monotonic()
            last_heartbeat = started_monotonic
            timed_out = False
            while process.poll() is None:
                now_monotonic = time.monotonic()
                if (
                    job.timeout_seconds is not None
                    and now_monotonic - started_monotonic > job.timeout_seconds
                ):
                    timed_out = True
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                if now_monotonic - last_heartbeat >= self.heartbeat_interval:
                    self._write_heartbeat(state)
                    last_heartbeat = now_monotonic
                time.sleep(self.poll_interval)
            returncode = process.wait()
            for handle in (stdout_handle, stderr_handle):
                handle.flush()
                os.fsync(handle.fileno())
            if timed_out:
                return self._technical_failure(
                    state,
                    f"subprocess exceeded timeout_seconds={job.timeout_seconds}",
                    returncode,
                )
            if returncode != 0:
                return self._technical_failure(
                    state,
                    f"subprocess exited with return code {returncode}",
                    returncode,
                )

            validation = validate_artifact(job, require_manifest=False)
            if not validation.valid:
                return self._technical_failure(
                    state,
                    f"subprocess exited successfully but artifact is invalid: {validation.reason}",
                    returncode,
                    validation,
                )
            self._write_manifest(job, validation)
            manifested = validate_artifact(job, require_manifest=True)
            if not manifested.valid:
                return self._technical_failure(
                    state,
                    f"artifact manifest verification failed: {manifested.reason}",
                    returncode,
                    manifested,
                )
            return self._validated_success(
                state, manifested, "subprocess and artifact validation succeeded"
            )
        except BaseException as exc:
            # Preserve resumability for ordinary scheduler errors.  KeyboardInterrupt
            # and SystemExit are re-raised after the running state/child PID are durable.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                if process is not None and process.poll() is not None:
                    self._technical_failure(
                        state,
                        f"scheduler interrupted after child exit: {exc}",
                        process.returncode,
                    )
                raise
            # Never mark a still-live child retryable: that could launch the same
            # immutable experiment twice.  Ordinary scheduler exceptions first
            # stop the managed child; hard process death is handled later by
            # stale-running recovery.
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            try:
                latest = self.load_state(job.experiment_id)
                if latest["status"] == "running":
                    return self._technical_failure(
                        latest,
                        f"scheduler exception: {type(exc).__name__}: {exc}",
                        process.returncode if process is not None else None,
                    )
            except SchedulerError:
                pass
            raise
        finally:
            for handle in (stdout_handle, stderr_handle):
                if handle is not None and not handle.closed:
                    handle.close()
            lock.release()

    def list_states(self) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        for path in sorted(self.state_dir.glob("*.json")):
            states.append(self._validate_state(_read_json(path)))
        return states

    def run_pending(self, *, dry_run: bool = False) -> list[RunOutcome]:
        """Run one attempt of each currently pending job."""

        self.recover_stale_jobs()
        outcomes: list[RunOutcome] = []
        for state in self.list_states():
            if state["status"] == "pending":
                outcomes.append(self.run_job(state["experiment_id"], dry_run=dry_run))
        return outcomes

    def run_until_idle(self, *, dry_run: bool = False) -> list[RunOutcome]:
        """Run pending jobs, including declared retries, until none remain."""

        if dry_run:
            return self.run_pending(dry_run=True)
        outcomes: list[RunOutcome] = []
        while True:
            batch = self.run_pending()
            outcomes.extend(batch)
            if not any(outcome.status == "pending" for outcome in batch):
                break
        return outcomes


def _print_outcomes(outcomes: Iterable[RunOutcome]) -> None:
    for outcome in outcomes:
        print(
            json.dumps(
                {
                    "experiment_id": outcome.experiment_id,
                    "status": outcome.status,
                    "attempt": outcome.attempt,
                    "returncode": outcome.returncode,
                    "message": outcome.message,
                },
                sort_keys=True,
            )
        )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="durable queue directory")
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--stale-after", type=float, default=300.0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="submit and run one JobSpec JSON")
    run_parser.add_argument("spec")
    run_parser.add_argument("--dry-run", action="store_true")
    pending_parser = subparsers.add_parser(
        "run-pending", help="run persisted pending jobs"
    )
    pending_parser.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("recover", help="recover stale running jobs")
    subparsers.add_parser("status", help="print durable job states")
    args = parser.parse_args(argv)
    scheduler = PersistentLocalScheduler(
        args.root,
        heartbeat_interval=args.heartbeat_interval,
        stale_after=args.stale_after,
    )
    if args.command == "run":
        job = JobSpec.from_dict(_read_json(args.spec))
        _print_outcomes([scheduler.run_job(job, dry_run=args.dry_run)])
    elif args.command == "run-pending":
        _print_outcomes(scheduler.run_until_idle(dry_run=args.dry_run))
    elif args.command == "recover":
        _print_outcomes(scheduler.recover_stale_jobs())
    elif args.command == "status":
        for state in scheduler.list_states():
            print(json.dumps(state, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
