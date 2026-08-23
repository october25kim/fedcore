"""Immutable artifact manifests and semantic configuration hashes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import tempfile
from typing import Any, Mapping, Sequence


def canonical_json(value: Any) -> str:
    """Stable JSON encoding used for semantic IDs and manifest comparisons."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def semantic_experiment_id(prefix: str, training_config: Mapping[str, Any]) -> str:
    """Human-readable prefix plus a collision-resistant training-config digest."""
    clean_prefix = "".join(
        c if c.isalnum() or c in "-_" else "-" for c in prefix
    ).strip("-")
    if not clean_prefix:
        raise ValueError("experiment prefix must contain an alphanumeric character")
    return f"{clean_prefix}__{semantic_hash(dict(training_config))[:16]}"


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    sha256: str
    size_bytes: int
    role: str

    @classmethod
    def from_path(cls, path: str, role: str) -> "ArtifactRecord":
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return cls(
            path=os.path.abspath(path),
            sha256=file_sha256(path),
            size_bytes=os.path.getsize(path),
            role=role,
        )


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    experiment_id: str
    status: str
    training_config: Mapping[str, Any]
    posthoc_config: Mapping[str, Any]
    seeds: Mapping[str, int]
    config_hash: str
    code_commit: str
    dataset_hash: str
    fold_hash: str
    started_at: str
    ended_at: str
    checkpoint_path: str
    stdout_path: str
    stderr_path: str
    artifacts: Sequence[ArtifactRecord] = field(default_factory=tuple)
    environment: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported manifest schema_version")
        if self.status not in {"completed", "failed", "blocked", "infeasible"}:
            raise ValueError(f"invalid terminal status {self.status!r}")
        expected = semantic_hash(dict(self.training_config))
        if self.config_hash != expected:
            raise ValueError("config_hash does not match canonical training_config")
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not self.code_commit:
            raise ValueError(
                "code_commit is required (use UNAVAILABLE explicitly if absent)"
            )
        for record in self.artifacts:
            if file_sha256(record.path) != record.sha256:
                raise ValueError(f"artifact checksum mismatch: {record.path}")
            if os.path.getsize(record.path) != record.size_bytes:
                raise ValueError(f"artifact size mismatch: {record.path}")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["artifacts"] = [asdict(record) for record in self.artifacts]
        return value


def environment_record() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_immutable_manifest(path: str, manifest: RunManifest) -> str:
    """Atomically create a manifest; refuse a semantically different overwrite."""
    manifest.validate()
    payload = canonical_json(manifest.as_dict()) + "\n"
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            existing = handle.read()
        if existing != payload:
            raise FileExistsError(f"refusing to rewrite immutable manifest {path}")
        return path
    fd, tmp = tempfile.mkstemp(
        prefix=".manifest.", dir=os.path.dirname(path), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # O_EXCL semantics at the destination: a race must not replace another job.
        try:
            os.link(tmp, path)
        except FileExistsError:
            with open(path, encoding="utf-8") as handle:
                if handle.read() != payload:
                    raise
        return path
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
