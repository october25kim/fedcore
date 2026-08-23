"""Campaign provenance, inventory, and restart-safe local scheduling.

Imports are lazy so ``python -m fedcore.campaign.scheduler`` can execute without
pre-importing its own module (and without a runpy warning).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SCHEDULER_EXPORTS = {
    "ArtifactValidation",
    "ExclusiveFileLock",
    "JobLockedError",
    "JobSpec",
    "PersistentLocalScheduler",
    "RunOutcome",
    "SchedulerError",
    "SpecMismatchError",
    "StateCorruptionError",
    "atomic_write_json",
    "sha256_file",
    "validate_artifact",
}

_ARTIFACT_EXPORTS = {
    "ArtifactRecord",
    "RunManifest",
    "canonical_json",
    "environment_record",
    "file_sha256",
    "semantic_experiment_id",
    "semantic_hash",
    "utc_now",
    "write_immutable_manifest",
}

__all__ = sorted(_SCHEDULER_EXPORTS | _ARTIFACT_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _SCHEDULER_EXPORTS:
        return getattr(import_module(".scheduler", __name__), name)
    if name in _ARTIFACT_EXPORTS:
        return getattr(import_module(".artifacts", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
