"""Immutable manifest tests (standalone and pytest compatible)."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcore.campaign.artifacts import (  # noqa: E402
    ArtifactRecord,
    RunManifest,
    semantic_hash,
    write_immutable_manifest,
)


def test_manifest_is_hash_checked_and_immutable():
    with tempfile.TemporaryDirectory() as td:
        artifact = os.path.join(td, "value.txt")
        with open(artifact, "w", encoding="utf-8") as handle:
            handle.write("fixed\n")
        config = {"dataset": "fixture", "seed": 3}
        manifest = RunManifest(
            schema_version=1,
            experiment_id="fixture__abc",
            status="completed",
            training_config=config,
            posthoc_config={},
            seeds={"train": 3},
            config_hash=semantic_hash(config),
            code_commit="UNAVAILABLE",
            dataset_hash="dataset-sha",
            fold_hash="fold-sha",
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="2026-01-01T00:01:00+00:00",
            checkpoint_path="",
            stdout_path="",
            stderr_path="",
            artifacts=(ArtifactRecord.from_path(artifact, "fixture"),),
        )
        path = os.path.join(td, "manifest.json")
        write_immutable_manifest(path, manifest)
        write_immutable_manifest(path, manifest)
        with open(artifact, "a", encoding="utf-8") as handle:
            handle.write("mutated\n")
        try:
            manifest.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("mutated artifact passed checksum validation")


def main():
    test_manifest_is_hash_checked_and_immutable()
    print("artifact manifest tests: PASS")


if __name__ == "__main__":
    main()
