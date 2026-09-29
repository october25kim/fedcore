#!/usr/bin/env python3
"""Verify the synthetic execution-binding bundle without third-party packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
EXPECTED_STATUS = (
    "PASS_SYNTHETIC_IMPLEMENTATION_BINDING_"
    "HOLD_SCIENTIFIC_RUNNER_AND_INPUT_BINDING"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    manifest_rows: list[tuple[str, str]] = []
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        observed = digest(ROOT / name)
        if observed != expected:
            raise SystemExit(f"INVALID_BUNDLE_HASH {name} {observed} != {expected}")
        if name != "SEAL.json":
            manifest_rows.append((name, f"{expected}  {name}\n"))
    seal = json.loads((ROOT / "SEAL.json").read_text(encoding="utf-8"))
    root_hash = hashlib.sha256(
        "".join(row for _, row in sorted(manifest_rows)).encode("utf-8")
    ).hexdigest()
    if root_hash != seal["bundle_root_sha256"]:
        raise SystemExit("INVALID_BUNDLE_ROOT")
    gate = json.loads((ROOT / "EXECUTION_GATE.json").read_text(encoding="utf-8"))
    if gate["status"] != EXPECTED_STATUS:
        raise SystemExit("INVALID_GATE_STATUS")
    for field in ("training_authorized", "pacs_training_started", "pacs_inference_started"):
        if gate[field] is not False:
            raise SystemExit(f"INVALID_GATE_FIELD {field}")
    source_manifest = ROOT / "SOURCE_MANIFEST.sha256"
    for line in source_manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        observed = digest(REPOSITORY_ROOT / name)
        if observed != expected:
            raise SystemExit(f"INVALID_SOURCE_HASH {name} {observed} != {expected}")
    integration = json.loads((ROOT / "SYNTHETIC_INTEGRATION.json").read_text())
    if integration["status"] != "PASS_SYNTHETIC_BINDING_TESTS":
        raise SystemExit("INVALID_SYNTHETIC_STATUS")
    if not integration["synthetic"] or integration["PACS_images_used"]:
        raise SystemExit("INVALID_SYNTHETIC_PROVENANCE")
    print("PASS_SYNTHETIC_BINDING_BUNDLE_HOLD_SCIENTIFIC_EXECUTION")
    print(f"bundle_root_sha256={root_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
