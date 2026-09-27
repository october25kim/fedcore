#!/usr/bin/env python3
"""Verify the compact IJAR claim-to-artifact release."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    release = json.loads((ROOT / "RELEASE.json").read_text(encoding="utf-8"))
    required = {
        "README.md",
        "RELEASE.json",
        "REPRODUCE.md",
        "SAMPLING_CONTRACT.md",
        "CLAIM_ARTIFACT.csv",
        "PROCEDURE_THEOREM_LEDGER.csv",
        "central_finding_source.csv",
        "tinyimagenet_n3_s050_summary.csv",
        "tinyimagenet_n3_s075_summary.csv",
    }
    missing = sorted(name for name in required if not (ROOT / name).is_file())
    if missing:
        raise SystemExit(f"FAIL missing required files: {missing}")

    expected: dict[str, str] = {}
    for raw in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, name = raw.split(maxsplit=1)
        expected[name.strip()] = digest
    checksum_errors = [
        name
        for name, digest in expected.items()
        if not (ROOT / name).is_file() or sha256(ROOT / name) != digest
    ]
    if checksum_errors:
        raise SystemExit(f"FAIL checksum mismatch: {checksum_errors}")

    with (ROOT / "CLAIM_ARTIFACT.csv").open(newline="", encoding="utf-8") as handle:
        claims = list(csv.DictReader(handle))
    with (ROOT / "PROCEDURE_THEOREM_LEDGER.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        procedures = list(csv.DictReader(handle))
    if not claims or not procedures:
        raise SystemExit("FAIL empty claim or procedure ledger")

    unresolved_sources: list[str] = []
    for claim in claims:
        source = claim.get("SourceArtifact", "")
        if source.startswith("NOT_APPLICABLE_"):
            continue
        for item in source.split(";"):
            candidate = (ROOT / item).resolve()
            if not candidate.is_file():
                unresolved_sources.append(f"{claim.get('ClaimID')}:{item}")
    if unresolved_sources:
        raise SystemExit(f"FAIL unresolved claim sources: {unresolved_sources}")

    print(
        "PASS IJAR release integrity: "
        f"{len(expected)} checksums, {len(claims)} claims, "
        f"{len(procedures)} procedure rows; status={release['status']}"
    )


if __name__ == "__main__":
    main()
