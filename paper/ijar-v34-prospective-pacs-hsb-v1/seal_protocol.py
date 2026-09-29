#!/usr/bin/env python3
"""Create the non-circular SHA-256 seal for the scientific protocol bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDED = {"SEAL.json", "SHA256SUMS.txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.iterdir()
        if path.is_file() and path.name not in EXCLUDED and not path.name.startswith(".")
    )


def checksum_line(path: Path) -> str:
    return f"{sha256(path)}  {path.name}\n"


def main() -> None:
    files = canonical_files()
    canonical_lines = [checksum_line(path) for path in files]
    canonical_root = hashlib.sha256("".join(canonical_lines).encode("utf-8")).hexdigest()
    protocol_hash = sha256(ROOT / "PROSPECTIVE_CONFIRMATION_PROTOCOL.json")
    seal = {
        "schema_version": 1,
        "protocol_id": "FEDCORE-IJAR-V34-PACS-PROSPECTIVE-HSB-v1",
        "sealed_at_utc": "2026-09-29T02:14:39Z",
        "state": "SEALED_SCIENTIFIC_PROTOCOL_NOT_EXECUTED",
        "preregistration_sha256": protocol_hash,
        "canonical_bundle_root_sha256": canonical_root,
        "canonical_file_count": len(files),
        "public_tag_planned": "pacs-prospective-hsb-v1",
        "gpu_training_started": False,
        "pacs_training_started": False,
        "pacs_inference_started": False,
        "training_authorized": False,
        "submission_authorized": False,
        "execution_gate": "HOLD_IMPLEMENTATION_BINDING",
        "note": "The public Git tag timestamps this content seal. A separate pre-outcome execution-binding seal is required before training."
    }
    (ROOT / "SEAL.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_lines = canonical_lines + [checksum_line(ROOT / "SEAL.json")]
    (ROOT / "SHA256SUMS.txt").write_text("".join(sorted(final_lines)), encoding="utf-8")
    print(json.dumps({"status": "SEALED", "protocol_sha256": protocol_hash, "bundle_root_sha256": canonical_root, "files": len(final_lines)}, sort_keys=True))


if __name__ == "__main__":
    main()
