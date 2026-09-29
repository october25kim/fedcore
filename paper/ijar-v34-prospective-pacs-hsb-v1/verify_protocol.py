#!/usr/bin/env python3
"""Verify the sealed scientific protocol without launching any experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise AssertionError(message)


def check_protocol() -> dict[str, object]:
    protocol = load_json("PROSPECTIVE_CONFIRMATION_PROTOCOL.json")
    pid = "FEDCORE-IJAR-V34-PACS-PROSPECTIVE-HSB-v1"
    if protocol["protocol_id"] != pid:
        fail("protocol_id mismatch")
    if protocol["state"] != "SEALED_SCIENTIFIC_PROTOCOL_NOT_EXECUTED":
        fail("unexpected protocol state")
    for key in ["gpu_training_started", "pacs_training_started", "pacs_inference_started", "submission_authorized"]:
        if protocol[key] is not False:
            fail(f"{key} must be false")
    if protocol["training"]["planned_models"] != 30:
        fail("planned_models must equal 30")
    primary = protocol["primary_contract"]
    expected = {
        "alpha": 0.2,
        "delta_r": 0.05,
        "delta_c": 0.05,
        "audit_budget_total": 512,
        "audit_draws_per_client": 128,
        "audit_replicate_id": 0,
        "models_in_primary_denominator": 30,
    }
    for key, value in expected.items():
        if primary[key] != value:
            fail(f"primary {key} mismatch")
    if protocol["selector_family"]["M"] != 12:
        fail("family size must equal 12")
    if protocol["dataset"]["J"] != 4:
        fail("PACS client count must equal four")
    if protocol["outcome_gates"]["absolute_utility_gate"]["minimum_mean_H_ECA"] != 0.05:
        fail("H ECA gate mismatch")
    if protocol["outcome_gates"]["client_division_gain_gate"]["minimum_mean_S_minus_B"] != 0.01:
        fail("S-B gate mismatch")
    for value in [
        protocol["dataset"]["exact_frame"]["archive_sha256"],
        protocol["dataset"]["exact_frame"]["image_manifest_sha256"],
        protocol["dataset"]["exact_frame"]["fold_manifest_sha256"],
        protocol["manuscript_binding"]["manuscript_docx_sha256"],
        protocol["manuscript_binding"]["rendered_pdf_sha256"],
    ]:
        if not HEX64.fullmatch(value):
            fail("invalid bound SHA-256")
    return protocol


def check_registered_manifests(protocol: dict[str, object]) -> dict[str, int]:
    with (ROOT / "PACS_MODEL_MATRIX.csv").open(newline="", encoding="utf-8") as handle:
        models = list(csv.DictReader(handle))
    if len(models) != 30 or len({row["model_id"] for row in models}) != 30:
        fail("model matrix must contain 30 unique models")
    if any(row["status"] != "NOT_RUN" for row in models):
        fail("every registered model must be NOT_RUN at seal")
    architecture_counts = {name: 0 for name in ["resnet18", "convnext_tiny"]}
    split_seed_pairs: set[tuple[int, int, str]] = set()
    for row in models:
        architecture_counts[row["architecture"]] += 1
        split_seed_pairs.add((int(row["split"]), int(row["nominal_seed"]), row["architecture"]))
    if architecture_counts != {"resnet18": 15, "convnext_tiny": 15}:
        fail("architecture denominator mismatch")
    if len(split_seed_pairs) != 30:
        fail("split-seed-architecture matrix incomplete")
    with (ROOT / "TRAINING_SEED_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        training_seeds = list(csv.DictReader(handle))
    if len(training_seeds) != 30 or len({row["model_init_seed_uint64"] for row in training_seeds}) != 30:
        fail("training seed manifest mismatch")
    with (ROOT / "AUDIT_SEED_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        audit_seeds = list(csv.DictReader(handle))
    if len(audit_seeds) != 2000:
        fail("audit seed manifest must contain 2,000 streams")
    primary_streams = [row for row in audit_seeds if row["primary"] == "true"]
    if len(primary_streams) != 20:
        fail("expected one primary stream for every split-client pair")
    if any(int(row["primary_prefix_draws"]) != 128 for row in audit_seeds):
        fail("primary prefix mismatch")
    return {"models": len(models), "training_seeds": len(training_seeds), "audit_streams": len(audit_seeds)}


def check_gate_and_schema() -> None:
    gate = load_json("EXECUTION_GATE.json")
    if gate["status"] != "HOLD_IMPLEMENTATION_BINDING" or gate["training_authorized"] is not False:
        fail("execution must remain on HOLD")
    if gate["gpu_training_started"] is not False or gate["submission_authorized"] is not False:
        fail("training/submission authorization must be false")
    schema = load_json("OUTCOME_SCHEMA.json")
    if schema["expected_primary_rows"] != 90:
        fail("primary output denominator mismatch")
    if schema["count_file"]["expected_rows"] != 1440:
        fail("primary count denominator mismatch")
    family = load_json("SELECTOR_FAMILY_CONTRACT.json")
    if len(family["ordering"]) != 12:
        fail("selector ordering mismatch")
    if [entry["candidate_index"] for entry in family["ordering"]] != list(range(12)):
        fail("selector indices are not contiguous")


def check_checksums() -> dict[str, int]:
    checksum_path = ROOT / "SHA256SUMS.txt"
    if not checksum_path.exists():
        fail("SHA256SUMS.txt is missing")
    checked = 0
    canonical_entries: list[tuple[str, str]] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            fail(f"missing sealed file: {relative}")
        observed = sha256(path)
        if observed != expected:
            fail(f"checksum mismatch: {relative}")
        if relative != "SEAL.json":
            canonical_entries.append((relative, expected))
        checked += 1
    seal = load_json("SEAL.json")
    if seal["preregistration_sha256"] != sha256(ROOT / "PROSPECTIVE_CONFIRMATION_PROTOCOL.json"):
        fail("protocol seal mismatch")
    canonical_lines = [
        f"{expected}  {relative}\n"
        for relative, expected in sorted(canonical_entries, key=lambda item: item[0])
    ]
    canonical_root = hashlib.sha256("".join(canonical_lines).encode("utf-8")).hexdigest()
    if seal["canonical_bundle_root_sha256"] != canonical_root:
        fail("canonical bundle root mismatch")
    return {"sealed_files": checked}


def main() -> int:
    protocol = check_protocol()
    counts = check_registered_manifests(protocol)
    check_gate_and_schema()
    checksum_counts = check_checksums()
    result = {
        "status": "PASS_SCIENTIFIC_PROTOCOL_SEALED_HOLD_EXECUTION",
        "protocol_id": protocol["protocol_id"],
        "gpu_training_started": False,
        "training_authorized": False,
        "submission_authorized": False,
        **counts,
        **checksum_counts,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}), file=sys.stderr)
        raise
