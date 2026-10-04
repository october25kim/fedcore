#!/usr/bin/env python3
"""Verify the FedCORE IJAR v37 manuscript and evidence release."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PACS = ROOT / "artifacts" / "pacs"
DOCX_SHA256 = "3cdf07988c670016e00be135bd93ee8e5d8477811b6f40284e0e945c0f703c4d"
PDF_SHA256 = "ef585347079765f70990312326a7c9e4107ecd29363b99e65d0a8fe8cd903699"
RELEASE_TAG = "v0.7.0"
SOURCE_TAG = "pacs-local-v1-fa016c1"
SOURCE_COMMIT = "fa016c199a7c846bbe20b64d54ab34164adf7647"
SOURCE_TREE = "8adbb6fea6b5f7eef4ea8e7f78b484103bf08024"
IMAGE_ID = "sha256:62c35527e56f5a05c026598ffda355e2b2b80256d966e5d9202ba639a5bfde8a"
PROTOCOL_ID = "FEDCORE-IJAR-V34-PACS-FRESH-SEED-REPLICATION-HSB-v3"


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv(path: Path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_gzip_csv(path: Path) -> int:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def verify_manifest() -> None:
    manifest = ROOT / "SHA256SUMS.txt"
    require(manifest.is_file(), "SHA256SUMS.txt is missing")
    listed = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, rel = raw.split("  ", 1)
        require(rel not in listed, f"duplicate checksum entry: {rel}")
        listed[rel] = digest
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != manifest
    }
    require(set(listed) == actual, "checksum manifest file set differs from release file set")
    for rel, expected in listed.items():
        require(sha256(ROOT / rel) == expected, f"checksum mismatch: {rel}")


def verify_binding() -> None:
    binding = read_json(ROOT / "MANUSCRIPT_BINDING.json")
    release = read_json(ROOT / "RELEASE.json")
    docx = ROOT / binding["manuscript_filename"]
    pdf = ROOT / binding["rendered_pdf_filename"]
    require(binding["manuscript_version"] == "v37", "wrong manuscript version")
    require(binding["release_tag"] == RELEASE_TAG, "wrong release tag in binding")
    require(binding["pacs_source_tag"] == SOURCE_TAG, "wrong source tag in binding")
    require(binding["manuscript_included"] is True, "manuscript must be included")
    require(sha256(docx) == DOCX_SHA256 == binding["manuscript_sha256"], "DOCX binding mismatch")
    require(sha256(pdf) == PDF_SHA256 == binding["rendered_pdf_sha256"], "PDF binding mismatch")
    require(release["release_tag"] == RELEASE_TAG, "wrong tag in RELEASE.json")
    require(release["pacs_source_tag"] == SOURCE_TAG, "wrong PACS source tag")
    require(release["status"] == "VERSIONED_RELEASE", "release is not final")
    require(release["ledger_composition"] == {"figures": 7, "tables": 6, "abstract_claims": 3, "prose_claims": 8}, "wrong ledger composition")
    retained = (ROOT / "RETAINED_NON_GOVERNING.md").read_text(encoding="utf-8")
    require("The sole claim-to-evidence authority for v37 is `CLAIM_ARTIFACT.csv`" in retained, "non-governing boundary is missing")


def verify_ledger() -> None:
    rows = read_csv(ROOT / "CLAIM_ARTIFACT.csv")
    expected = {*(f"F{i:02d}" for i in range(1, 8)), *(f"T{i:02d}" for i in range(1, 7)), *(f"H{i:02d}" for i in range(1, 4)), *(f"C{i:02d}" for i in range(1, 9))}
    ids = [row["LedgerID"] for row in rows]
    require(len(rows) == 24 and len(set(ids)) == 24 and set(ids) == expected, "ledger ID composition mismatch")
    require(all(row["Status"] == "GOVERNING_V37" for row in rows), "non-v37 governing ledger row")
    for row in rows:
        obj = row["ManuscriptObject"]
        if obj:
            require(sha256(ROOT / obj) == row["ManuscriptObjectSHA256"], f"manuscript object mismatch: {row['LedgerID']}")
        paths = row["GoverningArtifact"].split(";")
        digests = row["ArtifactSHA256"].split(";")
        require(len(paths) == len(digests), f"artifact/hash arity mismatch: {row['LedgerID']}")
        for rel, digest in zip(paths, digests):
            require(sha256(ROOT / rel) == digest, f"governing artifact mismatch: {row['LedgerID']} {rel}")
    table1 = read_csv(ROOT / "artifacts" / "tables" / "closest_method_comparison.csv")
    require(len(table1) == 6, "Table 1 must contain six component rows")
    require(set(table1[0]) == {"established_component", "standard_object", "target_specific_gap", "fedcore_result", "validity_scope", "evidence_scope"}, "Table 1 schema mismatch")


def verify_legacy_governing_claims() -> None:
    validity = read_csv(ROOT / "artifacts" / "figures" / "validity_B1000_per_cell.csv")
    fedcore = [row for row in validity if row["method"] == "fedcore_noJ" and row["selector_proposal_feasible"] == "1"]
    require(len(fedcore) == 443, "target-matched validity denominator must be 443")
    coverages = [float(row["empirical_ucb_coverage"]) for row in fedcore]
    require(min(coverages) >= 0.95 and close(min(coverages), 0.953, 5e-4), "target-matched validity claim mismatch")
    headline = read_csv(ROOT / "artifacts" / "headline" / "primary_headline_alpha020.csv")
    expected_cert = {"H": 177, "S": 177, "B": 130}
    expected_eca = {"H": 0.08342402472238086, "S": 0.0816592568534411, "B": 0.06018240780185294}
    for procedure in ("H", "S", "B"):
        rows = [row for row in headline if row["dataset"] == "ALL" and row["procedure"] == procedure]
        require(len(rows) == 1, f"missing ALL headline row for {procedure}")
        total = int(rows[0]["N_cells"])
        certified = int(rows[0]["certified_cells"])
        eca = float(rows[0]["EffectiveCertCov"])
        require(total == 450 and certified == expected_cert[procedure], f"headline denominator mismatch for {procedure}")
        require(close(eca, expected_eca[procedure]), f"headline ECA mismatch for {procedure}")


def verify_pacs_denominator_and_contract():
    models = read_csv(PACS / "MODEL_STATUS.csv")
    failures = read_csv(PACS / "FAILURE_LEDGER.csv")
    primary = read_csv(PACS / "PRIMARY_512_REP000.csv")
    require(len(models) == 30 and not failures and len(primary) == 90, "PACS complete denominator mismatch")
    require({row["architecture"] for row in models} == {"resnet18", "convnext_tiny"}, "PACS architecture roster mismatch")
    require({int(row["split"]) for row in models} == set(range(5)), "PACS split roster mismatch")
    require({int(row["nominal_training_seed"]) for row in models} == {0, 1, 2}, "PACS seed roster mismatch")
    require(all(row["training_status"] == "terminal_success" and row["model_evaluable"] == "true" for row in models), "non-terminal or unevaluable PACS model")
    require(len({(row["model_id"], row["procedure"]) for row in primary}) == 90, "duplicate PACS procedure row")
    for row in primary:
        require(row["protocol_id"] == PROTOCOL_ID, "PACS protocol ID drift")
        require(close(float(row["alpha"]), 0.20) and close(float(row["delta_r"]), 0.05) and close(float(row["delta_c"]), 0.05), "PACS confidence contract drift")
        require(row["mixture_set"] == "Delta^3" and int(row["budget_total"]) == 512 and int(row["replicate_id"]) == 0, "PACS target or budget drift")
        require(int(row["proposal_feasible_members"]) == 12, "PACS family size drift")
    return models, primary


def verify_pacs_results(models, primary) -> None:
    procedures = ("H", "S", "B")
    selected_expected = {
        "pacs__convnext_tiny__split02__seed0",
        "pacs__convnext_tiny__split02__seed1",
        "pacs__convnext_tiny__split02__seed2",
    }
    expected_eca = {"H": 0.03866772986037437, "S": 0.03866772986037437, "B": 0.03556827642844303}
    expected_cond = {"H": 0.38667729860374367, "S": 0.38667729860374367, "B": 0.35568276428443024}
    for procedure in procedures:
        rows = [row for row in primary if row["procedure"] == procedure]
        certified = [row for row in rows if row["certified"] == "true"]
        require(len(rows) == 30 and len(certified) == 3, f"PACS certified count mismatch for {procedure}")
        require({row["model_id"] for row in certified} == selected_expected, f"PACS selected model set mismatch for {procedure}")
        require(close(sum(float(row["ECA"]) for row in rows) / 30, expected_eca[procedure]), f"PACS ECA mismatch for {procedure}")
        require(close(sum(float(row["coverage_lcb"]) for row in certified) / 3, expected_cond[procedure]), f"PACS conditional coverage mismatch for {procedure}")
    require(close(expected_eca["S"] - expected_eca["B"], 0.003099453431931335), "PACS S-minus-B mismatch")
    summary = read_json(PACS / "PACS_SUMMARY.json")
    require(summary["planned_models"] == summary["evaluable_models"] == summary["accounted_models"] == 30, "PACS summary denominator mismatch")
    require(summary["selected_model_set"] == sorted(selected_expected), "PACS summary model-set mismatch")


def verify_pacs_candidate_and_figure(primary) -> None:
    candidates = read_gzip_csv(PACS / "PRIMARY_CANDIDATE_DECISIONS.csv.gz")
    proposal = read_gzip_csv(PACS / "PROPOSAL_FAMILY_MANIFEST.csv.gz")
    counts = read_gzip_csv(PACS / "PRIMARY_COUNTS.csv.gz")
    require(len(candidates) == 1080 and len(proposal) == 360 and len(counts) == 1440, "PACS primary row counts mismatch")
    require(len({(row["model_id"], row["candidate_index"], row["procedure"]) for row in candidates}) == 1080, "duplicate PACS candidate row")
    expected = {
        "H": (360, 360, 321, 11, 11, 3),
        "S": (360, 360, 321, 11, 11, 3),
        "B": (360, 360, 321, 8, 8, 3),
    }
    derived = {}
    for procedure in expected:
        rows = [row for row in candidates if row["procedure"] == procedure]
        derived[procedure] = (
            len(rows),
            sum(row["proposal_feasible"] == "true" for row in rows),
            sum(row["coverage_positive"] == "true" for row in rows),
            sum(row["risk_pass"] == "true" for row in rows),
            sum(row["candidate_certified"] == "true" for row in rows),
            sum(row["selected"] == "true" for row in rows),
        )
        require(derived[procedure] == expected[procedure], f"candidate gate mismatch for {procedure}")
    displayed = {row["procedure"]: tuple(int(row[key]) for key in ("total_candidates", "proposal_feasible", "positive_coverage_lcb", "risk_gate_passed", "candidate_certified", "selected_and_certified")) for row in read_csv(PACS / "PACS_CANDIDATE_GATE_SUMMARY.csv")}
    require(displayed == expected, "Figure 7 candidate summary mismatch")
    matrix = read_csv(PACS / "PACS_MODEL_CERTIFICATION_MATRIX.csv")
    require(len(matrix) == 10 and all(int(row["total_models"]) == 3 for row in matrix), "Figure 7 model matrix denominator mismatch")
    for row in matrix:
        expected_value = 3 if row["architecture"] == "convnext_tiny" and int(row["split"]) == 2 else 0
        require(all(int(row[f"{procedure}_certified"]) == expected_value for procedure in ("H", "S", "B")), "Figure 7 model matrix value mismatch")


def verify_truth_replay_and_gates(primary) -> None:
    truth = read_csv(PACS / "POST_DECISION_TRUTH.csv")
    require(len(truth) == 90, "PACS truth denominator mismatch")
    reasons = Counter(row["truth_unavailable_reason"] for row in truth)
    require(reasons == Counter({"certificate_refusal": 81, "none": 9}), "PACS truth reason mismatch")
    certified_keys = {(row["model_id"], row["procedure"]) for row in primary if row["certified"] == "true"}
    truth_keys = {(row["model_id"], row["procedure"]) for row in truth if row["truth_unavailable_reason"] == "none"}
    require(certified_keys == truth_keys, "truth availability does not match certified decisions")
    replay = read_json(PACS / "INDEPENDENT_REPLAY.json")
    require(replay["status"] == "PASS_INDEPENDENT_REPLAY" and replay["candidate_rows_replayed"] == 1080 and replay["mismatch_count"] == 0, "independent replay failed")
    require(replay["production_module_imported"] is False, "independent replay imported production module")
    inputs = {
        "candidate_decisions": PACS / "PRIMARY_CANDIDATE_DECISIONS.csv.gz",
        "counts": PACS / "PRIMARY_COUNTS.csv.gz",
        "proposal": PACS / "PROPOSAL_FAMILY_MANIFEST.csv.gz",
    }
    for key, path in inputs.items():
        require(sha256(path) == replay["input_sha256"][key], f"independent replay input mismatch: {key}")
    validation = read_json(PACS / "FINAL_VALIDATION.json")
    require(validation["status"] == "VALID_NEGATIVE", "PACS final classification mismatch")
    require(all(value is False for value in validation["registered_metrics_and_gates"]["H_absolute_gates"].values()), "an absolute H gate unexpectedly passed")
    gain = validation["registered_metrics_and_gates"]["gain_gates"]
    require(gain == {"architecture_means_nonnegative": True, "mean_S_minus_B_at_least_0_01": False, "strict_positive_cell_each_architecture": False}, "PACS gain gates mismatch")


def verify_pacs_chain() -> None:
    completion = read_json(PACS / "EXECUTION_COMPLETION_SEAL.json")
    readiness = read_json(PACS / "provenance" / "EXECUTION_READINESS.json")
    authorization = read_json(PACS / "provenance" / "RUN_AUTHORIZATION.json")
    runtime = read_json(PACS / "provenance" / "RUNTIME_INSPECT_RECEIPT.json")
    terminal = read_json(PACS / "provenance" / "HOST_TERMINAL_RECEIPT.json")
    current = read_json(PACS / "CURRENT_STATE.json")
    final_state = read_json(PACS / current["state_path"])
    lease = read_json(PACS / "CAMPAIGN_LEASE.json")
    require(completion["status"] == "PASS_AUTHORIZED_CAMPAIGN_COMPLETION_BINDING" and completion["campaign_finalized"] is True, "completion seal mismatch")
    require(completion["source_commit"] == SOURCE_COMMIT and completion["image_id"] == IMAGE_ID and completion["planned_cells"] == 30, "completion source/image/denominator mismatch")
    require(completion["readiness_sha256"] == readiness["readiness_sha256"] == "e522ac49cbf081f985ac944d7c34ed831fbb3364bb8e76bd73b22ad6993ee234", "readiness chain mismatch")
    require(completion["run_authorization_sha256"] == authorization["authorization_sha256"] == lease["run_authorization_sha256"], "authorization chain mismatch")
    require(completion["runtime_receipt_sha256"] == runtime["runtime_receipt_sha256"] == lease["runtime_receipt_sha256"], "runtime chain mismatch")
    require(completion["authorization_id"] == authorization["authorization_id"] == runtime["authorization_id"] == lease["authorization_id"], "authorization ID mismatch")
    require(completion["current_state_file_sha256"] == sha256(PACS / "CURRENT_STATE.json"), "CURRENT_STATE file hash mismatch")
    require(current["state_sha256"] == final_state["state_sha256"] == completion["final_state_sha256"], "final state internal hash mismatch")
    require(final_state["campaign_phase"] == "FINALIZED" and final_state["sequence_number"] == 213, "final campaign state mismatch")
    require(terminal["status"] == "PROCESS_EXITED_ZERO" and terminal["returncode"] == 0 and terminal["timed_out"] is False, "terminal receipt does not show a clean process exit")
    require(terminal["container_confirmed_stopped"] is False, "release scope assumes terminal receipt does not prove stopped state")


def verify_source_tag() -> None:
    binding = read_json(PACS / "provenance" / "SOURCE_BINDING.json")
    manifest_path = PACS / "provenance" / "SOURCE_MANIFEST.json"
    manifest = read_json(manifest_path)
    require(binding["source_commit"] == manifest["source_commit"] == SOURCE_COMMIT, "source commit mismatch")
    require(binding["source_tree"] == manifest["source_tree"] == SOURCE_TREE, "source tree mismatch")
    require(binding["runtime_file_count"] == len(manifest["runtime_files"]) == 217, "runtime manifest count mismatch")
    require(binding["source_manifest_sha256"] == sha256(manifest_path), "source manifest file hash mismatch")
    require(binding["execution_image_id"] == IMAGE_ID, "source binding image mismatch")
    try:
        commit = subprocess.check_output(["git", "rev-parse", f"{SOURCE_TAG}^{{commit}}"], cwd=REPO, text=True).strip()
        tree = subprocess.check_output(["git", "rev-parse", f"{SOURCE_TAG}^{{tree}}"], cwd=REPO, text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise VerificationError(f"source tag cannot be resolved: {SOURCE_TAG}") from exc
    require(commit == SOURCE_COMMIT and tree == SOURCE_TREE, "source tag does not resolve to execution source")
    for item in manifest["runtime_files"]:
        try:
            blob = subprocess.check_output(["git", "show", f"{SOURCE_TAG}:{item['path']}"], cwd=REPO)
        except subprocess.CalledProcessError as exc:
            raise VerificationError(f"source-tag file missing: {item['path']}") from exc
        require(len(blob) == item["size_bytes"], f"source-tag size mismatch: {item['path']}")
        require(hashlib.sha256(blob).hexdigest() == item["sha256"], f"source-tag hash mismatch: {item['path']}")


def main() -> int:
    checks = [
        ("checksum manifest", verify_manifest),
        ("manuscript binding", verify_binding),
        ("claim-artifact ledger", verify_ledger),
        ("legacy governing claims", verify_legacy_governing_claims),
    ]
    for label, check in checks:
        check()
        print(f"PASS: {label}")
    models, primary = verify_pacs_denominator_and_contract()
    print("PASS: PACS complete denominator and fixed contract")
    verify_pacs_results(models, primary)
    print("PASS: PACS certification and ECA")
    verify_pacs_candidate_and_figure(primary)
    print("PASS: PACS candidate gates and Figure 7")
    verify_truth_replay_and_gates(primary)
    print("PASS: PACS truth, independent replay, and outcome gates")
    verify_pacs_chain()
    print("PASS: PACS readiness-to-completion provenance chain")
    verify_source_tag()
    print("PASS: exact PACS source tag and 217-file runtime manifest")
    print("PASS: FedCORE IJAR v37 release; 7 figures, 6 tables, 3 abstract claims, 8 prose claims")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
