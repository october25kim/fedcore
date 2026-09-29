#!/usr/bin/env python3
"""Verify the sealed scientific protocol without launching any experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PID = "FEDCORE-IJAR-V34-PACS-PROSPECTIVE-HSB-v2"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def seed_for(label: str, parts: list[object]) -> int:
    compact = json.dumps(parts, separators=(",", ":"), ensure_ascii=True)
    payload = f"{PID}|{label}|{compact}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def fail(message: str) -> None:
    raise AssertionError(message)


def assert_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        fail(f"{label}: observed={observed!r}, expected={expected!r}")


def assert_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-18):
        fail(f"{label}: observed={observed!r}, expected={expected!r}")


def check_json_protocol_ids() -> None:
    for path in sorted(ROOT.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        assert_equal(obj.get("protocol_id"), PID, f"{path.name} protocol_id")


def check_protocol() -> dict[str, object]:
    protocol = load_json("PROSPECTIVE_CONFIRMATION_PROTOCOL.json")
    assert_equal(protocol["protocol_id"], PID, "protocol_id")
    assert_equal(protocol["state"], "SEALED_SCIENTIFIC_PROTOCOL_NOT_EXECUTED", "protocol state")
    for key in ["gpu_training_started", "pacs_training_started", "pacs_inference_started", "submission_authorized"]:
        assert_equal(protocol[key], False, key)
    assert_equal(protocol["pre_outcome_amendment"]["scientific_outcome_seen"], False, "scientific outcome flag")

    dataset = protocol["dataset"]
    training = protocol["training"]
    assert_equal(dataset["J"], 4, "PACS client count")
    assert_equal(dataset["clients"], ["art_painting", "cartoon", "photo", "sketch"], "client order")
    assert_equal(training["architectures"], ["resnet18", "convnext_tiny"], "architecture order")
    assert_equal(training["class_splits"], [0, 1, 2, 3, 4], "training split order")
    assert_equal(training["nominal_seeds"], [0, 1, 2], "nominal seeds")
    assert_equal(training["planned_models"], 30, "planned models")
    assert_equal(training["federated_rounds"], 30, "federated rounds")
    assert_equal(training["local_epochs_per_round"], 1, "local epochs")
    assert_equal(training["clients_per_round"], 4, "clients per round")
    assert_equal(training["batch_size"], 32, "batch size")
    assert_equal(training["drop_last"], False, "drop_last")
    assert_equal(training["client_iteration_order"], dataset["clients"], "client iteration order")
    for required in ["EXIF transpose", "RandomResizedCrop(224", "RandomHorizontalFlip(p=0.5)", "ImageNet mean/std"]:
        if required not in training["train_transform"]:
            fail(f"train transform missing: {required}")
    for required in ["EXIF transpose", "Resize shorter side to 256", "CenterCrop(224)", "ImageNet mean/std"]:
        if required not in training["inference_transform"]:
            fail(f"inference transform missing: {required}")
    optimizer = training["optimizer"]
    expected_optimizer = {
        "name": "AdamW",
        "base_learning_rate": 0.0001,
        "weight_decay": 0.05,
        "betas": [0.9, 0.999],
        "eps": 1e-08,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
        "warmup_rounds": 2,
        "schedule": "rounds 0 and 1 use base_lr*(round+1)/2; rounds 2 through 29 use base_lr*0.5*(1+cos(pi*(round-2)/28))",
        "reset_each_client_round": True,
    }
    assert_equal(optimizer, expected_optimizer, "optimizer and LR contract")
    assert_equal(training["local_training_seed_parts"], "[architecture,split,nominal_seed,round,domain]", "local training seed parts")
    assert_equal(training["data_loader_workers"], 0, "data-loader workers")
    split_ids = [int(item["split"]) for item in dataset["class_splits"]]
    assert_equal(split_ids, training["class_splits"], "dataset/training split alignment")
    for item in dataset["class_splits"]:
        known = item["known"]
        unknown = item["unknown"]
        if len(known) != 4 or len(unknown) != 3 or set(known) & set(unknown):
            fail(f"invalid known/unknown partition for split {item['split']}")
        if set(known) | set(unknown) != set(dataset["classes"]):
            fail(f"incomplete class partition for split {item['split']}")

    family = protocol["selector_family"]
    assert_equal(family["M"], 12, "family size")
    assert_equal(family["minimum_proposal_accepted_count"], 1, "proposal accepted-count floor")
    assert_equal(family["proposal_pooling"], "pool all proposal domains by record with no domain reweighting", "proposal pooling")
    forbidden_family_keys = {"minimum_proposal_coverage", "minimum_proposal_coverage_fraction"}
    if forbidden_family_keys & set(family):
        fail("an unregistered proposal-coverage floor is present")

    primary = protocol["primary_contract"]
    expected_primary = {
        "alpha": 0.2,
        "delta_r": 0.05,
        "delta_c": 0.05,
        "audit_budget_total": 512,
        "audit_draws_per_client": 128,
        "audit_replicate_id": 0,
        "models_in_primary_denominator": 30,
        "primary_procedure_rows": 90,
    }
    for key, value in expected_primary.items():
        assert_equal(primary[key], value, f"primary {key}")
    if "not simultaneous across 30 models" not in primary["joint_confidence_scope"]:
        fail("joint confidence scope is not explicit")

    secondary = protocol["secondary_contract"]
    assert_equal(secondary["registered_audit_budgets_total"], [256, 512, 1024], "secondary budget grid")
    assert_equal(secondary["registered_audit_draws_per_client"], [64, 128, 256], "secondary prefix grid")
    assert_equal(secondary["secondary_audit_configurations_per_model"], 299, "secondary configurations/model")
    assert_equal(secondary["secondary_procedure_rows"], 26910, "secondary procedure rows")
    assert_equal(secondary["secondary_candidate_rows"], 322920, "secondary candidate rows")
    assert_equal(secondary["secondary_count_rows"], 430560, "secondary count rows")

    numerical = protocol["numerical_contract"]
    assert_close(float(numerical["H_S_tail"]), 0.05 / 12, "H/S tail")
    assert_close(float(numerical["B_tail"]), 0.05 / 48, "B tail")
    assert_equal(numerical["replay_tolerance"], {"atol": 1e-12, "rtol": 0.0}, "replay tolerance")
    if "first failure" not in numerical["holm"] or "adjusted p" not in numerical["holm"]:
        fail("Holm boundary and adjusted-p recursion are not frozen")

    ledger = protocol["denominator_ledger"]
    expected_ledger = {
        "planned_models": 30,
        "primary_procedure_rows": 90,
        "primary_candidate_decision_rows": 1080,
        "primary_count_rows": 1440,
        "registered_audit_configurations_per_model": 300,
        "secondary_audit_configurations_per_model": 299,
        "secondary_procedure_rows": 26910,
        "secondary_candidate_decision_rows": 322920,
        "secondary_count_rows": 430560,
    }
    for key, value in expected_ledger.items():
        assert_equal(ledger[key], value, f"denominator ledger {key}")

    gates = protocol["outcome_gates"]
    assert_equal(gates["precedence"], ["INVALID", "HOLD_INCOMPLETE", "PASS_FULL_PROSPECTIVE_CONFIRMATION", "PASS_CORE_ONLY", "VALID_NEGATIVE"], "gate precedence")
    absolute = gates["absolute_utility_gate"]
    assert_equal(absolute["minimum_H_certified_models"], 6, "H certification gate")
    assert_equal(absolute["H_certification_required_in_each_architecture"], True, "architecture gate")
    assert_equal(absolute["minimum_class_splits_with_H_certification"], 3, "class-split gate")
    assert_equal(absolute["minimum_mean_H_ECA"], 0.05, "H ECA gate")
    gain = gates["client_division_gain_gate"]
    assert_equal(gain["minimum_mean_S_minus_B"], 0.01, "S-B gate")
    assert_equal(gain["S_minus_B_nonnegative_in_each_architecture"], True, "S-B architecture gate")

    required_invalid = "E_H is no less than E_S and E_S is no less than E_B within replay tolerance"
    if required_invalid not in protocol["invalid_invariants"]["cell_level"]:
        fail("H/S/B cell-level monotonicity is not an INVALID invariant")

    resources = protocol["resource_contract"]
    assert_equal(resources["execution_host"], "workstation4", "execution host")
    assert_equal(resources["gpu_hour_cap_total"], 48, "GPU-hour cap")
    assert_equal(resources["capacity_stage_gpu_hours_included"], 2, "capacity-stage GPU hours")

    for value in [
        dataset["exact_frame"]["archive_sha256"],
        dataset["exact_frame"]["image_manifest_sha256"],
        dataset["exact_frame"]["fold_manifest_sha256"],
        protocol["manuscript_binding"]["manuscript_docx_sha256"],
        protocol["manuscript_binding"]["rendered_pdf_sha256"],
        training["weights"]["resnet18"]["sha256"],
        training["weights"]["convnext_tiny"]["sha256"],
    ]:
        if not HEX64.fullmatch(value):
            fail(f"invalid bound SHA-256: {value!r}")
    return protocol


def check_analysis_plan(protocol: dict[str, object]) -> None:
    plan = load_json("H_S_B_ANALYSIS_PLAN.json")
    primary = plan["primary"]
    assert_equal(primary, {
        "alpha": 0.2,
        "delta_r": 0.05,
        "delta_c": 0.05,
        "J": 4,
        "M": 12,
        "mixture_set": "Delta^3",
        "budget_total": 512,
        "draws_per_client": 128,
        "replicate_id": 0,
        "model_denominator": 30,
        "procedures": ["H", "S", "B"],
    }, "analysis primary contract")
    tails = plan["numerical_contract"]["tails"]
    assert_close(float(tails["H_coverage"]), 0.05 / 12, "analysis H coverage tail")
    assert_close(float(tails["S_risk"]), 0.05 / 12, "analysis S risk tail")
    assert_close(float(tails["S_coverage"]), 0.05 / 12, "analysis S coverage tail")
    assert_close(float(tails["B_risk"]), 0.05 / 48, "analysis B risk tail")
    assert_close(float(tails["B_coverage"]), 0.05 / 48, "analysis B coverage tail")
    numeric = plan["numerical_contract"]
    for token in ["inclusive", "first failure", "mapped back", "no tolerance"]:
        if token not in " ".join(str(value) for value in numeric.values()):
            fail(f"numerical contract missing token: {token}")
    confidence = plan["confidence_scope"]
    for key in ["not_simultaneous_across_30_models", "not_simultaneous_across_H_S_B", "not_a_confidence_interval_for_theta_H", "not_source_population_transport"]:
        assert_equal(confidence[key], True, f"confidence scope {key}")
    grid = plan["registered_audit_grid"]
    assert_equal(grid["budgets_total"], [256, 512, 1024], "registered budget grid")
    assert_equal(grid["configurations_per_model"], 300, "registered configurations/model")
    assert_equal(grid["secondary_configurations_per_model"], 299, "secondary configurations/model")
    assert_equal(grid["secondary_procedure_rows"], 30 * 299 * 3, "secondary procedure formula")
    assert_equal(grid["secondary_candidate_rows"], 30 * 299 * 12 * 3, "secondary candidate formula")
    assert_equal(grid["secondary_count_rows"], 30 * 299 * 12 * 4, "secondary count formula")
    if not any("E_H>=E_S>=E_B" in item for item in plan["invalid_invariants"]["cell_level"]):
        fail("analysis plan lacks cell-level H/S/B monotonicity")


def check_registered_manifests(protocol: dict[str, object]) -> dict[str, int]:
    models = read_csv("PACS_MODEL_MATRIX.csv")
    expected_models: list[dict[str, str]] = []
    split_map = {int(item["split"]): item for item in protocol["dataset"]["class_splits"]}
    for split_id in protocol["training"]["class_splits"]:
        split = split_map[int(split_id)]
        for nominal_seed in protocol["training"]["nominal_seeds"]:
            for architecture in protocol["training"]["architectures"]:
                expected_models.append({
                    "model_id": f"pacs__{architecture}__split{int(split_id):02d}__seed{nominal_seed}",
                    "architecture": architecture,
                    "split": str(split_id),
                    "nominal_seed": str(nominal_seed),
                    "model_init_seed_uint64": str(seed_for("model_init", [architecture, int(split_id), nominal_seed])),
                    "known_classes": ";".join(split["known"]),
                    "unknown_classes": ";".join(split["unknown"]),
                    "status": "NOT_RUN",
                })
    assert_equal(models, expected_models, "model matrix exact product/order")
    if len({row["model_id"] for row in models}) != 30:
        fail("model IDs are not unique")
    if len({row["model_init_seed_uint64"] for row in models}) != 30:
        fail("training seeds are not unique")

    training_seeds = read_csv("TRAINING_SEED_MANIFEST.csv")
    projected = [
        {key: row[key] for key in ["model_id", "architecture", "split", "nominal_seed", "model_init_seed_uint64"]}
        for row in expected_models
    ]
    assert_equal(training_seeds, projected, "training seed manifest exact projection")

    audit_seeds = read_csv("AUDIT_SEED_MANIFEST.csv")
    expected_audits: list[dict[str, str]] = []
    for split_id in protocol["training"]["class_splits"]:
        for client in protocol["dataset"]["clients"]:
            for replicate_id in range(100):
                expected_audits.append({
                    "split": str(split_id),
                    "client": client,
                    "replicate_id": str(replicate_id),
                    "pcg64_seed_uint64": str(seed_for("audit_draw", [int(split_id), client, replicate_id])),
                    "maximum_prefix_draws": "256",
                    "primary_prefix_draws": "128",
                    "secondary_lower_prefix_draws": "64",
                    "secondary_upper_prefix_draws": "256",
                    "primary": str(replicate_id == 0).lower(),
                })
    assert_equal(audit_seeds, expected_audits, "audit seed manifest exact product/order")
    if len({(row["split"], row["client"], row["replicate_id"]) for row in audit_seeds}) != 2000:
        fail("audit stream keys are not unique")
    if len([row for row in audit_seeds if row["primary"] == "true"]) != 20:
        fail("primary audit stream count is not 20")
    models_per_split = {split: sum(int(row["split"]) == split for row in models) for split in range(5)}
    assert_equal(models_per_split, {0: 6, 1: 6, 2: 6, 3: 6, 4: 6}, "CRN-sharing model count per split")
    return {"models": len(models), "training_seeds": len(training_seeds), "audit_streams": len(audit_seeds)}


def check_gate_schema_and_family() -> None:
    gate = load_json("EXECUTION_GATE.json")
    assert_equal(gate["status"], "HOLD_IMPLEMENTATION_BINDING", "execution gate")
    for key in ["training_authorized", "gpu_training_started", "submission_authorized"]:
        assert_equal(gate[key], False, f"execution gate {key}")

    schema = load_json("OUTCOME_SCHEMA.json")
    exact_rows = {
        "expected_primary_rows": 90,
        "proposal_family_file.expected_rows": 360,
        "count_file.expected_rows": 1440,
        "primary_candidate_decision_file.expected_rows": 1080,
        "post_decision_truth_file.expected_rows": 90,
        "secondary_procedure_file.expected_rows": 26910,
        "secondary_candidate_decision_file.expected_rows": 322920,
        "secondary_count_file.expected_rows": 430560,
    }
    observed_rows = {
        "expected_primary_rows": schema["expected_primary_rows"],
        "proposal_family_file.expected_rows": schema["proposal_family_file"]["expected_rows"],
        "count_file.expected_rows": schema["count_file"]["expected_rows"],
        "primary_candidate_decision_file.expected_rows": schema["primary_candidate_decision_file"]["expected_rows"],
        "post_decision_truth_file.expected_rows": schema["post_decision_truth_file"]["expected_rows"],
        "secondary_procedure_file.expected_rows": schema["secondary_procedure_file"]["expected_rows"],
        "secondary_candidate_decision_file.expected_rows": schema["secondary_candidate_decision_file"]["expected_rows"],
        "secondary_count_file.expected_rows": schema["secondary_count_file"]["expected_rows"],
    }
    assert_equal(observed_rows, exact_rows, "outcome row denominators")
    assert_equal(schema["primary_candidate_decision_file"]["primary_key"], ["model_id", "candidate_index", "procedure"], "primary candidate key")
    assert_equal(schema["secondary_procedure_file"]["primary_key"], ["model_id", "budget_total", "replicate_id", "procedure"], "secondary procedure key")
    assert_equal(schema["secondary_candidate_decision_file"]["primary_key"], ["model_id", "budget_total", "replicate_id", "candidate_index", "procedure"], "secondary candidate key")
    assert_equal(schema["secondary_count_file"]["primary_key"], ["model_id", "budget_total", "replicate_id", "candidate_index", "client"], "secondary count key")
    refusal = schema["required_columns"]["refusal_reason"]
    if "accepted_count" in refusal or "width" in refusal:
        fail("unregistered accepted-count or width refusal gate remains")
    required_ledgers = {
        "PRIMARY_CANDIDATE_DECISIONS.csv.gz",
        "POST_DECISION_TRUTH.csv",
        "SECONDARY_COUNTS.csv.gz",
        "SECONDARY_CANDIDATE_DECISIONS.csv.gz",
        "SECONDARY_AUDIT_REPLICATES.csv.gz",
    }
    if not required_ledgers.issubset(schema["mandatory_ledgers"]):
        fail("mandatory primary/secondary ledgers are incomplete")
    if schema["numeric_invariants"]["violation_status"] != "INVALID":
        fail("numeric invariant breach is not INVALID")

    family = load_json("SELECTOR_FAMILY_CONTRACT.json")
    assert_equal(family["family_size"], 12, "selector family size")
    assert_equal([entry["candidate_index"] for entry in family["ordering"]], list(range(12)), "selector candidate order")
    expected_order = [
        {"candidate_index": index, "score": score, "gamma": gamma}
        for index, (score, gamma) in enumerate(
            (score, gamma)
            for score in ["msp", "energy_lse", "margin"]
            for gamma in [0.3, 0.5, 0.7, 1.0]
        )
    ]
    assert_equal(family["ordering"], expected_order, "selector product/order")
    if "No additional proposal-coverage floor" not in family["member_rule"]:
        fail("selector contract does not explicitly exclude a proposal-coverage floor")
    assert_equal(family["proposal_pooling"], "pool all proposal-domain records without domain reweighting", "selector proposal pooling")
    assert_equal(family["acceptance_rule"], "accept iff oriented score >= threshold", "selector acceptance rule")
    if "exact integer cross multiplication" not in family["comparison_arithmetic"]:
        fail("proposal risk comparison arithmetic is not exact")


def check_checksums() -> dict[str, int]:
    checksum_path = ROOT / "SHA256SUMS.txt"
    if not checksum_path.exists():
        fail("SHA256SUMS.txt is missing")
    listed_files: set[str] = set()
    canonical_entries: list[tuple[str, str]] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        if not HEX64.fullmatch(expected):
            fail(f"invalid manifest hash: {relative}")
        if relative in listed_files:
            fail(f"duplicate checksum entry: {relative}")
        listed_files.add(relative)
        path = ROOT / relative
        if not path.is_file():
            fail(f"missing sealed file: {relative}")
        if sha256(path) != expected:
            fail(f"checksum mismatch: {relative}")
        if relative != "SEAL.json":
            canonical_entries.append((relative, expected))

    actual_files = {
        path.name
        for path in ROOT.iterdir()
        if path.is_file() and path.name != "SHA256SUMS.txt" and not path.name.startswith(".")
    }
    if actual_files != listed_files:
        fail(f"checksum coverage mismatch: missing={sorted(actual_files-listed_files)}, extra={sorted(listed_files-actual_files)}")

    seal = load_json("SEAL.json")
    assert_equal(seal["canonical_file_count"], len(listed_files) - 1, "canonical file count")
    assert_equal(seal["preregistration_sha256"], sha256(ROOT / "PROSPECTIVE_CONFIRMATION_PROTOCOL.json"), "protocol seal")
    canonical_lines = [
        f"{expected}  {relative}\n"
        for relative, expected in sorted(canonical_entries, key=lambda item: item[0])
    ]
    root_hash = hashlib.sha256("".join(canonical_lines).encode("utf-8")).hexdigest()
    assert_equal(seal["canonical_bundle_root_sha256"], root_hash, "canonical bundle root")
    for key in ["gpu_training_started", "pacs_training_started", "pacs_inference_started", "training_authorized", "submission_authorized"]:
        assert_equal(seal[key], False, f"seal {key}")
    assert_equal(seal["execution_gate"], "HOLD_IMPLEMENTATION_BINDING", "seal execution gate")
    assert_equal(seal["public_tag_planned"], "pacs-prospective-hsb-v2", "planned public tag")
    return {"sealed_files": len(listed_files)}


def main() -> int:
    check_json_protocol_ids()
    protocol = check_protocol()
    check_analysis_plan(protocol)
    counts = check_registered_manifests(protocol)
    check_gate_schema_and_family()
    checksum_counts = check_checksums()
    result = {
        "status": "PASS_SCIENTIFIC_PROTOCOL_SEALED_HOLD_EXECUTION",
        "protocol_id": protocol["protocol_id"],
        "gpu_training_started": False,
        "training_authorized": False,
        "submission_authorized": False,
        "primary_rows": 90,
        "primary_candidate_rows": 1080,
        "primary_count_rows": 1440,
        "secondary_procedure_rows": 26910,
        "secondary_candidate_rows": 322920,
        "secondary_count_rows": 430560,
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
