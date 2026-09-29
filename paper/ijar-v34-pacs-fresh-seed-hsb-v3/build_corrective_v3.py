#!/usr/bin/env python3
"""Build the outcome-disclosed PACS fresh-seed replication protocol v3.

This script changes provenance, evidentiary labels, and execution-host binding
only.  It deliberately carries forward the v2 H/S/B numerical contract and
the exact unexecuted v2 seed manifests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V2_ROOT = ROOT.parent / "ijar-v34-prospective-pacs-hsb-v2"
V2_PID = "FEDCORE-IJAR-V34-PACS-PROSPECTIVE-HSB-v2"
V3_PID = "FEDCORE-IJAR-V34-PACS-FRESH-SEED-REPLICATION-HSB-v3"
SEALED_AT = "2026-09-29T08:15:00Z"
V2_RELEASE = "https://github.com/october25kim/fedcore/releases/tag/pacs-prospective-hsb-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def write(name: str, value: dict) -> None:
    (ROOT / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def update_protocol() -> None:
    protocol = load("PROSPECTIVE_CONFIRMATION_PROTOCOL.json")
    protocol["schema_version"] = 2
    protocol["protocol_id"] = V3_PID
    protocol["sealed_at_utc"] = SEALED_AT
    protocol["state"] = "SEALED_CORRECTIVE_SCIENTIFIC_PROTOCOL_NOT_EXECUTED"
    protocol["objective"] = (
        "This protocol prospectively registers a fresh-seed replication of the "
        "FedCORE H/S/B comparison on the previously studied, fixed PACS empirical "
        "frame. It evaluates 30 newly trained checkpoints under frozen selectors, "
        "audit streams, estimands, denominators, and decision rules. It does not "
        "constitute an untouched-dataset confirmation, an independent source-"
        "population replication, or evidence of prospective population transport."
    )
    protocol["supersession"] = {
        "invalidated_protocol_id": V2_PID,
        "invalidated_public_tag": "pacs-prospective-hsb-v2",
        "invalidated_release": V2_RELEASE,
        "invalidated_protocol_sha256": "7f5d976cf2bd0dfa0bd8ee51b8b6f833a5e611b33de2c8fd94e95d81b3444c68",
        "invalidated_tag_object": "ff3a5ff349bbc0394d1fa5d494b2491cf5f5255e",
        "invalidated_tagged_commit": "93418ed352e4212fef88e201a7d1e1d61972ff16",
        "v2_status": "INVALID_PRESEAL_PROVENANCE / DO NOT EXECUTE",
        "relationship": (
            "Version 3 supersedes the v2 scientific record because v2 incorrectly "
            "stated that no PACS training, inference, or scientific outcome existed "
            "before its seal. The v2 tag remains immutable as an audit record. "
            "Version 3 preserves the v2 H/S/B scientific contract and corrects its "
            "chronology, evidentiary label, execution host, and isolation requirements."
        ),
        "outcome_driven_change": "cannot be excluded and is not claimed absent",
    }
    protocol.pop("pre_outcome_amendment", None)
    protocol["prior_outcome_disclosure"] = {
        "prior_pacs_outcomes_exist": True,
        "prior_outcome_available_before_v3_seal": True,
        "prior_v1_protocol_id": "FEDCORE-V1-PACS-20260916-v1",
        "prior_v1_protocol_sha256": "a8c944d57246d2fc46b175982cdada76f1c797da30090a35fbb344bad0cb7c50",
        "prior_v1_decision": "HOLD",
        "prior_v1_decision_utc": "2026-09-17T12:51:21.891927+00:00",
        "prior_v1_decision_sha256": "027b18bceb69188531d1e423b64b61c981964f3515563ddd5a1264f86ce2312a",
        "prior_v1_models": 30,
        "prior_v1_rows_total": 162000,
        "prior_transfer_protocol_id": "FEDCORE-S1-DOMAINNET6-PACSCLS-20260925-v2",
        "prior_transfer_protocol_sha256": "06dd7e1a22f3d504dc76dd115b3f1803754ffe022a9d436e7dcef8a20c7c142b",
        "prior_transfer_decision": "KILL",
        "prior_transfer_label": "NOT_TRANSFER",
        "prior_transfer_decision_utc": "2026-09-25T12:48:57.641032+00:00",
        "prior_transfer_decision_sha256": "47c67e85854f81f67dfb2a3882e49d9de66aaea67d5ba24be19479fe8e3d1edd",
        "v3_specific_training_or_inference_seen": False,
        "v3_specific_primary_outcome_seen": False,
        "design_independence_from_prior_pacs_outcomes": "not claimed",
        "qualification": (
            "The H/S/B design was sealed after PACS outcomes from an earlier study "
            "were available. Outcome influence on the revised design cannot be "
            "excluded. Validity claims are limited to a prospectively registered "
            "fresh-seed replication after the v3 seal."
        ),
    }

    scope = protocol["scope"]
    scope["confirmation_type"] = (
        "prospectively registered fresh-seed replication on a previously studied "
        "fixed PACS empirical frame"
    )
    scope["new_dataset_relative_to_v34_headline"] = False
    scope["freshly_trained_checkpoints"] = 30
    scope["untouched_dataset_confirmation"] = False
    scope["independent_source_population_replication"] = False
    scope["external_validation_claimed"] = False
    scope["independence_qualification"] = (
        "The v3 checkpoints and random streams are fresh relative to the earlier "
        "PACS run. The source images, empirical frame, domain structure, class-split "
        "family, architectures, and ImageNet initialization families were previously "
        "studied. The 30 checkpoints are training-randomness replications conditional "
        "on this fixed frame, not independent datasets or source-population replicates."
    )

    frame = protocol["dataset"]["exact_frame"]
    frame["archive_path_on_execution_host"] = (
        "/home/sanghoon/Desktop/Workspace/Fedcore/data/v1_pacs/PACS_mirror.zip"
    )
    frame["archived_prior_audit_membership_overlap"] = (
        "SAME_HASHED_PACS_FRAME_AS_V1; source-disjointness from the V1 PACS study is not claimed"
    )

    training = protocol["training"]
    training["seed_namespace"] = {
        "seed_namespace_id": V2_PID,
        "seed_namespace_reason": (
            "Preserve the exact unexecuted v2 training and audit streams while "
            "correcting provenance only."
        ),
        "seed_manifest_change_from_v2": False,
        "v2_training_seed_manifest_sha256": "7c55b885e6c245b749ca8ba9d9462e077c85e60fa7a6b9f7cbe2ee0dc1e08bfc",
        "v2_audit_seed_manifest_sha256": "747b5416b2b5f09d0fbcdb2a94cbbc39825a1d44e2f80677ac8b0a859f290029",
        "v2_model_matrix_sha256": "ac91ffbf2ccebb1a35ffcd648051fc4c25db1fd7df6cc45c607d652e0e385104",
    }
    training["seed_derivation"] = (
        "big-endian integer from the first eight SHA-256 bytes of "
        "seed_namespace_id|label|compact_json(parts); seed_namespace_id is the "
        "frozen unexecuted v2 namespace"
    )

    gates = protocol["outcome_gates"]
    gates["precedence"] = [
        "INVALID",
        "HOLD_INCOMPLETE",
        "PASS_FRESH_SEED_REPLICATION",
        "PASS_CORE_ONLY",
        "VALID_NEGATIVE",
    ]
    gates["PASS_FRESH_SEED_REPLICATION"] = gates.pop(
        "PASS_FULL_PROSPECTIVE_CONFIRMATION"
    )

    protocol["resource_contract"] = {
        "execution_host_alias": "ubuntu-4070",
        "observed_hostname": "ubuntu",
        "isolated_execution_root": "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_v3_exec",
        "permitted_gpu_index": 0,
        "permitted_gpu_uuid": "GPU-ee5a0082-32e6-ede4-27d5-441bee0ca7c6",
        "permitted_gpu_name": "NVIDIA GeForce RTX 4070 Ti SUPER",
        "gpu_memory_mib": 16376,
        "gpu_hour_cap_total": 48,
        "capacity_stage_gpu_hours_included": 2,
        "cpu_core_hour_cap": 24,
        "cap_exhaustion_before_complete_accounting": (
            "HOLD_INCOMPLETE; do not extend the cap after observing a v3 scientific outcome"
        ),
        "gpu_assignment": (
            "the execution-binding seal must freeze physical GPU 0, its UUID, and "
            "single-process concurrency before PACS training"
        ),
    }
    protocol["runtime_isolation"] = {
        "legacy_workspace_read_only_reference": "/home/sanghoon/Desktop/Workspace/Fedcore",
        "isolated_execution_root": "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_v3_exec",
        "forbidden_runtime_mounts": [
            "/home/sanghoon/Desktop/Workspace/Fedcore",
            "/home/sanghoon/Desktop/Workspace/Fedcore/outputs",
            "/home/sanghoon/Desktop/Workspace/Fedcore/work",
        ],
        "permitted_read_only_input_types": [
            "PACS archive",
            "IMAGE_MANIFEST.csv",
            "FOLD_MANIFEST.csv",
            "ResNet-18 pretrained weight",
            "ConvNeXt-Tiny pretrained weight",
        ],
        "network_disabled_for_scientific_runtime": True,
        "prior_checkpoints_logits_counts_and_decisions_inaccessible": "required before PASS_EXECUTION_READY",
    }
    protocol["negative_result_policy"][-1] = (
        "A VALID_NEGATIVE is retained as the result of this fresh-seed replication "
        "and requires a new submission-readiness assessment."
    )
    protocol["preexecution_requirements"] = [
        "Preserve the v2 tag and publish its INVALID_PRESEAL_PROVENANCE / DO NOT EXECUTE notice.",
        "Publicly bind this corrected v3 protocol and its ordered SHA-256 manifest before any v3 PACS training or inference.",
        "Create and hash the isolated Ubuntu execution snapshot without mounting the legacy workspace or prior outputs.",
        "Freeze and verify the end-to-end 30-model orchestrator, resume logic, resource cap, output schema, H/S/B implementation, and independent replay implementation.",
        "Pin the exact container, dependencies, pretrained weights, CUDA runtime path, all execution-source hashes, and the permitted mount list.",
        "Pass PACS-free synthetic end-to-end, resume, failure-ledger, resource-cap, output-schema, H/S/B parity, and independent-replay tests.",
        "Publish a v3 execution-binding seal with PASS_EXECUTION_READY and obtain explicit user authorization to start PACS training.",
    ]
    protocol["output_policy"]["isolated_remote_directory"] = (
        "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_v3_exec/runs/scientific/"
        "FEDCORE-IJAR-V34-PACS-FRESH-SEED-REPLICATION-HSB-v3"
    )
    write("PROSPECTIVE_CONFIRMATION_PROTOCOL.json", protocol)


def update_analysis_plan() -> None:
    plan = load("H_S_B_ANALYSIS_PLAN.json")
    plan["protocol_id"] = V3_PID
    gates = plan["decision_gates"]
    gates["PASS_FRESH_SEED_REPLICATION"] = gates.pop(
        "PASS_FULL_PROSPECTIVE_CONFIRMATION"
    )
    write("H_S_B_ANALYSIS_PLAN.json", plan)


def update_supporting_files() -> None:
    for name in [
        "OUTCOME_SCHEMA.json",
        "SELECTOR_FAMILY_CONTRACT.json",
    ]:
        obj = load(name)
        obj["protocol_id"] = V3_PID
        write(name, obj)

    overlap = load("PACS_ROLE_OVERLAP_AUDIT.json")
    overlap["protocol_id"] = V3_PID
    overlap["archived_prior_audit_membership_binding"] = (
        "SAME_HASHED_PACS_FRAME_AS_V1; not source-disjoint from the prior PACS study"
    )
    overlap["execution_requirement"] = (
        "Do not claim PACS source-disjointness from V1. Retain only the verified "
        "separation from the listed non-PACS raw roots and the within-PACS role disjointness."
    )
    write("PACS_ROLE_OVERLAP_AUDIT.json", overlap)

    frozen = {
        "schema_version": 2,
        "protocol_id": V3_PID,
        "dataset": {
            "PACS_mirror.zip": {
                "path": "/home/sanghoon/Desktop/Workspace/Fedcore/data/v1_pacs/PACS_mirror.zip",
                "sha256": "42bf567f1ed8a01d522e47e4a677e2a3149577bbd6fcbb38bedfdd73cb59e147",
                "bytes": 184417365,
                "role": "governing source archive on the previously studied fixed frame",
            },
            "IMAGE_MANIFEST.csv": {
                "path": "/home/sanghoon/Desktop/Workspace/Fedcore/outputs/fedcore_v1_ubuntu/IMAGE_MANIFEST.csv",
                "sha256": "283d47bd8cfb7017d6e7fde6d65e07ff6479ab25127afcba135da3a06775afae",
                "records": 9991,
                "runtime_mount": "file-only read-only",
            },
            "FOLD_MANIFEST.csv": {
                "path": "/home/sanghoon/Desktop/Workspace/Fedcore/outputs/fedcore_v1_ubuntu/FOLD_MANIFEST.csv",
                "sha256": "fe5b64d237f47f503b8895aa5e566727bc818e05ee597f84c440df70e91e5344",
                "runtime_mount": "file-only read-only",
            },
        },
        "pretrained_weights": {
            "resnet18-f37072fd.pth": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
            "convnext_tiny-983f1562.pth": "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
        },
        "prior_outcome_references_not_runtime_inputs": {
            "V1_PROTOCOL.json": "a8c944d57246d2fc46b175982cdada76f1c797da30090a35fbb344bad0cb7c50",
            "V1_DECISION.json": "027b18bceb69188531d1e423b64b61c981964f3515563ddd5a1264f86ce2312a",
            "DOMAINNET_TRANSFER_DECISION.json": "47c67e85854f81f67dfb2a3882e49d9de66aaea67d5ba24be19479fe8e3d1edd",
        },
        "seed_carry_forward": {
            "namespace": V2_PID,
            "training_manifest_sha256": "7c55b885e6c245b749ca8ba9d9462e077c85e60fa7a6b9f7cbe2ee0dc1e08bfc",
            "audit_manifest_sha256": "747b5416b2b5f09d0fbcdb2a94cbbc39825a1d44e2f80677ac8b0a859f290029",
            "model_matrix_sha256": "ac91ffbf2ccebb1a35ffcd648051fc4c25db1fd7df6cc45c607d652e0e385104",
        },
        "environment": {
            "candidate_container_image_id": "sha256:f0f2b57a630805081d8a345fb11980b230556e5360223a3658cd40aceff61bb2",
            "candidate_only_not_execution_bound": True,
            "execution_environment_fully_bound": False,
        },
        "availability": {
            "full_dataset_and_manifests_publicly_redistributed_here": False,
            "hash_only_is_not_end_to_end_reproduction": True,
        },
    }
    write("FROZEN_INPUTS.json", frozen)

    environment = {
        "schema_version": 2,
        "protocol_id": V3_PID,
        "observation_host_alias": "ubuntu-4070",
        "observed_hostname": "ubuntu",
        "legacy_workspace": "/home/sanghoon/Desktop/Workspace/Fedcore",
        "legacy_workspace_git_repository": False,
        "isolated_execution_root": "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_v3_exec",
        "isolated_execution_root_created": False,
        "repository_binding": "PENDING_CLEAN_V3_GIT_SNAPSHOT",
        "candidate_container_image_id": "sha256:f0f2b57a630805081d8a345fb11980b230556e5360223a3658cd40aceff61bb2",
        "candidate_environment": {
            "python": "3.10.14",
            "torch": "2.3.0",
            "torchvision": "0.18.0",
            "numpy": "1.26.4",
            "scipy": "1.15.3",
            "Pillow": "10.2.0",
            "cuda": "12.1",
            "cudnn": 8902,
        },
        "gpu": {
            "index": 0,
            "name": "NVIDIA GeForce RTX 4070 Ti SUPER",
            "uuid": "GPU-ee5a0082-32e6-ede4-27d5-441bee0ca7c6",
            "memory_mib": 16376,
            "observed_idle_during_inventory": True,
        },
        "free_disk_gb_observed": 338,
        "execution_binding_status": "NOT_YET_FROZEN",
        "warning": (
            "The candidate environment is an inventory record only. A clean source "
            "commit, dedicated image, exact mount allowlist, and synthetic integration "
            "must be sealed before PACS training."
        ),
    }
    write("CODE_ENVIRONMENT_MANIFEST.json", environment)


def create_provenance_files() -> None:
    write(
        "PROVENANCE_INCIDENT.json",
        {
            "schema_version": 1,
            "protocol_id": V3_PID,
            "incident_id": "FEDCORE-PACS-PRESEAL-PROVENANCE-20260929",
            "detected_at_utc": "2026-09-29T07:00:00Z",
            "affected_release": V2_RELEASE,
            "affected_protocol_id": V2_PID,
            "classification": "INVALID_PRESEAL_PROVENANCE",
            "summary": (
                "The v2 seal asserted that no PACS training, inference, or scientific "
                "outcome existed, but a completed V1 PACS study and a later transfer "
                "probe predated that seal."
            ),
            "containment": [
                "Preserve the immutable v2 tag and files.",
                "Mark the public v2 release DO NOT EXECUTE.",
                "Do not use v2 to support an untouched prospective-confirmation claim.",
                "Create this outcome-disclosed corrective successor before new execution.",
            ],
            "v2_specific_experiment_observed": False,
            "scientific_consequence": (
                "A new run may test a prospectively registered fresh-seed replication "
                "conditional on the previously studied frame, but not untouched-dataset confirmation."
            ),
        },
    )
    write(
        "PRIOR_OUTCOME_LEDGER.json",
        {
            "schema_version": 1,
            "protocol_id": V3_PID,
            "records": [
                {
                    "role": "prior PACS scientific study",
                    "protocol_id": "FEDCORE-V1-PACS-20260916-v1",
                    "protocol_sha256": "a8c944d57246d2fc46b175982cdada76f1c797da30090a35fbb344bad0cb7c50",
                    "decision": "HOLD",
                    "decision_utc": "2026-09-17T12:51:21.891927+00:00",
                    "decision_sha256": "027b18bceb69188531d1e423b64b61c981964f3515563ddd5a1264f86ce2312a",
                    "models": 30,
                    "primary_rows": 54000,
                    "rows_total": 162000,
                    "execution_host_path": "/home/sanghoon/Desktop/Workspace/Fedcore/outputs/fedcore_v1_full_run/run_outputs/DECISION.json",
                },
                {
                    "role": "subsequent transfer probe using frozen PACS predictors",
                    "protocol_id": "FEDCORE-S1-DOMAINNET6-PACSCLS-20260925-v2",
                    "protocol_sha256": "06dd7e1a22f3d504dc76dd115b3f1803754ffe022a9d436e7dcef8a20c7c142b",
                    "decision": "KILL",
                    "transfer_label": "NOT_TRANSFER",
                    "decision_utc": "2026-09-25T12:48:57.641032+00:00",
                    "decision_sha256": "47c67e85854f81f67dfb2a3882e49d9de66aaea67d5ba24be19479fe8e3d1edd",
                    "models": 30,
                    "primary_rows": 54000,
                    "rows_total": 162000,
                    "execution_host_path": "/home/sanghoon/Desktop/Workspace/Fedcore/outputs/fedcore_s1v2_run_20260925/DECISION.json",
                },
            ],
            "v3_seal_after_all_listed_records": True,
            "design_independence_from_listed_records": "not claimed",
        },
    )
    write(
        "V2_SUPERSESSION_NOTICE.json",
        {
            "schema_version": 1,
            "protocol_id": V3_PID,
            "superseded_protocol_id": V2_PID,
            "superseded_release": V2_RELEASE,
            "superseded_status": "INVALID_PRESEAL_PROVENANCE / DO NOT EXECUTE",
            "tag_and_files_preserved": True,
            "tag_object": "ff3a5ff349bbc0394d1fa5d494b2491cf5f5255e",
            "tagged_commit": "93418ed352e4212fef88e201a7d1e1d61972ff16",
            "protocol_sha256": "7f5d976cf2bd0dfa0bd8ee51b8b6f833a5e611b33de2c8fd94e95d81b3444c68",
            "canonical_bundle_root_sha256": "36ff18ddae661715eae22c5b92f5ecb1888e301e524d75ade4bedf38d1f7a3e2",
            "seal_sha256": "5f5f92bba4267a843d145448514cd312d788a964cb8ea86c3e64e55ff72250e5",
            "training_authorized_by_v2": False,
            "submission_authorized_by_v2": False,
        },
    )
    write(
        "EXECUTION_ISOLATION_CONTRACT.json",
        {
            "schema_version": 1,
            "protocol_id": V3_PID,
            "execution_host_alias": "ubuntu-4070",
            "observed_hostname": "ubuntu",
            "execution_root": "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_v3_exec",
            "directory_layout": [
                "repo",
                "protocol",
                "inputs",
                "synthetic",
                "runs/synthetic",
                "runs/capacity",
                "runs/scientific",
                "seals",
                "logs",
            ],
            "legacy_workspace": "/home/sanghoon/Desktop/Workspace/Fedcore",
            "legacy_workspace_must_not_be_mounted": True,
            "forbidden_paths": [
                "/home/sanghoon/Desktop/Workspace/Fedcore/outputs",
                "/home/sanghoon/Desktop/Workspace/Fedcore/work",
                "/home/sanghoon/Desktop/Workspace/Fedcore/outputs/fedcore_v1_run",
                "/home/sanghoon/Desktop/Workspace/Fedcore/outputs/fedcore_v1_full_run",
                "/home/sanghoon/Desktop/Workspace/Fedcore/outputs/fedcore_s1v2_run_20260925",
            ],
            "runtime_conditions": {
                "network": "none",
                "root_filesystem": "read-only",
                "temporary_filesystem": "tmpfs /tmp",
                "inputs": "file-level read-only mounts only",
                "scientific_output": "one dedicated read-write run directory",
                "unexpected_mount_status": "INVALID_MOUNT_SCOPE",
            },
            "scientific_run_directory_must_be_empty_at_execution_seal": True,
            "status": "NOT_YET_IMPLEMENTED",
        },
    )


def update_review_gate_validation() -> None:
    review = {
        "schema_version": 2,
        "protocol_id": V3_PID,
        "review_type": "read-only cross-host provenance and corrective-design review",
        "reviewed_after_prior_pacs_outcome_disclosure": True,
        "reviewed_before_any_v3_specific_training_or_inference": True,
        "design_independence_from_prior_pacs_outcomes": "not claimed",
        "review_findings": {
            "chronology": "Prior PACS V1 and transfer outcomes predate this v3 seal and are fully disclosed.",
            "scientific_alignment": "The alpha, confidence budgets, Lambda, M, primary budget, denominators, selectors, H/S/B tails, and decision gates are carried forward from v2.",
            "seed_contract": "The exact unexecuted v2 training and audit seed manifests are carried forward byte-for-byte to avoid a provenance-driven scientific redesign.",
            "independence_boundary": "Fresh RNG and checkpoints do not create an untouched dataset, an independent source population, or external validation.",
            "implementation": "A clean Ubuntu source snapshot, dedicated image, mount isolation, and PACS-free synthetic integration are still required.",
            "negative_result_rule": "Every adverse v3 cell remains in the registered denominator and must be released.",
        },
        "verdict": "PASS_CORRECTIVE_SCIENTIFIC_PROTOCOL_HOLD_EXECUTION",
        "training_authorized": False,
        "submission_authorized": False,
    }
    write("INDEPENDENT_PRESEAL_REVIEW.json", review)

    gate = {
        "schema_version": 2,
        "protocol_id": V3_PID,
        "status": "HOLD_IMPLEMENTATION_BINDING",
        "scientific_protocol_sealed": True,
        "provenance_correction_complete": True,
        "training_authorized": False,
        "gpu_training_started": False,
        "pacs_inference_started": False,
        "submission_authorized": False,
        "reason": (
            "The corrected scientific contract is frozen, but no clean v3 source "
            "commit, dedicated runtime image, exact mount proof, or PACS-free "
            "synthetic end-to-end implementation binding exists yet."
        ),
        "required_before_PASS_EXECUTION_READY": [
            "Create the isolated Ubuntu directory and a clean v3 Git repository.",
            "Freeze every runner, H/S/B implementation, test, config, dependency, container, input hash, mount, and seed manifest.",
            "Prove that prior outputs, checkpoints, logits, counts, thresholds, and decisions are inaccessible from the runtime.",
            "Pass PACS-free synthetic end-to-end, resume, failure, cap, ordering, schema, monotonicity, and independent-replay tests.",
            "Publish a separate v3 execution-binding seal.",
            "Obtain explicit user authorization after PASS_EXECUTION_READY.",
        ],
        "forbidden_now": [
            "PACS model training",
            "PACS model inference",
            "proposal threshold construction from PACS logits",
            "PACS audit count generation",
            "v3 PACS performance inspection",
        ],
        "next_state": "PASS_EXECUTION_READY",
        "ready_submission_package_state": "NOT_APPLICABLE_UNTIL_EXPERIMENT_COMPLETES",
    }
    write("EXECUTION_GATE.json", gate)

    validation = {
        "schema_version": 2,
        "protocol_id": V3_PID,
        "observed_at_utc": SEALED_AT,
        "status": "PASS_CORRECTIVE_SCIENTIFIC_SEAL_HOLD_EXECUTION",
        "checks": {
            "prior_pacs_outcomes_disclosed": True,
            "v2_public_release_marked_do_not_execute": True,
            "v3_specific_checkpoint_files_found": 0,
            "v3_specific_result_files_found": 0,
            "v3_specific_training_started": False,
            "v3_specific_inference_started": False,
            "model_matrix_rows": 30,
            "model_matrix_NOT_RUN_rows": 30,
            "seed_manifests_byte_identical_to_v2": True,
            "pacs_images": 9991,
            "within_pacs_cross_role_identity_leakage": 0,
            "exact_CP_Holm_boundaries_registered": True,
            "primary_candidate_rows_registered": 1080,
            "secondary_procedure_rows_registered": 26910,
            "secondary_candidate_rows_registered": 322920,
            "secondary_count_rows_registered": 430560,
            "H_S_B_monotonicity_is_INVALID_invariant": True,
        },
        "known_execution_blockers": [
            "isolated Ubuntu execution root not yet created",
            "clean v3 Git source commit and dedicated image not yet bound",
            "no validated end-to-end 30-model H/S/B orchestrator",
            "mount isolation not yet demonstrated",
            "PACS-free synthetic integration and independent replay not yet passed",
        ],
        "conclusion": (
            "The outcome-disclosed fresh-seed replication contract may be timestamped. "
            "No v3 PACS scientific execution is authorized until a separate execution-binding seal passes."
        ),
    }
    write("PRESEAL_VALIDATION.json", validation)


def create_carryforward_ledger() -> None:
    files = [
        "TRAINING_SEED_MANIFEST.csv",
        "AUDIT_SEED_MANIFEST.csv",
        "PACS_MODEL_MATRIX.csv",
        "H_S_B_ANALYSIS_PLAN.json",
        "SELECTOR_FAMILY_CONTRACT.json",
        "OUTCOME_SCHEMA.json",
    ]
    ledger = {
        "schema_version": 1,
        "protocol_id": V3_PID,
        "v2_protocol_id": V2_PID,
        "purpose": "prove that provenance correction did not silently alter the H/S/B numerical contract or registered RNG streams",
        "files": {},
    }
    for name in files:
        ledger["files"][name] = {
            "v2_sha256": sha256(V2_ROOT / name),
            "v3_sha256": sha256(ROOT / name),
            "byte_identical": (V2_ROOT / name).read_bytes() == (ROOT / name).read_bytes(),
        }
    ledger["files"]["H_S_B_ANALYSIS_PLAN.json"]["allowed_delta"] = (
        "protocol_id and outcome-label rename only; numerical values and rules unchanged"
    )
    ledger["files"]["SELECTOR_FAMILY_CONTRACT.json"]["allowed_delta"] = "protocol_id only"
    ledger["files"]["OUTCOME_SCHEMA.json"]["allowed_delta"] = "protocol_id only"
    write("SCIENTIFIC_CONTRACT_CARRYFORWARD.json", ledger)


def update_readme() -> None:
    text = f"""# PACS H/S/B fresh-seed replication protocol v3

**Status: corrective scientific protocol sealed; implementation binding HOLD.**

This directory prospectively registers a fresh-seed FedCORE H/S/B replication
on a previously studied fixed PACS empirical frame. It does not authorize
training, inference, manuscript submission, or an untouched-dataset claim.

## Why v3 exists

The public v2 seal incorrectly stated that no PACS training, inference, or
scientific outcome existed before its seal. A read-only cross-host audit found
a completed 30-model PACS V1 study from 2026-09-17 and a later transfer probe
that reused the frozen PACS predictors. The v2 tag remains immutable and its
public release is marked `INVALID_PRESEAL_PROVENANCE / DO NOT EXECUTE`.

Version 3 discloses those outcomes and narrows the study label to:

> Prospectively registered fresh-seed replication on a previously studied
> fixed PACS empirical frame.

The v3 checkpoints and RNG streams will be fresh relative to V1. The dataset,
source frame, domain structure, class-split family, architectures, and
initialization families were previously studied. Consequently, this study is
not an independent source-population replication, external validation, or
evidence of unseen-population transport.

## Frozen scientific contract

The v2 H/S/B numerical contract is carried forward without changing its
selectors, candidate order, estimands, confidence budgets, audit budgets,
denominators, decision gates, or negative-result policy. The exact unexecuted
v2 training and audit seed manifests are retained byte-for-byte under the
frozen v2 seed namespace. `SCIENTIFIC_CONTRACT_CARRYFORWARD.json` records the
carry-forward hashes and permitted metadata-only differences.

The primary comparison uses 30 planned checkpoints, 12 proposal-frozen
candidates, `alpha=0.20`, `delta_r=delta_c=0.05`, the full simplex over four
PACS domains, and one budget-512 audit per model. H, S, and B must consume the
same count tensor. Any evidence-equality or monotonicity violation is
`INVALID`, not a scientific negative.

## Provenance and isolation

`PRIOR_OUTCOME_LEDGER.json` identifies both known pre-v3 outcome records.
`EXECUTION_ISOLATION_CONTRACT.json` prohibits mounting the legacy Fedcore
workspace or any prior PACS outputs. The future runtime may receive only the
five hash-bound input files as file-level read-only mounts and one dedicated
scientific output directory.

The planned execution root is:

```text
/home/sanghoon/Desktop/Workspace/Fedcore_HSB_v3_exec
```

The v3 runner, dedicated image, mount proof, resume logic, output schemas,
H/S/B parity, and independent replay have not yet been implementation-bound.

## Verification

From the repository root:

```bash
python paper/ijar-v34-pacs-fresh-seed-hsb-v3/verify_protocol.py
```

The expected state is
`PASS_CORRECTIVE_SCIENTIFIC_PROTOCOL_SEALED_HOLD_EXECUTION` with
`training_authorized=false` and `submission_authorized=false`.

## Next gate

The only permitted next stage is an isolated implementation-binding pass using
PACS-free synthetic data. PACS training and inference remain prohibited until
that stage produces a public `PASS_EXECUTION_READY` seal and the user gives a
separate explicit training authorization.
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    update_protocol()
    update_analysis_plan()
    update_supporting_files()
    create_provenance_files()
    update_review_gate_validation()
    create_carryforward_ledger()
    update_readme()


if __name__ == "__main__":
    main()
