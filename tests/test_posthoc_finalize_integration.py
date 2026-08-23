"""End-to-end plan cell -> post-hoc artifact -> conservative finalizer test."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from fedcore.campaign.artifacts import (
    ArtifactRecord,
    RunManifest,
    file_sha256,
    semantic_hash,
    write_immutable_manifest,
)
from fedcore.campaign.finalize import finalize_campaign
from fedcore.campaign.plan import expand_training_cells
from fedcore.experiments.run_oneshot_posthoc import (
    JOINT_CONDITIONAL_CERTIFICATE_VARIANT,
    PosthocRequest,
    run_posthoc,
)
from fedcore.seeds import SeedBundle


_DATASET_SHA256 = "b" * 64


def _write_json(path: str, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _plan() -> dict:
    return {
        "schema_version": 1,
        "campaign_seed": 71,
        "cifar": {
            "class_splits": [
                {"id": f"split-{index}", "unknown_classes": [index, index + 5]}
                for index in range(5)
            ],
            "model_seeds": [0, 1, 2],
            "dirichlet_alpha": [0.1, 0.5, 5.0],
            "training": {"dataset": "fixture-cifar", "n_clients": 3},
        },
        "medical": {
            "heldout_diagnoses": [f"diagnosis-{index}" for index in range(8)],
            "model_seeds": [0, 1, 2],
            "training": {"dataset": "fixture-medical", "n_clients": 3},
        },
        "posthoc": {
            "alpha": [0.1],
            "total_delta": [0.1],
            "rho": [None],
            "threshold_policies": ["global", "client_specific"],
            "allocation_policies": ["uniform", "proposal_informed"],
            "audit_budget_fractions": [1.0],
            "traffic_sample_sizes": [0],
            "scores": ["msp"],
            "certificate_variants": [JOINT_CONDITIONAL_CERTIFICATE_VARIANT],
            "gammas": [0.7],
        },
    }


def _fold(seed: int, prefix: str, n: int = 360):
    rng = np.random.default_rng(seed)
    clients = np.tile(np.arange(3, dtype=np.int64), n // 3)
    labels = rng.integers(0, 2, size=n, dtype=np.int64)
    unknown = rng.random(n) < 0.25
    labels[unknown] = -1
    logits = rng.normal(0.0, 0.2, size=(n, 2))
    known = labels >= 0
    logits[np.flatnonzero(known), labels[known]] += 3.0
    logits[unknown] *= 0.2
    ids = np.asarray([f"{prefix}:{index}" for index in range(n)], dtype="U32")
    return logits, labels, clients, ids


def test_real_posthoc_output_is_accepted_with_separate_plan_and_training_hashes():
    with tempfile.TemporaryDirectory() as root:
        plan = _plan()
        plan_path = os.path.join(root, "plan.json")
        _write_json(plan_path, plan)
        cells = expand_training_cells(plan)
        cell = cells[0]
        dynamic_config = {
            "schema_version": 1,
            "dataset": "fixture-cifar",
            "dataset_sha256": _DATASET_SHA256,
            "n_clients": 3,
            "dirichlet_alpha": cell.config["dirichlet_alpha"],
            "rounds": 2,
            "execution_detail": "deliberately-not-the-plan-cell-config",
            "plan_cell_config_sha256": cell.config_hash,
        }
        dynamic_hash = semantic_hash(dynamic_config)
        assert dynamic_hash != cell.config_hash
        artifact = os.path.join(root, "training.npz")
        seed_bundle = SeedBundle.derive(
            71, common_context={"campaign": "posthoc-finalize-fixture"}
        )
        arrays: dict[str, np.ndarray] = {
            "dataset": np.asarray("fixture-cifar"),
            "experiment_id": np.asarray(cell.experiment_id),
            "dataset_sha256": np.asarray(_DATASET_SHA256),
            "training_config_json": np.asarray(
                json.dumps(dynamic_config, sort_keys=True, separators=(",", ":"))
            ),
            "training_config_sha256": np.asarray(dynamic_hash),
            "plan_cell_config_json": np.asarray(
                json.dumps(cell.config, sort_keys=True, separators=(",", ":"))
            ),
            "plan_cell_config_sha256": np.asarray(cell.config_hash),
            "seed_ledger_json": np.asarray(seed_bundle.to_json()),
        }
        for index, fold in enumerate(("prop", "cert", "test"), start=1):
            logits, labels, clients, ids = _fold(index, fold)
            arrays[f"{fold}_logits"] = logits
            arrays[f"{fold}_y_open"] = labels
            arrays[f"{fold}_client"] = clients
            arrays[f"{fold}_sample_id"] = ids
        np.savez_compressed(artifact, **arrays)

        training_manifest = artifact + ".manifest.json"
        write_immutable_manifest(
            training_manifest,
            RunManifest(
                schema_version=1,
                experiment_id=cell.experiment_id,
                status="completed",
                training_config=dynamic_config,
                posthoc_config={},
                seeds=seed_bundle.seeds,
                config_hash=dynamic_hash,
                code_commit="UNAVAILABLE",
                dataset_hash=_DATASET_SHA256,
                fold_hash="bound-inside-artifact",
                started_at="2026-01-01T00:00:00+00:00",
                ended_at="2026-01-01T00:00:01+00:00",
                checkpoint_path="",
                stdout_path="",
                stderr_path="",
                artifacts=(
                    ArtifactRecord.from_path(artifact, "immutable_model_outputs"),
                ),
            ),
        )

        result_path = os.path.join(root, "posthoc.json")
        result_manifest = result_path + ".manifest.json"
        result = run_posthoc(
            PosthocRequest(
                input_path=artifact,
                output_path=result_path,
                manifest_path=result_manifest,
                scores=("msp",),
                gammas=(0.7,),
                alpha=0.1,
                total_delta=0.1,
                delta_conditional_risk=0.05,
                delta_acceptance_lower=0.05,
                delta_acceptance_upper=0.0,
                delta_mixture=0.0,
                mixture_mode="simplex",
                certificate_variant=JOINT_CONDITIONAL_CERTIFICATE_VARIANT,
            )
        )
        assert result["row_count"] == 4

        records = []
        for planned in cells:
            if planned.experiment_id == cell.experiment_id:
                records.append(
                    {
                        "experiment_id": cell.experiment_id,
                        "family": cell.family,
                        "config_hash": cell.config_hash,
                        "status": "succeeded",
                        "reason": "",
                        "artifact_path": artifact,
                        "artifact_sha256": file_sha256(artifact),
                        "manifest_path": training_manifest,
                        "posthoc_outputs": [
                            {
                                "path": result_path,
                                "sha256": file_sha256(result_path),
                                "manifest_path": result_manifest,
                            }
                        ],
                    }
                )
            else:
                records.append(
                    {
                        "experiment_id": planned.experiment_id,
                        "family": planned.family,
                        "config_hash": planned.config_hash,
                        "status": "blocked",
                        "reason": "fixture external prerequisite",
                        "posthoc_outputs": [],
                    }
                )
        records_path = os.path.join(root, "records.json")
        _write_json(
            records_path,
            {
                "schema_version": 1,
                "plan_sha256": semantic_hash(plan),
                "records": records,
            },
        )
        collection = finalize_campaign(
            plan_path, records_path, os.path.join(root, "final"), "allow-blocked"
        )
        assert len(collection.rows) == 4
        assert collection.report["manuscript_ready"] is False
        assert {row["plan_cell_config_sha256"] for row in collection.rows} == {
            cell.config_hash
        }
        assert {row["training_config_sha256"] for row in collection.rows} == {
            dynamic_hash
        }
