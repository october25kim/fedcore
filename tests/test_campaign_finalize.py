"""Focused tests for conservative final campaign validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcore.campaign.artifacts import (  # noqa: E402
    ArtifactRecord,
    RunManifest,
    file_sha256,
    semantic_hash,
    write_immutable_manifest,
)
from fedcore.campaign.finalize import (  # noqa: E402
    CampaignValidationError,
    collect_campaign,
    expected_posthoc_cells,
    finalize_campaign,
    finalize_unplanned_blocked_state,
    inspect_training_artifact,
)
from fedcore.campaign.plan import expand_training_cells  # noqa: E402
from fedcore.certificate.variants import (  # noqa: E402
    JOINT_CONDITIONAL_CERTIFICATE_VARIANT,
)


def _plan() -> dict:
    return {
        "schema_version": 1,
        "campaign_seed": 17,
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
            "rho": [0.0],
            "threshold_policies": ["global", "client_specific"],
            "allocation_policies": ["uniform", "proposal_informed"],
            "audit_budget_fractions": [1.0],
            "traffic_sample_sizes": [0],
            "scores": ["msp"],
            "certificate_variants": [JOINT_CONDITIONAL_CERTIFICATE_VARIANT],
            "gammas": [0.7],
        },
    }


def _write_json(path: str, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _actual_training_config(cell) -> dict:
    return {
        **cell.config,
        "rounds": 3,
        "local_epochs": 1,
        "optimizer": "fixture-sgd",
    }


def _write_training_artifact(path: str, cell) -> None:
    training_config = _actual_training_config(cell)
    arrays: dict[str, np.ndarray] = {
        "dataset": np.asarray(cell.config["dataset"]),
        "experiment_id": np.asarray(cell.experiment_id),
        "plan_cell_config_json": np.asarray(
            json.dumps(cell.config, sort_keys=True, separators=(",", ":"))
        ),
        "plan_cell_config_sha256": np.asarray(cell.config_hash),
        "training_config_json": np.asarray(
            json.dumps(training_config, sort_keys=True, separators=(",", ":"))
        ),
        "training_config_sha256": np.asarray(semantic_hash(training_config)),
    }
    labels = np.asarray([-1, 0, 1, 0] * 3, dtype=np.int64)
    clients = np.tile(np.arange(3), 4).astype(np.int64)
    for fold_index, fold in enumerate(("prop", "cert", "test")):
        logits = np.tile(np.asarray([[0.2, -0.1], [2.0, -1.0]]), (6, 1))
        arrays[f"{fold}_logits"] = logits
        arrays[f"{fold}_y_open"] = labels
        arrays[f"{fold}_client"] = clients
        arrays[f"{fold}_sample_id"] = np.asarray(
            [f"{fold}:{fold_index}:{index}" for index in range(12)], dtype="U32"
        )
    np.savez_compressed(path, **arrays)


def _training_manifest(
    path: str, cell, artifact: str, training_config_hash: str
) -> None:
    _write_json(
        path,
        {
            "manifest_type": "fedcore.campaign.artifact",
            "schema_version": 1,
            "experiment_id": cell.experiment_id,
            "config_hash": training_config_hash,
            "artifact": {
                "path": os.path.abspath(artifact),
                "sha256": file_sha256(artifact),
                "size_bytes": os.path.getsize(artifact),
            },
        },
    )


def _successful_record(
    root: str,
    plan: dict,
    cell,
    *,
    omit_last_grid_row: bool = False,
    bad_accounting: bool = False,
    metric_overrides: dict | None = None,
) -> dict:
    artifact = os.path.join(root, "runs", f"{cell.experiment_id}.npz")
    Path(artifact).parent.mkdir(parents=True, exist_ok=True)
    _write_training_artifact(artifact, cell)
    training_manifest = artifact + ".manifest.json"
    info = inspect_training_artifact(artifact, cell.experiment_id, cell.config_hash)
    _training_manifest(training_manifest, cell, artifact, info.training_config_hash)

    posthoc_config = {
        "fixture": True,
        "source_experiment_id": cell.experiment_id,
        "scores": ["msp"],
        "gammas": [0.7],
        "alpha": 0.1,
        "failure_budget": {
            "delta_total": 0.1,
            "delta_mixture": 0.0,
            "delta_conditional_risk": 0.05,
            "delta_acceptance_lower": 0.05,
            "delta_acceptance_upper": 0.0,
            "delta_spent": 0.09 if bad_accounting else 0.1,
            "delta_slack": 0.0,
        },
    }
    posthoc_hash = semantic_hash(posthoc_config)
    rows = []
    for grid in expected_posthoc_cells(plan):
        accepted = 9 if grid["threshold_policy"] == "global" else 7
        row = {
            "certified": False,
            "cert_risk_ucb": 0.2,
            "cert_coverage_lcb": 0.3,
            "cert_n": accepted,
            "cert_k": 0,
            "prop_coverage": 0.5,
            "prop_risk": 0.0,
            "test_coverage": 0.5,
            "test_risk": 0.0,
            "score_name": grid["score_name"],
            "gamma": grid["gamma"],
            "alpha": grid["alpha"],
            "delta": grid["delta"],
            "Lambda": "simplex",
            "dirichlet_alpha": cell.config.get("dirichlet_alpha"),
            "n_clients": 3,
            "threshold_policy": grid["threshold_policy"],
            "allocation_policy": grid["allocation_policy"],
            "certificate_variant": grid["certificate_variant"],
            "rho": grid["rho"],
            "audit_budget_fraction": grid["audit_budget_fraction"],
            "traffic_sample_size": grid["traffic_sample_size"],
            "delta_total": 0.1,
            "delta_mixture": 0.0,
            "delta_conditional_risk": 0.05,
            "delta_acceptance_lower": 0.05,
            "delta_acceptance_upper": 0.0,
            "delta_spent": 0.09 if bad_accounting else 0.1,
            "delta_slack": 0.0,
            "source_experiment_id": cell.experiment_id,
            "input_artifact_sha256": info.sha256,
            "training_config_sha256": info.training_config_hash,
            "fold_sha256": info.fold_hash,
            "posthoc_config_sha256": posthoc_hash,
            "cert_sample_ids_sha256": info.cert_sample_ids_sha256,
            "traffic_sample_ids_sha256": "",
            "cert_observation_count": 12,
            "certificate_feasible": True,
            "certificate_reason": "ok",
        }
        row.update(metric_overrides or {})
        rows.append(row)
    if omit_last_grid_row:
        rows.pop()

    result_path = os.path.join(root, "posthoc", f"{cell.experiment_id}.json")
    result = {
        "schema_version": 1,
        "artifact_type": "fedcore.oneshot.posthoc-results",
        "status": "completed",
        "source_experiment_id": cell.experiment_id,
        "posthoc_experiment_id": f"{cell.experiment_id}-posthoc-fixture",
        "input_artifact": {
            "path": os.path.abspath(artifact),
            "sha256": info.sha256,
            "size_bytes": os.path.getsize(artifact),
        },
        "training_config_sha256": info.training_config_hash,
        "fold_sha256": info.fold_hash,
        "posthoc_config": posthoc_config,
        "posthoc_config_sha256": posthoc_hash,
        "row_count": len(rows),
        "rows": rows,
    }
    _write_json(result_path, result)
    result_manifest = result_path + ".manifest.json"
    write_immutable_manifest(
        result_manifest,
        RunManifest(
            schema_version=1,
            experiment_id=result["posthoc_experiment_id"],
            status="completed",
            training_config=_actual_training_config(cell),
            posthoc_config={
                **posthoc_config,
                "posthoc_config_sha256": posthoc_hash,
            },
            seeds={},
            config_hash=info.training_config_hash,
            code_commit="UNAVAILABLE",
            dataset_hash="fixture",
            fold_hash=info.fold_hash,
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="2026-01-01T00:00:01+00:00",
            checkpoint_path="",
            stdout_path="",
            stderr_path="",
            artifacts=(
                ArtifactRecord.from_path(artifact, "immutable_model_outputs"),
                ArtifactRecord.from_path(result_path, "posthoc_results"),
            ),
        ),
    )
    return {
        "experiment_id": cell.experiment_id,
        "family": cell.family,
        "config_hash": cell.config_hash,
        "status": "succeeded",
        "reason": "",
        "artifact_path": os.path.abspath(artifact),
        "artifact_sha256": info.sha256,
        "manifest_path": os.path.abspath(training_manifest),
        "posthoc_outputs": [
            {
                "path": os.path.abspath(result_path),
                "sha256": file_sha256(result_path),
                "manifest_path": os.path.abspath(result_manifest),
            }
        ],
    }


def _blocked_record(cell, status: str = "blocked") -> dict:
    return {
        "experiment_id": cell.experiment_id,
        "family": cell.family,
        "config_hash": cell.config_hash,
        "status": status,
        "reason": "fixture external prerequisite is unavailable",
        "posthoc_outputs": [],
    }


def _fixture(
    root: str,
    *,
    success: bool = True,
    omit_record: bool = False,
    nonterminal: bool = False,
    omit_last_grid_row: bool = False,
    bad_accounting: bool = False,
    metric_overrides: dict | None = None,
):
    plan = _plan()
    plan_path = os.path.join(root, "plan.json")
    _write_json(plan_path, plan)
    cells = expand_training_cells(plan)
    records = [_blocked_record(cell) for cell in cells]
    if success:
        records[0] = _successful_record(
            root,
            plan,
            cells[0],
            omit_last_grid_row=omit_last_grid_row,
            bad_accounting=bad_accounting,
            metric_overrides=metric_overrides,
        )
    if nonterminal:
        records[1] = {
            **_blocked_record(cells[1]),
            "status": "running",
            "reason": "",
        }
    if omit_record:
        records.pop()
    records_path = os.path.join(root, "records.json")
    _write_json(
        records_path,
        {
            "schema_version": 1,
            "plan_sha256": semantic_hash(plan),
            "records": records,
        },
    )
    return plan_path, records_path


def _expect_validation_error(fn):
    try:
        fn()
    except CampaignValidationError:
        return
    raise AssertionError("expected CampaignValidationError")


def test_allow_blocked_emits_only_observed_negative_rows_with_checksums():
    with tempfile.TemporaryDirectory() as root:
        plan, records = _fixture(root)
        _expect_validation_error(lambda: collect_campaign(plan, records, "complete"))
        out = os.path.join(root, "final")
        result = finalize_campaign(plan, records, out, "allow-blocked")
        assert len(result.rows) == 4
        assert all(row["certified"] is False for row in result.rows)
        assert all(row["risk_output_type"] == "numerical_ucb" for row in result.rows)
        assert all(row["risk_pass"] is False for row in result.rows)
        assert all(row["family_procedure"] == "single_selector" for row in result.rows)
        assert all(row["delta_r"] == 0.05 for row in result.rows)
        assert all(row["delta_c"] == 0.05 for row in result.rows)
        assert result.report["manuscript_ready"] is False
        assert result.report["terminal_unsuccessful_training_record_count"] == 68
        assert Path(out, "all_runs.csv").is_file()
        assert Path(out, "all_runs.parquet").is_file()
        assert Path(out, "checksums.sha256").is_file()
        record_bundle = json.loads(Path(records).read_text(encoding="utf-8"))
        first_record = record_bundle["records"][0]
        training_manifest = json.loads(
            Path(first_record["manifest_path"]).read_text(encoding="utf-8")
        )
        # Scheduler/post-hoc provenance binds the full dynamic training config;
        # the record remains bound independently to the authoritative plan cell.
        assert training_manifest["config_hash"] != first_record["config_hash"]
        assert (
            result.rows[0]["training_config_sha256"] == training_manifest["config_hash"]
        )
        assert result.rows[0]["plan_cell_config_sha256"] == first_record["config_hash"]
        import pyarrow.parquet as pq

        assert pq.read_table(Path(out, "all_runs.parquet")).num_rows == 4


def test_missing_and_nonterminal_records_need_explicit_incomplete_policy():
    with tempfile.TemporaryDirectory() as root:
        plan, records = _fixture(
            root,
            success=False,
            omit_record=True,
            nonterminal=True,
        )
        _expect_validation_error(
            lambda: collect_campaign(plan, records, "allow-blocked")
        )
        result = finalize_campaign(
            plan, records, os.path.join(root, "final"), "allow-incomplete"
        )
        assert len(result.rows) == 0
        assert result.report["missing_training_record_count"] == 1
        assert result.report["nonterminal_training_record_count"] == 1
        with open(
            os.path.join(root, "final", "all_runs.csv"), encoding="utf-8"
        ) as handle:
            assert len(handle.read().splitlines()) == 1
        import pyarrow as pa
        import pyarrow.parquet as pq

        empty = pq.read_table(os.path.join(root, "final", "all_runs.parquet"))
        assert empty.num_rows == 0
        assert empty.schema.field("certified").type == pa.bool_()
        assert empty.schema.field("risk_pass").type == pa.bool_()
        assert empty.schema.field("cert_n").type == pa.int64()
        assert empty.schema.field("holm_rank").type == pa.int64()
        assert empty.schema.field("cert_risk_ucb").type == pa.float64()
        assert empty.schema.field("iut_raw_pvalue").type == pa.float64()


def test_fixed_alpha_decision_uses_decision_and_positive_coverage_not_a_ucb():
    fixed_alpha = {
        "certified": True,
        "cert_risk_ucb": None,
        "cert_coverage_lcb": 0.2,
        "delta_r": 0.05,
        "delta_c": 0.05,
        "family_procedure": "holm_iut",
        "risk_output_type": "fixed_alpha_decision",
        "risk_pass": True,
        "iut_raw_pvalue": 0.01,
        "holm_adjusted_pvalue": 0.02,
        "holm_rank": 1,
    }
    with tempfile.TemporaryDirectory() as root:
        plan, records = _fixture(root, metric_overrides=fixed_alpha)
        result = collect_campaign(plan, records, "allow-blocked")
        assert len(result.rows) == 4
        assert all(row["certified"] is True for row in result.rows)
        assert all(row["cert_risk_ucb"] is None for row in result.rows)

    invalid_cases = (
        {**fixed_alpha, "cert_risk_ucb": 0.05},
        {**fixed_alpha, "risk_pass": False},
        {**fixed_alpha, "cert_coverage_lcb": 0.0},
        {**fixed_alpha, "family_procedure": "simple_simultaneous"},
        {**fixed_alpha, "holm_adjusted_pvalue": None},
        {**fixed_alpha, "holm_adjusted_pvalue": 0.005},
    )
    for overrides in invalid_cases:
        with tempfile.TemporaryDirectory() as root:
            plan, records = _fixture(root, metric_overrides=overrides)
            _expect_validation_error(
                lambda plan=plan, records=records: collect_campaign(
                    plan, records, "allow-blocked"
                )
            )

    refused_without_selected_member = {
        **fixed_alpha,
        "certified": False,
        "risk_pass": False,
        "cert_coverage_lcb": 0.0,
        "iut_raw_pvalue": None,
        "holm_adjusted_pvalue": None,
        "holm_rank": None,
    }
    with tempfile.TemporaryDirectory() as root:
        plan, records = _fixture(
            root, metric_overrides=refused_without_selected_member
        )
        result = collect_campaign(plan, records, "allow-blocked")
        assert all(row["certified"] is False for row in result.rows)


def test_partial_posthoc_grid_is_reported_without_fabricating_the_missing_row():
    with tempfile.TemporaryDirectory() as root:
        plan, records = _fixture(root, omit_last_grid_row=True)
        _expect_validation_error(
            lambda: collect_campaign(plan, records, "allow-blocked")
        )
        result = finalize_campaign(
            plan, records, os.path.join(root, "final"), "allow-incomplete"
        )
        assert len(result.rows) == 3
        assert result.report["missing_posthoc_cell_count"] == 1


def test_accounting_corruption_is_fatal_even_in_incomplete_mode():
    with tempfile.TemporaryDirectory() as root:
        plan, records = _fixture(root, bad_accounting=True)
        _expect_validation_error(
            lambda: collect_campaign(plan, records, "allow-incomplete")
        )
        assert not os.path.exists(os.path.join(root, "final", "all_runs.csv"))


def test_checksum_and_fold_identity_corruption_are_always_fatal():
    with tempfile.TemporaryDirectory() as root:
        plan, records_path = _fixture(root)
        records = json.loads(Path(records_path).read_text(encoding="utf-8"))
        records["records"][0]["artifact_sha256"] = "0" * 64
        _write_json(records_path, records)
        _expect_validation_error(
            lambda: collect_campaign(plan, records_path, "allow-incomplete")
        )

    with tempfile.TemporaryDirectory() as root:
        plan = _plan()
        cell = expand_training_cells(plan)[0]
        artifact = os.path.join(root, "overlap.npz")
        _write_training_artifact(artifact, cell)
        with np.load(artifact, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
        arrays["test_sample_id"][0] = arrays["prop_sample_id"][0]
        np.savez_compressed(artifact, **arrays)
        _expect_validation_error(
            lambda: inspect_training_artifact(
                artifact, cell.experiment_id, cell.config_hash
            )
        )


def test_unplanned_blocked_state_emits_typed_zero_rows_and_never_manuscript_ready():
    with tempfile.TemporaryDirectory() as root:
        blocked = os.path.join(root, "blocked.json")
        _write_json(
            blocked,
            {
                "schema_version": 1,
                "campaign_status": "blocked_external_prerequisite",
                "primary_progress": {
                    "cifar_required": 45,
                    "cifar_executed": 0,
                    "fed_isic_required": 24,
                    "fed_isic_executed": 0,
                    "primary_jobs_queued": 0,
                },
                "blockers": [{"id": "missing-plan", "remediation": "restore plan"}],
            },
        )
        out = os.path.join(root, "final")
        result = finalize_unplanned_blocked_state(blocked, out)
        assert result.rows == ()
        assert result.report["coverage_policy_passed"] is False
        assert result.report["manuscript_ready"] is False
        import pyarrow.parquet as pq

        assert pq.read_table(os.path.join(out, "all_runs.parquet")).num_rows == 0
        assert Path(out, "tables", "BLOCKED.md").is_file()
        assert Path(out, "figures", "BLOCKED.md").is_file()

    with tempfile.TemporaryDirectory() as root:
        plan = _plan()
        cell = expand_training_cells(plan)[0]
        artifact = os.path.join(root, "stale-training-hash.npz")
        _write_training_artifact(artifact, cell)
        with np.load(artifact, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
        # The plan hash is valid but is not interchangeable with the dynamic
        # training hash.
        arrays["training_config_sha256"] = np.asarray(cell.config_hash)
        np.savez_compressed(artifact, **arrays)
        _expect_validation_error(
            lambda: inspect_training_artifact(
                artifact, cell.experiment_id, cell.config_hash
            )
        )


def main():
    test_allow_blocked_emits_only_observed_negative_rows_with_checksums()
    test_missing_and_nonterminal_records_need_explicit_incomplete_policy()
    test_fixed_alpha_decision_uses_decision_and_positive_coverage_not_a_ucb()
    test_partial_posthoc_grid_is_reported_without_fabricating_the_missing_row()
    test_accounting_corruption_is_fatal_even_in_incomplete_mode()
    test_checksum_and_fold_identity_corruption_are_always_fatal()
    test_unplanned_blocked_state_emits_typed_zero_rows_and_never_manuscript_ready()
    print("campaign finalize tests: PASS")


if __name__ == "__main__":
    main()
