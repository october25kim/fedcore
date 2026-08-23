"""Tests for plan-driven synthetic validity replay."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from fedcore.experiments.run_synthetic_validity import (
    draw_sampling_counts,
    evaluate_plan,
    main,
    validate_plan,
)


def _plan(repetitions: int = 40) -> dict:
    return {
        "schema_version": 1,
        "campaign_seed": 808,
        "samplings": [
            {
                "sampling_id": "fixed-population-0",
                "repetitions": repetitions,
                "n": [120, 100, 80],
                "acceptance_probability": [0.8, 0.55, 0.7],
                "conditional_risk": [0.03, 0.08, 0.05],
                "proposal_A": [60, 35, 45],
                "proposal_K": [1, 4, 2],
            }
        ],
        "analyses": [
            {
                "analysis_id": "simplex-uniform",
                "sampling_id": "fixed-population-0",
                "alpha": 0.1,
                "total_delta": 0.1,
                "allocation_policy": "uniform",
                "Lambda": "simplex",
            },
            {
                "analysis_id": "box-informed",
                "sampling_id": "fixed-population-0",
                "alpha": 0.2,
                "total_delta": 0.05,
                "allocation_policy": "proposal_informed",
                "Lambda": "bounded",
                "lambda_lower": [0.1, 0.1, 0.1],
                "lambda_upper": [0.7, 0.7, 0.7],
            },
        ],
    }


def test_analysis_grid_reuses_bitwise_identical_counts():
    plan = _plan()
    rows = evaluate_plan(plan)
    assert len(rows) == 2
    assert rows[0]["counts_sha256"] == rows[1]["counts_sha256"]
    assert rows[0]["audit_seed"] == rows[1]["audit_seed"]
    assert rows[0]["Lambda"] == "simplex"
    assert rows[1]["Lambda"] == "bounded"
    assert all(0.0 <= row["empirical_joint_miss_rate"] <= 1.0 for row in rows)


def test_draw_replay_and_seed_ignore_analysis_values():
    plan = _plan()
    sampling = plan["samplings"][0]
    first = draw_sampling_counts(sampling, plan["campaign_seed"])
    plan["analyses"][0]["alpha"] = 0.09
    plan["analyses"][0]["total_delta"] = 0.08
    second = draw_sampling_counts(sampling, plan["campaign_seed"])
    np.testing.assert_array_equal(first.A, second.A)
    np.testing.assert_array_equal(first.K, second.K)
    assert first.digest == second.digest and first.audit_seed == second.audit_seed


def test_plan_validation_rejects_guessed_or_malformed_cells():
    plan = _plan()
    plan["analyses"][0]["score_name"] = "msp"
    with pytest.raises(ValueError, match="unsupported key"):
        validate_plan(plan)
    plan = _plan()
    plan["samplings"][0]["proposal_K"][0] = 1000
    with pytest.raises(ValueError, match="proposal counts"):
        validate_plan(plan)


def test_cli_writes_atomic_source_rows_and_manifest(tmp_path: Path):
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "rows.csv"
    plan_path.write_text(json.dumps(_plan(repetitions=8)), encoding="utf-8")
    assert main(["--plan", str(plan_path), "--output", str(output)]) == 0
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    manifest = json.loads((tmp_path / "rows.csv.manifest.json").read_text())
    assert manifest["status"] == "succeeded" and manifest["row_count"] == 2
