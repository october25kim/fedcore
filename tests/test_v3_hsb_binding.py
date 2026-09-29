"""Synthetic-only execution-binding tests for the sealed PACS H/S/B v3 protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fedcore.certificate.cp import cp_lower, cp_upper
from fedcore.experiments.v3_contract import (
    ALPHA,
    CLIENTS,
    ContractError,
    candidate_family,
    load_model_cells,
    verify_scientific_contract,
)
from fedcore.experiments.v3_hsb import candidate_decisions, validate_hsb_invariants
from fedcore.experiments.v3_orchestrator import (
    COUNT_FIELDS,
    read_csv,
    run_synthetic,
    write_csv,
)
from fedcore.experiments.v3_proposal import ProposalMember, proposal_risk_eligible
from fedcore.experiments.v3_replay_independent import replay
from fedcore.experiments.v3_validate import (
    require_scientific_execution_authorization,
    validate_mount_scope,
    validate_synthetic_output,
)
from fedcore.officehome_rescue import holm_adjusted_pvalues, holm_step_down_reject


def _members() -> tuple[ProposalMember, ...]:
    return tuple(
        ProposalMember(item.candidate_index, item.score, item.gamma, True, 0.5, 100, 80, 0, 0.0)
        for item in candidate_family()
    )


def _strict_counts(model_id: str) -> list[dict[str, object]]:
    accepted = (64, 100, 128) + (128,) * 9
    errors = (3, 9, 14) + (15,) * 9
    return [
        {
            "model_id": model_id,
            "candidate_index": m,
            "client": client,
            "status": "ok",
            "n": 128,
            "A": accepted[m],
            "K": errors[m],
        }
        for m in range(12)
        for client in CLIENTS
    ]


def test_sealed_contract_loads_without_authorizing_training() -> None:
    report = verify_scientific_contract()
    assert report["models"] == 30
    assert report["candidates"] == 12
    assert report["scientific_training_authorized"] is False


@pytest.mark.parametrize(
    ("gamma", "pass_counts", "fail_counts"),
    [
        (0.3, (3, 50), (3, 49)),
        (0.5, (1, 10), (1, 9)),
        (0.7, (7, 50), (7, 49)),
        (1.0, (1, 5), (1, 4)),
    ],
)
def test_proposal_integer_boundary(
    gamma: float, pass_counts: tuple[int, int], fail_counts: tuple[int, int]
) -> None:
    assert proposal_risk_eligible(*pass_counts, gamma)
    assert not proposal_risk_eligible(*fail_counts, gamma)
    assert not proposal_risk_eligible(0, 0, gamma)


def test_clopper_pearson_registered_boundaries() -> None:
    eta_s = 0.05 / 12
    eta_b = 0.05 / 48
    assert cp_upper(13, 128, eta_s) == pytest.approx(0.19186463091460090)
    assert cp_upper(14, 128, eta_s) == pytest.approx(0.20157005639164052)
    assert cp_upper(12, 128, eta_b) == pytest.approx(0.19848546494117050)
    assert cp_upper(13, 128, eta_b) == pytest.approx(0.20856279829725974)
    assert cp_lower(0, 128, eta_s) == 0.0
    assert cp_lower(1, 128, eta_s) == pytest.approx(3.261955711505705e-5)
    assert cp_lower(128, 128, eta_s) == pytest.approx(0.95808623290774408)
    assert cp_lower(128, 128, eta_b) == pytest.approx(0.94776574064782815)
    assert cp_upper(0, 0, eta_s) == 1.0
    assert cp_upper(128, 128, eta_s) == 1.0


def test_holm_inclusive_boundary_stable_order_and_stop() -> None:
    boundary = 0.05 / 12
    equal = np.full(12, boundary)
    rejected = holm_step_down_reject(equal, 0.05)
    adjusted, ranks = holm_adjusted_pvalues(equal)
    assert rejected.tolist() == [True] * 12
    assert adjusted == pytest.approx(np.full(12, 0.05))
    assert ranks.tolist() == list(range(1, 13))
    above = equal.copy()
    above[0] = np.nextafter(boundary, np.inf)
    above[1:] = above[0]
    assert holm_step_down_reject(above, 0.05).tolist() == [False] * 12
    stop = np.array([0.001, 0.1, 0.003] + [0.9] * 9)
    stopped = holm_step_down_reject(stop, 0.05)
    assert stopped[0]
    assert stopped[2]
    assert not stopped[1]


def test_strict_h_greater_s_greater_b_fixture() -> None:
    model = load_model_cells()[0]
    rows = candidate_decisions(
        model,
        _members(),
        _strict_counts(model.model_id),
        checkpoint_sha256="a" * 64,
        proposal_seal_sha256="b" * 64,
        audit_sequence_sha256="c" * 64,
    )
    validate_hsb_invariants(rows)
    selected = {
        procedure: next(row for row in rows if row.procedure == procedure and row.selected)
        for procedure in "HSB"
    }
    assert {key: row.candidate_index for key, row in selected.items()} == {
        "H": 2,
        "S": 1,
        "B": 0,
    }
    assert selected["H"].coverage_lcb == pytest.approx(0.95808623290774408)
    assert selected["S"].coverage_lcb == pytest.approx(0.67093576525266840)
    assert selected["B"].coverage_lcb == pytest.approx(0.36299343365835451)


def test_full_synthetic_denominators_and_independent_replay(tmp_path: Path) -> None:
    state = run_synthetic(tmp_path)
    assert state["status"] == "PASS_SYNTHETIC_COMPLETE"
    report = validate_synthetic_output(tmp_path)
    assert report["status"] == "PASS_SYNTHETIC_BINDING_TESTS"
    assert report["row_counts"] == {
        "proposal": 360,
        "counts": 1440,
        "candidates": 1080,
        "primary": 90,
        "truth": 90,
    }
    assert report["certified_cells"] == {"H": 27, "S": 27, "B": 27}


def test_resume_matches_uninterrupted_artifact_hashes(tmp_path: Path) -> None:
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    first = run_synthetic(uninterrupted)
    partial = run_synthetic(resumed, max_new_cells=7)
    assert partial["status"] == "HOLD_INCOMPLETE_SYNTHETIC"
    assert partial["completed_cells"] == 7
    second = run_synthetic(resumed)
    assert second["status"] == "PASS_SYNTHETIC_COMPLETE"
    assert first["artifact_hashes"] == second["artifact_hashes"]
    assert first["artifact_root_sha256"] == second["artifact_root_sha256"]


def test_independent_replay_detects_count_tampering(tmp_path: Path) -> None:
    run_synthetic(tmp_path)
    rows = read_csv(tmp_path / "PRIMARY_COUNTS.csv.gz")
    row = next(item for item in rows if item["status"] == "ok" and int(item["K"]) < int(item["A"]))
    row["K"] = str(int(row["K"]) + 1)
    write_csv(tmp_path / "PRIMARY_COUNTS.csv.gz", rows, COUNT_FIELDS, compressed=True)
    report = replay(tmp_path, write_report=False)
    assert report["status"] == "INVALID_REPLAY_MISMATCH"
    assert report["mismatch_count"] > 0


def test_mount_scope_and_scientific_gate_fail_closed(tmp_path: Path) -> None:
    allowed_input = tmp_path / "input.bin"
    output = tmp_path / "run"
    allowed_input.write_bytes(b"x")
    output.mkdir()
    expected = [
        {
            "type": "bind",
            "source": str(allowed_input),
            "destination": "/inputs/input.bin",
            "read_only": True,
        },
        {
            "type": "bind",
            "source": str(output),
            "destination": "/run",
            "read_only": False,
        },
        {"type": "tmpfs", "source": "", "destination": "/tmp", "read_only": False},
    ]
    validate_mount_scope(expected, expected, forbidden_roots=[tmp_path / "legacy"])
    unexpected = expected + [
        {
            "type": "bind",
            "source": str(tmp_path / "legacy" / "outputs"),
            "destination": "/leak",
            "read_only": True,
        }
    ]
    with pytest.raises(ContractError, match="INVALID_MOUNT_SCOPE"):
        validate_mount_scope(unexpected, expected, forbidden_roots=[tmp_path / "legacy"])
    gate = tmp_path / "EXECUTION_GATE.json"
    gate.write_text(
        json.dumps(
            {
                "protocol_id": "FEDCORE-IJAR-V34-PACS-FRESH-SEED-REPLICATION-HSB-v3",
                "status": "HOLD_IMPLEMENTATION_BINDING",
                "training_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="PASS_EXECUTION_READY"):
        require_scientific_execution_authorization(gate)
