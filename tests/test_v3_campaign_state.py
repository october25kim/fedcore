from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from fedcore.experiments.v3_contract import ContractError, load_model_cells
from fedcore.experiments.v3_campaign_state import (
    initial_campaign_state,
    next_state,
    seal_state,
    validate_campaign_state,
    validate_state_transition,
    verify_state_history,
    write_state_history,
)


AUTHORIZATION = "a" * 64
H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
H4 = "4" * 64


def _all_primary_failures(state, cells):
    for index in range(30):
        state = next_state(
            state,
            cells=cells,
            authorization_sha256=AUTHORIZATION,
            cell_index=index,
            primary_stage="TRAINING",
        )
        state = next_state(
            state,
            cells=cells,
            authorization_sha256=AUTHORIZATION,
            cell_index=index,
            primary_stage="MODEL_FAILURE",
            proposal_seal_sha256=H2,
            primary_fragment_seal_sha256=H1,
            failure_stage="training_nonfinite",
        )
    return state


def _unlock_post(state, cells):
    state = next_state(
        state,
        cells=cells,
        authorization_sha256=AUTHORIZATION,
        campaign_phase="PRIMARY_SEALED",
        primary_seal_sha256=H2,
    )
    return next_state(
        state,
        cells=cells,
        authorization_sha256=AUTHORIZATION,
        campaign_phase="POST_PRIMARY_COLLECTION",
    )


def test_global_primary_barrier_and_complete_failure_path():
    cells = load_model_cells()
    state = initial_campaign_state(cells, authorization_sha256=AUTHORIZATION)
    with pytest.raises(ContractError):
        next_state(
            state,
            cells=cells,
            authorization_sha256=AUTHORIZATION,
            campaign_phase="PRIMARY_SEALED",
            primary_seal_sha256=H2,
        )
    state = _unlock_post(_all_primary_failures(state, cells), cells)
    assert all(row["post_primary_stage"] == "NOT_STARTED" for row in state["cells"])
    for index in range(30):
        state = next_state(
            state,
            cells=cells,
            authorization_sha256=AUTHORIZATION,
            cell_index=index,
            post_primary_stage="MODEL_FAILURE",
            truth_fragment_sha256=H3,
            secondary_fragment_sha256=H4,
            failure_stage="training_nonfinite",
        )
    state = next_state(
        state,
        cells=cells,
        authorization_sha256=AUTHORIZATION,
        campaign_phase="FINALIZED",
        final_validation_sha256="5" * 64,
    )
    assert validate_campaign_state(
        state, cells, authorization_sha256=AUTHORIZATION
    ) == state["state_sha256"]


def test_premature_hashes_and_stage_skip_are_rejected():
    cells = load_model_cells()
    initial = initial_campaign_state(cells, authorization_sha256=AUTHORIZATION)
    forged = deepcopy(initial)
    forged["cells"][0]["truth_fragment_sha256"] = H3
    forged = seal_state(forged)
    with pytest.raises(ContractError, match="before truth opening"):
        validate_campaign_state(forged, cells, authorization_sha256=AUTHORIZATION)
    with pytest.raises(ContractError, match="invalid primary-stage transition"):
        next_state(
            initial,
            cells=cells,
            authorization_sha256=AUTHORIZATION,
            cell_index=0,
            primary_stage="CHECKPOINT_SEALED",
            checkpoint_seal_sha256=H1,
        )


def test_actual_proposal_seal_hash_is_required_and_immutable() -> None:
    cells = load_model_cells()
    state = initial_campaign_state(cells, authorization_sha256=AUTHORIZATION)
    state = next_state(
        state,
        cells=cells,
        authorization_sha256=AUTHORIZATION,
        cell_index=0,
        primary_stage="TRAINING",
    )
    state = next_state(
        state,
        cells=cells,
        authorization_sha256=AUTHORIZATION,
        cell_index=0,
        primary_stage="CHECKPOINT_SEALED",
        checkpoint_seal_sha256=H1,
    )
    with pytest.raises(ContractError, match="actual proposal-seal"):
        next_state(
            state,
            cells=cells,
            authorization_sha256=AUTHORIZATION,
            cell_index=0,
            primary_stage="PROPOSAL_SEALED",
        )
    state = next_state(
        state,
        cells=cells,
        authorization_sha256=AUTHORIZATION,
        cell_index=0,
        primary_stage="PROPOSAL_SEALED",
        proposal_seal_sha256=H2,
    )
    with pytest.raises(ContractError, match="changed after sealing"):
        next_state(
            state,
            cells=cells,
            authorization_sha256=AUTHORIZATION,
            cell_index=0,
            primary_stage="PRIMARY_FRAGMENT_SEALED",
            proposal_seal_sha256=H3,
            primary_fragment_seal_sha256=H4,
        )


def test_primary_seal_cannot_change_during_cell_transition():
    cells = load_model_cells()
    state = _unlock_post(
        _all_primary_failures(
            initial_campaign_state(cells, authorization_sha256=AUTHORIZATION), cells
        ),
        cells,
    )
    current = next_state(
        state,
        cells=cells,
        authorization_sha256=AUTHORIZATION,
        cell_index=0,
        post_primary_stage="MODEL_FAILURE",
        truth_fragment_sha256=H3,
        secondary_fragment_sha256=H4,
        failure_stage="training_nonfinite",
    )
    forged = deepcopy(current)
    forged["primary_seal_sha256"] = "9" * 64
    forged = seal_state(forged)
    with pytest.raises(ContractError, match="primary seal changed"):
        validate_state_transition(
            state, forged, cells, authorization_sha256=AUTHORIZATION
        )


def test_state_history_is_exclusive_and_chain_verified(tmp_path: Path):
    cells = load_model_cells()
    initial = initial_campaign_state(cells, authorization_sha256=AUTHORIZATION)
    write_state_history(tmp_path, initial)
    with pytest.raises(ContractError, match="already exists"):
        write_state_history(tmp_path, initial)
    state = next_state(
        initial,
        cells=cells,
        authorization_sha256=AUTHORIZATION,
        cell_index=0,
        primary_stage="TRAINING",
    )
    write_state_history(tmp_path, state)
    verified = verify_state_history(
        tmp_path, cells, authorization_sha256=AUTHORIZATION
    )
    assert verified["state_sha256"] == state["state_sha256"]
    history = tmp_path / "state" / "000001.json"
    history.chmod(0o644)
    history.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ContractError):
        verify_state_history(tmp_path, cells, authorization_sha256=AUTHORIZATION)
