"""Regression tests for the serialized-Boolean defect in post-primary truth.

The v3 PACS campaign read its sealed primary decision CSV with ``read_csv``,
which yields ``dict[str, str]``.  ``post_decision_truth`` then evaluated
``bool(row["certified"])``; because ``bool("false") is True``, all 81 refused
H/S/B rows were treated as certified and their ``selected_candidate = -1`` was
used to index the candidate family, opening truth for refused cells.

The pre-existing suite passed ``summarize_primary_cell`` output directly, which
carries *native* booleans, so the defect was invisible.  Every test below
therefore drives values through the campaign's own CSV writer/reader pair
(``v3_orchestrator.write_csv`` / ``v3_orchestrator.read_csv``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fedcore.experiments.v3_campaign_finalizer import (
    VerifiedPrimarySeal,
    _validate_truth_fragments,
    truth_fragment_sha256,
)
from fedcore.experiments.v3_contract import (
    ALPHA,
    CLIENTS,
    EXPECTED_MODELS,
    PROCEDURES,
    PROTOCOL_ID,
    ContractError,
    load_model_cells,
    parse_serialized_bool,
)
from fedcore.experiments.v3_orchestrator import PRIMARY_FIELDS, read_csv, write_csv
from fedcore.experiments.v3_posttrain import LogitFrame, post_decision_truth
from fedcore.experiments.v3_proposal import ProposalMember

PRIMARY_HASH = "c" * 64
CANDIDATE_HASH = "b" * 64
SEAL_HASH = "d" * 64


# --------------------------------------------------------------------------
# 1-6: the strict parser itself
# --------------------------------------------------------------------------


def test_native_false_parses_as_false() -> None:
    assert parse_serialized_bool(False, "certified") is False


def test_native_true_parses_as_true() -> None:
    assert parse_serialized_bool(True, "certified") is True


def test_serialized_false_parses_as_false() -> None:
    """The exact defect: bool("false") is True, the parser must say False."""

    assert bool("false") is True
    assert parse_serialized_bool("false", "certified") is False


def test_serialized_true_parses_as_true() -> None:
    assert parse_serialized_bool("true", "certified") is True


@pytest.mark.parametrize(
    "value, expected",
    [
        ("False", False),
        ("TRUE", True),
        ("True", True),
        ("FALSE", False),
        (" false ", False),
        ("\ttrue\n", True),
        ("  True  ", True),
    ],
)
def test_mixed_case_and_whitespace_are_handled_deliberately(
    value: str, expected: bool
) -> None:
    assert parse_serialized_bool(value, "certified") is expected


@pytest.mark.parametrize(
    "value",
    ["", " ", "0", "1", "yes", "no", "t", "f", "none", "null", "truex", None, 0, 1, 1.0, [], {}],
)
def test_invalid_values_fail_closed(value: object) -> None:
    with pytest.raises(ContractError, match="is not a serialized boolean"):
        parse_serialized_bool(value, "certified")


def test_parser_names_the_offending_field() -> None:
    with pytest.raises(ContractError, match="model_evaluable is not a serialized boolean"):
        parse_serialized_bool("maybe", "model_evaluable")


# --------------------------------------------------------------------------
# fixtures: the campaign's real CSV path
# --------------------------------------------------------------------------


def _frame(client: str, n: int = 40, *, unknown: int = 4, offset: int = 0) -> LogitFrame:
    labels = np.arange(n, dtype=np.int64) % 4
    is_unknown = np.zeros(n, dtype=bool)
    if unknown:
        labels[-unknown:] = -1
        is_unknown[-unknown:] = True
    logits = np.full((n, 4), -4.0, dtype=np.float64)
    for i in range(n - unknown):
        logits[i, int(labels[i])] = 6.0
    if unknown:
        logits[-unknown:] = 0.0
    identities = tuple(f"{offset + i:064x}" for i in range(n))
    return LogitFrame(client, identities, logits, labels, is_unknown).validated()


def _frames() -> tuple[LogitFrame, ...]:
    return tuple(_frame(c, offset=i * 1000) for i, c in enumerate(CLIENTS))


def _member(index: int, *, threshold: float = -1e9) -> ProposalMember:
    """A member whose threshold accepts every sample, so truth can open."""

    return ProposalMember(
        candidate_index=index,
        score="msp",
        gamma=0.5,
        proposal_feasible=True,
        threshold=threshold,
        proposal_n=128,
        proposal_A=128,
        proposal_K=4,
        proposal_risk=0.03,
    )


class _TripwireMembers:
    """A members sequence that fails the test if the family is ever indexed."""

    def __init__(self, length: int = 12) -> None:
        self._length = length
        self.indexed: list[int] = []

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> ProposalMember:  # pragma: no cover
        self.indexed.append(index)
        raise AssertionError(f"candidate family was indexed with {index!r}")


def _primary_row(
    model_id: str,
    procedure: str,
    *,
    certified: bool,
    selected: int,
    refusal_reason: str,
    coverage_lcb: float = 0.0,
    model_evaluable: bool = True,
) -> dict[str, object]:
    """One primary procedure row carrying *native* booleans, as the writer sees it."""

    return {
        "protocol_id": PROTOCOL_ID,
        "model_id": model_id,
        "architecture": "resnet18",
        "split": 0,
        "nominal_training_seed": 0,
        "training_status": "terminal_success",
        "model_evaluable": model_evaluable,
        "failure_stage": "none",
        "checkpoint_sha256": "a" * 64,
        "proposal_seal_sha256": "e" * 64,
        "audit_sequence_sha256": "f" * 64,
        "count_tensor_sha256": "9" * 64,
        "candidate_decision_file_sha256": CANDIDATE_HASH,
        "procedure": procedure,
        "alpha": ALPHA,
        "delta_r": 0.05,
        "delta_c": 0.05,
        "mixture_set": "Delta^3",
        "budget_total": 512,
        "replicate_id": 0,
        "proposal_feasible_members": 12,
        "certified": certified,
        "selected_candidate": selected,
        "selected_score": "msp" if certified else "",
        "selected_gamma": 0.5 if certified else "",
        "selected_threshold": -1e9 if certified else "",
        "risk_decision": certified,
        "risk_ucb": 0.1 if certified else "",
        "raw_p_value": "",
        "holm_adjusted_p_value": "",
        "coverage_lcb": coverage_lcb,
        "ECA": 0.0,
        "refusal_reason": refusal_reason,
        "diagnostic_failure_subtype": "" if certified else "no_risk_pass",
    }


def _round_trip(tmp_path: Path, rows: list[dict[str, object]]) -> list[dict[str, str]]:
    """Serialize and re-read through the exact pair the campaign uses."""

    path = tmp_path / "PRIMARY_512_REP000.csv"
    write_csv(path, rows, PRIMARY_FIELDS)
    read_back = read_csv(path)
    assert all(isinstance(v, str) for row in read_back for v in row.values())
    return read_back


def _truth(rows: list[dict[str, str]], members: object) -> list[dict[str, object]]:
    return post_decision_truth(
        rows,
        members,
        _frames(),
        primary_decision_file_sha256=PRIMARY_HASH,
        primary_candidate_decision_file_sha256=CANDIDATE_HASH,
        primary_seal_verified=True,
    )


def _triplet(
    model_id: str, *, certified: bool, selected: int, refusal_reason: str, coverage_lcb: float = 0.0
) -> list[dict[str, object]]:
    return [
        _primary_row(
            model_id,
            procedure,
            certified=certified,
            selected=selected,
            refusal_reason=refusal_reason,
            coverage_lcb=coverage_lcb,
        )
        for procedure in PROCEDURES
    ]


# --------------------------------------------------------------------------
# 7-10: row-level behaviour through the CSV path
# --------------------------------------------------------------------------


def test_refused_row_with_minus_one_never_indexes_the_family(tmp_path: Path) -> None:
    """The exact v3 failure: certified="false", selected_candidate=-1."""

    rows = _round_trip(
        tmp_path,
        _triplet(
            "pacs__resnet18__split00__seed0",
            certified=False,
            selected=-1,
            refusal_reason="no_candidate_certified",
        ),
    )
    assert [row["certified"] for row in rows] == ["false", "false", "false"]
    tripwire = _TripwireMembers()
    truth = _truth(rows, tripwire)
    assert tripwire.indexed == []
    assert len(truth) == 3
    for row in truth:
        assert row["truth_unavailable_reason"] == "certificate_refusal"
        assert row["true_worst_client_risk"] is None
        assert row["true_min_client_acceptance"] is None
        assert row["validity_failure"] is None
        assert row["selected_candidate"] == -1


def test_proposal_infeasible_row_reports_its_own_reason(tmp_path: Path) -> None:
    rows = _round_trip(
        tmp_path,
        _triplet(
            "pacs__resnet18__split00__seed1",
            certified=False,
            selected=-1,
            refusal_reason="proposal_family_all_infeasible",
        ),
    )
    tripwire = _TripwireMembers()
    truth = _truth(rows, tripwire)
    assert tripwire.indexed == []
    assert all(row["truth_unavailable_reason"] == "proposal_infeasible" for row in truth)
    assert all(row["true_worst_client_risk"] is None for row in truth)


def test_certified_row_with_valid_candidate_opens_truth(tmp_path: Path) -> None:
    rows = _round_trip(
        tmp_path,
        _triplet(
            "pacs__convnext_tiny__split02__seed0",
            certified=True,
            selected=4,
            refusal_reason="none",
        ),
    )
    assert [row["certified"] for row in rows] == ["true", "true", "true"]
    members = tuple(_member(i) for i in range(12))
    truth = _truth(rows, members)
    assert len(truth) == 3
    for row in truth:
        assert row["truth_unavailable_reason"] == "none"
        assert isinstance(row["true_worst_client_risk"], float)
        assert isinstance(row["true_min_client_acceptance"], float)
        assert isinstance(row["validity_failure"], bool)
        assert row["selected_candidate"] == 4


def test_certified_row_with_minus_one_candidate_fails_closed(tmp_path: Path) -> None:
    """An inconsistent sealed row must stop the finalizer, not index members[-1]."""

    rows = _round_trip(
        tmp_path,
        _triplet(
            "pacs__resnet18__split00__seed2",
            certified=True,
            selected=-1,
            refusal_reason="none",
        ),
    )
    members = tuple(_member(i) for i in range(12))
    with pytest.raises(ContractError, match="certified row has an invalid selected candidate"):
        _truth(rows, members)


def test_certified_row_with_out_of_range_candidate_fails_closed(tmp_path: Path) -> None:
    rows = _round_trip(
        tmp_path,
        _triplet(
            "pacs__resnet18__split00__seed2",
            certified=True,
            selected=12,
            refusal_reason="none",
        ),
    )
    members = tuple(_member(i) for i in range(12))
    with pytest.raises(ContractError, match="certified row has an invalid selected candidate"):
        _truth(rows, members)


def test_corrupt_certified_token_fails_closed(tmp_path: Path) -> None:
    rows = _round_trip(
        tmp_path,
        _triplet(
            "pacs__resnet18__split00__seed0",
            certified=False,
            selected=-1,
            refusal_reason="no_candidate_certified",
        ),
    )
    for row in rows:
        row["certified"] = "maybe"
    with pytest.raises(ContractError, match="certified is not a serialized boolean"):
        _truth(rows, _TripwireMembers())


# --------------------------------------------------------------------------
# 11: CSV round trip preserves Boolean meaning
# --------------------------------------------------------------------------


def test_csv_round_trip_preserves_boolean_meaning(tmp_path: Path) -> None:
    native = _triplet(
        "pacs__resnet18__split00__seed0",
        certified=False,
        selected=-1,
        refusal_reason="no_candidate_certified",
    ) + _triplet(
        "pacs__convnext_tiny__split02__seed0",
        certified=True,
        selected=4,
        refusal_reason="none",
    )
    read_back = _round_trip(tmp_path, native)
    assert len(read_back) == len(native)
    for before, after in zip(native, read_back):
        assert isinstance(before["certified"], bool)
        assert isinstance(after["certified"], str)
        # the builtin conversion loses the distinction; the parser keeps it
        assert parse_serialized_bool(after["certified"], "certified") is before["certified"]
        assert parse_serialized_bool(
            after["model_evaluable"], "model_evaluable"
        ) is before["model_evaluable"]
    assert bool(read_back[0]["certified"]) is True  # the defect, still latent in builtins


# --------------------------------------------------------------------------
# 12: the full 90-row campaign shape, plus finalizer consistency
# --------------------------------------------------------------------------


def _ninety_row_fixture() -> list[dict[str, object]]:
    """27 fully-refused models (81 rows) + 3 fully-certified models (9 rows)."""

    models = load_model_cells()
    assert len(models) == EXPECTED_MODELS
    certified_ids = {
        "pacs__convnext_tiny__split02__seed0",
        "pacs__convnext_tiny__split02__seed1",
        "pacs__convnext_tiny__split02__seed2",
    }
    rows: list[dict[str, object]] = []
    for model in models:
        if model.model_id in certified_ids:
            rows.extend(
                _triplet(
                    model.model_id,
                    certified=True,
                    selected=4,
                    refusal_reason="none",
                    coverage_lcb=0.1,
                )
            )
        else:
            rows.extend(
                _triplet(
                    model.model_id,
                    certified=False,
                    selected=-1,
                    refusal_reason="no_candidate_certified",
                )
            )
    assert len(rows) == 90
    return rows


def test_ninety_row_fixture_reproduces_the_81_9_crosstabulation(tmp_path: Path) -> None:
    rows = _round_trip(tmp_path, _ninety_row_fixture())
    assert len(rows) == 90
    assert sum(row["certified"] == "false" for row in rows) == 81
    assert sum(row["certified"] == "true" for row in rows) == 9

    members = tuple(_member(i) for i in range(12))
    crosstab: dict[tuple[str, str], int] = {}
    by_model: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_model.setdefault(row["model_id"], []).append(row)

    for model_id, triplet in by_model.items():
        certified = triplet[0]["certified"] == "true"
        truth = _truth(triplet, members if certified else _TripwireMembers())
        for primary, observed in zip(triplet, truth):
            # the finalizer's registered expectation, read the correct way
            if primary["refusal_reason"] == "proposal_family_all_infeasible":
                expected = "proposal_infeasible"
            elif primary["certified"].lower() != "true":
                expected = "certificate_refusal"
            else:
                expected = "none"
            key = (expected, str(observed["truth_unavailable_reason"]))
            crosstab[key] = crosstab.get(key, 0) + 1

    # the repaired pipeline agrees on every row: no off-diagonal mass
    assert crosstab == {
        ("certificate_refusal", "certificate_refusal"): 81,
        ("none", "none"): 9,
    }
    assert sum(crosstab.values()) == 90


def test_ninety_row_fixture_passes_finalizer_consistency(tmp_path: Path) -> None:
    """The repaired truth rows satisfy the unchanged finalizer contract."""

    primary_rows = _round_trip(tmp_path, _ninety_row_fixture())
    members = tuple(_member(i) for i in range(12))
    models = load_model_cells()
    by_model: dict[str, list[dict[str, str]]] = {}
    for row in primary_rows:
        by_model.setdefault(row["model_id"], []).append(row)

    fragments = []
    for model in models:
        triplet = by_model[model.model_id]
        certified = triplet[0]["certified"] == "true"
        truth_rows = _truth(triplet, members if certified else _TripwireMembers())
        fragment = {
            "protocol_id": PROTOCOL_ID,
            "model_id": model.model_id,
            "primary_seal_sha256": SEAL_HASH,
            "rows": truth_rows,
        }
        fragment["fragment_sha256"] = truth_fragment_sha256(fragment)
        fragments.append(fragment)

    authority = VerifiedPrimarySeal(
        output_dir=tmp_path,
        primary_decision_file_sha256=PRIMARY_HASH,
        primary_candidate_decision_file_sha256=CANDIDATE_HASH,
        primary_seal_sha256=SEAL_HASH,
        artifact_hashes={},
    )
    validated = _validate_truth_fragments(fragments, models, authority, primary_rows)
    assert len(validated) == 90
    reasons = [row["truth_unavailable_reason"] for row in validated]
    assert reasons.count("certificate_refusal") == 81
    assert reasons.count("none") == 9


def test_unrepaired_conversion_would_have_failed_the_finalizer(tmp_path: Path) -> None:
    """Characterize the v3 breach: the builtin conversion disagrees on 81 rows."""

    rows = _round_trip(tmp_path, _ninety_row_fixture())
    disagreements = 0
    for row in rows:
        builtin = bool(row["certified"])
        strict = parse_serialized_bool(row["certified"], "certified")
        if builtin is not strict:
            disagreements += 1
    assert disagreements == 81
