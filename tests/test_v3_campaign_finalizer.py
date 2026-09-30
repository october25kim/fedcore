"""PACS-free campaign-global finalizer tests."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from fedcore.experiments.v3_campaign_finalizer import (
    VerifiedPrimarySeal,
    append_primary_fragment_bundle,
    primary_fragment_sha256,
    load_primary_fragment_bundle,
    seal_primary_campaign,
    secondary_fragment_sha256,
    validate_secondary_fragment,
    verify_primary_seal,
    write_primary_fragment_bundle,
    write_proposal_fragment_bundle,
)
from fedcore.experiments.v3_contract import (
    CLIENTS,
    PROCEDURES,
    PROTOCOL_ID,
    ContractError,
    canonical_json_sha256,
    candidate_family,
    load_model_cells,
)
from fedcore.experiments.v3_counts import model_failure_count_rows
from fedcore.experiments.v3_hsb import model_failure_candidate_decisions
from fedcore.experiments.v3_orchestrator import _generate_fragment, read_csv, read_json
from fedcore.experiments.v3_proposal import ProposalMember


def _primary_fragment_and_audit(ordinal, model):
    fragment = _generate_fragment(model, ordinal)
    fragment["protocol_id"] = PROTOCOL_ID
    fragment["model"] = {
        **asdict(model),
        "known_classes": list(model.known_classes),
        "unknown_classes": list(model.unknown_classes),
    }
    if fragment["training_status"] == "terminal_success":
        clients = [
            {
                "client": client,
                "reservoir_size": 300,
                "indices": list(range(256)),
            }
            for client in CLIENTS
        ]
        sequence_hash = canonical_json_sha256(
            {
                "split": model.split,
                "replicate_id": 0,
                "clients": [
                    {"client": row["client"], "indices": row["indices"]}
                    for row in clients
                ],
            }
        )
        for row in fragment["candidate_decisions"]:
            row["audit_sequence_sha256"] = sequence_hash
        audit = {
            "protocol_id": PROTOCOL_ID,
            "model_id": model.model_id,
            "status": "ok",
            "split": model.split,
            "replicate_id": 0,
            "clients": clients,
        }
    else:
        audit = {
            "protocol_id": PROTOCOL_ID,
            "model_id": model.model_id,
            "status": "model_failure",
            "split": model.split,
            "replicate_id": 0,
            "clients": [],
        }
    fragment["fragment_sha256"] = primary_fragment_sha256(fragment)
    return fragment, audit


def _primary_fragments(tmp_path: Path):
    fragments = []
    for ordinal, model in enumerate(load_model_cells()):
        fragment, audit = _primary_fragment_and_audit(ordinal, model)
        cell_dir = tmp_path / "bundles" / f"{ordinal:02d}_{model.model_id}"
        write_primary_fragment_bundle(cell_dir, fragment, audit)
        fragments.append(load_primary_fragment_bundle(cell_dir))
    return fragments


def _failure_members():
    return tuple(
        ProposalMember(
            candidate_index=item.candidate_index,
            score=item.score,
            gamma=item.gamma,
            proposal_feasible=False,
            threshold=None,
            proposal_n=0,
            proposal_A=0,
            proposal_K=0,
            proposal_risk=None,
        )
        for item in candidate_family()
    )


def test_primary_seal_requires_all_30_fragments_before_writing(tmp_path: Path) -> None:
    fragments = _primary_fragments(tmp_path)
    with pytest.raises(ContractError, match="all 30"):
        seal_primary_campaign(tmp_path, fragments[:-1])
    assert not (tmp_path / "PRIMARY_SEAL.json").exists()
    assert not (tmp_path / "PRIMARY_CANDIDATE_DECISIONS.csv.gz").exists()


def test_proposal_is_durable_and_immutable_before_primary_append(tmp_path: Path) -> None:
    model = load_model_cells()[0]
    fragment, audit = _primary_fragment_and_audit(0, model)
    bundle = tmp_path / "cell"

    proposal_seal = write_proposal_fragment_bundle(bundle, fragment)
    assert proposal_seal.name == "PROPOSAL_SEAL.json"
    assert {path.name for path in bundle.iterdir()} == {
        "PROPOSAL_FRAGMENT.json",
        "PROPOSAL_SEAL.json",
    }
    proposal_before = (bundle / "PROPOSAL_FRAGMENT.json").read_bytes()
    seal_before = proposal_seal.read_bytes()

    primary_seal = append_primary_fragment_bundle(bundle, fragment, audit)
    assert primary_seal.name == "PRIMARY_FRAGMENT_SEAL.json"
    assert (bundle / "PROPOSAL_FRAGMENT.json").read_bytes() == proposal_before
    assert proposal_seal.read_bytes() == seal_before
    assert load_primary_fragment_bundle(bundle)["fragment_sha256"] == fragment[
        "fragment_sha256"
    ]


def test_primary_append_requires_the_exact_sealed_proposal(tmp_path: Path) -> None:
    model = load_model_cells()[0]
    fragment, audit = _primary_fragment_and_audit(0, model)
    bundle = tmp_path / "cell"
    write_proposal_fragment_bundle(bundle, fragment)

    changed = json.loads(json.dumps(fragment))
    changed["members"][0]["proposal_A"] += 1
    changed["fragment_sha256"] = primary_fragment_sha256(changed)
    with pytest.raises(ContractError, match="differs from its sealed proposal"):
        append_primary_fragment_bundle(bundle, changed, audit)
    assert {path.name for path in bundle.iterdir()} == {
        "PROPOSAL_FRAGMENT.json",
        "PROPOSAL_SEAL.json",
    }


@pytest.mark.parametrize("mutation", ("split", "client_order", "index_range"))
def test_primary_append_rejects_audit_manifest_binding_drift(
    tmp_path: Path, mutation: str
) -> None:
    model = load_model_cells()[0]
    fragment, audit = _primary_fragment_and_audit(0, model)
    bundle = tmp_path / mutation
    write_proposal_fragment_bundle(bundle, fragment)
    changed = json.loads(json.dumps(audit))
    if mutation == "split":
        changed["split"] = (model.split + 1) % 5
        match = "split/model"
    elif mutation == "client_order":
        changed["clients"][0], changed["clients"][1] = (
            changed["clients"][1],
            changed["clients"][0],
        )
        match = "unique and in frozen order"
    else:
        changed["clients"][0]["reservoir_size"] = 255
        match = "in-range"
    with pytest.raises(ContractError, match=match):
        append_primary_fragment_bundle(bundle, fragment, changed)


def test_primary_seal_rejects_unsealed_mapping_and_bundle_tamper(tmp_path: Path) -> None:
    fragments = _primary_fragments(tmp_path)
    unsealed = [dict(row) for row in fragments]
    unsealed[0].pop("_bundle_receipt")
    with pytest.raises(ContractError, match="unsealed in-memory"):
        seal_primary_campaign(tmp_path / "unsealed-output", unsealed)

    bundle = Path(fragments[0]["_bundle_receipt"]["root"])
    audit = bundle / "AUDIT_DRAW_INDICES.json"
    audit.write_bytes(audit.read_bytes() + b"\n")
    with pytest.raises(ContractError, match="file-hash chain"):
        load_primary_fragment_bundle(bundle)


def test_primary_single_writer_materializes_global_hashes_then_authorizes(tmp_path: Path) -> None:
    authority = seal_primary_campaign(tmp_path, _primary_fragments(tmp_path))
    assert isinstance(authority, VerifiedPrimarySeal)
    assert len(read_csv(tmp_path / "PRIMARY_CANDIDATE_DECISIONS.csv.gz")) == 1080
    assert len(read_csv(tmp_path / "PRIMARY_512_REP000.csv")) == 90
    seal = read_json(tmp_path / "PRIMARY_SEAL.json")
    assert seal["status"] == "PRIMARY_SEALED"
    assert seal["truth_and_secondary_authorized"] is True
    assert seal["truth_opened"] is False
    assert seal["row_counts"] == {
        "models": 30,
        "proposal": 360,
        "counts": 1440,
        "candidates": 1080,
        "procedures": 90,
    }
    assert authority.primary_decision_file_sha256 == seal["primary_decision_file_sha256"]
    assert authority.primary_candidate_decision_file_sha256 == seal[
        "primary_candidate_decision_file_sha256"
    ]
    assert verify_primary_seal(tmp_path) == authority


def test_primary_seal_detects_post_seal_mutation(tmp_path: Path) -> None:
    seal_primary_campaign(tmp_path, _primary_fragments(tmp_path))
    path = tmp_path / "PRIMARY_512_REP000.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ContractError, match="changed after sealing"):
        verify_primary_seal(tmp_path)


def test_secondary_fragment_requires_all_299_registered_configs() -> None:
    model = load_model_cells()[0]
    members = _failure_members()
    counts = []
    candidates = []
    for budget in (256, 512, 1024):
        for replicate in range(100):
            if (budget, replicate) == (512, 0):
                continue
            counts.extend(
                {
                    **row,
                    "budget_total": budget,
                    "replicate_id": replicate,
                }
                for row in model_failure_count_rows(model.model_id)
            )
            candidates.extend(
                row.as_row()
                for row in model_failure_candidate_decisions(
                    model, members, budget_total=budget, replicate_id=replicate
                )
            )
    authority = VerifiedPrimarySeal(
        Path("/synthetic/no-io"), "a" * 64, "b" * 64, "c" * 64, {}
    )
    fragment = {
        "protocol_id": PROTOCOL_ID,
        "model_id": model.model_id,
        "primary_seal_sha256": authority.primary_seal_sha256,
        "counts": counts,
        "candidate_decisions": candidates,
    }
    fragment["fragment_sha256"] = secondary_fragment_sha256(fragment)
    observed_counts, observed_candidates = validate_secondary_fragment(
        fragment, model, authority
    )
    assert len(observed_counts) == 299 * 12 * len(CLIENTS)
    assert len(observed_candidates) == 299 * 12 * len(PROCEDURES)

    incomplete = json.loads(json.dumps(fragment))
    incomplete["counts"] = incomplete["counts"][:-48]
    incomplete["fragment_sha256"] = secondary_fragment_sha256(incomplete)
    with pytest.raises(ContractError, match="secondary count denominator"):
        validate_secondary_fragment(incomplete, model, authority)
