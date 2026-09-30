"""PACS-free tests for v3 post-training scientific primitives."""

from __future__ import annotations

import numpy as np
import pytest

from fedcore.experiments.v3_contract import CLIENTS, ContractError, load_model_cells
from fedcore.experiments.v3_posttrain import (
    LogitFrame,
    audit_index_plan,
    audit_sequence_sha256,
    draw_frames,
    infer_logit_frame,
    load_audit_seeds,
    post_decision_truth,
    primary_cell,
    secondary_cell,
    summarize_primary_cell,
)


def _frame(client: str, n: int, *, unknown: int = 0, offset: int = 0) -> LogitFrame:
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


def _reservoirs(n: int = 300) -> dict[str, LogitFrame]:
    return {
        client: _frame(client, n, unknown=12, offset=index * 1000)
        for index, client in enumerate(CLIENTS)
    }


def test_registered_audit_streams_are_deterministic_nested_and_shared() -> None:
    seeds = load_audit_seeds()
    assert len(seeds) == 2000
    sizes = {client: 300 for client in CLIENTS}
    first = audit_index_plan(0, 0, sizes, seeds=seeds)
    second = audit_index_plan(0, 0, sizes, seeds=seeds)
    assert all(np.array_equal(first[client], second[client]) for client in CLIENTS)
    assert first["art_painting"][:5].tolist() == [225, 106, 277, 207, 130]
    assert audit_sequence_sha256(0, 0, first) == audit_sequence_sha256(0, 0, second)
    reservoirs = _reservoirs()
    low = draw_frames(reservoirs, first, 64)
    primary = draw_frames(reservoirs, first, 128)
    high = draw_frames(reservoirs, first, 256)
    assert low[0].identities == primary[0].identities[:64]
    assert primary[0].identities == high[0].identities[:128]


def test_inference_preserves_order_and_restores_training_mode() -> None:
    torch = pytest.importorskip("torch")
    dataset = torch.utils.data.TensorDataset(
        torch.eye(4, dtype=torch.float32), torch.arange(4, dtype=torch.int64)
    )
    model = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(4))
    model.train()
    frame = infer_logit_frame(
        model,
        dataset,
        client="art_painting",
        identities=tuple(f"{i:064x}" for i in range(4)),
        batch_size=2,
    )
    assert model.training
    assert np.array_equal(np.argmax(frame.logits, axis=1), np.arange(4))


def test_primary_truth_ordering_and_secondary_denominators() -> None:
    model = load_model_cells()[0]
    proposal = tuple(_frame(client, 80, unknown=4, offset=i * 1000) for i, client in enumerate(CLIENTS))
    reservoirs = _reservoirs()
    seeds = load_audit_seeds()
    indices = audit_index_plan(model.split, 0, {c: 300 for c in CLIENTS}, seeds=seeds)
    primary_draws = draw_frames(reservoirs, indices, 128)
    sequence_hash = audit_sequence_sha256(model.split, 0, indices)
    primary = primary_cell(
        model,
        proposal,
        primary_draws,
        checkpoint_sha256="a" * 64,
        audit_sequence_hash=sequence_hash,
    )
    assert len(primary.members) == 12
    assert len(primary.count_rows) == 48
    assert len(primary.candidate_rows) == 36
    summary = summarize_primary_cell(
        model, primary, campaign_candidate_file_sha256="b" * 64
    )
    assert len(summary) == 3
    with pytest.raises(ContractError, match="forbidden before primary seal"):
        post_decision_truth(
            summary,
            primary.members,
            tuple(reservoirs[c] for c in CLIENTS),
            primary_decision_file_sha256="c" * 64,
            primary_candidate_decision_file_sha256="b" * 64,
            primary_seal_verified=False,
        )
    truth = post_decision_truth(
        summary,
        primary.members,
        tuple(reservoirs[c] for c in CLIENTS),
        primary_decision_file_sha256="c" * 64,
        primary_candidate_decision_file_sha256="b" * 64,
        primary_seal_verified=True,
    )
    assert len(truth) == 3
    assert all(row["truth_opened_after_primary_hash"] is True for row in truth)
    with pytest.raises(ContractError, match="forbidden before the primary seal"):
        secondary_cell(
            model,
            primary.members,
            reservoirs,
            checkpoint_sha256="a" * 64,
            proposal_seal_sha256=primary.proposal_manifest_sha256,
            primary_seal_verified=False,
            seeds=seeds,
        )
    counts, candidates = secondary_cell(
        model,
        primary.members,
        reservoirs,
        checkpoint_sha256="a" * 64,
        proposal_seal_sha256=primary.proposal_manifest_sha256,
        primary_seal_verified=True,
        seeds=seeds,
    )
    assert len(counts) == 299 * 48
    assert len(candidates) == 299 * 36
    assert not any(
        row["budget_total"] == 512 and row["replicate_id"] == 0 for row in counts
    )
