"""PACS-agnostic post-training primitives for the sealed v3 protocol.

The functions in this module operate on already ordered tensors.  They do not
open a PACS archive or manifest and do not select a checkpoint.  In particular,
``primary_cell`` accepts only the registered primary draws, while
``post_decision_truth`` refuses to run unless the caller supplies the hashes of
an already sealed primary decision file and candidate-decision file.

This module is intentionally not a campaign writer.  The official primary
files are campaign-global (30 models), so a separate single-writer finalizer is
still required before these primitives can be connected to scientific data.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fedcore.experiments.v3_contract import (
    ALPHA,
    CLIENTS,
    J,
    M,
    PRIMARY_BUDGET_TOTAL,
    PRIMARY_REPLICATE_ID,
    ContractError,
    ModelCell,
    canonical_json_sha256,
    default_protocol_dir,
    sha256_file,
)
from fedcore.experiments.v3_counts import AuditClient, count_audit_records
from fedcore.experiments.v3_hsb import (
    CandidateDecision,
    candidate_decisions,
    procedure_rows,
    validate_hsb_invariants,
)
from fedcore.experiments.v3_proposal import (
    ProposalMember,
    construct_proposal_family,
    oriented_scores,
    proposal_error_vector,
    proposal_manifest_sha256,
)


AUDIT_SEED_MANIFEST_SHA256 = (
    "747b5416b2b5f09d0fbcdb2a94cbbc39825a1d44e2f80677ac8b0a859f290029"
)
REGISTERED_BUDGETS = (256, 512, 1024)
REGISTERED_REPLICATES = tuple(range(100))
MAXIMUM_DRAWS_PER_CLIENT = 256


@dataclass(frozen=True)
class LogitFrame:
    """One identity-ordered proposal or audit frame."""

    client: str
    identities: tuple[str, ...]
    logits: np.ndarray
    labels: np.ndarray
    is_unknown: np.ndarray

    def validated(self) -> "LogitFrame":
        if self.client not in CLIENTS:
            raise ContractError(f"unregistered client {self.client!r}")
        logits = np.asarray(self.logits, dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.int64)
        unknown = np.asarray(self.is_unknown, dtype=bool)
        n = len(self.identities)
        if logits.shape != (n, 4):
            raise ContractError("logit frame must have shape (N,4)")
        if labels.shape != (n,) or unknown.shape != (n,):
            raise ContractError("label and unknown vectors must have shape (N,)")
        if n == 0 or len(set(self.identities)) != n:
            raise ContractError("logit frame identities must be nonempty and unique")
        if tuple(sorted(self.identities)) != self.identities:
            raise ContractError("logit frame must be source-identity sorted")
        if np.any(~np.isfinite(logits)):
            raise ContractError("logit frame contains a non-finite value")
        if np.any(unknown != (labels < 0)):
            raise ContractError("unknown markers must equal label<0")
        return LogitFrame(self.client, self.identities, logits, labels, unknown)

    def take(self, indices: np.ndarray) -> "LogitFrame":
        source = self.validated()
        index = np.asarray(indices, dtype=np.int64)
        if index.ndim != 1 or np.any(index < 0) or np.any(index >= len(source.identities)):
            raise ContractError("audit draw index is outside the frozen reservoir")
        return LogitFrame(
            client=source.client,
            identities=tuple(source.identities[int(i)] for i in index),
            logits=source.logits[index],
            labels=source.labels[index],
            is_unknown=source.is_unknown[index],
        )


@dataclass(frozen=True)
class AuditSeed:
    split: int
    client: str
    replicate_id: int
    seed_uint64: int


@dataclass(frozen=True)
class PrimaryCell:
    members: tuple[ProposalMember, ...]
    proposal_manifest_sha256: str
    audit_sequence_sha256: str
    count_rows: tuple[dict[str, object], ...]
    candidate_rows: tuple[CandidateDecision, ...]


def infer_logit_frame(
    model: Any,
    dataset: Any,
    *,
    client: str,
    identities: Sequence[str],
    device: str = "cpu",
    batch_size: int = 32,
) -> LogitFrame:
    """Run deterministic, order-preserving inference without opening data itself."""

    if batch_size <= 0 or len(dataset) != len(identities):
        raise ContractError("inference dataset/identity denominator mismatch")
    import torch

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    was_training = bool(model.training)
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for values, targets in loader:
                output = model(values.to(device))
                if output.ndim != 2 or output.shape[1] != 4:
                    raise ContractError("registered model must emit four known-class logits")
                logits.append(output.detach().cpu().to(torch.float64).numpy())
                labels.append(targets.detach().cpu().to(torch.int64).numpy())
    finally:
        model.train(was_training)
    y = np.concatenate(labels)
    return LogitFrame(
        client=client,
        identities=tuple(identities),
        logits=np.concatenate(logits),
        labels=y,
        is_unknown=y < 0,
    ).validated()


def load_audit_seeds(protocol_dir: Path | None = None) -> dict[tuple[int, str, int], AuditSeed]:
    root = default_protocol_dir() if protocol_dir is None else Path(protocol_dir)
    path = root / "AUDIT_SEED_MANIFEST.csv"
    if sha256_file(path) != AUDIT_SEED_MANIFEST_SHA256:
        raise ContractError("audit seed manifest hash drift")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_fields = (
        "split",
        "client",
        "replicate_id",
        "pcg64_seed_uint64",
        "maximum_prefix_draws",
        "primary_prefix_draws",
        "secondary_lower_prefix_draws",
        "secondary_upper_prefix_draws",
        "primary",
    )
    if tuple(rows[0]) != expected_fields or len(rows) != 5 * J * 100:
        raise ContractError("audit seed manifest schema or denominator drift")
    output: dict[tuple[int, str, int], AuditSeed] = {}
    for row in rows:
        split, client, replicate = int(row["split"]), row["client"], int(row["replicate_id"])
        key = (split, client, replicate)
        if key in output or split not in range(5) or client not in CLIENTS or replicate not in range(100):
            raise ContractError("invalid or duplicate audit seed key")
        if (
            int(row["maximum_prefix_draws"]) != 256
            or int(row["primary_prefix_draws"]) != 128
            or int(row["secondary_lower_prefix_draws"]) != 64
            or int(row["secondary_upper_prefix_draws"]) != 256
            or (row["primary"] == "true") != (replicate == 0)
        ):
            raise ContractError("audit prefix contract drift")
        output[key] = AuditSeed(split, client, replicate, int(row["pcg64_seed_uint64"]))
    return output


def audit_index_plan(
    split: int,
    replicate_id: int,
    reservoir_sizes: Mapping[str, int],
    *,
    seeds: Mapping[tuple[int, str, int], AuditSeed] | None = None,
) -> dict[str, np.ndarray]:
    """Generate the registered maximum PCG64 prefix for all four clients."""

    registered = load_audit_seeds() if seeds is None else seeds
    output: dict[str, np.ndarray] = {}
    for client in CLIENTS:
        size = int(reservoir_sizes.get(client, 0))
        if size <= 0:
            raise ContractError(f"empty audit reservoir for {client}")
        try:
            seed = registered[(split, client, replicate_id)].seed_uint64
        except KeyError as exc:
            raise ContractError("audit seed key is outside the registered grid") from exc
        rng = np.random.Generator(np.random.PCG64(seed))
        output[client] = rng.integers(
            0, size, size=MAXIMUM_DRAWS_PER_CLIENT, dtype=np.int64
        )
    return output


def audit_sequence_sha256(
    split: int, replicate_id: int, indices: Mapping[str, np.ndarray]
) -> str:
    """Hash the complete four-client maximum prefix (implementation provenance)."""

    return canonical_json_sha256(
        {
            "split": split,
            "replicate_id": replicate_id,
            "clients": [
                {"client": client, "indices": np.asarray(indices[client], dtype=np.int64).tolist()}
                for client in CLIENTS
            ],
        }
    )


def draw_frames(
    reservoirs: Mapping[str, LogitFrame],
    indices: Mapping[str, np.ndarray],
    draws_per_client: int,
) -> tuple[LogitFrame, ...]:
    if draws_per_client not in (64, 128, 256):
        raise ContractError("draw prefix must be one of 64, 128, or 256")
    return tuple(
        reservoirs[client].take(np.asarray(indices[client])[:draws_per_client])
        for client in CLIENTS
    )


def _audit_clients(frames: Sequence[LogitFrame]) -> tuple[AuditClient, ...]:
    if tuple(frame.client for frame in frames) != CLIENTS:
        raise ContractError(f"audit frames must be ordered as {CLIENTS!r}")
    return tuple(
        AuditClient(frame.client, frame.logits, frame.labels, frame.is_unknown)
        for frame in frames
    )


def proposal_members(frames: Sequence[LogitFrame]) -> tuple[ProposalMember, ...]:
    if tuple(frame.client for frame in frames) != CLIENTS:
        raise ContractError(f"proposal frames must be ordered as {CLIENTS!r}")
    checked = tuple(frame.validated() for frame in frames)
    return construct_proposal_family(
        np.concatenate([frame.logits for frame in checked]),
        np.concatenate([frame.labels for frame in checked]),
        np.concatenate([frame.is_unknown for frame in checked]),
    )


def primary_cell(
    model: ModelCell,
    proposal_frames: Sequence[LogitFrame],
    primary_draw_frames: Sequence[LogitFrame],
    *,
    checkpoint_sha256: str,
    audit_sequence_hash: str,
) -> PrimaryCell:
    """Compute one cell's primary fragment from proposal and 128-draw inputs."""

    if len(checkpoint_sha256) != 64 or len(audit_sequence_hash) != 64:
        raise ContractError("primary provenance hashes must be hex64 strings")
    members = proposal_members(proposal_frames)
    proposal_hash = proposal_manifest_sha256(members)
    counts = count_audit_records(model.model_id, members, _audit_clients(primary_draw_frames))
    if any(int(row["n"]) != 128 for row in counts):
        raise ContractError("primary audit must contain exactly 128 draws per client")
    decisions = candidate_decisions(
        model,
        members,
        counts,
        checkpoint_sha256=checkpoint_sha256,
        proposal_seal_sha256=proposal_hash,
        audit_sequence_sha256=audit_sequence_hash,
    )
    validate_hsb_invariants(decisions)
    return PrimaryCell(
        members=members,
        proposal_manifest_sha256=proposal_hash,
        audit_sequence_sha256=audit_sequence_hash,
        count_rows=tuple(counts),
        candidate_rows=tuple(decisions),
    )


def summarize_primary_cell(
    model: ModelCell,
    primary: PrimaryCell,
    *,
    campaign_candidate_file_sha256: str,
) -> list[dict[str, object]]:
    if len(campaign_candidate_file_sha256) != 64:
        raise ContractError("campaign candidate decision hash must be hex64")
    return procedure_rows(
        model,
        primary.candidate_rows,
        candidate_decision_file_sha256=campaign_candidate_file_sha256,
    )


def post_decision_truth(
    primary_rows: Sequence[Mapping[str, object]],
    members: Sequence[ProposalMember],
    full_audit_frames: Sequence[LogitFrame],
    *,
    primary_decision_file_sha256: str,
    primary_candidate_decision_file_sha256: str,
    primary_seal_verified: bool,
) -> list[dict[str, object]]:
    """Evaluate selected candidates only after the campaign primary seal exists."""

    if not primary_seal_verified:
        raise ContractError("full-frame truth is forbidden before primary seal verification")
    if len(primary_decision_file_sha256) != 64 or len(primary_candidate_decision_file_sha256) != 64:
        raise ContractError("truth must bind two campaign-global hex64 hashes")
    if len(primary_rows) != 3 or tuple(row["procedure"] for row in primary_rows) != ("H", "S", "B"):
        raise ContractError("truth requires one ordered H/S/B primary triplet")
    frames = tuple(frame.validated() for frame in full_audit_frames)
    if tuple(frame.client for frame in frames) != CLIENTS:
        raise ContractError("full audit frames are not in registered client order")
    output: list[dict[str, object]] = []
    for row in primary_rows:
        selected = int(row["selected_candidate"])
        certified = bool(row["certified"])
        if not certified:
            reason = (
                "proposal_infeasible"
                if row["refusal_reason"] == "proposal_family_all_infeasible"
                else "certificate_refusal"
            )
            true_risk = true_acceptance = validity_failure = None
        else:
            member = members[selected]
            risks: list[float] = []
            acceptances: list[float] = []
            for frame in frames:
                accepted = oriented_scores(frame.logits)[member.score] >= float(member.threshold)
                errors = proposal_error_vector(frame.logits, frame.labels, frame.is_unknown)
                a = int(accepted.sum())
                if a == 0:
                    raise ContractError("certified selector has zero full-frame client acceptance")
                risks.append(float(np.count_nonzero(accepted & errors) / a))
                acceptances.append(float(a / len(frame.identities)))
            true_risk = max(risks)
            true_acceptance = min(acceptances)
            validity_failure = bool(
                true_risk > ALPHA or float(row["coverage_lcb"]) > true_acceptance
            )
            reason = "none"
        output.append(
            {
                "model_id": row["model_id"],
                "procedure": row["procedure"],
                "primary_decision_file_sha256": primary_decision_file_sha256,
                "primary_candidate_decision_file_sha256": primary_candidate_decision_file_sha256,
                "truth_opened_after_primary_hash": True,
                "selected_candidate": selected,
                "true_worst_client_risk": true_risk,
                "true_min_client_acceptance": true_acceptance,
                "validity_failure": validity_failure,
                "truth_unavailable_reason": reason,
            }
        )
    return output


def secondary_cell(
    model: ModelCell,
    members: Sequence[ProposalMember],
    full_audit_frames: Mapping[str, LogitFrame],
    *,
    checkpoint_sha256: str,
    proposal_seal_sha256: str,
    primary_seal_verified: bool,
    seeds: Mapping[tuple[int, str, int], AuditSeed] | None = None,
) -> tuple[list[dict[str, object]], list[CandidateDecision]]:
    """Compute all 299 registered secondary count and candidate fragments."""

    if not primary_seal_verified:
        raise ContractError("secondary audits are forbidden before the primary seal")
    checked = {client: full_audit_frames[client].validated() for client in CLIENTS}
    sizes = {client: len(checked[client].identities) for client in CLIENTS}
    count_output: list[dict[str, object]] = []
    decision_output: list[CandidateDecision] = []
    for replicate_id in REGISTERED_REPLICATES:
        indices = audit_index_plan(model.split, replicate_id, sizes, seeds=seeds)
        sequence_hash = audit_sequence_sha256(model.split, replicate_id, indices)
        for budget_total in REGISTERED_BUDGETS:
            if (budget_total, replicate_id) == (PRIMARY_BUDGET_TOTAL, PRIMARY_REPLICATE_ID):
                continue
            frames = draw_frames(checked, indices, budget_total // J)
            counts = count_audit_records(model.model_id, members, _audit_clients(frames))
            for row in counts:
                count_output.append(
                    {**row, "budget_total": budget_total, "replicate_id": replicate_id}
                )
            decisions = candidate_decisions(
                model,
                members,
                counts,
                checkpoint_sha256=checkpoint_sha256,
                proposal_seal_sha256=proposal_seal_sha256,
                audit_sequence_sha256=sequence_hash,
                budget_total=budget_total,
                replicate_id=replicate_id,
            )
            validate_hsb_invariants(decisions)
            decision_output.extend(decisions)
    if len(count_output) != 299 * M * J or len(decision_output) != 299 * M * 3:
        raise ContractError("secondary per-cell denominator drift")
    return count_output, decision_output


__all__ = [
    "AUDIT_SEED_MANIFEST_SHA256",
    "AuditSeed",
    "LogitFrame",
    "PrimaryCell",
    "audit_index_plan",
    "audit_sequence_sha256",
    "draw_frames",
    "infer_logit_frame",
    "load_audit_seeds",
    "post_decision_truth",
    "primary_cell",
    "proposal_members",
    "secondary_cell",
    "summarize_primary_cell",
]
