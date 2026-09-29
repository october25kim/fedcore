"""Frozen execution contract for the FedCORE PACS H/S/B v3 replication.

This module contains no PACS data access and performs no training.  It is the
single implementation-side reader for the publicly sealed scientific contract.
Every executable v3 component imports the constants below instead of silently
re-declaring statistical budgets or family ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "FEDCORE-IJAR-V34-PACS-FRESH-SEED-REPLICATION-HSB-v3"
PROTOCOL_SHA256 = "d7fba4c748d449994b11912acd6004dfd690e5053825169e482eb7f64b2ee728"
BUNDLE_ROOT_SHA256 = "f81345a8e0ca70a7f64b76b3f01375bebff0a471e249710663e8bcbc82475feb"
PROTOCOL_DIRNAME = "ijar-v34-pacs-fresh-seed-hsb-v3"
CLIENTS = ("art_painting", "cartoon", "photo", "sketch")
PROCEDURES = ("H", "S", "B")
SCORES = ("msp", "energy_lse", "margin")
GAMMAS = (0.3, 0.5, 0.7, 1.0)
ALPHA = 0.20
DELTA_R = 0.05
DELTA_C = 0.05
M = 12
J = 4
PRIMARY_BUDGET_TOTAL = 512
PRIMARY_DRAWS_PER_CLIENT = 128
PRIMARY_REPLICATE_ID = 0
EXPECTED_MODELS = 30
EXPECTED_PROPOSAL_ROWS = EXPECTED_MODELS * M
EXPECTED_PRIMARY_COUNT_ROWS = EXPECTED_MODELS * M * J
EXPECTED_PRIMARY_CANDIDATE_ROWS = EXPECTED_MODELS * M * len(PROCEDURES)
EXPECTED_PRIMARY_PROCEDURE_ROWS = EXPECTED_MODELS * len(PROCEDURES)


class ContractError(RuntimeError):
    """Raised when implementation state diverges from the sealed contract."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_protocol_dir() -> Path:
    return repository_root() / "paper" / PROTOCOL_DIRNAME


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class Candidate:
    candidate_index: int
    score: str
    gamma: float


@dataclass(frozen=True)
class ModelCell:
    model_id: str
    architecture: str
    split: int
    nominal_training_seed: int
    model_init_seed_uint64: int
    known_classes: tuple[str, ...]
    unknown_classes: tuple[str, ...]


def candidate_family() -> tuple[Candidate, ...]:
    return tuple(
        Candidate(index, score, gamma)
        for index, (score, gamma) in enumerate(
            (score, gamma) for score in SCORES for gamma in GAMMAS
        )
    )


def load_model_cells(protocol_dir: Path | None = None) -> tuple[ModelCell, ...]:
    root = default_protocol_dir() if protocol_dir is None else Path(protocol_dir)
    rows = read_csv(root / "PACS_MODEL_MATRIX.csv")
    cells = tuple(
        ModelCell(
            model_id=row["model_id"],
            architecture=row["architecture"],
            split=int(row["split"]),
            nominal_training_seed=int(row["nominal_seed"]),
            model_init_seed_uint64=int(row["model_init_seed_uint64"]),
            known_classes=tuple(row["known_classes"].split(";")),
            unknown_classes=tuple(row["unknown_classes"].split(";")),
        )
        for row in rows
    )
    if len(cells) != EXPECTED_MODELS:
        raise ContractError(
            f"model denominator drift: expected {EXPECTED_MODELS}, observed {len(cells)}"
        )
    if len({cell.model_id for cell in cells}) != len(cells):
        raise ContractError("duplicate model_id in PACS_MODEL_MATRIX.csv")
    return cells


def _assert_close(actual: float, expected: float, label: str) -> None:
    if actual != expected:
        raise ContractError(f"{label} drift: expected {expected!r}, observed {actual!r}")


def verify_scientific_contract(protocol_dir: Path | None = None) -> dict[str, Any]:
    """Verify the fields the executable implementation is permitted to consume."""

    root = default_protocol_dir() if protocol_dir is None else Path(protocol_dir)
    plan = read_json(root / "H_S_B_ANALYSIS_PLAN.json")
    selector = read_json(root / "SELECTOR_FAMILY_CONTRACT.json")
    gate = read_json(root / "EXECUTION_GATE.json")
    if plan.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("analysis-plan protocol_id mismatch")
    if selector.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("selector protocol_id mismatch")
    if gate.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("execution-gate protocol_id mismatch")
    primary = plan["primary"]
    _assert_close(float(primary["alpha"]), ALPHA, "alpha")
    _assert_close(float(primary["delta_r"]), DELTA_R, "delta_r")
    _assert_close(float(primary["delta_c"]), DELTA_C, "delta_c")
    if int(primary["J"]) != J or int(primary["M"]) != M:
        raise ContractError("J/M drift")
    if int(primary["budget_total"]) != PRIMARY_BUDGET_TOTAL:
        raise ContractError("primary budget drift")
    if int(primary["draws_per_client"]) != PRIMARY_DRAWS_PER_CLIENT:
        raise ContractError("primary per-client draw drift")
    if tuple(primary["procedures"]) != PROCEDURES:
        raise ContractError("procedure ordering drift")
    expected_order = [
        {"candidate_index": item.candidate_index, "score": item.score, "gamma": item.gamma}
        for item in candidate_family()
    ]
    if selector.get("ordering") != expected_order:
        raise ContractError("candidate family ordering drift")
    if gate.get("training_authorized") is not False:
        raise ContractError("sealed scientific gate must keep training_authorized=false")
    load_model_cells(root)
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "bundle_root_sha256": BUNDLE_ROOT_SHA256,
        "models": EXPECTED_MODELS,
        "candidates": M,
        "clients": J,
        "procedures": list(PROCEDURES),
        "scientific_training_authorized": False,
    }


def assert_unique_keys(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in seen:
            raise ContractError(f"duplicate key {key!r} for fields {tuple(fields)!r}")
        seen.add(key)


__all__ = [
    "ALPHA",
    "BUNDLE_ROOT_SHA256",
    "CLIENTS",
    "ContractError",
    "DELTA_C",
    "DELTA_R",
    "EXPECTED_MODELS",
    "EXPECTED_PRIMARY_CANDIDATE_ROWS",
    "EXPECTED_PRIMARY_COUNT_ROWS",
    "EXPECTED_PRIMARY_PROCEDURE_ROWS",
    "EXPECTED_PROPOSAL_ROWS",
    "GAMMAS",
    "J",
    "M",
    "PRIMARY_BUDGET_TOTAL",
    "PRIMARY_DRAWS_PER_CLIENT",
    "PRIMARY_REPLICATE_ID",
    "PROCEDURES",
    "PROTOCOL_ID",
    "PROTOCOL_SHA256",
    "SCORES",
    "Candidate",
    "ModelCell",
    "assert_unique_keys",
    "candidate_family",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "default_protocol_dir",
    "load_model_cells",
    "read_csv",
    "read_json",
    "repository_root",
    "sha256_bytes",
    "sha256_file",
    "verify_scientific_contract",
]
