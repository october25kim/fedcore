"""Metadata-only Fed-ISIC2019 support audit and fold-plan selection.

No image, model, logit, prediction, AUROC, risk, coverage, or certification
outcome is read.  Primary rows are distinct patient/lesion audit units; repeated
images of a unit remain together in exactly one fold.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Iterable, Mapping, Sequence

import numpy as np

from fedcore.campaign.artifacts import file_sha256


FOLD_ROLES = ("train", "proposal", "certification", "test", "traffic")
CANDIDATE_PLANS: Mapping[str, tuple[float, ...]] = {
    "50_15_15_10_10": (0.50, 0.15, 0.15, 0.10, 0.10),
    "45_15_25_5_10": (0.45, 0.15, 0.25, 0.05, 0.10),
}


@dataclass(frozen=True)
class PreflightConfig:
    metadata_csv: str
    center_col: str
    diagnosis_col: str
    patient_col: str
    lesion_col: str
    image_col: str
    unit_col: str
    planning_acceptance_rate: float
    alpha: float
    delta: float
    seed: int = 0
    out_dir: str = "results/preflight"

    def validate(self) -> None:
        if self.unit_col not in {self.patient_col, self.lesion_col}:
            raise ValueError("unit_col must be the declared patient or lesion column")
        if not 0.0 < self.planning_acceptance_rate <= 1.0:
            raise ValueError("planning_acceptance_rate must lie in (0, 1]")
        if not 0.0 < self.alpha < 1.0 or not 0.0 < self.delta < 1.0:
            raise ValueError("alpha and delta must lie in (0, 1)")


def _read_metadata(config: PreflightConfig) -> list[dict[str, str]]:
    required = {
        config.center_col,
        config.diagnosis_col,
        config.patient_col,
        config.lesion_col,
        config.image_col,
        config.unit_col,
    }
    with open(config.metadata_csv, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"metadata is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("metadata is empty")
    for row_index, row in enumerate(rows, 2):
        for column in required:
            if not str(row[column]).strip():
                raise ValueError(f"blank {column!r} at CSV row {row_index}")
    images = [row[config.image_col] for row in rows]
    if len(images) != len(set(images)):
        raise ValueError("image IDs must be unique")
    return rows


def _unit_table(
    rows: Sequence[Mapping[str, str]], config: PreflightConfig
) -> dict[str, dict]:
    units: dict[str, dict] = {}
    for row in rows:
        unit = row[config.unit_col]
        record = units.setdefault(
            unit,
            {
                "unit_id": unit,
                "center": row[config.center_col],
                "diagnosis": row[config.diagnosis_col],
                "patient_id": row[config.patient_col],
                "lesion_ids": set(),
                "image_ids": [],
            },
        )
        if record["center"] != row[config.center_col]:
            raise ValueError(f"unit {unit!r} spans multiple centers")
        if record["diagnosis"] != row[config.diagnosis_col]:
            raise ValueError(f"unit {unit!r} spans multiple diagnoses")
        if (
            config.unit_col == config.lesion_col
            and record["patient_id"] != row[config.patient_col]
        ):
            raise ValueError(f"lesion {unit!r} maps to multiple patients")
        record["lesion_ids"].add(row[config.lesion_col])
        record["image_ids"].append(row[config.image_col])
    return units


def _largest_remainder_counts(n: int, fractions: Sequence[float]) -> np.ndarray:
    raw = np.asarray(fractions, dtype=float) * n
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remainder]] += 1
    assert int(counts.sum()) == n
    return counts


def _stable_order(unit_ids: Iterable[str], seed: int, stratum: str) -> list[str]:
    def key(unit: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}|{stratum}|{unit}".encode("utf-8")).hexdigest()
        return digest, unit

    return sorted(unit_ids, key=key)


def assign_folds(
    units: Mapping[str, Mapping], fractions: Sequence[float], seed: int
) -> dict[str, str]:
    """Stratified, deterministic, unit-level fold assignment."""
    if len(fractions) != len(FOLD_ROLES) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fold fractions must align with roles and sum to one")
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for unit, record in units.items():
        strata[(record["center"], record["diagnosis"])].append(unit)
    assignment: dict[str, str] = {}
    for (center, diagnosis), members in sorted(strata.items()):
        ordered = _stable_order(members, seed, f"{center}|{diagnosis}")
        counts = _largest_remainder_counts(len(ordered), fractions)
        cursor = 0
        for role, count in zip(FOLD_ROLES, counts):
            for unit in ordered[cursor : cursor + int(count)]:
                assignment[unit] = role
            cursor += int(count)
    if set(assignment) != set(units):
        raise RuntimeError("fold assignment did not cover every unit exactly once")
    return assignment


def _required_accepted_count(n_strata: int, alpha: float, delta: float) -> int:
    return int(math.ceil(math.log(n_strata / delta) / (-math.log1p(-alpha))))


def _support_rows(
    plan: str,
    units: Mapping[str, Mapping],
    assignment: Mapping[str, str],
    config: PreflightConfig,
) -> list[dict]:
    centers = sorted({record["center"] for record in units.values()})
    required = _required_accepted_count(len(centers), config.alpha, config.delta)
    rows = []
    for level, key_fn in (
        ("overall", lambda record: "ALL"),
        ("center", lambda record: record["center"]),
        ("diagnosis", lambda record: record["diagnosis"]),
        (
            "center_x_diagnosis",
            lambda record: f"{record['center']}|{record['diagnosis']}",
        ),
    ):
        keys = sorted({key_fn(record) for record in units.values()})
        for key in keys:
            for fold in FOLD_ROLES:
                selected = [
                    record
                    for unit, record in units.items()
                    if key_fn(record) == key and assignment[unit] == fold
                ]
                n_units = len(selected)
                expected = int(math.floor(n_units * config.planning_acceptance_rate))
                rows.append(
                    {
                        "plan_id": plan,
                        "support_level": level,
                        "support_key": key,
                        "fold": fold,
                        "n_units": n_units,
                        "n_patients": len({r["patient_id"] for r in selected}),
                        "n_lesions": (
                            len(set().union(*(r["lesion_ids"] for r in selected)))
                            if selected
                            else 0
                        ),
                        "n_images": sum(len(r["image_ids"]) for r in selected),
                        "planning_acceptance_rate": config.planning_acceptance_rate,
                        "expected_accepted_units": expected,
                        "required_accepted_units": required,
                        "accepted_count_margin": expected - required,
                        "planning_feasible": expected >= required,
                    }
                )
    return rows


def _plan_score(rows: Sequence[Mapping]) -> tuple[int, int, int]:
    # Primary planning constraint is per-center certification support. Diagnosis
    # cells are reported but do not masquerade as independent client strata.
    primary = [
        row
        for row in rows
        if row["support_level"] == "center" and row["fold"] == "certification"
    ]
    return (
        int(all(row["planning_feasible"] for row in primary)),
        min(int(row["accepted_count_margin"]) for row in primary),
        min(int(row["n_units"]) for row in primary),
    )


def _write_csv(path: str, fields: Sequence[str], rows: Sequence[Mapping]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def run_preflight(config: PreflightConfig) -> dict:
    config.validate()
    rows = _read_metadata(config)
    units = _unit_table(rows, config)
    all_support = []
    assignments = {}
    plan_scores = {}
    for plan, fractions in CANDIDATE_PLANS.items():
        assignment = assign_folds(units, fractions, config.seed)
        support = _support_rows(plan, units, assignment, config)
        assignments[plan] = assignment
        all_support.extend(support)
        plan_scores[plan] = _plan_score(support)
    selected = max(sorted(CANDIDATE_PLANS), key=lambda plan: plan_scores[plan])
    selected_assignment = assignments[selected]

    os.makedirs(config.out_dir, exist_ok=True)
    support_path = os.path.join(config.out_dir, "fed_isic_support.csv")
    support_fields = list(all_support[0])
    _write_csv(support_path, support_fields, all_support)

    fold_rows = []
    for unit in sorted(units):
        record = units[unit]
        fold_rows.append(
            {
                "audit_unit_id": unit,
                "fold": selected_assignment[unit],
                "center": record["center"],
                "diagnosis": record["diagnosis"],
                "patient_id": record["patient_id"],
                "lesion_ids_json": json.dumps(
                    sorted(record["lesion_ids"]), separators=(",", ":")
                ),
                "image_ids_json": json.dumps(
                    sorted(record["image_ids"]), separators=(",", ":")
                ),
            }
        )
    folds_path = os.path.join(config.out_dir, "fed_isic_folds.csv")
    _write_csv(folds_path, list(fold_rows[0]), fold_rows)
    fold_hash = file_sha256(folds_path)
    metadata_hash = file_sha256(config.metadata_csv)

    decision_path = os.path.join(config.out_dir, "fed_isic_fold_decision.md")
    decision = (
        "# Fed-ISIC2019 Metadata-Only Fold Decision\n\n"
        f"- Selected plan: `{selected}` ({'/'.join(str(int(100*x)) for x in CANDIDATE_PLANS[selected])})\n"
        f"- Fold roles: `{', '.join(FOLD_ROLES)}`\n"
        f"- Primary audit unit: `{config.unit_col}`\n"
        f"- Distinct units: {len(units)}; images: {len(rows)}\n"
        f"- Planning acceptance rate (predeclared): {config.planning_acceptance_rate}\n"
        f"- Theorem-2 planning target: alpha={config.alpha}, delta={config.delta}\n"
        f"- Candidate scores `(all-centers-feasible, min accepted margin, min cert units)`: "
        f"`{json.dumps(plan_scores, sort_keys=True)}`\n"
        f"- Metadata SHA-256: `{metadata_hash}`\n"
        f"- Frozen fold SHA-256: `{fold_hash}`\n\n"
        "Selection used support counts only. No image pixels, predictions, labels beyond "
        "the declared diagnosis metadata, AUROC, risk, coverage, or certificate outcome were read.\n"
    )
    tmp = f"{decision_path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(decision)
    os.replace(tmp, decision_path)
    return {
        "selected_plan": selected,
        "plan_scores": plan_scores,
        "metadata_sha256": metadata_hash,
        "fold_sha256": fold_hash,
        "support_path": support_path,
        "folds_path": folds_path,
        "decision_path": decision_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("metadata_csv")
    ap.add_argument("--center-col", required=True)
    ap.add_argument("--diagnosis-col", required=True)
    ap.add_argument("--patient-col", required=True)
    ap.add_argument("--lesion-col", required=True)
    ap.add_argument("--image-col", required=True)
    ap.add_argument("--unit-col", required=True)
    ap.add_argument("--planning-acceptance-rate", type=float, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/preflight")
    args = ap.parse_args()
    result = run_preflight(
        PreflightConfig(
            metadata_csv=args.metadata_csv,
            center_col=args.center_col,
            diagnosis_col=args.diagnosis_col,
            patient_col=args.patient_col,
            lesion_col=args.lesion_col,
            image_col=args.image_col,
            unit_col=args.unit_col,
            planning_acceptance_rate=args.planning_acceptance_rate,
            alpha=args.alpha,
            delta=args.delta,
            seed=args.seed,
            out_dir=args.out_dir,
        )
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
