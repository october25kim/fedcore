"""Authoritative one-shot plan validation and training-cell expansion.

The plan is deliberately external data.  This module refuses to guess the five
CIFAR class splits, eight diagnoses, seed roots, or post-hoc grids when the
governing run/seed document is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from fedcore.campaign.artifacts import (
    canonical_json,
    semantic_experiment_id,
    semantic_hash,
)
from fedcore.certificate.variants import SUPPORTED_CERTIFICATE_VARIANTS


POSTHOC_ONLY_KEYS = frozenset(
    {
        "alpha",
        "delta",
        "total_delta",
        "rho",
        "threshold_policy",
        "allocation_policy",
        "score",
        "score_name",
        "certificate_variant",
        "audit_budget_fraction",
        "traffic_sample_size",
        "solver",
    }
)

TERMINAL_STATUSES = frozenset({"succeeded", "failed", "blocked", "infeasible"})


@dataclass(frozen=True)
class TrainingCell:
    experiment_id: str
    family: str
    config: Mapping[str, Any]
    config_hash: str


def training_cell_experiment_id(config: Mapping[str, Any]) -> str:
    """Return the single canonical ID convention used by plans and runners."""

    family = config.get("family")
    if family == "cifar":
        required = ("split_id", "model_seed", "dirichlet_alpha")
        if any(key not in config for key in required):
            raise ValueError("CIFAR plan cell lacks split/model/d identity")
        prefix = (
            f"cifar-{config['split_id']}-m{int(config['model_seed'])}-"
            f"d{float(config['dirichlet_alpha']):g}"
        )
    elif family == "fed_isic2019":
        required = ("heldout_diagnosis", "model_seed")
        if any(key not in config for key in required):
            raise ValueError("medical plan cell lacks diagnosis/model identity")
        prefix = f"fed-isic-{config['heldout_diagnosis']}-m{int(config['model_seed'])}"
    else:
        raise ValueError("plan cell family must be cifar or fed_isic2019")
    return semantic_experiment_id(prefix, config)


def load_training_cell_config(path: str) -> dict[str, Any]:
    """Load one exact flat ``TrainingCell.config`` JSON object."""

    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("plan-cell config must be a JSON object")
    # Strict canonicalization rejects NaN/Infinity and unsupported values.
    canonical_json(value)
    training_cell_experiment_id(value)
    return value


def validate_training_cell_binding(
    config: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Fail if runner selectors disagree with their authoritative plan cell."""

    missing = sorted(set(expected) - set(config))
    if missing:
        raise ValueError(f"plan-cell config is missing selectors: {missing}")
    mismatched = [
        key
        for key, value in expected.items()
        if canonical_json(config[key]) != canonical_json(value)
    ]
    if mismatched:
        raise ValueError(
            f"runner arguments disagree with plan-cell config: {mismatched}"
        )


def _require_exact_sequence(value: Any, expected: Sequence[Any], name: str) -> tuple:
    if not isinstance(value, list) or tuple(value) != tuple(expected):
        raise ValueError(f"{name} must be exactly {list(expected)!r}")
    return tuple(value)


def _assert_no_posthoc(config: Mapping[str, Any]) -> None:
    overlap = POSTHOC_ONLY_KEYS & set(config)
    if overlap:
        raise ValueError(
            f"training config contains post-hoc-only keys: {sorted(overlap)}"
        )


def validate_plan(plan: Mapping[str, Any]) -> None:
    for key in ("schema_version", "campaign_seed", "cifar", "medical", "posthoc"):
        if key not in plan:
            raise ValueError(f"plan is missing {key!r}")
    if plan["schema_version"] != 1:
        raise ValueError("unsupported plan schema_version")
    if (
        isinstance(plan["campaign_seed"], bool)
        or not isinstance(plan["campaign_seed"], int)
        or plan["campaign_seed"] < 0
    ):
        raise ValueError("campaign_seed must be a non-negative integer")

    cifar = plan["cifar"]
    splits = cifar.get("class_splits")
    if not isinstance(splits, list) or len(splits) != 5:
        raise ValueError(
            "cifar.class_splits must contain exactly five predeclared splits"
        )
    split_ids = []
    for split in splits:
        if set(split) != {"id", "unknown_classes"}:
            raise ValueError(
                "each CIFAR split must contain only id and unknown_classes"
            )
        if not isinstance(split["id"], str) or not split["id"]:
            raise ValueError("CIFAR split id must be a non-empty string")
        unknown = split["unknown_classes"]
        if (
            not isinstance(unknown, list)
            or len(unknown) == 0
            or len(set(unknown)) != len(unknown)
        ):
            raise ValueError("CIFAR unknown_classes must be a non-empty unique list")
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in unknown
        ):
            raise ValueError("CIFAR unknown_classes must contain integer class IDs")
        if unknown != sorted(unknown):
            raise ValueError("CIFAR unknown_classes must be sorted canonically")
        split_ids.append(split["id"])
    if len(set(split_ids)) != 5:
        raise ValueError("CIFAR split IDs must be unique")
    _require_exact_sequence(cifar.get("model_seeds"), [0, 1, 2], "cifar.model_seeds")
    _require_exact_sequence(
        cifar.get("dirichlet_alpha"), [0.1, 0.5, 5.0], "cifar.dirichlet_alpha"
    )

    medical = plan["medical"]
    diagnoses = medical.get("heldout_diagnoses")
    if (
        not isinstance(diagnoses, list)
        or len(diagnoses) != 8
        or len(set(diagnoses)) != 8
    ):
        raise ValueError(
            "medical.heldout_diagnoses must list exactly eight unique diagnoses"
        )
    _require_exact_sequence(
        medical.get("model_seeds"), [0, 1, 2], "medical.model_seeds"
    )

    posthoc = plan["posthoc"]
    required_grids = {
        "alpha",
        "total_delta",
        "rho",
        "threshold_policies",
        "allocation_policies",
        "audit_budget_fractions",
        "traffic_sample_sizes",
        "scores",
        "certificate_variants",
        "gammas",
    }
    missing = required_grids - set(posthoc)
    if missing:
        raise ValueError(f"posthoc plan is missing grids: {sorted(missing)}")
    for key in required_grids:
        if not isinstance(posthoc[key], list) or not posthoc[key]:
            raise ValueError(f"posthoc.{key} must be a non-empty list")
    if set(posthoc["threshold_policies"]) != {"global", "client_specific"}:
        raise ValueError(
            "threshold policy grid must contain global and client_specific"
        )
    if set(posthoc["allocation_policies"]) != {"uniform", "proposal_informed"}:
        raise ValueError(
            "allocation policy grid must contain uniform and proposal_informed"
        )
    if set(posthoc["gammas"]) - {0.5, 0.7, 1.0}:
        raise ValueError("every gamma must be one of {0.5, 0.7, 1.0}")
    variants = posthoc["certificate_variants"]
    if any(not isinstance(variant, str) or not variant for variant in variants):
        raise ValueError("posthoc.certificate_variants must contain non-empty names")
    if len(set(variants)) != len(variants):
        raise ValueError("posthoc.certificate_variants must be unique")
    unsupported = sorted(set(variants) - set(SUPPORTED_CERTIFICATE_VARIANTS))
    if unsupported:
        raise ValueError(
            "posthoc plan names unimplemented certificate variants: "
            f"{unsupported}; implemented={sorted(SUPPORTED_CERTIFICATE_VARIANTS)}"
        )


def load_plan(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        plan = json.load(handle)
    validate_plan(plan)
    return plan


def expand_training_cells(plan: Mapping[str, Any]) -> list[TrainingCell]:
    validate_plan(plan)
    cells: list[TrainingCell] = []
    cifar = plan["cifar"]
    shared_cifar = dict(cifar.get("training", {}))
    _assert_no_posthoc(shared_cifar)
    for split in cifar["class_splits"]:
        for model_seed in cifar["model_seeds"]:
            for d in cifar["dirichlet_alpha"]:
                config = {
                    **shared_cifar,
                    "family": "cifar",
                    "split_id": split["id"],
                    "unknown_classes": list(split["unknown_classes"]),
                    "model_seed": int(model_seed),
                    "dirichlet_alpha": float(d),
                    "campaign_seed": int(plan["campaign_seed"]),
                }
                cells.append(
                    TrainingCell(
                        training_cell_experiment_id(config),
                        "cifar",
                        config,
                        semantic_hash(config),
                    )
                )
    medical = plan["medical"]
    shared_medical = dict(medical.get("training", {}))
    _assert_no_posthoc(shared_medical)
    for diagnosis in medical["heldout_diagnoses"]:
        for model_seed in medical["model_seeds"]:
            config = {
                **shared_medical,
                "family": "fed_isic2019",
                "heldout_diagnosis": diagnosis,
                "model_seed": int(model_seed),
                "campaign_seed": int(plan["campaign_seed"]),
            }
            cells.append(
                TrainingCell(
                    training_cell_experiment_id(config),
                    "fed_isic2019",
                    config,
                    semantic_hash(config),
                )
            )
    if sum(cell.family == "cifar" for cell in cells) != 45:
        raise AssertionError("CIFAR matrix expansion is not 45 cells")
    if sum(cell.family == "fed_isic2019" for cell in cells) != 24:
        raise AssertionError("medical matrix expansion is not 24 cells")
    return cells


def validate_terminal_coverage(
    cells: Sequence[TrainingCell], statuses: Mapping[str, str]
) -> None:
    expected = {cell.experiment_id for cell in cells}
    unknown = set(statuses) - expected
    missing = expected - set(statuses)
    nonterminal = {
        key: value
        for key, value in statuses.items()
        if key in expected and value not in TERMINAL_STATUSES
    }
    if unknown or missing or nonterminal:
        raise RuntimeError(
            f"matrix coverage failure: missing={len(missing)}, unknown={len(unknown)}, "
            f"nonterminal={len(nonterminal)}"
        )
