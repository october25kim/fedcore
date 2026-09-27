"""Conservative final validation and aggregation for a Fed-CORE campaign.

The finalizer consumes two authoritative inputs:

* the frozen one-shot plan accepted by :mod:`fedcore.campaign.plan`;
* a plan-bound run-record bundle whose paths and SHA-256 digests identify every
  training artifact, terminal manifest, and post-hoc result artifact.

It never invents a missing run or result row.  Coverage problems are fatal under
``complete`` policy.  ``allow-blocked`` permits explicitly recorded terminal
failures/blockers, while ``allow-incomplete`` additionally permits missing or
nonterminal records and partial post-hoc grids.  Integrity failures (bad hashes,
overlapping identities, invalid accounting, duplicate/unknown cells, or malformed
metrics) are fatal under every policy.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from fedcore.campaign.artifacts import canonical_json, file_sha256, semantic_hash
from fedcore.campaign.plan import TrainingCell, expand_training_cells, load_plan


CANONICAL_METRIC_FIELDS = (
    "certified",
    "risk_output_type",
    "risk_pass",
    "cert_risk_ucb",
    "cert_coverage_lcb",
    "iut_raw_pvalue",
    "holm_adjusted_pvalue",
    "holm_rank",
    "cert_n",
    "cert_k",
    "prop_coverage",
    "prop_risk",
    "test_coverage",
    "test_risk",
    "score_name",
    "gamma",
    "alpha",
    "delta",
    "delta_r",
    "delta_c",
    "family_procedure",
    "Lambda",
    "dirichlet_alpha",
    "n_clients",
)

GRID_FIELDS = (
    "alpha",
    "delta",
    "rho",
    "threshold_policy",
    "allocation_policy",
    "audit_budget_fraction",
    "traffic_sample_size",
    "score_name",
    "certificate_variant",
    "gamma",
)

ACCOUNTING_FIELDS = (
    "delta_total",
    "delta_mixture",
    "delta_conditional_risk",
    "delta_acceptance_lower",
    "delta_acceptance_upper",
    "delta_spent",
    "delta_slack",
)

PROVENANCE_FIELDS = (
    "source_experiment_id",
    "input_artifact_sha256",
    "training_config_sha256",
    "fold_sha256",
    "posthoc_config_sha256",
    "cert_sample_ids_sha256",
    "traffic_sample_ids_sha256",
    "cert_observation_count",
)

OUTPUT_LEADING_FIELDS = (
    "source_experiment_id",
    "family",
    "run_status",
    *CANONICAL_METRIC_FIELDS,
    "threshold_policy",
    "allocation_policy",
    "certificate_variant",
    "rho",
    "audit_budget_fraction",
    "traffic_sample_size",
    *ACCOUNTING_FIELDS,
    "input_artifact_sha256",
    "plan_cell_config_sha256",
    "training_config_sha256",
    "fold_sha256",
    "posthoc_config_sha256",
    "cert_sample_ids_sha256",
    "traffic_sample_ids_sha256",
    "cert_observation_count",
)

RECORD_STATUSES = frozenset(
    {
        "succeeded",
        "failed",
        "blocked",
        "infeasible",
        "pending",
        "running",
        "queued",
        "skipped",
        "not_attempted",
    }
)
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "blocked", "infeasible"})
NONTERMINAL_STATUSES = RECORD_STATUSES - TERMINAL_STATUSES
POLICIES = frozenset({"complete", "allow-blocked", "allow-incomplete"})
_HEX = frozenset("0123456789abcdef")

_PLAN_TO_ROW_GRID = (
    ("alpha", "alpha"),
    ("total_delta", "delta"),
    ("rho", "rho"),
    ("threshold_policies", "threshold_policy"),
    ("allocation_policies", "allocation_policy"),
    ("audit_budget_fractions", "audit_budget_fraction"),
    ("traffic_sample_sizes", "traffic_sample_size"),
    ("scores", "score_name"),
    ("certificate_variants", "certificate_variant"),
    ("gammas", "gamma"),
)


class CampaignValidationError(RuntimeError):
    """The campaign cannot be aggregated without violating provenance."""


@dataclass(frozen=True)
class TrainingArtifactInfo:
    path: str
    sha256: str
    experiment_id: str
    plan_cell_config_hash: str
    training_config_hash: str
    fold_hash: str
    cert_sample_ids_sha256: str
    n_clients: int


@dataclass(frozen=True)
class CampaignCollection:
    rows: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _require_hash(value: Any, name: str) -> str:
    if not _is_hash(value):
        raise CampaignValidationError(f"{name} must be a lowercase SHA-256 digest")
    return str(value)


def _strict_json_check(value: Any, path: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CampaignValidationError(f"{path} contains a non-finite JSON number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _strict_json_check(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CampaignValidationError(f"{path} contains a non-string key")
            _strict_json_check(item, f"{path}.{key}")
        return
    raise CampaignValidationError(
        f"{path} contains unsupported type {type(value).__name__}"
    )


def _read_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(
                handle,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    CampaignValidationError(
                        f"non-standard JSON constant {token!r} in {path}"
                    )
                ),
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignValidationError(f"could not read JSON {path}: {exc}") from exc
    _strict_json_check(value, path)
    return value


def _scalar(archive: Mapping[str, np.ndarray], key: str) -> Any:
    if key not in archive:
        raise CampaignValidationError(f"training artifact is missing {key!r}")
    value = np.asarray(archive[key])
    if value.size != 1:
        raise CampaignValidationError(f"training artifact scalar {key!r} is not scalar")
    return value.reshape(()).item()


def _integer_vector(value: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind == "b":
        raise CampaignValidationError(f"{name} is not a one-dimensional integer vector")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise CampaignValidationError(f"{name} is not numeric") from exc
    if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise CampaignValidationError(f"{name} contains non-integer values")
    return numeric.astype(np.int64)


def _stable_ids(value: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"U", "S"}:
        raise CampaignValidationError(f"{name} is not a persisted stable string vector")
    ids = raw.astype(str)
    if np.any(np.char.str_len(ids) == 0) or len(set(ids.tolist())) != len(ids):
        raise CampaignValidationError(f"{name} contains empty or duplicate identities")
    return ids


def _ids_digest(ids: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for value in np.asarray(ids).astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _fold_digest(ids: np.ndarray, clients: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256(b"fedcore.posthoc.fold-identity.v1\x00")
    for sample_id in ids.astype(str):
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    digest.update(np.asarray(clients, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(labels, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()


def inspect_training_artifact(
    path: str,
    expected_experiment_id: str,
    expected_plan_cell_config_hash: str,
) -> TrainingArtifactInfo:
    """Validate a primary NPZ and return its immutable identity summary."""

    absolute = os.path.abspath(path)
    if not os.path.isfile(absolute):
        raise CampaignValidationError(f"training artifact does not exist: {absolute}")
    digest_before = file_sha256(absolute)
    try:
        with np.load(absolute, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
    except Exception as exc:
        raise CampaignValidationError(
            f"invalid training NPZ {absolute}: {exc}"
        ) from exc
    if file_sha256(absolute) != digest_before:
        raise CampaignValidationError(
            f"training artifact changed while reading: {absolute}"
        )

    experiment_id = str(_scalar(arrays, "experiment_id"))
    if experiment_id != expected_experiment_id:
        raise CampaignValidationError(
            f"artifact experiment_id mismatch: {experiment_id!r} != {expected_experiment_id!r}"
        )
    plan_config_json = _scalar(arrays, "plan_cell_config_json")
    if not isinstance(plan_config_json, str):
        raise CampaignValidationError("plan_cell_config_json is not a string")
    try:
        plan_config = json.loads(plan_config_json)
    except json.JSONDecodeError as exc:
        raise CampaignValidationError("plan_cell_config_json is invalid") from exc
    if not isinstance(plan_config, dict):
        raise CampaignValidationError("plan_cell_config_json is not an object")
    plan_config_hash = semantic_hash(plan_config)
    if plan_config_hash != expected_plan_cell_config_hash:
        raise CampaignValidationError(
            f"artifact plan-cell configuration does not match {expected_experiment_id}"
        )
    if str(_scalar(arrays, "plan_cell_config_sha256")) != plan_config_hash:
        raise CampaignValidationError(
            "artifact plan_cell_config_sha256 is stale or forged"
        )

    training_config_json = _scalar(arrays, "training_config_json")
    if not isinstance(training_config_json, str):
        raise CampaignValidationError("training_config_json is not a string")
    try:
        training_config = json.loads(training_config_json)
    except json.JSONDecodeError as exc:
        raise CampaignValidationError("training_config_json is invalid") from exc
    if not isinstance(training_config, dict):
        raise CampaignValidationError("training_config_json is not an object")
    training_config_hash = semantic_hash(training_config)
    if str(_scalar(arrays, "training_config_sha256")) != training_config_hash:
        raise CampaignValidationError(
            "artifact training_config_sha256 is stale or forged"
        )
    raw_n_clients = training_config.get("n_clients")
    if (
        isinstance(raw_n_clients, bool)
        or not isinstance(raw_n_clients, int)
        or raw_n_clients <= 0
    ):
        raise CampaignValidationError("training config has invalid n_clients")
    n_clients = int(raw_n_clients)

    fold_ids: dict[str, np.ndarray] = {}
    fold_hashes: dict[str, str] = {}
    for fold in ("prop", "cert", "test"):
        required = [
            f"{fold}_logits",
            f"{fold}_y_open",
            f"{fold}_client",
            f"{fold}_sample_id",
        ]
        missing = [key for key in required if key not in arrays]
        if missing:
            raise CampaignValidationError(
                f"artifact is missing {fold} fields: {missing}"
            )
        logits = np.asarray(arrays[f"{fold}_logits"], dtype=float)
        labels = _integer_vector(arrays[f"{fold}_y_open"], f"{fold}_y_open")
        clients = _integer_vector(arrays[f"{fold}_client"], f"{fold}_client")
        ids = _stable_ids(arrays[f"{fold}_sample_id"], f"{fold}_sample_id")
        if logits.ndim != 2 or not (
            len(logits) == len(labels) == len(clients) == len(ids)
        ):
            raise CampaignValidationError(f"artifact {fold} arrays are not aligned")
        if len(ids) == 0 or np.any(~np.isfinite(logits)):
            raise CampaignValidationError(
                f"artifact {fold} is empty or has non-finite logits"
            )
        if np.any(clients < 0) or np.any(clients >= n_clients):
            raise CampaignValidationError(
                f"artifact {fold} client IDs exceed the roster"
            )
        if np.any((labels < -1) | (labels >= logits.shape[1])):
            raise CampaignValidationError(
                f"artifact {fold} labels violate open-set encoding"
            )
        fold_ids[fold] = ids
        fold_hashes[fold] = _fold_digest(ids, clients, labels)
    for index, left in enumerate(("prop", "cert", "test")):
        for right in ("prop", "cert", "test")[index + 1 :]:
            if set(fold_ids[left].tolist()) & set(fold_ids[right].tolist()):
                raise CampaignValidationError(
                    f"artifact stable IDs overlap between {left} and {right}"
                )
    fold_hash = semantic_hash(
        {fold: fold_hashes[fold] for fold in ("prop", "cert", "test")}
    )
    return TrainingArtifactInfo(
        path=absolute,
        sha256=digest_before,
        experiment_id=experiment_id,
        plan_cell_config_hash=plan_config_hash,
        training_config_hash=training_config_hash,
        fold_hash=fold_hash,
        cert_sample_ids_sha256=_ids_digest(fold_ids["cert"]),
        n_clients=n_clients,
    )


def _resolve_path(value: Any, base: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CampaignValidationError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return str(path.resolve(strict=False))


def _validate_training_manifest(
    manifest_path: str,
    cell: TrainingCell,
    artifact: TrainingArtifactInfo,
) -> None:
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise CampaignValidationError("training manifest is not an object")
    if manifest.get("manifest_type") == "fedcore.campaign.artifact":
        if manifest.get("experiment_id") != cell.experiment_id:
            raise CampaignValidationError("scheduler manifest experiment ID mismatch")
        if manifest.get("config_hash") != artifact.training_config_hash:
            raise CampaignValidationError(
                "scheduler manifest training config hash mismatch"
            )
        record = manifest.get("artifact")
        if not isinstance(record, dict):
            raise CampaignValidationError("scheduler manifest lacks artifact record")
        if record.get("sha256") != artifact.sha256:
            raise CampaignValidationError("scheduler manifest artifact hash mismatch")
        if int(record.get("size_bytes", -1)) != os.path.getsize(artifact.path):
            raise CampaignValidationError("scheduler manifest artifact size mismatch")
        # macOS exposes /var through /private/var.  Compare canonical real paths
        # so a manifest made from one spelling validates under the other.
        if os.path.realpath(str(record.get("path", ""))) != os.path.realpath(artifact.path):
            raise CampaignValidationError("scheduler manifest artifact path mismatch")
        return
    if manifest.get("experiment_id") != cell.experiment_id:
        raise CampaignValidationError("training manifest experiment ID mismatch")
    if manifest.get("status") != "completed":
        raise CampaignValidationError("training manifest is not terminal-completed")
    if manifest.get("config_hash") != artifact.training_config_hash:
        raise CampaignValidationError("training manifest config hash mismatch")
    manifest_training = manifest.get("training_config")
    if (
        not isinstance(manifest_training, dict)
        or semantic_hash(manifest_training) != artifact.training_config_hash
    ):
        raise CampaignValidationError(
            "training manifest configuration content mismatch"
        )
    records = manifest.get("artifacts")
    if not isinstance(records, list) or not any(
        isinstance(record, dict)
        and record.get("sha256") == artifact.sha256
        and os.path.realpath(str(record.get("path", "")))
        == os.path.realpath(artifact.path)
        for record in records
    ):
        raise CampaignValidationError(
            "training manifest does not bind the primary artifact"
        )


def expected_posthoc_cells(plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Literal Cartesian expansion of the authoritative post-hoc grid arrays."""

    posthoc = plan.get("posthoc")
    if not isinstance(posthoc, dict):
        raise CampaignValidationError("plan posthoc section is not an object")
    arrays: list[list[Any]] = []
    row_names: list[str] = []
    for plan_name, row_name in _PLAN_TO_ROW_GRID:
        values = posthoc.get(plan_name)
        if not isinstance(values, list) or not values:
            raise CampaignValidationError(
                f"posthoc.{plan_name} must be a non-empty explicit list for final validation"
            )
        tokens = [canonical_json(value) for value in values]
        if len(set(tokens)) != len(tokens):
            raise CampaignValidationError(
                f"posthoc.{plan_name} contains duplicate values"
            )
        for value in values:
            if isinstance(value, (dict, list)):
                raise CampaignValidationError(
                    f"posthoc.{plan_name} grid values must be JSON scalars"
                )
        arrays.append(values)
        row_names.append(row_name)
    cells = tuple(
        dict(zip(row_names, values, strict=True))
        for values in itertools.product(*arrays)
    )
    if not cells:
        raise CampaignValidationError(
            "authoritative post-hoc grid expands to zero cells"
        )
    return cells


def _grid_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        return tuple(canonical_json(row[field]) for field in GRID_FIELDS)
    except KeyError as exc:
        raise CampaignValidationError(
            f"result row is missing grid field {exc.args[0]!r}"
        ) from exc


def _number(value: Any, name: str, *, nullable: bool = False) -> Optional[float]:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignValidationError(f"{name} must be finite")
    return result


def _normalize_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical risk-output schema, including legacy defaults.

    Historical one-shot artifacts contain only numerical risk UCBs and the
    component-level failure-budget names.  They remain readable as single-
    selector numerical-UCB rows.  New producers should write every field
    explicitly, especially for fixed-alpha Holm/IUT decisions.
    """

    normalized = dict(row)
    normalized.setdefault("delta_r", normalized.get("delta_conditional_risk"))
    normalized.setdefault("delta_c", normalized.get("delta_acceptance_lower"))
    normalized.setdefault("family_procedure", "single_selector")
    normalized.setdefault("risk_output_type", "numerical_ucb")
    if "risk_pass" not in normalized:
        risk = normalized.get("cert_risk_ucb")
        alpha = normalized.get("alpha")
        normalized["risk_pass"] = bool(
            isinstance(risk, (int, float))
            and not isinstance(risk, bool)
            and math.isfinite(float(risk))
            and isinstance(alpha, (int, float))
            and not isinstance(alpha, bool)
            and math.isfinite(float(alpha))
            and float(risk) <= float(alpha)
        )
    normalized.setdefault("iut_raw_pvalue", None)
    normalized.setdefault("holm_adjusted_pvalue", None)
    normalized.setdefault("holm_rank", None)
    return normalized


def _validate_metric_row(row: Mapping[str, Any], cell: TrainingCell) -> None:
    missing = set(
        CANONICAL_METRIC_FIELDS + GRID_FIELDS + ACCOUNTING_FIELDS + PROVENANCE_FIELDS
    ) - set(row)
    if missing:
        raise CampaignValidationError(
            f"result row is missing canonical fields: {sorted(missing)}"
        )
    if not isinstance(row["certified"], bool):
        raise CampaignValidationError("certified must be boolean")
    alpha = _number(row["alpha"], "alpha")
    delta_r = _number(row["delta_r"], "delta_r")
    delta_c = _number(row["delta_c"], "delta_c")
    gamma = _number(row["gamma"], "gamma")
    audit_fraction = _number(row["audit_budget_fraction"], "audit_budget_fraction")
    rho = _number(row["rho"], "rho", nullable=True)
    if alpha is None or not 0.0 < alpha < 1.0:
        raise CampaignValidationError("alpha must lie in (0, 1)")
    if delta_r is None or not 0.0 < delta_r < 1.0:
        raise CampaignValidationError("delta_r must lie in (0, 1)")
    if delta_c is None or not 0.0 < delta_c < 1.0:
        raise CampaignValidationError("delta_c must lie in (0, 1)")
    if (
        not isinstance(row["family_procedure"], str)
        or not row["family_procedure"]
    ):
        raise CampaignValidationError("family_procedure must be non-empty")
    risk_output_type = row["risk_output_type"]
    if risk_output_type not in {"numerical_ucb", "fixed_alpha_decision"}:
        raise CampaignValidationError(
            "risk_output_type must be numerical_ucb or fixed_alpha_decision"
        )
    risk_pass = row["risk_pass"]
    if not isinstance(risk_pass, bool):
        raise CampaignValidationError("risk_pass must be boolean")
    for name in ("iut_raw_pvalue", "holm_adjusted_pvalue"):
        pvalue = _number(row[name], name, nullable=True)
        if pvalue is not None and not 0.0 <= pvalue <= 1.0:
            raise CampaignValidationError(f"{name} must be null or lie in [0, 1]")
    holm_rank = row["holm_rank"]
    if holm_rank is not None and (
        isinstance(holm_rank, bool)
        or not isinstance(holm_rank, int)
        or holm_rank <= 0
    ):
        raise CampaignValidationError("holm_rank must be null or a positive integer")
    if gamma not in {0.5, 0.7, 1.0}:
        raise CampaignValidationError("gamma must be one of {0.5, 0.7, 1.0}")
    if audit_fraction is None or not 0.0 < audit_fraction <= 1.0:
        raise CampaignValidationError("audit_budget_fraction must lie in (0, 1]")
    if rho is not None and rho < 0.0:
        raise CampaignValidationError("rho must be null or non-negative")
    if row["dirichlet_alpha"] is not None:
        d_value = _number(row["dirichlet_alpha"], "dirichlet_alpha")
        if d_value is None or d_value <= 0.0:
            raise CampaignValidationError("dirichlet_alpha must be positive or null")
    metric_values: dict[str, float] = {}
    for name in (
        "cert_coverage_lcb",
        "prop_coverage",
        "prop_risk",
        "test_coverage",
        "test_risk",
    ):
        value = _number(row[name], name)
        if value is None or not 0.0 <= value <= 1.0:
            raise CampaignValidationError(f"{name} must lie in [0, 1]")
        metric_values[name] = value
    risk = _number(row["cert_risk_ucb"], "cert_risk_ucb", nullable=True)
    feasible = row.get("certificate_feasible")
    if risk_output_type == "numerical_ucb":
        if any(
            row[name] is not None
            for name in ("iut_raw_pvalue", "holm_adjusted_pvalue", "holm_rank")
        ):
            raise CampaignValidationError(
                "numerical_ucb rows must not carry Holm/IUT decision fields"
            )
        if risk is None:
            if feasible is not False or row["certified"]:
                raise CampaignValidationError(
                    "null cert_risk_ucb is allowed only for an explicitly infeasible certificate"
                )
        elif not 0.0 <= risk <= 1.0:
            raise CampaignValidationError("cert_risk_ucb must lie in [0, 1]")
        expected_risk_pass = bool(risk is not None and risk <= alpha)
        if risk_pass != expected_risk_pass:
            raise CampaignValidationError(
                "numerical-UCB risk_pass is inconsistent with cert_risk_ucb and alpha"
            )
        if row["certified"] and (risk is None or risk > alpha):
            raise CampaignValidationError("certified row has risk UCB above alpha")
    else:
        if risk is not None:
            raise CampaignValidationError(
                "fixed_alpha_decision requires null cert_risk_ucb"
            )
        if row["family_procedure"] != "holm_iut":
            raise CampaignValidationError(
                "fixed_alpha_decision requires family_procedure='holm_iut'"
            )
        raw_pvalue = _number(
            row["iut_raw_pvalue"], "iut_raw_pvalue", nullable=True
        )
        adjusted_pvalue = _number(
            row["holm_adjusted_pvalue"], "holm_adjusted_pvalue", nullable=True
        )
        holm_fields = (raw_pvalue, adjusted_pvalue, holm_rank)
        if any(value is None for value in holm_fields) and not all(
            value is None for value in holm_fields
        ):
            raise CampaignValidationError(
                "Holm raw p-value, adjusted p-value, and rank must be jointly present or null"
            )
        if risk_pass and adjusted_pvalue is None:
            raise CampaignValidationError(
                "a passing fixed-alpha decision requires its Holm evidence"
            )
        if (
            adjusted_pvalue is not None
            and raw_pvalue is not None
            and adjusted_pvalue + 1e-15 < raw_pvalue
        ):
            raise CampaignValidationError(
                "Holm-adjusted p-value must not be smaller than its raw IUT p-value"
            )
        if adjusted_pvalue is not None and risk_pass != bool(adjusted_pvalue <= delta_r):
            raise CampaignValidationError(
                "fixed-alpha risk_pass is inconsistent with adjusted p-value and delta_r"
            )
        if row["certified"] and (
            not risk_pass or metric_values["cert_coverage_lcb"] <= 0.0
        ):
            raise CampaignValidationError(
                "fixed-alpha certified row requires risk_pass=true and positive coverage LCB"
            )
        if row["certified"] and feasible is False:
            raise CampaignValidationError(
                "explicitly infeasible fixed-alpha row cannot be certified"
            )
    for name in (
        "cert_n",
        "cert_k",
        "n_clients",
        "cert_observation_count",
        "traffic_sample_size",
    ):
        value = row[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CampaignValidationError(f"{name} must be a non-negative integer")
    if row["n_clients"] <= 0 or row["cert_k"] > row["cert_n"]:
        raise CampaignValidationError("invalid client or certification counts")
    if row["cert_n"] > row["cert_observation_count"]:
        raise CampaignValidationError(
            "accepted certification count exceeds audit observations"
        )
    if not isinstance(row["score_name"], str) or not row["score_name"]:
        raise CampaignValidationError("score_name must be non-empty")
    if not isinstance(row["Lambda"], str) or not row["Lambda"]:
        raise CampaignValidationError("Lambda must be non-empty")
    for field in ("threshold_policy", "allocation_policy", "certificate_variant"):
        if not isinstance(row[field], str) or not row[field]:
            raise CampaignValidationError(f"{field} must be non-empty")
    if row["source_experiment_id"] != cell.experiment_id:
        raise CampaignValidationError(
            "row source_experiment_id does not match plan cell"
        )
    for field in (
        "input_artifact_sha256",
        "training_config_sha256",
        "fold_sha256",
        "posthoc_config_sha256",
        "cert_sample_ids_sha256",
    ):
        _require_hash(row[field], field)
    traffic_size = int(row["traffic_sample_size"])
    traffic_digest = row["traffic_sample_ids_sha256"]
    if traffic_size > 0:
        _require_hash(traffic_digest, "traffic_sample_ids_sha256")
    elif traffic_digest not in {"", None} and not _is_hash(traffic_digest):
        raise CampaignValidationError("invalid optional traffic_sample_ids_sha256")

    total = _number(row["delta_total"], "delta_total")
    delta = _number(row["delta"], "delta")
    components = [
        _number(row[name], name)
        for name in (
            "delta_mixture",
            "delta_conditional_risk",
            "delta_acceptance_lower",
            "delta_acceptance_upper",
        )
    ]
    spent = _number(row["delta_spent"], "delta_spent")
    slack = _number(row["delta_slack"], "delta_slack")
    assert None not in [total, delta, spent, slack, *components]
    if not 0.0 < float(total) < 1.0 or not math.isclose(
        float(delta), float(total), abs_tol=1e-12
    ):
        raise CampaignValidationError("row delta and delta_total are inconsistent")
    if any(float(value) < 0.0 for value in components):
        raise CampaignValidationError("failure-budget components must be non-negative")
    component_sum = math.fsum(float(value) for value in components)
    if not math.isclose(float(spent), component_sum, rel_tol=0.0, abs_tol=1e-12):
        raise CampaignValidationError(
            "delta_spent does not equal declared component sum"
        )
    if component_sum > float(total) + 1e-12:
        raise CampaignValidationError("failure-budget components exceed total delta")
    if not math.isclose(
        float(slack), float(total) - component_sum, rel_tol=0.0, abs_tol=1e-12
    ):
        raise CampaignValidationError("delta_slack is inconsistent")


def _validate_posthoc_manifest(
    path: str,
    result_path: str,
    result_sha: str,
    training: TrainingArtifactInfo,
    result: Mapping[str, Any],
) -> None:
    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        raise CampaignValidationError("post-hoc manifest is not an object")
    if manifest.get("experiment_id") != result.get("posthoc_experiment_id"):
        raise CampaignValidationError("post-hoc manifest experiment ID mismatch")
    if manifest.get("status") != "completed":
        raise CampaignValidationError("post-hoc manifest is not terminal-completed")
    if manifest.get("config_hash") != training.training_config_hash:
        raise CampaignValidationError("post-hoc manifest training config hash mismatch")
    manifest_training = manifest.get("training_config")
    if (
        not isinstance(manifest_training, dict)
        or semantic_hash(manifest_training) != training.training_config_hash
    ):
        raise CampaignValidationError(
            "post-hoc manifest training configuration mismatch"
        )
    if manifest.get("fold_hash") != training.fold_hash:
        raise CampaignValidationError("post-hoc manifest fold hash mismatch")
    manifest_posthoc = manifest.get("posthoc_config")
    if not isinstance(manifest_posthoc, dict) or manifest_posthoc.get(
        "posthoc_config_sha256"
    ) != result.get("posthoc_config_sha256"):
        raise CampaignValidationError("post-hoc manifest configuration hash mismatch")
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise CampaignValidationError("post-hoc manifest lacks artifact records")
    expected = {
        (os.path.realpath(training.path), training.sha256),
        (os.path.realpath(result_path), result_sha),
    }
    observed = {
        (os.path.realpath(str(record.get("path", ""))), str(record.get("sha256", "")))
        for record in records
        if isinstance(record, dict)
    }
    if not expected.issubset(observed):
        raise CampaignValidationError(
            "post-hoc manifest does not bind input and output artifacts"
        )
    for record in records:
        if not isinstance(record, dict):
            raise CampaignValidationError(
                "post-hoc manifest has malformed artifact record"
            )
        record_path = str(record.get("path", ""))
        if not os.path.isfile(record_path) or file_sha256(record_path) != record.get(
            "sha256"
        ):
            raise CampaignValidationError(
                "post-hoc manifest artifact checksum is invalid"
            )
        if os.path.getsize(record_path) != record.get("size_bytes"):
            raise CampaignValidationError("post-hoc manifest artifact size is invalid")


def _load_posthoc_result(
    entry: Mapping[str, Any],
    base: str,
    cell: TrainingCell,
    training: TrainingArtifactInfo,
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(entry, Mapping):
        raise CampaignValidationError("posthoc_outputs entries must be objects")
    result_path = _resolve_path(entry.get("path"), base, "posthoc output path")
    declared_sha = _require_hash(entry.get("sha256"), "posthoc output sha256")
    if not os.path.isfile(result_path) or file_sha256(result_path) != declared_sha:
        raise CampaignValidationError(
            f"post-hoc output checksum mismatch: {result_path}"
        )
    result = _read_json(result_path)
    if not isinstance(result, dict):
        raise CampaignValidationError("post-hoc result is not an object")
    if (
        result.get("schema_version") != 1
        or result.get("artifact_type") != "fedcore.oneshot.posthoc-results"
    ):
        raise CampaignValidationError("unsupported post-hoc result schema")
    if result.get("status") != "completed":
        raise CampaignValidationError("post-hoc result is not completed")
    if result.get("source_experiment_id") != cell.experiment_id:
        raise CampaignValidationError("post-hoc result source experiment mismatch")
    input_record = result.get("input_artifact")
    if (
        not isinstance(input_record, dict)
        or input_record.get("sha256") != training.sha256
    ):
        raise CampaignValidationError(
            "post-hoc result is not bound to the training artifact"
        )
    if os.path.realpath(str(input_record.get("path", ""))) != os.path.realpath(
        training.path
    ) or input_record.get("size_bytes") != os.path.getsize(training.path):
        raise CampaignValidationError("post-hoc input artifact path or size mismatch")
    if result.get("training_config_sha256") != training.training_config_hash:
        raise CampaignValidationError("post-hoc result training config hash mismatch")
    if result.get("fold_sha256") != training.fold_hash:
        raise CampaignValidationError("post-hoc result fold identity hash mismatch")
    config = result.get("posthoc_config")
    if not isinstance(config, dict):
        raise CampaignValidationError("post-hoc result lacks a configuration object")
    config_hash = semantic_hash(config)
    if result.get("posthoc_config_sha256") != config_hash:
        raise CampaignValidationError("post-hoc configuration hash mismatch")
    rows = result.get("rows")
    if not isinstance(rows, list) or result.get("row_count") != len(rows):
        raise CampaignValidationError("post-hoc row_count is inconsistent")
    typed_rows: list[dict[str, Any]] = []
    declared_budget = config.get("failure_budget")
    if not isinstance(declared_budget, dict):
        raise CampaignValidationError(
            "post-hoc config lacks complete failure_budget accounting"
        )
    for raw in rows:
        if not isinstance(raw, dict):
            raise CampaignValidationError("post-hoc rows must be objects")
        row = _normalize_metric_row(raw)
        _validate_metric_row(row, cell)
        if row["input_artifact_sha256"] != training.sha256:
            raise CampaignValidationError("row input artifact hash mismatch")
        if row["training_config_sha256"] != training.training_config_hash:
            raise CampaignValidationError("row training config hash mismatch")
        if row["fold_sha256"] != training.fold_hash:
            raise CampaignValidationError("row fold hash mismatch")
        if row["posthoc_config_sha256"] != config_hash:
            raise CampaignValidationError("row post-hoc config hash mismatch")
        for field in ACCOUNTING_FIELDS:
            if field not in declared_budget or not math.isclose(
                float(row[field]),
                float(declared_budget[field]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise CampaignValidationError(
                    f"row {field} differs from its hashed post-hoc configuration"
                )
        if int(row["n_clients"]) != training.n_clients:
            raise CampaignValidationError(
                "row n_clients differs from training artifact roster"
            )
        if "alpha" in config and row["alpha"] != config["alpha"]:
            raise CampaignValidationError(
                "row alpha differs from post-hoc configuration"
            )
        if "scores" in config and row["score_name"] not in config["scores"]:
            raise CampaignValidationError(
                "row score is absent from post-hoc configuration"
            )
        if "gammas" in config and row["gamma"] not in config["gammas"]:
            raise CampaignValidationError(
                "row gamma is absent from post-hoc configuration"
            )
        typed_rows.append(row)
    manifest_path = _resolve_path(
        entry.get("manifest_path"), base, "posthoc manifest path"
    )
    _validate_posthoc_manifest(
        manifest_path,
        result_path,
        declared_sha,
        training,
        result,
    )
    return typed_rows, str(result.get("posthoc_experiment_id", ""))


def _load_record_bundle(path: str, plan_hash: str) -> tuple[list[dict[str, Any]], str]:
    absolute = os.path.abspath(path)
    bundle = _read_json(absolute)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise CampaignValidationError("unsupported run-record bundle schema")
    if bundle.get("plan_sha256") != plan_hash:
        raise CampaignValidationError(
            "run-record bundle is not bound to the authoritative plan"
        )
    records = bundle.get("records")
    if not isinstance(records, list):
        raise CampaignValidationError("run-record bundle records must be a list")
    if any(not isinstance(record, dict) for record in records):
        raise CampaignValidationError("every run record must be an object")
    return [dict(record) for record in records], os.path.dirname(absolute)


def _coverage_failure(report: Mapping[str, Any], policy: str) -> Optional[str]:
    missing = int(report["missing_training_record_count"])
    nonterminal = int(report["nonterminal_training_record_count"])
    unsuccessful = int(report["terminal_unsuccessful_training_record_count"])
    missing_posthoc = int(report["missing_posthoc_cell_count"])
    if policy == "complete":
        if missing or nonterminal or unsuccessful or missing_posthoc:
            return (
                "complete policy rejected campaign: "
                f"missing_training={missing}, nonterminal={nonterminal}, "
                f"terminal_unsuccessful={unsuccessful}, missing_posthoc={missing_posthoc}"
            )
    elif policy == "allow-blocked":
        if missing or nonterminal or missing_posthoc:
            return (
                "allow-blocked policy requires all 45+24 training records, terminal states, "
                f"and complete grids for succeeded runs: missing_training={missing}, "
                f"nonterminal={nonterminal}, missing_posthoc={missing_posthoc}"
            )
    return None


def collect_campaign(
    plan_path: str, records_path: str, policy: str
) -> CampaignCollection:
    """Validate every observed object and return only observed scientific rows."""

    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {sorted(POLICIES)}")
    plan = load_plan(plan_path)
    plan_hash = semantic_hash(plan)
    cells = expand_training_cells(plan)
    if len(cells) != 69:
        raise CampaignValidationError(
            "authoritative training matrix is not 45+24 cells"
        )
    expected_grid = expected_posthoc_cells(plan)
    expected_grid_keys = {_grid_key(cell) for cell in expected_grid}
    records, base = _load_record_bundle(records_path, plan_hash)
    cell_by_id = {cell.experiment_id: cell for cell in cells}
    record_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        experiment_id = record.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise CampaignValidationError("run record lacks experiment_id")
        if experiment_id not in cell_by_id:
            raise CampaignValidationError(
                f"run record has unknown plan cell {experiment_id!r}"
            )
        if experiment_id in record_by_id:
            raise CampaignValidationError(f"duplicate run record for {experiment_id!r}")
        record_by_id[experiment_id] = record

    missing_ids = sorted(set(cell_by_id) - set(record_by_id))
    nonterminal_ids: list[str] = []
    terminal_unsuccessful_ids: list[str] = []
    missing_posthoc: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    status_counts = {status: 0 for status in sorted(RECORD_STATUSES)}
    status_ids: dict[str, list[str]] = {
        status: [] for status in sorted(RECORD_STATUSES)
    }
    posthoc_ids: set[str] = set()
    artifact_by_experiment: dict[str, TrainingArtifactInfo] = {}

    for experiment_id in sorted(record_by_id):
        record = record_by_id[experiment_id]
        cell = cell_by_id[experiment_id]
        status = record.get("status")
        if status not in RECORD_STATUSES:
            raise CampaignValidationError(
                f"record {experiment_id!r} has invalid status {status!r}"
            )
        status_counts[str(status)] += 1
        status_ids[str(status)].append(experiment_id)
        if record.get("family") != cell.family:
            raise CampaignValidationError(f"record {experiment_id!r} family mismatch")
        if record.get("config_hash") != cell.config_hash:
            raise CampaignValidationError(
                f"record {experiment_id!r} config hash mismatch"
            )
        outputs = record.get("posthoc_outputs", [])
        if not isinstance(outputs, list):
            raise CampaignValidationError("posthoc_outputs must be a list")
        if status in NONTERMINAL_STATUSES:
            nonterminal_ids.append(experiment_id)
            if outputs:
                raise CampaignValidationError(
                    "nonterminal record cannot claim post-hoc outputs"
                )
            continue
        if status != "succeeded":
            terminal_unsuccessful_ids.append(experiment_id)
            if (
                not isinstance(record.get("reason"), str)
                or not record["reason"].strip()
            ):
                raise CampaignValidationError(
                    f"terminal non-success record {experiment_id!r} requires an exact reason"
                )
            if outputs:
                raise CampaignValidationError(
                    "failed/blocked/infeasible training record cannot carry scientific rows"
                )
            continue

        artifact_path = _resolve_path(
            record.get("artifact_path"), base, "artifact_path"
        )
        declared_artifact_sha = _require_hash(
            record.get("artifact_sha256"), "artifact_sha256"
        )
        artifact = inspect_training_artifact(
            artifact_path,
            experiment_id,
            cell.config_hash,
        )
        if artifact.sha256 != declared_artifact_sha:
            raise CampaignValidationError(
                f"record artifact hash mismatch for {experiment_id}"
            )
        manifest_path = _resolve_path(
            record.get("manifest_path"), base, "training manifest path"
        )
        _validate_training_manifest(manifest_path, cell, artifact)
        artifact_by_experiment[experiment_id] = artifact

        observed_grid: dict[tuple[str, ...], dict[str, Any]] = {}
        for entry in outputs:
            result_rows, posthoc_id = _load_posthoc_result(
                entry,
                base,
                cell,
                artifact,
            )
            if not posthoc_id or posthoc_id in posthoc_ids:
                raise CampaignValidationError(
                    "duplicate or empty post-hoc experiment ID"
                )
            posthoc_ids.add(posthoc_id)
            for row in result_rows:
                key = _grid_key(row)
                if key not in expected_grid_keys:
                    raise CampaignValidationError(
                        f"observed post-hoc row is outside authoritative grid for {experiment_id}"
                    )
                if key in observed_grid:
                    raise CampaignValidationError(
                        f"duplicate post-hoc grid cell for {experiment_id}"
                    )
                observed_grid[key] = row
        missing_keys = expected_grid_keys - set(observed_grid)
        for key in sorted(missing_keys):
            missing_posthoc.append(
                {
                    "experiment_id": experiment_id,
                    "grid_key": list(key),
                }
            )
        for key in sorted(observed_grid):
            row = dict(observed_grid[key])
            row["family"] = cell.family
            row["run_status"] = "succeeded"
            row["plan_cell_config_sha256"] = cell.config_hash
            all_rows.append(row)

    # Competing analyses must use common audit/traffic observations.  Allocation
    # policies may alter confidence bounds but must not alter A/K for a fixed rule.
    cert_groups: dict[tuple[Any, ...], set[str]] = {}
    traffic_groups: dict[tuple[Any, ...], set[str]] = {}
    count_groups: dict[tuple[Any, ...], set[tuple[int, int]]] = {}
    for row in all_rows:
        experiment_id = str(row["source_experiment_id"])
        redraw = row.get("audit_redraw_index", 0)
        cert_group = (
            experiment_id,
            canonical_json(row["audit_budget_fraction"]),
            redraw,
        )
        cert_groups.setdefault(cert_group, set()).add(
            str(row["cert_sample_ids_sha256"])
        )
        traffic_group = (
            experiment_id,
            int(row["traffic_sample_size"]),
            row.get("traffic_draw_index", 0),
        )
        traffic_groups.setdefault(traffic_group, set()).add(
            str(row["traffic_sample_ids_sha256"] or "")
        )
        count_key = tuple(
            canonical_json(row[field])
            for field in GRID_FIELDS
            if field != "allocation_policy"
        ) + (experiment_id, redraw)
        count_groups.setdefault(count_key, set()).add(
            (int(row["cert_n"]), int(row["cert_k"]))
        )
    if any(len(values) != 1 for values in cert_groups.values()):
        raise CampaignValidationError(
            "competing methods use different certification IDs"
        )
    if any(len(values) != 1 for values in traffic_groups.values()):
        raise CampaignValidationError("competing methods use different traffic IDs")
    if any(len(values) != 1 for values in count_groups.values()):
        raise CampaignValidationError(
            "allocation policies do not reuse identical A/K counts"
        )
    for (experiment_id, fraction_token, _redraw), values in cert_groups.items():
        fraction = json.loads(fraction_token)
        if isinstance(fraction, (int, float)) and math.isclose(
            float(fraction), 1.0, abs_tol=1e-12
        ):
            expected_digest = artifact_by_experiment[
                experiment_id
            ].cert_sample_ids_sha256
            if next(iter(values)) != expected_digest:
                raise CampaignValidationError(
                    "full-audit result certification digest does not match training artifact"
                )

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "fedcore.campaign.final-validation",
        "validation_policy": policy,
        "plan_sha256": plan_hash,
        "records_sha256": file_sha256(records_path),
        "expected_training_cell_count": 69,
        "expected_cifar_cell_count": 45,
        "expected_medical_cell_count": 24,
        "observed_training_record_count": len(record_by_id),
        "missing_training_record_count": len(missing_ids),
        "missing_training_experiment_ids": missing_ids,
        "nonterminal_training_record_count": len(nonterminal_ids),
        "nonterminal_training_experiment_ids": sorted(nonterminal_ids),
        "terminal_unsuccessful_training_record_count": len(terminal_unsuccessful_ids),
        "terminal_unsuccessful_training_experiment_ids": sorted(
            terminal_unsuccessful_ids
        ),
        "training_status_counts": status_counts,
        "training_experiment_ids_by_status": {
            status: sorted(experiment_ids)
            for status, experiment_ids in status_ids.items()
        },
        "expected_posthoc_cells_per_succeeded_run": len(expected_grid_keys),
        "expected_posthoc_cell_count_if_all_training_succeeded": (
            len(cells) * len(expected_grid_keys)
        ),
        "observed_scientific_row_count": len(all_rows),
        "missing_posthoc_cell_count": len(missing_posthoc),
        "missing_posthoc_cells": missing_posthoc,
        "posthoc_cells_unavailable_due_terminal_training_status": (
            len(terminal_unsuccessful_ids) * len(expected_grid_keys)
        ),
        "posthoc_cells_unavailable_due_missing_or_nonterminal_training": (
            (len(missing_ids) + len(nonterminal_ids)) * len(expected_grid_keys)
        ),
        "scientific_rows_are_observed_only": True,
        "negative_and_infeasible_rows_preserved": True,
    }
    rejection = _coverage_failure(report, policy)
    report["coverage_policy_passed"] = rejection is None
    report["manuscript_ready"] = bool(
        not missing_ids
        and not nonterminal_ids
        and not terminal_unsuccessful_ids
        and not missing_posthoc
    )
    if rejection is not None:
        raise CampaignValidationError(rejection)
    sorted_rows = tuple(
        sorted(
            all_rows, key=lambda row: (str(row["source_experiment_id"]), _grid_key(row))
        )
    )
    return CampaignCollection(sorted_rows, report)


def _output_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    observed = {key for row in rows for key in row}
    leading = [
        field for field in OUTPUT_LEADING_FIELDS if field in observed or not rows
    ]
    return leading + sorted(observed - set(leading))


def _flat_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return canonical_json(value)


def _atomic_bytes(path: str, payload: bytes) -> str:
    absolute = os.path.abspath(path)
    directory = os.path.dirname(absolute) or "."
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(absolute):
        raise FileExistsError(f"refusing to replace final artifact {absolute}")
    fd, temporary = tempfile.mkstemp(prefix=".finalize.", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, absolute)
        return absolute
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_csv(
    path: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _flat_value(row.get(column)) for column in columns})
    return _atomic_bytes(path, stream.getvalue().encode("utf-8"))


def _parquet_type(column: str, values: Sequence[Any]):
    import pyarrow as pa

    boolean_fields = {
        "certified",
        "risk_pass",
        "proposal_feasible",
        "certificate_feasible",
    }
    integer_fields = {
        "cert_n",
        "cert_k",
        "n_clients",
        "cert_observation_count",
        "traffic_sample_size",
        "audit_redraw_index",
        "traffic_draw_index",
        "holm_rank",
    }
    float_fields = {
        "cert_risk_ucb",
        "cert_coverage_lcb",
        "prop_coverage",
        "prop_risk",
        "test_coverage",
        "test_risk",
        "gamma",
        "alpha",
        "delta",
        "delta_r",
        "delta_c",
        "iut_raw_pvalue",
        "holm_adjusted_pvalue",
        "dirichlet_alpha",
        "rho",
        "audit_budget_fraction",
        *ACCOUNTING_FIELDS,
    }
    if column in boolean_fields:
        return pa.bool_()
    if column in integer_fields:
        return pa.int64()
    if column in float_fields:
        return pa.float64()
    nonnull = [_flat_value(value) for value in values if value is not None]
    if nonnull and all(isinstance(value, bool) for value in nonnull):
        return pa.bool_()
    if nonnull and all(
        isinstance(value, int) and not isinstance(value, bool) for value in nonnull
    ):
        return pa.int64()
    if nonnull and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in nonnull
    ):
        return pa.float64()
    return pa.string()


def _write_parquet(
    path: str,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> str:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise CampaignValidationError(
            "pyarrow is required for mandated final Parquet"
        ) from exc
    fields = []
    arrays = []
    for column in columns:
        raw_values = [row.get(column) for row in rows]
        data_type = _parquet_type(column, raw_values)
        values = [_flat_value(value) for value in raw_values]
        if pa.types.is_string(data_type):
            values = [None if value is None else str(value) for value in values]
        fields.append(pa.field(column, data_type))
        arrays.append(pa.array(values, type=data_type))
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    absolute = os.path.abspath(path)
    directory = os.path.dirname(absolute) or "."
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(absolute):
        raise FileExistsError(f"refusing to replace final artifact {absolute}")
    fd, temporary = tempfile.mkstemp(
        prefix=".finalize.", suffix=".parquet", dir=directory
    )
    os.close(fd)
    try:
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, absolute)
        return absolute
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def finalize_campaign(
    plan_path: str,
    records_path: str,
    out_dir: str,
    policy: str,
) -> CampaignCollection:
    """Validate first; only then emit observed rows and checksums."""

    collection = collect_campaign(plan_path, records_path, policy)
    directory = os.path.abspath(out_dir)
    paths = {
        "csv": os.path.join(directory, "all_runs.csv"),
        "parquet": os.path.join(directory, "all_runs.parquet"),
        "report": os.path.join(directory, "validation.json"),
        "checksums": os.path.join(directory, "checksums.sha256"),
    }
    existing = [path for path in paths.values() if os.path.exists(path)]
    if existing:
        raise FileExistsError(f"refusing to replace final artifacts: {existing}")
    columns = _output_columns(collection.rows)
    csv_path = _write_csv(paths["csv"], collection.rows, columns)
    parquet_path = _write_parquet(paths["parquet"], collection.rows, columns)
    report = {
        **collection.report,
        "all_runs_columns": columns,
        "all_runs_csv": os.path.abspath(csv_path),
        "all_runs_parquet": os.path.abspath(parquet_path),
    }
    report_path = _atomic_bytes(
        paths["report"],
        (
            json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )
    checksum_lines = [
        f"{file_sha256(path)}  {os.path.basename(path)}"
        for path in sorted((csv_path, parquet_path, report_path), key=os.path.basename)
    ]
    _atomic_bytes(
        paths["checksums"], ("\n".join(checksum_lines) + "\n").encode("ascii")
    )
    return CampaignCollection(collection.rows, report)


def finalize_unplanned_blocked_state(
    blocked_path: str,
    out_dir: str,
) -> CampaignCollection:
    """Emit typed empty final assets when the authoritative plan itself is absent.

    This is not an incomplete scientific aggregation: without class splits and
    diagnosis identities there are no legitimate plan-cell IDs to enumerate.
    The function therefore permits no result input and always records failed
    coverage/manuscript readiness. It exists solely for completion criterion B.
    """

    blocked = _read_json(blocked_path)
    if not isinstance(blocked, dict) or blocked.get("schema_version") != 1:
        raise CampaignValidationError("blocked state must use schema_version 1")
    if not str(blocked.get("campaign_status", "")).startswith("blocked"):
        raise CampaignValidationError(
            "unplanned finalization requires blocked campaign state"
        )
    blockers = blocked.get("blockers")
    if not isinstance(blockers, list) or not blockers:
        raise CampaignValidationError(
            "unplanned finalization requires at least one blocker"
        )
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("id"), str)
        or not item["id"]
        or not isinstance(item.get("remediation"), str)
        or not item["remediation"]
        for item in blockers
    ):
        raise CampaignValidationError("blocked state has a malformed blocker")
    progress = blocked.get("primary_progress")
    if not isinstance(progress, dict):
        raise CampaignValidationError("blocked state lacks primary_progress")
    expected = {
        "cifar_required": 45,
        "cifar_executed": 0,
        "fed_isic_required": 24,
        "fed_isic_executed": 0,
        "primary_jobs_queued": 0,
    }
    if any(progress.get(key) != value for key, value in expected.items()):
        raise CampaignValidationError(
            "unplanned blocked-state emission is only valid before any primary job is queued"
        )

    directory = os.path.abspath(out_dir)
    paths = {
        "csv": os.path.join(directory, "all_runs.csv"),
        "parquet": os.path.join(directory, "all_runs.parquet"),
        "report": os.path.join(directory, "validation.json"),
        "table_marker": os.path.join(directory, "tables", "BLOCKED.md"),
        "figure_marker": os.path.join(directory, "figures", "BLOCKED.md"),
        "source_marker": os.path.join(directory, "source_data", "README.md"),
        "checksums": os.path.join(directory, "checksums.sha256"),
    }
    existing = [path for path in paths.values() if os.path.exists(path)]
    if existing:
        raise FileExistsError(f"refusing to replace final artifacts: {existing}")
    rows: tuple[dict[str, Any], ...] = ()
    columns = _output_columns(rows)
    csv_path = _write_csv(paths["csv"], rows, columns)
    parquet_path = _write_parquet(paths["parquet"], rows, columns)
    report = {
        "schema_version": 1,
        "artifact_type": "fedcore.campaign.blocked-without-authoritative-plan",
        "validation_policy": "unplanned-blocked-only",
        "blocked_state_sha256": file_sha256(blocked_path),
        "blocker_ids": [item["id"] for item in blockers],
        "expected_training_cell_count": 69,
        "expected_cifar_cell_count": 45,
        "expected_medical_cell_count": 24,
        "observed_training_record_count": 0,
        "observed_scientific_row_count": 0,
        "scientific_rows_are_observed_only": True,
        "negative_and_infeasible_rows_preserved": True,
        "authoritative_plan_available": False,
        "coverage_policy_passed": False,
        "manuscript_ready": False,
        "all_runs_columns": columns,
        "all_runs_csv": os.path.abspath(csv_path),
        "all_runs_parquet": os.path.abspath(parquet_path),
    }
    report_path = _atomic_bytes(
        paths["report"],
        (
            json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )
    marker = (
        "# Blocked\n\nNo manuscript numerical asset is generated because the "
        "authoritative one-shot plan and all primary results are unavailable. "
        "See `validation.json` and `docs/agent/blocked.json`.\n"
    ).encode("utf-8")
    table_marker = _atomic_bytes(paths["table_marker"], marker)
    figure_marker = _atomic_bytes(paths["figure_marker"], marker)
    source_marker = _atomic_bytes(
        paths["source_marker"],
        (
            "# Final source-data status\n\nNo primary scientific rows exist. "
            "Plan-independent legacy reservoir accounting remains under "
            "`results/accounting/` and is not promoted into `all_runs`.\n"
        ).encode("utf-8"),
    )
    checksum_targets = (
        csv_path,
        parquet_path,
        report_path,
        table_marker,
        figure_marker,
        source_marker,
    )
    checksum_lines = [
        f"{file_sha256(path)}  {os.path.relpath(path, directory)}"
        for path in sorted(
            checksum_targets, key=lambda value: os.path.relpath(value, directory)
        )
    ]
    _atomic_bytes(
        paths["checksums"], ("\n".join(checksum_lines) + "\n").encode("ascii")
    )
    return CampaignCollection(rows, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", required=True, help="authoritative one-shot plan JSON"
    )
    parser.add_argument(
        "--records", required=True, help="plan-bound run-record bundle JSON"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="complete")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    result = finalize_campaign(args.plan, args.records, args.out_dir, args.policy)
    print(
        json.dumps(
            {
                "status": "validated",
                "validation_policy": args.policy,
                "manuscript_ready": result.report["manuscript_ready"],
                "observed_scientific_rows": len(result.rows),
                "out_dir": os.path.abspath(args.out_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
