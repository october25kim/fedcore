"""Post-hoc Fed-CORE evaluation from one immutable logit artifact.

This runner never trains a model and never chooses an analysis grid.  Every
score, gamma, confidence-budget component, and deployment-mixture choice must
be supplied explicitly by the caller (normally the frozen campaign plan).

The input contract is intentionally strict.  An NPZ must contain logits,
open-set labels, client identifiers, and stable sample identifiers for the
original disjoint ``prop``, ``cert``, and ``test`` folds.  Thresholds and
confidence allocations are selected from ``prop`` only; ``cert`` supplies the
conditional certificate counts; ``test`` is evaluation-only.  A traffic-derived
mixture consumes only stable traffic identifiers and client/site identifiers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import subprocess
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from fedcore.budget import FailureBudget
from fedcore.certificate.variants import (
    JOINT_CONDITIONAL_CERTIFICATE_VARIANT,
    SUPPORTED_CERTIFICATE_VARIANTS,
    validate_certificate_variant,
)
from fedcore.campaign.artifacts import (
    ArtifactRecord,
    RunManifest,
    environment_record,
    file_sha256,
    semantic_experiment_id,
    semantic_hash,
    utc_now,
    write_immutable_manifest,
)
from fedcore.data.fedosr_split import fold_identity_fingerprint
from fedcore.mixture import rho_mixture_box
from fedcore.posthoc import evaluate_four_policies, mixture_set_from_traffic
from fedcore.sampling import (
    nested_without_replacement_indices,
    stratified_nested_without_replacement_indices,
)
from fedcore.scores import scored_views
from fedcore.seeds import SEED_NAMESPACES, SeedBundle, derive_seed


_AUDIT_FOLDS = ("prop", "cert", "test")
_ALLOWED_TRAFFIC_FIELDS = frozenset(
    {
        "traffic_sample_id",
        "traffic_client",
        "traffic_site_id",
        "traffic_group_id",
        "traffic_fp",
    }
)


@dataclass(frozen=True)
class PosthocRequest:
    """One fully explicit post-hoc analysis request.

    ``mixture_mode`` is one of ``simplex``, ``rho``, or ``traffic``.  A
    ``traffic`` request may additionally provide ``rho`` and a declared center,
    in which case the traffic confidence box is intersected with that rho box.
    """

    input_path: str
    output_path: str
    manifest_path: str
    scores: tuple[str, ...]
    gammas: tuple[float, ...]
    alpha: float
    total_delta: float
    delta_conditional_risk: float
    delta_acceptance_lower: float
    delta_acceptance_upper: float
    delta_mixture: float
    mixture_mode: str
    certificate_variant: str
    rho: Optional[float] = None
    mixture_center: Optional[tuple[float, ...]] = None
    audit_fraction: float = 1.0
    audit_redraw_index: int = 0
    traffic_sample_size: Optional[int] = None
    traffic_draw_index: int = 0


def _strict_json_value(value: Any) -> Any:
    """Return a strict-JSON value, mapping non-finite estimates to null."""

    if isinstance(value, np.ndarray):
        return [_strict_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _strict_canonical_json(value: Any) -> str:
    return json.dumps(
        _strict_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _write_immutable_text(path: str, payload: str) -> str:
    """Durably create ``path`` and refuse every overwrite."""

    destination = os.path.abspath(path)
    directory = os.path.dirname(destination) or "."
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(destination):
        raise FileExistsError(f"refusing to replace immutable output {destination}")
    fd, temporary = tempfile.mkstemp(prefix=".posthoc.", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to replace immutable output {destination}"
            ) from exc
        return destination
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _scalar(arrays: Mapping[str, np.ndarray], key: str) -> Any:
    if key not in arrays:
        raise ValueError(f"input artifact is missing required scalar {key!r}")
    value = np.asarray(arrays[key])
    if value.size != 1:
        raise ValueError(f"input scalar {key!r} must contain exactly one value")
    return value.reshape(()).item()


def _required_sha256(arrays: Mapping[str, np.ndarray], key: str) -> str:
    value = _scalar(arrays, key)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{key} must be a lowercase 64-character SHA-256 digest")
    return value


def _integer_vector(value: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be a one-dimensional integer vector")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer vector") from exc
    if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ValueError(f"{name} must contain finite integers")
    return numeric.astype(np.int64)


def _stable_ids(value: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a one-dimensional persisted string array")
    ids = raw.astype(str)
    if np.any(np.char.str_len(ids) == 0):
        raise ValueError(f"{name} contains an empty sample identifier")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{name} contains duplicate sample identifiers")
    return ids


def _fold_digest(ids: np.ndarray, clients: np.ndarray, labels: np.ndarray) -> str:
    """Outcome-committing ordered fold digest used only for provenance."""

    import hashlib

    digest = hashlib.sha256(b"fedcore.posthoc.fold-identity.v1\x00")
    for sample_id in ids.astype(str):
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    digest.update(np.asarray(clients, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(labels, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()


def _sampling_identity_digest(ids: np.ndarray, clients: np.ndarray) -> str:
    """Label-free ordered identity digest safe for audit/traffic seed context."""

    import hashlib

    digest = hashlib.sha256(b"fedcore.posthoc.sampling-identity.v1\x00")
    for sample_id in ids.astype(str):
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    digest.update(np.asarray(clients, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()


def _sample_id_digest(ids: np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha256(b"fedcore.posthoc.sample-ids.v1\x00")
    for sample_id in ids.astype(str):
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _load_input(path: str) -> tuple[dict[str, np.ndarray], str, int]:
    absolute = os.path.abspath(path)
    if not os.path.isfile(absolute):
        raise FileNotFoundError(absolute)
    digest_before = file_sha256(absolute)
    size_before = os.path.getsize(absolute)
    try:
        with np.load(absolute, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True) for name in archive.files
            }
    except ValueError as exc:
        raise ValueError(
            "input NPZ must be readable with allow_pickle=False and contain no object arrays"
        ) from exc
    if (
        file_sha256(absolute) != digest_before
        or os.path.getsize(absolute) != size_before
    ):
        raise RuntimeError("input artifact changed while it was being loaded")
    return arrays, digest_before, size_before


def _training_provenance(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], str, dict[str, int]]:
    raw_json = _scalar(arrays, "training_config_json")
    if not isinstance(raw_json, str):
        raise ValueError("training_config_json must be a persisted JSON string")
    try:
        training_config = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError("training_config_json is invalid") from exc
    if not isinstance(training_config, dict):
        raise ValueError("training_config_json must encode a JSON object")
    declared_hash = str(_scalar(arrays, "training_config_sha256"))
    actual_hash = semantic_hash(training_config)
    if declared_hash != actual_hash:
        raise ValueError(
            "training_config_sha256 does not match the canonical persisted configuration"
        )

    if "seed_ledger_json" not in arrays:
        raise ValueError(
            "completed one-shot artifacts require a replay-validated seed_ledger_json; "
            "raw training-config seed integers are legacy accounting inputs only"
        )
    ledger_json = _scalar(arrays, "seed_ledger_json")
    if not isinstance(ledger_json, str) or not ledger_json:
        raise ValueError("seed_ledger_json must be a non-empty persisted JSON string")
    seeds = SeedBundle.from_json(ledger_json).seeds
    if tuple(seeds) != SEED_NAMESPACES:
        raise ValueError("seed ledger must contain the exact canonical namespace set")
    return training_config, actual_hash, seeds


def _validate_request(request: PosthocRequest) -> None:
    if os.path.abspath(request.input_path) in {
        os.path.abspath(request.output_path),
        os.path.abspath(request.manifest_path),
    }:
        raise ValueError("input, output, and manifest paths must be distinct")
    if os.path.abspath(request.output_path) == os.path.abspath(request.manifest_path):
        raise ValueError("output and manifest paths must be distinct")
    if not request.scores or len(set(request.scores)) != len(request.scores):
        raise ValueError("scores must be a non-empty unique sequence")
    if not request.gammas or len(set(request.gammas)) != len(request.gammas):
        raise ValueError("gammas must be a non-empty unique sequence")
    if any(value not in {0.5, 0.7, 1.0} for value in request.gammas):
        raise ValueError("every gamma must be one of {0.5, 0.7, 1.0}")
    if not 0.0 < request.alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if request.mixture_mode not in {"simplex", "rho", "traffic"}:
        raise ValueError("mixture_mode must be simplex, rho, or traffic")
    validate_certificate_variant(request.certificate_variant, request.mixture_mode)
    if request.mixture_mode == "simplex":
        if request.rho is not None or request.mixture_center is not None:
            raise ValueError("simplex mode does not accept rho or mixture_center")
        if request.delta_acceptance_upper != 0.0 or request.delta_mixture != 0.0:
            raise ValueError(
                "simplex mode must not spend acceptance-upper or mixture budget"
            )
    elif request.mixture_mode == "rho":
        if request.rho is None or request.mixture_center is None:
            raise ValueError("rho mode requires rho and a declared mixture_center")
        if request.delta_acceptance_upper <= 0.0 or request.delta_mixture != 0.0:
            raise ValueError(
                "rho mode requires acceptance-upper budget and no mixture budget"
            )
    else:
        if request.delta_acceptance_upper <= 0.0 or request.delta_mixture <= 0.0:
            raise ValueError(
                "traffic mode requires acceptance-upper and mixture confidence budgets"
            )
        if (request.rho is None) != (request.mixture_center is None):
            raise ValueError(
                "traffic rho intersection requires both rho and a declared mixture_center"
            )
    if request.delta_conditional_risk <= 0.0 or request.delta_acceptance_lower <= 0.0:
        raise ValueError("risk and acceptance-lower budgets must be positive")
    if (
        not math.isfinite(request.audit_fraction)
        or not 0.0 < request.audit_fraction <= 1.0
    ):
        raise ValueError("audit_fraction must lie in (0, 1]")
    for name, value in (
        ("audit_redraw_index", request.audit_redraw_index),
        ("traffic_draw_index", request.traffic_draw_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if request.audit_fraction == 1.0 and request.audit_redraw_index != 0:
        raise ValueError("a full audit fold has no redraw; set audit_redraw_index=0")
    if request.traffic_sample_size is not None and (
        isinstance(request.traffic_sample_size, bool)
        or not isinstance(request.traffic_sample_size, int)
        or request.traffic_sample_size <= 0
    ):
        raise ValueError("traffic_sample_size must be a positive integer or omitted")
    if request.mixture_mode != "traffic" and (
        request.traffic_sample_size is not None or request.traffic_draw_index != 0
    ):
        raise ValueError("traffic subsampling options require mixture_mode=traffic")
    if request.traffic_sample_size is None and request.traffic_draw_index != 0:
        raise ValueError("traffic_draw_index requires traffic_sample_size")


def _validated_views(
    arrays: Mapping[str, np.ndarray], training_config: Mapping[str, Any]
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, str],
    int,
    str,
]:
    raw_n_clients = training_config.get("n_clients")
    if isinstance(raw_n_clients, bool) or not isinstance(raw_n_clients, int):
        raise ValueError("training_config must declare integer n_clients")
    n_clients = int(raw_n_clients)
    if n_clients <= 0:
        raise ValueError("training_config n_clients must be positive")

    fold_arrays: dict[str, dict[str, np.ndarray]] = {}
    identities: dict[str, np.ndarray] = {}
    fold_digests: dict[str, str] = {}
    sampling_identity_digests: dict[str, str] = {}
    n_classes: Optional[int] = None
    for fold in _AUDIT_FOLDS:
        required = {
            "logits": f"{fold}_logits",
            "y_open": f"{fold}_y_open",
            "client": f"{fold}_client",
            "sample_id": f"{fold}_sample_id",
        }
        missing = [key for key in required.values() if key not in arrays]
        if missing:
            raise ValueError(f"input artifact is missing {fold} fields: {missing}")
        logits = np.asarray(arrays[required["logits"]], dtype=float)
        if logits.ndim != 2 or len(logits) == 0 or logits.shape[1] < 2:
            raise ValueError(
                f"{fold}_logits must be a non-empty N x C array with C >= 2"
            )
        if np.any(~np.isfinite(logits)):
            raise ValueError(f"{fold}_logits contains non-finite values")
        if n_classes is None:
            n_classes = int(logits.shape[1])
        elif logits.shape[1] != n_classes:
            raise ValueError("all audit folds must have the same logit dimension")
        labels = _integer_vector(arrays[required["y_open"]], required["y_open"])
        clients = _integer_vector(arrays[required["client"]], required["client"])
        ids = _stable_ids(arrays[required["sample_id"]], required["sample_id"])
        if not (len(logits) == len(labels) == len(clients) == len(ids)):
            raise ValueError(f"{fold} logits/labels/clients/sample IDs do not align")
        if np.any(clients < 0) or np.any(clients >= n_clients):
            raise ValueError(
                f"{fold}_client contains an ID outside the declared roster"
            )
        if np.any((labels < -1) | (labels >= logits.shape[1])):
            raise ValueError(f"{fold}_y_open is outside the open-set label convention")
        if not np.any(labels == -1) or not np.any(labels >= 0):
            raise ValueError(
                f"{fold} must contain labeled known and unknown observations"
            )

        # New CIFAR exports also carry an index-based identity digest.  Verify it
        # when present; the generic string-ID digest below remains the cross-domain
        # contract (medical IDs are opaque strings, not integer indices).
        index_key = f"{fold}_sample_idx"
        declared_key = f"{fold}_identity_sha256"
        if index_key in arrays and declared_key in arrays:
            expected = fold_identity_fingerprint(
                arrays[index_key],
                clients,
                labels,
            )
            if str(_scalar(arrays, declared_key)) != expected:
                raise ValueError(f"{declared_key} does not match the persisted fold")

        fold_arrays[fold] = {"logits": logits, "y_open": labels, "client": clients}
        identities[fold] = ids
        fold_digests[fold] = _fold_digest(ids, clients, labels)
        sampling_identity_digests[fold] = _sampling_identity_digest(ids, clients)

    for index, left in enumerate(_AUDIT_FOLDS):
        left_ids = set(identities[left].tolist())
        for right in _AUDIT_FOLDS[index + 1 :]:
            overlap = left_ids & set(identities[right].tolist())
            if overlap:
                raise ValueError(
                    f"stable sample IDs overlap between {left} and {right}: "
                    f"{sorted(overlap)[:3]}"
                )
    combined_fold_hash = semantic_hash(
        {fold: fold_digests[fold] for fold in _AUDIT_FOLDS}
    )
    return (
        fold_arrays,
        identities,
        sampling_identity_digests,
        n_clients,
        combined_fold_hash,
    )


def _traffic_identities(
    arrays: Mapping[str, np.ndarray],
    n_clients: int,
    audit_identities: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    missing = [
        name for name in ("traffic_sample_id", "traffic_client") if name not in arrays
    ]
    if missing:
        raise ValueError(f"traffic mode requires persisted identity fields: {missing}")
    unexpected = sorted(
        key
        for key in arrays
        if key.startswith("traffic_") and key not in _ALLOWED_TRAFFIC_FIELDS
    )
    if unexpected:
        raise ValueError(
            "traffic mixture input must be identity-only; unexpected fields found: "
            f"{unexpected}"
        )
    ids = _stable_ids(arrays["traffic_sample_id"], "traffic_sample_id")
    clients = _integer_vector(arrays["traffic_client"], "traffic_client")
    if len(ids) == 0 or len(ids) != len(clients):
        raise ValueError(
            "traffic_sample_id and traffic_client must be aligned and non-empty"
        )
    if np.any(clients < 0) or np.any(clients >= n_clients):
        raise ValueError("traffic_client contains an ID outside the declared roster")
    traffic_set = set(ids.tolist())
    for fold, fold_ids in audit_identities.items():
        overlap = traffic_set & set(fold_ids.tolist())
        if overlap:
            raise ValueError(
                f"traffic sample IDs overlap the {fold} audit fold: {sorted(overlap)[:3]}"
            )
    return ids, clients


def _code_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "UNAVAILABLE"
    return result.stdout.strip() or "UNAVAILABLE"


def _assert_common_certification(
    rows: Sequence[Mapping[str, Any]], cert_size: int
) -> None:
    digests = {str(row["cert_sample_ids_sha256"]) for row in rows}
    if len(digests) != 1 or not next(iter(digests)):
        raise RuntimeError(
            "four-policy evaluation did not preserve one certification sample set"
        )
    for threshold_policy in ("global", "client_specific"):
        matching = [row for row in rows if row["threshold_policy"] == threshold_policy]
        count_pairs = {(int(row["cert_n"]), int(row["cert_k"])) for row in matching}
        if len(matching) != 2 or len(count_pairs) != 1:
            raise RuntimeError(
                "confidence-allocation policies did not reuse identical certificate counts"
            )
    if cert_size <= 0:
        raise RuntimeError("certification fold is empty")


def run_posthoc(request: PosthocRequest) -> dict[str, Any]:
    """Validate, evaluate, and durably manifest one immutable post-hoc request."""

    _validate_request(request)
    if os.path.exists(request.output_path):
        raise FileExistsError(
            f"refusing to replace immutable output {request.output_path}"
        )
    if os.path.exists(request.manifest_path):
        raise FileExistsError(
            f"refusing to replace immutable manifest {request.manifest_path}"
        )
    started_at = utc_now()
    arrays, input_sha256, input_size = _load_input(request.input_path)
    training_config, training_config_sha256, seeds = _training_provenance(arrays)
    dataset_sha256 = _required_sha256(arrays, "dataset_sha256")
    if training_config.get("dataset_sha256") != dataset_sha256:
        raise ValueError(
            "training_config dataset_sha256 must match the persisted dataset digest"
        )
    (
        fold_arrays,
        identities,
        sampling_identity_digests,
        n_clients,
        fold_sha256,
    ) = _validated_views(arrays, training_config)
    source_experiment_id = str(_scalar(arrays, "experiment_id"))

    audit_seed: Optional[int] = None
    if request.audit_fraction < 1.0:
        if "audit_draw" not in seeds:
            raise ValueError("audit subsampling requires a persisted audit_draw seed")
        audit_seed = derive_seed(
            seeds["audit_draw"],
            "audit_draw",
            experiment_id=source_experiment_id,
            sampling_identity_sha256=sampling_identity_digests["cert"],
            draw_index=request.audit_redraw_index,
        )
        selected = stratified_nested_without_replacement_indices(
            fold_arrays["cert"]["client"], [request.audit_fraction], audit_seed
        )[request.audit_fraction]
        for name in tuple(fold_arrays["cert"]):
            fold_arrays["cert"][name] = fold_arrays["cert"][name][selected]
        identities["cert"] = identities["cert"][selected]

    budget = FailureBudget(
        total=request.total_delta,
        mixture=request.delta_mixture,
        conditional_risk=request.delta_conditional_risk,
        acceptance_lower=request.delta_acceptance_lower,
        acceptance_upper=request.delta_acceptance_upper,
    )
    lambda_lower: Optional[np.ndarray] = None
    lambda_upper: Optional[np.ndarray] = None
    traffic_ids: Optional[np.ndarray] = None
    traffic_seed: Optional[int] = None
    traffic_sampling_identity_sha256: Optional[str] = None
    mixture_center = request.mixture_center
    if mixture_center is not None and len(mixture_center) != n_clients:
        raise ValueError(
            "mixture_center length does not equal the declared client roster"
        )
    if request.mixture_mode == "rho":
        mixture = rho_mixture_box(mixture_center, request.rho)  # type: ignore[arg-type]
        lambda_lower, lambda_upper = mixture.lower, mixture.upper
    elif request.mixture_mode == "traffic":
        traffic_ids, traffic_clients = _traffic_identities(
            arrays, n_clients, identities
        )
        traffic_sampling_identity_sha256 = _sampling_identity_digest(
            traffic_ids, traffic_clients
        )
        if request.traffic_sample_size is not None:
            if request.traffic_sample_size > len(traffic_ids):
                raise ValueError(
                    "traffic_sample_size exceeds the frozen traffic reservoir"
                )
            if "traffic_draw" not in seeds:
                raise ValueError(
                    "traffic subsampling requires a persisted traffic_draw seed"
                )
            traffic_seed = derive_seed(
                seeds["traffic_draw"],
                "traffic_draw",
                experiment_id=source_experiment_id,
                sampling_identity_sha256=traffic_sampling_identity_sha256,
                draw_index=request.traffic_draw_index,
            )
            selected = nested_without_replacement_indices(
                len(traffic_ids), [request.traffic_sample_size], traffic_seed
            )[request.traffic_sample_size]
            traffic_ids = traffic_ids[selected]
            traffic_clients = traffic_clients[selected]
        mixture = mixture_set_from_traffic(
            traffic_clients,
            n_clients=n_clients,
            mixture_delta=budget.mixture,
            rho=request.rho,
            center=mixture_center,
        )
        lambda_lower, lambda_upper = mixture.lower, mixture.upper

    score_views = {
        fold: scored_views(
            fold_arrays[fold]["logits"],
            fold_arrays[fold]["y_open"],
            fold_arrays[fold]["client"],
            list(request.scores),
        )
        for fold in _AUDIT_FOLDS
    }
    dirichlet_alpha = training_config.get("dirichlet_alpha")
    rows: list[dict[str, Any]] = []
    for score_name in request.scores:
        for gamma in request.gammas:
            cell_rows = evaluate_four_policies(
                score_views["prop"][score_name],
                score_views["cert"][score_name],
                score_views["test"][score_name],
                score_name=score_name,
                gamma=float(gamma),
                alpha=request.alpha,
                budget=budget,
                n_clients=n_clients,
                dirichlet_alpha=dirichlet_alpha,  # type: ignore[arg-type]
                Lambda=("simplex" if request.mixture_mode == "simplex" else "bounded"),
                lambda_lower=lambda_lower,
                lambda_upper=lambda_upper,
                cert_sample_ids=identities["cert"],
            )
            _assert_common_certification(cell_rows, len(identities["cert"]))
            rows.extend(cell_rows)

    dataset_value = _scalar(arrays, "dataset")
    if not isinstance(dataset_value, str) or not dataset_value:
        raise ValueError("dataset must be a non-empty persisted string")
    dataset = dataset_value
    posthoc_config: dict[str, Any] = {
        "schema_version": 1,
        "source_experiment_id": source_experiment_id,
        "source_artifact_sha256": input_sha256,
        "scores": list(request.scores),
        "gammas": list(request.gammas),
        "alpha": request.alpha,
        "failure_budget": budget.as_dict(),
        "mixture_mode": request.mixture_mode,
        "certificate_variant": request.certificate_variant,
        "rho": request.rho,
        "mixture_center": list(mixture_center) if mixture_center is not None else None,
        "lambda_lower": lambda_lower.tolist() if lambda_lower is not None else None,
        "lambda_upper": lambda_upper.tolist() if lambda_upper is not None else None,
        "selection_source": "proposal_only",
        "allocation_source": "proposal_only",
        "certification_source": "certification_only",
        "test_usage": "evaluation_only",
        "traffic_usage": (
            "identity_only" if request.mixture_mode == "traffic" else "unused"
        ),
        "audit_fraction": request.audit_fraction,
        "audit_redraw_index": request.audit_redraw_index,
        "audit_sampling_seed": audit_seed,
        "cert_sampling_identity_sha256": sampling_identity_digests["cert"],
        "audit_sample_ids_sha256": _sample_id_digest(identities["cert"]),
        "traffic_sample_size": request.traffic_sample_size,
        "traffic_draw_index": request.traffic_draw_index,
        "traffic_sampling_seed": traffic_seed,
        "traffic_sampling_identity_sha256": traffic_sampling_identity_sha256,
        "traffic_sample_ids_sha256": (
            _sample_id_digest(traffic_ids) if traffic_ids is not None else None
        ),
    }
    posthoc_config_sha256 = semantic_hash(posthoc_config)
    analysis_identity = {
        "training_config_sha256": training_config_sha256,
        "fold_sha256": fold_sha256,
        "posthoc_config_sha256": posthoc_config_sha256,
    }
    posthoc_experiment_id = semantic_experiment_id(
        f"{source_experiment_id}-posthoc", analysis_identity
    )
    common = {
        "source_experiment_id": source_experiment_id,
        "posthoc_experiment_id": posthoc_experiment_id,
        "input_artifact_sha256": input_sha256,
        "training_config_sha256": training_config_sha256,
        "fold_sha256": fold_sha256,
        "posthoc_config_sha256": posthoc_config_sha256,
        "mixture_mode": request.mixture_mode,
        "certificate_variant": request.certificate_variant,
        "rho": request.rho,
        "audit_budget_fraction": request.audit_fraction,
        "audit_redraw_index": request.audit_redraw_index,
        "traffic_sample_size": len(traffic_ids) if traffic_ids is not None else 0,
        "traffic_draw_index": request.traffic_draw_index,
        "traffic_sample_ids_sha256": (
            _sample_id_digest(traffic_ids) if traffic_ids is not None else ""
        ),
        "cert_observation_count": len(identities["cert"]),
        "traffic_observation_count": len(traffic_ids) if traffic_ids is not None else 0,
    }
    for row in rows:
        row.update(common)

    # Recheck the complete source artifact after all numerical work and before
    # making any result durable.  A changed input can never receive a manifest.
    if (
        file_sha256(request.input_path) != input_sha256
        or os.path.getsize(request.input_path) != input_size
    ):
        raise RuntimeError("input artifact changed during post-hoc evaluation")

    result = {
        "schema_version": 1,
        "artifact_type": "fedcore.oneshot.posthoc-results",
        "status": "completed",
        "dataset": dataset,
        "source_experiment_id": source_experiment_id,
        "posthoc_experiment_id": posthoc_experiment_id,
        "input_artifact": {
            "path": os.path.abspath(request.input_path),
            "sha256": input_sha256,
            "size_bytes": input_size,
        },
        "training_config_sha256": training_config_sha256,
        "fold_sha256": fold_sha256,
        "posthoc_config": posthoc_config,
        "posthoc_config_sha256": posthoc_config_sha256,
        "row_count": len(rows),
        "rows": rows,
    }
    output_path = _write_immutable_text(
        request.output_path, _strict_canonical_json(result) + "\n"
    )
    ended_at = utc_now()
    manifest = RunManifest(
        schema_version=1,
        experiment_id=posthoc_experiment_id,
        status="completed",
        training_config=training_config,
        posthoc_config={
            **posthoc_config,
            "posthoc_config_sha256": posthoc_config_sha256,
        },
        seeds=seeds,
        config_hash=training_config_sha256,
        code_commit=_code_commit(),
        dataset_hash=dataset_sha256,
        fold_hash=fold_sha256,
        started_at=started_at,
        ended_at=ended_at,
        checkpoint_path="",
        stdout_path="",
        stderr_path="",
        artifacts=(
            ArtifactRecord.from_path(request.input_path, "immutable_model_outputs"),
            ArtifactRecord.from_path(output_path, "posthoc_results"),
        ),
        environment=environment_record(),
    )
    write_immutable_manifest(request.manifest_path, manifest)
    return _strict_json_value(result)


def _center_json(value: Optional[str]) -> Optional[tuple[float, ...]]:
    if value is None:
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("mixture center must be a JSON array") from exc
    if not isinstance(decoded, list) or not decoded:
        raise argparse.ArgumentTypeError(
            "mixture center must be a non-empty JSON array"
        )
    try:
        return tuple(float(item) for item in decoded)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "mixture center entries must be numbers"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="immutable NPZ logit artifact")
    parser.add_argument("--output", required=True, help="new immutable result JSON")
    parser.add_argument(
        "--manifest",
        default=None,
        help="new immutable manifest JSON (default: OUTPUT.manifest.json)",
    )
    parser.add_argument("--score", dest="scores", action="append", required=True)
    parser.add_argument(
        "--gamma", dest="gammas", action="append", required=True, type=float
    )
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--total-delta", required=True, type=float)
    parser.add_argument("--delta-conditional-risk", required=True, type=float)
    parser.add_argument("--delta-acceptance-lower", required=True, type=float)
    parser.add_argument("--delta-acceptance-upper", type=float, default=0.0)
    parser.add_argument("--delta-mixture", type=float, default=0.0)
    parser.add_argument(
        "--mixture-mode",
        choices=("simplex", "rho", "traffic"),
        required=True,
    )
    parser.add_argument(
        "--certificate-variant",
        choices=tuple(sorted(SUPPORTED_CERTIFICATE_VARIANTS)),
        required=True,
    )
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument(
        "--mixture-center-json",
        default=None,
        help="declared center as a JSON probability array; required with rho",
    )
    parser.add_argument("--audit-fraction", type=float, default=1.0)
    parser.add_argument("--audit-redraw-index", type=int, default=0)
    parser.add_argument("--traffic-sample-size", type=int, default=None)
    parser.add_argument("--traffic-draw-index", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    output = os.path.abspath(args.output)
    request = PosthocRequest(
        input_path=os.path.abspath(args.input),
        output_path=output,
        manifest_path=os.path.abspath(args.manifest or f"{output}.manifest.json"),
        scores=tuple(args.scores),
        gammas=tuple(args.gammas),
        alpha=args.alpha,
        total_delta=args.total_delta,
        delta_conditional_risk=args.delta_conditional_risk,
        delta_acceptance_lower=args.delta_acceptance_lower,
        delta_acceptance_upper=args.delta_acceptance_upper,
        delta_mixture=args.delta_mixture,
        mixture_mode=args.mixture_mode,
        certificate_variant=args.certificate_variant,
        rho=args.rho,
        mixture_center=_center_json(args.mixture_center_json),
        audit_fraction=args.audit_fraction,
        audit_redraw_index=args.audit_redraw_index,
        traffic_sample_size=args.traffic_sample_size,
        traffic_draw_index=args.traffic_draw_index,
    )
    result = run_posthoc(request)
    print(
        _strict_canonical_json(
            {
                "status": result["status"],
                "posthoc_experiment_id": result["posthoc_experiment_id"],
                "row_count": result["row_count"],
                "output": output,
                "manifest": request.manifest_path,
            }
        )
    )


if __name__ == "__main__":
    main()
