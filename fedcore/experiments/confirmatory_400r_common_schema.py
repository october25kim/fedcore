"""Confirmatory-400R common per-observation export schema (license-neutral).

This module is a pure numpy parser/assembler. It defines ONE canonical per-obs
schema that every training path (CIFAR PROSER-FedAvg, Office-Home ConvNeXt
full-FT and frozen-linear) must emit, so the four smoke artifacts can be
compared field-for-field. It contains NO training code and NO GPL-derived
PROSER surgery -- only arithmetic over already-exported known-class logits plus
a precomputed family-native accept score. It may therefore live in fedcore2.

Canonical fields (identical NAMES and DTYPES across all four paths):

    immutable_source_id            str      opaque, per-observation identity
    client_id                      int64    federated client index
    fold_role                      str      proposal | certification | test/evaluation
    true_global_class              int64    original dataset class id (>=0)
    true_known_class_index_or_neg1 int64    remapped known index, or -1 if unknown
    known_or_unknown               str      "known" | "unknown"
    known_logits                   float32  (N, n_known); WIDTH varies by n_known
    predicted_known_index          int64    argmax over known_logits
    energy_score                   float32  fedcore.scores.energy (higher => ID)
    known_margin_score             float32  fedcore.scores.margin (higher => ID)
    native_score                   float32  family-native accept score (higher => ID)

Per-fold scalar metadata (identical names across paths):

    native_score_name  str   e.g. "proser_dummy_vs_known" (CIFAR) or "msp" (OH)
    n_known            int64  known-class count (explains known_logits width)
    family             str    "cifar_proser_fedavg" | "officehome_convnext_*"

No field is named only ``logit`` or ``score``; every score column carries an
explicit, unambiguous name (``energy_score``, ``known_margin_score``,
``native_score`` + ``native_score_name``), and the logits are a named 2-D
``known_logits`` matrix.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from fedcore.scores import energy as _energy, margin as _margin

# Ordered per-observation field names and their canonical numpy dtype *kinds*.
# ('U' = unicode string of any width; 'i8' = int64; 'f4' = float32; 'f4-2d' =
# float32 matrix). Width of unicode / of the known_logits 2nd axis is allowed to
# differ across paths; the KIND and the name must match.
PER_OBS_FIELDS: tuple[tuple[str, str], ...] = (
    ("immutable_source_id", "U"),
    ("client_id", "i8"),
    ("fold_role", "U"),
    ("true_global_class", "i8"),
    ("true_known_class_index_or_neg1", "i8"),
    ("known_or_unknown", "U"),
    ("known_logits", "f4-2d"),
    ("predicted_known_index", "i8"),
    ("energy_score", "f4"),
    ("known_margin_score", "f4"),
    ("native_score", "f4"),
)

SCALAR_META_FIELDS: tuple[tuple[str, str], ...] = (
    ("native_score_name", "U"),
    ("n_known", "i8"),
    ("family", "U"),
)

FOLD_ROLES: tuple[str, ...] = ("proposal", "certification", "test")


def cifar_proser_accept_score(known_logits: np.ndarray, dummy_logit: np.ndarray) -> np.ndarray:
    """PROSER dummy-vs-known accept score (higher => more likely a known class).

    FedPD's native PROSER open-set confidence is
    ``conf = softmax([known, max_dummy])[unknown] - max_k softmax[known]``
    (higher => more OOD). The Fed-CORE accept orientation is ``-conf``. This is
    plain arithmetic over the already-exported known logits and the (single)
    dummy logit; it is NOT the GPL PROSER training surgery.
    """
    known = np.asarray(known_logits, dtype=np.float64)
    dummy = np.asarray(dummy_logit, dtype=np.float64).reshape(-1, 1)
    total = np.concatenate([known, dummy], axis=1)
    z = total - total.max(axis=1, keepdims=True)
    e = np.exp(z)
    sm = e / e.sum(axis=1, keepdims=True)
    dummy_conf = sm[:, -1]
    max_known = sm[:, :known.shape[1]].max(axis=1)
    conf = dummy_conf - max_known           # higher => OOD
    return (-conf).astype(np.float32)       # higher => ID (accept)


def assemble_fold(
    *,
    immutable_source_id: np.ndarray,
    client_id: np.ndarray,
    fold_role: str,
    true_global_class: np.ndarray,
    true_known_class_index_or_neg1: np.ndarray,
    known_logits: np.ndarray,
    native_score: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble one fold's canonical per-obs arrays (all length N, ID-aligned)."""
    known_logits = np.asarray(known_logits, dtype=np.float32)
    if known_logits.ndim != 2:
        raise ValueError("known_logits must be (N, n_known)")
    n = known_logits.shape[0]
    y = np.asarray(true_known_class_index_or_neg1, dtype=np.int64)
    for name, arr in (
        ("immutable_source_id", immutable_source_id),
        ("client_id", client_id),
        ("true_global_class", true_global_class),
        ("true_known_class_index_or_neg1", y),
        ("native_score", native_score),
    ):
        if len(np.asarray(arr)) != n:
            raise ValueError(f"{name} length {len(np.asarray(arr))} != {n}")
    known_or_unknown = np.where(y >= 0, "known", "unknown").astype(str)
    predicted = known_logits.argmax(axis=1).astype(np.int64)
    energy = _energy(known_logits).astype(np.float32)
    margin_s = _margin(known_logits).astype(np.float32)
    return {
        "immutable_source_id": np.asarray(immutable_source_id, dtype=str),
        "client_id": np.asarray(client_id, dtype=np.int64),
        "fold_role": np.asarray([str(fold_role)] * n, dtype=str),
        "true_global_class": np.asarray(true_global_class, dtype=np.int64),
        "true_known_class_index_or_neg1": y,
        "known_or_unknown": known_or_unknown,
        "known_logits": known_logits,
        "predicted_known_index": predicted,
        "energy_score": energy,
        "known_margin_score": margin_s,
        "native_score": np.asarray(native_score, dtype=np.float32),
    }


def pack_common_npz(
    folds: Mapping[str, Mapping[str, np.ndarray]],
    *,
    native_score_name: str,
    n_known: int,
    family: str,
) -> dict[str, np.ndarray]:
    """Flatten per-fold arrays into a single npz payload (``<role>__<field>``)."""
    payload: dict[str, np.ndarray] = {}
    for role, arrays in folds.items():
        present = {name for name, _ in PER_OBS_FIELDS}
        missing = present - set(arrays)
        if missing:
            raise ValueError(f"fold {role!r} missing fields {sorted(missing)}")
        for name, _kind in PER_OBS_FIELDS:
            payload[f"{role}__{name}"] = arrays[name]
    payload["native_score_name"] = np.asarray(str(native_score_name))
    payload["n_known"] = np.asarray(int(n_known), dtype=np.int64)
    payload["family"] = np.asarray(str(family))
    payload["schema_id"] = np.asarray("confirmatory_400r_common_obs_v1")
    payload["fold_roles"] = np.asarray(list(folds), dtype=str)
    return payload


def _dtype_kind(arr: np.ndarray) -> str:
    a = np.asarray(arr)
    if a.dtype.kind in ("U", "S"):
        return "U"
    if a.ndim == 2 and a.dtype.kind == "f":
        return "f4-2d"
    if a.dtype.kind == "f":
        return "f4"
    if a.dtype.kind in ("i", "u"):
        return "i8"
    return a.dtype.kind


def describe_common_npz(path: str) -> dict:
    """Return a field->(kind, shape) description for the schema comparison."""
    data = np.load(path, allow_pickle=False)
    roles = [str(r) for r in data["fold_roles"]]
    per_field: dict[str, dict] = {}
    for name, _kind in PER_OBS_FIELDS:
        key = f"{roles[0]}__{name}"
        arr = data[key]
        per_field[name] = {
            "kind": _dtype_kind(arr),
            "dtype": str(arr.dtype),
            "ndim": int(np.asarray(arr).ndim),
            "width": int(arr.shape[1]) if np.asarray(arr).ndim == 2 else None,
        }
    meta = {
        "native_score_name": str(data["native_score_name"]),
        "n_known": int(data["n_known"]),
        "family": str(data["family"]),
        "schema_id": str(data["schema_id"]),
        "roles": roles,
    }
    return {"per_field": per_field, "meta": meta}


def normalize_officehome_to_common(
    native_path: str, out_path: str, *, family: str
) -> str:
    """Map a run_officehome native artifact into the common per-obs schema.

    The Office-Home runner emits per-output-role arrays (``prop_*``/``cert_*``/
    ``eval_*``) with logits, ``y_open``, ``client``, ``sample_id``, ``klass``.
    This reads them (no torch), computes the shared/native scores, and re-emits
    the canonical common-schema npz with fold roles ``proposal``/``certification``
    /``test`` (Office-Home's ``evaluation`` fold maps to the common ``test``
    role). Native score = MSP (higher => more likely a known class).
    """
    from fedcore.scores import msp as _msp

    data = np.load(native_path, allow_pickle=False)
    known = [str(c) for c in np.asarray(data["known_classes"], dtype=str)]
    unknown = [str(c) for c in np.asarray(data["unknown_classes"], dtype=str)]
    vocab = sorted(set(known) | set(unknown))
    class_to_global = {c: i for i, c in enumerate(vocab)}
    n_known = int(len(known))

    role_map = (("prop", "proposal"), ("cert", "certification"), ("eval", "test"))
    folds: dict[str, dict] = {}
    for src_role, common_role in role_map:
        logits = np.asarray(data[f"{src_role}_logits"], dtype=np.float32)
        y_open = np.asarray(data[f"{src_role}_y_open"], dtype=np.int64)
        client = np.asarray(data[f"{src_role}_client"], dtype=np.int64)
        sample_id = np.asarray(data[f"{src_role}_sample_id"], dtype=str)
        klass = np.asarray(data[f"{src_role}_klass"], dtype=str)
        true_global = np.asarray([class_to_global[str(k)] for k in klass], dtype=np.int64)
        native = _msp(logits).astype(np.float32)
        folds[common_role] = assemble_fold(
            immutable_source_id=sample_id,
            client_id=client,
            fold_role=common_role,
            true_global_class=true_global,
            true_known_class_index_or_neg1=y_open,
            known_logits=logits,
            native_score=native,
        )
    payload = pack_common_npz(
        folds, native_score_name="msp", n_known=n_known, family=family)
    payload["dataset"] = np.asarray("officehome")
    payload["split_id"] = np.asarray(str(data["split_id"]))
    import os
    tmp = f"{out_path}.tmp.{os.getpid()}.npz"  # .npz so savez does not re-suffix
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, out_path)
    os.chmod(out_path, 0o644)
    return out_path


def load_fold_counts_inputs(path: str, role: str):
    """Return ``(native_score, energy, margin, y_open, client, known_logits)``.

    Consumers build per-candidate accept/error counts for the certification
    dry-run from these; ``native_score`` is the family-native accept score.
    """
    data = np.load(path, allow_pickle=False)
    y = np.asarray(data[f"{role}__true_known_class_index_or_neg1"], dtype=np.int64)
    return {
        "native_score": np.asarray(data[f"{role}__native_score"], dtype=np.float64),
        "energy_score": np.asarray(data[f"{role}__energy_score"], dtype=np.float64),
        "known_margin_score": np.asarray(data[f"{role}__known_margin_score"], dtype=np.float64),
        "y_open": y,
        "client": np.asarray(data[f"{role}__client_id"], dtype=np.int64),
        "predicted_known_index": np.asarray(data[f"{role}__predicted_known_index"], dtype=np.int64),
        "known_logits": np.asarray(data[f"{role}__known_logits"], dtype=np.float64),
    }
