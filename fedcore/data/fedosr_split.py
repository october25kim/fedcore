"""Open-set splitting, Dirichlet non-IID partitioning, and calibration folds.

Three pieces:

* :func:`open_set_split` -- pick ``n_known`` known classes, treat the rest as
  unknown, and remap known classes to ``[0, n_known)``.
* :func:`dirichlet_partition` -- partition known TRAIN indices across clients so
  each client's label distribution is ``Dirichlet(alpha)`` (smaller alpha = more
  non-IID).
* :func:`build_calibration` -- build per-client trusted ``{prop, cert, test}``
  folds, each with clean known points (true remapped label) and injected
  unknowns (``y_open = -1``). Calibration/test STAY CLEAN.
* :func:`build_identity_only_traffic` -- reserve a disjoint, label-free set of
  sample identities and simulated deployment-client identities.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np


def _probability_vector(values, dimension: int, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape != (dimension,):
        raise ValueError(f"{name} must have length {dimension}")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must sum to one")
    return probabilities / probabilities.sum()


def build_identity_only_traffic(
    candidate_idx,
    *,
    n_traffic: int,
    n_clients: int,
    client_probabilities,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Reserve traffic identities without accepting labels or predictions.

    The API deliberately has no ``y`` or logit argument. It first takes a
    without-replacement prefix of one candidate-identity permutation, then draws
    a deployment client for each identity from the predeclared mixture. Callers
    remove the returned identities before constructing labeled audit folds.
    """

    identities = np.asarray(candidate_idx)
    if identities.ndim != 1 or len(set(identities.tolist())) != len(identities):
        raise ValueError(
            "candidate_idx must be a one-dimensional unique identity vector"
        )
    if isinstance(n_traffic, bool) or not isinstance(n_traffic, (int, np.integer)):
        raise ValueError("n_traffic must be an integer")
    n_traffic = int(n_traffic)
    if n_traffic < 0 or n_traffic > len(identities):
        raise ValueError("n_traffic must lie in [0, len(candidate_idx)]")
    if (
        isinstance(n_clients, bool)
        or not isinstance(n_clients, (int, np.integer))
        or int(n_clients) <= 0
    ):
        raise ValueError("n_clients must be a positive integer")
    n_clients = int(n_clients)
    probabilities = _probability_vector(
        client_probabilities, n_clients, "client_probabilities"
    )
    rng = np.random.default_rng(int(seed))
    selected = identities[rng.permutation(len(identities))[:n_traffic]].copy()
    clients = rng.choice(n_clients, size=n_traffic, p=probabilities).astype(np.int64)
    return {"idx": selected, "client": clients}


def open_set_split(
    labels, n_known: int, seed: int, unknown_classes=None
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """Choose known/unknown classes and remap knowns to ``[0, n_known)``.

    Returns ``(known_classes, unknown_classes, remap)`` where ``remap`` maps each
    original known class id to its contiguous index in ``[0, n_known)``.

    If ``unknown_classes`` is given (an explicit iterable of class ids), it FIXES
    the open-set split independently of ``seed`` (the known set is the complement).
    This lets a robustness study hold a pre-declared unknown-class set constant
    across training seeds. Default ``None`` preserves the seed-driven random split.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if unknown_classes is not None:
        unknown_classes = np.sort(
            np.asarray(list(unknown_classes), dtype=classes.dtype)
        )
        known_classes = np.sort(
            np.array(
                [c for c in classes if c not in set(unknown_classes.tolist())],
                dtype=classes.dtype,
            )
        )
        if len(known_classes) != n_known:
            raise ValueError(
                f"explicit unknown_classes leaves {len(known_classes)} known "
                f"classes but n_known={n_known}"
            )
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(classes)
        known_classes = np.sort(perm[:n_known])
        unknown_classes = np.sort(perm[n_known:])
    remap = {int(c): i for i, c in enumerate(known_classes)}
    return known_classes, unknown_classes, remap


def dirichlet_partition(
    indices,
    labels_remapped,
    n_clients: int,
    alpha: float,
    seed: int,
) -> List[np.ndarray]:
    """Partition ``indices`` across ``n_clients`` with per-client Dirichlet labels.

    ``indices`` and ``labels_remapped`` are aligned arrays (the remapped label of
    ``indices[i]`` is ``labels_remapped[i]``). For each class, its points are
    split across clients by a ``Dirichlet(alpha)`` draw.
    """
    indices = np.asarray(indices)
    labels_remapped = np.asarray(labels_remapped)
    rng = np.random.default_rng(seed)
    n_known = int(labels_remapped.max()) + 1 if len(labels_remapped) else 0

    client_idx: List[List[int]] = [[] for _ in range(n_clients)]
    for c in range(n_known):
        cls_pos = np.where(labels_remapped == c)[0]
        if len(cls_pos) == 0:
            continue
        rng.shuffle(cls_pos)
        props = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(props) * len(cls_pos)).astype(int)[:-1]
        chunks = np.split(cls_pos, cuts)
        for j in range(n_clients):
            client_idx[j].extend(indices[chunks[j]].tolist())

    return [np.array(sorted(c), dtype=int) for c in client_idx]


def build_calibration(
    known_idx,
    known_y_remapped,
    unknown_idx,
    n_clients: int,
    folds: Tuple[float, float, float],
    unknown_contamination: float,
    seed: int,
) -> List[Dict[str, Dict[str, np.ndarray]]]:
    """Per-client trusted ``{prop, cert, test}`` folds with injected unknowns.

    Each client receives a slice of the clean known calibration points (with
    their true remapped labels) plus enough unknowns (label ``-1``) so that the
    unknown fraction of each fold is ``unknown_contamination``.

    Returns a list (one per client) of ``{fold: {"idx", "y_open"}}``.
    """
    rng = np.random.default_rng(seed)
    known_idx = np.asarray(known_idx)
    known_y = np.asarray(known_y_remapped)
    unk = np.asarray(unknown_idx).copy()
    rng.shuffle(unk)

    perm = rng.permutation(len(known_idx))
    known_idx = known_idx[perm]
    known_y = known_y[perm]

    client_pos = np.array_split(np.arange(len(known_idx)), n_clients)
    fold_names = ("prop", "cert", "test")
    pf, cf, _tf = folds
    contam = unknown_contamination

    result: List[Dict[str, Dict[str, np.ndarray]]] = []
    unk_ptr = 0
    for j in range(n_clients):
        pos = client_pos[j]
        kidx = known_idx[pos]
        ky = known_y[pos]
        n = len(pos)
        n_prop = int(round(n * pf))
        n_cert = int(round(n * cf))
        bounds = [(0, n_prop), (n_prop, n_prop + n_cert), (n_prop + n_cert, n)]

        client_folds: Dict[str, Dict[str, np.ndarray]] = {}
        for (s, e), fn in zip(bounds, fold_names):
            fk_idx = kidx[s:e]
            fk_y = ky[s:e]
            n_k = len(fk_idx)
            if contam >= 1.0:
                n_unk = n_k
            else:
                n_unk = int(round(contam / (1.0 - contam) * n_k))
            take = unk[unk_ptr : unk_ptr + n_unk]
            if len(take) != n_unk:
                raise ValueError(
                    "insufficient unknown observations for the declared calibration "
                    "contamination and fold sizes"
                )
            unk_ptr += len(take)

            idxs = np.concatenate([fk_idx, take]).astype(int)
            yopen = np.concatenate([fk_y, -np.ones(len(take), dtype=int)]).astype(int)
            client_folds[fn] = {"idx": idxs, "y_open": yopen}
        result.append(client_folds)

    return result


# --------------------------------------------------------------------------- #
# split provenance (cross-environment drift detection)
# --------------------------------------------------------------------------- #
def fold_fingerprint(client, y_open) -> str:
    """Deterministic 12-hex digest of a fold's ``(client, y_open)`` assignment.

    Two exports of the SAME ``(dataset, dirichlet_alpha, seed)`` must produce the
    same fingerprint. A mismatch flags fold-split drift -- e.g. numpy RNG-stream
    changes (``Generator.dirichlet`` / ``shuffle``) across environments, which a
    pinned seed alone does NOT prevent.
    """
    h = hashlib.md5()
    h.update(np.ascontiguousarray(np.asarray(client)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(y_open)).tobytes())
    return h.hexdigest()[:12]


def fold_identity_fingerprint(sample_idx, client, y_open) -> str:
    """SHA-256 digest of an ordered fold including immutable sample identity.

    ``fold_fingerprint`` remains as the legacy cross-environment oracle.  New
    artifacts additionally carry this stronger digest, which detects swapping two
    same-label examples and therefore commits to the actual audit observations.
    """
    arrays = (
        np.asarray(sample_idx, dtype=np.int64),
        np.asarray(client, dtype=np.int64),
        np.asarray(y_open, dtype=np.int64),
    )
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("sample_idx, client, and y_open must be aligned")
    h = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        h.update(str(contiguous.shape).encode("ascii"))
        h.update(contiguous.tobytes())
    return h.hexdigest()


def add_split_fingerprint(raw: Dict[str, np.ndarray], seed: int) -> None:
    """Attach split-provenance metadata to an npz payload dict, in place.

    ``raw`` must already carry ``{fold}_client`` and ``{fold}_y_open`` for each
    audit fold. Adds ``split_seed``, ``numpy_version`` and a per-fold ``{fold}_fp``
    so a consumer can recompute the fingerprint from the stored folds and detect
    divergence. Supports the single-export-source rule: ws4090 is the only
    exporter; the laptop never re-exports an existing run.
    """
    raw["split_seed"] = np.asarray(int(seed))
    raw["numpy_version"] = np.asarray(np.__version__)
    for fold in ("prop", "cert", "test"):
        raw[f"{fold}_fp"] = np.asarray(
            fold_fingerprint(raw[f"{fold}_client"], raw[f"{fold}_y_open"])
        )
        sample_key = f"{fold}_sample_idx"
        if sample_key in raw:
            raw[f"{fold}_identity_sha256"] = np.asarray(
                fold_identity_fingerprint(
                    raw[sample_key], raw[f"{fold}_client"], raw[f"{fold}_y_open"]
                )
            )
