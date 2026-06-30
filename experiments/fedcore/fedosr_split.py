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
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def open_set_split(
    labels, n_known: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """Choose known/unknown classes and remap knowns to ``[0, n_known)``.

    Returns ``(known_classes, unknown_classes, remap)`` where ``remap`` maps each
    original known class id to its contiguous index in ``[0, n_known)``.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
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
            take = unk[unk_ptr:unk_ptr + n_unk]
            unk_ptr += len(take)

            idxs = np.concatenate([fk_idx, take]).astype(int)
            yopen = np.concatenate(
                [fk_y, -np.ones(len(take), dtype=int)]
            ).astype(int)
            client_folds[fn] = {"idx": idxs, "y_open": yopen}
        result.append(client_folds)

    return result
