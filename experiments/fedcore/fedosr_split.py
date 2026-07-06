"""Open-set + non-IID partition utilities for FedOSR (dataset-agnostic, numpy).

Operates on a 1-D array of integer class labels, so it works for CIFAR-10/100 or
any classification dataset. Produces:
  * an open-set class split (known vs. held-out unknown classes),
  * a Dirichlet non-IID partition of the KNOWN-class training data across clients,
  * a per-client trusted calibration pool (known points + injected unknown-class
    points) split into proposal / certification / test folds.

Open-set label convention used downstream: a point's ``y_open`` is its known
class index in [0, n_known) after remapping, or -1 if it is an unknown-class point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OpenSetSplit:
    known_classes: np.ndarray      # original class ids treated as "known"
    unknown_classes: np.ndarray    # original class ids held out as "unknown"
    remap: dict[int, int]          # original known id -> contiguous [0, n_known)


def open_set_split(labels: np.ndarray, n_known: int, seed: int) -> OpenSetSplit:
    """Choose ``n_known`` classes as known; the rest are unknown."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    perm = rng.permutation(classes)
    known = np.sort(perm[:n_known])
    unknown = np.sort(perm[n_known:])
    remap = {int(c): i for i, c in enumerate(known)}
    return OpenSetSplit(known_classes=known, unknown_classes=unknown, remap=remap)


def dirichlet_partition(
    indices: np.ndarray,
    labels_remapped: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int,
) -> list[np.ndarray]:
    """Partition known-class TRAIN ``indices`` across clients by label Dirichlet(alpha).

    ``labels_remapped`` are the contiguous known labels aligned with ``indices``.
    Smaller ``alpha`` => more heterogeneous (skewed) client label distributions.
    """
    rng = np.random.default_rng(seed)
    n_class = int(labels_remapped.max()) + 1
    client_bins: list[list[int]] = [[] for _ in range(n_clients)]
    for c in range(n_class):
        idx_c = indices[labels_remapped == c]
        rng.shuffle(idx_c)
        props = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(props)[:-1] * len(idx_c)).astype(int)
        for j, part in enumerate(np.split(idx_c, cuts)):
            client_bins[j].extend(part.tolist())
    return [np.array(sorted(b), dtype=int) for b in client_bins]


def build_calibration(
    known_idx: np.ndarray,
    known_y_remapped: np.ndarray,
    unknown_idx: np.ndarray,
    n_clients: int,
    folds: tuple[float, float, float],
    unknown_contamination: float,
    seed: int,
) -> list[dict[str, dict[str, np.ndarray]]]:
    """Build a per-client trusted calibration pool split into prop/cert/test.

    Each client receives a share of held-out known points (with true remapped
    labels) plus a contamination of unknown-class points (label -1). Returns a
    list (per client) of dicts: fold_name -> {'idx': dataset indices,
    'y_open': open-set labels aligned with idx}.

    The returned dataset indices are used to export logits from a trained model.
    """
    rng = np.random.default_rng(seed)
    known_idx = known_idx.copy(); rng.shuffle(known_idx)
    unknown_idx = unknown_idx.copy(); rng.shuffle(unknown_idx)

    # map dataset index -> remapped known label for quick lookup
    y_lookup = {int(i): int(y) for i, y in zip(known_idx, known_y_remapped)}

    known_per_client = np.array_split(known_idx, n_clients)
    # size unknown share so that unknowns are ~unknown_contamination of each client
    out: list[dict[str, dict[str, np.ndarray]]] = []
    fr_prop, fr_cert, fr_test = folds
    u_ptr = 0
    for j in range(n_clients):
        kc = known_per_client[j]
        n_known_pts = len(kc)
        n_unk = int(round(unknown_contamination / (1 - unknown_contamination) * n_known_pts))
        uc = unknown_idx[u_ptr:u_ptr + n_unk]; u_ptr += n_unk

        idx = np.concatenate([kc, uc])
        y_open = np.concatenate([
            np.array([y_lookup[int(i)] for i in kc], dtype=int),
            np.full(len(uc), -1, dtype=int),
        ])
        order = rng.permutation(len(idx))
        idx, y_open = idx[order], y_open[order]

        n = len(idx)
        c1 = int(fr_prop * n); c2 = c1 + int(fr_cert * n)
        client_folds = {
            "prop": {"idx": idx[:c1], "y_open": y_open[:c1]},
            "cert": {"idx": idx[c1:c2], "y_open": y_open[c1:c2]},
            "test": {"idx": idx[c2:], "y_open": y_open[c2:]},
        }
        out.append(client_folds)
    return out
