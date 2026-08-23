"""Recover the immutable original sample ID for every point in a frozen reservoir.

Why this module exists
----------------------
``run_cifar._gather_fold`` computes ``idx`` -- the original torchvision test-set
index of each calibration point -- and then DISCARDS it: the exported npz stores
only ``{fold}_logits``, ``{fold}_y_open``, ``{fold}_client``. All 335 archived
reservoirs predate the fix, so identity cannot be read out of any of them, and the
Phase-1 brief forbids inferring identity from floating-point logits.

What we do instead
------------------
The calibration split is a pure deterministic function of the run config, driven by
seeded ``np.random.default_rng``. We RECOMPUTE it and then REQUIRE exact,
element-wise, order-sensitive agreement with the stored arrays:

    recon(y_open) == stored y_open   AND   recon(client) == stored client

for all three folds. A CIFAR-100 run puts 8570 ordered positions behind that check.
Anything short of full agreement raises ``IdRecoveryError`` -- we never fall back to
a heuristic, and we never match on logits.

Residual assumption (A3 in docs/agent_plan_phase1.md): a *different* index
assignment producing an identical ordered ``(y_open, client)`` pair is not excluded
by this check. It is excluded for new exports, which persist ``{fold}_sample_idx``
directly; ``recover_fold_ids`` prefers that key whenever present.

ID format
---------
``"{dataset}:test:{original_index}"``. Every trusted calibration point (known and
unknown alike) is drawn from the CLEAN torchvision test set, so the test-set index
is globally unique within a dataset and stable across runs, seeds, and backbones.
"""

from __future__ import annotations

import functools
import os
import pickle
from typing import Dict, Tuple

import numpy as np

from fedcore.accounting.provenance import RunSpec
from fedcore.config import FedOSRConfig
from fedcore.data.fedosr_split import build_calibration, open_set_split

FOLDS = ("prop", "cert", "test")


class IdRecoveryError(RuntimeError):
    """Raised when a reservoir's sample IDs cannot be recovered AND verified.

    Never caught-and-defaulted inside this package: a run whose identity is not
    proven is reported as unresolved, never accounted with guessed IDs.
    """


# --------------------------------------------------------------------------- #
# CIFAR label loading (no torch / torchvision; matches torchvision's ordering)
# --------------------------------------------------------------------------- #
def _unpickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


@functools.lru_cache(maxsize=4)
def _cifar_labels(dataset: str, data_root: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(train_labels, test_labels)`` in torchvision's index order.

    torchvision.datasets.CIFAR10 concatenates data_batch_1..5 in order for train and
    reads test_batch for test; CIFAR100 reads the 'train'/'test' files' fine_labels.
    Reproducing that ordering here is what makes the recovered index meaningful --
    and the y_open/client verification is what proves we reproduced it.
    """
    if dataset == "cifar10":
        root = os.path.join(data_root, "cifar-10-batches-py")
        train = np.concatenate(
            [
                np.array(_unpickle(os.path.join(root, f"data_batch_{i}"))[b"labels"])
                for i in range(1, 6)
            ]
        )
        test = np.array(_unpickle(os.path.join(root, "test_batch"))[b"labels"])
    elif dataset == "cifar100":
        root = os.path.join(data_root, "cifar-100-python")
        train = np.array(_unpickle(os.path.join(root, "train"))[b"fine_labels"])
        test = np.array(_unpickle(os.path.join(root, "test"))[b"fine_labels"])
    else:
        raise IdRecoveryError(f"no label loader for dataset {dataset!r}")
    return train, test


# --------------------------------------------------------------------------- #
# split recomputation
# --------------------------------------------------------------------------- #
def recompute_split(
    spec: RunSpec, data_root: str = "data"
) -> Dict[str, Dict[str, np.ndarray]]:
    """Recompute ``{fold: {idx, y_open, client}}`` exactly as run_cifar built it.

    Mirrors run_cifar.py's calibration path (open_set_split -> build_calibration ->
    _gather_fold) by CALLING the same functions -- not by reimplementing them -- so
    the recomputation cannot drift from the pipeline it audits.
    """
    train_labels, test_labels = _cifar_labels(spec.dataset, data_root)

    known, unknown, remap = open_set_split(
        train_labels,
        spec.n_known,
        spec.seed,
        unknown_classes=list(spec.unknown_classes) if spec.unknown_classes else None,
    )
    test_known_idx = np.where(np.isin(test_labels, known))[0]
    test_known_remapped = np.array([remap[int(c)] for c in test_labels[test_known_idx]])
    test_unknown_idx = np.where(np.isin(test_labels, unknown))[0]

    # run_cifar never exposes the fold fractions or contamination on the CLI, so every
    # archived run used the FedOSRConfig defaults. The verification below is what
    # actually establishes that -- it is not assumed.
    cfg = FedOSRConfig(
        dataset=spec.dataset,
        n_known=spec.n_known,
        n_clients=spec.n_clients,
        seed=spec.seed,
    )
    calib = build_calibration(
        test_known_idx,
        test_known_remapped,
        test_unknown_idx,
        spec.n_clients,
        cfg.folds(),
        cfg.unknown_contamination,
        spec.seed,
    )

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for fold in FOLDS:
        out[fold] = {
            "idx": np.concatenate(
                [np.asarray(calib[j][fold]["idx"]) for j in range(spec.n_clients)]
            ),
            "y_open": np.concatenate(
                [np.asarray(calib[j][fold]["y_open"]) for j in range(spec.n_clients)]
            ),
            "client": np.concatenate(
                [
                    np.full(len(calib[j][fold]["idx"]), j, dtype=int)
                    for j in range(spec.n_clients)
                ]
            ),
        }
    return out


def make_sample_ids(dataset: str, idx: np.ndarray) -> np.ndarray:
    """Immutable IDs for original test-set indices."""
    return np.array([f"{dataset}:test:{int(i)}" for i in np.asarray(idx)], dtype=object)


def recover_fold_ids(
    spec: RunSpec,
    data_root: str = "data",
    npz: "np.lib.npyio.NpzFile | None" = None,
) -> Dict[str, np.ndarray]:
    """Verified ``{fold: sample_id array}`` for one reservoir.

    Prefers the persisted ``{fold}_sample_idx`` (new exports). Otherwise recomputes
    the split and verifies it element-wise against the stored ``y_open``/``client``.

    Raises
    ------
    IdRecoveryError
        If verification fails on any fold. The run is then reported as unresolved.
    """
    close_after = npz is None
    z = np.load(spec.npz_path, allow_pickle=True) if npz is None else npz
    try:
        # Fast path: identity persisted at reservoir creation time.
        if all(f"{fold}_sample_idx" in z.files for fold in FOLDS):
            return {
                fold: make_sample_ids(spec.dataset, z[f"{fold}_sample_idx"])
                for fold in FOLDS
            }

        recon = recompute_split(spec, data_root)
        for fold in FOLDS:
            got_y, got_c = z[f"{fold}_y_open"], z[f"{fold}_client"]
            exp_y, exp_c = recon[fold]["y_open"], recon[fold]["client"]
            if got_y.shape != exp_y.shape:
                raise IdRecoveryError(
                    f"{spec.run_id}: fold {fold!r} size mismatch "
                    f"(stored {got_y.shape}, recomputed {exp_y.shape}) -- "
                    f"config from {spec.provenance_source} is wrong for this reservoir"
                )
            if not np.array_equal(got_y, exp_y):
                n_bad = int((got_y != exp_y).sum())
                raise IdRecoveryError(
                    f"{spec.run_id}: fold {fold!r} y_open mismatch at {n_bad}/{len(got_y)} "
                    f"positions -- refusing to guess sample identity"
                )
            if not np.array_equal(got_c, exp_c):
                n_bad = int((got_c != exp_c).sum())
                raise IdRecoveryError(
                    f"{spec.run_id}: fold {fold!r} client mismatch at {n_bad}/{len(got_c)} "
                    f"positions -- refusing to guess sample identity"
                )
        return {
            fold: make_sample_ids(spec.dataset, recon[fold]["idx"]) for fold in FOLDS
        }
    finally:
        if close_after:
            z.close()
