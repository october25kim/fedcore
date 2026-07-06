"""Client-side label corruption for the FedOSR TRAIN labels (numpy).

CRITICAL INVARIANT: noise is injected into TRAINING labels only. The trusted
calibration/test folds (built in ``fedosr_split.build_calibration``) keep their
true labels by definition -- the whole point of Fed-CORE is to certify against a
clean trusted set despite corrupted training. Never call this on calibration.

Schemes (over the remapped known label space [0, n_known)):
  * symmetric  : with prob ``rate``, flip to a uniformly random OTHER known class.
  * asymmetric : with prob ``rate``, flip to ``(y + 1) % n_known`` (a fixed,
                 class-conditional, structured corruption -- the systematic case
                 where confidence deformation is strongest). Generic circular
                 pairing is used so it is well-defined for any known-class subset.
"""
from __future__ import annotations

import numpy as np


def make_label_noise(
    remapped_labels: np.ndarray,
    indices: np.ndarray,
    noise_type: str,
    rate: float,
    n_known: int,
    seed: int,
) -> dict[int, int]:
    """Return a sparse override {dataset_index -> noisy remapped label}.

    ``remapped_labels`` are the true known labels aligned with ``indices``. Only
    flipped points appear in the returned dict; unflipped points keep their true
    label downstream.
    """
    if noise_type in ("none", None) or rate <= 0.0:
        return {}
    rng = np.random.default_rng(seed)
    flip = rng.random(len(indices)) < rate
    override: dict[int, int] = {}
    for pos in np.where(flip)[0]:
        idx = int(indices[pos])
        y = int(remapped_labels[pos])
        if noise_type == "symmetric":
            ny = int(rng.integers(0, n_known - 1))
            if ny >= y:  # map uniformly onto the other n_known-1 classes
                ny += 1
        elif noise_type == "asymmetric":
            ny = (y + 1) % n_known
        else:
            raise ValueError(f"unknown noise_type {noise_type!r}")
        override[idx] = ny
    return override
