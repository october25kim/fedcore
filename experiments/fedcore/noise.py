"""Client-side TRAIN-label corruption.

Corruption affects **training labels only**. The trusted calibration/test folds
must stay clean -- NEVER call :func:`make_label_noise` on calibration indices.

* ``symmetric``  -- flip to a uniformly chosen *other* known class.
* ``asymmetric`` -- flip ``y -> (y + 1) % n_known``.
* ``none`` / ``rate <= 0`` -- no flips.

No self-flips are ever produced.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def make_label_noise(
    remapped_labels,
    indices: Sequence[int],
    noise_type: str,
    rate: float,
    n_known: int,
    seed: int,
) -> Dict[int, int]:
    """Return ``{dataset_index: noisy_label}`` for flipped TRAIN points only.

    Parameters
    ----------
    remapped_labels : mapping (dict or array) from dataset index to the clean
        remapped known-class label in ``[0, n_known)``.
    indices : the TRAIN dataset indices eligible for corruption.
    """
    if noise_type in (None, "none") or rate <= 0.0:
        return {}

    rng = np.random.default_rng(seed)
    out: Dict[int, int] = {}
    for idx in indices:
        y = int(remapped_labels[idx])
        if rng.random() >= rate:
            continue
        if noise_type == "symmetric":
            choices = [c for c in range(n_known) if c != y]
            if not choices:
                continue
            ny = int(rng.choice(choices))
        elif noise_type == "asymmetric":
            ny = (y + 1) % n_known
        else:
            raise ValueError(f"unknown noise_type={noise_type!r}")
        if ny != y:
            out[int(idx)] = ny
    return out
