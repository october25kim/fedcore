"""FLamby Fed-ISIC2019 recipe transcription checks (torch-free parts).

Guards the two things most easily got wrong when moving FLamby's 8-class recipe
onto the open-set protocol: the alpha restriction policy, and the fact that the
pre-registered alpha vector really is FLamby's.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcore.data.fed_isic2019 import CLASSES
from fedcore.medical.flamby import (
    FLAMBY_CLASS_ORDER,
    FLAMBY_FOCAL_ALPHA_FULL8,
    FLAMBY_FOCAL_GAMMA,
    restrict_focal_alpha,
)


def test_flamby_class_order_matches_the_dataset_layer():
    assert FLAMBY_CLASS_ORDER == CLASSES


def test_prereg_alpha_and_gamma_are_flamby_values():
    """Transcribed from flamby/datasets/fed_isic2019/loss.py BaselineLoss."""
    assert FLAMBY_FOCAL_ALPHA_FULL8 == (
        5.5813,
        2.0472,
        7.0204,
        26.1194,
        9.5369,
        101.0707,
        92.5224,
        38.3443,
    )
    assert FLAMBY_FOCAL_GAMMA == 2.0
    assert len(FLAMBY_FOCAL_ALPHA_FULL8) == 8


def test_alpha_is_restricted_by_index_without_renormalization():
    """recipe.focal_alpha_policy: restrict by index, NO renormalization.

    The weights are per-class multipliers, not a distribution, so the kept entries
    must be bit-identical to FLamby's -- not rescaled to sum to anything.
    """
    # Roster split 0 holds out [MEL, BCC] -> knowns are NV, AK, BKL, DF, VASC, SCC.
    known = ["NV", "AK", "BKL", "DF", "VASC", "SCC"]
    alpha = restrict_focal_alpha(known)
    expected = np.asarray([2.0472, 9.5369, 26.1194, 101.0707, 92.5224, 38.3443])
    # By index in FLamby's order, NOT by the caller's order.
    lookup = {name: FLAMBY_FOCAL_ALPHA_FULL8[CLASSES.index(name)] for name in known}
    assert np.allclose(alpha, [lookup[name] for name in known])
    assert np.allclose(np.sort(alpha), np.sort(expected))
    assert len(alpha) == len(known)
    # Explicitly NOT renormalized.
    assert not np.isclose(alpha.sum(), 1.0)


def test_alpha_follows_the_head_order_not_flamby_order():
    """alpha[i] must weight the class the head's column i predicts."""
    known = ["SCC", "NV"]  # deliberately not FLamby order
    alpha = restrict_focal_alpha(known)
    assert np.isclose(alpha[0], 38.3443)  # SCC
    assert np.isclose(alpha[1], 2.0472)  # NV


def test_alpha_rejects_unknown_or_duplicate_classes():
    for bad in (["NV", "NOT_A_CLASS"], ["NV", "NV"], []):
        try:
            restrict_focal_alpha(bad)
        except ValueError:
            continue
        raise AssertionError(f"restrict_focal_alpha accepted {bad!r}")


def main():
    tests = (
        test_flamby_class_order_matches_the_dataset_layer,
        test_prereg_alpha_and_gamma_are_flamby_values,
        test_alpha_is_restricted_by_index_without_renormalization,
        test_alpha_follows_the_head_order_not_flamby_order,
        test_alpha_rejects_unknown_or_duplicate_classes,
    )
    for test in tests:
        test()
        print(f"  [PASS] {test.__name__}")
    print(f"flamby recipe checks: PASS ({len(tests)} tests)")


if __name__ == "__main__":
    main()
