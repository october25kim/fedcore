"""Fed-ISIC2019 center-derivation regression tests (WP-C data layer).

The center definition is load-bearing: J=6 and every per-client count in the
certificate flows from it. These tests pin it against FLamby's documented sizes
and, where the staged inputs allow, against FLamby's own shipped split.

Tests that need the ISIC metadata skip cleanly when it is not staged, so the
suite still runs on a machine without the (CC-BY-NC, 9.8GB) dataset.

Run: python -m pytest tests/test_fed_isic2019_centers.py -q
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from fedcore.data.fed_isic2019 import (
    CENTER_NAMES,
    CENTER_NAME_TO_INDEX,
    CLASSES,
    DOCUMENTED_CENTER_SIZES,
    DOCUMENTED_TRAIN_SIZES,
    N_CENTERS,
    N_IMAGES_TOTAL,
    assign_center_names,
    center_sizes,
    class_counts_by_center,
    cross_center_lesions,
    default_root,
    load_center_table,
    verify_center_sizes,
)
from fedcore.experiments.isic_preflight import _largest_remainder, project_center


def _has_data() -> bool:
    root = default_root()
    return all(
        os.path.isfile(os.path.join(root, name))
        for name in (
            "ISIC_2019_Training_Metadata.csv",
            "ISIC_2019_Training_GroundTruth.csv",
            "HAM10000_metadata.csv",
        )
    )


needs_data = pytest.mark.skipif(not _has_data(), reason="ISIC-2019 metadata not staged")


# --------------------------------------------------------------------------- #
# constants (no data required)
# --------------------------------------------------------------------------- #
def test_center_constants_are_consistent():
    assert len(CENTER_NAME_TO_INDEX) == N_CENTERS
    assert sorted(CENTER_NAME_TO_INDEX.values()) == list(range(N_CENTERS))
    assert len(CENTER_NAMES) == N_CENTERS
    assert len(DOCUMENTED_CENTER_SIZES) == N_CENTERS
    assert sum(DOCUMENTED_CENTER_SIZES) == N_IMAGES_TOTAL
    assert len(CLASSES) == 8


def test_ham_contributes_four_subcenters_summing_to_ham10000():
    """HAM10000 splits into FOUR sub-centers (not three) totalling exactly 10015."""
    ham_idx = [i for i, n in enumerate(CENTER_NAMES) if n.startswith("HAM_")]
    assert len(ham_idx) == 4
    assert sum(DOCUMENTED_CENTER_SIZES[i] for i in ham_idx) == 10015
    # the two non-HAM archives
    assert DOCUMENTED_CENTER_SIZES[CENTER_NAME_TO_INDEX["BCN_nan"]] == 12413
    assert DOCUMENTED_CENTER_SIZES[CENTER_NAME_TO_INDEX["MSK4nan"]] == 819


def test_msk_center_name_keeps_the_prefix_digit():
    """lesion_id.str[:4] over MSK4_* yields 'MSK4', hence 'MSK4nan' not 'MSK_nan'."""
    assert "MSK4nan" in CENTER_NAME_TO_INDEX
    assert "MSK_nan" not in CENTER_NAME_TO_INDEX


def test_assign_center_names_replicates_flamby_concatenation():
    """The 'nan' literal for non-HAM rows is load-bearing, not an accident."""
    metadata = pd.DataFrame(
        {
            "image": ["ISIC_1", "ISIC_2", "ISIC_3"],
            "lesion_id": ["BCN_0000001", "HAM_0000118", "MSK4_0000001"],
        }
    )
    ham = pd.DataFrame({"image_id": ["ISIC_2"], "dataset": ["vidir_modern"]})
    names = assign_center_names(metadata, ham)
    assert list(names) == ["BCN_nan", "HAM_vidir_modern", "MSK4nan"]
    assert set(names) <= set(CENTER_NAME_TO_INDEX)


def test_assign_center_names_rejects_null_lesion_ids():
    metadata = pd.DataFrame({"image": ["ISIC_1"], "lesion_id": [None]})
    ham = pd.DataFrame({"image_id": [], "dataset": []})
    with pytest.raises(ValueError, match="null lesion_id"):
        assign_center_names(metadata, ham)


# --------------------------------------------------------------------------- #
# derivation against the real metadata
# --------------------------------------------------------------------------- #
@needs_data
def test_derived_center_sizes_match_flamby_documented():
    table = load_center_table()
    assert len(table) == N_IMAGES_TOTAL
    assert center_sizes(table) == list(DOCUMENTED_CENTER_SIZES)


@needs_data
def test_verify_reports_agreement_with_official_split():
    report = verify_center_sizes()
    assert report["sizes_match"] and report["total_match"]
    if report["official_split_available"]:
        # independent oracle: FLamby's own per-image center index
        assert report["official_center_agreement"]
        assert report["official_train_sizes"] == list(DOCUMENTED_TRAIN_SIZES)


@needs_data
def test_msk_center_holds_only_three_classes():
    """Drives A3: MSK (center 4) has zero BCC/AK/DF/VASC/SCC, so any unknown pair
    drawn from those five starves it of labeled unknowns."""
    counts = class_counts_by_center(load_center_table())
    msk = CENTER_NAME_TO_INDEX["MSK4nan"]
    present = [c for c in CLASSES if int(counts.loc[msk, c]) > 0]
    assert present == ["MEL", "NV", "BKL"]


@needs_data
def test_cross_center_lesions_are_reported_not_hidden():
    """9 HAM lesions straddle vidir_molemax/vidir_modern -- a real defect that the
    lesion-as-audit-unit consumers must resolve."""
    bad = cross_center_lesions(load_center_table())
    assert bad["lesion_id"].nunique() == 9
    assert set(bad["center_name"]) == {"HAM_vidir_molemax", "HAM_vidir_modern"}


# --------------------------------------------------------------------------- #
# preflight projection arithmetic (no data required)
# --------------------------------------------------------------------------- #
def test_largest_remainder_is_exact_and_integral():
    for total in (0, 1, 7, 23, 1000):
        parts = _largest_remainder(total, (0.40, 0.30, 0.30))
        assert sum(parts) == total
        assert all(isinstance(p, int) and p >= 0 for p in parts)


def test_project_center_zero_unknowns_starves():
    proj = project_center(819, 0, unknown_frac=0.30)
    assert proj["audit_pool"] == 0
    assert proj["n_cert"] == 0
    assert proj["cert_unknown"] == 0


def test_project_center_is_unknown_supply_limited_when_unknowns_are_scarce():
    proj = project_center(10_000, 30, unknown_frac=0.30)
    assert proj["binding_constraint"] == "unknown-supply"
    assert proj["audit_pool"] == 100  # floor(30 / 0.30)
    assert proj["pool_unknown"] == 30
    assert proj["fold_sizes"]["certification"] == proj["n_cert"]
    assert sum(proj["fold_sizes"].values()) == proj["audit_pool"]


def test_project_center_respects_the_known_budget_cap():
    uncapped = project_center(10_000, 3_000, unknown_frac=0.30)
    capped = project_center(10_000, 3_000, unknown_frac=0.30, max_known_audit_frac=0.1)
    assert capped["binding_constraint"] == "known-supply"
    assert capped["audit_pool"] < uncapped["audit_pool"]


def test_project_center_never_invents_unknowns():
    """pool_unknown may never exceed the center's real unknown supply."""
    for n_unknown in (0, 1, 5, 37, 404):
        proj = project_center(10_000, n_unknown, unknown_frac=0.30)
        assert proj["pool_unknown"] <= n_unknown
        assert proj["cert_unknown"] <= n_unknown
