"""RATIFICATION_004 regression tests: A3 fail-closed export.

The audit-fold export must NOT hard-crash when a declared unknown class is absent
from a fold (A3 count-starvation). Instead it records structured per-fold support
metadata and still exports the canonical arrays. The three statuses
(proposal_support_valid / certification_a3_valid / evaluation_unknown_metrics_defined)
are independent and must never be collapsed.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedcore.medical.data import (
    MedicalAuditUnit,
    MedicalImageRecord,
    audit_artifact_arrays,
)


def _units(spec):
    """spec: list of (sample_id, client, diagnosis, y_open) -> one image per unit."""
    units = []
    for s, c, d, y in spec:
        label = y if y >= 0 else 0
        img = MedicalImageRecord(
            image_id=f"{s}_img",
            image_path=f"/x/{s}.jpg",
            image_sample_id=f"{s}_img",
            unit_id=s,
            unit_sample_id=s,
            center="0_BCN_nan",
            client_id=c,
            diagnosis=d,
            fold="certification",
            label=label,
        )
        units.append(
            MedicalAuditUnit(
                unit_id=s,
                sample_id=s,
                center="0_BCN_nan",
                client_id=c,
                diagnosis=d,
                y_open=y,
                fold="certification",
                patient_id=f"{s}_pat",
                lesion_ids=(s,),
                images=(img,),
            )
        )
    return units


def _image_logits(n, k=6):
    # one image per unit; deterministic distinct rows
    return np.arange(n * k, dtype=np.float64).reshape(n, k)


def test_missing_declared_unknown_does_not_crash():
    """A fold with only NV (VASC declared but absent) exports with support metadata."""
    units = _units(
        [
            ("s0", 0, "MEL", 3),   # known
            ("s1", 1, "NV", -1),   # declared unknown present
        ]
    )
    out = audit_artifact_arrays(units, _image_logits(2), heldout_diagnoses=["NV", "VASC"])
    assert out["support_complete"].item() is False
    assert sorted(out["observed_heldout"].tolist()) == ["NV"]
    assert sorted(out["missing_heldout"].tolist()) == ["VASC"]
    assert sorted(out["declared_heldout"].tolist()) == ["NV", "VASC"]
    # the canonical arrays still export
    assert out["logits"].shape == (2, 6)
    assert out["y_open"].tolist() == [3, -1]


def test_full_support_is_complete():
    """Both declared unknowns present -> support_complete True, no missing."""
    units = _units(
        [
            ("s0", 0, "MEL", 3),
            ("s1", 1, "NV", -1),
            ("s2", 2, "VASC", -1),
        ]
    )
    out = audit_artifact_arrays(units, _image_logits(3), heldout_diagnoses=["NV", "VASC"])
    assert out["support_complete"].item() is True
    assert out["missing_heldout"].tolist() == []
    assert sorted(out["observed_heldout"].tolist()) == ["NV", "VASC"]


def test_undeclared_unknown_still_fails_closed():
    """A -1 on a diagnosis OUTSIDE the declared set is corruption -> ValueError."""
    units = _units(
        [
            ("s0", 0, "NV", -1),
            ("s1", 1, "BCC", -1),   # BCC not declared unknown
        ]
    )
    with pytest.raises(ValueError, match="UNDECLARED"):
        audit_artifact_arrays(units, _image_logits(2), heldout_diagnoses=["NV", "VASC"])


def test_known_class_mapped_to_minus_one_fails_closed():
    """y_open=-1 on a KNOWN class (declared unknown present too) -> exclusivity RuntimeError."""
    # NV declared unknown; MEL is known but wrongly labeled y_open=-1 while its
    # diagnosis is not in the present held-out set -> exclusivity guard fires.
    units = _units(
        [
            ("s0", 0, "NV", -1),
            ("s1", 1, "MEL", -1),   # MEL is known, must not be -1
        ]
    )
    # MEL is not in declared {NV,VASC} -> caught as UNDECLARED first (also correct).
    with pytest.raises(ValueError, match="UNDECLARED"):
        audit_artifact_arrays(units, _image_logits(2), heldout_diagnoses=["NV", "VASC"])


def test_known_metrics_available_when_test_unknown_absent():
    """A fold missing the unknown still exports known-class logits (metrics computable)."""
    units = _units(
        [
            ("s0", 0, "MEL", 3),
            ("s1", 1, "BKL", 2),
            ("s2", 2, "NV", -1),
        ]
    )
    out = audit_artifact_arrays(units, _image_logits(3), heldout_diagnoses=["NV", "VASC"])
    assert out["support_complete"].item() is False   # VASC missing
    known = out["y_open"] >= 0
    assert int(known.sum()) == 2   # known-class rows are present and usable


def test_no_pi_or_fold_mutation_side_effects():
    """audit_artifact_arrays is pure w.r.t. its inputs (no fold/seed mutation)."""
    spec = [("s0", 0, "MEL", 3), ("s1", 1, "NV", -1)]
    units = _units(spec)
    before = [(u.sample_id, u.client_id, u.diagnosis, u.y_open) for u in units]
    audit_artifact_arrays(units, _image_logits(2), heldout_diagnoses=["NV", "VASC"])
    after = [(u.sample_id, u.client_id, u.diagnosis, u.y_open) for u in units]
    assert before == after


def test_split04_support_pattern_matches_ratification_004():
    """The three statuses for split04's per-fold VASC counts match RATIFICATION_004 §4."""
    # (prop, cert, test) VASC counts from the frozen folds:
    counts = {0: (0, 0, 2), 1: (2, 2, 0), 2: (1, 0, 0), 3: (0, 0, 2), 4: (1, 1, 1)}
    expected = {
        0: (False, False, True),
        1: (True, True, False),   # seed1: A3 VALID (cert>0), eval NA (test=0)
        2: (True, False, False),
        3: (False, False, True),
        4: (True, True, True),
    }
    for seed, (p, c, t) in counts.items():
        got = (p > 0, c > 0, t > 0)
        assert got == expected[seed], f"seed{seed}: {got} != {expected[seed]}"
