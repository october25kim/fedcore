"""The pre-registered Fed-ISIC audit-unit design (prereg data.fed_isic2019, A-002).

Asserts the DECLARED clauses, on synthetic metadata where the expected answer is
computable by hand:

* train/audit boundary = FLamby's canonical per-center train/test split;
* audit units are lesions, with EXACTLY ONE image drawn per lesion;
* the 2,302 straddling and 9 cross-center lesions leave the audit pool and stay in
  training;
* unknown classes are never trained on.

The real-data counts live in ``test_fed_isic2019_centers.py``; this file must stay
runnable without the 9.2 GB corpus.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from fedcore.experiments.isic_preflight import PREREG_ROSTER
from fedcore.experiments.isic_source_data import (
    FOLD_FIELDS,
    METADATA_FIELDS,
    assign_folds,
    build_fold_rows,
    unit_table,
)
from fedcore.medical.data import MedicalDataConfig, load_fed_isic_job

KNOWN = "NV"
UNKNOWN = "MEL"


def _row(image, lesion, center_index, diagnosis, flamby_fold, eligible, reason=""):
    return {
        "image_id": image,
        "lesion_id": lesion,
        "patient_id": lesion,
        "center": f"{center_index}_C{center_index}",
        "center_index": center_index,
        "center_name": f"C{center_index}",
        "diagnosis": diagnosis,
        "flamby_fold": flamby_fold,
        "audit_eligible": eligible,
        "audit_exclusion_reason": reason,
    }


def _synthetic_rows():
    """One center, with every lesion archetype the declared design must handle."""
    rows = []
    # Train-side known lesions: training supply. Two images each.
    for i in range(6):
        for k in range(2):
            rows.append(_row(f"tr{i}_{k}", f"LTR{i}", 0, KNOWN, "train", 0, "flamby_train"))
    # Train-side unknown lesion: never trainable (unknown), never auditable (train side).
    rows.append(_row("tu0", "LTU0", 0, UNKNOWN, "train", 0, "flamby_train"))
    # Test-side known lesions: audit candidates, several images each.
    for i in range(14):
        for k in range(3):
            rows.append(_row(f"te{i}_{k}", f"LTE{i}", 0, KNOWN, "test", 1))
    # Test-side unknown lesions: audit candidates carrying the labeled unknowns.
    for i in range(6):
        rows.append(_row(f"tx{i}", f"LTX{i}", 0, UNKNOWN, "test", 1))
    # A straddling lesion: images on BOTH sides -> excluded from audit, kept in train.
    rows.append(_row("st0", "LST0", 0, KNOWN, "train", 0, "straddling_lesion"))
    rows.append(_row("st1", "LST0", 0, KNOWN, "test", 0, "straddling_lesion"))
    return rows


def _emit(tmp_path, rows, unknown_classes=(UNKNOWN,), seed=7):
    units = unit_table(rows)
    assignment = assign_folds(units, list(unknown_classes), seed=seed)
    meta = tmp_path / "metadata.csv"
    folds = tmp_path / "folds.csv"
    for path, fields, payload in (
        (meta, METADATA_FIELDS, rows),
        (folds, FOLD_FIELDS, build_fold_rows(units, assignment)),
    ):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(payload)
    return meta, folds, units, assignment


def _config(tmp_path, meta, folds):
    root = tmp_path / "images"
    root.mkdir(exist_ok=True)
    for row in _synthetic_rows():
        (root / f"{row['image_id']}.jpg").write_bytes(b"x")
    return MedicalDataConfig(
        metadata_csv=str(meta),
        folds_csv=str(folds),
        center_col="center",
        diagnosis_col="diagnosis",
        patient_col="patient_id",
        lesion_col="lesion_id",
        image_col="image_id",
        unit_col="lesion_id",
        image_root=str(root),
        image_extension=".jpg",
    )


def test_audit_units_carry_exactly_one_image_per_lesion(tmp_path):
    """``sampling``: "draw EXACTLY ONE image per lesion" -- the i.i.d. requirement.

    Test-side known lesions have 3 images each; an audit unit must still carry 1.
    """
    meta, folds, _, _ = _emit(tmp_path, _synthetic_rows())
    job = load_fed_isic_job(_config(tmp_path, meta, folds), UNKNOWN)

    for fold in ("proposal", "certification", "test"):
        units = job.fold_units(fold)
        assert units, f"{fold} fold is empty"
        assert {u.image_multiplicity for u in units} == {1}


def test_training_uses_only_flamby_train_side_known_images(tmp_path):
    """``train_pool``: "that center's FLamby TRAIN images, known classes only"."""
    meta, folds, _, _ = _emit(tmp_path, _synthetic_rows())
    job = load_fed_isic_job(_config(tmp_path, meta, folds), UNKNOWN)

    trained = [r for records in job.training_by_client for r in records]
    by_id = {row["image_id"]: row for row in _synthetic_rows()}
    assert trained, "training pool must not be empty"
    for record in trained:
        assert by_id[record.image_id]["flamby_fold"] == "train"
        assert record.diagnosis != UNKNOWN


def test_unknown_classes_are_never_trained_on(tmp_path):
    meta, folds, _, _ = _emit(tmp_path, _synthetic_rows())
    job = load_fed_isic_job(_config(tmp_path, meta, folds), UNKNOWN)

    assert UNKNOWN not in {
        r.diagnosis for records in job.training_by_client for r in records
    }


def test_no_lesion_is_both_trained_and_audited(tmp_path):
    """The whole point of the straddling/cross-center exclusion."""
    meta, folds, _, _ = _emit(tmp_path, _synthetic_rows())
    job = load_fed_isic_job(_config(tmp_path, meta, folds), UNKNOWN)

    trained = {r.unit_id for records in job.training_by_client for r in records}
    audited = {
        u.unit_id
        for fold in ("proposal", "certification", "test")
        for u in job.fold_units(fold)
    }
    assert trained & audited == set()

    train_images = {r.image_id for records in job.training_by_client for r in records}
    audit_images = {
        r.image_id
        for fold in ("proposal", "certification", "test")
        for u in job.fold_units(fold)
        for r in u.images
    }
    assert train_images & audit_images == set()


def test_straddling_lesion_leaves_the_audit_pool_but_stays_in_training(tmp_path):
    """``straddling_lesions.policy``, on the one synthetic straddler (LST0).

    Its TEST-side image must not be audited, and must not be smuggled into training
    either: ``train_pool`` is the FLamby TRAIN images, so only ``st0`` may train.
    """
    meta, folds, _, assignment = _emit(tmp_path, _synthetic_rows())
    fold, images = assignment["LST0"]
    assert fold == "train"
    assert set(images) == {"st0"}, "only the train-side image is training supply"

    job = load_fed_isic_job(_config(tmp_path, meta, folds), UNKNOWN)
    audited = {
        u.unit_id
        for f in ("proposal", "certification", "test")
        for u in job.fold_units(f)
    }
    assert "LST0" not in audited


def test_train_side_unknown_lesion_is_unused_not_trained(tmp_path):
    """It can be neither trained on (unknown) nor audited (train side)."""
    _, _, _, assignment = _emit(tmp_path, _synthetic_rows())
    assert assignment["LTU0"] == ("unused", ())


def test_every_lesion_gets_an_accounting_row(tmp_path):
    """prohibitions: cells are "REPORTED with accounting rows, never silently dropped"."""
    rows = _synthetic_rows()
    _, folds, units, _ = _emit(tmp_path, rows)
    with open(folds, newline="", encoding="utf-8") as handle:
        emitted = list(csv.DictReader(handle))

    assert {r["audit_unit_id"] for r in emitted} == set(units)
    assert any(r["fold"] == "unused" for r in emitted), "unused must be explicit"


def test_fold_assignment_is_deterministic_and_seed_sensitive(tmp_path):
    rows = _synthetic_rows()
    units = unit_table(rows)

    a = assign_folds(units, [UNKNOWN], seed=7)
    b = assign_folds(units, [UNKNOWN], seed=7)
    c = assign_folds(units, [UNKNOWN], seed=8)

    assert a == b, "same seed must reproduce the draw exactly"
    assert a != c, "the draw must actually depend on the seed"
    # Sizes are pure integer arithmetic on lesion counts, so they must NOT move.
    sizes = lambda m: sorted(  # noqa: E731
        (fold, sum(1 for v in m.values() if v[0] == fold))
        for fold in {v[0] for v in m.values()}
    )
    assert sizes(a) == sizes(c), "fold SIZES are seed-independent; only membership moves"


def test_audit_pool_never_contains_a_flamby_train_image(tmp_path):
    rows = _synthetic_rows()
    units = unit_table(rows)
    assignment = assign_folds(units, [UNKNOWN], seed=3)
    side = {row["image_id"]: row["flamby_fold"] for row in rows}

    for lesion, (fold, images) in assignment.items():
        if fold in ("proposal", "certification", "test"):
            assert all(side[i] == "test" for i in images)


@pytest.mark.skipif(
    not Path("results/preregistration.yaml").is_file(),
    reason="sealed preregistration artifact absent",
)
def test_preflight_roster_matches_the_sealed_preregistration():
    """``split_roster.drawn`` is final; the preflight's mirror must not drift."""
    import yaml

    with open("results/preregistration.yaml", encoding="utf-8") as handle:
        prereg = yaml.safe_load(handle)
    drawn = prereg["data"]["fed_isic2019"]["split_roster"]["drawn"]

    assert len(PREREG_ROSTER) == len(drawn)
    for (split_id, pair), entry in zip(PREREG_ROSTER, drawn):
        assert split_id == entry["split_id"]
        assert list(pair) == list(entry["unknown"])


def test_emitted_folds_reject_an_image_that_is_not_the_units(tmp_path):
    """The subset rule still forbids inventing an image for a unit."""
    rows = _synthetic_rows()
    meta, folds, units, assignment = _emit(tmp_path, rows)
    fold_rows = build_fold_rows(units, assignment)
    for row in fold_rows:
        if row["fold"] == "certification":
            row["image_ids_json"] = json.dumps(["not_an_image_of_this_lesion"])
            break
    with open(folds, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FOLD_FIELDS))
        writer.writeheader()
        writer.writerows(fold_rows)

    with pytest.raises(ValueError, match="not images of unit"):
        load_fed_isic_job(_config(tmp_path, meta, folds), UNKNOWN)
