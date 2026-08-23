"""Office-Home data-layer tests: frozen invariants + fail-closed provenance."""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from fedcore.data.officehome import (
    AUDIT_ROLES,
    DOMAINS,
    FOLD_ROLES,
    OfficeHomeDataConfig,
    OfficeHomeDataError,
    audit_role_arrays,
    load_officehome_job,
    traffic_identity_arrays,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "results/officehome/dedup/retained_canonical_manifest.csv")
CLASS_SPLITS = os.path.join(REPO, "results/officehome/preflight/class_splits.csv")
IMAGE_ROOT = os.path.join(REPO, "data/officehome/OfficeHomeDataset")


def _folds(split_id: str) -> str:
    return os.path.join(REPO, f"results/officehome/folds/folds_{split_id}.csv")


def _config(split_id: str = "officehome_split_0") -> OfficeHomeDataConfig:
    return OfficeHomeDataConfig(
        manifest_csv=MANIFEST,
        folds_csv=_folds(split_id),
        class_splits_csv=CLASS_SPLITS,
        split_id=split_id,
        image_root=IMAGE_ROOT,
    )


pytestmark = pytest.mark.skipif(
    not os.path.isfile(MANIFEST), reason="frozen Office-Home manifest absent"
)


@pytest.mark.parametrize("split_index", range(5))
def test_all_splits_load_and_are_disjoint(split_index):
    split_id = f"officehome_split_{split_index}"
    job = load_officehome_job(_config(split_id), check_image_files=False)
    assert job.n_clients == 4 and job.domains == DOMAINS
    assert job.n_known == 45 and len(job.unknown_classes) == 20
    assert not (set(job.known_classes) & set(job.unknown_classes))
    # 10 pairwise role-identity disjointness (validate_training_ready ran in load).
    ids = {role: {r.sample_id for r in job.role_records(role)} for role in FOLD_ROLES}
    for i, left in enumerate(FOLD_ROLES):
        for right in FOLD_ROLES[i + 1 :]:
            assert not (ids[left] & ids[right])
    # Every audit role has both known and unknown support.
    for role in AUDIT_ROLES:
        recs = job.role_records(role)
        assert any(r.y_open == -1 for r in recs)
        assert any(r.y_open >= 0 for r in recs)
    # No unknown class in train.
    for records in job.training_by_client:
        assert all(r.y_open >= 0 for r in records)


def test_y_open_maps_unknown_classes_to_minus_one():
    job = load_officehome_job(_config(), check_image_files=False)
    unknown = set(job.unknown_classes)
    for role in AUDIT_ROLES:
        for r in job.role_records(role):
            if r.klass in unknown:
                assert r.y_open == -1
            else:
                assert r.y_open == job.class_to_label[r.klass]


def test_sample_id_round_trip_through_audit_arrays():
    job = load_officehome_job(_config(), check_image_files=False)
    recs = job.role_records("certification")
    logits = np.zeros((len(recs), job.n_known), dtype=float)
    arrays = audit_role_arrays(recs, logits, job.known_classes, job.unknown_classes)
    # sample_ids come straight from the manifest and are 1:1 with the records.
    assert list(arrays["sample_id"]) == [r.sample_id for r in recs]
    assert np.array_equal(arrays["y_open"], np.array([r.y_open for r in recs]))
    assert np.array_equal(arrays["client"], np.array([r.client_id for r in recs]))
    # y_open == -1 exactly on the split's unknowns.
    unknown = set(job.unknown_classes)
    expected = np.array([r.klass in unknown for r in recs])
    assert np.array_equal(arrays["y_open"] == -1, expected)


def test_traffic_arrays_are_identity_only():
    job = load_officehome_job(_config(), check_image_files=False)
    arrays = traffic_identity_arrays(job.role_records("traffic"))
    assert set(arrays) == {"sample_id", "client", "site_id"}
    assert "y_open" not in arrays and "logits" not in arrays


# ---------------------------------------------------------------------------- #
# Fail-closed provenance tests over small synthetic CSVs.
# ---------------------------------------------------------------------------- #
def _write_csv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _tiny_dataset(tmp_path):
    # 2 classes x 4 domains; class A known, class B unknown.
    man_header = ["sample_id", "domain", "klass", "filename", "normalized_relpath", "content_sha256", "bytes"]
    fold_header = ["domain", "class", "role", "sample_id"]
    cs_header = ["split_id", "seed", "role", "class"]
    man_rows, fold_rows = [], []
    sid = 0
    roles = ["train", "proposal", "certification", "traffic", "evaluation"]
    for domain in DOMAINS:
        for klass, role_cycle in (("A", roles), ("B", roles[1:])):  # B never train
            for role in role_cycle:
                s = f"id{sid:04d}"
                rel = f"{domain}/{klass}/{sid}.jpg"
                man_rows.append([s, domain, klass, f"{sid}.jpg", rel, "0" * 64, 10])
                fold_rows.append([domain, klass, role, s])
                sid += 1
    cs_rows = [["tiny", 123, "known", "A"], ["tiny", 123, "unknown", "B"]]
    man = tmp_path / "man.csv"
    fold = tmp_path / "fold.csv"
    cs = tmp_path / "cs.csv"
    _write_csv(man, man_header, man_rows)
    _write_csv(fold, fold_header, fold_rows)
    _write_csv(cs, cs_header, cs_rows)
    root = tmp_path / "root"
    for domain in DOMAINS:
        os.makedirs(root / domain, exist_ok=True)
    return OfficeHomeDataConfig(
        manifest_csv=str(man), folds_csv=str(fold), class_splits_csv=str(cs),
        split_id="tiny", image_root=str(root),
    ), man, fold


def test_tiny_dataset_loads(tmp_path):
    cfg, _, _ = _tiny_dataset(tmp_path)
    job = load_officehome_job(cfg, check_image_files=False)
    assert job.n_known == 1 and job.unknown_classes == ("B",)


def test_fail_closed_on_provenance_mismatch(tmp_path):
    cfg, _, fold = _tiny_dataset(tmp_path)
    rows = list(csv.reader(open(fold)))
    rows[1][1] = "Clipart"  # domain disagrees with manifest for this sample
    _write_csv(fold, rows[0], rows[1:])
    with pytest.raises(OfficeHomeDataError):
        load_officehome_job(cfg, check_image_files=False)


def test_fail_closed_on_unknown_in_train(tmp_path):
    cfg, _, fold = _tiny_dataset(tmp_path)
    rows = list(csv.reader(open(fold)))
    # Flip a class-B (unknown) row into the train role.
    for r in rows[1:]:
        if r[1] == "B":
            r[2] = "train"
            break
    _write_csv(fold, rows[0], rows[1:])
    with pytest.raises(OfficeHomeDataError):
        load_officehome_job(cfg, check_image_files=False)


def test_fail_closed_on_duplicate_sample_id(tmp_path):
    cfg, man, _ = _tiny_dataset(tmp_path)
    rows = list(csv.reader(open(man)))
    dup = list(rows[1])
    dup[0] = rows[2][0]  # duplicate an existing sample_id
    rows.append(dup)
    _write_csv(man, rows[0], rows[1:])
    with pytest.raises(OfficeHomeDataError):
        load_officehome_job(cfg, check_image_files=False)


def test_audit_arrays_reject_mislabelled_unknown(tmp_path):
    cfg, _, _ = _tiny_dataset(tmp_path)
    job = load_officehome_job(cfg, check_image_files=False)
    recs = list(job.role_records("certification"))
    logits = np.zeros((len(recs), job.n_known))
    # Corrupt: claim there are no unknown classes -> y_open==-1 no longer exclusive.
    with pytest.raises(OfficeHomeDataError):
        audit_role_arrays(recs, logits, job.known_classes, ())
