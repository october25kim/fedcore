"""PACS-free tests for stage-scoped metadata and truth access."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from fedcore.experiments import v3_pacs_data as pacs_data
from fedcore.experiments.v3_contract import CLIENTS, sha256_file
from fedcore.experiments.v3_pacs_data import (
    FOLD_MANIFEST_COLUMNS,
    IMAGE_MANIFEST_COLUMNS,
    PACSDataError,
    audit_frame_by_domain,
    load_pacs_audit_frame,
    load_pacs_full_audit_records,
    load_pacs_proposal_records,
    load_pacs_selected_audit_records,
    load_pacs_train_records,
    records_by_use_and_domain,
)


KNOWN = ("dog", "elephant", "giraffe", "guitar")


def _write_csv(path: Path, columns, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _stage_manifests(
    tmp_path: Path,
    *,
    opaque_uses: frozenset[str] = frozenset(),
    invalid_second_audit: bool = False,
) -> tuple[Path, Path]:
    images: list[dict[str, str]] = []
    folds: list[dict[str, str]] = []
    definitions = (
        ("dog", "train", "train", "0", "train"),
        ("horse", "proposal", "proposal", "-1", "proposal"),
        ("elephant", "audit", "audit", "1", "audit_selected"),
        ("giraffe", "audit", "audit", "2", "audit_nonselected"),
        ("house", "train", "unused", "-1", "unused"),
    )
    ordinal = 0
    for domain in CLIENTS:
        for class_name, role, use, label, variant in definitions:
            identity = f"{ordinal:064x}"
            opaque = use in opaque_uses or (
                invalid_second_audit and variant == "audit_nonselected"
            )
            if opaque:
                visible_class = f"{use.upper()}_CLASS_MUST_REMAIN_OPAQUE"
                visible_label = f"{use.upper()}_LABEL_MUST_REMAIN_OPAQUE"
                manifest_path = f"{use.upper()}_PATH_MUST_REMAIN_OPAQUE"
            else:
                visible_class = class_name
                visible_label = label
                relative = f"PACS/{domain}/{class_name}/fixture_{ordinal:03d}.png"
                manifest_path = f"data/v1_pacs/extracted/{relative}"
            images.append(
                {
                    "path": manifest_path,
                    "domain": domain,
                    "class": visible_class,
                    "identity": identity,
                    "width": "8",
                    "height": "8",
                    "role": role,
                }
            )
            folds.append(
                {
                    "split": "0",
                    "identity": identity,
                    "domain": domain,
                    "class": visible_class,
                    "role": role,
                    "use": use,
                    "label": visible_label,
                    "path": manifest_path,
                }
            )
            ordinal += 1
    image_path = tmp_path / "IMAGE_MANIFEST.csv"
    fold_path = tmp_path / "FOLD_MANIFEST.csv"
    _write_csv(image_path, IMAGE_MANIFEST_COLUMNS, images)
    _write_csv(fold_path, FOLD_MANIFEST_COLUMNS, folds)
    return image_path, fold_path


def _seal(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text('{"status":"sealed"}\n', encoding="utf-8")
    return path


def _indices(value: int, draws: int = 128) -> dict[str, list[int]]:
    return {client: [value] * draws for client in CLIENTS}


def _assert_exact_stage(records, stage: str, per_client: int) -> None:
    assert len(records) == len(CLIENTS) * per_client
    assert {record.use for record in records} == {stage}
    assert len({record.identity for record in records}) == len(records)
    grouped = records_by_use_and_domain(records, stage)
    assert tuple(grouped) == CLIENTS
    assert all(len(grouped[client]) == per_client for client in CLIENTS)


def _spy_semantic_parsing(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    parsed_records: list[str] = []
    original_parse = pacs_data._parse_csv_values

    def record_parse(record: str, **kwargs):
        parsed_records.append(record)
        return original_parse(record, **kwargs)

    monkeypatch.setattr(pacs_data, "_parse_csv_values", record_parse)
    return parsed_records


def test_train_loader_never_decodes_nontrain_class_label_or_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path, fold_path = _stage_manifests(
        tmp_path, opaque_uses=frozenset({"proposal", "audit", "unused"})
    )
    parsed = _spy_semantic_parsing(monkeypatch)
    records = load_pacs_train_records(
        image_path,
        fold_path,
        split=0,
        known_classes=KNOWN,
        expected_image_count=20,
    )
    _assert_exact_stage(records, "train", 1)
    parsed_text = "".join(parsed)
    for token in ("PROPOSAL_CLASS", "AUDIT_LABEL", "UNUSED_PATH"):
        assert token not in parsed_text


def test_proposal_loader_never_decodes_nonproposal_class_label_or_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path, fold_path = _stage_manifests(
        tmp_path, opaque_uses=frozenset({"train", "audit", "unused"})
    )
    parsed = _spy_semantic_parsing(monkeypatch)
    records = load_pacs_proposal_records(
        image_path,
        fold_path,
        split=0,
        known_classes=KNOWN,
        expected_image_count=20,
    )
    _assert_exact_stage(records, "proposal", 1)
    parsed_text = "".join(parsed)
    for token in ("TRAIN_CLASS", "AUDIT_LABEL", "UNUSED_PATH"):
        assert token not in parsed_text


def test_audit_frame_gate_precedes_any_manifest_open(tmp_path: Path) -> None:
    with pytest.raises(PACSDataError, match="seal is absent"):
        load_pacs_audit_frame(
            tmp_path / "manifest-must-not-open.csv",
            tmp_path / "fold-must-not-open.csv",
            split=0,
            known_classes=KNOWN,
            proposal_seal_path=tmp_path / "PROPOSAL_SEAL.json",
            expected_proposal_seal_sha256="a" * 64,
        )


def test_audit_frame_requires_exact_proposal_seal_hash(tmp_path: Path) -> None:
    seal = _seal(tmp_path, "PROPOSAL_SEAL.json")
    with pytest.raises(PACSDataError, match="hash differs"):
        load_pacs_audit_frame(
            tmp_path / "manifest-must-not-open.csv",
            tmp_path / "fold-must-not-open.csv",
            split=0,
            known_classes=KNOWN,
            proposal_seal_path=seal,
            expected_proposal_seal_sha256="b" * 64,
        )


def test_stage_three_frame_does_not_decode_any_audit_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path, fold_path = _stage_manifests(
        tmp_path, opaque_uses=frozenset({"audit"})
    )
    seal = _seal(tmp_path, "PROPOSAL_SEAL.json")
    parsed = _spy_semantic_parsing(monkeypatch)
    frame = load_pacs_audit_frame(
        image_path,
        fold_path,
        split=0,
        known_classes=KNOWN,
        proposal_seal_path=seal,
        expected_proposal_seal_sha256=sha256_file(seal),
        expected_image_count=20,
    )
    grouped = audit_frame_by_domain(frame)
    assert all(len(grouped[client]) == 2 for client in CLIENTS)
    assert all(not hasattr(row, "label") for row in frame)
    assert all(not hasattr(row, "class_name") for row in frame)
    assert all(not hasattr(row, "manifest_path") for row in frame)
    parsed_text = "".join(parsed)
    for token in ("AUDIT_CLASS", "AUDIT_LABEL", "AUDIT_PATH"):
        assert token not in parsed_text


def test_only_selected_unique_primary_truth_is_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path, fold_path = _stage_manifests(tmp_path, invalid_second_audit=True)
    seal = _seal(tmp_path, "PROPOSAL_SEAL.json")
    seal_hash = sha256_file(seal)
    frame = load_pacs_audit_frame(
        image_path,
        fold_path,
        split=0,
        known_classes=KNOWN,
        proposal_seal_path=seal,
        expected_proposal_seal_sha256=seal_hash,
        expected_image_count=20,
    )
    parsed = _spy_semantic_parsing(monkeypatch)
    records = load_pacs_selected_audit_records(
        image_path,
        fold_path,
        split=0,
        known_classes=KNOWN,
        audit_frame=frame,
        indices=_indices(0),
        draws_per_client=128,
        proposal_seal_path=seal,
        expected_proposal_seal_sha256=seal_hash,
        expected_image_count=20,
    )
    _assert_exact_stage(records, "audit", 1)
    parsed_text = "".join(parsed)
    for token in ("AUDIT_CLASS", "AUDIT_LABEL", "AUDIT_PATH"):
        assert token not in parsed_text


def test_invalid_selected_primary_truth_fails_closed(tmp_path: Path) -> None:
    image_path, fold_path = _stage_manifests(tmp_path, invalid_second_audit=True)
    seal = _seal(tmp_path, "PROPOSAL_SEAL.json")
    seal_hash = sha256_file(seal)
    frame = load_pacs_audit_frame(
        image_path,
        fold_path,
        split=0,
        known_classes=KNOWN,
        proposal_seal_path=seal,
        expected_proposal_seal_sha256=seal_hash,
        expected_image_count=20,
    )
    with pytest.raises(PACSDataError, match="unknown PACS class"):
        load_pacs_selected_audit_records(
            image_path,
            fold_path,
            split=0,
            known_classes=KNOWN,
            audit_frame=frame,
            indices=_indices(1),
            draws_per_client=128,
            proposal_seal_path=seal,
            expected_proposal_seal_sha256=seal_hash,
            expected_image_count=20,
        )


def test_full_audit_truth_refuses_before_or_under_mismatched_primary_seal(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "PRIMARY_SEAL.json"
    with pytest.raises(PACSDataError, match="seal is absent"):
        load_pacs_full_audit_records(
            tmp_path / "manifest-must-not-open.csv",
            tmp_path / "fold-must-not-open.csv",
            split=0,
            known_classes=KNOWN,
            primary_seal_path=missing,
            expected_primary_seal_sha256="a" * 64,
        )
    seal = _seal(tmp_path, "PRIMARY_SEAL.json")
    with pytest.raises(PACSDataError, match="hash differs"):
        load_pacs_full_audit_records(
            tmp_path / "manifest-must-not-open.csv",
            tmp_path / "fold-must-not-open.csv",
            split=0,
            known_classes=KNOWN,
            primary_seal_path=seal,
            expected_primary_seal_sha256="b" * 64,
        )


def test_full_audit_truth_opens_only_after_exact_global_primary_seal(
    tmp_path: Path,
) -> None:
    image_path, fold_path = _stage_manifests(tmp_path)
    seal = _seal(tmp_path, "PRIMARY_SEAL.json")
    records = load_pacs_full_audit_records(
        image_path,
        fold_path,
        split=0,
        known_classes=KNOWN,
        primary_seal_path=seal,
        expected_primary_seal_sha256=sha256_file(seal),
        expected_image_count=20,
    )
    _assert_exact_stage(records, "audit", 2)


def test_visible_stage_still_rejects_its_own_invalid_label(tmp_path: Path) -> None:
    image_path, fold_path = _stage_manifests(tmp_path)
    with fold_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["label"] = "VISIBLE_TRAIN_LABEL_IS_INVALID"
    _write_csv(fold_path, FOLD_MANIFEST_COLUMNS, rows)
    with pytest.raises(PACSDataError, match="invalid train label"):
        load_pacs_train_records(
            image_path,
            fold_path,
            split=0,
            known_classes=KNOWN,
            expected_image_count=20,
        )
