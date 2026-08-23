"""Fed-ISIC2019 metadata joins and distinct-unit logit aggregation.

This module is deliberately torch-free.  It validates the immutable metadata
and frozen fold artifacts produced by :mod:`fedcore.medical.preflight`, builds
one leave-one-diagnosis-out training cell, and keeps image observations separate
from the patient/lesion units used by the statistical audit.

The only place repeated images become an audit observation is
``aggregate_unit_logits``: image-level logits are averaged once per frozen unit.
Traffic exports contain opaque unit identity and site identity only -- never a
diagnosis, label, image, or model output.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from fedcore.campaign.artifacts import file_sha256


#: ``unused`` is an ACCOUNTING role, not a fold: under the pre-registered Fed-ISIC
#: audit design (AMENDMENT A-002) a lesion may be neither training supply nor an
#: audit draw -- an unknown-class lesion on the FLamby train side (never trainable,
#: never auditable), or a test-side lesion the 0.30-unknown-fraction pool did not
#: draw.  The emitter writes a row for it anyway so that every lesion reconciles and
#: nothing is silently dropped, per the prereg's prohibitions.  Units in this role
#: contribute no training image and no audit observation.
FOLD_ROLES = ("train", "proposal", "certification", "test", "traffic", "unused")
AUDIT_FOLDS = ("proposal", "certification", "test")
OUTPUT_FOLD_NAMES = {
    "proposal": "prop",
    "certification": "cert",
    "test": "test",
}
_FOLD_ALIASES = {
    "prop": "proposal",
    "proposal": "proposal",
    "cert": "certification",
    "certification": "certification",
    "test": "test",
    "train": "train",
    "traffic": "traffic",
    "unused": "unused",
}


@dataclass(frozen=True)
class MedicalDataConfig:
    """Column and path contract shared with the metadata preflight."""

    metadata_csv: str
    folds_csv: str
    center_col: str
    diagnosis_col: str
    patient_col: str
    lesion_col: str
    image_col: str
    unit_col: str
    image_path_col: str | None = None
    image_root: str | None = None
    image_extension: str = ""
    dataset_name: str = "fed-isic2019"

    def validate(self) -> None:
        if self.unit_col not in {self.patient_col, self.lesion_col}:
            raise ValueError("unit_col must be the declared patient or lesion column")
        if not self.dataset_name.strip():
            raise ValueError("dataset_name must be non-empty")
        if not os.path.isfile(self.metadata_csv):
            raise FileNotFoundError(self.metadata_csv)
        if not os.path.isfile(self.folds_csv):
            raise FileNotFoundError(self.folds_csv)
        semantic_columns = (
            self.center_col,
            self.diagnosis_col,
            self.patient_col,
            self.lesion_col,
            self.image_col,
        )
        if any(
            not str(column).strip() for column in (*semantic_columns, self.unit_col)
        ):
            raise ValueError("metadata column names must be non-empty")
        if len(set(semantic_columns)) != len(semantic_columns):
            raise ValueError(
                "center, diagnosis, patient, lesion, and image columns must be distinct"
            )
        extension = self.image_extension
        if extension and (
            os.path.sep in extension or (os.path.altsep and os.path.altsep in extension)
        ):
            raise ValueError("image_extension must be a suffix, not a path")

    @property
    def unit_kind(self) -> str:
        return "patient" if self.unit_col == self.patient_col else "lesion"

    def resolve_image_path(self, row: Mapping[str, str]) -> str:
        raw = row[self.image_path_col] if self.image_path_col else row[self.image_col]
        raw = str(raw).strip()
        if not raw:
            source = self.image_path_col or self.image_col
            raise ValueError(f"blank image path derived from column {source!r}")
        extension = self.image_extension.strip()
        if extension and not extension.startswith("."):
            extension = "." + extension
        if extension and not raw.casefold().endswith(extension.casefold()):
            raw += extension
        if os.path.isabs(raw):
            return os.path.normpath(raw)
        root = self.image_root
        if root is None:
            root = os.path.dirname(os.path.abspath(self.metadata_csv))
        return os.path.abspath(os.path.join(root, raw))


@dataclass(frozen=True)
class MedicalImageRecord:
    """One physical image used for training or unit-level inference."""

    image_id: str
    image_path: str
    image_sample_id: str
    unit_id: str
    unit_sample_id: str
    center: str
    client_id: int
    diagnosis: str
    fold: str
    label: int


@dataclass(frozen=True)
class MedicalAuditUnit:
    """One independent patient/lesion audit observation."""

    unit_id: str
    sample_id: str
    center: str
    client_id: int
    diagnosis: str
    y_open: int
    fold: str
    patient_id: str
    lesion_ids: tuple[str, ...]
    images: tuple[MedicalImageRecord, ...]

    @property
    def image_multiplicity(self) -> int:
        return len(self.images)


@dataclass(frozen=True)
class FedISICJobData:
    """Validated immutable inputs for one open-set-split x model-replicate job.

    The held-out (unknown) set is a TUPLE, not a single diagnosis: the sealed
    pre-registration's ``data.fed_isic2019.split_roster`` holds out a PAIR of
    classes per split (``split_semantics``: "Each roster entry holds out a PAIR of
    classes as unknown (2 of 8)").  A one-element tuple reproduces the legacy
    leave-one-diagnosis-out behaviour exactly.
    """

    config: MedicalDataConfig
    heldout_diagnoses: tuple[str, ...]
    centers: tuple[str, ...]
    diagnoses: tuple[str, ...]
    known_diagnoses: tuple[str, ...]
    center_to_client: Mapping[str, int]
    diagnosis_to_label: Mapping[str, int]
    training_by_client: tuple[tuple[MedicalImageRecord, ...], ...]
    units_by_fold: Mapping[str, tuple[MedicalAuditUnit, ...]]
    metadata_sha256: str
    fold_sha256: str

    @property
    def dataset_sha256(self) -> str:
        """Alias used by run manifests (the dataset input is the metadata CSV)."""

        return self.metadata_sha256

    @property
    def heldout_label(self) -> str:
        """Stable identity label for the held-out set, e.g. ``MEL+BCC``.

        Order follows ``heldout_diagnoses``, which is normalized to sorted order at
        load time, so the label is a deterministic function of the SET.
        """

        return "+".join(self.heldout_diagnoses)

    @property
    def heldout_diagnosis(self) -> str:
        """Back-compatible accessor for a single held-out diagnosis.

        Kept for the leave-one-out callers and artifacts that predate pair splits.
        Raises for a genuine pair rather than silently returning one of the two.
        """

        if len(self.heldout_diagnoses) != 1:
            raise ValueError(
                "this job holds out "
                f"{len(self.heldout_diagnoses)} diagnoses ({self.heldout_label}); "
                "use .heldout_diagnoses or .heldout_label"
            )
        return self.heldout_diagnoses[0]

    @property
    def n_clients(self) -> int:
        return len(self.centers)

    @property
    def n_known(self) -> int:
        return len(self.known_diagnoses)

    def fold_units(self, fold: str) -> tuple[MedicalAuditUnit, ...]:
        try:
            canonical = _FOLD_ALIASES[fold]
        except KeyError as exc:
            raise ValueError(f"unknown medical fold {fold!r}") from exc
        return self.units_by_fold.get(canonical, ())

    def validate_training_ready(self) -> None:
        """Fail before importing torch or touching an image."""

        if self.n_known < 1:
            raise ValueError(
                "a held-out diagnosis requires at least one known diagnosis"
            )
        if sum(len(records) for records in self.training_by_client) == 0:
            raise ValueError("no known-diagnosis image rows exist in the train fold")
        for fold in AUDIT_FOLDS:
            units = self.fold_units(fold)
            if not units:
                raise ValueError(f"frozen {fold} fold contains no audit units")
            if not any(unit.y_open == -1 for unit in units):
                raise ValueError(
                    f"frozen {fold} fold contains no labeled held-out-diagnosis unit"
                )
            if not any(unit.y_open >= 0 for unit in units):
                raise ValueError(f"frozen {fold} fold contains no known-diagnosis unit")
        # A traffic fold is OPTIONAL, and the sealed pre-registration declares none
        # for Fed-ISIC: data.folds spends the whole audit pool on
        # proposal/certification/test (0.40+0.30+0.30 = 1.0), and the pre-registered
        # certification path (grids.targets = [simplex, box]) never enters
        # run_oneshot_posthoc's traffic mixture mode.  Requiring one here belongs to
        # the SUPERSEDED 45/24 plan; a frozen fold artifact that carries traffic
        # units still validates, and one that omits them no longer fails closed.

        # This is deliberately an identity check, not an index-position check.
        fold_ids = {
            fold: {unit.sample_id for unit in self.fold_units(fold)}
            for fold in FOLD_ROLES
        }
        for i, left in enumerate(FOLD_ROLES):
            for right in FOLD_ROLES[i + 1 :]:
                overlap = fold_ids[left] & fold_ids[right]
                if overlap:
                    raise ValueError(
                        f"unit sample IDs overlap between {left} and {right}: "
                        f"{sorted(overlap)[:3]}"
                    )

    def summary(self) -> dict:
        return {
            "dataset": self.config.dataset_name,
            "dataset_sha256": self.dataset_sha256,
            "dataset_hash_scope": "metadata-csv-bytes",
            "metadata_sha256": self.metadata_sha256,
            "fold_sha256": self.fold_sha256,
            # ``heldout_diagnosis`` stays singular-valued for a leave-one-out job so
            # legacy consumers are unchanged; ``heldout_diagnoses`` is the general form.
            "heldout_diagnosis": (
                self.heldout_diagnoses[0]
                if len(self.heldout_diagnoses) == 1
                else self.heldout_label
            ),
            "heldout_diagnoses": list(self.heldout_diagnoses),
            "unit_kind": self.config.unit_kind,
            "n_centers": self.n_clients,
            "centers": list(self.centers),
            "n_diagnoses": len(self.diagnoses),
            "known_diagnoses": list(self.known_diagnoses),
            "train_images_by_center": {
                center: len(self.training_by_client[index])
                for index, center in enumerate(self.centers)
            },
            "units_by_fold": {fold: len(self.fold_units(fold)) for fold in FOLD_ROLES},
            "images_by_audit_fold": {
                fold: sum(unit.image_multiplicity for unit in self.fold_units(fold))
                for fold in AUDIT_FOLDS
            },
            "unknown_units_by_audit_fold": {
                fold: sum(unit.y_open == -1 for unit in self.fold_units(fold))
                for fold in AUDIT_FOLDS
            },
        }


def _read_csv(path: str, required: Iterable[str], *, name: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        if len(fields) != len(set(fields)):
            raise ValueError(f"{name} CSV contains duplicate column names")
        missing = set(required) - set(fields)
        if missing:
            raise ValueError(f"{name} CSV is missing columns: {sorted(missing)}")
        rows = []
        for row_number, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"too many fields in {name} CSV row {row_number}")
            rows.append(
                {
                    key: "" if value is None else str(value).strip()
                    for key, value in row.items()
                }
            )
    if not rows:
        raise ValueError(f"{name} CSV is empty")
    for row_number, row in enumerate(rows, 2):
        for column in required:
            if not row[column]:
                raise ValueError(f"blank {column!r} in {name} CSV row {row_number}")
    return rows


def _opaque_sample_id(dataset: str, kind: str, raw_id: str) -> str:
    payload = f"{dataset}\x00{kind}\x00{raw_id}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"{dataset}:{kind}:{digest[:32]}"


def make_unit_sample_id(dataset: str, unit_kind: str, unit_id: str) -> str:
    """Return a stable opaque identifier without exporting patient/lesion IDs."""

    if not dataset or not unit_kind or not unit_id:
        raise ValueError("dataset, unit_kind, and unit_id must be non-empty")
    return _opaque_sample_id(dataset, unit_kind, unit_id)


def _parse_json_ids(value: str, *, field: str, unit: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {field} for frozen unit {unit!r}") from exc
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) and item.strip() for item in parsed
    ):
        raise ValueError(f"{field} for frozen unit {unit!r} must be a JSON string list")
    values = tuple(item.strip() for item in parsed)
    if len(values) != len(set(values)):
        raise ValueError(f"{field} for frozen unit {unit!r} contains duplicates")
    return values


def _metadata_units(
    rows: Sequence[Mapping[str, str]], config: MedicalDataConfig
) -> dict[str, dict]:
    images: set[str] = set()
    units: dict[str, dict] = {}
    lesion_patients: dict[str, str] = {}
    resolved_paths: dict[str, str] = {}
    for row in rows:
        image_id = row[config.image_col]
        if image_id in images:
            raise ValueError(
                f"metadata image IDs must be unique; duplicate {image_id!r}"
            )
        images.add(image_id)
        unit_id = row[config.unit_col]
        lesion_id = row[config.lesion_col]
        patient_id = row[config.patient_col]
        prior_patient = lesion_patients.setdefault(lesion_id, patient_id)
        if prior_patient != patient_id:
            raise ValueError(f"lesion {lesion_id!r} maps to multiple patients")
        record = units.setdefault(
            unit_id,
            {
                "unit_id": unit_id,
                "center": row[config.center_col],
                "centers": set(),
                "diagnosis": row[config.diagnosis_col],
                "patient_id": patient_id,
                "lesion_ids": set(),
                "rows": [],
            },
        )
        # A unit that spans centers is NOT rejected here, because the real data
        # contains 9 such lesions (all MEL, across HAM vidir_molemax/vidir_modern)
        # and the sealed prereg's policy is to retain them IN TRAINING while
        # excluding them from the audit pool -- so rejecting them outright would
        # refuse the pre-registered design. Per-client independence is what the
        # single-center requirement protects, and that binds only on AUDIT units:
        # load_fed_isic_job enforces it there, and routes each training image to the
        # client of ITS OWN metadata row rather than to a unit-level center.
        record["centers"].add(row[config.center_col])
        if record["diagnosis"] != row[config.diagnosis_col]:
            raise ValueError(f"unit {unit_id!r} spans multiple diagnoses")
        if config.unit_kind == "lesion" and record["patient_id"] != patient_id:
            raise ValueError(f"lesion unit {unit_id!r} maps to multiple patients")
        path = config.resolve_image_path(row)
        prior_image = resolved_paths.setdefault(path, image_id)
        if prior_image != image_id:
            raise ValueError(
                f"different image IDs resolve to the same image path: "
                f"{prior_image!r}, {image_id!r}"
            )
        record["lesion_ids"].add(lesion_id)
        row_copy = dict(row)
        row_copy["__image_path__"] = path
        record["rows"].append(row_copy)
    return units


def _validate_frozen_folds(
    units: Mapping[str, Mapping],
    fold_rows: Sequence[Mapping[str, str]],
    config: MedicalDataConfig,
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Return ``unit_id -> (fold, frozen_image_ids)``.

    The frozen fold CSV is authoritative for WHICH images a unit contributes; see
    the ``image_ids_json`` subset rule below.
    """
    assignment: dict[str, tuple[str, tuple[str, ...]]] = {}
    for row in fold_rows:
        unit_id = row["audit_unit_id"]
        if unit_id in assignment:
            raise ValueError(
                f"frozen folds contain duplicate/overlapping unit {unit_id!r}"
            )
        fold = row["fold"]
        if fold not in FOLD_ROLES:
            raise ValueError(f"unknown frozen fold {fold!r} for unit {unit_id!r}")
        if unit_id not in units:
            raise ValueError(f"frozen unit {unit_id!r} is absent from metadata")
        metadata = units[unit_id]
        comparisons = {
            "diagnosis": metadata["diagnosis"],
            "patient_id": metadata["patient_id"],
        }
        # A cross-center unit has no single metadata center to compare against; the
        # frozen row records one of them. Checked only where it is well defined.
        if len(metadata["centers"]) == 1:
            comparisons["center"] = metadata["center"]
        for field, expected in comparisons.items():
            if row[field] != expected:
                raise ValueError(
                    f"frozen {field} disagrees with metadata for unit {unit_id!r}: "
                    f"{row[field]!r} != {expected!r}"
                )
        frozen_lesions = _parse_json_ids(
            row["lesion_ids_json"], field="lesion_ids_json", unit=unit_id
        )
        frozen_images = _parse_json_ids(
            row["image_ids_json"], field="image_ids_json", unit=unit_id
        )
        metadata_lesions = {str(value) for value in metadata["lesion_ids"]}
        metadata_images = {record[config.image_col] for record in metadata["rows"]}
        if set(frozen_lesions) != metadata_lesions:
            raise ValueError(
                f"frozen lesion IDs disagree with metadata for unit {unit_id!r}"
            )
        # SUBSET, not equality. Under the pre-registered audit design (A-002) a unit
        # contributes only the images its ROLE calls for: exactly ONE drawn image for
        # an audit unit ("draw EXACTLY ONE image per lesion"), and only the FLamby
        # train-side images for a train unit. Demanding equality here would forbid
        # the declared design. Every frozen image must still be a real image of this
        # unit, which is the property that actually prevents leakage.
        unknown_images = set(frozen_images) - metadata_images
        if unknown_images:
            raise ValueError(
                f"frozen image IDs are not images of unit {unit_id!r}: "
                f"{sorted(unknown_images)[:3]}"
            )
        if not frozen_images and fold != "unused":
            raise ValueError(
                f"frozen unit {unit_id!r} in fold {fold!r} carries no image; only "
                "the 'unused' accounting role may be empty"
            )
        assignment[unit_id] = (fold, frozen_images)

    missing = set(units) - set(assignment)
    if missing:
        raise ValueError(
            f"metadata units are missing from frozen folds: {sorted(missing)[:5]}"
        )
    extra = set(assignment) - set(units)
    if extra:  # defensive; the loop above already catches this case
        raise ValueError(f"frozen folds contain unknown units: {sorted(extra)[:5]}")
    return assignment


def normalize_heldout(heldout_diagnoses: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize a held-out spec to a sorted, deduplicated, non-empty tuple.

    Accepts a single diagnosis (legacy leave-one-out), a sequence, or a
    comma-separated string (``"MEL,BCC"``) as the roster spells pairs on a CLI.
    Sorting makes the resulting identity a function of the SET, so ``MEL,BCC`` and
    ``BCC,MEL`` are the same split and cannot produce two different experiment IDs.
    """

    if isinstance(heldout_diagnoses, str):
        parts = [part.strip() for part in heldout_diagnoses.split(",")]
    else:
        parts = []
        for item in heldout_diagnoses:
            if not isinstance(item, str):
                raise ValueError("held-out diagnoses must be strings")
            parts.extend(part.strip() for part in item.split(","))
    names = [part for part in parts if part]
    if not names:
        raise ValueError("heldout_diagnoses must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError(f"heldout_diagnoses contains duplicates: {names}")
    return tuple(sorted(names))


def load_fed_isic_job(
    config: MedicalDataConfig,
    heldout_diagnoses: str | Sequence[str],
    *,
    check_image_files: bool = False,
) -> FedISICJobData:
    """Load and validate one open-set-split Fed-ISIC job.

    ``heldout_diagnoses`` is the unknown-class set: a PAIR under the sealed
    pre-registration's roster, or a single diagnosis for the legacy
    leave-one-diagnosis-out behaviour.  Every held-out diagnosis maps to
    ``y_open = -1``; the knowns are relabelled contiguously from 0.

    ``check_image_files=False`` is the metadata-only pre-training/dry-run path.
    No image library or torch module is imported in either case; the optional
    check only verifies that every image needed by training/inference is a file.
    """

    config.validate()
    heldout = normalize_heldout(heldout_diagnoses)
    metadata_sha256 = file_sha256(config.metadata_csv)
    fold_sha256 = file_sha256(config.folds_csv)
    metadata_required = {
        config.center_col,
        config.diagnosis_col,
        config.patient_col,
        config.lesion_col,
        config.image_col,
        config.unit_col,
    }
    if config.image_path_col:
        metadata_required.add(config.image_path_col)
    metadata_rows = _read_csv(config.metadata_csv, metadata_required, name="metadata")
    units = _metadata_units(metadata_rows, config)
    fold_required = {
        "audit_unit_id",
        "fold",
        "center",
        "diagnosis",
        "patient_id",
        "lesion_ids_json",
        "image_ids_json",
    }
    fold_rows = _read_csv(config.folds_csv, fold_required, name="frozen folds")
    assignment = _validate_frozen_folds(units, fold_rows, config)
    if file_sha256(config.metadata_csv) != metadata_sha256:
        raise RuntimeError("metadata CSV changed while the job was being validated")
    if file_sha256(config.folds_csv) != fold_sha256:
        raise RuntimeError("frozen fold CSV changed while the job was being validated")

    centers = tuple(sorted({str(record["center"]) for record in units.values()}))
    diagnoses = tuple(sorted({str(record["diagnosis"]) for record in units.values()}))
    absent = [name for name in heldout if name not in diagnoses]
    if absent:
        raise ValueError(
            f"held-out diagnoses {absent!r} are absent; available={list(diagnoses)!r}"
        )
    heldout_set = set(heldout)
    known = tuple(diagnosis for diagnosis in diagnoses if diagnosis not in heldout_set)
    center_to_client = {center: index for index, center in enumerate(centers)}
    diagnosis_to_label = {diagnosis: index for index, diagnosis in enumerate(known)}

    by_fold: dict[str, list[MedicalAuditUnit]] = {fold: [] for fold in FOLD_ROLES}
    training: list[list[MedicalImageRecord]] = [[] for _ in centers]
    seen_sample_ids: set[str] = set()
    for unit_id in sorted(units):
        record = units[unit_id]
        fold, frozen_images = assignment[unit_id]
        diagnosis = str(record["diagnosis"])
        center = str(record["center"])
        # Per-client independence binds on AUDIT units: a unit in two centers cannot
        # be one client's draw. Training does not need it (each image is routed to
        # its own client below), which is what lets the prereg keep the 9 real
        # cross-center lesions in training instead of discarding them.
        if fold in AUDIT_FOLDS and len(record["centers"]) > 1:
            raise ValueError(
                f"audit unit {unit_id!r} spans multiple centers "
                f"{sorted(record['centers'])}; per-client independence would break"
            )
        client_id = center_to_client[center]
        y_open = -1 if diagnosis in heldout_set else diagnosis_to_label[diagnosis]
        unit_sample_id = make_unit_sample_id(
            config.dataset_name, config.unit_kind, unit_id
        )
        if unit_sample_id in seen_sample_ids:
            raise ValueError(f"unit sample-ID collision for {unit_id!r}")
        seen_sample_ids.add(unit_sample_id)
        # The FROZEN image list decides which images this unit contributes -- exactly
        # one for an audit unit, the train-side images for a train unit.
        frozen = set(frozen_images)
        image_records = []
        for row in sorted(record["rows"], key=lambda item: item[config.image_col]):
            image_id = row[config.image_col]
            if image_id not in frozen:
                continue
            image_center = str(row[config.center_col])
            image_records.append(
                MedicalImageRecord(
                    image_id=image_id,
                    image_path=row["__image_path__"],
                    image_sample_id=_opaque_sample_id(
                        config.dataset_name, "image", image_id
                    ),
                    unit_id=unit_id,
                    unit_sample_id=unit_sample_id,
                    center=image_center,
                    client_id=center_to_client[image_center],
                    diagnosis=diagnosis,
                    fold=fold,
                    label=y_open,
                )
            )
        audit_unit = MedicalAuditUnit(
            unit_id=unit_id,
            sample_id=unit_sample_id,
            center=center,
            client_id=client_id,
            diagnosis=diagnosis,
            y_open=y_open,
            fold=fold,
            patient_id=str(record["patient_id"]),
            lesion_ids=tuple(sorted(str(value) for value in record["lesion_ids"])),
            images=tuple(image_records),
        )
        by_fold[fold].append(audit_unit)
        # Unknown-diagnosis train rows are intentionally excluded, never remapped:
        # the model must never see a held-out class. Each image trains at the client
        # of ITS OWN metadata row, so a cross-center lesion's images stay with the
        # centers that actually acquired them.
        if fold == "train" and diagnosis not in heldout_set:
            for image in image_records:
                training[image.client_id].append(image)

    frozen_by_fold = {
        fold: tuple(sorted(records, key=lambda unit: unit.sample_id))
        for fold, records in by_fold.items()
    }
    frozen_training = tuple(
        tuple(sorted(records, key=lambda image: image.image_sample_id))
        for records in training
    )
    job = FedISICJobData(
        config=config,
        heldout_diagnoses=heldout,
        centers=centers,
        diagnoses=diagnoses,
        known_diagnoses=known,
        center_to_client=MappingProxyType(dict(center_to_client)),
        diagnosis_to_label=MappingProxyType(dict(diagnosis_to_label)),
        training_by_client=frozen_training,
        units_by_fold=MappingProxyType(dict(frozen_by_fold)),
        metadata_sha256=metadata_sha256,
        fold_sha256=fold_sha256,
    )
    job.validate_training_ready()
    if check_image_files:
        validate_required_image_files(job)
    return job


# A descriptive alias for callers that key on the dataset rather than the job cell.
load_fed_isic2019 = load_fed_isic_job


def validate_required_image_files(job: FedISICJobData) -> None:
    """Check every image used for training or proposal/cert/test inference."""

    paths: dict[str, str] = {}
    for records in job.training_by_client:
        for record in records:
            paths[record.image_sample_id] = record.image_path
    for fold in AUDIT_FOLDS:
        for unit in job.fold_units(fold):
            for record in unit.images:
                paths[record.image_sample_id] = record.image_path
    missing = [path for path in paths.values() if not os.path.isfile(path)]
    if missing:
        preview = ", ".join(repr(path) for path in sorted(missing)[:5])
        suffix = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise FileNotFoundError(
            f"required Fed-ISIC image files are missing: {preview}{suffix}"
        )


def flatten_unit_images(
    units: Sequence[MedicalAuditUnit],
) -> tuple[tuple[MedicalImageRecord, ...], np.ndarray]:
    """Flatten images while retaining their parent unit sample IDs."""

    images: list[MedicalImageRecord] = []
    parent_ids: list[str] = []
    for unit in units:
        if not unit.images:
            raise ValueError(f"audit unit {unit.sample_id!r} has no images")
        images.extend(unit.images)
        parent_ids.extend([unit.sample_id] * len(unit.images))
    return tuple(images), np.asarray(parent_ids, dtype=str)


def aggregate_unit_logits(
    image_logits: np.ndarray,
    image_unit_ids: Sequence[str],
    unit_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean image logits once per distinct unit.

    Returns ``(unit_logits, image_multiplicity)`` in ``unit_order``.  If no
    order is supplied, first appearance is used.  Duplicate entries in
    ``unit_order``, missing parents, non-finite logits, and unconsumed image rows
    fail closed.
    """

    try:
        logits = np.asarray(image_logits, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("image_logits must be numeric") from exc
    ids = np.asarray(image_unit_ids, dtype=str)
    if logits.ndim != 2:
        raise ValueError("image_logits must be a two-dimensional array")
    if logits.shape[1] == 0:
        raise ValueError("image_logits must contain at least one known-class column")
    if ids.ndim != 1:
        raise ValueError("image_unit_ids must be one-dimensional")
    if logits.shape[0] != ids.shape[0]:
        raise ValueError("image_logits and image_unit_ids must be aligned")
    if logits.shape[0] == 0:
        raise ValueError("cannot aggregate an empty image-logit array")
    if not np.all(np.isfinite(logits)):
        raise ValueError("image_logits contain a non-finite value")
    if np.any(ids == ""):
        raise ValueError("image_unit_ids must be non-empty")
    if unit_order is None:
        order = tuple(dict.fromkeys(ids.tolist()))
    else:
        order = tuple(str(value) for value in unit_order)
    if len(order) != len(set(order)):
        raise ValueError("unit_order contains duplicates")
    observed = set(ids.tolist())
    expected = set(order)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"unit/image parent mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    means = []
    multiplicities = []
    for unit_id in order:
        selected = logits[ids == unit_id]
        if (
            selected.shape[0] == 0
        ):  # structural after the set equality, retained defensively
            raise ValueError(f"unit {unit_id!r} has no image logits")
        means.append(selected.mean(axis=0, dtype=np.float64))
        multiplicities.append(selected.shape[0])
    return (
        np.asarray(means, dtype=np.float64),
        np.asarray(multiplicities, dtype=np.int64),
    )


aggregate_repeated_image_logits = aggregate_unit_logits


def audit_artifact_arrays(
    units: Sequence[MedicalAuditUnit],
    image_logits: np.ndarray,
    heldout_diagnoses: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Build one row per patient/lesion for a proposal/cert/test artifact.

    ``heldout_diagnoses`` is the job's declared unknown set (a PAIR under the
    pre-registered roster).  When supplied it is ENFORCED: ``y_open = -1`` must hold
    exactly on those diagnoses, so a fold that silently lost one member of the pair,
    or mapped a known class to -1, fails closed.  When omitted the set is inferred
    from the fold, which still catches a -1 leaking onto an unexpected diagnosis but
    cannot notice a pair member that is entirely absent.
    """

    units = tuple(units)
    if not units:
        raise ValueError("cannot export an empty audit fold")
    _, parent_ids = flatten_unit_images(units)
    order = [unit.sample_id for unit in units]
    logits, multiplicity = aggregate_unit_logits(image_logits, parent_ids, order)
    y_open = np.asarray([unit.y_open for unit in units], dtype=np.int64)
    if np.any(y_open < -1):
        raise ValueError("y_open labels must be -1 or a non-negative known-class index")
    observed_heldout = {unit.diagnosis for unit in units if unit.y_open == -1}
    if heldout_diagnoses is None:
        if not observed_heldout:
            raise ValueError("an audit fold must map at least one diagnosis to y_open=-1")
        expected_heldout = observed_heldout
    else:
        expected_heldout = set(normalize_heldout(heldout_diagnoses))
    # RATIFICATION_004: a fold that is missing a DECLARED unknown class (A3
    # count-starvation) no longer hard-crashes. The pre-registration requires such
    # cells to be PRESERVED as documented A3-infeasible findings, never dropped or
    # re-drawn. We record the missing support as structured fail-closed metadata and
    # still export the canonical logit artifact. Downstream, per-fold support flags
    # gate proposal support, certification A3 eligibility, and evaluation-metric
    # definability INDEPENDENTLY. Note this only ever ADDS to observed_heldout being a
    # subset of expected_heldout; a -1 leaking onto an UNexpected diagnosis (a genuine
    # corruption) is still caught by the exclusivity guard below, because that would put
    # a non-declared diagnosis into observed_heldout.
    missing_heldout = expected_heldout - observed_heldout
    unexpected_heldout = observed_heldout - expected_heldout
    if unexpected_heldout:
        raise ValueError(
            "audit fold maps an UNDECLARED diagnosis to y_open=-1: "
            f"unexpected={sorted(unexpected_heldout)}, declared={sorted(expected_heldout)}"
        )
    # Exclusivity: y_open=-1 must fall exactly on the declared unknowns that are
    # PRESENT in this fold (i.e. observed_heldout). A known class mapped to -1, or a
    # declared-unknown unit left at a known label, still fails closed.
    present_heldout = expected_heldout - missing_heldout
    diagnosis_is_heldout = np.asarray(
        [unit.diagnosis in present_heldout for unit in units], dtype=bool
    )
    if not np.array_equal(y_open == -1, diagnosis_is_heldout):
        raise RuntimeError("y_open=-1 is not exclusive to the held-out diagnoses")
    return {
        "logits": logits,
        "y_open": y_open,
        "client": np.asarray([unit.client_id for unit in units], dtype=np.int64),
        "sample_id": np.asarray(order, dtype=str),
        "image_multiplicity": multiplicity,
        # RATIFICATION_004 support metadata (per fold):
        "declared_heldout": np.asarray(sorted(expected_heldout), dtype=str),
        "observed_heldout": np.asarray(sorted(observed_heldout), dtype=str),
        "missing_heldout": np.asarray(sorted(missing_heldout), dtype=str),
        "support_complete": np.asarray(len(missing_heldout) == 0, dtype=bool),
    }


def traffic_identity_arrays(units: Sequence[MedicalAuditUnit]) -> dict[str, np.ndarray]:
    """Export only opaque sample identity and public site/client identity."""

    units = tuple(units)
    if not units:
        raise ValueError("cannot export an empty traffic fold")
    return {
        "sample_id": np.asarray([unit.sample_id for unit in units], dtype=str),
        "client": np.asarray([unit.client_id for unit in units], dtype=np.int64),
        "site_id": np.asarray([unit.center for unit in units], dtype=str),
    }


class MedicalImageDataset:
    """Minimal Dataset protocol with lazy PIL loading.

    The class intentionally does not import or subclass torch Dataset.  FedAvg
    and ``export_logits`` only need ``__len__``/``__getitem__``, allowing the
    metadata and dry-run paths to import this module on torch-free hosts.
    """

    def __init__(
        self,
        records: Sequence[MedicalImageRecord],
        *,
        transform: Callable | None = None,
        image_loader: Callable[[str], object] | None = None,
    ) -> None:
        self.records = tuple(records)
        self.transform = transform
        self.image_loader = image_loader or self._pil_loader

    @staticmethod
    def _pil_loader(path: str):
        from PIL import Image

        with Image.open(path) as image:
            return image.convert("RGB").copy()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = self.image_loader(record.image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, record.label


__all__ = [
    "AUDIT_FOLDS",
    "FOLD_ROLES",
    "OUTPUT_FOLD_NAMES",
    "FedISICJobData",
    "MedicalAuditUnit",
    "MedicalDataConfig",
    "MedicalImageDataset",
    "MedicalImageRecord",
    "aggregate_repeated_image_logits",
    "aggregate_unit_logits",
    "audit_artifact_arrays",
    "flatten_unit_images",
    "load_fed_isic2019",
    "load_fed_isic_job",
    "make_unit_sample_id",
    "normalize_heldout",
    "traffic_identity_arrays",
    "validate_required_image_files",
]
