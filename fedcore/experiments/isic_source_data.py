"""Emit the canonical Fed-ISIC2019 source-data CSVs the runner consumes.

Torch-free; images are never opened.  Everything here is DERIVED, never invented:
the center assignment, the cross-center defect, and the audit-supply projection all
come from modules that already verified themselves against FLamby.

Reused, not duplicated
----------------------
* :mod:`fedcore.data.fed_isic2019` -- ``load_center_table`` (the FLamby center
  replication, verified against FLamby's ``train_test_split`` oracle on all 23,247
  images) and ``cross_center_lesions`` (the 9-lesion defect).
* :mod:`fedcore.experiments.isic_preflight` -- ``project_center`` /
  ``_largest_remainder``, the audit-supply projection whose default reproduces the
  sealed prereg's ``a3_preflight_finding`` exactly.

Outputs
-------
``--emit-metadata PATH``
    The canonical per-image metadata CSV for
    :class:`fedcore.medical.data.MedicalDataConfig`.  Fully determined by the
    sealed prereg; no free parameters.

``--emit-folds PATH``
    The frozen per-unit fold assignment, per the sealed prereg's
    ``data.fed_isic2019.audit_unit`` (AMENDMENT A-002).  The train/audit boundary is
    FLamby's canonical per-center train/test split, so this emitter no longer needs
    (or accepts) a ``--max-known-audit-frac``: see :func:`assign_folds`.

Column contract (metadata CSV)
------------------------------
``center``      ``"<flamby_index>_<center_name>"``, e.g. ``0_BCN_nan``.
    The index prefix is load-bearing, not decoration.
    :func:`fedcore.medical.data.load_fed_isic_job` assigns ``client_id`` by
    ``sorted()`` order over the center strings.  Sorting the bare FLamby names
    yields ``BCN_nan, HAM_rosendahl, HAM_vidir_modern, HAM_vidir_molemax,
    HAM_vienna_dias, MSK4nan`` -- which maps client 1 to rosendahl, not molemax.
    That silently permutes the prereg's ``group_partition_G2``
    (``HAM_derived: [1,2,3,5]``, ``other: [0,4]``) and its per-center A3 verdicts.
    The zero-padded index prefix makes lexicographic order equal FLamby index order.
``diagnosis``   one of the 8 FLamby classes.
``lesion_id``   the audit unit (prereg: "the audit unit is the lesion").
``patient_id``  **placeholder, equal to ``lesion_id``.**  ISIC-2019 distributes no
    patient identifier (``ISIC_2019_Training_Metadata.csv`` columns are
    ``image, age_approx, anatom_site_general, lesion_id, sex``; HAM10000's
    metadata adds no patient key either).  The lesion is therefore the finest
    available unit, which is what the prereg fixes as the audit unit.  The column
    exists only because ``MedicalDataConfig`` requires one.  **Do not pass
    ``--unit-col patient_id``**: it would assert one-lesion-per-patient, which
    HAM10000 contradicts, and would silently coarsen the audit unit.
``image_id``    ISIC image id; ``<image_id>.jpg`` under ``--image-root``.
``flamby_fold``  ``train``/``test`` -- FLamby's canonical split, which AMENDMENT
    A-002 makes the TRAIN/AUDIT BOUNDARY.
``audit_eligible``  ``1`` iff this image may enter the audit pool: it is on the
    ``test`` side AND its lesion is neither cross-center nor straddling.
``audit_exclusion_reason``  why an image is not audit-eligible (``flamby_train``,
    ``cross_center_lesion``, ``straddling_lesion``), else empty.  Accounting, so
    nothing is silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Mapping, Sequence

import numpy as np

from fedcore.data.fed_isic2019 import (
    CENTER_NAME_TO_INDEX,
    CLASSES,
    N_CENTERS,
    attach_official_fold,
    cross_center_lesions,
    load_center_table,
    straddling_lesions,
)
from fedcore.experiments.isic_preflight import (
    DEFAULT_FOLD_FRACTIONS,
    DEFAULT_UNKNOWN_FRAC,
    FOLD_NAMES,
    _largest_remainder,
    project_center,
)

METADATA_FIELDS = (
    "image_id",
    "lesion_id",
    "patient_id",
    "center",
    "center_index",
    "center_name",
    "diagnosis",
    "flamby_fold",
    "audit_eligible",
    "audit_exclusion_reason",
)

FOLD_FIELDS = (
    "audit_unit_id",
    "fold",
    "center",
    "diagnosis",
    "patient_id",
    "lesion_ids_json",
    "image_ids_json",
)


def center_label(index: int, name: str) -> str:
    """``"0_BCN_nan"`` -- lexicographic order equals FLamby index order."""
    return f"{int(index)}_{name}"


def build_metadata_rows(root: str | None = None) -> list[dict]:
    """The canonical per-image metadata rows. Deterministic; no RNG, no images.

    Audit eligibility implements ``data.fed_isic2019.audit_unit`` exactly:

    * ``audit_pool``: "that center's FLamby TEST images" -- so a ``train``-side image
      is never audit-eligible; it is training supply.
    * ``straddling_lesions``: the 2,302 lesions with images on BOTH sides of FLamby's
      split are "EXCLUDED from the audit pool and retained in training only".
    * ``cross_center_lesions.policy``: the same treatment for the 9 cross-center ones.
    """
    table = attach_official_fold(load_center_table(root), root)
    cross = set(cross_center_lesions(table)["lesion_id"].unique())
    straddle = set(straddling_lesions(table))
    index_to_name = {index: name for name, index in CENTER_NAME_TO_INDEX.items()}
    rows = []
    for record in table.itertuples(index=False):
        index = int(record.center)
        lesion = str(record.lesion_id)
        flamby_fold = str(record.fold)
        # Order matters only for the REASON string, not for eligibility: a lesion may
        # be both cross-center and straddling (4 of the 9 are). The lesion-level
        # defects are reported first because they exclude the lesion outright,
        # whereas flamby_train merely routes the image to training.
        if lesion in cross:
            reason = "cross_center_lesion"
        elif lesion in straddle:
            reason = "straddling_lesion"
        elif flamby_fold != "test":
            reason = "flamby_train"
        else:
            reason = ""
        rows.append(
            {
                "image_id": str(record.image),
                "lesion_id": lesion,
                # Placeholder: ISIC-2019 ships no patient key. See module docstring.
                "patient_id": lesion,
                "center": center_label(index, index_to_name[index]),
                "center_index": index,
                "center_name": index_to_name[index],
                "diagnosis": str(record.diagnosis),
                "flamby_fold": flamby_fold,
                "audit_eligible": 0 if reason else 1,
                "audit_exclusion_reason": reason,
            }
        )
    return rows


def _write_csv(path: str, fields: Sequence[str], rows: Sequence[Mapping]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def unit_table(rows: Sequence[Mapping]) -> Dict[str, dict]:
    """Collapse image rows into lesion units (the audit unit).

    Each unit keeps its images SPLIT BY FLamby SIDE, because the two sides play
    different roles under ``audit_unit``: ``train_image_ids`` is training supply,
    ``audit_image_ids`` is the pool from which exactly one image per lesion is drawn.
    A cross-center lesion spans centers by definition, so ``center`` is recorded from
    its first row and the span is reported by ``cross_center_lesions`` rather than
    raising here.
    """
    units: Dict[str, dict] = {}
    for row in rows:
        lesion = str(row["lesion_id"])
        unit = units.setdefault(
            lesion,
            {
                "lesion_id": lesion,
                "patient_id": str(row["patient_id"]),
                "center": str(row["center"]),
                "center_index": int(row["center_index"]),
                "diagnosis": str(row["diagnosis"]),
                "audit_eligible": int(row["audit_eligible"]),
                "image_ids": [],
                "train_image_ids": [],
                "audit_image_ids": [],
            },
        )
        if unit["diagnosis"] != str(row["diagnosis"]):
            raise ValueError(f"lesion {lesion!r} spans diagnoses")
        if int(row["audit_eligible"]) != unit["audit_eligible"]:
            raise ValueError(f"lesion {lesion!r} has inconsistent audit eligibility")
        unit["image_ids"].append(str(row["image_id"]))
        if str(row["flamby_fold"]) == "train":
            unit["train_image_ids"].append(str(row["image_id"]))
        elif int(row["audit_eligible"]) == 1:
            unit["audit_image_ids"].append(str(row["image_id"]))
    for unit in units.values():
        for key in ("image_ids", "train_image_ids", "audit_image_ids"):
            unit[key].sort()
    return units


def assign_folds(
    units: Mapping[str, Mapping],
    unknown_classes: Sequence[str],
    *,
    seed: int,
    unknown_frac: float = DEFAULT_UNKNOWN_FRAC,
    fold_fractions: Sequence[float] = DEFAULT_FOLD_FRACTIONS,
) -> Dict[str, tuple[str, tuple[str, ...]]]:
    """Assign every lesion unit to train / proposal / certification / test / unused.

    Implements ``data.fed_isic2019.audit_unit`` (AMENDMENT A-002) literally.  Returns
    ``lesion_id -> (fold, image_ids)``: the fold decides WHICH images the unit
    contributes, so the images are returned with it rather than re-derived.

    The declared design, clause by clause
    -------------------------------------
    * ``train_audit_boundary``: FLamby's canonical per-center train/test split.
    * ``train_pool``: "that center's FLamby TRAIN images, known classes only" -- a
      known-class unit with train-side images becomes a ``train`` unit carrying
      exactly its train-side images.  Unknown-class units are NEVER trained on.
    * ``audit_pool``: "that center's FLamby TEST images, both known and unknown
      classes", minus the straddling and cross-center lesions.
    * ``sampling``: "group audit-pool images by lesion_id, draw EXACTLY ONE image per
      lesion into the 0.40/0.30/0.30 folds" -- so an audit unit carries exactly one
      image and ``aggregate_unit_logits`` sees multiplicity 1.  This is what makes
      ``K_j | A_j ~ Bin(A_j, r_j)`` an i.i.d. binomial rather than a correlated sum.
    * ``audit_unknown_fraction`` (0.30) sizes the pool, per ``audit_folds_note``
      ("applies to Fed-ISIC exactly as to CIFAR").  Units not drawn are ``unused``
      and are still emitted as accounting rows.

    Why there is no ``max_known_audit_frac`` any more
    ------------------------------------------------
    That parameter existed because nothing declared the train/audit boundary, and
    diverting a known unit into the audit pool cost training data.  A-002 declares
    the boundary, and it dissolves the trade-off: training draws from the FLamby
    TRAIN side and the audit draws from the FLamby TEST side, so a test-side known
    unit that the audit does not draw is simply ``unused`` -- it was never training
    supply to begin with.  There is no longer a fraction to choose, which is exactly
    why the runner's surviving blocker is now cleared rather than guessed.

    Determinism: the only RNG is ``default_rng(seed)``, consumed in a fixed order --
    per center, permute knowns then unknowns, then draw one image per selected unit.
    """
    unknown_set = {str(name) for name in unknown_classes}
    if not unknown_set:
        raise ValueError("unknown_classes must be non-empty")
    rng = np.random.default_rng(int(seed))
    assignment: Dict[str, tuple[str, tuple[str, ...]]] = {}

    for center in range(N_CENTERS):
        local = [u for u in units.values() if int(u["center_index"]) == center]

        # --- non-audit units: training supply, or nothing --------------------
        # Straddling / cross-center / FLamby-train-side lesions all land here.
        # A known-class unit contributes its TRAIN-side images; everything else,
        # including every unknown-class unit, is unused. Applied before any RNG.
        for unit in local:
            if int(unit["audit_eligible"]) == 1:
                continue
            lesion = str(unit["lesion_id"])
            train_images = tuple(unit["train_image_ids"])
            if str(unit["diagnosis"]) not in unknown_set and train_images:
                assignment[lesion] = ("train", train_images)
            else:
                assignment[lesion] = ("unused", ())

        # --- audit-eligible units: one image per lesion into the folds -------
        eligible = [u for u in local if int(u["audit_eligible"]) == 1]
        knowns = sorted(
            (u for u in eligible if str(u["diagnosis"]) not in unknown_set),
            key=lambda u: str(u["lesion_id"]),
        )
        unknowns = sorted(
            (u for u in eligible if str(u["diagnosis"]) in unknown_set),
            key=lambda u: str(u["lesion_id"]),
        )
        # The pool is counted in LESIONS, not images: one image per lesion means the
        # unit count is the image count, which is the whole point of A-002.
        projection = project_center(
            len(knowns),
            len(unknowns),
            unknown_frac=unknown_frac,
            fold_fractions=fold_fractions,
            max_known_audit_frac=1.0,
        )
        n_known_pool = int(projection["pool_known"])
        n_unknown_pool = int(projection["pool_unknown"])

        known_order = [knowns[i] for i in rng.permutation(len(knowns))]
        unknown_order = [unknowns[i] for i in rng.permutation(len(unknowns))]

        # Audit-eligible units are on the FLamby TEST side, so a unit the audit does
        # not draw cannot fall back to training: it is unused, not train.
        for unit in known_order[n_known_pool:] + unknown_order[n_unknown_pool:]:
            assignment[str(unit["lesion_id"])] = ("unused", ())

        for pool, count in (
            (known_order[:n_known_pool], n_known_pool),
            (unknown_order[:n_unknown_pool], n_unknown_pool),
        ):
            sizes = _largest_remainder(count, fold_fractions)
            start = 0
            for name, size in zip(FOLD_NAMES, sizes):
                for unit in pool[start : start + size]:
                    candidates = list(unit["audit_image_ids"])
                    if not candidates:
                        raise RuntimeError(
                            f"audit-eligible lesion {unit['lesion_id']!r} has no "
                            "test-side image; eligibility and images disagree"
                        )
                    # "draw EXACTLY ONE image per lesion" -- a seeded draw, not
                    # always-first, so acquisition order cannot bias the fold.
                    drawn = candidates[int(rng.integers(len(candidates)))]
                    assignment[str(unit["lesion_id"])] = (name, (drawn,))
                start += size

    missing = set(units) - set(assignment)
    if missing:
        raise RuntimeError(f"unassigned units: {sorted(missing)[:5]}")
    return assignment


def build_fold_rows(
    units: Mapping[str, Mapping],
    assignment: Mapping[str, tuple[str, Sequence[str]]],
) -> list[dict]:
    """Frozen fold rows -- one per lesion, INCLUDING ``unused``.

    Every lesion gets a row.  ``unused`` is an explicit accounting fold, not an
    omission: the prereg's prohibitions require that units be "REPORTED with
    accounting rows, never silently dropped", and an emitted ``unused`` row lets a
    reader reconcile 11,847 lesions exactly.  ``image_ids_json`` carries the images
    the unit actually contributes in its role -- ONE drawn image for an audit unit,
    the train-side images for a train unit -- which is a SUBSET of the lesion's
    images in general.
    """
    import json

    rows = []
    for lesion_id in sorted(units):
        fold, image_ids = assignment[lesion_id]
        unit = units[lesion_id]
        rows.append(
            {
                "audit_unit_id": lesion_id,
                "fold": fold,
                "center": unit["center"],
                "diagnosis": unit["diagnosis"],
                "patient_id": unit["patient_id"],
                "lesion_ids_json": json.dumps([lesion_id]),
                "image_ids_json": json.dumps(list(image_ids)),
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="data/isic2019 directory")
    parser.add_argument("--emit-metadata", default=None, help="output metadata CSV")
    parser.add_argument("--emit-folds", default=None, help="output frozen folds CSV")
    parser.add_argument(
        "--unknown-classes", default=None, help="comma-separated 2-of-8 unknown pair"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--unknown-frac", type=float, default=DEFAULT_UNKNOWN_FRAC)
    args = parser.parse_args(argv)

    rows = build_metadata_rows(args.root)
    if args.emit_metadata:
        _write_csv(args.emit_metadata, METADATA_FIELDS, rows)
        reasons: Dict[str, int] = {}
        for row in rows:
            if row["audit_exclusion_reason"]:
                key = str(row["audit_exclusion_reason"])
                reasons[key] = reasons.get(key, 0) + 1
        eligible = sum(1 for row in rows if row["audit_eligible"])
        print(
            f"[isic_source_data] metadata -> {args.emit_metadata} "
            f"({len(rows)} images, {len({r['lesion_id'] for r in rows})} lesions; "
            f"{eligible} audit-eligible images over "
            f"{len({r['lesion_id'] for r in rows if r['audit_eligible']})} lesions; "
            f"excluded {reasons})"
        )

    if args.emit_folds:
        if args.unknown_classes is None or args.seed is None:
            raise SystemExit("--emit-folds requires --unknown-classes and --seed")
        unknown = [c.strip() for c in args.unknown_classes.split(",") if c.strip()]
        bad = set(unknown) - set(CLASSES)
        if bad:
            raise SystemExit(f"unknown classes {sorted(bad)} are not ISIC classes")
        units = unit_table(rows)
        assignment = assign_folds(
            units,
            unknown,
            seed=args.seed,
            unknown_frac=args.unknown_frac,
        )
        fold_rows = build_fold_rows(units, assignment)
        _write_csv(args.emit_folds, FOLD_FIELDS, fold_rows)
        counts: Dict[str, int] = {}
        images: Dict[str, int] = {}
        for row in fold_rows:
            import json as _json

            counts[row["fold"]] = counts.get(row["fold"], 0) + 1
            images[row["fold"]] = images.get(row["fold"], 0) + len(
                _json.loads(row["image_ids_json"])
            )
        print(
            f"[isic_source_data] folds -> {args.emit_folds} "
            f"(units {counts}; images {images})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
