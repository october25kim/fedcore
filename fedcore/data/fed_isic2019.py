"""Fed-ISIC2019 dataset layer: deterministic, FLamby-compatible center assignment.

WP-C data layer.  Metadata only -- this module never opens an image and never
imports torch.  It answers exactly one question: *which of the J=6 federated
centers does each ISIC-2019 image belong to, and what is its diagnosis?*

Licensing
---------
ISIC-2019 is distributed under **CC-BY-NC** (non-commercial).  See
https://challenge.isic-archive.com/data/ .  The derived center assignment in
this module is metadata only; redistribution of the images themselves is
governed by that license.  HAM10000 (Tschandl et al., 2018) is likewise CC-BY-NC
and is one of the three source archives pooled into ISIC-2019.

Provenance of the center definition
-----------------------------------
Centers are NOT invented here and NOT recalled from memory.  They replicate
``owkin/FLamby`` verbatim, from
``flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py``::

    ISIC_2019_Training_Metadata["dataset"] = ISIC_2019_Training_Metadata["lesion_id"].str[:4]
    result = pd.merge(ISIC_2019_Training_Metadata, HAM10000_metadata, how="left", on="image")
    result["dataset"] = result["dataset_x"] + result["dataset_y"].astype(str)

Three facts follow, each verified against the real metadata (see
:func:`verify_center_sizes`):

1. Rows with a null ``lesion_id`` have no attributable acquisition site.  FLamby
   *deletes* them (image, metadata row, and ground-truth row).  This drops
   25331 -> 23247 images.  The 2084 dropped images are exactly the MEL/NV/BKL
   surplus that distinguishes the raw class counts (MEL 4522, NV 12875,
   BKL 2624) from the center-attributable ones (MEL 4185, NV 11326, BKL 2426).

   *Deliberate deviation:* FLamby ``rm``s those 2084 JPEGs from disk.  This
   module deletes nothing -- it excludes them at the table level instead.  The
   table returned by :func:`load_center_table` is the single source of truth for
   membership, so the surplus files on disk are inert and the raw download stays
   re-usable without a re-fetch.  Consumers must therefore select images via the
   table, never by globbing the image directory.
2. ``lesion_id.str[:4]`` yields three source archives: ``BCN_`` (12413),
   ``HAM_`` (10015), ``MSK4`` (819).  Note ``MSK4`` -- MSK lesion ids are
   ``MSK4_*``, so the 4-character prefix keeps the digit.  This is why the
   concatenated center name is ``MSK4nan`` and not ``MSK_nan``.
3. ``.astype(str)`` on the merged HAM column turns missing values into the
   literal string ``"nan"``, so non-HAM rows become ``BCN_nan`` / ``MSK4nan``,
   while HAM rows expand into their four HAM10000 acquisition sub-sites.

**HAM10000 contributes FOUR sub-centers, not three**: ``vidir_molemax`` (3954),
``vidir_modern`` (3363), ``rosendahl`` (2259), ``vienna_dias`` (439).  They sum
to exactly 10015 = |HAM10000|.  J = 1 (BCN) + 4 (HAM) + 1 (MSK) = 6.

Known data defects (surfaced, not silently patched)
---------------------------------------------------
:func:`cross_center_lesions` reports 9 HAM lesions whose images are split across
``vidir_molemax`` and ``vidir_modern``.  Consumers that treat the *lesion* as the
independent audit unit (``fedcore.medical.data``) must resolve these, because a
unit that lives in two centers violates per-client independence.  4 of the 9 also
straddle FLamby's own official train/test fold, i.e. the upstream split has
lesion-level leakage for those 4.  This module does not choose a resolution; it
reports the defect.

Required inputs (under ``root``, default ``<repo>/data/isic2019``)
-----------------------------------------------------------------
* ``ISIC_2019_Training_Metadata.csv``   -- ISIC archive (has ``lesion_id``)
* ``ISIC_2019_Training_GroundTruth.csv``-- ISIC archive (one-hot diagnoses)
* ``HAM10000_metadata.csv``             -- ships in the FLamby repo; supplies the
  ``dataset`` column that expands HAM into its four sub-sites
* ``FLamby_train_test_split.csv``       -- optional; FLamby's official split, used
  by :func:`verify_center_sizes` as an independent oracle
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

#: The 8 ISIC-2019 lesion classes, in ground-truth column order.  ``UNK`` is a
#: 9th column in the raw ground truth but is identically zero on the training
#: set and is therefore not a class.
CLASSES: Tuple[str, ...] = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")

#: Center name -> FLamby center index.  Verified image-by-image against FLamby's
#: shipped ``train_test_split`` for all 23247 images: the map is a bijection and
#: agrees on every row (see :func:`verify_center_sizes`).
CENTER_NAME_TO_INDEX: Dict[str, int] = {
    "BCN_nan": 0,
    "HAM_vidir_molemax": 1,
    "HAM_vidir_modern": 2,
    "HAM_rosendahl": 3,
    "MSK4nan": 4,
    "HAM_vienna_dias": 5,
}

#: Index-ordered center names.
CENTER_NAMES: Tuple[str, ...] = tuple(
    name for name, _ in sorted(CENTER_NAME_TO_INDEX.items(), key=lambda kv: kv[1])
)

#: Human-readable acquisition site per center index.
CENTER_DESCRIPTIONS: Tuple[str, ...] = (
    "BCN_20000 / Hospital Clinic de Barcelona",
    "HAM10000 / ViDIR MoleMax (Vienna)",
    "HAM10000 / ViDIR Modern (Vienna)",
    "HAM10000 / Rosendahl (Queensland)",
    "MSK / Memorial Sloan Kettering",
    "HAM10000 / ViDIR Dias (Vienna)",
)

#: FLamby's documented per-center image counts (train + test), by center index.
DOCUMENTED_CENTER_SIZES: Tuple[int, ...] = (12413, 3954, 3363, 2259, 819, 439)

#: FLamby's documented per-center TRAIN counts (the 80% split), by center index.
DOCUMENTED_TRAIN_SIZES: Tuple[int, ...] = (9930, 3163, 2691, 1807, 655, 351)

#: Total center-attributable images (25331 raw minus 2084 with null lesion_id).
N_IMAGES_TOTAL: int = 23247

N_CENTERS: int = 6


def default_root() -> str:
    """``<repo>/data/isic2019`` -- resolved from this file, not the cwd."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "data", "isic2019")


# --------------------------------------------------------------------------- #
# center assignment (the FLamby replication)
# --------------------------------------------------------------------------- #
def assign_center_names(
    metadata: pd.DataFrame, ham_metadata: pd.DataFrame
) -> pd.Series:
    """Return the FLamby center *name* per row of ``metadata``.

    Replicates ``download_isic.py`` exactly.  ``metadata`` must already have had
    its null-``lesion_id`` rows dropped (see :func:`drop_unattributable`); this
    function does not drop anything, so the returned Series is row-aligned with
    the input.

    Deterministic: no RNG, no ordering dependence -- the value is a pure function
    of ``(lesion_id, image)``.
    """
    for column in ("image", "lesion_id"):
        if column not in metadata.columns:
            raise ValueError(f"metadata is missing required column {column!r}")
    if metadata["lesion_id"].isnull().any():
        raise ValueError(
            "metadata still contains null lesion_id rows; call drop_unattributable() "
            "first -- FLamby removes these images entirely"
        )

    ham = ham_metadata.rename(columns={"image_id": "image"})
    if "dataset" not in ham.columns:
        raise ValueError("HAM10000 metadata is missing the 'dataset' column")
    ham = ham[["image", "dataset"]]
    if ham["image"].duplicated().any():
        raise ValueError(
            "HAM10000 metadata has duplicate image ids; merge would fan out"
        )

    # FLamby: dataset := first 4 chars of lesion_id  ("BCN_", "HAM_", "MSK4")
    prefix = metadata["lesion_id"].str[:4]

    # FLamby: left-merge HAM sub-site, then string-concatenate.  Missing HAM
    # values become the literal "nan" via .astype(str) -- this is load-bearing,
    # it is what produces the BCN_nan / MSK4nan center names.
    merged = pd.merge(
        metadata[["image"]].assign(_prefix=prefix.values), ham, how="left", on="image"
    )
    if len(merged) != len(metadata):
        raise ValueError("HAM merge changed the row count; duplicate image ids?")

    # pandas 3 preserves a missing scalar through string arithmetic instead of
    # producing the historical literal ``"nan"``.  Spell out the FLamby-era
    # conversion so the center map is stable across supported pandas versions.
    ham_site = merged["dataset"].fillna("nan").astype(str)
    names = merged["_prefix"].astype(str) + ham_site
    names.index = metadata.index
    return names.rename("center_name")


def drop_unattributable(
    metadata: pd.DataFrame, ground_truth: Optional[pd.DataFrame] = None
):
    """Drop rows with a null ``lesion_id`` (no attributable acquisition site).

    FLamby deletes these 2084 images outright.  If ``ground_truth`` is given it
    is filtered by the same positional mask and returned alongside; the two
    frames are checked for image alignment first (FLamby assumes it silently).
    """
    keep = ~metadata["lesion_id"].isnull()
    meta_kept = metadata[keep].copy()
    if ground_truth is None:
        return meta_kept
    if len(ground_truth) != len(metadata):
        raise ValueError("metadata and ground_truth have different row counts")
    gt_kept = ground_truth[keep.values].copy()
    if not (meta_kept["image"].values == gt_kept["image"].values).all():
        raise ValueError("metadata/ground_truth image columns are not aligned")
    return meta_kept, gt_kept


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_center_table(root: Optional[str] = None) -> pd.DataFrame:
    """Build the canonical per-image center + diagnosis table.

    Returns a DataFrame with one row per center-attributable image and columns:

    ``image``       ISIC image id
    ``lesion_id``   source lesion id (the natural audit unit; repeats)
    ``center_name`` FLamby center name, e.g. ``HAM_rosendahl``
    ``center``      FLamby center index in ``[0, 6)``
    ``diagnosis``   one of :data:`CLASSES`
    ``target``      index of ``diagnosis`` in :data:`CLASSES`

    Deterministic and RNG-free; row order follows the ISIC metadata file.
    """
    root = root or default_root()
    meta_path = os.path.join(root, "ISIC_2019_Training_Metadata.csv")
    gt_path = os.path.join(root, "ISIC_2019_Training_GroundTruth.csv")
    ham_path = os.path.join(root, "HAM10000_metadata.csv")
    for path in (meta_path, gt_path, ham_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found. See this module's docstring for required inputs "
                "(ISIC archive CSVs + FLamby's HAM10000_metadata)."
            )

    metadata = pd.read_csv(meta_path)
    ground_truth = pd.read_csv(gt_path)
    ham = pd.read_csv(ham_path)

    metadata, ground_truth = drop_unattributable(metadata, ground_truth)
    center_name = assign_center_names(metadata, ham)

    unknown_names = set(center_name.unique()) - set(CENTER_NAME_TO_INDEX)
    if unknown_names:
        raise ValueError(
            f"unexpected center name(s) {sorted(unknown_names)}; the FLamby center "
            "definition does not cover this metadata"
        )

    onehot = ground_truth[list(CLASSES)].values
    if (onehot.sum(axis=1) != 1).any():
        raise ValueError("ground truth is not one-hot over the 8 classes")
    target = onehot.argmax(axis=1)

    table = pd.DataFrame(
        {
            "image": metadata["image"].values,
            "lesion_id": metadata["lesion_id"].values,
            "center_name": center_name.values,
            "center": [CENTER_NAME_TO_INDEX[n] for n in center_name.values],
            "diagnosis": [CLASSES[t] for t in target],
            "target": target,
        }
    )
    return table


def load_official_train_test_split(
    root: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """FLamby's shipped official split, or ``None`` if not staged locally."""
    root = root or default_root()
    path = os.path.join(root, "FLamby_train_test_split.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)[["image", "center", "fold"]]


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def center_sizes(table: pd.DataFrame) -> List[int]:
    """Per-center image counts, ordered by center index."""
    counts = table["center"].value_counts()
    return [int(counts.get(j, 0)) for j in range(N_CENTERS)]


def verify_center_sizes(
    table: Optional[pd.DataFrame] = None, root: Optional[str] = None
) -> Dict:
    """Check the derived assignment against FLamby's documented sizes.

    Also cross-checks against FLamby's shipped official split when available --
    an independent oracle that validates the *index* mapping, not just the sizes.
    """
    table = load_center_table(root) if table is None else table
    derived = center_sizes(table)
    report: Dict = {
        "n_images": int(len(table)),
        "n_images_expected": N_IMAGES_TOTAL,
        "derived_sizes": derived,
        "documented_sizes": list(DOCUMENTED_CENTER_SIZES),
        "sizes_match": derived == list(DOCUMENTED_CENTER_SIZES),
        "total_match": len(table) == N_IMAGES_TOTAL,
    }
    official = load_official_train_test_split(root)
    if official is None:
        report["official_split_available"] = False
    else:
        merged = table[["image", "center"]].merge(
            official.rename(columns={"center": "center_official"}),
            on="image",
            how="inner",
        )
        report["official_split_available"] = True
        report["official_n_joined"] = int(len(merged))
        report["official_center_agreement"] = bool(
            len(merged) == len(table)
            and (merged["center"] == merged["center_official"]).all()
        )
        train = official[official["fold"] == "train"]["center"].value_counts()
        report["official_train_sizes"] = [
            int(train.get(j, 0)) for j in range(N_CENTERS)
        ]
        report["documented_train_sizes"] = list(DOCUMENTED_TRAIN_SIZES)
    return report


def cross_center_lesions(table: pd.DataFrame) -> pd.DataFrame:
    """Lesions whose images span more than one center (a unit-independence defect).

    Returns the offending rows (empty if none).  See the module docstring: the
    real data has 9 such HAM lesions.  Reported, not silently repaired.
    """
    spans = table.groupby("lesion_id")["center"].nunique()
    bad = spans[spans > 1].index
    return table[table["lesion_id"].isin(bad)].sort_values(["lesion_id", "center"])


def attach_official_fold(
    table: Optional[pd.DataFrame] = None, root: Optional[str] = None
) -> pd.DataFrame:
    """Add FLamby's official ``train``/``test`` fold label to every image row.

    The sealed pre-registration (``data.fed_isic2019.audit_unit``, AMENDMENT A-002)
    makes this split the TRAIN/AUDIT BOUNDARY: a center's FLamby TRAIN images are
    its training pool, its FLamby TEST images are its audit pool.  The split is
    therefore no longer merely an oracle for :func:`verify_center_sizes`; it is
    required input, so this raises rather than returning ``None`` when it is absent.
    """
    table = load_center_table(root) if table is None else table
    official = load_official_train_test_split(root)
    if official is None:
        raise FileNotFoundError(
            "FLamby_train_test_split.csv is required: the sealed prereg's "
            "audit_unit.train_audit_boundary is FLamby's canonical per-center "
            "train/test split. Stage it under the isic2019 root."
        )
    merged = table.merge(official[["image", "fold"]], on="image", how="left")
    if len(merged) != len(table):
        raise ValueError("joining FLamby's official split changed the row count")
    if merged["fold"].isna().any():
        missing = int(merged["fold"].isna().sum())
        raise ValueError(
            f"{missing} image(s) have no fold in FLamby's official split; the "
            "train/audit boundary would be undefined for them"
        )
    bad = set(merged["fold"].unique()) - {"train", "test"}
    if bad:
        raise ValueError(f"unexpected FLamby fold label(s): {sorted(bad)}")
    return merged


def straddling_lesions(table_with_fold: pd.DataFrame) -> List[str]:
    """Lesions with images on BOTH sides of FLamby's official train/test split.

    A second upstream lesion-level defect, of the same kind as
    :func:`cross_center_lesions` but far larger: the real data has 2,302 of 11,847
    such lesions.  Under the pre-registered lesion-as-audit-unit design they are a
    train/audit LEAK -- the same physical lesion would be both trained on and
    audited -- so the prereg (``audit_unit.straddling_lesions``) excludes them from
    the audit pool and retains them in training only.  This function reports the
    defect; the exclusion policy lives in ``fedcore.experiments.isic_source_data``.

    ``table_with_fold`` must carry a ``fold`` column (see :func:`attach_official_fold`).
    """
    if "fold" not in table_with_fold.columns:
        raise ValueError("table must carry a 'fold' column; see attach_official_fold")
    spans = table_with_fold.groupby("lesion_id")["fold"].nunique()
    return sorted(str(lesion) for lesion in spans[spans > 1].index)


def class_counts_by_center(table: pd.DataFrame) -> pd.DataFrame:
    """Center index x class contingency table (rows 0..5, columns :data:`CLASSES`)."""
    counts = pd.crosstab(table["center"], table["diagnosis"])
    return counts.reindex(index=range(N_CENTERS), columns=list(CLASSES), fill_value=0)


if __name__ == "__main__":  # pragma: no cover - manual verification entry point
    import json

    tbl = load_center_table()
    print(json.dumps(verify_center_sizes(tbl), indent=2, default=str))
    print("\nclass counts by center:")
    print(class_counts_by_center(tbl).to_string())
    bad = cross_center_lesions(tbl)
    print(f"\ncross-center lesions: {bad['lesion_id'].nunique()} ({len(bad)} images)")
