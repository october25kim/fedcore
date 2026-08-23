"""Office-Home natural-domain metadata joins and open-set fold assembly.

This module is deliberately torch-free.  It consumes the FROZEN, owner-hashed
Office-Home artifacts and never regenerates or mutates them:

* ``results/officehome/dedup/retained_canonical_manifest.csv`` -- the immutable
  sample identity table (``sample_id`` derived from normalized path + content
  hash).  15080 images after conflict-aware content dedup.
* ``results/officehome/preflight/class_splits.csv`` -- per split, which of the
  65 shared classes are ``known`` (45) vs ``unknown`` (20).  The class
  permutation comes from a SHA-256 semantic key, never from counts/scores.
* ``results/officehome/folds/folds_officehome_split_{0..4}.csv`` -- per sample,
  its role in ``{train, proposal, certification, traffic, evaluation}``.

One loaded job is exactly one immutable ``(class_split, domain-clients)`` unit.
J = 4 clients = domains ``{Art, Clipart, Product, Real_World}``.  A sample's
open-set label ``y_open`` is the contiguous known-class index when its class is
known in this split, or ``-1`` when its class is one of the split's unknowns.
Unknown-class images never appear in the train role (the model must never see a
held-out class); they appear only in the audit roles.

Traffic exports carry opaque sample identity and public domain identity only --
never a label, image, or model output -- so the traffic-derived deployment
mixture is constructed without touching any outcome.

Fail-closed on: missing artifact, provenance mismatch (a fold row whose
domain/class disagrees with the manifest), duplicate sample IDs, any sample
outside the manifest, an unknown class leaking into the train role, an audit
role missing known or unknown support, or any pairwise role-identity overlap.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import os
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from fedcore.campaign.artifacts import file_sha256


# Sorted domain order fixes the client index (Art=0, Clipart=1, Product=2,
# Real_World=3).  The manifest normalizes the domain with an underscore; the
# frozen dataset directory spells "Real World" with a space.
DOMAINS: tuple[str, ...] = ("Art", "Clipart", "Product", "Real_World")
DOMAIN_TO_DIR: Mapping[str, str] = MappingProxyType(
    {"Art": "Art", "Clipart": "Clipart", "Product": "Product", "Real_World": "Real World"}
)

FOLD_ROLES: tuple[str, ...] = (
    "train",
    "proposal",
    "certification",
    "traffic",
    "evaluation",
)
#: Roles that carry a labelled audit observation (logits + y_open exported).
AUDIT_ROLES: tuple[str, ...] = ("proposal", "certification", "evaluation")
#: Role that carries opaque identity only (no label / no model output).
TRAFFIC_ROLE: str = "traffic"
OUTPUT_FOLD_NAMES: Mapping[str, str] = MappingProxyType(
    {"proposal": "prop", "certification": "cert", "evaluation": "eval"}
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class OfficeHomeDataError(ValueError):
    """Raised on any fail-closed Office-Home provenance/identity violation."""


@dataclass(frozen=True)
class OfficeHomeDataConfig:
    """Path + column contract for the frozen Office-Home artifacts."""

    manifest_csv: str
    folds_csv: str
    class_splits_csv: str
    split_id: str
    image_root: str
    dataset_name: str = "officehome"
    dataset_version: str = "officehome_dedup_v1"

    def validate(self) -> None:
        for path in (self.manifest_csv, self.folds_csv, self.class_splits_csv):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
        if not os.path.isdir(self.image_root):
            raise FileNotFoundError(self.image_root)
        if not str(self.split_id).strip():
            raise OfficeHomeDataError("split_id must be non-empty")
        if not str(self.dataset_name).strip():
            raise OfficeHomeDataError("dataset_name must be non-empty")

    def resolve_image_path(self, domain: str, normalized_relpath: str) -> str:
        """Map a manifest relpath to an on-disk path (Real_World -> 'Real World')."""
        parts = str(normalized_relpath).split("/")
        if len(parts) < 2:
            raise OfficeHomeDataError(
                f"malformed normalized_relpath {normalized_relpath!r}"
            )
        directory = DOMAIN_TO_DIR.get(domain, domain)
        return os.path.abspath(os.path.join(self.image_root, directory, *parts[1:]))


@dataclass(frozen=True)
class OfficeHomeImageRecord:
    """One physical image with its immutable manifest identity and role."""

    sample_id: str
    image_path: str
    domain: str
    client_id: int
    klass: str
    role: str
    y_open: int


@dataclass(frozen=True)
class OfficeHomeJobData:
    """Validated immutable inputs for one open-set-split x domain-clients job."""

    config: OfficeHomeDataConfig
    split_id: str
    split_seed: int
    domains: tuple[str, ...]
    known_classes: tuple[str, ...]
    unknown_classes: tuple[str, ...]
    class_to_label: Mapping[str, int]
    training_by_client: tuple[tuple[OfficeHomeImageRecord, ...], ...]
    records_by_role: Mapping[str, tuple[OfficeHomeImageRecord, ...]]
    manifest_sha256: str
    folds_sha256: str
    class_splits_sha256: str

    @property
    def dataset_sha256(self) -> str:
        return self.manifest_sha256

    @property
    def n_clients(self) -> int:
        return len(self.domains)

    @property
    def n_known(self) -> int:
        return len(self.known_classes)

    def role_records(self, role: str) -> tuple[OfficeHomeImageRecord, ...]:
        if role not in FOLD_ROLES:
            raise OfficeHomeDataError(f"unknown Office-Home role {role!r}")
        return self.records_by_role.get(role, ())

    def validate_training_ready(self) -> None:
        """Fail before importing torch or opening an image."""

        if self.n_known < 1:
            raise OfficeHomeDataError("a split needs at least one known class")
        if sum(len(rec) for rec in self.training_by_client) == 0:
            raise OfficeHomeDataError("no known-class train rows exist")
        for records in self.training_by_client:
            for record in records:
                if record.y_open < 0:
                    raise OfficeHomeDataError(
                        "unknown-class image leaked into the train role"
                    )
        for role in AUDIT_ROLES:
            records = self.role_records(role)
            if not records:
                raise OfficeHomeDataError(f"audit role {role!r} is empty")
            if not any(r.y_open == -1 for r in records):
                raise OfficeHomeDataError(
                    f"audit role {role!r} has no labelled unknown-class support"
                )
            if not any(r.y_open >= 0 for r in records):
                raise OfficeHomeDataError(
                    f"audit role {role!r} has no known-class support"
                )
        if not self.role_records(TRAFFIC_ROLE):
            raise OfficeHomeDataError("traffic role is empty")

        # 10 pairwise role-identity disjointness assertions (C(5,2)=10). Each
        # sample carries exactly one role row, so this is guaranteed, but the
        # frozen-rules require it be ASSERTED, not assumed.
        ids = {
            role: {r.sample_id for r in self.role_records(role)} for role in FOLD_ROLES
        }
        for i, left in enumerate(FOLD_ROLES):
            for right in FOLD_ROLES[i + 1 :]:
                overlap = ids[left] & ids[right]
                if overlap:
                    raise OfficeHomeDataError(
                        f"role identity overlap {left}/{right}: {sorted(overlap)[:3]}"
                    )

    def summary(self) -> dict:
        return {
            "dataset": self.config.dataset_name,
            "dataset_version": self.config.dataset_version,
            "split_id": self.split_id,
            "split_seed": int(self.split_seed),
            "manifest_sha256": self.manifest_sha256,
            "folds_sha256": self.folds_sha256,
            "class_splits_sha256": self.class_splits_sha256,
            "domains": list(self.domains),
            "n_clients": self.n_clients,
            "n_known": self.n_known,
            "n_unknown": len(self.unknown_classes),
            "known_classes": list(self.known_classes),
            "unknown_classes": list(self.unknown_classes),
            "train_images_by_client": {
                domain: len(self.training_by_client[i])
                for i, domain in enumerate(self.domains)
            },
            "records_by_role": {
                role: len(self.role_records(role)) for role in FOLD_ROLES
            },
            "unknown_by_audit_role": {
                role: int(sum(r.y_open == -1 for r in self.role_records(role)))
                for role in AUDIT_ROLES
            },
            "traffic_by_client": {
                domain: int(
                    sum(r.client_id == i for r in self.role_records(TRAFFIC_ROLE))
                )
                for i, domain in enumerate(self.domains)
            },
        }


def _read_csv(path: str, required: Iterable[str], *, name: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        if len(fields) != len(set(fields)):
            raise OfficeHomeDataError(f"{name} CSV has duplicate column names")
        missing = set(required) - set(fields)
        if missing:
            raise OfficeHomeDataError(f"{name} CSV is missing columns: {sorted(missing)}")
        rows = []
        for row_number, row in enumerate(reader, 2):
            if None in row:
                raise OfficeHomeDataError(f"too many fields in {name} CSV row {row_number}")
            rows.append(
                {k: "" if v is None else str(v).strip() for k, v in row.items()}
            )
    if not rows:
        raise OfficeHomeDataError(f"{name} CSV is empty")
    for row_number, row in enumerate(rows, 2):
        for column in required:
            if not row[column]:
                raise OfficeHomeDataError(
                    f"blank {column!r} in {name} CSV row {row_number}"
                )
    return rows


def load_officehome_job(
    config: OfficeHomeDataConfig,
    *,
    check_image_files: bool = False,
) -> OfficeHomeJobData:
    """Load and validate one open-set Office-Home split job.

    ``check_image_files=False`` is the metadata-only dry-run path -- no PIL or
    torch import, no image opened.  The optional check verifies that every image
    required for training/inference is a real file on disk.
    """

    config.validate()
    manifest_sha256 = file_sha256(config.manifest_csv)
    folds_sha256 = file_sha256(config.folds_csv)
    class_splits_sha256 = file_sha256(config.class_splits_csv)

    manifest_rows = _read_csv(
        config.manifest_csv,
        {"sample_id", "domain", "klass", "filename", "normalized_relpath"},
        name="manifest",
    )
    manifest: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        sample_id = row["sample_id"]
        if sample_id in manifest:
            raise OfficeHomeDataError(f"duplicate manifest sample_id {sample_id!r}")
        if row["domain"] not in DOMAINS:
            raise OfficeHomeDataError(
                f"manifest domain {row['domain']!r} is not one of {DOMAINS}"
            )
        manifest[sample_id] = row

    class_rows = _read_csv(
        config.class_splits_csv, {"split_id", "seed", "role", "class"}, name="class_splits"
    )
    split_rows = [r for r in class_rows if r["split_id"] == config.split_id]
    if not split_rows:
        raise OfficeHomeDataError(f"split_id {config.split_id!r} absent from class_splits")
    seeds = {int(r["seed"]) for r in split_rows}
    if len(seeds) != 1:
        raise OfficeHomeDataError(f"split {config.split_id!r} has inconsistent seeds")
    split_seed = seeds.pop()
    known = tuple(sorted({r["class"] for r in split_rows if r["role"] == "known"}))
    unknown = tuple(sorted({r["class"] for r in split_rows if r["role"] == "unknown"}))
    if set(known) & set(unknown):
        raise OfficeHomeDataError("a class is both known and unknown in the split")
    all_classes = {r["klass"] for r in manifest.values()}
    stray = (set(known) | set(unknown)) - all_classes
    if stray:
        raise OfficeHomeDataError(f"class split names absent from manifest: {sorted(stray)}")
    class_to_label = {name: idx for idx, name in enumerate(known)}

    fold_rows = _read_csv(
        config.folds_csv, {"domain", "class", "role", "sample_id"}, name="folds"
    )
    assigned: dict[str, str] = {}
    by_role: dict[str, list[OfficeHomeImageRecord]] = {r: [] for r in FOLD_ROLES}
    training: list[list[OfficeHomeImageRecord]] = [[] for _ in DOMAINS]
    for row in fold_rows:
        sample_id = row["sample_id"]
        if sample_id in assigned:
            raise OfficeHomeDataError(
                f"fold sample_id {sample_id!r} appears in two roles "
                f"({assigned[sample_id]} and {row['role']})"
            )
        role = row["role"]
        if role not in FOLD_ROLES:
            raise OfficeHomeDataError(f"unknown fold role {role!r}")
        meta = manifest.get(sample_id)
        if meta is None:
            raise OfficeHomeDataError(f"fold sample_id {sample_id!r} absent from manifest")
        if row["domain"] != meta["domain"] or row["class"] != meta["klass"]:
            raise OfficeHomeDataError(
                f"fold row disagrees with manifest for {sample_id!r}: "
                f"({row['domain']},{row['class']}) != ({meta['domain']},{meta['klass']})"
            )
        assigned[sample_id] = role
        client_id = DOMAINS.index(meta["domain"])
        klass = meta["klass"]
        y_open = class_to_label[klass] if klass in class_to_label else -1
        record = OfficeHomeImageRecord(
            sample_id=sample_id,
            image_path=config.resolve_image_path(meta["domain"], meta["normalized_relpath"]),
            domain=meta["domain"],
            client_id=client_id,
            klass=klass,
            role=role,
            y_open=y_open,
        )
        by_role[role].append(record)
        if role == "train":
            if y_open < 0:
                raise OfficeHomeDataError(
                    f"unknown-class sample {sample_id!r} present in the train role"
                )
            training[client_id].append(record)

    if file_sha256(config.manifest_csv) != manifest_sha256:
        raise OfficeHomeDataError("manifest changed while the job was being validated")
    if file_sha256(config.folds_csv) != folds_sha256:
        raise OfficeHomeDataError("folds changed while the job was being validated")
    if file_sha256(config.class_splits_csv) != class_splits_sha256:
        raise OfficeHomeDataError("class_splits changed while the job was being validated")

    frozen_by_role = {
        role: tuple(sorted(records, key=lambda r: r.sample_id))
        for role, records in by_role.items()
    }
    frozen_training = tuple(
        tuple(sorted(records, key=lambda r: r.sample_id)) for records in training
    )
    job = OfficeHomeJobData(
        config=config,
        split_id=config.split_id,
        split_seed=split_seed,
        domains=DOMAINS,
        known_classes=known,
        unknown_classes=unknown,
        class_to_label=MappingProxyType(dict(class_to_label)),
        training_by_client=frozen_training,
        records_by_role=MappingProxyType(dict(frozen_by_role)),
        manifest_sha256=manifest_sha256,
        folds_sha256=folds_sha256,
        class_splits_sha256=class_splits_sha256,
    )
    job.validate_training_ready()
    if check_image_files:
        validate_required_image_files(job)
    return job


def validate_required_image_files(job: OfficeHomeJobData) -> None:
    """Check every image used for training or proposal/cert/eval inference."""

    paths: dict[str, str] = {}
    for records in job.training_by_client:
        for record in records:
            paths[record.sample_id] = record.image_path
    for role in AUDIT_ROLES:
        for record in job.role_records(role):
            paths[record.sample_id] = record.image_path
    missing = [p for p in paths.values() if not os.path.isfile(p)]
    if missing:
        preview = ", ".join(repr(p) for p in sorted(missing)[:5])
        suffix = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise FileNotFoundError(
            f"required Office-Home image files are missing: {preview}{suffix}"
        )


def audit_role_arrays(
    records: Sequence[OfficeHomeImageRecord],
    image_logits: np.ndarray,
    known_classes: Sequence[str],
    unknown_classes: Sequence[str],
) -> dict[str, np.ndarray]:
    """One row per image for a proposal/cert/eval artifact, ID-aligned to logits.

    Enforces the open-set label contract: ``y_open == -1`` exactly on this
    split's unknown classes.  A known class mapped to ``-1``, or an unknown-class
    row left at a known label, fails closed.
    """

    records = tuple(records)
    if not records:
        raise OfficeHomeDataError("cannot export an empty audit role")
    logits = np.asarray(image_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] != len(records):
        raise OfficeHomeDataError("image_logits must be (len(records), n_known)")
    if logits.shape[1] != len(known_classes):
        raise OfficeHomeDataError("image_logits width must equal the known-class count")
    if not np.all(np.isfinite(logits)):
        raise OfficeHomeDataError("image_logits contain a non-finite value")
    unknown_set = set(unknown_classes)
    y_open = np.asarray([r.y_open for r in records], dtype=np.int64)
    is_unknown = np.asarray([r.klass in unknown_set for r in records], dtype=bool)
    if not np.array_equal(y_open == -1, is_unknown):
        raise OfficeHomeDataError("y_open=-1 is not exclusive to the split's unknowns")
    return {
        "logits": logits,
        "y_open": y_open,
        "client": np.asarray([r.client_id for r in records], dtype=np.int64),
        "sample_id": np.asarray([r.sample_id for r in records], dtype=str),
        "klass": np.asarray([r.klass for r in records], dtype=str),
    }


def traffic_identity_arrays(
    records: Sequence[OfficeHomeImageRecord],
) -> dict[str, np.ndarray]:
    """Export opaque sample identity and public domain identity only."""

    records = tuple(records)
    if not records:
        raise OfficeHomeDataError("cannot export an empty traffic role")
    return {
        "sample_id": np.asarray([r.sample_id for r in records], dtype=str),
        "client": np.asarray([r.client_id for r in records], dtype=np.int64),
        "site_id": np.asarray([r.domain for r in records], dtype=str),
    }


def build_transforms(image_size: int = 224):
    """Return ``(train_transform, eval_transform)`` (torchvision, imported lazily).

    * train: ``RandomResizedCrop(image_size) + RandomHorizontalFlip`` + ImageNet norm.
    * eval: ``Resize(image_size * 256 // 224) + CenterCrop(image_size)`` + ImageNet norm.
    """

    import torchvision.transforms as transforms

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    resize = int(round(image_size * 256 / 224))
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluate = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train, evaluate


class OfficeHomeImageDataset:
    """Minimal ``__len__``/``__getitem__`` dataset with lazy PIL loading.

    Does not import or subclass torch, so metadata/dry-run paths import this
    module on torch-free hosts.
    """

    def __init__(
        self,
        records: Sequence[OfficeHomeImageRecord],
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
        label = record.y_open if record.y_open >= 0 else 0
        return image, label


def domain_client_roster_hash(domains: Sequence[str]) -> str:
    """Stable hash of the ordered domain->client roster (for seed provenance)."""

    payload = "\x00".join(str(d) for d in domains).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "AUDIT_ROLES",
    "DOMAINS",
    "DOMAIN_TO_DIR",
    "FOLD_ROLES",
    "OUTPUT_FOLD_NAMES",
    "TRAFFIC_ROLE",
    "OfficeHomeDataConfig",
    "OfficeHomeDataError",
    "OfficeHomeImageDataset",
    "OfficeHomeImageRecord",
    "OfficeHomeJobData",
    "audit_role_arrays",
    "build_transforms",
    "domain_client_roster_hash",
    "load_officehome_job",
    "traffic_identity_arrays",
    "validate_required_image_files",
]
