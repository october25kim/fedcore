"""Strict PACS data primitives for the sealed FedCORE v3 replication.

The module deliberately contains no experiment launcher.  It joins the two
registered CSV manifests, enforces the frozen train/proposal/audit separation,
and reads images directly from the single mounted PACS archive.  Torch,
torchvision, and Pillow are imported lazily so mount and authorization checks
can run without opening PACS or initializing a GPU runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import io
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence
import zipfile

from fedcore.experiments.v3_contract import CLIENTS, ContractError


PACS_CLASSES = (
    "dog",
    "elephant",
    "giraffe",
    "guitar",
    "horse",
    "house",
    "person",
)
ROLES = ("train", "proposal", "audit")
USES = ("train", "proposal", "audit", "unused")
IMAGE_MANIFEST_COLUMNS = (
    "path",
    "domain",
    "class",
    "identity",
    "width",
    "height",
    "role",
)
FOLD_MANIFEST_COLUMNS = (
    "split",
    "identity",
    "domain",
    "class",
    "role",
    "use",
    "label",
    "path",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PACSDataError(ContractError):
    """Raised when mounted PACS metadata diverges from the sealed frame."""


@dataclass(frozen=True)
class PACSRecord:
    """One registered image under one class split.

    ``role`` is the global source-identity partition.  ``use`` is the effective
    split-specific use.  Unknown-class records whose global role is ``train``
    must have ``use='unused'`` and are never reallocated.
    """

    split: int
    identity: str
    domain: str
    class_name: str
    role: str
    use: str
    label: int
    manifest_path: str
    archive_member: str
    width: int
    height: int


def _read_exact_csv(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise PACSDataError(
                f"{path.name} columns drift: expected {tuple(expected_columns)!r}, "
                f"observed {tuple(reader.fieldnames or ())!r}"
            )
        rows = list(reader)
    if not rows:
        raise PACSDataError(f"{path.name} is empty")
    return rows


def archive_member_from_manifest_path(value: str) -> str:
    """Map the frozen extracted-tree path to its member in ``PACS_mirror.zip``."""

    text = str(value).replace("\\", "/")
    marker = "PACS/"
    start = text.find(marker)
    if start < 0:
        raise PACSDataError(f"manifest path does not contain {marker!r}: {value!r}")
    member = PurePosixPath(text[start:])
    if member.is_absolute() or ".." in member.parts or member.parts[0] != "PACS":
        raise PACSDataError(f"unsafe PACS archive member: {value!r}")
    if len(member.parts) != 4:
        raise PACSDataError(f"unexpected PACS archive layout: {member.as_posix()!r}")
    return member.as_posix()


def _parse_nonnegative_int(value: str, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PACSDataError(f"invalid integer for {label}: {value!r}") from exc
    if result < 0:
        raise PACSDataError(f"negative integer for {label}: {result}")
    return result


def load_pacs_records(
    image_manifest: Path,
    fold_manifest: Path,
    *,
    split: int,
    known_classes: Sequence[str],
    expected_image_count: int | None = None,
) -> tuple[PACSRecord, ...]:
    """Join and validate the two mounted manifests for one registered split.

    The function does not open the PACS archive.  Returned records include
    ``unused`` rows so the caller can prove that unknown training records were
    retained in accounting rather than silently reassigned.
    """

    if split not in range(5):
        raise PACSDataError(f"split must be in [0,4], observed {split}")
    known = tuple(known_classes)
    if len(known) != 4 or tuple(sorted(known)) != known:
        raise PACSDataError("known_classes must contain four classes in alphabetical order")
    if not set(known).issubset(PACS_CLASSES):
        raise PACSDataError("known_classes contains a non-PACS class")
    unknown = set(PACS_CLASSES) - set(known)

    image_rows = _read_exact_csv(Path(image_manifest), IMAGE_MANIFEST_COLUMNS)
    if expected_image_count is not None and len(image_rows) != expected_image_count:
        raise PACSDataError(
            f"image denominator drift: expected {expected_image_count}, observed {len(image_rows)}"
        )
    images: dict[str, dict[str, str]] = {}
    global_roles: dict[str, set[str]] = {role: set() for role in ROLES}
    for row in image_rows:
        identity = row["identity"]
        if len(identity) != 64 or any(ch not in "0123456789abcdef" for ch in identity):
            raise PACSDataError(f"invalid decoded-image identity: {identity!r}")
        if identity in images:
            raise PACSDataError(f"duplicate decoded-image identity: {identity}")
        if row["domain"] not in CLIENTS:
            raise PACSDataError(f"unknown PACS domain: {row['domain']!r}")
        if row["class"] not in PACS_CLASSES:
            raise PACSDataError(f"unknown PACS class: {row['class']!r}")
        if row["role"] not in ROLES:
            raise PACSDataError(f"invalid global role: {row['role']!r}")
        _parse_nonnegative_int(row["width"], "width")
        _parse_nonnegative_int(row["height"], "height")
        archive_member_from_manifest_path(row["path"])
        images[identity] = row
        global_roles[row["role"]].add(identity)
    for left_index, left in enumerate(ROLES):
        for right in ROLES[left_index + 1 :]:
            if global_roles[left] & global_roles[right]:
                raise PACSDataError(f"global {left}/{right} identity intersection is nonempty")

    fold_rows = _read_exact_csv(Path(fold_manifest), FOLD_MANIFEST_COLUMNS)
    selected = [row for row in fold_rows if _parse_nonnegative_int(row["split"], "split") == split]
    if len(selected) != len(images):
        raise PACSDataError(
            f"split {split} fold denominator drift: expected {len(images)}, observed {len(selected)}"
        )
    records: list[PACSRecord] = []
    seen: set[str] = set()
    label_by_class = {name: index for index, name in enumerate(known)}
    for row in selected:
        identity = row["identity"]
        if identity in seen:
            raise PACSDataError(f"duplicate fold identity in split {split}: {identity}")
        seen.add(identity)
        image = images.get(identity)
        if image is None:
            raise PACSDataError(f"fold identity is absent from IMAGE_MANIFEST: {identity}")
        for field in ("domain", "class", "role", "path"):
            if row[field] != image[field]:
                raise PACSDataError(
                    f"manifest disagreement for {identity}, field {field}: "
                    f"{image[field]!r} != {row[field]!r}"
                )
        use = row["use"]
        if use not in USES:
            raise PACSDataError(f"invalid split use: {use!r}")
        label = int(row["label"])
        class_name = row["class"]
        if class_name in label_by_class:
            if use != row["role"] or label != label_by_class[class_name]:
                raise PACSDataError(
                    f"known-class use/label mismatch for {identity}: use={use!r}, label={label}"
                )
        else:
            if class_name not in unknown or label != -1:
                raise PACSDataError(f"unknown-class label mismatch for {identity}")
            expected_use = "unused" if row["role"] == "train" else row["role"]
            if use != expected_use:
                raise PACSDataError(
                    f"unknown record was reallocated for {identity}: expected {expected_use!r}, "
                    f"observed {use!r}"
                )
        records.append(
            PACSRecord(
                split=split,
                identity=identity,
                domain=row["domain"],
                class_name=class_name,
                role=row["role"],
                use=use,
                label=label,
                manifest_path=row["path"],
                archive_member=archive_member_from_manifest_path(row["path"]),
                width=_parse_nonnegative_int(image["width"], "width"),
                height=_parse_nonnegative_int(image["height"], "height"),
            )
        )
    if seen != set(images):
        raise PACSDataError(f"split {split} does not account for every image identity")
    return tuple(sorted(records, key=lambda item: item.identity))


def records_by_use_and_domain(
    records: Iterable[PACSRecord], use: str
) -> dict[str, tuple[PACSRecord, ...]]:
    """Return identity-sorted, nonempty client records for one effective use."""

    if use not in ROLES:
        raise PACSDataError(f"use must be one of {ROLES!r}")
    result: dict[str, tuple[PACSRecord, ...]] = {}
    rows = tuple(records)
    for domain in CLIENTS:
        subset = tuple(sorted((r for r in rows if r.domain == domain and r.use == use), key=lambda r: r.identity))
        if not subset:
            raise PACSDataError(f"empty registered {domain}/{use} stratum")
        if len({record.identity for record in subset}) != len(subset):
            raise PACSDataError(f"duplicate identity in {domain}/{use}")
        if use == "train" and any(record.label < 0 for record in subset):
            raise PACSDataError(f"unknown label entered {domain}/train")
        result[domain] = subset
    return result


class _ExifTransposeRGB:
    def __call__(self, image):
        from PIL import ImageOps

        return ImageOps.exif_transpose(image).convert("RGB")


def build_registered_transforms():
    """Return the exact registered ``(train, inference)`` torchvision transforms."""

    from torchvision import transforms as T

    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train = T.Compose(
        [
            _ExifTransposeRGB(),
            T.RandomResizedCrop(
                224,
                scale=(0.08, 1.0),
                ratio=(0.75, 4.0 / 3.0),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalize,
        ]
    )
    inference = T.Compose(
        [
            _ExifTransposeRGB(),
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ]
    )
    return train, inference


class PACSZipDataset:
    """Identity-ordered dataset that reads only registered members from one zip."""

    def __init__(
        self,
        archive_path: Path,
        records: Sequence[PACSRecord],
        *,
        transform: Callable | None,
        allow_unknown: bool,
    ) -> None:
        self.archive_path = Path(archive_path)
        self.records = tuple(records)
        self.transform = transform
        self.allow_unknown = bool(allow_unknown)
        self._archive: zipfile.ZipFile | None = None
        if not self.records:
            raise PACSDataError("PACSZipDataset cannot be empty")
        if len({row.identity for row in self.records}) != len(self.records):
            raise PACSDataError("PACSZipDataset contains duplicate identities")
        if not self.allow_unknown and any(row.label < 0 for row in self.records):
            raise PACSDataError("unknown label supplied to a known-only dataset")

    def __len__(self) -> int:
        return len(self.records)

    def _zip(self) -> zipfile.ZipFile:
        """Open the immutable archive once per dataset process.

        The registered loader uses ``num_workers=0``.  Keeping one read-only
        handle avoids reopening the 184 MB central directory for every sample
        without changing record order or decoded bytes.  ``__getstate__``
        deliberately drops the handle if the object is ever pickled.
        """

        if self._archive is None:
            self._archive = zipfile.ZipFile(self.archive_path, "r")
        return self._archive

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_archive"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, index: int):
        from PIL import Image, ImageOps

        record = self.records[index]
        try:
            payload = self._zip().read(record.archive_member)
        except KeyError as exc:
            raise PACSDataError(
                f"registered PACS member is absent: {record.archive_member}"
            ) from exc
        with Image.open(io.BytesIO(payload)) as image:
            raw = image.copy()
        decoded = ImageOps.exif_transpose(raw).convert("RGB")
        if decoded.size != (record.width, record.height):
            raise PACSDataError(
                f"decoded dimensions drift for {record.identity}: "
                f"expected {(record.width, record.height)}, observed {decoded.size}"
            )
        # The registered transform begins with EXIF transpose and RGB conversion.
        # Pass the untransposed copy to it so that operation occurs exactly once.
        value = self.transform(raw) if self.transform is not None else decoded
        return value, record.label


__all__ = [
    "FOLD_MANIFEST_COLUMNS",
    "IMAGE_MANIFEST_COLUMNS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PACS_CLASSES",
    "PACSDataError",
    "PACSRecord",
    "PACSZipDataset",
    "ROLES",
    "USES",
    "archive_member_from_manifest_path",
    "build_registered_transforms",
    "load_pacs_records",
    "records_by_use_and_domain",
]
