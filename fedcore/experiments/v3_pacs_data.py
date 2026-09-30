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
from typing import Callable, Iterable, Mapping, Sequence
import zipfile

from fedcore.experiments.v3_contract import CLIENTS, ContractError, sha256_file


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


@dataclass(frozen=True)
class PACSAuditFrameRecord:
    """Class-free audit reservoir atom available after a proposal seal.

    The object intentionally has no class, numeric label, manifest path, or
    archive member.  It is sufficient to freeze identity-sorted client
    reservoirs, denominators, and registered with-replacement draw indices.
    """

    split: int
    identity: str
    domain: str


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


def _parse_csv_values(record: str, *, path: Path, line_number: int) -> list[str]:
    """Parse one physical CSV record and attach fail-closed source context."""

    try:
        values = next(csv.reader([record], strict=True))
    except (csv.Error, StopIteration) as exc:
        raise PACSDataError(
            f"invalid CSV record in {path.name} at line {line_number}"
        ) from exc
    return values


def _routing_fields(
    record: str,
    *,
    indexes: Sequence[int],
    path: Path,
    line_number: int,
) -> dict[int, str]:
    """Decode only explicitly requested fields from one physical CSV record.

    Delimiter scanning is syntactic rather than semantic.  Fields crossed on
    the way to the largest requested index are not decoded or retained.  This
    is essential because ``class``, ``label``, and class-bearing ``path`` must
    remain unopened for a non-target stage even though they share a mounted
    physical CSV with routing metadata.
    """

    requested = tuple(sorted(set(int(value) for value in indexes)))
    if not requested or requested[0] < 0:
        raise PACSDataError("CSV routing indexes must be nonempty and nonnegative")
    maximum = requested[-1]
    spans: list[tuple[int, int]] = []
    start = 0
    in_quotes = False
    index = 0
    while index < len(record):
        character = record[index]
        if character == '"':
            if in_quotes and index + 1 < len(record) and record[index + 1] == '"':
                index += 2
                continue
            in_quotes = not in_quotes
        elif character == "," and not in_quotes:
            spans.append((start, index))
            start = index + 1
            if len(spans) > maximum:
                break
        index += 1
    if in_quotes or len(spans) <= maximum:
        raise PACSDataError(
            f"truncated CSV routing prefix in {path.name} at line {line_number}"
        )
    output: dict[int, str] = {}
    for field_index in requested:
        left, right = spans[field_index]
        values = _parse_csv_values(
            record[left:right], path=path, line_number=line_number
        )
        if len(values) != 1:
            raise PACSDataError(
                f"CSV routing field drift in {path.name} at line {line_number}"
            )
        output[field_index] = values[0]
    return output


def _validate_stage_request(
    *, split: int, known_classes: Sequence[str], stage: str
) -> tuple[str, ...]:
    """Validate the public stage-loader contract without opening a manifest."""

    if stage not in ROLES:
        raise PACSDataError(f"stage must be one of {ROLES!r}, observed {stage!r}")
    if split not in range(5):
        raise PACSDataError(f"split must be in [0,4], observed {split}")
    known = tuple(known_classes)
    if len(known) != 4 or tuple(sorted(known)) != known:
        raise PACSDataError("known_classes must contain four classes in alphabetical order")
    if not set(known).issubset(PACS_CLASSES):
        raise PACSDataError("known_classes contains a non-PACS class")
    return known


def _verify_seal_gate(
    seal_path: Path,
    expected_sha256: str,
    *,
    expected_name: str,
    label: str,
) -> None:
    """Fail closed before a later-stage manifest is opened."""

    path = Path(seal_path)
    expected = str(expected_sha256)
    if path.name != expected_name:
        raise PACSDataError(f"{label} access requires the exact {expected_name} path")
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise PACSDataError(f"{label} access requires a lowercase SHA-256 seal")
    if path.is_symlink() or not path.is_file():
        raise PACSDataError(f"{label} access refused: seal is absent or not regular")
    observed = sha256_file(path)
    if observed != expected:
        raise PACSDataError(
            f"{label} access refused: seal hash differs from the campaign state"
        )


def _verify_proposal_seal_gate(
    proposal_seal_path: Path,
    expected_proposal_seal_sha256: str,
) -> None:
    _verify_seal_gate(
        proposal_seal_path,
        expected_proposal_seal_sha256,
        expected_name="PROPOSAL_SEAL.json",
        label="audit-frame",
    )


def _verify_primary_seal_gate(
    primary_seal_path: Path,
    expected_primary_seal_sha256: str,
) -> None:
    _verify_seal_gate(
        primary_seal_path,
        expected_primary_seal_sha256,
        expected_name="PRIMARY_SEAL.json",
        label="full-audit-truth",
    )


def _load_pacs_stage_records(
    image_manifest: Path,
    fold_manifest: Path,
    *,
    split: int,
    known_classes: Sequence[str],
    stage: str,
    expected_image_count: int | None,
    selected_identities: frozenset[str] | None = None,
) -> tuple[PACSRecord, ...]:
    """Load exactly one stage without tokenizing another stage's labels.

    The full identity columns are streamed to verify the frozen denominator and
    the one-to-one manifest join. Crucially, ``row["label"]`` is converted to
    an integer only after both the registered split and the requested ``use``
    match. Rows from other stages expose only split, identity, domain, role,
    and use. Their semantic class, numeric label, and class-bearing path fields
    are neither decoded nor retained. The CSV bytes remain physically mounted
    and are scanned as raw records for routing because this is a procedural,
    rather than physical, isolation contract.
    """

    known = _validate_stage_request(
        split=split, known_classes=known_classes, stage=stage
    )
    label_by_class = {name: index for index, name in enumerate(known)}
    unknown = set(PACS_CLASSES) - set(known)

    selected: dict[str, dict[str, object]] = {}
    fold_identities: set[str] = set()
    split_row_count = 0
    fold_path = Path(fold_manifest)
    with fold_path.open("r", encoding="utf-8", newline="") as handle:
        header_record = handle.readline()
        header = _parse_csv_values(header_record, path=fold_path, line_number=1)
        if tuple(header) != FOLD_MANIFEST_COLUMNS:
            raise PACSDataError(
                f"{fold_path.name} columns drift: expected {FOLD_MANIFEST_COLUMNS!r}, "
                f"observed {tuple(header)!r}"
            )
        observed_any = False
        for line_number, record in enumerate(handle, start=2):
            if not record.strip():
                raise PACSDataError(
                    f"blank CSV record in {fold_path.name} at line {line_number}"
                )
            observed_any = True
            routing = _routing_fields(
                record,
                indexes=(0, 1, 2, 4, 5),
                path=fold_path,
                line_number=line_number,
            )
            row_split = _parse_nonnegative_int(routing[0], "split")
            if row_split != split:
                continue
            split_row_count += 1
            identity = routing[1]
            if len(identity) != 64 or any(
                ch not in "0123456789abcdef" for ch in identity
            ):
                raise PACSDataError(f"invalid fold identity: {identity!r}")
            if identity in fold_identities:
                raise PACSDataError(
                    f"duplicate fold identity in split {split}: {identity}"
                )
            fold_identities.add(identity)
            if routing[2] not in CLIENTS:
                raise PACSDataError(f"unknown PACS domain: {routing[2]!r}")
            if routing[4] not in ROLES:
                raise PACSDataError(f"invalid global role: {routing[4]!r}")
            if routing[5] not in USES:
                raise PACSDataError(f"invalid split use: {routing[5]!r}")

            # Stage-isolation boundary: do not inspect, convert, validate, or
            # retain the label cell for any other stage.  Only the prefix above
            # has been tokenized at this point.
            if routing[5] != stage or (
                selected_identities is not None and identity not in selected_identities
            ):
                continue
            values = _parse_csv_values(
                record, path=fold_path, line_number=line_number
            )
            if len(values) != len(FOLD_MANIFEST_COLUMNS):
                raise PACSDataError(
                    f"CSV width drift in {fold_path.name} at line {line_number}"
                )
            row = dict(zip(FOLD_MANIFEST_COLUMNS, values, strict=True))
            if row["class"] not in PACS_CLASSES:
                raise PACSDataError(f"unknown PACS class: {row['class']!r}")
            class_name = row["class"]
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise PACSDataError(
                    f"invalid {stage} label for {identity}: {row['label']!r}"
                ) from exc
            if row["role"] != stage:
                raise PACSDataError(
                    f"{stage} row has mismatched global role for {identity}: "
                    f"{row['role']!r}"
                )
            if class_name in label_by_class:
                if label != label_by_class[class_name]:
                    raise PACSDataError(
                        f"known-class {stage} label mismatch for {identity}: {label}"
                    )
            elif class_name not in unknown or label != -1:
                raise PACSDataError(
                    f"unknown-class {stage} label mismatch for {identity}: {label}"
                )
            if stage == "train" and label < 0:
                raise PACSDataError(f"unknown label entered train stage: {identity}")
            selected[identity] = {
                "domain": row["domain"],
                "class": class_name,
                "role": row["role"],
                "use": row["use"],
                "label": label,
                "path": row["path"],
            }
        if not observed_any:
            raise PACSDataError(f"{fold_path.name} is empty")

    if not selected:
        raise PACSDataError(f"split {split} contains no {stage} rows")
    if selected_identities is not None and set(selected) != set(selected_identities):
        missing = sorted(set(selected_identities) - set(selected))[:3]
        raise PACSDataError(
            f"requested {stage} truth identities are absent from the frame: {missing!r}"
        )

    image_path = Path(image_manifest)
    image_identities: set[str] = set()
    records: list[PACSRecord] = []
    image_row_count = 0
    with image_path.open("r", encoding="utf-8", newline="") as handle:
        header_record = handle.readline()
        header = _parse_csv_values(header_record, path=image_path, line_number=1)
        if tuple(header) != IMAGE_MANIFEST_COLUMNS:
            raise PACSDataError(
                f"{image_path.name} columns drift: expected {IMAGE_MANIFEST_COLUMNS!r}, "
                f"observed {tuple(header)!r}"
            )
        for line_number, record in enumerate(handle, start=2):
            if not record.strip():
                raise PACSDataError(
                    f"blank CSV record in {image_path.name} at line {line_number}"
                )
            image_row_count += 1
            identity = _routing_fields(
                record,
                indexes=(3,),
                path=image_path,
                line_number=line_number,
            )[3]
            if len(identity) != 64 or any(
                ch not in "0123456789abcdef" for ch in identity
            ):
                raise PACSDataError(f"invalid decoded-image identity: {identity!r}")
            if identity in image_identities:
                raise PACSDataError(f"duplicate decoded-image identity: {identity}")
            image_identities.add(identity)
            # Stream non-target rows for identity accounting only.  In
            # particular, do not decode their path or class fields.
            fold = selected.get(identity)
            if fold is None:
                continue
            values = _parse_csv_values(
                record, path=image_path, line_number=line_number
            )
            if len(values) != len(IMAGE_MANIFEST_COLUMNS):
                raise PACSDataError(
                    f"CSV width drift in {image_path.name} at line {line_number}"
                )
            row = dict(zip(IMAGE_MANIFEST_COLUMNS, values, strict=True))
            if row["domain"] not in CLIENTS:
                raise PACSDataError(f"unknown PACS domain: {row['domain']!r}")
            if row["role"] not in ROLES:
                raise PACSDataError(f"invalid global role: {row['role']!r}")
            if row["class"] not in PACS_CLASSES:
                raise PACSDataError(f"unknown PACS class: {row['class']!r}")
            width = _parse_nonnegative_int(row["width"], "width")
            height = _parse_nonnegative_int(row["height"], "height")
            archive_member = archive_member_from_manifest_path(row["path"])
            for field in ("domain", "class", "role", "path"):
                if row[field] != fold[field]:
                    raise PACSDataError(
                        f"manifest disagreement for {identity}, field {field}: "
                        f"{row[field]!r} != {fold[field]!r}"
                    )
            records.append(
                PACSRecord(
                    split=split,
                    identity=identity,
                    domain=row["domain"],
                    class_name=row["class"],
                    role=row["role"],
                    use=stage,
                    label=int(fold["label"]),
                    manifest_path=row["path"],
                    archive_member=archive_member,
                    width=width,
                    height=height,
                )
            )

    if image_row_count == 0:
        raise PACSDataError(f"{image_path.name} is empty")
    if expected_image_count is not None and image_row_count != expected_image_count:
        raise PACSDataError(
            f"image denominator drift: expected {expected_image_count}, "
            f"observed {image_row_count}"
        )
    if split_row_count != image_row_count:
        raise PACSDataError(
            f"split {split} fold denominator drift: expected {image_row_count}, "
            f"observed {split_row_count}"
        )
    if fold_identities != image_identities:
        missing = sorted(image_identities - fold_identities)[:3]
        extra = sorted(fold_identities - image_identities)[:3]
        raise PACSDataError(
            f"split {split} identity frame differs from IMAGE_MANIFEST: "
            f"missing={missing!r}, extra={extra!r}"
        )
    if {record.identity for record in records} != set(selected):
        raise PACSDataError(f"split {split} {stage} rows did not join one-to-one")
    return tuple(sorted(records, key=lambda item: item.identity))


def load_pacs_train_records(
    image_manifest: Path,
    fold_manifest: Path,
    *,
    split: int,
    known_classes: Sequence[str],
    expected_image_count: int | None = None,
) -> tuple[PACSRecord, ...]:
    """Load only known-class training rows; later-stage labels remain opaque."""

    return _load_pacs_stage_records(
        image_manifest,
        fold_manifest,
        split=split,
        known_classes=known_classes,
        stage="train",
        expected_image_count=expected_image_count,
    )


def load_pacs_proposal_records(
    image_manifest: Path,
    fold_manifest: Path,
    *,
    split: int,
    known_classes: Sequence[str],
    expected_image_count: int | None = None,
) -> tuple[PACSRecord, ...]:
    """Load only proposal rows; audit labels remain opaque."""

    return _load_pacs_stage_records(
        image_manifest,
        fold_manifest,
        split=split,
        known_classes=known_classes,
        stage="proposal",
        expected_image_count=expected_image_count,
    )


def _load_pacs_audit_frame(
    image_manifest: Path,
    fold_manifest: Path,
    *,
    split: int,
    expected_image_count: int | None = None,
) -> tuple[PACSAuditFrameRecord, ...]:
    """Load only identity, domain, and denominator metadata for audit routing."""

    fold_path = Path(fold_manifest)
    fold_identities: set[str] = set()
    audit_frame: list[PACSAuditFrameRecord] = []
    split_row_count = 0
    with fold_path.open("r", encoding="utf-8", newline="") as handle:
        header = _parse_csv_values(handle.readline(), path=fold_path, line_number=1)
        if tuple(header) != FOLD_MANIFEST_COLUMNS:
            raise PACSDataError(
                f"{fold_path.name} columns drift: expected {FOLD_MANIFEST_COLUMNS!r}, "
                f"observed {tuple(header)!r}"
            )
        observed_any = False
        for line_number, record in enumerate(handle, start=2):
            if not record.strip():
                raise PACSDataError(
                    f"blank CSV record in {fold_path.name} at line {line_number}"
                )
            observed_any = True
            routing = _routing_fields(
                record,
                indexes=(0, 1, 2, 4, 5),
                path=fold_path,
                line_number=line_number,
            )
            if _parse_nonnegative_int(routing[0], "split") != split:
                continue
            split_row_count += 1
            identity = routing[1]
            if len(identity) != 64 or any(
                character not in "0123456789abcdef" for character in identity
            ):
                raise PACSDataError(f"invalid fold identity: {identity!r}")
            if identity in fold_identities:
                raise PACSDataError(
                    f"duplicate fold identity in split {split}: {identity}"
                )
            fold_identities.add(identity)
            domain, role, use = routing[2], routing[4], routing[5]
            if domain not in CLIENTS:
                raise PACSDataError(f"unknown PACS domain: {domain!r}")
            if role not in ROLES or use not in USES:
                raise PACSDataError("invalid audit-frame role/use routing metadata")
            if use == "audit":
                if role != "audit":
                    raise PACSDataError(
                        f"audit row has mismatched global role for {identity}: {role!r}"
                    )
                audit_frame.append(PACSAuditFrameRecord(split, identity, domain))
        if not observed_any:
            raise PACSDataError(f"{fold_path.name} is empty")

    image_path = Path(image_manifest)
    image_identities: set[str] = set()
    image_row_count = 0
    with image_path.open("r", encoding="utf-8", newline="") as handle:
        header = _parse_csv_values(handle.readline(), path=image_path, line_number=1)
        if tuple(header) != IMAGE_MANIFEST_COLUMNS:
            raise PACSDataError(
                f"{image_path.name} columns drift: expected {IMAGE_MANIFEST_COLUMNS!r}, "
                f"observed {tuple(header)!r}"
            )
        for line_number, record in enumerate(handle, start=2):
            if not record.strip():
                raise PACSDataError(
                    f"blank CSV record in {image_path.name} at line {line_number}"
                )
            image_row_count += 1
            identity = _routing_fields(
                record,
                indexes=(3,),
                path=image_path,
                line_number=line_number,
            )[3]
            if len(identity) != 64 or any(
                character not in "0123456789abcdef" for character in identity
            ):
                raise PACSDataError(f"invalid decoded-image identity: {identity!r}")
            if identity in image_identities:
                raise PACSDataError(f"duplicate decoded-image identity: {identity}")
            image_identities.add(identity)

    if expected_image_count is not None and image_row_count != expected_image_count:
        raise PACSDataError(
            f"image denominator drift: expected {expected_image_count}, "
            f"observed {image_row_count}"
        )
    if split_row_count != image_row_count or fold_identities != image_identities:
        raise PACSDataError("audit frame identity/denominator drift")
    if not audit_frame:
        raise PACSDataError(f"split {split} contains no audit rows")
    result = tuple(sorted(audit_frame, key=lambda item: item.identity))
    audit_frame_by_domain(result)
    return result


def audit_frame_by_domain(
    records: Iterable[PACSAuditFrameRecord],
) -> dict[str, tuple[PACSAuditFrameRecord, ...]]:
    """Return source-identity-sorted, class-free audit reservoirs."""

    rows = tuple(records)
    if len({row.identity for row in rows}) != len(rows):
        raise PACSDataError("duplicate identity in the class-free audit frame")
    output: dict[str, tuple[PACSAuditFrameRecord, ...]] = {}
    for domain in CLIENTS:
        subset = tuple(sorted(
            (row for row in rows if row.domain == domain),
            key=lambda row: row.identity,
        ))
        if not subset:
            raise PACSDataError(f"empty registered {domain}/audit frame")
        output[domain] = subset
    if sum(len(rows) for rows in output.values()) != len(rows):
        raise PACSDataError("audit frame contains an unknown domain")
    return output


def load_pacs_audit_frame(
    image_manifest: Path,
    fold_manifest: Path,
    *,
    split: int,
    known_classes: Sequence[str],
    proposal_seal_path: Path,
    expected_proposal_seal_sha256: str,
    expected_image_count: int | None = None,
) -> tuple[PACSAuditFrameRecord, ...]:
    """Open class-free audit routing metadata after the per-cell proposal seal."""

    _verify_proposal_seal_gate(
        proposal_seal_path, expected_proposal_seal_sha256
    )
    _validate_stage_request(split=split, known_classes=known_classes, stage="audit")
    return _load_pacs_audit_frame(
        image_manifest,
        fold_manifest,
        split=split,
        expected_image_count=expected_image_count,
    )


def load_pacs_selected_audit_records(
    image_manifest: Path,
    fold_manifest: Path,
    *,
    split: int,
    known_classes: Sequence[str],
    audit_frame: Sequence[PACSAuditFrameRecord],
    indices: Mapping[str, Sequence[int]],
    draws_per_client: int,
    proposal_seal_path: Path,
    expected_proposal_seal_sha256: str,
    expected_image_count: int | None = None,
) -> tuple[PACSRecord, ...]:
    """Open truth only for unique identities selected by registered indices."""

    _verify_proposal_seal_gate(
        proposal_seal_path, expected_proposal_seal_sha256
    )
    if draws_per_client not in (64, 128, 256):
        raise PACSDataError("registered audit draw count must be 64, 128, or 256")
    observed_frame = _load_pacs_audit_frame(
        image_manifest,
        fold_manifest,
        split=split,
        expected_image_count=expected_image_count,
    )
    if tuple(audit_frame) != observed_frame:
        raise PACSDataError("selected audit truth request differs from the frozen frame")
    frame_by_domain = audit_frame_by_domain(audit_frame)
    selected: set[str] = set()
    for client in CLIENTS:
        reservoir = frame_by_domain[client]
        prefix = tuple(int(value) for value in indices[client][:draws_per_client])
        if len(prefix) != draws_per_client or any(
            value < 0 or value >= len(reservoir) for value in prefix
        ):
            raise PACSDataError("registered audit prefix is outside its reservoir")
        selected.update(reservoir[value].identity for value in prefix)
    if not selected:
        raise PACSDataError("registered audit prefix selected no identities")
    return _load_pacs_stage_records(
        image_manifest,
        fold_manifest,
        split=split,
        known_classes=known_classes,
        stage="audit",
        expected_image_count=expected_image_count,
        selected_identities=frozenset(selected),
    )


def load_pacs_full_audit_records(
    image_manifest: Path,
    fold_manifest: Path,
    *,
    split: int,
    known_classes: Sequence[str],
    primary_seal_path: Path,
    expected_primary_seal_sha256: str,
    expected_image_count: int | None = None,
) -> tuple[PACSRecord, ...]:
    """Open the complete audit truth only after the global primary barrier."""

    _verify_primary_seal_gate(primary_seal_path, expected_primary_seal_sha256)
    return _load_pacs_stage_records(
        image_manifest,
        fold_manifest,
        split=split,
        known_classes=known_classes,
        stage="audit",
        expected_image_count=expected_image_count,
    )


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
    "PACSAuditFrameRecord",
    "PACSDataError",
    "PACSRecord",
    "PACSZipDataset",
    "ROLES",
    "USES",
    "archive_member_from_manifest_path",
    "audit_frame_by_domain",
    "build_registered_transforms",
    "load_pacs_audit_frame",
    "load_pacs_full_audit_records",
    "load_pacs_proposal_records",
    "load_pacs_selected_audit_records",
    "load_pacs_train_records",
    "records_by_use_and_domain",
]
