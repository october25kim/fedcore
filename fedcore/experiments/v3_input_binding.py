"""Fail-closed binding of the five registered PACS v3 scientific inputs.

This module is a host-side preparation utility.  It copies only the five
scientific inputs registered by the public v3 protocol into the isolated
execution root, verifies their bytes while copying, makes the staged files
read-only, and emits a deterministic receipt.  It performs no image decoding,
model construction, inference, or training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Sequence

from fedcore.experiments.v3_contract import (
    PROTOCOL_ID,
    ContractError,
    canonical_json_bytes,
)


ISOLATED_INPUT_DIRECTORY = Path(
    "/home/sanghoon/Desktop/Workspace/Fedcore_HSB_v3_exec/inputs"
)


@dataclass(frozen=True)
class ScientificInputSpec:
    """One immutable host input and its isolated runtime identity."""

    name: str
    source_path: str
    container_path: str
    sha256: str
    size_bytes: int


REGISTERED_SCIENTIFIC_INPUTS: tuple[ScientificInputSpec, ...] = (
    ScientificInputSpec(
        name="PACS_mirror.zip",
        source_path=(
            "/home/sanghoon/Desktop/Workspace/Fedcore/data/v1_pacs/"
            "PACS_mirror.zip"
        ),
        container_path="/inputs/PACS_mirror.zip",
        sha256="42bf567f1ed8a01d522e47e4a677e2a3149577bbd6fcbb38bedfdd73cb59e147",
        size_bytes=184_417_365,
    ),
    ScientificInputSpec(
        name="IMAGE_MANIFEST.csv",
        source_path=(
            "/home/sanghoon/Desktop/Workspace/Fedcore/outputs/"
            "fedcore_v1_ubuntu/IMAGE_MANIFEST.csv"
        ),
        container_path="/inputs/IMAGE_MANIFEST.csv",
        sha256="283d47bd8cfb7017d6e7fde6d65e07ff6479ab25127afcba135da3a06775afae",
        size_bytes=1_537_910,
    ),
    ScientificInputSpec(
        name="FOLD_MANIFEST.csv",
        source_path=(
            "/home/sanghoon/Desktop/Workspace/Fedcore/outputs/"
            "fedcore_v1_ubuntu/FOLD_MANIFEST.csv"
        ),
        container_path="/inputs/FOLD_MANIFEST.csv",
        sha256="fe5b64d237f47f503b8895aa5e566727bc818e05ee597f84c440df70e91e5344",
        size_bytes=7_853_382,
    ),
    ScientificInputSpec(
        name="resnet18-f37072fd.pth",
        source_path=(
            "/home/sanghoon/Desktop/Workspace/Fedcore/data/v1_pacs/torch_cache/"
            "hub/checkpoints/resnet18-f37072fd.pth"
        ),
        container_path="/inputs/resnet18-f37072fd.pth",
        sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
        size_bytes=46_830_571,
    ),
    ScientificInputSpec(
        name="convnext_tiny-983f1562.pth",
        source_path=(
            "/home/sanghoon/Desktop/Workspace/Fedcore/data/v1_pacs/torch_cache/"
            "hub/checkpoints/convnext_tiny-983f1562.pth"
        ),
        container_path="/inputs/convnext_tiny-983f1562.pth",
        sha256="983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
        size_bytes=114_419_221,
    ),
)


def _validate_specs(specs: Sequence[ScientificInputSpec]) -> tuple[ScientificInputSpec, ...]:
    frozen = tuple(specs)
    if len(frozen) != 5:
        raise ContractError(f"exactly five scientific inputs are required, observed {len(frozen)}")
    names = [item.name for item in frozen]
    destinations = [item.container_path for item in frozen]
    if len(names) != len(set(names)):
        raise ContractError("duplicate scientific input name")
    if len(destinations) != len(set(destinations)):
        raise ContractError("duplicate scientific input container path")
    for item in frozen:
        if Path(item.name).name != item.name or item.name in ("", ".", ".."):
            raise ContractError(f"scientific input name is not a basename: {item.name!r}")
        if Path(item.container_path) != Path("/inputs") / item.name:
            raise ContractError(f"container path drift for {item.name}")
        if len(item.sha256) != 64 or any(ch not in "0123456789abcdef" for ch in item.sha256):
            raise ContractError(f"invalid SHA-256 for {item.name}")
        if item.size_bytes < 0:
            raise ContractError(f"invalid byte size for {item.name}")
    return frozen


def _require_plain_directory(path: Path) -> None:
    if path.is_symlink():
        raise ContractError(f"isolated input directory must not be a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise ContractError(f"isolated input path is not a directory: {path}")


def _require_regular_nonsymlink(path: Path, label: str) -> os.stat_result:
    try:
        link_status = path.lstat()
    except FileNotFoundError as exc:
        raise ContractError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(link_status.st_mode):
        raise ContractError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(link_status.st_mode):
        raise ContractError(f"{label} must be a regular file: {path}")
    return link_status


def _hash_regular_file(path: Path, label: str) -> tuple[str, int]:
    """Hash a regular file through a no-follow descriptor when supported."""

    _require_regular_nonsymlink(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractError(f"cannot safely open {label}: {path}") from exc
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ContractError(f"{label} changed type while opening: {path}")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total += len(chunk)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest(), total


def _verify_file(path: Path, spec: ScientificInputSpec, label: str) -> None:
    digest, size = _hash_regular_file(path, label)
    if size != spec.size_bytes:
        raise ContractError(
            f"{label} size mismatch for {spec.name}: {size} != {spec.size_bytes}"
        )
    if digest != spec.sha256:
        raise ContractError(
            f"{label} SHA-256 mismatch for {spec.name}: {digest} != {spec.sha256}"
        )


def _copy_verified(source: Path, destination: Path, spec: ScientificInputSpec) -> None:
    """Copy one verified source without overwriting an existing staged file."""

    if destination.exists() or destination.is_symlink():
        _verify_file(destination, spec, "staged input")
        os.chmod(destination, 0o444, follow_symlinks=False)
        return

    _require_regular_nonsymlink(source, "source input")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_fd = os.open(source, flags)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{spec.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(source_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ContractError(f"source input changed type while opening: {source}")
        with os.fdopen(source_fd, "rb", closefd=True) as reader:
            source_fd = -1
            with os.fdopen(temporary_fd, "wb", closefd=True) as writer:
                temporary_fd = -1
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    digest.update(chunk)
                    total += len(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
        if total != spec.size_bytes or digest.hexdigest() != spec.sha256:
            raise ContractError(f"source input changed or mismatched while copying: {spec.name}")
        os.chmod(temporary, 0o444, follow_symlinks=False)
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    _verify_file(destination, spec, "staged input")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def bind_scientific_inputs(
    isolated_input_directory: Path = ISOLATED_INPUT_DIRECTORY,
    *,
    specs: Sequence[ScientificInputSpec] = REGISTERED_SCIENTIFIC_INPUTS,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Stage and bind the registered inputs without opening their scientific content.

    The destination may be empty, partially populated with already-correct files,
    or fully populated with the same five files.  Any extra entry, symlink, wrong
    byte, or non-regular file fails closed.  Existing wrong files are never
    replaced.
    """

    frozen = _validate_specs(specs)
    input_dir = Path(isolated_input_directory)
    _require_plain_directory(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    _require_plain_directory(input_dir)
    expected_names = {item.name for item in frozen}
    initial_names = {item.name for item in input_dir.iterdir()}
    extras = sorted(initial_names - expected_names)
    if extras:
        raise ContractError(f"unexpected isolated input entries: {extras!r}")

    records: list[dict[str, Any]] = []
    for spec in frozen:
        source = Path(spec.source_path)
        destination = input_dir / spec.name
        _verify_file(source, spec, "source input")
        if destination.is_symlink():
            raise ContractError(f"staged input must not be a symlink: {destination}")
        if source.resolve() == destination.resolve(strict=False):
            raise ContractError(f"source and staged input are the same path: {source}")
        _copy_verified(source, destination, spec)
        _verify_file(source, spec, "source input after copy")
        _verify_file(destination, spec, "staged input after copy")
        mode = stat.S_IMODE(destination.lstat().st_mode)
        if mode & 0o222:
            raise ContractError(f"staged input remains writable: {destination}")
        records.append(
            {
                **asdict(spec),
                "source_resolved_path": str(source.resolve(strict=True)),
                "staged_path": str(destination.resolve(strict=True)),
                "staged_mode_octal": f"{mode:04o}",
                "regular_file": True,
                "symlink": False,
                "read_only_mount_required": True,
            }
        )

    final_entries = tuple(sorted(item.name for item in input_dir.iterdir()))
    if final_entries != tuple(sorted(expected_names)):
        raise ContractError(f"isolated input directory is not exact: {final_entries!r}")
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS_FIVE_INPUT_BINDING",
        "isolated_input_directory": str(input_dir.resolve(strict=True)),
        "isolated_input_file_count": len(records),
        "exact_directory_contents": True,
        "all_regular_nonsymlink": True,
        "all_read_only": True,
        "bytes_read_for_hashing_and_copy_only": True,
        "scientific_content_interpreted": False,
        "pacs_training_started": False,
        "pacs_inference_started": False,
        "files": records,
    }
    if receipt_path is not None:
        receipt_target = Path(receipt_path)
        resolved_receipt = receipt_target.resolve(strict=False)
        resolved_input_dir = input_dir.resolve(strict=True)
        if (
            resolved_receipt == resolved_input_dir
            or resolved_input_dir in resolved_receipt.parents
        ):
            raise ContractError("input binding receipt must be stored outside the exact input directory")
        _atomic_write(receipt_target, canonical_json_bytes(receipt))
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isolated-input-directory",
        type=Path,
        default=ISOLATED_INPUT_DIRECTORY,
    )
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = bind_scientific_inputs(
        args.isolated_input_directory,
        receipt_path=args.receipt,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ISOLATED_INPUT_DIRECTORY",
    "REGISTERED_SCIENTIFIC_INPUTS",
    "ScientificInputSpec",
    "bind_scientific_inputs",
    "main",
]
