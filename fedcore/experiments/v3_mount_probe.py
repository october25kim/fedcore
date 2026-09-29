"""Harmless no-GPU write probe for the FedCORE v3 container boundary.

The probe never decodes PACS, reads labels, imports torch, or starts training.
It checks only filesystem metadata and whether writes are rejected or allowed at
the six registered bind destinations, the read-only image root, and ``/tmp``.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat


INPUT_PATHS = (
    Path("/inputs/PACS_mirror.zip"),
    Path("/inputs/IMAGE_MANIFEST.csv"),
    Path("/inputs/FOLD_MANIFEST.csv"),
    Path("/inputs/resnet18-f37072fd.pth"),
    Path("/inputs/convnext_tiny-983f1562.pth"),
)
ROOT_PROBE = Path("/workspace/.fedcore-v3-write-probe")
TMP_PROBE = Path("/tmp/.fedcore-v3-write-probe")
OUTPUT_PROBE = Path("/output/.fedcore-v3-write-probe")


def _must_reject_write(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    except OSError:
        return
    os.close(descriptor)
    raise RuntimeError(f"write unexpectedly succeeded: {path}")


def _must_allow_create_then_remove(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, b"mount-probe\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.unlink()


def main() -> int:
    for path in INPUT_PATHS:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"registered input is not a regular file: {path}")
        _must_reject_write(path)
    _must_reject_write(ROOT_PROBE)
    _must_allow_create_then_remove(TMP_PROBE)
    _must_allow_create_then_remove(OUTPUT_PROBE)
    print("PASS_INPUTS_READ_ONLY")
    print("PASS_ROOT_READ_ONLY")
    print("PASS_TMP_RW")
    print("PASS_OUTPUT_RW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["INPUT_PATHS", "OUTPUT_PROBE", "ROOT_PROBE", "TMP_PROBE", "main"]
