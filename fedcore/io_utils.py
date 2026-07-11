"""CSV writers used by experiment runners and aggregators.

The helpers centralize two file-write patterns used across the project:

- full CSV rewrites through a same-directory temp file + os.replace;
- append-only CSV writes guarded by flock when POSIX locking is available.
"""

from __future__ import annotations

import csv
import os
import tempfile
from typing import Iterable, Mapping, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


def atomic_write_csv(
    path: str,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping],
    extrasaction: str = "raise",
) -> None:
    """Write rows to path atomically."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(fieldnames), extrasaction=extrasaction
            )
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def append_csv_locked(
    path: str,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping],
    extrasaction: str = "ignore",
) -> None:
    """Append rows under an exclusive lock; write the header for a new file."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    with open(path, "a", newline="") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(
                f, fieldnames=list(fieldnames), extrasaction=extrasaction
            )
            if os.fstat(f.fileno()).st_size == 0:
                writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)
