"""Atomic / race-free CSV writers (becomes fedcore/io.py inside the package).

Named ``atomic_io`` (NOT ``io``) in the flat experiments/ dir to avoid shadowing the Python
stdlib ``io`` module for scripts run from here.

- :func:`atomic_write_csv` -- full rewrite via a unique temp file + ``os.replace`` (atomic on
  POSIX), so a concurrent reader never sees a half-written file. Use for aggregators.
- :func:`append_csv_locked` -- append guarded by an exclusive ``flock`` so concurrent
  per-process appends cannot interleave (this removes the clean+launch race that previously
  injected duplicate/smoke rows). Use for the experiment runners.

Both produce byte-identical CSV content to the previous ``open(...,'w'/'a') + csv.DictWriter``
path (same fieldnames, rows, quoting); behaviour is unchanged, only the write is made safe.
"""

from __future__ import annotations

import csv
import os
import tempfile
from typing import Iterable, Mapping, Sequence

try:
    import fcntl  # POSIX file locking
except ImportError:  # pragma: no cover
    fcntl = None


def atomic_write_csv(path: str, fieldnames: Sequence[str], rows: Iterable[Mapping],
                     extrasaction: str = "raise") -> None:
    """Write ``rows`` to ``path`` atomically (temp file in the same dir + os.replace)."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction=extrasaction)
            w.writeheader()
            w.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)            # atomic
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def append_csv_locked(path: str, fieldnames: Sequence[str], rows: Iterable[Mapping],
                      extrasaction: str = "ignore") -> None:
    """Append ``rows`` under an exclusive lock; write the header iff the file is new/empty."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    with open(path, "a", newline="") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction=extrasaction)
            if os.fstat(f.fileno()).st_size == 0:
                w.writeheader()
            w.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)
