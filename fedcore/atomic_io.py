"""Backward-compatible import shim for CSV I/O helpers.

New code should import from :mod:`fedcore.io_utils`. This module remains so older
experiment scripts and saved commands keep working.
"""

from fedcore.io_utils import append_csv_locked, atomic_write_csv, atomic_write_text

__all__ = ["atomic_write_csv", "append_csv_locked", "atomic_write_text"]
