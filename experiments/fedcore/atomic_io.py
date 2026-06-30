"""Backward-compat shim -> fedcore.atomic_io (structure-only refactor)."""
from fedcore.atomic_io import atomic_write_csv, append_csv_locked  # noqa: F401
