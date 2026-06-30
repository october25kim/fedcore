"""Backward-compat shim -> fedcore.data.fedosr_split (structure-only refactor)."""
from fedcore.data.fedosr_split import (  # noqa: F401
    open_set_split, dirichlet_partition, build_calibration,
)
