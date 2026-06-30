"""Backward-compat shim -> fedcore.experiments.selftrain (structure-only refactor)."""
from fedcore.experiments.selftrain import (  # noqa: F401
    MappedSubset, _gather, best_gamma_selector, naive_selector, partition_selftrain, run_self_training,
)
