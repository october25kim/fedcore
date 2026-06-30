"""Backward-compat shim -> fedcore.selector (structure-only refactor)."""
from fedcore.selector import (  # noqa: F401
    open_set_error, Selector, empirical_risk_coverage, choose_threshold, counts_per_client,
)
