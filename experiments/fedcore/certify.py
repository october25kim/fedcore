"""Backward-compat shim -> fedcore.certify (structure-only refactor)."""
from fedcore.certify import (  # noqa: F401
    _coverage_lcb, certify_for_score, certify_best_gamma, certify_best_gamma_grouped,
    certify_grid, METRIC_KEYS,
)
