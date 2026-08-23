"""Reservoir accounting: identity, draw replay, and audit-burden reporting.

This package is READ-ONLY with respect to ``runs/``. It answers one question the
certification pipeline never asks: *how many distinct labelled examples actually
support each certified number, versus how many nominal draws were taken?*

Nothing here may change a certificate. See ``docs/agent_plan_phase1.md``.
"""

from fedcore.accounting.ids import IdRecoveryError, recover_fold_ids
from fedcore.accounting.occupancy import expected_unique_count
from fedcore.accounting.provenance import RunSpec, resolve_run

__all__ = [
    "IdRecoveryError",
    "RunSpec",
    "expected_unique_count",
    "recover_fold_ids",
    "resolve_run",
]
