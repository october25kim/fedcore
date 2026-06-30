"""Theorem 2 feasibility floor.

Extracts the per-group accepted-count floor formula used inline across the codebase
(``ln(J/delta) / (-ln(1-alpha))``) into one named pure function. Behaviour-identical to
the inline expression; no caller is changed by this extraction (internal call sites may be
migrated in a later import-migration step).
"""

from __future__ import annotations

import numpy as np


def thm2_floor(J: int, delta: float, alpha: float) -> float:
    """Theorem-2 per-group accepted-count floor: ``ln(J/delta) / (-ln(1-alpha))``.

    A group whose accepted count falls below this floor cannot certify selective risk
    ``<= alpha`` at confidence ``1 - delta`` (infeasible round / non-deployable).
    """
    return float(np.log(J / delta) / (-np.log(1 - alpha)))
