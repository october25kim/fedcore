"""Zero-error feasibility floor -- MANUSCRIPT NUMBERING: **Theorem 3**.

Extracts the per-group accepted-count floor formula used inline across the codebase
(``ln(J/delta) / (-ln(1-alpha))``) into one named pure function. Behaviour-identical to
the inline expression; no caller is changed by this extraction (internal call sites may be
migrated in a later import-migration step).

CODE/MANUSCRIPT NUMBERING COLLISION -- READ BEFORE CITING THIS FUNCTION
----------------------------------------------------------------------
The symbol ``thm2_floor`` is STALE and is deliberately NOT renamed: it is load-bearing
for the golden/regression gate, and renaming it would move legacy results for a purely
cosmetic gain.

The quantity it computes is **Theorem 3** in the manuscript. Two independent, sealed
authorities agree on that numbering:

* ``results/preregistration.yaml`` -> ``statistics.theorem_3``:
  "zero-error floor A_j >= ceil(ln(J/delta_r)/(-ln(1-alpha)))";
* ``fedcore/certificate/allocation.py``'s docstring, which already uses Theorem 3.

Only this module's legacy symbol and ``CLAUDE.md`` section 2 still say "Theorem 2"
(``CLAUDE.md`` additionally reserves "Theorem 2" for the feasibility statement, which is
the origin of the clash). MANUSCRIPT-FACING OUTPUT MUST SAY "Theorem 3" -- e.g.
``results/fedisic/count_feasibility_by_budget.csv`` carries the column
``A_min_theorem_manuscript = "Theorem 3"``. Do not propagate the ``thm2`` spelling into
any table, figure label, or paper text.
"""

from __future__ import annotations

import numpy as np


def thm2_floor(J: int, delta: float, alpha: float) -> float:
    """Per-group accepted-count floor: ``ln(J/delta) / (-ln(1-alpha))``.

    MANUSCRIPT NUMBERING: this is **Theorem 3**, not Theorem 2 -- the function name is a
    stale spelling retained only for golden-regression stability. See the module
    docstring. Manuscript-facing output must say "Theorem 3".

    A group whose accepted count falls below this floor cannot certify selective risk
    ``<= alpha`` at confidence ``1 - delta`` (infeasible round / non-deployable).
    """
    return float(np.log(J / delta) / (-np.log(1 - alpha)))
