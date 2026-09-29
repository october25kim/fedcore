"""Fail-closed PACS training entry point for the v3 execution-binding phase.

No training implementation is reachable in the current binding.  The command
exists so that an accidental scientific dispatch fails before importing torch,
opening PACS, or touching a GPU.  A later, separately reviewed execution commit
must replace the final hold after the dedicated runner and execution seal exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fedcore.experiments.v3_contract import ContractError
from fedcore.experiments.v3_validate import require_scientific_execution_authorization


def scientific_preflight(execution_gate: Path) -> None:
    require_scientific_execution_authorization(execution_gate)
    raise ContractError(
        "scientific training remains disabled in this synthetic-binding commit; "
        "a separately reviewed runner-binding commit is required"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-gate", type=Path, required=True)
    args = parser.parse_args()
    scientific_preflight(args.execution_gate)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["scientific_preflight"]
