"""Backward-compat shim -> fedcore.experiments.tables (structure-only refactor)."""
from fedcore.experiments.tables import main  # noqa: F401
if __name__ == "__main__":
    main()
