"""Backward-compat shim -> fedcore.experiments.run_tabular (structure-only refactor)."""
from fedcore.experiments.run_tabular import main  # noqa: F401
if __name__ == "__main__":
    main()
