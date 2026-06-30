"""Backward-compat shim -> fedcore.experiments.exp_hetero_collapse (structure-only refactor)."""
from fedcore.experiments.exp_hetero_collapse import main  # noqa: F401
if __name__ == "__main__":
    main()
