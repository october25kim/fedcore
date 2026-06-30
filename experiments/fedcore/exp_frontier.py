"""Backward-compat shim -> fedcore.experiments.exp_frontier (structure-only refactor)."""
from fedcore.experiments.exp_frontier import main  # noqa: F401
if __name__ == "__main__":
    main()
