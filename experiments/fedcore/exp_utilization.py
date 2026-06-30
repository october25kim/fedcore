"""Backward-compat shim -> fedcore.experiments.exp_utilization (structure-only refactor)."""
from fedcore.experiments.exp_utilization import main  # noqa: F401
if __name__ == "__main__":
    main()
