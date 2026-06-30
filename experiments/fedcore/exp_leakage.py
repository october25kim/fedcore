"""Backward-compat shim -> fedcore.experiments.exp_leakage (structure-only refactor)."""
from fedcore.experiments.exp_leakage import main  # noqa: F401
if __name__ == "__main__":
    main()
