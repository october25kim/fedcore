"""Backward-compat shim -> fedcore.experiments.exp_validity (structure-only refactor)."""
from fedcore.experiments.exp_validity import main  # noqa: F401
if __name__ == "__main__":
    main()
