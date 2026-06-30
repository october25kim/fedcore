"""Backward-compat shim -> fedcore.experiments.exp_pooling_fail (structure-only refactor)."""
from fedcore.experiments.exp_pooling_fail import main  # noqa: F401
if __name__ == "__main__":
    main()
