"""Backward-compat shim -> fedcore.experiments.exp_superiority (structure-only refactor)."""
from fedcore.experiments.exp_superiority import main  # noqa: F401
if __name__ == "__main__":
    main()
