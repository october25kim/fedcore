"""Backward-compat shim -> fedcore.experiments.exp_necessity_real (structure-only refactor)."""
from fedcore.experiments.exp_necessity_real import main  # noqa: F401
if __name__ == "__main__":
    main()
