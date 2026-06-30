"""Backward-compat shim -> fedcore.experiments.exp_necessity (structure-only refactor)."""
from fedcore.experiments.exp_necessity import main  # noqa: F401
if __name__ == "__main__":
    main()
