"""Backward-compat shim -> fedcore.experiments.exp_feasibility_lever (structure-only refactor)."""
from fedcore.experiments.exp_feasibility_lever import main  # noqa: F401
if __name__ == "__main__":
    main()
