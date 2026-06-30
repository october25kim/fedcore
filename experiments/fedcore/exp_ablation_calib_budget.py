"""Backward-compat shim -> fedcore.experiments.exp_ablation_calib_budget (structure-only refactor)."""
from fedcore.experiments.exp_ablation_calib_budget import main  # noqa: F401
if __name__ == "__main__":
    main()
