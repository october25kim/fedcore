"""Backward-compat shim -> fedcore.experiments.exp_score_agnostic (structure-only refactor)."""
from fedcore.experiments.exp_score_agnostic import main  # noqa: F401
if __name__ == "__main__":
    main()
