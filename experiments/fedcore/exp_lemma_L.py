"""Backward-compat shim -> fedcore.experiments.exp_lemma_L (structure-only refactor)."""
from fedcore.experiments.exp_lemma_L import main  # noqa: F401
if __name__ == "__main__":
    main()
