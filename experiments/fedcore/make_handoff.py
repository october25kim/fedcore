"""Backward-compat shim -> fedcore.experiments.make_handoff (structure-only refactor)."""
from fedcore.experiments.make_handoff import main  # noqa: F401
if __name__ == "__main__":
    main()
