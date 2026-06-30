"""Backward-compat shim -> fedcore.aggregate.covtype (structure-only refactor)."""
from fedcore.aggregate.covtype import main  # noqa: F401
if __name__ == "__main__":
    main()
