"""Backward-compat shim -> fedcore.aggregate.selftrain (structure-only refactor)."""
from fedcore.aggregate.selftrain import main  # noqa: F401
if __name__ == "__main__":
    main()
