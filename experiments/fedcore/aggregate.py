"""Backward-compat shim -> fedcore.aggregate.main (structure-only refactor)."""
from fedcore.aggregate.main import main  # noqa: F401
if __name__ == "__main__":
    main()
