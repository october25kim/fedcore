"""Backward-compat shim -> fedcore.aggregate.t8 (structure-only refactor)."""
from fedcore.aggregate.t8 import main  # noqa: F401
if __name__ == "__main__":
    main()
