"""Backward-compat shim -> fedcore.plotting.make_F8 (structure-only refactor)."""
from fedcore.plotting.make_F8 import main  # noqa: F401
if __name__ == "__main__":
    main()
