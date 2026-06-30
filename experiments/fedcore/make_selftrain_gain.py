"""Backward-compat shim -> fedcore.plotting.make_selftrain_gain (structure-only refactor)."""
from fedcore.plotting.make_selftrain_gain import main  # noqa: F401
if __name__ == "__main__":
    main()
