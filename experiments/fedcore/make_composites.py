"""Backward-compat shim -> fedcore.plotting.make_composites (structure-only refactor)."""
from fedcore.plotting.make_composites import main  # noqa: F401
if __name__ == "__main__":
    main()
