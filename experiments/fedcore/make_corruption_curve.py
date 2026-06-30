"""Backward-compat shim -> fedcore.plotting.make_corruption_curve (structure-only refactor)."""
from fedcore.plotting.make_corruption_curve import main  # noqa: F401
if __name__ == "__main__":
    main()
