"""Backward-compat shim -> fedcore.experiments.run_selftrain_pkg (structure-only refactor)."""
from fedcore.experiments.run_selftrain_pkg import setup_data, main  # noqa: F401
if __name__ == "__main__":
    main()
