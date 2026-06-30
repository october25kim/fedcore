"""Backward-compat shim -> fedcore.experiments.run_foogd_cifar (structure-only refactor)."""
from fedcore.experiments.run_foogd_cifar import main  # noqa: F401
if __name__ == "__main__":
    main()
