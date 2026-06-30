"""Backward-compat shim -> fedcore.experiments.run_cifar (structure-only refactor)."""
from fedcore.experiments.run_cifar import (  # noqa: F401
    _LabelRemapSubset, _NORM, _gather_fold, _load_cifar, main,
)
if __name__ == "__main__":
    main()
