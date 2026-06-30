"""Backward-compat shim -> fedcore.plotting.make_figures (structure-only refactor)."""
from fedcore.plotting.make_figures import (  # noqa: F401
    ALPHA, BASE, CB, DELTA, FIGS, _staircase_by_G, main,
)
if __name__ == "__main__":
    main()
