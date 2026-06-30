"""Backward-compat shim -> fedcore.plotting.make_problem_diagram (structure-only refactor)."""
from fedcore.plotting.make_problem_diagram import main  # noqa: F401
if __name__ == "__main__":
    main()
