"""Backward-compat shim -> fedcore.experiments.run_smoke (structure-only refactor)."""
from fedcore.experiments.run_smoke import (  # noqa: F401
    SmokeSpec, generate_smoke, print_metric_table, save_csv, best_gamma_sanity, main,
)
if __name__ == "__main__":
    main()
