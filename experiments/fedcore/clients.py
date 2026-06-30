"""Backward-compat shim -> fedcore.data.clients (structure-only refactor)."""
from fedcore.data.clients import (  # noqa: F401
    ClientPopulation, draw_counts, heterogeneous_population,
)
