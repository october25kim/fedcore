"""Backward-compat shim -> fedcore.config (structure-only refactor)."""
from fedcore.config import (  # noqa: F401
    FedOSRConfig, CANONICAL_SCHEMA, SELFTRAIN_MIN_ACC,
)
