"""Backward-compat shim -> fedcore.experiments.self_training (structure-only refactor)."""
from fedcore.experiments.self_training import (  # noqa: F401
    RoundDecision, round_decision, simulate_self_training,
)
