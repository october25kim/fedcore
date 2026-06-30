"""Backward-compat shim -> fedcore.scores (structure-only refactor)."""
from fedcore.scores import (  # noqa: F401
    softmax, msp, neg_entropy, margin, energy, score_norm, compute_score, scored_views,
)
