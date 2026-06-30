"""Backward-compat shim -> fedcore.models.fed_train (structure-only refactor)."""
from fedcore.models.fed_train import (  # noqa: F401
    local_train, _average_state_dicts, fedavg, export_logits,
)
