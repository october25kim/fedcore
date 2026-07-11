"""FedAvg training and logit export (torch).

``fedavg`` performs size-weighted averaging of client model parameters over a
number of communication rounds; non-float buffers (e.g. BatchNorm
``num_batches_tracked``) are copied from the first client rather than averaged.
``export_logits`` produces logits over the known classes for a set of dataset
indices, which then feed the (torch-free) certification core.
"""

from __future__ import annotations

import copy
from typing import Callable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


def local_train(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
) -> nn.Module:
    """Train ``model`` in place with SGD+momentum on cross-entropy."""
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def _average_state_dicts(state_dicts: List[dict], weights: Sequence[float]) -> dict:
    """Size-weighted average of float tensors; non-float buffers from client 0."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    avg = copy.deepcopy(state_dicts[0])
    for key, ref in avg.items():
        if torch.is_floating_point(ref):
            stacked = torch.stack(
                [sd[key].float() * w for sd, w in zip(state_dicts, weights)], dim=0
            )
            avg[key] = stacked.sum(dim=0).to(ref.dtype)
        else:
            avg[key] = state_dicts[0][key]  # e.g. num_batches_tracked
    return avg


def fedavg(
    make_model_fn: Callable[[], nn.Module],
    client_datasets: List[Dataset],
    rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> nn.Module:
    """Run ``rounds`` of FedAvg and return the final global model."""
    global_model = make_model_fn().to(device)
    sizes = [len(ds) for ds in client_datasets]

    for r in range(rounds):
        local_states: List[dict] = []
        used_sizes: List[int] = []
        for ds, size in zip(client_datasets, sizes):
            if size == 0:
                continue
            local = make_model_fn().to(device)
            local.load_state_dict(copy.deepcopy(global_model.state_dict()))
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=True, drop_last=False
            )
            local_train(local, loader, local_epochs, lr, device)
            local_states.append(local.state_dict())
            used_sizes.append(size)
        if local_states:
            global_model.load_state_dict(
                _average_state_dicts(local_states, used_sizes)
            )
    return global_model


@torch.no_grad()
def export_logits(
    model: nn.Module,
    base_dataset: Dataset,
    indices: Sequence[int],
    device: str,
    bs: int = 256,
) -> np.ndarray:
    """Return logits of shape ``(len(indices), n_known)`` for ``indices``."""
    model.to(device).eval()
    subset = Subset(base_dataset, list(indices))
    loader = DataLoader(subset, batch_size=bs, shuffle=False)
    out: List[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0,))
