"""Minimal but correct FedAvg training + logit export (torch).

The trained model classifies the KNOWN classes only; unknown-class points are
never seen in training and appear solely in the trusted calibration/test folds.
"""
from __future__ import annotations

import copy

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
    """One client's local SGD on cross-entropy over known classes."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def _weighted_average(states: list[dict], weights: np.ndarray) -> dict:
    w = weights / weights.sum()
    avg = copy.deepcopy(states[0])
    for k in avg:
        if avg[k].dtype.is_floating_point:
            avg[k] = sum(w[i] * states[i][k] for i in range(len(states)))
        else:  # e.g. BN num_batches_tracked: take from first
            avg[k] = states[0][k]
    return avg


def fedavg(
    make_model_fn,
    client_datasets: list[Dataset],
    rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> nn.Module:
    """Standard FedAvg: weighted average of locally-updated client states."""
    global_model = make_model_fn().to(device)
    sizes = np.array([len(ds) for ds in client_datasets], dtype=float)
    loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
        for ds in client_datasets
    ]
    for _ in range(rounds):
        states = []
        for loader in loaders:
            local = copy.deepcopy(global_model)
            local_train(local, loader, local_epochs, lr, device)
            states.append({k: v.detach().clone() for k, v in local.state_dict().items()})
        global_model.load_state_dict(_weighted_average(states, sizes))
    return global_model


@torch.no_grad()
def export_logits(
    model: nn.Module,
    base_dataset: Dataset,
    indices: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Return logits (len(indices), n_known) for the given dataset indices."""
    model.eval()
    loader = DataLoader(Subset(base_dataset, list(indices)),
                        batch_size=batch_size, shuffle=False)
    out = []
    for xb, _ in loader:
        out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0,))
