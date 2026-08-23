"""Office-Home client update rule (AdamW) and the cosine LR schedule.

The FedAvg client update is AdamW over the *trainable* parameters (all of them
for pipeline A, only the linear head for pipeline B) with cross-entropy. A fresh
optimizer is built each round from the current global weights, so no optimizer
state crosses rounds -- the update is a deterministic function of the round's
weights and its minibatch order, which is exactly what makes ``--resume`` exact.

The learning rate is supplied per round by :func:`cosine_warmup_lr` through
FedAvg's ``lr_schedule`` hook. Keeping the schedule a pure function of the round
index (not of optimizer state) is what lets an interrupted run resume bit-exactly.
"""

from __future__ import annotations

import math
from typing import Callable


def cosine_warmup_lr(
    round_index: int, base_lr: float, total_rounds: int, warmup_rounds: int
) -> float:
    """Cosine-decayed LR with a linear warmup, as a pure function of the round.

    Rounds ``0 .. warmup_rounds-1`` ramp linearly ``base_lr/warmup .. base_lr``;
    the remaining rounds follow a half-cosine decay to 0. Depending only on the
    round index (and the fixed base/total/warmup constants), it recomputes
    identically after a resume.
    """
    if total_rounds <= 0:
        raise ValueError("total_rounds must be positive")
    if warmup_rounds < 0:
        raise ValueError("warmup_rounds must be non-negative")
    r = int(round_index)
    if r < 0 or r >= total_rounds:
        raise ValueError(f"round_index {r} out of range [0, {total_rounds})")
    warmup = min(int(warmup_rounds), int(total_rounds))
    if r < warmup:
        return float(base_lr) * float(r + 1) / float(warmup)
    denom = max(1, total_rounds - warmup)
    progress = (r - warmup) / denom
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_officehome_local_train_fn(
    *, weight_decay: float = 0.05, frozen_encoder: bool = False, criterion=None
) -> Callable[..., "object"]:
    """Return a FedAvg-compatible ``(model, loader, epochs, lr, device)`` update.

    ``frozen_encoder=True`` (pipeline B) runs the whole model in ``eval`` mode so
    the frozen ConvNeXt encoder is deterministic (stochastic depth off) and
    optimizes only the parameters left with ``requires_grad=True`` (the linear
    head). ``frozen_encoder=False`` (pipeline A) runs in ``train`` mode and
    optimizes every parameter.

    ``criterion`` defaults to ``None`` -> plain :class:`~torch.nn.CrossEntropyLoss`
    (the frozen Office-Home behaviour, byte-identical). Passing an explicit loss
    module (e.g. FLamby's imbalance-aware weighted focal loss) swaps ONLY the loss
    while keeping the AdamW client update and cosine-warmup schedule unchanged --
    used by the corrected Fed-ISIC ConvNeXt intervention.
    """

    import torch
    import torch.nn as nn

    def local_train_fn(model, loader, epochs, lr, device):
        model.to(device)
        if frozen_encoder:
            model.eval()  # deterministic frozen features (no stochastic depth)
        else:
            model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("no trainable parameters for the client update")
        opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
        if criterion is None:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = criterion.to(device) if hasattr(criterion, "to") else criterion
        for _ in range(int(epochs)):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
        return model

    return local_train_fn


__all__ = ["cosine_warmup_lr", "make_officehome_local_train_fn"]
