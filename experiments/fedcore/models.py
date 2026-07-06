"""Small CNN classifier over the known classes (torch). CIFAR-sized inputs.

Kept intentionally light so the FedAvg smoke train (project's
docker_cifar_smoke_train.sh analogue) is fast; swap for a ResNet for full runs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A compact 3-block CNN for 32x32x3 -> n_known logits."""

    def __init__(self, n_known: int, in_ch: int = 3, width: int = 64):
        super().__init__()
        def block(ci: int, co: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(in_ch, width),
            block(width, width * 2),
            block(width * 2, width * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(width * 4, n_known),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def make_model(n_known: int) -> nn.Module:
    return SimpleCNN(n_known=n_known)
