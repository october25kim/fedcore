"""Backbones over the known classes (torch).

SimpleCNN (default) and a CIFAR-variant ResNet-18 (3x3 stem, no maxpool). The
backbone is the lever for lowering realized risk ``rhat`` -- the Theorem-2 sample
requirement scales as ``(alpha - rhat)^-2``, so a stronger backbone shrinks the
per-group accepted count needed to certify. Imported only by the torch path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """3 conv blocks (Conv-BN-ReLU-MaxPool) -> global average pool -> Linear."""

    def __init__(self, n_known: int, in_channels: int = 3, width: int = 64):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, width),
            block(width, width * 2),
            block(width * 2, width * 4),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(width * 4, n_known)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


def _norm(norm: str, c: int) -> nn.Module:
    """Normalization layer: BatchNorm (bn) or GroupNorm-32 (gn, FL-appropriate)."""
    if norm == "gn":
        return nn.GroupNorm(min(32, c), c)
    return nn.BatchNorm2d(c)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin: int, cout: int, stride: int = 1, norm: str = "bn"):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = _norm(norm, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = _norm(norm, cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), _norm(norm, cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.short(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """CIFAR-variant ResNet-18: 3x3 stem, no maxpool, [2,2,2,2] basic blocks."""

    def __init__(self, n_known: int, in_channels: int = 3, norm: str = "bn"):
        super().__init__()
        self.in_planes = 64
        self.norm = norm
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),
            _norm(norm, 64), nn.ReLU(inplace=True))
        self.layer1 = self._make(64, 2, 1)
        self.layer2 = self._make(128, 2, 2)
        self.layer3 = self._make(256, 2, 2)
        self.layer4 = self._make(512, 2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, n_known)

    def _make(self, planes, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_BasicBlock(self.in_planes, planes, s, norm=self.norm))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.gap(x).flatten(1)

    def forward(self, x):
        return self.classifier(self.features(x))


def make_model(n_known: int, backbone: str = "simplecnn",
               pretrained: bool = False, norm: str = "bn", **kwargs):
    """Factory: ``backbone in {'simplecnn','resnet18'}``, ``norm in {'bn','gn'}``.

    GroupNorm (``gn``) is the FL-appropriate normalization (BatchNorm running stats
    diverge under non-IID FedAvg). ``pretrained=True`` loads torchvision ResNet-18
    ImageNet weights (fc replaced with an ``n_known`` head).
    """
    if backbone == "simplecnn":
        return SimpleCNN(n_known=n_known, **kwargs)
    if backbone == "resnet18":
        if pretrained:
            import torchvision
            m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            m.fc = nn.Linear(m.fc.in_features, n_known)
            return m
        return ResNet18(n_known=n_known, norm=norm, **kwargs)
    raise ValueError(f"unknown backbone {backbone!r}")
