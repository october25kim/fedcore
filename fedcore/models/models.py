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


class _WRNBasicBlock(nn.Module):
    """Pre-activation WideResNet basic block (BN-ReLU-Conv x2 + shortcut)."""

    def __init__(self, cin: int, cout: int, stride: int, norm: str = "bn", drop: float = 0.0):
        super().__init__()
        self.bn1 = _norm(norm, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = _norm(norm, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.drop = float(drop)
        self.equal = (cin == cout and stride == 1)
        self.short = None if self.equal else nn.Conv2d(cin, cout, 1, stride, 0, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = x if self.equal else self.short(out)
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        if self.drop > 0:
            out = F.dropout(out, p=self.drop, training=self.training)
        out = self.conv2(out)
        return out + shortcut


class WideResNet(nn.Module):
    """CIFAR pre-activation WideResNet. ``depth=28, widen_factor=10`` == WRN-28-10.

    Standard 32x32 CIFAR WRN with a single ``Linear(64*widen, n_known)`` head (a
    clean from-scratch module -- NOT the FedPD/PROSER bottleneck-head variant).
    ``norm`` selects BatchNorm (bn) or GroupNorm (gn, FL-appropriate).
    """

    def __init__(self, n_known: int, depth: int = 28, widen_factor: int = 10,
                 norm: str = "bn", drop: float = 0.0, in_channels: int = 3):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must be 6n+4")
        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(in_channels, widths[0], 3, 1, 1, bias=False)
        self.block1 = self._stage(widths[0], widths[1], n, 1, norm, drop)
        self.block2 = self._stage(widths[1], widths[2], n, 2, norm, drop)
        self.block3 = self._stage(widths[2], widths[3], n, 2, norm, drop)
        self.bn = _norm(norm, widths[3])
        self.classifier = nn.Linear(widths[3], n_known)
        self._feat_dim = widths[3]

    @staticmethod
    def _stage(cin, cout, n, stride, norm, drop):
        layers = [_WRNBasicBlock(cin, cout, stride, norm, drop)]
        for _ in range(n - 1):
            layers.append(_WRNBasicBlock(cout, cout, 1, norm, drop))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.block3(self.block2(self.block1(x)))
        x = F.relu(self.bn(x))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        return self.classifier(self.features(x))


class _ResNeXtBottleneck(nn.Module):
    """Aggregated-transform bottleneck (1x1 -> grouped 3x3 -> 1x1) + shortcut."""

    def __init__(self, in_planes: int, out_planes: int, stride: int,
                 cardinality: int, base_width: int, widen_factor: int, norm: str = "bn"):
        super().__init__()
        width_ratio = out_planes / (widen_factor * 64.0)
        d = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_planes, d, 1, 1, 0, bias=False)
        self.bn_reduce = _norm(norm, d)
        self.conv_conv = nn.Conv2d(d, d, 3, stride, 1, groups=cardinality, bias=False)
        self.bn = _norm(norm, d)
        self.conv_expand = nn.Conv2d(d, out_planes, 1, 1, 0, bias=False)
        self.bn_expand = _norm(norm, out_planes)
        self.shortcut = nn.Sequential()
        if in_planes != out_planes or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False),
                _norm(norm, out_planes),
            )

    def forward(self, x):
        b = F.relu(self.bn_reduce(self.conv_reduce(x)))
        b = F.relu(self.bn(self.conv_conv(b)))
        b = self.bn_expand(self.conv_expand(b))
        return F.relu(b + self.shortcut(x))


class ResNeXt29(nn.Module):
    """CIFAR ResNeXt-29. ``cardinality=8, base_width=64`` == ResNeXt-29 (8x64d).

    3 stages x 3 aggregated bottleneck blocks (depth 29) at 32x32, GAP -> a single
    ``Linear(256*widen, n_known)`` head. From-scratch plain-FedAvg backbone.
    ``norm`` selects BatchNorm (bn) or GroupNorm (gn).
    """

    def __init__(self, n_known: int, cardinality: int = 8, base_width: int = 64,
                 widen_factor: int = 4, depth: int = 29, norm: str = "bn", in_channels: int = 3):
        super().__init__()
        if (depth - 2) % 9 != 0:
            raise ValueError("ResNeXt depth must be 9n+2")
        n = (depth - 2) // 9
        self.cardinality = cardinality
        self.base_width = base_width
        self.widen_factor = widen_factor
        stages = [64, 64 * widen_factor, 128 * widen_factor, 256 * widen_factor]
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        self.bn1 = _norm(norm, 64)
        self.stage1 = self._make(stages[0], stages[1], n, 1, norm)
        self.stage2 = self._make(stages[1], stages[2], n, 2, norm)
        self.stage3 = self._make(stages[2], stages[3], n, 2, norm)
        self.classifier = nn.Linear(stages[3], n_known)
        self._feat_dim = stages[3]

    def _make(self, cin, cout, n, stride, norm):
        layers = []
        for i in range(n):
            layers.append(_ResNeXtBottleneck(
                cin if i == 0 else cout, cout, stride if i == 0 else 1,
                self.cardinality, self.base_width, self.widen_factor, norm))
        return nn.Sequential(*layers)

    def features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage3(self.stage2(self.stage1(x)))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        return self.classifier(self.features(x))


def efficientnet_b0(n_known: int, pretrained: bool = True):
    """FLamby's Fed-ISIC2019 baseline backbone (``flamby/datasets/fed_isic2019/model.py``).

    Replicates ``Baseline`` verbatim: ``efficientnet_pytorch.EfficientNet``
    ``efficientnet-b0``, ImageNet-pretrained, with ``_fc`` replaced by a
    ``Linear(1280, n_known)`` head. FLamby hard-codes 8 outputs; the open-set
    protocol trains only on the kept known classes, so the head width is
    ``n_known``.

    ``efficientnet_pytorch`` is imported lazily: it is an extra installed only in
    the Fed-ISIC container, and the CIFAR/FedPD arms must import this module
    without it.
    """
    from efficientnet_pytorch import EfficientNet

    base = (
        EfficientNet.from_pretrained("efficientnet-b0")
        if pretrained
        else EfficientNet.from_name("efficientnet-b0")
    )
    base._fc = nn.Linear(base._fc.in_features, n_known)
    return base


def convnext_tiny_model(
    n_known: int, pretrained: bool = True, frozen_encoder: bool = False
):
    """Office-Home arm backbone: torchvision ``convnext_tiny``.

    The ImageNet-1k head (``classifier[2] = Linear(768, 1000)``) is replaced with
    a freshly-initialized ``Linear(768, n_known)`` known-class head. Two pipelines
    share this factory:

    * pipeline A (``frozen_encoder=False``) -- full FedAvg fine-tune, every
      parameter trainable.
    * pipeline B (``frozen_encoder=True``) -- the encoder is frozen
      (``requires_grad=False``) and only the linear known-class head is federated.

    ``timm`` is deliberately not used: weights come from torchvision only.
    """
    import torchvision

    weights = "IMAGENET1K_V1" if pretrained else None
    model = torchvision.models.convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, n_known)
    if frozen_encoder:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.classifier[2].parameters():
            parameter.requires_grad_(True)
    return model


def dinov2_model(
    n_known: int,
    variant: str = "vit_large_patch14_dinov2.lvd142m",
    pretrained: bool = True,
    img_size: int = 224,
    frozen_encoder: bool = True,
):
    """Fed-ISIC DINOv2 arm: a ``timm`` DINOv2 ViT (self-supervised ImageNet features).

    Builds ``variant`` with a fresh ``Linear(embed_dim, n_known)`` known-class head
    (``num_classes=n_known``). ``frozen_encoder=True`` (default -- the linear-probe
    recipe) freezes every encoder parameter and federates ONLY the linear head; the
    frozen ViT runs in ``eval`` (no drop-path), so its features are deterministic.

    ``img_size=224`` gives a 16x16 patch grid (patch-14); ``timm`` interpolates the
    518-grid pretrained position embeddings at build time. Weights load from the
    baked Hugging Face cache in the ``fedcore-c400r-dino`` image (offline). ``timm``
    is imported lazily -- it lives only in that sibling image, so the CIFAR/FedPD
    arms import this module without it.
    """
    import timm

    model = timm.create_model(
        variant, pretrained=pretrained, num_classes=n_known, img_size=img_size
    )
    if frozen_encoder:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.get_classifier().parameters():
            parameter.requires_grad_(True)
    return model


def make_model(n_known: int, backbone: str = "simplecnn",
               pretrained: bool = False, norm: str = "bn", **kwargs):
    """Factory: ``backbone in {'simplecnn','resnet18','efficientnet-b0','convnext_tiny','dinov2_vitl14'}``.

    GroupNorm (``gn``) is the FL-appropriate normalization (BatchNorm running stats
    diverge under non-IID FedAvg). ``pretrained=True`` loads torchvision ResNet-18
    ImageNet weights (fc replaced with an ``n_known`` head).

    ``efficientnet-b0`` is the Fed-ISIC (FLamby) arm; it carries EfficientNet's own
    BatchNorm and therefore ignores ``norm``.

    ``convnext_tiny`` is the Office-Home arm; ``frozen_encoder`` (kwarg) selects the
    frozen-linear pipeline B, and it ignores ``norm``.
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
    if backbone == "wrn28_10":
        return WideResNet(n_known=n_known, depth=28, widen_factor=10, norm=norm)
    if backbone == "resnext29_8x64d":
        return ResNeXt29(n_known=n_known, cardinality=8, base_width=64,
                         widen_factor=4, depth=29, norm=norm)
    if backbone == "efficientnet-b0":
        return efficientnet_b0(n_known=n_known, pretrained=pretrained, **kwargs)
    if backbone == "convnext_tiny":
        return convnext_tiny_model(
            n_known=n_known,
            pretrained=pretrained,
            frozen_encoder=bool(kwargs.get("frozen_encoder", False)),
        )
    if backbone == "dinov2_vitl14":
        # Frozen DINOv2 ViT-L/14 linear-probe by default (run_fed_isic does not
        # thread frozen_encoder, so the default here IS the recipe).
        return dinov2_model(
            n_known=n_known,
            variant="vit_large_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            frozen_encoder=bool(kwargs.get("frozen_encoder", True)),
        )
    raise ValueError(f"unknown backbone {backbone!r}")
