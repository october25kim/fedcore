"""FLamby Fed-ISIC2019 recipe pieces: weighted focal loss and 200px transforms.

Transcribed from ``owkin/FLamby`` @ main, fetched and read (not recalled), per the
pre-registration's own instruction ("verify the exact recipe from the FLamby repo
... do not trust memory"):

* ``flamby/datasets/fed_isic2019/loss.py``    -- ``BaselineLoss`` (weighted focal)
* ``flamby/datasets/fed_isic2019/dataset.py`` -- the albumentations pipelines
* ``flamby/datasets/fed_isic2019/common.py``  -- Adam, lr 5e-4, batch 64

Torch is imported lazily so the metadata/dry-run paths keep working on torch-free
hosts, matching :mod:`fedcore.medical.data`.

Two DECLARED DEVIATIONS from FLamby (surfaced, never silently patched)
---------------------------------------------------------------------
1. **Train augmentations.** FLamby's train pipeline is eight ops::

       RandomScale(0.07), Rotate(50), RandomBrightnessContrast(0.15, 0.1),
       Flip(p=0.5), Affine(shear=0.1), RandomCrop(200, 200),
       CoarseDropout(random.randint(1, 8), 16, 16), Normalize(always_apply=True)

   The sealed pre-registration (``data.fed_isic2019.recipe.train_transform``)
   declares only ``RandomCrop(200,200) + Normalize``.  The pre-registration is the
   sole authority for the main run, so :func:`train_transform` implements the
   DECLARED two-op pipeline.  The six dropped ops are recorded in
   :data:`FLAMBY_TRAIN_OPS_NOT_PREREGISTERED` rather than silently reinstated.
   The eval pipeline needs no such note: the prereg's
   ``CenterCrop(200,200) + Normalize`` matches FLamby exactly.

2. **Input preprocessing is NOT pre-registered and NOT applied here.** FLamby does
   not read raw ISIC JPEGs.  Its ``dataset_creation_scripts/resize_images.py``
   writes ``ISIC_2019_Training_Input_preprocessed`` by resizing each image so the
   SHORTER SIDE IS 224px (aspect ratio preserved, bilinear) and applying
   Shades-of-Gray colour constancy (``color_constancy.py``, power=6); the dataset
   class then reads only that preprocessed directory.  ``RandomCrop(200,200)``
   is therefore a MILD crop of a ~224px-short-side image in FLamby.  This repo
   stages only the raw archive (``data/isic2019/ISIC_2019_Training_Input``, images
   up to 1024x1024), where the same crop takes a ~4%-area patch that usually misses
   the lesion.  Neither the resize target nor the colour constancy is declared in
   the sealed prereg, so this module does NOT invent them; see
   :data:`FLAMBY_PREPROCESSING_NOT_PREREGISTERED`.  Resolve before a real run.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


#: FLamby class order for ``focal_alpha_full8`` (``ISIC_2019_Training_GroundTruth``
#: column order).  Matches ``fedcore.data.fed_isic2019.CLASSES``.
FLAMBY_CLASS_ORDER: tuple[str, ...] = (
    "MEL",
    "NV",
    "BCC",
    "AK",
    "BKL",
    "DF",
    "VASC",
    "SCC",
)

#: ``flamby/datasets/fed_isic2019/loss.py`` ``BaselineLoss.__init__`` default alpha.
FLAMBY_FOCAL_ALPHA_FULL8: tuple[float, ...] = (
    5.5813,
    2.0472,
    7.0204,
    26.1194,
    9.5369,
    101.0707,
    92.5224,
    38.3443,
)

FLAMBY_FOCAL_GAMMA: float = 2.0

#: Ops FLamby applies in training that the sealed prereg does not declare.
FLAMBY_TRAIN_OPS_NOT_PREREGISTERED: tuple[str, ...] = (
    "RandomScale(0.07)",
    "Rotate(50)",
    "RandomBrightnessContrast(0.15, 0.1)",
    "Flip(p=0.5)",
    "Affine(shear=0.1)",
    "CoarseDropout(random.randint(1, 8), 16, 16)",
)

#: FLamby's mandatory pre-training image preprocessing, absent from the prereg.
FLAMBY_PREPROCESSING_NOT_PREREGISTERED: str = (
    "resize_images.py: shorter side -> 224px, aspect preserved, PIL BILINEAR; "
    "then color_constancy.py: Shades-of-Gray, power=6. FLamby reads only "
    "ISIC_2019_Training_Input_preprocessed. Not declared in the sealed prereg; "
    "not applied by this module."
)


def restrict_focal_alpha(
    known_diagnoses: Sequence[str],
    alpha_full8: Sequence[float] = FLAMBY_FOCAL_ALPHA_FULL8,
) -> np.ndarray:
    """Restrict FLamby's 8-class alpha to the kept known classes, BY INDEX.

    Implements ``recipe.focal_alpha_policy`` verbatim: alpha is indexed over all 8
    classes in FLamby's order; the unknown classes are never trained on, so alpha is
    restricted to the kept classes by index with **no renormalization** (the weights
    are per-class multipliers, not a distribution).

    ``known_diagnoses`` must be ordered exactly as the model's head, i.e. the label
    mapping used at training time (``FedISICJobData.known_diagnoses``).
    """
    alpha_full8 = tuple(float(value) for value in alpha_full8)
    if len(alpha_full8) != len(FLAMBY_CLASS_ORDER):
        raise ValueError(
            f"alpha_full8 must have {len(FLAMBY_CLASS_ORDER)} entries, "
            f"one per FLamby class; got {len(alpha_full8)}"
        )
    known = [str(name).strip() for name in known_diagnoses]
    if not known:
        raise ValueError("known_diagnoses must be non-empty")
    if len(set(known)) != len(known):
        raise ValueError(f"known_diagnoses contains duplicates: {known}")
    unexpected = [name for name in known if name not in FLAMBY_CLASS_ORDER]
    if unexpected:
        raise ValueError(
            f"known diagnoses {unexpected} are not FLamby classes {FLAMBY_CLASS_ORDER}"
        )
    index = {name: position for position, name in enumerate(FLAMBY_CLASS_ORDER)}
    return np.asarray([alpha_full8[index[name]] for name in known], dtype=np.float64)


def make_focal_loss(
    known_diagnoses: Sequence[str],
    *,
    gamma: float = FLAMBY_FOCAL_GAMMA,
    alpha_full8: Sequence[float] = FLAMBY_FOCAL_ALPHA_FULL8,
):
    """FLamby ``BaselineLoss`` (weighted focal), restricted to the known classes.

    Transcribed from ``flamby/datasets/fed_isic2019/loss.py``::

        targets = targets.view(-1, 1).type_as(inputs)
        logpt = F.log_softmax(inputs, dim=1).gather(1, targets.long()).view(-1)
        pt = logpt.exp()
        at = self.alpha.gather(0, targets.data.view(-1).long())
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean()
    """
    import torch
    from torch.nn import functional as F
    from torch.nn.modules.loss import _Loss

    restricted = restrict_focal_alpha(known_diagnoses, alpha_full8)
    if not float(gamma) >= 0.0:
        raise ValueError("focal gamma must be non-negative")

    class BaselineLoss(_Loss):
        """FLamby's weighted focal loss over the kept known classes."""

        def __init__(self) -> None:
            super().__init__()
            self.alpha = torch.tensor(restricted).to(torch.float)
            self.gamma = float(gamma)

        def forward(self, inputs, targets):
            targets = targets.view(-1, 1).type_as(inputs)
            logpt = F.log_softmax(inputs, dim=1)
            logpt = logpt.gather(1, targets.long())
            logpt = logpt.view(-1)
            pt = logpt.exp()
            self.alpha = self.alpha.to(targets.device)
            at = self.alpha.gather(0, targets.data.view(-1).long())
            logpt = logpt * at
            loss = -1 * (1 - pt) ** self.gamma * logpt
            return loss.mean()

    return BaselineLoss()


class AlbumentationsImageTransform:
    """Adapt an albumentations pipeline to the ``PIL.Image -> CHW tensor`` contract.

    :class:`fedcore.medical.data.MedicalImageDataset` hands a PIL image to its
    transform and expects a torch tensor.  FLamby's dataset instead feeds
    ``np.array(Image.open(path))`` to albumentations and finishes with
    ``np.transpose(image, (2, 0, 1))`` -- there is no ``ToTensor``, because
    ``Normalize`` already returns float32 HWC.  This adapter reproduces that
    exact ordering.
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def __call__(self, image):
        import torch

        array = np.asarray(image)
        augmented = self.pipeline(image=array)["image"]
        chw = np.transpose(augmented, (2, 0, 1)).astype(np.float32)
        return torch.tensor(chw, dtype=torch.float32)


def train_transform(image_size: int = 200) -> AlbumentationsImageTransform:
    """The PRE-REGISTERED train pipeline: ``RandomCrop(sz, sz) + Normalize``.

    NOT FLamby's full train pipeline -- see this module's docstring, deviation 1.
    """
    import albumentations

    return AlbumentationsImageTransform(
        albumentations.Compose(
            [
                albumentations.RandomCrop(image_size, image_size),
                # FLamby spells this Normalize(always_apply=True); always_apply is
                # deprecated in current albumentations and is a no-op here, since
                # Normalize already defaults to p=1.0. Behaviour is identical.
                albumentations.Normalize(),
            ]
        )
    )


def eval_transform(image_size: int = 200) -> AlbumentationsImageTransform:
    """``CenterCrop(sz, sz) + Normalize`` -- prereg and FLamby agree exactly."""
    import albumentations

    return AlbumentationsImageTransform(
        albumentations.Compose(
            [
                albumentations.CenterCrop(image_size, image_size),
                # FLamby spells this Normalize(always_apply=True); always_apply is
                # deprecated in current albumentations and is a no-op here, since
                # Normalize already defaults to p=1.0. Behaviour is identical.
                albumentations.Normalize(),
            ]
        )
    )


def make_local_train_fn(known_diagnoses: Sequence[str], *, gamma: float = FLAMBY_FOCAL_GAMMA):
    """A ``local_train``-compatible callable using FLamby's Adam + focal loss.

    Signature matches :func:`fedcore.models.fed_train.local_train` so it can be
    passed to ``fedavg(..., local_train_fn=...)``; only the optimizer and the loss
    differ from the CIFAR arm's SGD+momentum/cross-entropy.  FLamby's
    ``common.py`` declares ``Optimizer = torch.optim.Adam`` and ``LR = 5e-4`` with
    no momentum/weight-decay overrides, so torch's Adam defaults are used.
    """
    import torch

    loss_fn = make_focal_loss(known_diagnoses, gamma=gamma)

    def local_train(model, loader, epochs: int, lr: float, device: str):
        model.to(device).train()
        criterion = loss_fn.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()
        return model

    return local_train


__all__ = [
    "AlbumentationsImageTransform",
    "FLAMBY_CLASS_ORDER",
    "FLAMBY_FOCAL_ALPHA_FULL8",
    "FLAMBY_FOCAL_GAMMA",
    "FLAMBY_PREPROCESSING_NOT_PREREGISTERED",
    "FLAMBY_TRAIN_OPS_NOT_PREREGISTERED",
    "eval_transform",
    "make_focal_loss",
    "make_local_train_fn",
    "restrict_focal_alpha",
    "train_transform",
]
