"""Configuration for Fed-CORE FedOSR runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FedOSRConfig:
    """All knobs for a federated open-set certification run.

    The calibration set is split into three disjoint folds: ``proposal`` (choose
    the selector threshold), ``certification`` (compute the certificate), and
    ``test`` (final unbiased evaluation). ``folds()`` returns the fractions
    normalized to sum to 1.
    """

    dataset: str = "cifar10"
    n_known: int = 6
    seed: int = 0

    # federation / heterogeneity
    n_clients: int = 5
    dirichlet_alpha: float = 0.1

    # client-side TRAIN-label corruption (calibration/test stay clean)
    noise_type: str = "none"  # 'none' | 'symmetric' | 'asymmetric'
    noise_rate: float = 0.0

    # calibration fold fractions (proposal / certification / test)
    prop_frac: float = 0.4
    cert_frac: float = 0.3
    test_frac: float = 0.3
    unknown_contamination: float = 0.30

    # certification targets
    alpha: float = 0.10
    delta: float = 0.10
    gammas: Tuple[float, ...] = (0.2, 0.3, 0.5, 0.7, 1.0)
    Lambda: str = "simplex"
    box_radius: float = 0.15

    scores: Tuple[str, ...] = ("msp", "neg_entropy", "margin", "energy")

    # training
    rounds: int = 50
    local_epochs: int = 2
    batch_size: int = 64
    lr: float = 0.01

    def folds(self) -> Tuple[float, float, float]:
        """Return (prop, cert, test) fractions normalized to sum to 1."""
        total = self.prop_frac + self.cert_frac + self.test_frac
        if total <= 0:
            raise ValueError("fold fractions must sum to a positive number")
        return (
            self.prop_frac / total,
            self.cert_frac / total,
            self.test_frac / total,
        )
