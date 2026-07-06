"""Configuration for the Fed-CORE FedOSR pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FedOSRConfig:
    # data / open-set split
    dataset: str = "cifar10"            # cifar10 | cifar100
    n_known: int = 6                    # number of known classes (rest are unknown)
    seed: int = 0

    # federation / heterogeneity
    n_clients: int = 5
    dirichlet_alpha: float = 0.1        # smaller => more non-IID

    # client-side TRAIN-label corruption (calibration stays clean)
    noise_type: str = "none"            # none | symmetric | asymmetric
    noise_rate: float = 0.0             # e.g. 0.35 (sym) or 0.20 (asym)

    # trusted calibration folds (fractions of the trusted clean pool)
    prop_frac: float = 0.34             # proposal fold
    cert_frac: float = 0.33             # certification fold
    test_frac: float = 0.33             # held-out test fold (deployment estimate)
    unknown_contamination: float = 0.30 # fraction of trusted points that are unknown-class

    # certification target
    alpha: float = 0.10                 # accepted selective-risk tolerance
    delta: float = 0.10                 # certificate failure probability
    gammas: tuple[float, ...] = (0.5, 0.7, 1.0)   # risk-buffer candidates
    Lambda: str = "simplex"             # simplex | box | known
    box_radius: float = 0.15            # for Lambda='box'

    # scores to evaluate (score-agnostic claim)
    scores: tuple[str, ...] = ("msp", "neg_entropy", "margin", "energy")

    # FedAvg training (torch path only)
    rounds: int = 50
    local_epochs: int = 2
    batch_size: int = 64
    lr: float = 0.01

    def folds(self) -> tuple[float, float, float]:
        s = self.prop_frac + self.cert_frac + self.test_frac
        return (self.prop_frac / s, self.cert_frac / s, self.test_frac / s)
