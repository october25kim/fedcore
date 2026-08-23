"""Proposal-only global score+threshold selection for the Office-Home arm.

The selector is chosen on the PROPOSAL fold only. It never sees certification or
evaluation labels -- structurally, because :func:`select_global_policy` accepts
only proposal arrays, and (in the post-hoc pipeline) because
:func:`select_from_artifact` reads only ``prop_*`` keys of the logit artifact.

Selection contract (frozen, no favorable/outcome-driven choice):

* score in ``{msp, energy, margin}``; ``gamma in {0.3, 0.5, 0.7, 1.0}``.
* For each ``(score, gamma)`` candidate, one GLOBAL threshold is chosen on the
  proposal fold as the coverage-maximizing threshold whose empirical proposal
  selective risk respects the buffer ``gamma * alpha`` (via
  :func:`fedcore.selector.choose_threshold`).
* ``kappa`` (default 0.01) is a minimum proposal accepted-coverage floor: a
  candidate is *proposal-feasible* only if its chosen threshold accepts at least
  ``kappa`` of the proposal fold. This rejects degenerate near-zero-coverage
  thresholds without ever consulting a certification/test outcome.
* Among proposal-feasible candidates, pick the one with the LARGEST proposal
  accepted coverage (deterministic tie-break: smaller proposal risk, then score
  name, then gamma). No mixture/alpha/delta/solver knob enters selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from fedcore.scores import compute_score
from fedcore.selector import choose_threshold, empirical_risk_coverage, open_set_error


DEFAULT_SCORES: tuple[str, ...] = ("msp", "energy", "margin")
DEFAULT_GAMMAS: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0)
DEFAULT_KAPPA: float = 0.01


@dataclass(frozen=True)
class SelectionCandidate:
    """One ``(score, gamma)`` proposal-fold candidate and its diagnostics."""

    score_name: str
    gamma: float
    threshold: float
    threshold_feasible: bool
    prop_coverage: float
    prop_risk: float
    proposal_feasible: bool

    def as_dict(self) -> dict:
        return {
            "score_name": self.score_name,
            "gamma": float(self.gamma),
            "threshold": float(self.threshold),
            "threshold_feasible": bool(self.threshold_feasible),
            "prop_coverage": float(self.prop_coverage),
            "prop_risk": float(self.prop_risk),
            "proposal_feasible": bool(self.proposal_feasible),
        }


@dataclass(frozen=True)
class SelectedPolicy:
    """The frozen global policy selected on the proposal fold."""

    score_name: str
    gamma: float
    threshold: float
    feasible: bool
    prop_coverage: float
    prop_risk: float
    alpha: float
    kappa: float

    def accept(self, score: Sequence[float]) -> np.ndarray:
        """Boolean accept mask; empty (reject-all) when the policy is infeasible."""
        arr = np.asarray(score, dtype=float)
        if not self.feasible:
            return np.zeros(arr.shape, dtype=bool)
        return arr >= self.threshold

    def as_dict(self) -> dict:
        return {
            "score_name": self.score_name,
            "gamma": float(self.gamma),
            "threshold": float(self.threshold),
            "feasible": bool(self.feasible),
            "prop_coverage": float(self.prop_coverage),
            "prop_risk": float(self.prop_risk),
            "alpha": float(self.alpha),
            "kappa": float(self.kappa),
        }


def select_global_policy(
    prop_logits: np.ndarray,
    prop_y_open: np.ndarray,
    *,
    alpha: float,
    scores: Sequence[str] = DEFAULT_SCORES,
    gammas: Sequence[float] = DEFAULT_GAMMAS,
    kappa: float = DEFAULT_KAPPA,
    n_grid: int = 300,
) -> tuple[SelectedPolicy, list[SelectionCandidate]]:
    """Select one global (score, gamma, threshold) using proposal data only.

    ``prop_logits`` is ``(N, C)`` over the known classes; ``prop_y_open`` uses the
    open-set convention (known-class id or ``-1``). Certification and evaluation
    arrays are intentionally NOT parameters of this function.
    """

    logits = np.asarray(prop_logits, dtype=float)
    y_open = np.asarray(prop_y_open)
    if logits.ndim != 2 or logits.shape[0] != y_open.shape[0]:
        raise ValueError("prop_logits must be (N, C) aligned with prop_y_open")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0.0 <= kappa < 1.0:
        raise ValueError("kappa must lie in [0, 1)")

    pred = logits.argmax(axis=-1)
    err = open_set_error(pred, y_open)

    candidates: list[SelectionCandidate] = []
    for score_name in scores:
        score_vec = compute_score(score_name, logits)
        for gamma in gammas:
            sel = choose_threshold(score_vec, pred, y_open, float(gamma), float(alpha), n_grid=n_grid)
            cov, risk = empirical_risk_coverage(score_vec, err, sel.threshold)
            proposal_feasible = bool(sel.feasible) and cov >= kappa
            candidates.append(
                SelectionCandidate(
                    score_name=score_name,
                    gamma=float(gamma),
                    threshold=float(sel.threshold),
                    threshold_feasible=bool(sel.feasible),
                    prop_coverage=float(cov),
                    prop_risk=float(risk),
                    proposal_feasible=proposal_feasible,
                )
            )

    feasible = [c for c in candidates if c.proposal_feasible]
    if not feasible:
        chosen = SelectedPolicy(
            score_name="",
            gamma=float("nan"),
            threshold=float("inf"),
            feasible=False,
            prop_coverage=0.0,
            prop_risk=0.0,
            alpha=float(alpha),
            kappa=float(kappa),
        )
        return chosen, candidates

    # Deterministic, outcome-independent tie-break: maximize coverage, then
    # minimize risk, then score name, then gamma.
    best = min(
        feasible,
        key=lambda c: (-c.prop_coverage, c.prop_risk, c.score_name, c.gamma),
    )
    chosen = SelectedPolicy(
        score_name=best.score_name,
        gamma=best.gamma,
        threshold=best.threshold,
        feasible=True,
        prop_coverage=best.prop_coverage,
        prop_risk=best.prop_risk,
        alpha=float(alpha),
        kappa=float(kappa),
    )
    return chosen, candidates


def select_from_artifact(
    artifact: Mapping[str, np.ndarray],
    *,
    alpha: float,
    scores: Sequence[str] = DEFAULT_SCORES,
    gammas: Sequence[float] = DEFAULT_GAMMAS,
    kappa: float = DEFAULT_KAPPA,
    n_grid: int = 300,
) -> tuple[SelectedPolicy, list[SelectionCandidate]]:
    """Select from a loaded logit artifact, reading only its ``prop_*`` keys.

    This function deliberately touches ``prop_logits`` and ``prop_y_open`` only.
    A guard placed over ``cert_*``/``eval_*`` keys (see the selector monkeypatch
    test) confirms selection never reads a certification/evaluation outcome.
    """

    prop_logits = np.asarray(artifact["prop_logits"], dtype=float)
    prop_y_open = np.asarray(artifact["prop_y_open"])
    return select_global_policy(
        prop_logits,
        prop_y_open,
        alpha=alpha,
        scores=scores,
        gammas=gammas,
        kappa=kappa,
        n_grid=n_grid,
    )


__all__ = [
    "DEFAULT_GAMMAS",
    "DEFAULT_KAPPA",
    "DEFAULT_SCORES",
    "SelectedPolicy",
    "SelectionCandidate",
    "select_from_artifact",
    "select_global_policy",
]
