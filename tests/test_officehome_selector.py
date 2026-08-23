"""Office-Home selector: proposal-only, deterministic, kappa floor, no leakage."""

from __future__ import annotations

import numpy as np
import pytest

from fedcore.officehome_selector import (
    DEFAULT_GAMMAS,
    DEFAULT_SCORES,
    select_from_artifact,
    select_global_policy,
)


def _synthetic_proposal(n=600, c=45, seed=0, unknown_frac=0.3):
    rng = np.random.default_rng(seed)
    y = np.full(n, -1, dtype=int)
    known_mask = rng.random(n) >= unknown_frac
    y[known_mask] = rng.integers(0, c, size=int(known_mask.sum()))
    logits = rng.normal(size=(n, c))
    for i in range(n):
        if y[i] >= 0:
            logits[i, y[i]] += 3.0  # knowns are separable; unknowns diffuse
    return logits, y


class _LeakGuardArtifact(dict):
    """A dict that raises if any certification/evaluation key is READ."""

    def __getitem__(self, key):
        if str(key).startswith(("cert_", "eval_")):
            raise AssertionError(f"selection illegally read a held-out key: {key!r}")
        return super().__getitem__(key)


def test_selection_is_deterministic():
    logits, y = _synthetic_proposal()
    a, _ = select_global_policy(logits, y, alpha=0.20)
    b, _ = select_global_policy(logits, y, alpha=0.20)
    assert a.as_dict() == b.as_dict()


def test_selection_enumerates_all_score_gamma_candidates():
    logits, y = _synthetic_proposal()
    _, cands = select_global_policy(logits, y, alpha=0.20)
    assert len(cands) == len(DEFAULT_SCORES) * len(DEFAULT_GAMMAS)
    keys = {(c.score_name, c.gamma) for c in cands}
    assert keys == {(s, g) for s in DEFAULT_SCORES for g in DEFAULT_GAMMAS}


def test_selection_maximizes_proposal_coverage_among_feasible():
    logits, y = _synthetic_proposal()
    pol, cands = select_global_policy(logits, y, alpha=0.20)
    feasible = [c for c in cands if c.proposal_feasible]
    assert pol.feasible
    assert pol.prop_coverage == max(c.prop_coverage for c in feasible)


def test_kappa_floor_rejects_degenerate_coverage():
    logits, y = _synthetic_proposal()
    # An impossibly high coverage floor makes every candidate proposal-infeasible.
    pol, cands = select_global_policy(logits, y, alpha=0.20, kappa=0.999999)
    assert not pol.feasible
    assert all(not c.proposal_feasible for c in cands)


def test_selection_reads_no_certification_or_evaluation_labels():
    logits, y = _synthetic_proposal()
    artifact = _LeakGuardArtifact(
        prop_logits=logits,
        prop_y_open=y,
        cert_logits=np.zeros((10, 45)),
        cert_y_open=np.zeros(10),
        eval_logits=np.zeros((10, 45)),
        eval_y_open=np.zeros(10),
    )
    # If selection touched any cert_*/eval_* key, __getitem__ would raise.
    pol, _ = select_from_artifact(artifact, alpha=0.20)
    assert pol.feasible


def test_infeasible_policy_accepts_nothing():
    logits, y = _synthetic_proposal()
    pol, _ = select_global_policy(logits, y, alpha=0.20, kappa=0.999999)
    scores = np.linspace(0, 1, 100)
    assert not pol.accept(scores).any()
