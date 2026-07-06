"""Acceptance scores from classifier logits (score-agnostic certificate).

Every score is oriented so that a HIGHER value means "more confidently a known
class" (i.e. more likely to be accepted). The certificate's validity does not
depend on which score is used; the score only affects how much coverage is
certified.
"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def msp(logits: np.ndarray) -> np.ndarray:
    """Maximum softmax probability."""
    return softmax(logits).max(axis=1)


def neg_entropy(logits: np.ndarray) -> np.ndarray:
    """Negative predictive entropy (higher => more confident)."""
    p = softmax(logits)
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
    return -ent


def margin(logits: np.ndarray) -> np.ndarray:
    """Gap between the top-2 softmax probabilities."""
    p = np.sort(softmax(logits), axis=1)
    return p[:, -1] - p[:, -2]


def energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Energy score. Free energy is -T*logsumexp(logits/T); ID data has LOW
    energy, so we return its negation (= T*logsumexp) to keep 'higher => accept'.
    """
    T = temperature
    z = logits / T
    m = z.max(axis=1, keepdims=True)
    lse = (m.squeeze(1) + np.log(np.exp(z - m).sum(axis=1)))
    return T * lse


SCORES = {
    "msp": msp,
    "neg_entropy": neg_entropy,
    "margin": margin,
    "energy": energy,
}


def compute_score(name: str, logits: np.ndarray) -> np.ndarray:
    if name not in SCORES:
        raise ValueError(f"unknown score {name!r}; choose from {list(SCORES)}")
    return SCORES[name](logits)


def scored_views(
    logits: np.ndarray,
    y_open: np.ndarray,
    client: np.ndarray,
    score_names,
) -> dict:
    """Build per-score fold views consumed by ``certify.certify_grid``.

    Returns ``{score_name: {'score','pred','y_open','client'}}``. Shared by both
    the fake-logit smoke and the real CIFAR runner so the certification path is
    identical for synthetic and real logits.
    """
    pred = logits.argmax(axis=1)
    out = {}
    for s in score_names:
        out[s] = {
            "score": compute_score(s, logits),
            "pred": pred,
            "y_open": y_open,
            "client": client,
        }
    return out
