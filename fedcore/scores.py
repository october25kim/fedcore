"""Open-set confidence scores and their ``scored_views`` packaging.

Every score is oriented so that **higher => more likely a known (ID) class =>
more likely to accept**. Scores operate on raw logits of shape
``(N, C)`` over the ``C`` known classes.

Implemented scores:

* ``msp``        -- maximum softmax probability.
* ``neg_entropy``-- negative Shannon entropy of the softmax (higher = sharper).
* ``margin``     -- top1 minus top2 of the softmax.
* ``energy``     -- ``T * logsumexp(logits / T)`` (higher = more ID).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically-stable softmax."""
    logits = np.asarray(logits, dtype=float)
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def msp(logits: np.ndarray) -> np.ndarray:
    """Maximum softmax probability."""
    return softmax(logits).max(axis=-1)


def neg_entropy(logits: np.ndarray) -> np.ndarray:
    """Negative Shannon entropy of the softmax (higher = more confident)."""
    p = softmax(logits)
    ent = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=-1)
    return -ent


def margin(logits: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 softmax probability."""
    p = softmax(logits)
    part = np.sort(p, axis=-1)
    return part[..., -1] - part[..., -2]


def energy(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Energy score ``T * logsumexp(logits / T)`` (higher = more ID)."""
    logits = np.asarray(logits, dtype=float)
    z = logits / T
    m = z.max(axis=-1, keepdims=True)
    lse = m.squeeze(-1) + np.log(np.exp(z - m).sum(axis=-1))
    return T * lse


def score_norm(logits: np.ndarray) -> np.ndarray:
    """FOOGD-style score-norm detector (logit-space proxy).

    FOOGD scores OOD by the norm of a feature/score-function response; here we use
    the L2 norm of the logit vector as a logit-space proxy (higher norm => more
    confident/ID => accept). Faithful feature-space score-norm would require
    exporting penultimate features; this proxy reuses the exported logits.
    """
    logits = np.asarray(logits, dtype=float)
    return np.linalg.norm(logits, axis=-1)


_SCORE_FNS = {
    "msp": msp,
    "neg_entropy": neg_entropy,
    "margin": margin,
    "energy": energy,
    "score_norm": score_norm,
}


def compute_score(name: str, logits: np.ndarray) -> np.ndarray:
    """Dispatch to a named score function."""
    try:
        fn = _SCORE_FNS[name]
    except KeyError as exc:
        raise ValueError(f"unknown score {name!r}") from exc
    return fn(logits)


def scored_views(
    logits: np.ndarray,
    y_open: np.ndarray,
    client: np.ndarray,
    score_names: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Package per-score views of a batch of predictions.

    Returns ``{score_name: {"score", "pred", "y_open", "client"}}`` where
    ``pred = argmax(logits)`` (the predicted known class). ``y_open`` uses the
    open-set convention (known class id in ``[0, C)`` or ``-1`` for unknown).
    """
    logits = np.asarray(logits, dtype=float)
    y_open = np.asarray(y_open)
    client = np.asarray(client)
    pred = logits.argmax(axis=-1)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name in score_names:
        out[name] = {
            "score": compute_score(name, logits),
            "pred": pred,
            "y_open": y_open,
            "client": client,
        }
    return out
