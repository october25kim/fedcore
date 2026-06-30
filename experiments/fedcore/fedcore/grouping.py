"""Shared grouping / repartition / view helpers for aggregation and plotting.

Extracted (behaviour-preserving) from exp_feasibility_lever (``_group_map``, ``_repartition``)
and make_handoff (``_views_from_parts``) so the aggregators and figure generators import ONE
copy instead of reaching into experiment scripts. Bodies moved verbatim.
"""

from __future__ import annotations

import numpy as np

from fedcore.scores import scored_views


def _group_map(n_clients: int, G: int) -> np.ndarray:
    """Public, data-independent client->group map (contiguous balanced blocks)."""
    return np.array([c * G // n_clients for c in range(n_clients)], dtype=int)


def _repartition(pool, cert_frac, test_frac, seed):
    """Split the pooled trusted points into disjoint prop/cert/test folds."""
    rng = np.random.default_rng(seed)
    n = len(pool["y_open"])
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_cert = int(round(n * cert_frac))
    idx = {"test": perm[:n_test],
           "cert": perm[n_test:n_test + n_cert],
           "prop": perm[n_test + n_cert:]}
    out = {}
    for fold, ix in idx.items():
        out[fold] = {k: pool[k][ix] for k in ("logits", "y_open", "client")}
    return out


def _views_from_parts(parts, score):
    return {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                             parts[fn]["client"], [score])[score] for fn in ("prop", "cert", "test")}
