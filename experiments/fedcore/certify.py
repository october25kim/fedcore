"""Glue: select a risk-buffered selector, certify it, and emit the metric schema.

Ties the score -> selector -> per-client counts -> stratified certificate path
together and returns the project's standard metric dictionary.
"""
from __future__ import annotations

import numpy as np

from certificates import conditional_risk_certificate, cp_lower, stratified_certificate
from selector import (
    Selector,
    choose_threshold,
    counts_per_client,
    empirical_risk_coverage,
    open_set_error,
)


def _coverage_lcb(
    alow: np.ndarray,
    Lambda: str,
    lam: np.ndarray | None,
    box: tuple[np.ndarray, np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Worst-case-over-Lambda lower bound on accepted coverage = inf_lam sum lam*a.

    Uses the per-client lower acceptance bounds ``alow`` from the certificate.
    For the full simplex the infimum is ``min_j alow_j``; for a box it is
    approximated by sampling lambda in the box; for known lambda it is the dot.
    """
    if Lambda == "known" and lam is not None:
        return float((np.asarray(lam, float) * alow).sum())
    if Lambda == "box" and box is not None:
        rng = rng or np.random.default_rng(0)
        lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
        best = 1.0
        for _ in range(1000):
            ll = lo + (hi - lo) * rng.random(len(alow))
            s = ll.sum()
            if s > 0:
                best = min(best, float((ll / s * alow).sum()))
        return best
    return float(np.min(alow))  # simplex


def certify_for_score(
    *,
    score_name: str,
    prop: dict,
    cert: dict,
    test: dict,
    gamma: float,
    alpha: float,
    delta: float,
    n_clients: int,
    dirichlet_alpha: float,
    Lambda: str = "simplex",
    lam: np.ndarray | None = None,
    box: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Run the full proposal/certification/test pipeline for one (score, gamma).

    ``prop``/``cert``/``test`` are dicts with keys: 'score', 'pred', 'y_open',
    'client' (only cert needs 'client').
    """
    # 1) proposal: pick risk-buffered threshold on the proposal fold
    sel: Selector = choose_threshold(
        prop["score"], prop["pred"], prop["y_open"], gamma=gamma, alpha=alpha
    )
    prop_cov, prop_risk = empirical_risk_coverage(
        prop["score"], open_set_error(prop["pred"], prop["y_open"]), sel.threshold
    )

    # 2) certification: per-client counts + conditional selective-risk certificate
    #    (Theorem 1/1'; the sharper main result). The mass-ratio bound remains
    #    available via `stratified_certificate` as an Appendix-C baseline.
    A, K, n = counts_per_client(
        cert["score"], cert["pred"], cert["y_open"], cert["client"], sel, n_clients
    )
    res = conditional_risk_certificate(A, K, n, delta, Lambda=Lambda, lam=lam, box=box)
    cert_risk_ucb = res.U
    eps_a = delta / (2.0 * n_clients)
    alow = np.array([cp_lower(int(A[j]), int(n[j]), eps_a) for j in range(n_clients)])
    cert_cov_lcb = _coverage_lcb(alow, Lambda, lam, box=box)

    # 3) test fold: empirical deployment estimate
    test_cov, test_risk = empirical_risk_coverage(
        test["score"], open_set_error(test["pred"], test["y_open"]), sel.threshold
    )

    certified = bool(sel.feasible and cert_risk_ucb <= alpha)
    return {
        "score_name": score_name,
        "gamma": gamma,
        "alpha": alpha,
        "delta": delta,
        "Lambda": Lambda,
        "dirichlet_alpha": dirichlet_alpha,
        "n_clients": n_clients,
        "certified": certified,
        "cert_risk_ucb": round(cert_risk_ucb, 4),
        "cert_coverage_lcb": round(cert_cov_lcb, 4),
        "cert_n": int(A.sum()),
        "cert_k": int(K.sum()),
        "prop_coverage": round(prop_cov, 4),
        "prop_risk": round(prop_risk, 4),
        "test_coverage": round(test_cov, 4),
        "test_risk": round(test_risk, 4),
        "feasible_proposal": sel.feasible,
    }


def certify_grid(
    *,
    prop: dict,
    cert: dict,
    test: dict,
    score_names,
    gammas,
    alpha: float,
    delta: float,
    n_clients: int,
    dirichlet_alpha: float,
    Lambda: str = "simplex",
    lam: np.ndarray | None = None,
    box: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[dict]:
    """Sweep (score x gamma) and return one metric row per combination.

    ``prop``/``cert``/``test`` map score_name -> {'score','pred','y_open',[client]}.
    """
    rows: list[dict] = []
    for s in score_names:
        for g in gammas:
            rows.append(
                certify_for_score(
                    score_name=s,
                    prop=prop[s],
                    cert=cert[s],
                    test=test[s],
                    gamma=g,
                    alpha=alpha,
                    delta=delta,
                    n_clients=n_clients,
                    dirichlet_alpha=dirichlet_alpha,
                    Lambda=Lambda,
                    lam=lam,
                    box=box,
                )
            )
    return rows
