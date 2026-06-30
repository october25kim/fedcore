"""Glue: proposal -> certification -> test, emitting the canonical metric schema.

The certification path is IDENTICAL for the synthetic smoke and real CIFAR runs:

1. choose the selector threshold on the PROPOSAL fold (risk buffer ``gamma*alpha``);
2. compute per-client counts on the CERTIFICATION fold;
3. certify the selective risk with the **conditional** certificate (Theorem 1/1',
   PRIMARY) and derive a coverage lower confidence bound;
4. evaluate empirically on the held-out TEST fold.

Metric schema keys (do not rename) -- see ``CLAUDE.md`` section 3.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from fedcore.certificate import (
    _sample_lambdas,
    conditional_risk_certificate,
    cp_lower,
)
from fedcore.selector import (
    choose_threshold,
    counts_per_client,
    empirical_risk_coverage,
    open_set_error,
)


def _coverage_lcb(
    A: np.ndarray,
    n: np.ndarray,
    delta: float,
    Lambda: str,
    lam: Optional[Sequence[float]],
    box: float,
    seed: int,
) -> float:
    """Worst-case-over-Lambda coverage lower confidence bound.

    Uses ``alow_j = U-(A_j, n_j; delta/(2J))`` and minimizes the achievable
    accepted coverage over the deployment mixture set.
    """
    J = len(A)
    eps = delta / (2.0 * J)
    alow = np.array(
        [cp_lower(int(A[j]), int(n[j]), eps) for j in range(J)], dtype=float
    )
    if Lambda == "simplex":
        return float(np.min(alow))
    if Lambda == "known":
        lam_arr = np.asarray(lam if lam is not None else np.full(J, 1.0 / J))
        return float(np.sum(lam_arr * alow))
    if Lambda == "box":
        rng = np.random.default_rng(seed)
        lams = _sample_lambdas(J, box, 256, rng)
        return float(min(np.sum(l * alow) for l in lams))
    raise ValueError(f"unknown Lambda={Lambda!r}")


def certify_for_score(
    score_name: str,
    prop_view: Dict[str, np.ndarray],
    cert_view: Dict[str, np.ndarray],
    test_view: Dict[str, np.ndarray],
    *,
    gamma: float,
    alpha: float,
    delta: float,
    Lambda: str,
    n_clients: int,
    dirichlet_alpha: float,
    lam: Optional[Sequence[float]] = None,
    box: float = 0.15,
    seed: int = 0,
) -> Dict[str, object]:
    """Run the full proposal -> certify -> test path for one (score, gamma, Lambda)."""
    # (1) selector on the PROPOSAL fold only
    sel = choose_threshold(
        prop_view["score"], prop_view["pred"], prop_view["y_open"], gamma, alpha
    )
    prop_err = open_set_error(prop_view["pred"], prop_view["y_open"])
    prop_cov, prop_risk = empirical_risk_coverage(
        prop_view["score"], prop_err, sel.threshold
    )

    # (2) per-client counts on the CERTIFICATION fold
    A, K, n = counts_per_client(
        cert_view["score"], cert_view["pred"], cert_view["y_open"],
        cert_view["client"], sel, n_clients,
    )

    # (3) PRIMARY: conditional selective-risk certificate (Theorem 1/1')
    cert = conditional_risk_certificate(
        A, K, n, delta, Lambda=Lambda, lam=lam, box=box, seed=seed
    )
    cert_coverage_lcb = _coverage_lcb(A, n, delta, Lambda, lam, box, seed)

    # (4) empirical evaluation on the held-out TEST fold
    test_err = open_set_error(test_view["pred"], test_view["y_open"])
    test_cov, test_risk = empirical_risk_coverage(
        test_view["score"], test_err, sel.threshold
    )

    certified = bool(cert.feasible and cert.U <= alpha)

    return {
        "score_name": score_name,
        "gamma": gamma,
        "alpha": alpha,
        "delta": delta,
        "Lambda": Lambda,
        "dirichlet_alpha": dirichlet_alpha,
        "n_clients": n_clients,
        "certified": certified,
        "cert_risk_ucb": float(cert.U),
        "cert_coverage_lcb": float(cert_coverage_lcb),
        "cert_n": int(np.sum(A)),
        "cert_k": int(np.sum(K)),
        "prop_coverage": float(prop_cov),
        "prop_risk": float(prop_risk),
        "test_coverage": float(test_cov),
        "test_risk": float(test_risk),
    }


def certify_best_gamma(
    prop_view: Dict[str, np.ndarray],
    cert_view: Dict[str, np.ndarray],
    test_view: Dict[str, np.ndarray],
    *,
    score_name: str,
    gammas: Sequence[float],
    alpha: float,
    delta: float,
    n_clients: int,
    dirichlet_alpha: float,
    Lambda: str = "simplex",
    lam: Optional[Sequence[float]] = None,
    box: float = 0.15,
    seed: int = 0,
    margin: float = 0.0,
) -> Dict[str, object]:
    """Certified-coverage-maximizing selector, VALIDITY-PRESERVING.

    The risk buffer ``gamma`` is chosen on the PROPOSAL fold only: for each gamma
    we build the risk-buffered selector and a PROPOSAL-side proxy certificate, then
    pick ``gamma*`` = the most aggressive buffer whose proposal-side certificate
    clears ``alpha - margin`` (max proposal coverage among those). The single chosen
    selector ``t_{gamma*}`` is then certified ONCE on the CERTIFICATION fold at the
    FULL ``delta`` -- no union/selection penalty, because ``t_{gamma*}`` is a
    function of the proposal fold alone (independent of the certification fold).

    ``margin`` (>=0) is a proposal-side safety buffer: requiring the proxy to clear
    ``alpha - margin`` makes the chosen operating point less likely to fail on the
    certification fold (fixes alpha-frontier non-monotonicity from proxy optimism).
    """
    prop_err = open_set_error(prop_view["pred"], prop_view["y_open"])

    # (1)+(2) per-gamma proposal-side selector + proxy certificate
    cands = []
    for gamma in gammas:
        sel = choose_threshold(
            prop_view["score"], prop_view["pred"], prop_view["y_open"], gamma, alpha
        )
        cov_p, risk_p = empirical_risk_coverage(
            prop_view["score"], prop_err, sel.threshold
        )
        Ap, Kp, np_ = counts_per_client(
            prop_view["score"], prop_view["pred"], prop_view["y_open"],
            prop_view["client"], sel, n_clients,
        )
        u_proxy = conditional_risk_certificate(
            Ap, Kp, np_, delta, Lambda=Lambda, lam=lam, box=box, seed=seed
        ).U
        cands.append({"gamma": gamma, "sel": sel, "cov_p": cov_p, "u_proxy": u_proxy})

    # (3) gamma* = argmax proposal coverage among proxy-certified; else smallest gamma
    feas = [c for c in cands if c["sel"].feasible and c["u_proxy"] <= alpha - margin]
    if feas:
        chosen = max(feas, key=lambda c: c["cov_p"])
    else:
        chosen = min(cands, key=lambda c: c["gamma"])
    gamma_star, sel = chosen["gamma"], chosen["sel"]

    prop_cov, prop_risk = empirical_risk_coverage(
        prop_view["score"], prop_err, sel.threshold
    )

    # (4) certify the single chosen selector on the CERT fold at FULL delta
    A, K, n = counts_per_client(
        cert_view["score"], cert_view["pred"], cert_view["y_open"],
        cert_view["client"], sel, n_clients,
    )
    cert = conditional_risk_certificate(
        A, K, n, delta, Lambda=Lambda, lam=lam, box=box, seed=seed
    )
    cert_coverage_lcb = _coverage_lcb(A, n, delta, Lambda, lam, box, seed)

    test_err = open_set_error(test_view["pred"], test_view["y_open"])
    test_cov, test_risk = empirical_risk_coverage(
        test_view["score"], test_err, sel.threshold
    )
    certified = bool(cert.feasible and cert.U <= alpha)

    return {
        "score_name": score_name,
        "gamma": gamma_star,
        "alpha": alpha,
        "delta": delta,
        "Lambda": Lambda,
        "dirichlet_alpha": dirichlet_alpha,
        "n_clients": n_clients,
        "certified": certified,
        "cert_risk_ucb": float(cert.U),
        "cert_coverage_lcb": float(cert_coverage_lcb),
        "cert_n": int(np.sum(A)),
        "cert_k": int(np.sum(K)),
        "prop_coverage": float(prop_cov),
        "prop_risk": float(prop_risk),
        "test_coverage": float(test_cov),
        "test_risk": float(test_risk),
        "gamma_star": gamma_star,
        "u_proxy": float(chosen["u_proxy"]),
    }


def certify_best_gamma_grouped(
    prop_view: Dict[str, np.ndarray],
    cert_view: Dict[str, np.ndarray],
    test_view: Dict[str, np.ndarray],
    *,
    score_name: str,
    group_map: np.ndarray,           # client id -> group id (PUBLIC, data-independent)
    G: int,
    gammas: Sequence[float],
    alpha: float,
    delta: float,
    Lambda: str = "box",
    box: float = 0.15,
    seed: int = 0,
    margin: float = 0.0,
) -> Dict[str, object]:
    """Grouped-stratified certificate (paper sec 4.4): worst-GROUP guarantee.

    Relabels each point's client id to its PUBLIC group id and applies the
    conditional certificate over ``G`` groups (eps = delta/G). Larger groups carry
    more accepted points, raising per-group counts toward the Theorem-2 floor. This
    is also the privacy compromise (secure-aggregate within groups). ``G=1`` is the
    pooled certificate -- valid only under matched mixture; label accordingly.
    """
    def regroup(view):
        v = dict(view)
        v["client"] = group_map[np.asarray(view["client"])]
        return v

    return certify_best_gamma(
        regroup(prop_view), regroup(cert_view), regroup(test_view),
        score_name=score_name, gammas=gammas, alpha=alpha, delta=delta,
        n_clients=G, dirichlet_alpha=float("nan"), Lambda=Lambda,
        box=box, seed=seed, margin=margin,
    )


def certify_grid(
    prop_views: Dict[str, Dict[str, np.ndarray]],
    cert_views: Dict[str, Dict[str, np.ndarray]],
    test_views: Dict[str, Dict[str, np.ndarray]],
    *,
    scores: Sequence[str],
    gammas: Sequence[float],
    alpha: float,
    delta: float,
    Lambdas: Sequence[str] = ("simplex", "box"),
    n_clients: int,
    dirichlet_alpha: float,
    box: float = 0.15,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Sweep score x gamma x Lambda and return a list of metric rows."""
    rows: List[Dict[str, object]] = []
    for Lambda in Lambdas:
        for sname in scores:
            for gamma in gammas:
                rows.append(
                    certify_for_score(
                        sname,
                        prop_views[sname],
                        cert_views[sname],
                        test_views[sname],
                        gamma=gamma,
                        alpha=alpha,
                        delta=delta,
                        Lambda=Lambda,
                        n_clients=n_clients,
                        dirichlet_alpha=dirichlet_alpha,
                        box=box,
                        seed=seed,
                    )
                )
    return rows


# Canonical schema is defined once in config.py; re-exported here for backward compatibility.
from fedcore.config import CANONICAL_SCHEMA as METRIC_KEYS  # noqa: E402
