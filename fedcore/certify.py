"""Glue: proposal -> certification -> test, emitting the canonical metric schema.

The certification path is IDENTICAL for the synthetic smoke and real CIFAR runs:

1. choose the selector threshold on the PROPOSAL fold (risk buffer ``gamma*alpha``);
2. compute per-client counts on the CERTIFICATION fold;
3. split ``delta`` equally into predeclared risk/coverage tails, certify the
   selective risk, and derive a positive coverage lower confidence bound;
4. evaluate empirically on the held-out TEST fold.

Metric schema keys (do not rename) -- see ``CLAUDE.md`` section 3.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from fedcore.certificate import (
    conditional_risk_certificate,
    cp_lower,
)
from fedcore.certificate.lambda_sets import (
    NormalizedBox,
    solve_normalized_box_coverage,
    uniform_box,
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
) -> tuple[float, Dict[str, object]]:
    """Worst-case-over-Lambda coverage lower confidence bound.

    For the full simplex, Corollary 1 uses the complete member-level tail at
    every stratum and takes their minimum (no stratum-count penalty).  A strict
    mixture restriction needs simultaneous acceptance endpoints and therefore
    divides its coverage tail across strata.
    """
    J = len(A)
    eps = delta if Lambda == "simplex" else delta / J
    alow = np.array(
        [cp_lower(int(A[j]), int(n[j]), eps) for j in range(J)], dtype=float
    )
    if Lambda == "simplex":
        return float(np.min(alow)), {
            "coverage_solver_status": "closed_form_full_simplex",
            "coverage_solver_certificate_valid": True,
            "coverage_solver_tolerance": 0.0,
            "coverage_solver_iterations": 0,
        }
    if Lambda == "known":
        lam_arr = np.asarray(lam if lam is not None else np.full(J, 1.0 / J))
        if lam_arr.shape != (J,) or np.any(lam_arr < 0.0) or lam_arr.sum() <= 0.0:
            raise ValueError("lam must be a non-negative length-J vector with positive sum")
        lam_arr = lam_arr / lam_arr.sum()
        result = solve_normalized_box_coverage(
            alow, NormalizedBox(lam_arr, lam_arr)
        )
        return float(result.value), result.diagnostics("coverage_solver")
    if Lambda == "box":
        result = solve_normalized_box_coverage(alow, uniform_box(J, box))
        return float(result.value), result.diagnostics("coverage_solver")
    raise ValueError(f"unknown Lambda={Lambda!r}")


def _solver_metadata(cert, coverage_meta: Dict[str, object]) -> Dict[str, object]:
    risk_valid = bool(cert.solver_certificate_valid)
    coverage_valid = bool(
        coverage_meta.get("coverage_solver_certificate_valid", False)
    )
    return {
        "risk_solver_status": cert.solver_status,
        "risk_solver_certificate_valid": risk_valid,
        "risk_solver_tolerance": float(cert.solver_tolerance),
        "risk_solver_iterations": int(cert.solver_iterations),
        "risk_solver_bracket_lower": float(cert.solver_bracket_lower),
        "risk_solver_bracket_upper": float(cert.solver_bracket_upper),
        "risk_solver_residual_lower": float(cert.solver_residual_lower),
        "risk_solver_residual_upper": float(cert.solver_residual_upper),
        "risk_solver_witness_value": float(cert.solver_witness_value),
        "risk_solver_reason": cert.solver_reason,
        **coverage_meta,
        "solver_status": (
            f"risk:{cert.solver_status};coverage:"
            f"{coverage_meta.get('coverage_solver_status', 'unknown')}"
        ),
        "solver_certificate_valid": bool(risk_valid and coverage_valid),
    }


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
    """Run the full proposal -> certify -> test path for one cell.

    ``delta`` is the total declared failure budget.  This compact legacy-facing
    entry point uses the predeclared equal split ``delta_r = delta_c = delta/2``.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    delta_r = delta / 2.0
    delta_c = delta / 2.0
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

    # (3) Current full-simplex / conservative-endpoint strict-mixture certificate.
    cert = conditional_risk_certificate(
        A, K, n, delta_r, Lambda=Lambda, lam=lam, box=box, seed=seed
    )
    cert_coverage_lcb, coverage_meta = _coverage_lcb(
        A, n, delta_c, Lambda, lam, box, seed
    )
    solver_meta = _solver_metadata(cert, coverage_meta)

    # (4) empirical evaluation on the held-out TEST fold
    test_err = open_set_error(test_view["pred"], test_view["y_open"])
    test_cov, test_risk = empirical_risk_coverage(
        test_view["score"], test_err, sel.threshold
    )

    certified = bool(
        cert.feasible
        and solver_meta["solver_certificate_valid"]
        and cert.U <= alpha
        and cert_coverage_lcb > 0.0
    )

    return {
        "score_name": score_name,
        "gamma": gamma,
        "alpha": alpha,
        "delta": delta,
        "delta_r": delta_r,
        "delta_c": delta_c,
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
        **solver_meta,
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
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    delta_r = delta / 2.0
    delta_c = delta / 2.0
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
            Ap, Kp, np_, delta_r, Lambda=Lambda, lam=lam, box=box, seed=seed
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
        A, K, n, delta_r, Lambda=Lambda, lam=lam, box=box, seed=seed
    )
    cert_coverage_lcb, coverage_meta = _coverage_lcb(
        A, n, delta_c, Lambda, lam, box, seed
    )
    solver_meta = _solver_metadata(cert, coverage_meta)

    test_err = open_set_error(test_view["pred"], test_view["y_open"])
    test_cov, test_risk = empirical_risk_coverage(
        test_view["score"], test_err, sel.threshold
    )
    certified = bool(
        cert.feasible
        and solver_meta["solver_certificate_valid"]
        and cert.U <= alpha
        and cert_coverage_lcb > 0.0
    )

    return {
        "score_name": score_name,
        "gamma": gamma_star,
        "alpha": alpha,
        "delta": delta,
        "delta_r": delta_r,
        "delta_c": delta_c,
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
        **solver_meta,
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
    mixture_spec=None,
) -> Dict[str, object]:
    """Grouped-stratified certificate (paper sec 4.4): worst-GROUP guarantee.

    TWO PATHS, and they are NOT interchangeable
    -------------------------------------------
    * ``mixture_spec`` GIVEN -> the EXACT path. The certification sample is DRAWN from
      the declared group mixture via the pre-registered sampler
      (``fedcore.group_draw.draw_group_certification_sample``): per observation,
      ``client ~ Categorical(pi_{.|g})`` then one unit from that client's frozen
      certification reservoir WITH REPLACEMENT. Each observation is i.i.d. from the
      group-g mixture, so ``K_g | A_g ~ Bin(A_g, r_g)`` holds EXACTLY and the result is
      a grouped-mixture CERTIFICATE.

    * ``mixture_spec`` OMITTED -> the LEGACY DIAGNOSTIC path. It merely relabels each
      point's client id to its group id and certifies the fixed-quota certification
      fold. Fixed quotas are deterministic largest-remainder allocations, NOT draws from
      the group mixture, so the conditional binomial law does NOT hold exactly. The
      returned row is tagged ``draw_construction='fixed_quota_largest_remainder'`` and
      ``theorem_exact=False``: it is a diagnostic and MUST NOT be reported as a
      grouped-mixture certificate result.

    ``eps = delta/G``. ``G=1`` is the pooled certificate -- valid only under matched
    mixture; label accordingly.

    The selector still sees the PROPOSAL fold only: the drawn sample replaces the
    CERTIFICATION fold alone, which is exactly where the conditional law must hold.
    """
    def regroup(view):
        v = dict(view)
        v["client"] = group_map[np.asarray(view["client"])]
        return v

    cert_for_certificate = regroup(cert_view)
    draw_tags: Dict[str, object] = {
        "draw_construction": "fixed_quota_largest_remainder",
        "sampler_invoked": "none",
        "theorem_exact": False,
        "artifact_class": "diagnostic_fixed_quota",
        "manuscript_status": (
            "not_theorem_exact__not_manuscript_headline__"
            "superseded_until_reproduced_under_exact_sampler"
        ),
    }

    if mixture_spec is not None:
        from fedcore.group_draw import draw_group_certification_sample

        cert_for_certificate, record = draw_group_certification_sample(
            cert_view, mixture_spec
        )
        draw_tags = {
            "draw_construction": record.draw_construction,
            "sampler_invoked": record.sampler,
            "theorem_exact": True,
            "artifact_class": "exact_group_mixture_certificate",
            "manuscript_status": "manuscript_facing_exact_sampler",
            "n_g": dict(record.n_g),
            "seed_per_group": dict(record.seed_per_group),
            "draw_record": record,
        }

    out = certify_best_gamma(
        regroup(prop_view), cert_for_certificate, regroup(test_view),
        score_name=score_name, gammas=gammas, alpha=alpha, delta=delta,
        n_clients=G, dirichlet_alpha=float("nan"), Lambda=Lambda,
        box=box, seed=seed, margin=margin,
    )
    out.update(draw_tags)
    return out


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
