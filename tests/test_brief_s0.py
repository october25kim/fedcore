"""Pre-registered section-0 unit tests.

Every expected value here is fixed by the brief and was independently reproduced
before the implementation was written. These tests are the acceptance gate: they
must pass before any tier of the campaign is launched.

Run: python -m pytest tests/test_brief_s0.py -q
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from fedcore.certificate.allocation import (
    allocate_R1,
    allocate_R2,
    allocated_risk_certificate,
    uniform_allocation,
    zero_error_floor,
    zero_error_frontier,
)
from fedcore.certificate.allocation import _zero_error_floor_real
from fedcore.certificate.cp import cp_lower, cp_upper
from fedcore.certificate.lambda_sets import (
    NormalizedBox,
    exact_risk_supremum,
    uniform_box,
    vertex_enumeration_reference,
)
from fedcore.mixture import traffic_mixture_confidence_box


# --------------------------------------------------------------------------- #
# 1. Zero-error Clopper-Pearson closed form
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("A", [20, 36, 44, 100])
@pytest.mark.parametrize("eps", [0.02, 0.01, 0.05, 0.1 / 3.0])
def test_cp_upper_zero_error_closed_form(A: int, eps: float) -> None:
    """U(0, A, eps) == 1 - eps^(1/A) to 1e-9."""
    assert cp_upper(0, A, eps) == pytest.approx(1.0 - eps ** (1.0 / A), abs=1e-9)


@pytest.mark.parametrize(
    "A,eps", [(400, 8.4e-19), (400, 1e-17), (600, 1e-25), (9000, 1e-300)]
)
def test_cp_upper_survives_tiny_eps(A: int, eps: float) -> None:
    """The allocated rules hand out eps far below 1.1e-16; the bound must hold.

    Computing ppf(1 - eps) collapses to 1.0 there and would refuse exactly the
    count-rich clients R1/R3 exist to certify.
    """
    assert cp_upper(0, A, eps) == pytest.approx(1.0 - eps ** (1.0 / A), abs=1e-9)
    assert cp_upper(0, A, eps) < 1.0


def test_cp_conventions() -> None:
    """U = 1 when A == 0 or K == A; L = 0 when K == 0."""
    assert cp_upper(0, 0, 0.05) == 1.0
    assert cp_upper(7, 7, 0.05) == 1.0
    assert cp_lower(0, 50, 0.05) == 0.0
    assert cp_lower(0, 0, 0.05) == 0.0


# --------------------------------------------------------------------------- #
# 2. Theorem 3 zero-error floors
# --------------------------------------------------------------------------- #
def test_theorem3_floor_J5() -> None:
    """J=5 at alpha=0.10: delta_r=0.10 -> 38; delta_r=0.05 -> 44."""
    assert zero_error_floor(0.10 / 5, 0.10) == 38
    assert zero_error_floor(0.05 / 5, 0.10) == 44


@pytest.mark.parametrize("G,expected", [(5, 44), (3, 39), (2, 36), (1, 29)])
def test_theorem3_floor_group_ladder(G: int, expected: int) -> None:
    """G in {5,3,2,1} at delta_r=0.05, alpha=0.10 -> {44, 39, 36, 29}."""
    assert zero_error_floor(0.05 / G, 0.10) == expected


def test_theorem3_floor_matches_uniform_certificate() -> None:
    """The floor is exactly the count at which the uniform certificate deploys."""
    J, delta_r, alpha = 5, 0.10, 0.10
    floor = zero_error_floor(delta_r / J, alpha)
    eps = uniform_allocation(J, delta_r)
    A_at = np.full(J, floor)
    A_below = np.full(J, floor - 1)
    K = np.zeros(J, dtype=int)
    assert allocated_risk_certificate(A_at, K, eps, delta_r, alpha=alpha).U <= alpha
    assert allocated_risk_certificate(A_below, K, eps, delta_r, alpha=alpha).U > alpha


# --------------------------------------------------------------------------- #
# 3. Pooling counterexample arithmetic
# --------------------------------------------------------------------------- #
def test_pooling_counterexample_calibration_risk() -> None:
    """a=(.7,.7,.7,.7,.5), r=(.02,.02,.02,.02,.3) -> pooled sum(a r)/sum(a) = 0.0624."""
    a = np.array([0.7, 0.7, 0.7, 0.7, 0.5])
    r = np.array([0.02, 0.02, 0.02, 0.02, 0.3])
    pooled = float((a * r).sum() / a.sum())
    assert pooled == pytest.approx(0.0624, abs=5e-5)
    # The pooled value hides a client at 0.3: pooling is not conservative.
    assert pooled < r.max()


# --------------------------------------------------------------------------- #
# 4. R1 equalizes the zero-error slack
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "A_hat",
    [
        [150, 40, 40, 40, 40],
        [900, 120, 60, 45, 30],
        [10, 10, 10, 10, 10],
        [1, 2, 3, 400, 5000],
    ],
)
def test_R1_equalizes_slack(A_hat) -> None:
    """floors_j - A_hat_j is constant across j to 1e-6."""
    delta_r, alpha = 0.10, 0.10
    eps = allocate_R1(A_hat, delta_r, alpha)
    slack = np.array([_zero_error_floor_real(e, alpha) - a for e, a in zip(eps, A_hat)])
    assert slack.max() - slack.min() == pytest.approx(0.0, abs=1e-6)
    assert float(eps.sum()) == pytest.approx(delta_r, rel=1e-12)


def test_R1_does_not_underflow_on_large_counts() -> None:
    """(1-alpha)^A underflows for A ~ 9000; R1 must stay a valid allocation.

    The count-rich client's true R1 budget is ~1e-412, unrepresentable in float64.
    Clamping it to the smallest positive normal keeps sum(eps) == delta_r (so the
    union bound is intact) and only tightens that client's rbar, so the cell stays
    certifiable instead of collapsing to rbar = 1.
    """
    delta_r, alpha = 0.10, 0.10
    eps = allocate_R1([9000, 40, 40], delta_r, alpha)
    assert np.all(eps > 0.0)
    assert float(eps.sum()) == pytest.approx(delta_r, rel=1e-12)
    # The clamped client must still certify at alpha on its own accepted mass.
    cert = allocated_risk_certificate(
        [9000, 40, 40], [0, 0, 0], eps, delta_r, rule="R1", alpha=alpha
    )
    assert cert.rbar[0] <= alpha
    assert cert.budget_respected


# --------------------------------------------------------------------------- #
# 5. Corollary 2 worked example
# --------------------------------------------------------------------------- #
def test_corollary2_worked_example() -> None:
    """alpha=.10, delta_r=.10, A=(200,36,36,36,36), K=0.

    Uniform floor 38 => no deploy (max rbar ~ 0.103); frontier ~ 0.0901 <= 0.10;
    R1 with A_hat=(150,40,40,40,40) => floors (145.01, 35.01 x4), deploys, max
    rbar ~ 0.0974.
    """
    alpha, delta_r, J = 0.10, 0.10, 5
    A = np.array([200, 36, 36, 36, 36])
    K = np.zeros(J, dtype=int)

    # Uniform: refuses.
    unif = allocated_risk_certificate(
        A, K, uniform_allocation(J, delta_r), delta_r, alpha=alpha
    )
    assert unif.U == pytest.approx(0.103, abs=5e-4)
    assert unif.U > alpha
    assert zero_error_floor(delta_r / J, alpha) == 38

    # Frontier says an allocation exists.
    frontier = zero_error_frontier(A, alpha)
    assert frontier == pytest.approx(0.0901, abs=5e-4)
    assert frontier <= delta_r

    # R1 on proposal-fold counts: deploys.
    A_hat = np.array([150, 40, 40, 40, 40])
    eps = allocate_R1(A_hat, delta_r, alpha)
    floors = np.array([_zero_error_floor_real(e, alpha) for e in eps])
    assert floors[0] == pytest.approx(145.01, abs=0.02)
    assert np.allclose(floors[1:], 35.01, atol=0.02)
    r1 = allocated_risk_certificate(A, K, eps, delta_r, rule="R1", alpha=alpha)
    assert r1.U == pytest.approx(0.0974, abs=5e-4)
    assert r1.U <= alpha


def test_corollary2_frontier_is_exactly_the_feasibility_boundary() -> None:
    """An allocation certifying at alpha exists iff sum_j (1-alpha)^A_j <= delta_r."""
    alpha, delta_r = 0.10, 0.10
    rng = np.random.default_rng(11)
    for _ in range(300):
        A = rng.integers(20, 90, size=4)
        # The minimal spend certifying every client at alpha is exactly the frontier.
        need = np.power(1.0 - alpha, A.astype(float))
        feasible_by_frontier = float(need.sum()) <= delta_r
        if feasible_by_frontier:
            # Spending exactly the per-client need certifies all of them.
            cert = allocated_risk_certificate(
                A, np.zeros(4, int), need, delta_r, alpha=alpha
            )
            assert cert.U <= alpha + 1e-12
        else:
            # No budget-respecting allocation can certify: the cheapest one already
            # overspends, and any cheaper eps_j only raises rbar_j.
            with pytest.raises(ValueError):
                allocated_risk_certificate(
                    A, np.zeros(4, int), need, delta_r, alpha=alpha
                )


def test_corollary2_strictly_weaker_than_theorem3_under_heterogeneity() -> None:
    """The frontier certifies cells the uniform Theorem-3 floor refuses."""
    alpha, delta_r, J = 0.10, 0.10, 5
    A = np.array([200, 36, 36, 36, 36])
    assert zero_error_frontier(A, alpha) <= delta_r  # frontier: feasible
    assert A.min() < zero_error_floor(delta_r / J, alpha)  # uniform: infeasible


# --------------------------------------------------------------------------- #
# 6. R2 degenerates to uniform on identical clients
# --------------------------------------------------------------------------- #
def test_R2_identical_clients_matches_uniform() -> None:
    """R2 with (K_hat=1, A_hat=100) x 5 equals the uniform max-UCB to 1e-3."""
    J, delta_r = 5, 0.10
    A_hat = np.full(J, 100)
    K_hat = np.full(J, 1)
    eps_r2 = allocate_R2(A_hat, K_hat, delta_r)
    eps_unif = uniform_allocation(J, delta_r)

    A, K = np.full(J, 100), np.full(J, 1)
    u_r2 = allocated_risk_certificate(A, K, eps_r2, delta_r, rule="R2").U
    u_unif = allocated_risk_certificate(A, K, eps_unif, delta_r, rule="uniform").U
    assert u_r2 == pytest.approx(u_unif, abs=1e-3)
    assert float(eps_r2.sum()) <= delta_r + 1e-12


def test_R2_reduces_to_R1_structure_when_zero_errors() -> None:
    """With K_hat = 0 the R2 water level reproduces R1's ordering and budget."""
    delta_r = 0.10
    A_hat = np.array([150, 40, 40, 40, 40])
    eps_r2 = allocate_R2(A_hat, np.zeros(5, int), delta_r)
    eps_r1 = allocate_R1(A_hat, delta_r, 0.10)
    assert float(eps_r2.sum()) <= delta_r + 1e-12
    # Same ordering: the count-rich client receives the least budget.
    assert np.argsort(eps_r2).tolist() == np.argsort(eps_r1).tolist()
    assert int(np.argmin(eps_r2)) == 0


# --------------------------------------------------------------------------- #
# 7. LFP solver vs exact vertex enumeration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("J", [2, 3, 4, 5])
def test_lfp_matches_vertex_enumeration(J: int) -> None:
    """box-Lambda_G supremum equals the exact vertex-enumeration value."""
    rng = np.random.default_rng(1234 + J)
    for _ in range(60):
        rbar = rng.uniform(0.0, 0.6, J)
        alow = rng.uniform(0.05, 0.6, J)
        ahigh = alow + rng.uniform(0.0, 0.35, J)
        rho = float(rng.choice([0.0, 0.05, 0.15, 0.25, 0.5]))
        box = uniform_box(J, rho)
        got, feasible = exact_risk_supremum(rbar, alow, ahigh, box)
        want = vertex_enumeration_reference(rbar, alow, ahigh, box)
        assert feasible
        assert got == pytest.approx(want, abs=1e-8)


def test_lfp_matches_vertex_enumeration_asymmetric_box() -> None:
    """Also exact on non-uniform centers (grouped strata of unequal declared mass)."""
    rng = np.random.default_rng(99)
    for _ in range(60):
        J = 4
        lo = rng.uniform(0.0, 0.2, J)
        hi = lo + rng.uniform(0.01, 0.4, J)
        box = NormalizedBox(lo, hi)
        rbar = rng.uniform(0.0, 0.5, J)
        alow = rng.uniform(0.1, 0.5, J)
        ahigh = alow + rng.uniform(0.0, 0.3, J)
        got, _ = exact_risk_supremum(rbar, alow, ahigh, box)
        want = vertex_enumeration_reference(rbar, alow, ahigh, box)
        assert got == pytest.approx(want, abs=1e-8)


def test_lfp_rho_zero_is_the_declared_center() -> None:
    """rho = 0 collapses to the single declared mixture."""
    J = 4
    rbar = np.array([0.05, 0.10, 0.20, 0.40])
    alow = np.full(J, 0.4)
    ahigh = np.full(J, 0.6)
    got, _ = exact_risk_supremum(rbar, alow, ahigh, uniform_box(J, 0.0))
    want = vertex_enumeration_reference(rbar, alow, ahigh, uniform_box(J, 0.0))
    assert got == pytest.approx(want, abs=1e-9)


def test_lfp_never_exceeds_the_simplex_bound() -> None:
    """No mixture set can beat max_j rbar_j -- Theorem 1 dominates Theorem 2."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        J = int(rng.integers(2, 6))
        rbar = rng.uniform(0.0, 0.9, J)
        alow = rng.uniform(0.05, 0.5, J)
        ahigh = alow + rng.uniform(0.0, 0.4, J)
        rho = float(rng.choice([0.0, 0.15, 0.5, 1.0]))
        got, _ = exact_risk_supremum(rbar, alow, ahigh, uniform_box(J, rho))
        assert got <= rbar.max() + 1e-12


def test_lfp_is_an_upper_bound_on_sampled_mixtures() -> None:
    """The exact sup dominates any sampled interior mixture -- the legacy defect.

    The retired path took a max over 256 random interior draws, which cannot reach
    a vertex and therefore under-reported the supremum. Guard against regression.
    """
    rng = np.random.default_rng(3)
    J = 5
    rbar = np.array([0.05, 0.05, 0.05, 0.05, 0.30])
    alow = np.full(J, 0.5)
    ahigh = np.full(J, 0.8)
    box = uniform_box(J, 0.15)
    exact, _ = exact_risk_supremum(rbar, alow, ahigh, box)
    sampled_max = -np.inf
    for _ in range(2000):
        lam = rng.uniform(box.lo, box.hi)
        lam = lam / lam.sum()
        a = rng.uniform(alow, ahigh)
        sampled_max = max(sampled_max, float((lam * a * rbar).sum() / (lam * a).sum()))
    assert exact >= sampled_max - 1e-12


# --------------------------------------------------------------------------- #
# 8. Data-derived mixture box coverage (Monte Carlo)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m", [500, 2000])
@pytest.mark.parametrize(
    "lam_star",
    [
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.5, 0.2, 0.15, 0.1, 0.05],
        [0.8, 0.05, 0.05, 0.05, 0.05],
    ],
)
def test_traffic_box_covers_lambda_star(m: int, lam_star) -> None:
    """Pr(lambda* in Lambda_hat) >= 1 - delta_lambda over 3000 reps."""
    delta_lambda = 0.02
    reps = 3000
    lam = np.asarray(lam_star, dtype=float)
    rng = np.random.default_rng(int(1000 * m + 7 * len(lam_star)))
    counts = rng.multinomial(m, lam, size=reps)
    hits = 0
    for row in counts:
        box = traffic_mixture_confidence_box(row, delta_lambda)
        if np.all(lam >= box.raw_lower - 1e-12) and np.all(
            lam <= box.raw_upper + 1e-12
        ):
            hits += 1
    empirical = hits / reps
    # Clopper-Pearson is conservative, so coverage should sit at or above 1-delta.
    # Allow only MC noise below the nominal level.
    assert (
        empirical >= 1.0 - delta_lambda - 0.004
    ), f"coverage {empirical:.4f} below nominal {1 - delta_lambda:.4f} (m={m}, lam*={lam_star})"


def test_traffic_box_is_label_free() -> None:
    """The traffic box consumes client identities only -- never labels or scores."""
    box = traffic_mixture_confidence_box([120, 80, 50, 30, 20], 0.02)
    assert box.total == 300
    assert box.tail_probability == pytest.approx(0.02 / (2 * 5))


def test_traffic_box_zero_counts_gives_full_simplex() -> None:
    """No traffic evidence must not shrink the mixture set."""
    box = traffic_mixture_confidence_box([0, 0, 0], 0.02)
    assert np.allclose(box.raw_lower, 0.0)
    assert np.allclose(box.raw_upper, 1.0)


# --------------------------------------------------------------------------- #
# Theorem 4 budget discipline
# --------------------------------------------------------------------------- #
def test_theorem4_refuses_to_overspend() -> None:
    """An allocation exceeding delta_r must raise, never return a bound."""
    with pytest.raises(ValueError, match="overspends"):
        allocated_risk_certificate([100, 100], [1, 1], [0.09, 0.09], 0.10, alpha=0.10)


def test_theorem4_zero_accepted_client_is_not_dropped() -> None:
    """A_j = 0 makes the cell non-deployable rather than silently vanishing."""
    cert = allocated_risk_certificate([0, 300], [0, 3], [0.05, 0.05], 0.10, alpha=0.10)
    assert cert.rbar[0] == 1.0
    assert cert.U == 1.0


def test_theorem4_uniform_reproduces_theorem1() -> None:
    """Uniform allocation is exactly the Theorem 1 simplex certificate."""
    A = np.array([300, 250, 400, 180, 220])
    K = np.array([9, 8, 10, 7, 30])
    delta_r, J = 0.10, 5
    cert = allocated_risk_certificate(A, K, uniform_allocation(J, delta_r), delta_r)
    expected = max(cp_upper(int(k), int(a), delta_r / J) for a, k in zip(A, K))
    assert cert.U == pytest.approx(expected, abs=1e-12)
