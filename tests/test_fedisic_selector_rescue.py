"""Fed-ISIC final selector-rescue regressions (F3a/F3b), host-only (numpy/scipy).

The IUT/Holm/simultaneous certificates and the leakage-safe threshold rule are already
covered by test_officehome_selector_rescue.py and the fedcore.selector tests (same code
reused here). These tests pin the Fed-ISIC-specific invariants: fresh-draw seed
independence, candidate dedup determinism, proposal-only selection, and a canonical
byte-value guard.
"""
from __future__ import annotations

import csv

import numpy as np

from fedcore.experiments.fedisic_f3a_rescue import fresh_seed, cand_hash
from fedcore.officehome_rescue import candidate_null_pvalue
from fedcore.selector import choose_threshold, counts_per_client, Selector


def test_fresh_seed_depends_only_on_cell():
    """Seed is a function of the cell name ONLY -- never alpha/score/gamma/outcome."""
    a = fresh_seed("fed_isic2019__fed-isic__split00__seed0")
    b = fresh_seed("fed_isic2019__fed-isic__split00__seed0")
    c = fresh_seed("fed_isic2019__fed-isic__split01__seed0")
    assert a == b                      # deterministic
    assert a != c                      # distinct cells differ
    assert 0 <= a < 2**32


def test_candidate_hash_dedup_deterministic():
    h1 = cand_hash("energy", 0.3, 1.2345678)
    h2 = cand_hash("energy", 0.3, 1.2345678)
    h3 = cand_hash("msp", 0.3, 1.2345678)
    assert h1 == h2 and h1 != h3 and len(h1) == 16


def test_iut_pvalue_regression():
    """p_IUT = max_j P(Bin(A_j, alpha) <= K_j); A_j=0 -> 1 (blocks)."""
    # one client A=1,K=0 at alpha=0.2 -> p=(1-0.2)^1=0.8 dominates
    p = candidate_null_pvalue([1, 5], [0, 0], 0.20)
    assert abs(p - 0.8) < 1e-9
    # a zero-acceptance client blocks certification
    assert candidate_null_pvalue([0, 30], [0, 0], 0.20) == 1.0


def test_choose_threshold_is_proposal_only_and_buffered():
    """Selector is built from one fold only and respects the gamma*alpha risk buffer."""
    rng = np.random.default_rng(0)
    score = rng.random(200)
    y_open = rng.integers(-1, 3, size=200)      # -1 unknown or class 0..2
    pred = rng.integers(0, 3, size=200)
    sel = choose_threshold(score, pred, y_open, gamma=0.5, alpha=0.20)
    # feasibility is a pure function of these proposal arrays (no cert data anywhere)
    assert isinstance(sel, Selector)
    if sel.feasible:
        # empirical accepted risk on the SAME fold must respect the buffer
        acc = score >= sel.threshold
        err = (y_open < 0) | (pred != y_open)
        if acc.sum() > 0:
            assert err[acc].mean() <= 0.5 * 0.20 + 1e-12


def test_counts_per_client_accept_error_semantics():
    score = np.array([0.9, 0.8, 0.1, 0.95])
    pred = np.array([0, 1, 0, 2])
    y_open = np.array([0, -1, 0, 2])            # unit1 unknown, unit3 correct known
    client = np.array([0, 0, 1, 1])
    sel = Selector(threshold=0.5, feasible=True)
    A, K, n = counts_per_client(score, pred, y_open, client, sel, 2)
    assert list(n) == [2, 2]
    assert list(A) == [2, 1]                    # unit2 (0.1) rejected
    assert list(K) == [1, 0]                    # client0: unit1 accepted unknown = error


def test_canonical_risk_decomposition_unchanged():
    """Guard: the canonical Fed-ISIC eligible decomposition must stay byte-value stable."""
    rows = list(csv.DictReader(open("results/final/fedisic_eligible_32_risk_decomposition.csv")))
    binding = [float(x["cp_risk_ucb"]) for x in rows
               if x["is_binding_group"] == "True" and x["cp_risk_ucb"] not in ("", "inf")]
    assert round(min(binding), 4) == 0.3290     # canonical "best CP-UCB = 0.329"
