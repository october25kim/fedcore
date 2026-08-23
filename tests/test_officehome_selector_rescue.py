"""Tests for the Office-Home selector-rescue follow-up (Phases S1/S2/S3).

These enforce the finite-sample validity of the two family theorems, the frozen
candidate-family provenance, the fresh-draw independence, and -- critically --
that the ORIGINAL PRIMARY output stays byte-identical.  No packages are
installed; only numpy/scipy/pytest already in the environment are used.

The Monte-Carlo validity tests use fixed seeds and conservative tolerances so
they are deterministic and non-flaky.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np
import pytest
from scipy.stats import binom as _binom

from fedcore.certificate.cp import cp_lower, cp_upper
from fedcore.officehome_selector import select_from_artifact
from fedcore.scores import compute_score
from fedcore.selector import choose_threshold, empirical_risk_coverage, open_set_error
from fedcore.officehome_rescue import (
    FAMILY_GAMMAS,
    FAMILY_NGRID,
    FAMILY_SCORES,
    M_CANDIDATES,
    CandidateKey,
    candidate_null_pvalue,
    family_keys,
    fresh_draw_seed,
    holm_family_certificate,
    holm_step_down_reject,
    select_family_candidate,
    selector_definition_hash,
    simulate_conditional_counts,
    simultaneous_family_certificate,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGITS_DIR = os.path.join(REPO, "runs", "oneshot", "officehome", "logits")
AUDIT_CSV = os.path.join(REPO, "results", "officehome", "selector_rescue",
                         "candidate_family_audit.csv")
RESCUE_DIR = os.path.join(REPO, "results", "officehome", "selector_rescue")
PRIMARY_CSV = os.path.join(REPO, "results", "officehome", "final_cell_results.csv")

# The frozen sha256 of the ORIGINAL PRIMARY output. This tier of the analysis
# must never touch it; if this hash changes the primary was overwritten.
PRIMARY_SHA256 = "b2ae8339e44aada430ec0cc6a7fab2096aaf8591d82dfcae4967e5272add3c02"

_HAVE_ARTIFACTS = os.path.isdir(LOGITS_DIR) and any(
    f.endswith("_logits.npz") for f in os.listdir(LOGITS_DIR)
) if os.path.isdir(LOGITS_DIR) else False
_HAVE_RESCUE = os.path.isfile(os.path.join(RESCUE_DIR, "simultaneous_family_results.csv"))

needs_artifacts = pytest.mark.skipif(not _HAVE_ARTIFACTS, reason="frozen logits absent")
needs_rescue = pytest.mark.skipif(not _HAVE_RESCUE, reason="rescue outputs not generated")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cell(name: str) -> dict:
    path = os.path.join(LOGITS_DIR, name + "_logits.npz")
    with np.load(path, allow_pickle=True) as arch:
        return {k: np.array(arch[k], copy=True) for k in arch.files}


def _cell_names() -> list:
    return sorted(
        f[: -len("_logits.npz")]
        for f in os.listdir(LOGITS_DIR)
        if f.endswith("_logits.npz")
    )


# --------------------------------------------------------------------------- #
# 1. Selector-hash round trip
# --------------------------------------------------------------------------- #
def test_selector_hash_deterministic_and_injective() -> None:
    keys = family_keys(0.20)
    assert len(keys) == M_CANDIDATES == 12
    hashes = [k.selector_hash for k in keys]
    # deterministic: recomputation matches
    for k in keys:
        assert k.selector_hash == selector_definition_hash(k.score_name, k.gamma, k.alpha)
    # injective across the 12 candidates
    assert len(set(hashes)) == len(hashes)
    # 16 lowercase hex chars
    for h in hashes:
        assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)


def test_selector_hash_depends_on_definition_not_cell() -> None:
    # alpha changes the hash; the same (score,gamma,alpha) is stable.
    a = selector_definition_hash("msp", 0.3, 0.20)
    b = selector_definition_hash("msp", 0.3, 0.25)
    assert a != b
    assert selector_definition_hash("msp", 0.3, 0.20) == a


# --------------------------------------------------------------------------- #
# 2. Candidate-family freeze / provenance (recompute == frozen audit)
# --------------------------------------------------------------------------- #
@needs_artifacts
def test_candidate_family_matches_frozen_audit() -> None:
    if not os.path.isfile(AUDIT_CSV):
        pytest.skip("frozen candidate audit absent")
    rows = list(csv.DictReader(open(AUDIT_CSV)))
    assert len(rows) == 3000  # 50 cells x 5 alpha x 12 candidates
    assert all(r["proposal_only"] == "True" for r in rows)
    assert all(r["frozen_before_cert"] == "True" for r in rows)
    by_cell: dict = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    assert len(by_cell) == 50
    for cell, rs in by_cell.items():
        arr = _load_cell(cell)
        logits = arr["prop_logits"]
        y = arr["prop_y_open"]
        cl = arr["prop_client"]
        pred = logits.argmax(-1)
        err = open_set_error(pred, y)
        sv_cache = {s: compute_score(s, logits) for s in FAMILY_SCORES}
        for r in rs:
            sv = sv_cache[r["score_family"]]
            sel = choose_threshold(sv, pred, y, float(r["gamma"]), float(r["alpha"]),
                                   n_grid=FAMILY_NGRID)
            cov, risk = empirical_risk_coverage(sv, err, sel.threshold)
            acc = sv >= sel.threshold
            A = [int(((cl == j) & acc).sum()) for j in range(4)]
            K = [int(((cl == j) & acc & err).sum()) for j in range(4)]
            # threshold reproduces to machine precision; counts exactly; cov/risk
            # to the audit's 4-decimal rounding; feasibility = threshold feasible.
            assert abs(sel.threshold - float(r["threshold"])) < 1e-12
            assert A == json.loads(r["prop_A"])
            assert K == json.loads(r["prop_K"])
            assert round(cov, 4) == round(float(r["proposal_coverage"]), 4)
            assert round(risk, 4) == round(float(r["proposal_risk"]), 4)
            assert str(bool(sel.feasible)) == r["proposal_feasible"]


# --------------------------------------------------------------------------- #
# 3. Cert/eval-label monkeypatch cannot alter candidate construction
# --------------------------------------------------------------------------- #
@needs_artifacts
def test_cert_eval_labels_do_not_touch_selector() -> None:
    name = _cell_names()[0]
    arr = _load_cell(name)
    _, cands0 = select_from_artifact(arr, alpha=0.20, scores=FAMILY_SCORES, gammas=FAMILY_GAMMAS)
    thr0 = [c.threshold for c in cands0]
    # Corrupt every certification / evaluation array; the selector must not move.
    poisoned = dict(arr)
    for key in list(poisoned):
        if key.startswith("cert_") or key.startswith("eval_"):
            v = poisoned[key]
            if np.issubdtype(np.asarray(v).dtype, np.number):
                poisoned[key] = np.asarray(v) * 0 - 999
    _, cands1 = select_from_artifact(poisoned, alpha=0.20, scores=FAMILY_SCORES, gammas=FAMILY_GAMMAS)
    thr1 = [c.threshold for c in cands1]
    assert thr0 == thr1


# --------------------------------------------------------------------------- #
# 4. Simultaneous-family theorem Monte-Carlo validity
# --------------------------------------------------------------------------- #
def test_simultaneous_family_simultaneous_coverage() -> None:
    """Family fixed independent of cert -> simultaneous coverage >= 1-dr-dc."""
    rng = np.random.default_rng(20260720)
    M, J = 4, 3
    delta_r = delta_c = 0.10
    n = np.array([400, 350, 500])
    # adversarial-ish truths: some candidates near the alpha boundary.
    true_r = np.array([
        [0.05, 0.08, 0.19],
        [0.19, 0.02, 0.10],
        [0.12, 0.15, 0.09],
        [0.03, 0.19, 0.19],
    ])
    true_a = np.array([
        [0.5, 0.6, 0.4],
        [0.3, 0.7, 0.5],
        [0.6, 0.4, 0.5],
        [0.4, 0.5, 0.6],
    ])
    reps = 4000
    simultaneous_hits = 0
    false_certified = 0
    alpha = 0.20
    keys = family_keys(alpha)[:M]  # arbitrary fixed keys for the tie-break
    for _ in range(reps):
        A, K = simulate_conditional_counts(true_a, true_r, n, rng)
        fc = simultaneous_family_certificate(A, K, n, alpha=alpha, delta_r=delta_r, delta_c=delta_c)
        risk_ok = np.all(fc.rbar >= true_r - 1e-12)
        cov_ok = np.all(fc.alow <= true_a + 1e-12)
        if risk_ok and cov_ok:
            simultaneous_hits += 1
        # selection-level: certifying a candidate whose true worst-client risk > alpha
        sel = select_family_candidate(keys, fc.certified, fc.C)
        if sel is not None and float(np.max(true_r[sel])) > alpha + 1e-12:
            false_certified += 1
    coverage = simultaneous_hits / reps
    # The theorem: simultaneous coverage of all M*J risk + M*J coverage events is
    # at least 1 - delta_r - delta_c.
    assert coverage >= 1.0 - delta_r - delta_c, f"simultaneous coverage {coverage:.4f}"
    # Non-trivial: with truths placed at the CP-tight boundary, some events really
    # do fail, so the bounds are not vacuously [0, 1].
    assert coverage < 1.0
    # Selection-level guarantee: certifying a candidate whose true worst-client
    # risk exceeds alpha happens at most delta_r + delta_c of the time.
    assert false_certified / reps <= delta_r + delta_c


def test_simultaneous_certificate_uses_multiplicity_budget() -> None:
    A = np.full((M_CANDIDATES, 4), 200)
    K = np.zeros((M_CANDIDATES, 4), dtype=int)
    n = np.full(4, 500)
    fc = simultaneous_family_certificate(A, K, n, alpha=0.20, delta_r=0.05, delta_c=0.05)
    assert fc.eps_r == pytest.approx(0.05 / (M_CANDIDATES * 4))
    assert fc.eps_c == pytest.approx(0.05 / (M_CANDIDATES * 4))
    # a candidate certified under multiplicity has a strictly larger rbar than the
    # same counts at the single-selector budget delta/J (the multiplicity price).
    rbar_mult = cp_upper(0, 200, 0.05 / (M_CANDIDATES * 4))
    rbar_single = cp_upper(0, 200, 0.05 / 4)
    assert rbar_mult > rbar_single


# --------------------------------------------------------------------------- #
# 5. Holm candidate-null p-value validity + FWER control
# --------------------------------------------------------------------------- #
def test_candidate_null_pvalue_is_valid_under_boundary_null() -> None:
    """Under r=alpha, P(p_jm <= u) <= u (super-uniform / conservative)."""
    rng = np.random.default_rng(7)
    alpha = 0.20
    A = 300
    reps = 20000
    K = rng.binomial(A, alpha, size=reps)
    p = _binom.cdf(K, A, alpha)
    for u in (0.05, 0.10, 0.20):
        assert (p <= u).mean() <= u + 0.01


def test_holm_family_controls_fwer() -> None:
    """All candidates true nulls (worst client r=alpha) -> P(reject any) <= delta_r."""
    rng = np.random.default_rng(101)
    M, J = 6, 3
    delta_r = delta_c = 0.10
    alpha = 0.20
    n = np.array([400, 400, 400])
    # each candidate has at least one client exactly at the boundary r=alpha.
    true_r = np.full((M, J), 0.05)
    true_r[:, 0] = alpha
    true_a = np.full((M, J), 0.5)
    reps = 4000
    any_reject = 0
    for _ in range(reps):
        A, K = simulate_conditional_counts(true_a, true_r, n, rng)
        holm = holm_family_certificate(A, K, n, alpha=alpha, delta_r=delta_r, delta_c=delta_c)
        if holm.holm_reject.any():
            any_reject += 1
    fwer = any_reject / reps
    assert fwer <= delta_r + 0.02, f"empirical FWER {fwer:.4f}"


def test_holm_step_down_matches_manual() -> None:
    p = np.array([0.001, 0.02, 0.5, 0.009])
    fwer = 0.05
    rej = holm_step_down_reject(p, fwer)
    # sorted: 0.001(thr .0125), 0.009(thr .0167), 0.02(thr .025), 0.5(thr .05)
    # reject 0.001, 0.009, 0.02; stop at 0.5.
    assert rej.tolist() == [True, True, False, True]


# --------------------------------------------------------------------------- #
# 6. Family-selection deterministic replay + tie-break
# --------------------------------------------------------------------------- #
def test_family_selection_deterministic_replay() -> None:
    keys = family_keys(0.20)
    certified = [False] * 12
    C = [0.0] * 12
    # make several candidates certified with an exact C tie to force the tie-break
    for idx in (11, 3, 7):
        certified[idx] = True
        C[idx] = 0.1234
    s1 = select_family_candidate(keys, certified, C)
    s2 = select_family_candidate(keys, certified, C)
    assert s1 == s2
    # tie-break prefers smaller gamma, then lexicographic score, then hash.
    chosen = keys[s1]
    tied = [keys[i] for i in (11, 3, 7)]
    best = min(tied, key=lambda k: k.tiebreak_key())
    assert chosen.tiebreak_key() == best.tiebreak_key()


def test_family_selection_prefers_larger_coverage() -> None:
    keys = family_keys(0.20)
    certified = [False] * 12
    C = [0.0] * 12
    certified[2] = certified[9] = True
    C[2] = 0.10
    C[9] = 0.20
    assert select_family_candidate(keys, certified, C) == 9


# --------------------------------------------------------------------------- #
# 7. Fresh-draw independence + seed rule
# --------------------------------------------------------------------------- #
def test_fresh_draw_seed_is_deterministic_and_cell_only() -> None:
    s1 = fresh_draw_seed("officehome__A__split0__rep0")
    s2 = fresh_draw_seed("officehome__A__split0__rep0")
    assert s1 == s2
    assert fresh_draw_seed("officehome__A__split0__rep1") != s1
    assert 0 <= s1 < 2**32


@needs_artifacts
def test_fresh_draw_seed_distinct_from_frozen_streams() -> None:
    for name in _cell_names():
        arr = _load_cell(name)
        seed = fresh_draw_seed(name)
        assert seed != int(arr["audit_draw_seed"])
        assert seed != int(arr["traffic_draw_seed"])


@needs_artifacts
def test_fresh_draw_positions_stay_in_reservoir() -> None:
    name = _cell_names()[0]
    arr = _load_cell(name)
    cert_client = arr["cert_client"]
    seed = fresh_draw_seed(name)
    rng = np.random.default_rng(seed)
    for j in range(4):
        res = np.where(cert_client == j)[0]
        pos = res[rng.integers(0, res.size, size=500)]
        assert np.all(cert_client[pos] == j)  # never crosses domains


# --------------------------------------------------------------------------- #
# 8. ORIGINAL PRIMARY byte-identical
# --------------------------------------------------------------------------- #
def test_primary_final_cell_results_byte_identical() -> None:
    assert os.path.isfile(PRIMARY_CSV)
    assert _sha256(PRIMARY_CSV) == PRIMARY_SHA256


# --------------------------------------------------------------------------- #
# 9-10. No cell omission; no alpha treated as an independent model
# --------------------------------------------------------------------------- #
@needs_rescue
def test_no_cell_omission_and_all_alphas_reported() -> None:
    for fname in ("simultaneous_family_results.csv", "holm_family_results.csv",
                  "failure_transition_matrix.csv"):
        rows = list(csv.DictReader(open(os.path.join(RESCUE_DIR, fname))))
        cells = sorted({r["cell"] for r in rows})
        alphas = sorted({r["alpha"] for r in rows})
        assert len(cells) == 50, fname
        assert set(alphas) == {"0.1", "0.15", "0.2", "0.25", "0.3"}, fname
        # every cell reports every alpha (correlated endpoints, not independent).
        for cell in cells:
            got = sorted({r["alpha"] for r in rows if r["cell"] == cell})
            assert got == ["0.1", "0.15", "0.2", "0.25", "0.3"], (fname, cell)


# --------------------------------------------------------------------------- #
# 11. Canonical Theorem-2 budget regression + secondary 0/50 unchanged
# --------------------------------------------------------------------------- #
@needs_rescue
def test_canonical_theorem2_budget_and_secondary_conclusion() -> None:
    J = 4
    lam = list(csv.DictReader(open(os.path.join(RESCUE_DIR, "canonical_budget_lambda_results.csv"))))
    rho = list(csv.DictReader(open(os.path.join(RESCUE_DIR, "canonical_budget_rho_results.csv"))))
    diff = list(csv.DictReader(open(os.path.join(RESCUE_DIR, "budget_convention_diff.csv"))))
    # budget convention pinned exactly
    for r in lam + rho:
        assert abs(float(r["eps_r_per_client"]) - 0.04 / (3.0 * J)) < 1e-15
        assert abs(float(r["eps_c_per_client"]) - 0.04 / J) < 1e-15
    # secondary conclusion unchanged: traffic (m=1000) + all rho remain 0/50.
    assert sum(r["canonical_certified"] == "True" for r in lam if r["primary_m"] == "True") == 0
    assert sum(r["canonical_certified"] == "True" for r in rho) == 0
    assert sum(r["original_certified"] == "True" for r in lam if r["primary_m"] == "True") == 0
    # canonical is at least as conservative -> deploy decision never flips on.
    assert sum(r["certified_changed"] == "True" for r in diff) == 0


# --------------------------------------------------------------------------- #
# 12. Reservoir / overlap accounting
# --------------------------------------------------------------------------- #
@needs_rescue
def test_fresh_draw_reservoir_accounting_is_sound() -> None:
    rows = list(csv.DictReader(open(os.path.join(RESCUE_DIR, "fresh_draw_reservoir_accounting.csv"))))
    assert len(rows) == 50 * 4
    for r in rows:
        assert r["all_positions_in_domain_reservoir"] == "True"
        assert r["distinct_from_audit_seed"] == "True"
        assert int(r["fold_overlap_total"]) == 0
        assert 0.0 <= float(r["duplication_rate"]) < 1.0
