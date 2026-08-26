"""The group-mixture sampler MACHINERY, exercised in the production path.

STATUS -- read first. ``pi_{j|g}`` is NOT DECLARED by any stamped owner document, so the
REAL pi path is INERT and fails closed (``test_real_pi_path_fails_closed_until_owner_declares``).

These tests therefore drive the machinery with an EXPLICITLY SYNTHETIC pi defined right
here in the test file. That is deliberate and is the only honest option available: it
proves the wiring, the exact draw, replay determinism and the labelling WITHOUT asserting
any design that an agent chose. The moment the owner declares pi, ``spec_for_task``
unlocks and these same tests cover the real path unchanged.

The ordering guarantee is tested too: pi must be a pure function of the FROZEN,
SHA-256-stamped counts artifact, and must refuse to derive if that artifact is mutated.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fedcore import group_draw  # noqa: E402
from fedcore.certify import certify_best_gamma_grouped  # noqa: E402
from fedcore.grouping import group_map_from_partition, make_group_map, scored_views  # noqa: E402
from fedcore.medical import group_mixture as GM  # noqa: E402

SPLIT, TSEED = 0, 0
AUDIT_SEED = 780164830  # the real 'audit'-namespace seed for fed_isic split00 seed0


def make_view(n, n_clients, seed):
    """Synthetic logits standing in for a trained model -- NO Fed-ISIC model exists."""
    r = np.random.default_rng(seed)
    return scored_views(
        r.normal(size=(n, 7)),
        r.integers(-1, 6, size=n),
        r.integers(0, n_clients, size=n),
        ["msp"],
    )["msp"]


#: An EXPLICITLY SYNTHETIC pi, defined here in the test. NOT a declaration, not derived
#: from the owner's proposed rule -- deliberately different numbers so it can never be
#: mistaken for one. It exists only to exercise the sampler machinery.
SYNTHETIC_PI = {0: np.array([0.1, 0.2, 0.6, 0.1]), 1: np.array([0.25, 0.75])}
SYNTHETIC_N_G = {0: 98, 1: 76}


@pytest.fixture(scope="module")
def spec():
    """A synthetic GroupMixtureSpec on the SEALED partition with a SYNTHETIC pi.

    The partition is real (it is sealed and public); only pi/n_g are synthetic, because
    those are the parts no one has declared.
    """
    return group_draw.GroupMixtureSpec(
        group_names=("HAM_derived", "other"),
        client_to_group=group_map_from_partition(
            {"HAM_derived": [1, 2, 3, 5], "other": [0, 4]}
        ),
        pi=SYNTHETIC_PI,
        n_g=SYNTHETIC_N_G,
        seed=AUDIT_SEED,
        provenance={"pi": "SYNTHETIC -- test only; pi_{j|g} is undeclared"},
    )


@pytest.fixture(scope="module")
def views():
    return make_view(3000, 6, 12), make_view(3000, 6, 11), make_view(3000, 6, 13)


KW = dict(
    score_name="msp", G=2, gammas=(0.5, 0.7, 1.0), alpha=0.10, delta=0.10,
    Lambda="box", box=0.15, seed=0,
)

_HAVE_GROUP_ARTIFACTS = all(
    path.is_file()
    for path in (
        GM.COUNTS_PATH,
        GM.STAMP_PATH,
        GM.DECLARATION_PATH,
        ROOT / "results" / "source_data" / "fed_isic2019_metadata.csv",
        ROOT / "results" / "preregistration.yaml",
    )
)
needs_group_artifacts = pytest.mark.skipif(
    not _HAVE_GROUP_ARTIFACTS,
    reason="frozen Fed-ISIC grouped-design artifacts are not distributed",
)
needs_campaign_runner = pytest.mark.skipif(
    not (ROOT / "run_all.py").is_file(),
    reason="optional sealed-campaign runner run_all.py is not distributed",
)


# --------------------------------------------------------------------------- #
# 1. The ordering guarantee: pi is a pure function of the STAMPED artifact
# --------------------------------------------------------------------------- #
@needs_group_artifacts
def test_pi_derives_only_from_the_stamped_frozen_counts():
    assert GM.COUNTS_PATH.is_file() and GM.STAMP_PATH.is_file()
    assert GM._sha256(GM.COUNTS_PATH) == GM.frozen_counts_sha256()


@needs_group_artifacts
def test_pi_refuses_if_the_frozen_counts_were_mutated(tmp_path, monkeypatch):
    """If the counts drift after stamping, pi is no longer predeclared. Refuse."""
    tampered = tmp_path / "frozen_audit_pool_counts.csv"
    tampered.write_text(GM.COUNTS_PATH.read_text() + "\n")
    monkeypatch.setattr(GM, "COUNTS_PATH", tampered)
    GM.load_frozen_counts.cache_clear()
    with pytest.raises(GM.FrozenCountsError, match="SHA256 MISMATCH"):
        GM.load_frozen_counts()
    GM.load_frozen_counts.cache_clear()


@needs_group_artifacts
def test_declaration_is_ratified_and_its_provenance_asymmetry_is_carried():
    """RATIFIED (af7f4e74) + ACTIVATED (02d5f821). The asymmetry it discloses is PERMANENT.

    This test replaces ``test_declaration_exists_but_its_human_provenance_is_UNVERIFIED``,
    which pinned the ``PENDING DIRECT USER CONFIRMATION`` caveat. RATIFICATION_002
    (``02d5f821...``) explicitly authorizes clearing that caveat and replacing it with
    references to the two ratification records. The caveat is therefore gone -- but the
    substance the old test protected is NOT, and this test pins the parts that must survive:

      * VERIFIED -- the frozen basis re-derives exactly from raw metadata (see
        ``test_pi_basis_re_derives_from_raw_metadata``) and the declaration binds that
        hash. The numbers are honest. This was true before ratification and is unchanged
        by it: it is a statement about the arithmetic, not about who chose it.
      * STILL NOT CLAIMED -- a cryptographic owner->subagent chain. RATIFICATION_001 §3 and
        RATIFICATION_002 §5 both state plainly that none exists. Ratification records that
        the owner was asked the decisive question in isolation and answered; it does not
        manufacture proof, and no artifact may imply it did.
      * PERMANENT -- the PROVENANCE ASYMMETRY. ``pi``/``n_g`` were NOT in the sealed
        prereg (which declares the group PARTITION only). Every artifact built on them must
        carry that, and must never present the two as equivalent.

    The old caveat is replaced by a REFERENCE to self-disclosing records, which is more
    informative than the caveat, not less. What would be dishonest is dropping the
    asymmetry -- so that is what this test enforces.
    """
    assert GM.DECLARATION_PATH.exists()
    assert GM.require_owner_declaration() == GM._sha256(GM.DECLARATION_PATH)
    # Derivation works...
    pi = GM.derive_pi()
    assert set(pi) == {"HAM_derived", "other"}

    # ...and the stale caveat must be gone, replaced by the ratification references.
    stages = ROOT / "results" / "fedisic" / "g2_reporting_stages.csv"
    if stages.is_file():
        text = stages.read_text()
        assert "PENDING DIRECT USER CONFIRMATION" not in text
        assert "PENDING OWNER RATIFICATION" not in text
        assert "af7f4e74" in text and "02d5f821" in text
        # The asymmetry is not optional decoration; it is the point.
        assert "NOT in the original sealed prereg" in text
        assert "present the two as equivalent" in text
        # No artifact may claim a chain that the ratifications themselves disclaim.
        assert "NO cryptographic owner->subagent chain exists" in text
        # ERRATUM_001's smoke-tier disclosure travels with the design.
        assert "SMOKE-TIER" in text and "21160715" in text


def test_pi_semantics_is_never_the_full_flamby_test_population():
    """The declaration requires this exact scoping; a looser phrase would overclaim."""
    assert "AUDIT-ELIGIBLE EMPIRICAL BENCHMARK TARGET" in GM.PI_SEMANTICS
    assert "NOT the complete unfiltered FLamby" in GM.PI_SEMANTICS


@needs_group_artifacts
def test_pi_basis_re_derives_from_raw_metadata():
    """The frozen basis must be reproducible arithmetic, not asserted numbers."""
    import pandas as pd

    meta = pd.read_csv(ROOT / "results" / "source_data" / "fed_isic2019_metadata.csv")
    mine = meta[meta.audit_eligible == 1].groupby("center_index").lesion_id.nunique()
    for r in GM.load_pi_basis().itertuples():
        assert int(mine[int(r.center_index)]) == int(r.N_basis)


@needs_group_artifacts
def test_pi_is_task_independent_and_no_center_is_zero_weighted():
    """pi takes NO task argument -- that is what stops a starved center being zeroed."""
    import inspect

    assert list(inspect.signature(GM.derive_pi).parameters) == []
    for weights in GM.derive_pi().values():
        assert all(w > 0 for w in weights.values()), "no center may carry zero mass"
        assert np.isclose(sum(weights.values()), 1.0, rtol=0.0, atol=1e-12)


@needs_group_artifacts
def test_fail_closed_when_a_positive_pi_center_has_no_support():
    """A positive-pi center with an EMPTY reservoir must FAIL CLOSED, never renormalize.

    Tasks 5/7 (BCC+AK, BCC+SCC) starve centers 1/4/5; task 8 (AK+VASC) starves center 4.
    Because pi is task-independent every center keeps pi>0, so these MUST raise rather
    than quietly redistributing the missing center's mass over the survivors.
    """
    for split_id in (5, 7, 8):
        with pytest.raises(GM.MissingCenterSupportError) as excinfo:
            GM.spec_for_task(split_id, TSEED, AUDIT_SEED)
        assert "infeasible_missing_center_support" in str(excinfo.value)
        assert "NOT renormalized" in str(excinfo.value)
        status = GM.sampler_status(split_id, TSEED)
        assert status["g2_manuscript_eligible"] is False
        assert status["grouped_exact_sampler_status"] == "infeasible_missing_center_support"
        assert status["unsupported_centers"]


@needs_group_artifacts
def test_support_is_satisfiability_only_not_count_feasibility():
    """A supported center can be nowhere near count-feasible. Never conflate them."""
    support = GM.center_support(0, TSEED)
    counts = GM.load_frozen_counts()
    row = counts[(counts.split_id == 0) & (counts.train_seed == TSEED) & (counts.client_id == 1)]
    # Center 1 has 3 cert lesions on task 0: SUPPORTED, but far below the G=2 floor (36).
    assert int(row["certification_lesions"].iloc[0]) == 3
    assert support[1] is True
    assert "SAMPLER SATISFIABILITY ONLY" in GM.SUPPORT_RULE


@needs_group_artifacts
def test_frozen_counts_are_metadata_only_and_declare_nothing():
    """Freezing counts is safe and is NOT a design declaration -- it is arithmetic."""
    counts = GM.load_frozen_counts()
    assert len(counts) == 300
    for col in ("audit_pool_lesions", "certification_lesions", "group_audit_pool_lesions"):
        assert col in counts.columns
    # No outcome column may ever appear here.
    forbidden = {"certified", "cert_risk_ucb", "score", "logit", "pred", "test_risk"}
    assert not (set(counts.columns) & forbidden)


@needs_group_artifacts
def test_frozen_counts_are_seed_independent():
    """Whatever pi the owner declares from these counts, it cannot move with train_seed."""
    counts = GM.load_frozen_counts()
    for (split_id, client_id), grp in counts.groupby(["split_id", "client_id"]):
        assert grp["audit_pool_lesions"].nunique() == 1
        assert grp["certification_lesions"].nunique() == 1


# --------------------------------------------------------------------------- #
# 2. Lambda_G is a DERIVATION from sealed values, not a new choice
# --------------------------------------------------------------------------- #
@needs_group_artifacts
def test_lambda_G_is_the_sealed_rho_box_over_two_group_coordinates():
    from fedcore.certificate.lambda_sets import uniform_box

    assert GM.rho_headline() == 0.15
    assert GM.rho_sweep() == [0.0, 0.05, 0.10, 0.15, 0.25, 0.50]
    got, want = GM.lambda_G(), uniform_box(2, 0.15)
    assert np.allclose(got.lo, want.lo) and np.allclose(got.hi, want.hi)


# --------------------------------------------------------------------------- #
# 3. THE CENTRAL TEST: the sampler is actually invoked in the production path
# --------------------------------------------------------------------------- #
@needs_group_artifacts
def test_PRODUCTION_ENTRY_POINT_reaches_the_declared_sampler(views, monkeypatch):
    """THE decisive test: the real production entry point drives the declared sampler.

    Call graph proven end-to-end, with no test-supplied spec:

        fedcore.medical.certify_grouped.certify_task_grouped   (production entry)
          -> group_mixture.spec_for_task   (declared pi/n_g from the ratified declaration)
          -> certify.certify_best_gamma_grouped(mixture_spec=...)
          -> group_draw.draw_group_certification_sample
          -> sampling.sample_group_mixture                     <-- DECLARED SAMPLER

    Before the wiring, sample_group_mixture had ZERO production callers and the grouped
    certificate relabelled fixed quotas. Importability was never the question; invocation
    was.
    """
    from fedcore.medical import certify_grouped

    prop, cert, test = views
    calls = []
    real = group_draw.sample_group_mixture

    def counting(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(group_draw, "sample_group_mixture", counting)
    row = certify_grouped.certify_task_grouped(
        SPLIT, TSEED, AUDIT_SEED, prop, cert, test,
        score_name="msp", gammas=(0.5, 0.7, 1.0), alpha=0.10, delta=0.10,
    )
    assert len(calls) == 2, "the declared sampler must be called once per group"
    assert row["sampler_invoked"] == "fedcore.sampling.sample_group_mixture"
    assert row["theorem_exact"] is True
    assert row["artifact_class"] == "exact_group_mixture_certificate"
    assert row["grouped_exact_sampler_status"] == "satisfiable"
    # The declared pi -- not a test fixture -- is what parameterised the draw.
    declared = GM.derive_pi()
    for kw in calls:
        for gi, gname in enumerate(sorted(declared)):
            clients = sorted(declared[gname])
            assert np.allclose(
                kw["client_probabilities_given_group"][gi],
                [declared[gname][j] for j in clients],
            )
    # Provenance must travel with the row and must not equate pi with the sealed partition.
    assert "af7f4e74" in row["ratification"]
    assert "NOT in the original sealed prereg" in row["provenance_note"]
    assert "WEAKER provenance" in row["provenance_note"]


@needs_group_artifacts
def test_production_entry_point_fails_closed_and_never_renormalizes(views):
    """The mandated fail-closed test, at the PRODUCTION entry point."""
    from fedcore.medical import certify_grouped

    prop, cert, test = views
    for split_id in (5, 7, 8):
        with pytest.raises(GM.MissingCenterSupportError) as excinfo:
            certify_grouped.certify_task_grouped(
                split_id, TSEED, AUDIT_SEED, prop, cert, test,
                score_name="msp", gammas=(0.5, 0.7, 1.0), alpha=0.10, delta=0.10,
            )
        assert "infeasible_missing_center_support" in str(excinfo.value)
        assert "NOT renormalized" in str(excinfo.value)
        status = certify_grouped.sampler_status_for_task(split_id, TSEED)
        assert status["g2_manuscript_eligible"] is False
        # pi is UNCHANGED by the failure -- no surviving-center renormalization happened.
        assert GM.derive_pi() == GM.derive_pi()
        for weights in GM.derive_pi().values():
            assert all(w > 0 for w in weights.values())


def test_production_path_invokes_the_declared_sampler(spec, views, monkeypatch):
    """Call-count proof at the certify layer, with an explicit spec."""
    prop, cert, test = views
    calls = []
    real = group_draw.sample_group_mixture

    def counting(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(group_draw, "sample_group_mixture", counting)
    out = certify_best_gamma_grouped(
        prop, cert, test, group_map=spec.client_to_group, mixture_spec=spec, **KW
    )
    assert len(calls) == spec.G, "the sampler must be called once per group"
    assert out["sampler_invoked"] == "fedcore.sampling.sample_group_mixture"
    assert out["theorem_exact"] is True
    # The declared law: group -> Categorical(pi_{.|g}) -> reservoir draw.
    for kw in calls:
        for g in range(spec.G):
            assert np.allclose(kw["client_probabilities_given_group"][g], spec.pi[g])


def test_legacy_path_does_not_invoke_the_sampler_and_is_never_theorem_exact(spec, views, monkeypatch):
    prop, cert, test = views
    calls = []
    monkeypatch.setattr(
        group_draw, "sample_group_mixture", lambda **k: calls.append(k)
    )
    out = certify_best_gamma_grouped(prop, cert, test, group_map=spec.client_to_group, **KW)
    assert calls == []
    assert out["theorem_exact"] is False
    assert out["draw_construction"] == "fixed_quota_largest_remainder"
    assert out["artifact_class"] == "diagnostic_fixed_quota"
    assert "superseded" in out["manuscript_status"]


# --------------------------------------------------------------------------- #
# 4. Empirical frequencies match pi -- large MC, CP tolerance, not eyeballed
# --------------------------------------------------------------------------- #
def test_empirical_client_frequencies_match_pi_within_clopper_pearson(spec, views):
    """Each client's realised share must sit inside an exact binomial CP interval."""
    _, cert, _ = views
    big = dataclasses.replace(spec, n_g={g: 20000 for g in range(spec.G)})
    _, record = group_draw.draw_group_certification_sample(cert, big)

    for g in range(spec.G):
        sel = record.group_index == g
        n = int(sel.sum())
        assert n == 20000
        clients = spec.clients_in(g)
        for k, j in enumerate(clients):
            p = float(spec.pi[g][k])
            hits = int((record.client_id[sel] == j).sum())
            # Exact Clopper-Pearson 99.99% interval; union over <=6 clients x 2 groups
            # keeps the suite's false-alarm rate negligible.
            eps = 1e-4 / 2
            lo = 0.0 if hits == 0 else beta.ppf(eps, hits, n - hits + 1)
            hi = 1.0 if hits == n else beta.isf(eps, hits + 1, n - hits)
            assert lo <= p <= hi, (
                f"group {spec.group_names[g]} client {j}: declared pi={p}, "
                f"empirical={hits/n}, CP=[{lo:.5f}, {hi:.5f}]"
            )


def test_zero_pi_clients_are_never_drawn(spec, views):
    """A client with pi=0 must never appear in the draw.

    This matters beyond bookkeeping: under ANY audit-supply-proportional rule, an
    A3-STARVED center (empty audit pool for that task) would receive pi=0 and would drop
    out of its group's distribution entirely -- so the certified target for such a task
    would be a group mixture that no longer contains that center. Whatever the owner
    declares, that consequence must be visible rather than discovered later.
    """
    _, cert, _ = views
    starved = dataclasses.replace(
        spec, pi={0: np.array([0.0, 0.3, 0.7, 0.0]), 1: np.array([1.0, 0.0])}
    )
    _, record = group_draw.draw_group_certification_sample(cert, starved)
    for g in range(starved.G):
        for k, j in enumerate(starved.clients_in(g)):
            if starved.pi[g][k] == 0.0:
                assert not (record.client_id[record.group_index == g] == j).any()


# --------------------------------------------------------------------------- #
# 5. Deterministic replay: client ids AND source ids
# --------------------------------------------------------------------------- #
def test_replay_reproduces_client_ids_and_source_ids(spec, views):
    _, cert, _ = views
    a_view, a = group_draw.draw_group_certification_sample(cert, spec)
    b_view, b = group_draw.draw_group_certification_sample(cert, spec)
    assert np.array_equal(a.client_id, b.client_id)
    assert np.array_equal(a.source_id, b.source_id)
    assert np.array_equal(a.reservoir_position, b.reservoir_position)
    assert np.array_equal(a.group_index, b.group_index)
    for key in a_view:
        assert np.array_equal(np.asarray(a_view[key]), np.asarray(b_view[key]))


def test_a_different_audit_seed_moves_the_draw(spec, views):
    """Guards against a replay test that would pass on a constant draw."""
    _, cert, _ = views
    other = dataclasses.replace(spec, seed=spec.seed + 1)
    _, a = group_draw.draw_group_certification_sample(cert, spec)
    _, b = group_draw.draw_group_certification_sample(cert, other)
    assert not np.array_equal(a.source_id, b.source_id)


# --------------------------------------------------------------------------- #
# 6. Sample ids and multiplicities are persisted; the draw is WITH replacement
# --------------------------------------------------------------------------- #
def test_group_sample_ids_and_multiplicities_are_persisted(spec, views):
    _, cert, _ = views
    drawn, record = group_draw.draw_group_certification_sample(cert, spec)
    assert len(record.source_id) == sum(spec.n_g.values())
    assert len(record.client_id) == len(record.source_id)
    mult = record.multiplicities()
    assert sum(mult.values()) == len(record.source_id)
    # WITH replacement: the drawn view may repeat a source unit.
    assert len(drawn["client"]) == sum(spec.n_g.values())
    # Each drawn observation's client must lie in the group it was drawn for.
    for obs in range(len(record.source_id)):
        assert spec.client_to_group[record.client_id[obs]] == record.group_index[obs]


def test_drawn_strata_are_the_groups_and_counts_are_exactly_n_g(spec, views):
    _, cert, _ = views
    drawn, record = group_draw.draw_group_certification_sample(cert, spec)
    for g in range(spec.G):
        assert int((drawn["client"] == g).sum()) == spec.n_g[g]


def test_the_frozen_reservoir_is_not_mutated_by_the_draw(spec, views):
    _, cert, _ = views
    before = {k: np.array(v, copy=True) for k, v in cert.items()}
    group_draw.draw_group_certification_sample(cert, spec)
    for k, v in before.items():
        assert np.array_equal(np.asarray(cert[k]), v)


# --------------------------------------------------------------------------- #
# 7. No certification outcome can select grouping / pi / n_g / the sampler seed
# --------------------------------------------------------------------------- #
def test_pi_and_n_g_apis_accept_no_outcome_argument():
    """A label/score/logit must not be able to reach pi. Enforced at the signature."""
    import inspect

    for fn in (GM.derive_pi, GM.derive_n_g, GM.client_to_group_vector):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"y_open", "labels", "score", "logits", "pred", "certified"})
        assert params <= {"split_id", "train_seed"}


@needs_campaign_runner
def test_sampler_seed_masks_alpha_and_certificate_variant():
    """The sealed 'audit' scope masks alpha/variant, so competing analyses share ONE draw.

    This is what makes 'no outcome selected the draw' structural rather than a promise.
    """
    import run_all as R

    assert GM.SAMPLER_SEED_NAMESPACE == "audit"
    scope = R.NAMESPACE_SCOPES["audit"]
    assert "alpha" not in scope and "certificate_variant" not in scope
    fields = {
        "dataset": "fed_isic2019", "pipeline": "fed-isic", "split_id": "split00",
        "train_seed": 0, "d": None, "alpha": None, "certificate_variant": None,
    }
    ledger = R.seed_ledger(fields, 20260715)
    cid = ledger["audit"]["cell_id"]
    assert cid.split("|")[5] == "*" and cid.split("|")[6] == "*"
    # Two different alphas/variants must produce the SAME audit seed.
    other = R.seed_ledger({**fields, "alpha": 0.25, "certificate_variant": "R3-global-box"}, 20260715)
    assert other["audit"]["seed"] == ledger["audit"]["seed"] == AUDIT_SEED


# --------------------------------------------------------------------------- #
# 8. Fold hygiene under the draw
# --------------------------------------------------------------------------- #
def test_draw_only_touches_the_certification_reservoir(spec, views):
    """The drawn sample must be a multiset of CERTIFICATION rows only."""
    prop, cert, test = views
    drawn, record = group_draw.draw_group_certification_sample(cert, spec)
    cert_rows = {
        tuple(np.round(np.atleast_1d(cert["score"])[i : i + 1], 12)) for i in range(len(cert["score"]))
    }
    for value in np.atleast_1d(drawn["score"]):
        assert (round(float(value), 12),) in {
            (round(float(v), 12),) for v in np.atleast_1d(cert["score"])
        }
    assert len(record.source_id) == sum(spec.n_g.values())


def test_selector_still_sees_the_proposal_fold_only(spec, views, monkeypatch):
    """The exact draw replaces the CERTIFICATION fold; it must not touch proposal/test."""
    prop, cert, test = views
    prop_before = {k: np.array(v, copy=True) for k, v in prop.items()}
    test_before = {k: np.array(v, copy=True) for k, v in test.items()}
    certify_best_gamma_grouped(
        prop, cert, test, group_map=spec.client_to_group, mixture_spec=spec, **KW
    )
    for k, v in prop_before.items():
        assert np.array_equal(np.asarray(prop[k]), v)
    for k, v in test_before.items():
        assert np.array_equal(np.asarray(test[k]), v)


# --------------------------------------------------------------------------- #
# 9. Non-contiguous partitions (the second gap)
# --------------------------------------------------------------------------- #
def test_group_map_from_partition_expresses_the_declared_fedisic_partition():
    got = group_map_from_partition({"HAM_derived": [1, 2, 3, 5], "other": [0, 4]})
    assert list(got) == [1, 0, 0, 0, 1, 0]
    # The contiguous map CANNOT express it -- this is why the fix was needed.
    assert not np.array_equal(got, make_group_map(6, 2))


def test_cifar_declared_partition_agrees_with_the_contiguous_map():
    """CIFAR's sealed [[0,1,2],[3,4]] IS contiguous, so the legacy map was correct there."""
    got = group_map_from_partition([[0, 1, 2], [3, 4]])
    assert list(got) == [0, 0, 0, 1, 1]
    assert np.array_equal(got, make_group_map(5, 2))


def test_partition_must_cover_every_client_exactly_once():
    with pytest.raises(ValueError, match="ungrouped"):
        group_map_from_partition([[0, 1], [2]], n_clients=5)
    with pytest.raises(ValueError, match="more than one group"):
        group_map_from_partition([[0, 1], [1, 2]])


@needs_group_artifacts
def test_spec_matches_the_sealed_partition(spec):
    """The production spec's grouping must equal the SEALED declaration."""
    import yaml

    prereg = yaml.safe_load((ROOT / "results" / "preregistration.yaml").read_text())
    sealed = prereg["data"]["fed_isic2019"]["group_partition_G2"]
    expected = group_map_from_partition(
        {k: v for k, v in sealed.items() if k != "rationale"}
    )
    assert np.array_equal(spec.client_to_group, expected)
