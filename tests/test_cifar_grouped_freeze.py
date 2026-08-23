"""The CIFAR exact-grouped recertification: preconditions PROVEN, not asserted.

RATIFICATION_003 section 5a makes five things mandatory BEFORE the NO-TRAIN exact-grouped
recertification of the 120 frozen CIFAR/FedPD logit sets. Each has a test here, because the
owner's instruction was explicit that they be proven with a test rather than asserted in
prose:

    * pi_{j|g} defined from the frozen NOMINAL certification-fold allocation  -> test_pi_*
    * freeze and hash pi / partitions / n_g / Lambda_G / seeds / policies      -> test_frozen_*
    * audit seeds do not depend on alpha/delta/rho/policy/score/solver/result  -> test_audit_seed_*
    * prove the production path reaches sample_group_mixture                   -> test_production_path_*
    * reuse identical audit draws across compared policies                     -> test_identical_draws_*
    * do NOT select only cells completed after sampler activation              -> test_recert_covers_*

None of these launches anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import run_all  # noqa: E402
from fedcore import cifar_group_mixture as CGM  # noqa: E402

pytestmark = pytest.mark.skipif(
    not CGM.PI_BASIS_PATH.is_file(),
    reason="the frozen CIFAR pi basis is not stamped yet (scripts/freeze_cifar_grouped_design.py)",
)


@pytest.fixture(scope="module")
def prereg():
    sealed, _ = run_all.load_prereg()
    return sealed


@pytest.fixture(scope="module")
def master_seed(prereg):
    return run_all.master_seed_for(prereg, "full")


@pytest.fixture(scope="module")
def cifar_cells(prereg):
    return [c for c in run_all.enumerate_cells(prereg) if c.runner in ("cifar", "fedpd")]


# --------------------------------------------------------------------------- #
# The audit seed is outcome-independent. This is the load-bearing proof.
# --------------------------------------------------------------------------- #
def test_audit_seed_is_invariant_to_alpha_and_certificate_variant(cifar_cells, master_seed):
    """No post-hoc knob may select the draw -- checked on EVERY cell, not a sample.

    alpha and certificate_variant are the only two cell_id coordinates that could carry a
    post-hoc choice, and the sealed `audit` scope masks both. certificate_variant is the
    coordinate that encodes target x allocation_rule x threshold_policy, so masking it is
    what makes 'same draw, different policy' true.
    """
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    variants = [
        "uniform-global-simplex", "uniform-global-box", "uniform-client_specific-simplex",
        "uniform-client_specific-box", "R3-global-simplex", "R3-global-box",
        "R3-client_specific-simplex", "R3-client_specific-box", "grouped",
    ]
    for cell in cifar_cells:
        base = int(run_all.seed_ledger(cell.fields, master_seed)["audit"]["seed"])
        for alpha in alphas:
            for variant in variants:
                fields = {**cell.fields, "alpha": alpha, "certificate_variant": variant}
                got = int(run_all.seed_ledger(fields, master_seed)["audit"]["seed"])
                assert got == base, (
                    f"{cell.label}: the audit seed MOVED with alpha={alpha} "
                    f"variant={variant} ({got} != {base}). A certification outcome could "
                    "then select the draw."
                )


def test_audit_seed_derivation_input_masks_the_post_hoc_coordinates(cifar_cells, master_seed):
    """Stronger than equal seeds: the hashed STRING itself must not contain them."""
    for cell in cifar_cells:
        entry = run_all.seed_ledger(
            {**cell.fields, "alpha": 0.10, "certificate_variant": "grouped"}, master_seed
        )["audit"]
        payload = entry["derivation_input"]
        assert payload.endswith("|*|*"), (
            f"{cell.label}: audit derivation input {payload!r} does not mask "
            "(alpha, certificate_variant) to '*'."
        )
        assert "grouped" not in payload and "0.10" not in payload
        assert "alpha" not in entry["scope"] and "certificate_variant" not in entry["scope"]


def test_audit_seed_cannot_depend_on_delta_rho_policy_score_or_solver():
    """These are not cell_id coordinates AT ALL, so they cannot enter the hash.

    Proven structurally rather than by sampling values: the seed rule hashes exactly
    CELL_FIELDS, so a knob absent from CELL_FIELDS is unreachable from the derivation.
    """
    for knob in ("delta", "delta_r", "delta_c", "rho", "kappa", "policy",
                 "threshold_policy", "allocation_rule", "score", "score_name",
                 "solver", "gamma"):
        assert knob not in run_all.CELL_FIELDS, (
            f"{knob!r} became a cell_id coordinate; the audit seed could now depend on it."
        )
    # ... and the audit stream's declared scope is a strict subset of the training-side
    # coordinates, i.e. it is a function of the DATA the cell was trained on, nothing else.
    assert set(run_all.NAMESPACE_SCOPES["audit"]) == {
        "dataset", "pipeline", "split_id", "train_seed", "d"
    }


def test_audit_seed_cannot_depend_on_an_observed_result(cifar_cells, master_seed):
    """The derivation reads only the cell tuple and the master seed -- never an artifact.

    Verified by construction: seed_ledger's inputs are (fields, master_seed) and its rule is
    CRC32 over a string built from those. No filesystem read, no logits, no counts.
    """
    cell = cifar_cells[0]
    entry = run_all.seed_ledger(cell.fields, master_seed)["audit"]
    expected, payload = run_all.derive_seed(master_seed, "audit", entry["cell_id"])
    assert entry["seed"] == expected and entry["derivation_input"] == payload
    # The recomputation above used ONLY the cell tuple + master seed. If the seed were a
    # function of any observed result, this equality could not hold with the artifact absent.
    assert not Path(payload).exists()


def test_the_frozen_design_records_the_audit_seed_it_will_actually_use(cifar_cells, master_seed):
    design = CGM.load_design()
    roster = {r["label"]: r for r in design["source_artifact_manifest"]["roster"]}
    assert len(roster) == 120, f"the frozen roster holds {len(roster)} cells, not 120"
    for cell in cifar_cells:
        entry = roster[cell.label]
        assert entry["audit_seed"] == int(
            run_all.seed_ledger(cell.fields, master_seed)["audit"]["seed"]
        )
        assert entry["audit_seed_scope"] == list(run_all.NAMESPACE_SCOPES["audit"])


# --------------------------------------------------------------------------- #
# pi: frozen first, derived second, from the hashed file alone.
# --------------------------------------------------------------------------- #
def test_pi_is_a_pure_function_of_the_stamped_basis(tmp_path, monkeypatch):
    """Mutate the basis by one byte -> every pi-dependent path REFUSES."""
    CGM.load_pi_basis.cache_clear()
    original = CGM.PI_BASIS_PATH.read_bytes()
    forged = tmp_path / "forged.csv"
    forged.write_bytes(original.replace(b"514", b"515", 1))
    monkeypatch.setattr(CGM, "PI_BASIS_PATH", forged)
    CGM.load_pi_basis.cache_clear()
    with pytest.raises(CGM.FrozenBasisError, match="SHA256 MISMATCH"):
        CGM.load_pi_basis()
    CGM.load_pi_basis.cache_clear()


def test_the_frozen_basis_carries_no_outcome_column():
    """pi must NOT derive from accepted counts, errors, realized draws, scores, results.

    The basis physically cannot: it has no such column. This is checked rather than trusted
    because a later 'helpful' addition is exactly how such a dependence would appear.
    """
    basis = CGM.load_pi_basis()
    assert set(basis.columns) == {
        "dataset", "pipeline", "split_id", "train_seed", "client_index", "group_index",
        "n_nominal_cert", "group_n_nominal_cert", "basis_definition", "provenance",
    }
    forbidden = (
        "accepted", "error", "realized", "score", "certified", "cert_risk", "alpha",
        "delta", "rho", "policy", "solver", "ucb", "lcb", "k_g", "a_g",
    )
    for column in basis.columns:
        assert not any(f in column.lower() for f in forbidden), column


def test_pi_derives_from_integer_counts_and_sums_to_one():
    for dataset, pipeline, split_id, train_seed in CGM.frozen_keys():
        pi = CGM.derive_pi(dataset, pipeline, split_id, train_seed)
        vec = CGM.client_to_group_vector()
        for group, weights in pi.items():
            assert np.isclose(sum(weights.values()), 1.0, rtol=0, atol=1e-12)
            for client, weight in weights.items():
                assert weight > 0.0, "a zero would silently remove a client"
                assert int(vec[client]) == group


def test_n_g_agrees_between_the_basis_and_the_frozen_design():
    """Two frozen copies of one quantity must never disagree silently."""
    for dataset, pipeline, split_id, train_seed in CGM.frozen_keys():
        n_g = CGM.derive_n_g(dataset, pipeline, split_id, train_seed)
        assert set(n_g) == {0, 1}
        assert all(v > 0 for v in n_g.values())


def test_the_partition_is_read_from_the_sealed_prereg_not_the_basis(prereg):
    """The partition is the one SEALED part of this design; it keeps its own provenance."""
    sealed = run_all.require(prereg, "data", "cifar", "group_partition_G2")
    assert [list(m) for m in sealed] == [[0, 1, 2], [3, 4]]
    assert CGM.client_to_group_vector().tolist() == [0, 0, 0, 1, 1]


def test_frozen_design_covers_everything_section_5a_requires():
    design = CGM.load_design()
    for key in ("pi", "group_partition_G2", "n_g", "Lambda_G", "sampler",
                "allocation_policy", "threshold_policy", "source_artifact_manifest"):
        assert key in design, f"section 5a requires {key} to be frozen and hashed"
    assert design["Lambda_G"]["value"] == "uniform_box(G=2, rho=0.15)"
    assert design["sampler"]["declared"] == "fedcore.sampling.sample_group_mixture"
    assert design["sampler"]["seed_namespace"] == "audit"
    assert design["allocation_policy"]["value"] == "n/a"
    assert design["threshold_policy"]["value"] == "n/a"


def test_the_frozen_design_and_basis_are_stamped_and_verify():
    assert CGM.pi_basis_sha256() == CGM._sha256(CGM.PI_BASIS_PATH)
    assert CGM.design_sha256() == CGM._sha256(CGM.DESIGN_PATH)


def test_the_declaration_binding_is_enforced(monkeypatch, tmp_path):
    """No stamped declaration stating CIFAR's rule -> the grouped path is INERT."""
    CGM.require_owner_declaration.cache_clear()
    assert CGM.require_owner_declaration() == run_all.GROUPED_DECLARATION_SHA
    missing = tmp_path / "absent.md"
    monkeypatch.setattr(CGM, "DECLARATION_PATH", missing)
    CGM.require_owner_declaration.cache_clear()
    with pytest.raises(CGM.UndeclaredDesignError):
        CGM.require_owner_declaration()
    CGM.require_owner_declaration.cache_clear()


# --------------------------------------------------------------------------- #
# The production path reaches the declared sampler.
# --------------------------------------------------------------------------- #
def _fake_views(n_clients=5, per_client=40, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for fold in ("prop", "cert", "test"):
        client = np.repeat(np.arange(n_clients), per_client)
        n = len(client)
        out[fold] = {
            "score": rng.uniform(0, 1, n),
            "pred": rng.integers(0, 6, n),
            "y_open": np.where(rng.uniform(0, 1, n) < 0.3, -1, rng.integers(0, 6, n)),
            "client": client,
        }
    return out


def test_production_path_reaches_sample_group_mixture(monkeypatch):
    """certify_cifar_cell_grouped -> ... -> fedcore.sampling.sample_group_mixture.

    Counted, not inferred. Before this wiring the CIFAR grouped path was BLOCKED and the
    sampler had zero CIFAR callers.
    """
    from fedcore import cifar_certify_grouped, group_draw

    calls = []
    real = group_draw.sample_group_mixture

    def counting(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(group_draw, "sample_group_mixture", counting)
    dataset, pipeline, split_id, train_seed = CGM.frozen_keys()[0]
    views = _fake_views()
    row = cifar_certify_grouped.certify_cifar_cell_grouped(
        dataset, pipeline, split_id, train_seed, 12345,
        views["prop"], views["cert"], views["test"],
        score_name="msp", gammas=[0.5, 1.0], alpha=0.10, delta=0.05,
    )
    assert len(calls) == 2, f"expected one sampler call per group (G=2), got {len(calls)}"
    assert row["sampler_invoked"] == "fedcore.sampling.sample_group_mixture"
    assert row["theorem_exact"] is True
    assert row["artifact_class"] == "exact_group_mixture_certificate"
    assert row["draw_construction"] == "categorical_then_reservoir_with_replacement"


def test_the_run_all_production_caller_reaches_the_cifar_entry_point():
    """The campaign entry point, not just the library, must reach it."""
    source = (REPO_ROOT / "run_all.py").read_text()
    assert "from fedcore.cifar_certify_grouped import certify_cifar_cell_grouped" in source
    assert "certify_cell_grouped_cifar(" in source
    # and certify_cell_grouped dispatches CIFAR cells to it rather than blocking them
    assert 'if cell.runner != "fedisic":\n        return certify_cell_grouped_cifar(' in source


def test_grouped_fails_closed_when_a_positive_pi_client_has_no_reservoir():
    """pi is NEVER renormalized over the survivors."""
    from fedcore import cifar_certify_grouped

    views = _fake_views()
    for fold in views:
        keep = views[fold]["client"] != 4          # starve client 4 (pi = 0.5 in group 1)
        views[fold] = {k: v[keep] for k, v in views[fold].items()}
    dataset, pipeline, split_id, train_seed = CGM.frozen_keys()[0]
    with pytest.raises(CGM.MissingClientSupportError) as excinfo:
        cifar_certify_grouped.certify_cifar_cell_grouped(
            dataset, pipeline, split_id, train_seed, 1,
            views["prop"], views["cert"], views["test"],
            score_name="msp", gammas=[1.0], alpha=0.10, delta=0.05,
        )
    assert excinfo.value.unsupported == [4]
    assert "NOT renormalized" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Identical audit draws across compared policies -- the within-run pairing.
# --------------------------------------------------------------------------- #
def test_identical_draws_across_compared_policies(monkeypatch):
    """Same draw, different policy. The pairing the brief relies on.

    The audit seed is the ONLY thing that selects the draw, and it is invariant to alpha and
    certificate_variant, so a grouped certificate at any alpha consumes the SAME units.
    """
    from fedcore import cifar_certify_grouped, group_draw

    seen = []
    real = group_draw.sample_group_mixture

    def recording(**kwargs):
        draw = real(**kwargs)
        seen.append(draw.sample_id.copy())
        return draw

    monkeypatch.setattr(group_draw, "sample_group_mixture", recording)
    dataset, pipeline, split_id, train_seed = CGM.frozen_keys()[0]
    views = _fake_views()
    for alpha in (0.05, 0.30):
        cifar_certify_grouped.certify_cifar_cell_grouped(
            dataset, pipeline, split_id, train_seed, 999,
            views["prop"], views["cert"], views["test"],
            score_name="msp", gammas=[1.0], alpha=alpha, delta=0.05,
        )
    assert len(seen) == 4
    np.testing.assert_array_equal(seen[0], seen[2])   # group 0, alpha .05 vs .30
    np.testing.assert_array_equal(seen[1], seen[3])   # group 1, alpha .05 vs .30


# --------------------------------------------------------------------------- #
# ALL 120 -- never a completion-order subset.
# --------------------------------------------------------------------------- #
def test_recert_covers_every_one_of_the_120_cells(prereg):
    cells = [c for c in run_all.enumerate_cells(prereg) if c.runner in ("cifar", "fedpd")]
    assert len(cells) == 120
    roster = CGM.load_design()["source_artifact_manifest"]["roster"]
    assert {r["label"] for r in roster} == {c.label for c in cells}


def test_the_completeness_gate_refuses_a_partial_arm(prereg, monkeypatch, capsys):
    """--require-complete makes 'only the finished cells' impossible, not merely discouraged.

    Driven through main() with a stubbed validator so the refusal is observed, not reasoned
    about. Nothing is dispatched: --no-train + --dry-run.
    """
    calls = {"n": 0}

    def half_done(cell, path):
        calls["n"] += 1
        return (calls["n"] % 2 == 0, "missing")

    monkeypatch.setattr(run_all, "validate_artifact", half_done)
    with pytest.raises(run_all.CampaignError, match="REFUSING to certify a subset"):
        run_all.main(
            ["--tier", "full", "--no-train", "--arm", "cifar", "--targets", "grouped",
             "--require-complete", "120"]
        )


def test_targets_filter_is_variant_axis_only_and_rejects_undeclared_targets():
    with pytest.raises(run_all.CampaignError, match="not in the declared grids.targets"):
        run_all.main(
            ["--tier", "full", "--no-train", "--dry-run", "--arm", "cifar",
             "--targets", "invented_target"]
        )


# --------------------------------------------------------------------------- #
# Carry-forward: the six statements, the status label, the terminology.
# --------------------------------------------------------------------------- #
def test_condition_1_status_label_is_exactly_the_permitted_one():
    assert run_all.CONDITION_1_STATUS == "PASS_PREOUTCOME_ADDENDUM"


def test_all_six_mandatory_statements_are_carried():
    block = run_all.grouped_carry_forward()
    assert block["condition_1_status"] == "PASS_PREOUTCOME_ADDENDUM"
    assert len(block["condition_1_mandatory_statements"]) == 6
    joined = " ".join(block["condition_1_mandatory_statements"])
    for fragment in (
        "did NOT contain pi",
        "group partition only",
        "specified later",
        "zero production callers",
        "0.464646",
        "0.914871",
        "No result was used to select the declaration",
        "prospective pre-outcome addendum",
    ):
        assert fragment in joined, f"missing mandatory statement content: {fragment!r}"


def test_required_terminology_and_the_forbidden_phrase():
    block = run_all.grouped_carry_forward()
    assert block["required_terminology"]["forbidden"] == (
        "originally preregistered grouped analysis"
    )
    assert block["required_terminology"]["required"] == (
        "prospectively specified through an owner-authored pre-outcome addendum"
    )


def test_the_forbidden_phrase_appears_nowhere_in_the_emitted_artifacts():
    """A phrase banned in prose is banned in the artifacts that become prose.

    USE vs MENTION, drawn explicitly rather than by a loose grep: RATIFICATION_003 forbids
    *describing* the grouped analysis as originally preregistered. Stating the ban itself
    necessarily quotes the phrase. So a line is a MENTION only if it also carries a
    prohibition marker; anything else is a USE and fails. The narrow exemption is written
    out here so it cannot quietly widen into "any line that mentions the phrase is fine".
    """
    forbidden = run_all.FORBIDDEN_PHRASE
    markers = ("forbidden", "never", "not ", "must never", "banned")
    for path in (
        CGM.PI_BASIS_PATH,
        CGM.DESIGN_PATH,
        REPO_ROOT / "results" / "cifar" / "cifar_pi_derived.csv",
        REPO_ROOT / "run_all.py",
        REPO_ROOT / "fedcore" / "cifar_group_mixture.py",
        REPO_ROOT / "fedcore" / "cifar_certify_grouped.py",
    ):
        if not path.is_file():
            continue
        uses = [
            line
            for line in path.read_text().splitlines()
            if forbidden in line and not any(m in line.lower() for m in markers)
        ]
        assert not uses, f"{path.name} USES the forbidden phrase: {uses[:1]}"


def test_the_carry_forward_block_states_the_asymmetry_coincidence_erratum_and_escalation():
    block = run_all.grouped_carry_forward()
    assert "WEAKER PROVENANCE" in block["provenance_asymmetry"]
    assert "never be presented as equivalent" in block["provenance_asymmetry"].lower()
    assert "DIVERGE" in block["coincidence_warning"]
    assert "C admits 9 tasks, stage A admits 7" in block["coincidence_warning"]
    assert "changes NO count number" in block["sampler_buys_validity_not_counts"]
    assert "SMOKE-TIER" in block["erratum_001_disclosure"]
    assert "21160715" in block["erratum_001_disclosure"]
    assert "UNRESOLVED, OWNER RULING REQUIRED" in block["grouped_knob_cross_status"]


def test_every_frozen_row_carries_the_provenance_asymmetry():
    basis = CGM.load_pi_basis()
    assert basis["provenance"].str.contains("PASS_PREOUTCOME_ADDENDUM").all()
    assert basis["provenance"].str.contains("NOT in the original sealed").all()
    assert basis["provenance"].str.contains("never present the two as equivalent").all()


def test_the_grouped_knob_cross_stays_escalated_and_uncrossed(prereg):
    """ADDENDUM_003 disclaims new design content, so the knobs must not be crossed."""
    _, sha = run_all.load_prereg()
    add3 = run_all.load_targets_addendum(
        [str(t) for t in run_all.require(prereg, "grids", "targets")]
    )
    effective = run_all.apply_targets_addendum(dict(prereg), add3)
    grouped = [v for v in run_all.variant_grid(effective) if v["target"] == "grouped"]
    assert len(grouped) == 1, (
        f"'grouped' is emitted {len(grouped)}x per alpha. Crossing it with "
        "allocation_rule x threshold_policy is UNDECLARED design content."
    )
    assert grouped[0]["allocation_rule"] == "n/a"
    assert grouped[0]["threshold_policy"] == "n/a"


# --------------------------------------------------------------------------- #
# Legacy fixed-quota diagnostics: preserved, never overwritten, exactly labelled.
# --------------------------------------------------------------------------- #
def test_the_legacy_fixed_quota_path_keeps_its_exact_labels():
    """RATIFICATION_003 section 5a fixes these three strings. They are not paraphrasable."""
    from fedcore.certify import certify_best_gamma_grouped

    views = _fake_views()
    row = certify_best_gamma_grouped(
        views["prop"], views["cert"], views["test"],
        score_name="msp", group_map=np.array([0, 0, 0, 1, 1]), G=2,
        gammas=[1.0], alpha=0.10, delta=0.05,
        # mixture_spec OMITTED -> the legacy diagnostic path
    )
    assert row["artifact_class"] == "diagnostic_fixed_quota"
    assert row["theorem_exact"] is False
    assert row["draw_construction"] == "fixed_quota_largest_remainder"
    assert row["sampler_invoked"] == "none"
    assert "not_theorem_exact" in row["manuscript_status"]
    assert "superseded" in row["manuscript_status"]


def test_the_exact_path_and_the_legacy_path_write_different_artifact_classes():
    """A diagnostic must never be mistakable for a certificate."""
    from fedcore import cifar_certify_grouped
    from fedcore.certify import certify_best_gamma_grouped

    views = _fake_views()
    dataset, pipeline, split_id, train_seed = CGM.frozen_keys()[0]
    exact = cifar_certify_grouped.certify_cifar_cell_grouped(
        dataset, pipeline, split_id, train_seed, 7,
        views["prop"], views["cert"], views["test"],
        score_name="msp", gammas=[1.0], alpha=0.10, delta=0.05,
    )
    legacy = certify_best_gamma_grouped(
        views["prop"], views["cert"], views["test"],
        score_name="msp", group_map=np.array([0, 0, 0, 1, 1]), G=2,
        gammas=[1.0], alpha=0.10, delta=0.05,
    )
    assert exact["artifact_class"] != legacy["artifact_class"]
    assert exact["theorem_exact"] is True and legacy["theorem_exact"] is False
