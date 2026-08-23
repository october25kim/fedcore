"""The Fed-ISIC precision addendum gate, and the grouped-sampler escalation tripwire.

Two things are pinned here:

1. ``run_all.py`` reads the SHA-stamped addendum and enumerates 10 tasks x 5 reps = 50
   WP-C cells (170 campaign total), while the sealed ``train_rep in {0,1}`` cells keep
   their cell_ids and CRC32-derived seeds BIT-IDENTICALLY.

2. A TRIPWIRE on the grouped-sampling escalation. ``pi_{j|g}`` (within-group client
   weights) is NOT declared in the sealed prereg, so the declared sampler
   ``fedcore.sampling.sample_group_mixture`` CANNOT be wired without inventing design.
   ``test_pi_j_given_g_tripwire`` FAILS LOUDLY the moment the prereg gains a pi
   declaration -- that failure is the signal to wire the sampler, not a regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_all as R  # noqa: E402


@pytest.fixture(scope="module")
def loaded():
    sealed, sha = R.load_prereg()
    add = R.load_addendum(sha)
    effective = R.apply_addendum(sealed, add)
    return sealed, sha, add, effective


# --------------------------------------------------------------------------- #
# 1. The addendum gate
# --------------------------------------------------------------------------- #
def test_addendum_sha_is_verified_against_its_stamp(loaded):
    _, sha, add, _ = loaded
    assert add.sha256 == R.sha256_file(R.ADDENDUM_PATH)
    assert add.parent_sha256 == sha, "the addendum must name the sealed prereg as parent"


def test_addendum_refuses_on_sha_mismatch(tmp_path, monkeypatch, loaded):
    """A mutated addendum must stop the runner, exactly as a mutated prereg does."""
    _, sha, _, _ = loaded
    tampered = tmp_path / "fedisic_precision_addendum.md"
    tampered.write_text(R.ADDENDUM_PATH.read_text() + "\nsilently appended\n")
    monkeypatch.setattr(R, "ADDENDUM_PATH", tampered)
    with pytest.raises(R.CampaignError, match="ADDENDUM SHA256 MISMATCH"):
        R.load_addendum(sha)


def test_addendum_refuses_when_parent_prereg_differs(loaded):
    with pytest.raises(R.CampaignError, match="PARENT MISMATCH"):
        R.load_addendum("0" * 64)


def test_addendum_declares_10_tasks_x_5_reps_totalling_50(loaded):
    _, _, add, _ = loaded
    assert add.n_tasks == 10
    assert add.train_reps == (0, 1, 2, 3, 4)
    assert add.wp_c_total == 50
    assert add.n_tasks * len(add.train_reps) == add.wp_c_total


def test_confirmatory_and_exploratory_partition_the_reps(loaded):
    _, _, add, _ = loaded
    assert add.confirmatory == (0, 1)
    assert add.exploratory == (2, 3, 4)
    assert not set(add.confirmatory) & set(add.exploratory)
    assert sorted(add.confirmatory + add.exploratory) == sorted(add.train_reps)
    assert [add.role(r) for r in (0, 1)] == ["confirmatory"] * 2
    assert [add.role(r) for r in (2, 3, 4)] == ["exploratory"] * 3


def test_confirmatory_subset_equals_the_sealed_preregs_train_seeds(loaded):
    sealed, _, add, _ = loaded
    sealed_reps = tuple(R.require(sealed, "run_matrix", "WP_C_fedisic", "train_seeds"))
    assert tuple(add.confirmatory) == sealed_reps


# --------------------------------------------------------------------------- #
# 2. Enumeration: 170 = 120 + 50
# --------------------------------------------------------------------------- #
def test_full_matrix_enumerates_170_cells(loaded):
    _, _, add, effective = loaded
    cells = R.enumerate_cells(effective)
    assert len(cells) == 170 == add.campaign_total
    by_pipeline = {}
    for c in cells:
        by_pipeline[c.pipeline] = by_pipeline.get(c.pipeline, 0) + 1
    assert by_pipeline["fed-isic"] == 50
    assert sum(v for k, v in by_pipeline.items() if k != "fed-isic") == 120


def test_sealed_prereg_alone_still_enumerates_140(loaded):
    """The overlay is in memory only: the sealed file is untouched and still says 140."""
    sealed, _, _, _ = loaded
    assert len(R.enumerate_cells(sealed)) == 140
    assert R.require(sealed, "run_matrix", "WP_C_fedisic", "total_runs") == 20


def test_no_fedisic_cell_is_blocked(loaded):
    _, _, _, effective = loaded
    isic = [c for c in R.enumerate_cells(effective) if c.runner == "fedisic"]
    assert len(isic) == 50
    for cell in isic:
        assert R.blocking_reasons(cell, effective) == [], cell.label


# --------------------------------------------------------------------------- #
# 3. The sealed cells must not move -- bit-identical
# --------------------------------------------------------------------------- #
def test_sealed_cell_ids_and_seeds_are_bit_identical(loaded):
    sealed, _, add, effective = loaded
    master = R.master_seed_for(effective, "full")
    proof = R.verify_sealed_cells_bit_identical(sealed, effective, add, master)
    assert proof["cell_ids_bit_identical"]
    assert proof["seed_ledgers_bit_identical"]
    assert proof["sealed_cells_checked"] == 20
    assert proof["seeds_compared"] == 20 * len(R.NAMESPACE_SCOPES)


def test_exploratory_seeds_are_derived_by_the_unchanged_rule(loaded):
    """train_rep 2-4 seeds are CRC32 of the pre-registered payload -- never chosen."""
    _, _, _, effective = loaded
    master = R.master_seed_for(effective, "full")
    assert master == 20260715
    cells = {
        (c.split_id, c.train_seed): c
        for c in R.enumerate_cells(effective)
        if c.runner == "fedisic"
    }
    for rep in (2, 3, 4):
        cell = cells[("split00", rep)]
        entry = R.seed_ledger(cell.fields, master)["fold"]
        expected, payload = R.derive_seed(master, "fold", entry["cell_id"])
        assert entry["seed"] == expected
        assert entry["derivation_input"] == payload == f"20260715:fold:{entry['cell_id']}"


def test_addendum_that_drops_a_sealed_rep_is_refused(loaded):
    sealed, _, add, _ = loaded
    import dataclasses

    dropped = dataclasses.replace(add, confirmatory=(0,), train_reps=(0, 2, 3, 4), wp_c_total=40)
    with pytest.raises(R.CampaignError, match="does not equal the sealed prereg"):
        R.apply_addendum(sealed, dropped)


# --------------------------------------------------------------------------- #
# 4. Run records carry the confirmatory/exploratory role
# --------------------------------------------------------------------------- #
def test_wpc_records_tag_the_train_rep_role(loaded):
    _, sha, add, effective = loaded
    master = R.master_seed_for(effective, "full")
    cells = {
        (c.split_id, c.train_seed): c
        for c in R.enumerate_cells(effective)
        if c.runner == "fedisic"
    }
    sealed_rec = R.base_record(cells[("split00", 0)], effective, "full", master, sha, add)
    new_rec = R.base_record(cells[("split00", 4)], effective, "full", master, sha, add)
    assert sealed_rec["addendum"]["train_rep_role"] == "confirmatory"
    assert new_rec["addendum"]["train_rep_role"] == "exploratory"
    assert new_rec["addendum"]["addendum_sha256"] == add.sha256
    assert "SEPARATELY" in new_rec["addendum"]["reporting_requirement"]


def test_non_wpc_records_carry_no_addendum_block(loaded):
    """The addendum's scope is WP-C only; a CIFAR record must not mention it."""
    _, sha, add, effective = loaded
    master = R.master_seed_for(effective, "full")
    cifar = next(c for c in R.enumerate_cells(effective) if c.runner == "cifar")
    assert "addendum" not in R.base_record(cifar, effective, "full", master, sha, add)


# --------------------------------------------------------------------------- #
# 5. TRIPWIRE -- the grouped-sampler escalation
# --------------------------------------------------------------------------- #
PI_TOKENS = (
    "pi_j_given_g",
    "within_group_weights",
    "client_probabilities_given_group",
    "group_client_weights",
)


def test_pi_j_given_g_tripwire():
    """FAILS LOUDLY once pi_{j|g} is genuinely DECLARED. That failure is the GO signal.

    STATUS: pi is still NOT declared. An inter-agent message asserted an owner decision,
    but an agent message is not the owner's declaration and not user consent -- and the
    owner sealed preregistration/*, requiring a STOP-and-report rather than an
    agent-authored addendum. So this tripwire stays armed.

    It fires on EITHER of the two legitimate declaration sites: the sealed prereg (which
    only the owner may edit) or a stamped owner addendum. When it fires, wire the real pi
    into fedcore/medical/group_mixture.py::spec_for_task -- the machinery, the exact
    sampler draw and the non-contiguous partition map are already implemented and tested
    against a synthetic pi, so only the declaration is missing.
    """
    from fedcore.medical import group_mixture as GM

    sealed = (ROOT / "results" / "preregistration.yaml").read_text()
    declared_in_prereg = [tok for tok in PI_TOKENS if tok in sealed]
    assert not declared_in_prereg, (
        f"The SEALED prereg now declares {declared_in_prereg}. The sealed prereg must "
        "never be edited; pi belongs in a separate stamped declaration."
    )

    # THE TRIPWIRE FIRED on 2026-07-16 and was RESOLVED THE ONLY WAY IT COULD BE: by
    # going to the owner. Its step 1 was "verify the OWNER authored it". That could not be
    # verified from a relay, so the arm was held until
    # RATIFICATION_001_grouped_sampling_declaration.md recorded that the owner was asked
    # the decisive question in isolation -- "did you personally choose straddlers-OUT
    # (BCN = 0.4646, not 0.9149)?" -- shown both outcomes, and affirmed reading B.
    #
    # This test now pins the RESOLVED state: declaration present AND ratified. If the
    # ratification ever disappears while the declaration remains, the arm must go back to
    # being held -- that is what the second assertion enforces.
    assert GM.DECLARATION_PATH.exists()
    ratification = ROOT / "preregistration" / "RATIFICATION_001_grouped_sampling_declaration.md"
    assert ratification.is_file(), (
        "The declaration exists but its RATIFICATION is gone. The production sampler may "
        "not run on an unratified declaration: an inter-agent relay is not consent, and "
        "git authorship cannot distinguish human from agent here."
    )
    text = ratification.read_text()
    # The ratification must name what it ratifies, and must not overclaim.
    assert GM.require_owner_declaration() in text, "ratification does not name the declaration hash"
    assert GM.pi_basis_sha256() in text, "ratification does not name the frozen basis hash"
    assert "does NOT claim" in text and "cryptographic" in text, (
        "the ratification must state its own limits rather than imply a proof it lacks"
    )


def test_ratification_carries_the_weaker_provenance_forward():
    """pi/n_g are ratified, NOT sealed-from-the-start. The two must never be equated."""
    ratification = (
        ROOT / "preregistration" / "RATIFICATION_001_grouped_sampling_declaration.md"
    ).read_text()
    assert "NOT" in ratification and "original sealed" in ratification
    assert "weaker provenance" in ratification.lower()
    # And the sealed prereg still declares the PARTITION only.
    sealed = (ROOT / "results" / "preregistration.yaml").read_text()
    assert "group_partition_G2" in sealed
    assert not [tok for tok in PI_TOKENS if tok in sealed]


def test_sample_group_mixture_has_a_production_caller():
    """The inverse of the original guard: the declared sampler IS now in production.

    Importability was never the question -- invocation was. If this ever returns to
    zero callers, the grouped certificate has silently regressed to relabelling fixed
    quotas and MUST NOT be reported as a grouped-mixture result.
    """
    callers = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith((".git", "tests/", "scripts/fedisic_prelaunch")):
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if "sample_group_mixture(" in line and "def sample_group_mixture(" not in line:
                callers.append(f"{rel}:{line_no}")
    assert callers, (
        "sample_group_mixture has NO production caller. The grouped certificate has "
        "regressed to fixed-quota relabelling; item 7 condition 2 no longer holds."
    )


# --------------------------------------------------------------------------- #
# ADDENDUM_003 (416f5dbe...) -- the grouped certificate variant, and the two
# pre-existing latent defects found while wiring it. Both were LAUNCH-BLOCKING and
# both were invisible to the running campaign, which is exactly why they are pinned.
# --------------------------------------------------------------------------- #
def _run_all():
    import importlib
    import sys

    sys.path.insert(0, str(ROOT))
    return importlib.import_module("run_all")


def test_addendum_003_targets_are_parsed_never_hardcoded():
    """'grouped' must be READ from the stamped chain, exactly as the precision addendum is."""
    ra = _run_all()
    sealed, _ = ra.load_prereg()
    sealed_targets = [str(t) for t in sealed["grids"]["targets"]]
    add3 = ra.load_targets_addendum(sealed_targets)

    assert add3.sha256.startswith("416f5dbe")
    assert "grouped" in add3.targets
    # The sealed values survive the overlay verbatim -- an addendum may ADD, never drop.
    for t in sealed_targets:
        assert t in add3.targets
    # The sealed FILE is never written: on its own it still declares [simplex, box].
    assert "grouped" not in sealed_targets


def test_addendum_003_must_bind_the_ratified_declaration():
    """'grouped' has no content of its own -- it IS the ratified declaration's target."""
    ra = _run_all()
    text = ra.ADDENDUM3_PATH.read_text()
    assert ra.GROUPED_DECLARATION_SHA[:8] in text
    # A prefix is not integrity: the bound file's FULL hash is re-verified on disk.
    decl = ROOT / "preregistration" / "fedisic_grouped_sampling_addendum.md"
    import hashlib

    assert hashlib.sha256(decl.read_bytes()).hexdigest() == ra.GROUPED_DECLARATION_SHA


def test_addendum_003_refuses_to_drop_a_sealed_target():
    ra = _run_all()
    with pytest.raises(ra.CampaignError, match="DROP sealed target"):
        ra.load_targets_addendum(["simplex", "box", "phantom_target"])


def test_grouped_variant_is_reachable_from_the_campaign():
    """Condition 2: certify_cell must actually reach the grouped path."""
    ra = _run_all()
    sealed, _ = ra.load_prereg()
    add3 = ra.load_targets_addendum([str(t) for t in sealed["grids"]["targets"]])
    eff = ra.apply_targets_addendum(sealed, add3)
    ids = [v["id"] for v in ra.variant_grid(eff)]
    assert "grouped" in ids
    # Emitted ONCE, not crossed 4x with knobs the grouped certificate does not accept.
    assert ids.count("grouped") == 1
    grouped = next(v for v in ra.variant_grid(eff) if v["id"] == "grouped")
    assert grouped["allocation_rule"] == "n/a"
    assert grouped["threshold_policy"] == "n/a"
    # And the sealed 2x2x2 design is untouched.
    assert len(ids) == len(sealed["grids"]["allocation_rules"]) * len(
        sealed["grids"]["threshold_policies"]
    ) * len(sealed["grids"]["targets"]) + 1


def test_certify_cell_uses_the_cells_own_n_clients_not_cifars():
    """DEFECT 1 (ADDENDUM_003 section 6). Fed-ISIC must certify at J=6, never CIFAR's J=5.

    At J=5 on 6-centre data, centre 5 (HAM_vienna_dias) drops out of the union bound and
    eps = delta_r/J is divided by the wrong J: an INVALID certificate that still looks
    well-formed. The running CIFAR campaign was unaffected (5 is correct for CIFAR), which
    is precisely why this survived unnoticed.
    """
    ra = _run_all()
    sealed, _ = ra.load_prereg()
    cells = ra.enumerate_cells(sealed)
    isic = next(c for c in cells if c.runner == "fedisic")
    cifar = next(c for c in cells if c.runner != "fedisic")
    assert ra.n_clients_for(isic, sealed) == int(sealed["data"]["fed_isic2019"]["n_clients"]) == 6
    assert ra.n_clients_for(cifar, sealed) == int(sealed["data"]["cifar"]["n_clients"]) == 5


def test_certify_cell_binds_add_and_does_not_raise_nameerror():
    """DEFECT 2. 'add' was referenced in certify_cell but bound NOWHERE -> NameError.

    Verified at runtime, not inferred. It was invisible because the RUNNING dispatcher holds
    an older in-memory copy of run_all and never reloads from disk. Left unfixed, the
    Fed-ISIC launch would have trained all 50 models and produced ZERO certificates -- every
    cell caught and written as CERTFAIL.
    """
    import inspect

    ra = _run_all()
    for fn in (ra.certify_cell, ra.certify_cell_grouped):
        assert "add" in inspect.signature(fn).parameters, (
            f"{fn.__name__} must RECEIVE add; a bare module-level 'add' does not exist"
        )
    assert not hasattr(ra, "add"), "no module-level 'add' -- it must be threaded, not global"
