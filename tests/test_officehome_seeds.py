"""Semantic seed-registry tests for the Office-Home arm.

Seeds must be a deterministic function of immutable identity (split / train_rep /
pipeline / frozen hashes) and INDEPENDENT of alpha/delta/rho/policy/score/solver.
"""

from __future__ import annotations

import os

import pytest

from fedcore.data.officehome import OfficeHomeDataConfig, load_officehome_job
from fedcore.experiments.run_officehome import build_seed_bundle
from fedcore.seeds import ForbiddenSeedContextError, SeedBundle, derive_seed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "results/officehome/dedup/retained_canonical_manifest.csv")
CLASS_SPLITS = os.path.join(REPO, "results/officehome/preflight/class_splits.csv")
IMAGE_ROOT = os.path.join(REPO, "data/officehome/OfficeHomeDataset")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(MANIFEST), reason="frozen Office-Home manifest absent"
)


def _job(split_id="officehome_split_0"):
    cfg = OfficeHomeDataConfig(
        manifest_csv=MANIFEST,
        folds_csv=os.path.join(REPO, f"results/officehome/folds/folds_{split_id}.csv"),
        class_splits_csv=CLASS_SPLITS,
        split_id=split_id,
        image_root=IMAGE_ROOT,
    )
    return load_officehome_job(cfg, check_image_files=False)


def test_seed_bundle_is_deterministic_from_identity():
    job = _job()
    a = build_seed_bundle(job, pipeline_name="convnext_tiny_full_fedavg", campaign_seed=0, train_rep=0)
    b = build_seed_bundle(job, pipeline_name="convnext_tiny_full_fedavg", campaign_seed=0, train_rep=0)
    assert a.to_json() == b.to_json()
    # Replay validation: the ledger recomputes every seed.
    assert SeedBundle.from_json(a.to_json()).to_json() == a.to_json()


def test_pipeline_and_rep_change_training_seeds():
    job = _job()
    base = build_seed_bundle(job, pipeline_name="convnext_tiny_full_fedavg", campaign_seed=0, train_rep=0)
    other_pipe = build_seed_bundle(job, pipeline_name="convnext_tiny_frozen_linear", campaign_seed=0, train_rep=0)
    other_rep = build_seed_bundle(job, pipeline_name="convnext_tiny_full_fedavg", campaign_seed=0, train_rep=1)
    assert base.model_init != other_pipe.model_init
    assert base.model_init != other_rep.model_init
    assert base.audit_draw != other_pipe.audit_draw
    assert base.audit_draw != other_rep.audit_draw


def test_split_changes_class_split_seed():
    s0 = build_seed_bundle(_job("officehome_split_0"), pipeline_name="convnext_tiny_full_fedavg", campaign_seed=0, train_rep=0)
    s1 = build_seed_bundle(_job("officehome_split_1"), pipeline_name="convnext_tiny_full_fedavg", campaign_seed=0, train_rep=0)
    assert s0.class_split != s1.class_split
    assert s0.model_init != s1.model_init


def test_audit_and_traffic_draws_reject_posthoc_knobs():
    # Draw-namespace contexts may not depend on analysis knobs.
    for knob in ("alpha", "delta", "rho", "score_name", "solver", "certificate_variant"):
        with pytest.raises(ForbiddenSeedContextError):
            derive_seed(0, "audit_draw", {"experiment_id": "cell", knob: 0.2})
        with pytest.raises(ForbiddenSeedContextError):
            derive_seed(0, "traffic_draw", {"experiment_id": "cell", knob: 0.2})


def test_all_namespaces_present_and_replayable():
    bundle = build_seed_bundle(_job(), pipeline_name="convnext_tiny_full_fedavg", campaign_seed=0, train_rep=0)
    # Full ledger with every namespace present (SeedBundle enforces this).
    seeds = bundle.seeds
    for ns in ("class_split", "partition", "fold", "model_init", "loader",
               "label_noise", "audit_draw", "traffic_draw", "solver", "stability"):
        assert ns in seeds and 0 <= seeds[ns] <= (1 << 32) - 1
