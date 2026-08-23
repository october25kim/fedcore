"""50-cell Office-Home schedule: exactly 50 unique jobs, GPU policy, no launch."""

from __future__ import annotations

import os

import pytest

from fedcore.experiments.officehome_schedule import (
    ALLOWED_GPUS,
    PIPELINE_KEYS,
    SPLIT_IDS,
    TRAIN_REPS,
    build_command,
    enumerate_cells,
    scheduler_dry_run,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "results/officehome/dedup/retained_canonical_manifest.csv")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(MANIFEST), reason="frozen Office-Home manifest absent"
)


def _kwargs():
    return dict(
        manifest_csv=MANIFEST,
        folds_template=os.path.join(REPO, "results/officehome/folds/folds_{split_id}.csv"),
        class_splits_csv=os.path.join(REPO, "results/officehome/preflight/class_splits.csv"),
        image_root=os.path.join(REPO, "data/officehome/OfficeHomeDataset"),
    )


def test_exactly_50_unique_cells():
    report = scheduler_dry_run(**_kwargs())
    assert report["n_cells"] == 50
    assert report["exactly_50_unique_jobs"]
    assert report["unique_experiment_ids"] == 50
    assert report["unique_cell_triples"] == 50


def test_matrix_is_the_full_5x5x2_cross_product():
    cells = enumerate_cells(**_kwargs())
    triples = {(c.split_id, c.train_rep, c.pipeline) for c in cells}
    expected = {
        (s, r, p) for s in SPLIT_IDS for r in TRAIN_REPS for p in PIPELINE_KEYS
    }
    assert triples == expected
    assert len(expected) == 50


def test_gpu_policy_excludes_gpu0():
    report = scheduler_dry_run(**_kwargs())
    assert report["gpu_excluded"] == 0
    assert set(report["allowed_gpus"]) == {1, 2, 3}
    assert report["cuda_device_order"] == "PCI_BUS_ID"
    for entry in report["commands"]:
        assert entry["gpu"] in ALLOWED_GPUS
        assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in entry["command"]
        assert "run_officehome" in entry["command"]


def test_command_pins_the_cell_experiment_id():
    cells = enumerate_cells(**_kwargs())
    cmd = build_command(cells[0], 1)
    assert cells[0].experiment_id in cmd
    assert cells[0].split_id in cmd
