"""Deterministic 50-cell Office-Home training schedule (enumeration + dry-run).

The matrix is a paired block over ``(class_split_id, train_rep)`` crossed with
the two pipelines: ``5 splits x 5 reps x 2 pipelines = 50`` immutable neural
training cells. This module ENUMERATES the cells and resolves each one's semantic
``experiment_id`` / ``training_config_sha256`` via the same torch-free
``run_officehome.prepare_job`` binding that a real run would use -- so a dry-run
proves exactly 50 unique jobs before any GPU time is spent.

It never launches training. ``build_command`` renders the exact module invocation
a launcher would ``setsid``-detach; GPU assignment round-robins over the allowed
physical GPUs (1, 2, 3) with ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` and GPU 0 excluded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

from fedcore.data.officehome import OfficeHomeDataConfig, load_officehome_job
from fedcore.experiments.run_officehome import (
    PIPELINES,
    build_seed_bundle,
    build_training_config,
)
from fedcore.campaign.artifacts import semantic_experiment_id, semantic_hash


SPLIT_IDS: tuple[str, ...] = tuple(f"officehome_split_{i}" for i in range(5))
TRAIN_REPS: tuple[int, ...] = tuple(range(5))
PIPELINE_KEYS: tuple[str, ...] = ("A", "B")
ALLOWED_GPUS: tuple[int, ...] = (1, 2, 3)

DEFAULT_MANIFEST = "results/officehome/dedup/retained_canonical_manifest.csv"
DEFAULT_FOLDS_TEMPLATE = "results/officehome/folds/folds_{split_id}.csv"
DEFAULT_CLASS_SPLITS = "results/officehome/preflight/class_splits.csv"
DEFAULT_IMAGE_ROOT = "data/officehome/OfficeHomeDataset"


@dataclass(frozen=True)
class Cell:
    """One immutable neural-training cell of the 50-cell matrix."""

    split_id: str
    train_rep: int
    pipeline: str
    pipeline_name: str
    experiment_id: str
    training_config_sha256: str
    manifest_sha256: str
    folds_sha256: str
    class_splits_sha256: str

    def as_dict(self) -> dict:
        return {
            "split_id": self.split_id,
            "train_rep": self.train_rep,
            "pipeline": self.pipeline,
            "pipeline_name": self.pipeline_name,
            "experiment_id": self.experiment_id,
            "training_config_sha256": self.training_config_sha256,
        }


def _folds_path(template: str, split_id: str) -> str:
    return template.format(split_id=split_id)


def _resolve_cell(job, split_id: str, train_rep: int, pipeline_key: str) -> Cell:
    pipeline = PIPELINES[pipeline_key]
    seed_bundle = build_seed_bundle(
        job,
        pipeline_name=pipeline["name"],
        campaign_seed=0,
        train_rep=train_rep,
    )
    args = SimpleNamespace(
        lr=None,
        pretrained=True,
        rounds=30,
        local_epochs=1,
        batch_size=32,
        image_size=224,
        weight_decay=0.05,
        warmup_rounds=2,
        train_rep=train_rep,
        campaign_seed=0,
        device="auto",
    )
    training_config = build_training_config(args, job, pipeline_key, seed_bundle)
    training_config_sha256 = semantic_hash(training_config)
    prefix = f"officehome-{split_id}-rep{train_rep}-{pipeline_key}"
    experiment_id = semantic_experiment_id(prefix, training_config)
    return Cell(
        split_id=split_id,
        train_rep=train_rep,
        pipeline=pipeline_key,
        pipeline_name=pipeline["name"],
        experiment_id=experiment_id,
        training_config_sha256=training_config_sha256,
        manifest_sha256=job.manifest_sha256,
        folds_sha256=job.folds_sha256,
        class_splits_sha256=job.class_splits_sha256,
    )


def enumerate_cells(
    *,
    manifest_csv: str = DEFAULT_MANIFEST,
    folds_template: str = DEFAULT_FOLDS_TEMPLATE,
    class_splits_csv: str = DEFAULT_CLASS_SPLITS,
    image_root: str = DEFAULT_IMAGE_ROOT,
    splits: Sequence[str] = SPLIT_IDS,
    train_reps: Sequence[int] = TRAIN_REPS,
    pipelines: Sequence[str] = PIPELINE_KEYS,
) -> list[Cell]:
    """Return all cells; one torch-free job load per split (folds validated)."""

    cells: list[Cell] = []
    for split_id in splits:
        config = OfficeHomeDataConfig(
            manifest_csv=manifest_csv,
            folds_csv=_folds_path(folds_template, split_id),
            class_splits_csv=class_splits_csv,
            image_root=image_root,
            split_id=split_id,
        )
        job = load_officehome_job(config, check_image_files=False)
        for train_rep in train_reps:
            for pipeline_key in pipelines:
                cells.append(_resolve_cell(job, split_id, train_rep, pipeline_key))
    return cells


def build_command(
    cell: Cell,
    gpu: int,
    *,
    manifest_csv: str = DEFAULT_MANIFEST,
    folds_template: str = DEFAULT_FOLDS_TEMPLATE,
    class_splits_csv: str = DEFAULT_CLASS_SPLITS,
    image_root: str = DEFAULT_IMAGE_ROOT,
) -> str:
    """Render the exact (never executed here) training invocation for one cell."""

    return (
        f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={gpu} "
        "python -m fedcore.experiments.run_officehome "
        f"--manifest-csv {manifest_csv} "
        f"--folds-csv {_folds_path(folds_template, cell.split_id)} "
        f"--class-splits-csv {class_splits_csv} "
        f"--split-id {cell.split_id} --image-root {image_root} "
        f"--pipeline {cell.pipeline} --train-rep {cell.train_rep} "
        f"--experiment-id {cell.experiment_id}"
    )


def scheduler_dry_run(**kwargs) -> dict:
    """Enumerate the matrix and assert exactly 50 unique cells (fail closed)."""

    cells = enumerate_cells(**kwargs)
    experiment_ids = [c.experiment_id for c in cells]
    triples = [(c.split_id, c.train_rep, c.pipeline) for c in cells]
    n_expected = len(SPLIT_IDS) * len(TRAIN_REPS) * len(PIPELINE_KEYS)
    unique_ids = len(set(experiment_ids))
    unique_triples = len(set(triples))
    ok = (
        len(cells) == n_expected
        and unique_ids == n_expected
        and unique_triples == n_expected
    )
    commands = [
        {
            "cell": cell.as_dict(),
            "gpu": ALLOWED_GPUS[i % len(ALLOWED_GPUS)],
            "command": build_command(cell, ALLOWED_GPUS[i % len(ALLOWED_GPUS)]),
        }
        for i, cell in enumerate(cells)
    ]
    return {
        "n_cells": len(cells),
        "n_expected": n_expected,
        "unique_experiment_ids": unique_ids,
        "unique_cell_triples": unique_triples,
        "exactly_50_unique_jobs": ok,
        "allowed_gpus": list(ALLOWED_GPUS),
        "gpu_excluded": 0,
        "cuda_device_order": "PCI_BUS_ID",
        "cells": [c.as_dict() for c in cells],
        "commands": commands,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", default=DEFAULT_MANIFEST)
    parser.add_argument("--folds-template", default=DEFAULT_FOLDS_TEMPLATE)
    parser.add_argument("--class-splits-csv", default=DEFAULT_CLASS_SPLITS)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--show-commands", action="store_true")
    args = parser.parse_args(argv)
    report = scheduler_dry_run(
        manifest_csv=args.manifest_csv,
        folds_template=args.folds_template,
        class_splits_csv=args.class_splits_csv,
        image_root=args.image_root,
    )
    if not args.show_commands:
        report_view = dict(report)
        report_view.pop("commands", None)
        report_view["cells"] = f"{len(report['cells'])} cells (omitted; use --show-commands)"
        print(json.dumps(report_view, indent=2))
    else:
        print(json.dumps(report, indent=2))
    return 0 if report["exactly_50_unique_jobs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
