"""Confirmatory-400R restart-safe dispatcher DRY-RUN against the BOUND matrix.

Targets ``matrix_history/final_training_matrix_v2_bound.csv`` (section 7 output),
so every Office-Home cell dispatches its FRESH confirmatory fold via the bound
``fold_path`` + ``fold_sha256`` columns -- never a historical
``folds_officehome_split_*`` fold.  Submits NOTHING.

Dry-run assertions (section 10-11):
  * 400 rows classified (150 cifar10 + 150 cifar100 + 50 full + 50 frozen);
  * 0 duplicate semantic_ids;
  * 0 missing OH fold files / 0 fold-hash mismatches vs the bound column;
  * 0 historical OH fold references;
  * 0 submitted / 0 launched processes;
  * 50 cifar10 + 50 cifar100 three-d blocks (each 3 Dirichlet alphas);
  * 50 Office-Home paired blocks (full + frozen);
  * atomic O_EXCL lock self-test.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os

from fedcore.experiments.confirmatory_400r_dispatcher import (
    _cifar_unknowns, _sha256, acquire_lock, cell_paths, disk_ok, is_complete,
    _lock_selftest, DISK_GUARD_GB, SIBLING_RUNNER,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMP = os.path.join(REPO, "results", "confirmatory_400r")
BOUND_MATRIX = os.path.join(CAMP, "final_training_matrix_v2_bound.csv")
OH_MANIFEST = os.path.join(REPO, "results", "officehome", "dedup", "retained_canonical_manifest.csv")
OH_CLASS_SPLITS = os.path.join(REPO, "results", "confirmatory_400r", "prelaunch",
                               "officehome_folds", "officehome_c400r_class_splits.csv")
OH_IMAGE_ROOT = "/data/officehome/OfficeHomeDataset"


def _load_matrix():
    with open(BOUND_MATRIX, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def build_command_bound(cell, paths):
    ds = cell["dataset"]
    if ds.startswith("cifar"):
        unknowns = _cifar_unknowns(ds, cell["split_id"])
        n_known = 6 if ds == "cifar10" else 60
        cmd = [
            "python", SIBLING_RUNNER, "smoke",
            "--dataset", ds, "--split-id", cell["split_id"],
            "--n-known", str(n_known), "--n-clients", "5",
            "--dirichlet-alpha", str(cell["d"]),
            "--rounds", "50", "--local-epochs", "2", "--batch-size", "128", "--lr", "0.01",
            "--seed", str(cell["train_rep"]), "--data-root", "/data",
            "--unknown-classes", unknowns,
            "--experiment-id", cell["semantic_id"], "--config-sha", cell["semantic_id"],
            "--out", paths["logits"], "--checkpoint", paths["checkpoint"], "--marker", paths["marker"],
        ]
        return cmd, {"kind": "cifar"}
    # Office-Home: use the BOUND fresh fold columns.
    pipeline = "A" if cell["arm"] == "convnext_full_ft" else "B"
    fold_rel = cell["fold_path"]
    fold_abs = os.path.join(REPO, fold_rel)
    fold_present = os.path.isfile(fold_abs)
    bound_sha = cell["fold_sha256"]
    on_disk_sha = _sha256(fold_abs) if fold_present else None
    fold_hash_ok = bool(fold_present and on_disk_sha == bound_sha)
    historical_ref = ("folds_officehome_split_" in fold_rel) or ("legacy" in cell.get("fold_split_id", ""))
    cmd = [
        "python", "-m", "fedcore.experiments.run_officehome",
        "--manifest-csv", OH_MANIFEST, "--folds-csv", fold_abs,
        "--class-splits-csv", OH_CLASS_SPLITS, "--split-id", cell["fold_split_id"],
        "--image-root", OH_IMAGE_ROOT, "--pipeline", pipeline,
        "--train-rep", str(cell["train_rep"]), "--campaign-seed", "20260715",
        "--rounds", "30", "--out", paths["logits"], "--checkpoint", paths["checkpoint"],
    ]
    return cmd, {"kind": "officehome", "fold_split_id": cell["fold_split_id"],
                 "fold_path": fold_rel, "fold_present": fold_present,
                 "bound_fold_sha256": bound_sha, "on_disk_fold_sha256": on_disk_sha,
                 "fold_hash_ok": fold_hash_ok, "historical_ref": historical_ref}


def dry_run(out_root, gpu_uuids):
    rows = _load_matrix()
    seed_fields = [k for k in rows[0] if k.startswith("seed_")]
    classified = collections.Counter()
    plan = []
    seen_ids = collections.Counter()
    oh_missing_fold = 0
    oh_fold_hash_mismatch = 0
    oh_historical_ref = 0
    seeds_frozen = True
    for i, cell in enumerate(rows):
        seen_ids[cell["semantic_id"]] += 1
        paths = cell_paths(out_root, cell)
        cmd, extra = build_command_bound(cell, paths)
        for sf in seed_fields:
            if not str(cell.get(sf, "")).strip():
                seeds_frozen = False
        done, why = is_complete(paths)
        locked = os.path.exists(paths["lock"])
        state = "complete" if done else ("retry" if locked else "pending")
        classified[state] += 1
        if extra.get("kind") == "officehome":
            if not extra["fold_present"]:
                oh_missing_fold += 1
            if not extra["fold_hash_ok"]:
                oh_fold_hash_mismatch += 1
            if extra["historical_ref"]:
                oh_historical_ref += 1
        plan.append({
            "idx": i, "semantic_id": cell["semantic_id"], "dataset": cell["dataset"],
            "pipeline_id": cell["pipeline_id"], "split_id": cell["split_id"],
            "fold_split_id": cell.get("fold_split_id", ""), "state": state, "reason": why,
            "assigned_gpu_uuid": gpu_uuids[i % len(gpu_uuids)] if gpu_uuids else None,
            "oh_fold_present": extra.get("fold_present", True),
            "oh_fold_hash_ok": extra.get("fold_hash_ok", True),
            "oh_historical_ref": extra.get("historical_ref", False),
            "reserve_seed_replacement": False,
            "command_preview": " ".join(cmd[:6]) + " ...",
        })

    duplicate_ids = {k: v for k, v in seen_ids.items() if v > 1}
    # block accounting
    def blocks_for(pred):
        pb = collections.defaultdict(list)
        for r in rows:
            if pred(r):
                pb[r["paired_block"]].append(r)
        return pb
    cifar10_blocks = blocks_for(lambda r: r["pipeline_id"] == "cifar10_proser_fedavg")
    cifar100_blocks = blocks_for(lambda r: r["pipeline_id"] == "cifar100_proser_fedavg")
    oh_blocks = blocks_for(lambda r: r["dataset"] == "officehome")
    cifar10_3d = sum(1 for v in cifar10_blocks.values() if len(v) == 3)
    cifar100_3d = sum(1 for v in cifar100_blocks.values() if len(v) == 3)
    oh_paired = sum(1 for v in oh_blocks.values()
                    if sorted(r["pipeline_id"] for r in v) ==
                    ["officehome_convnext_frozen", "officehome_convnext_full"])

    ok_disk, free_gb = disk_ok(out_root)
    n_oh = sum(1 for r in rows if r["dataset"] == "officehome")
    summary = {
        "campaign": "confirmatory_400r",
        "matrix": os.path.relpath(BOUND_MATRIX, REPO),
        "mode": "DRY_RUN", "submitted": 0, "launched_processes": 0,
        "total_cells": len(rows),
        "cells_by_pipeline": dict(collections.Counter(r["pipeline_id"] for r in rows)),
        "classified": dict(classified),
        "duplicate_semantic_ids": duplicate_ids,
        "n_duplicate_semantic_ids": len(duplicate_ids),
        "seeds_all_frozen_present": seeds_frozen,
        "reserve_seed_replacement_used": False,
        "officehome_cells": n_oh,
        "officehome_missing_fold_files": oh_missing_fold,
        "officehome_fold_hash_mismatches": oh_fold_hash_mismatch,
        "officehome_historical_fold_references": oh_historical_ref,
        "cifar10_three_d_blocks": cifar10_3d,
        "cifar100_three_d_blocks": cifar100_3d,
        "officehome_paired_blocks": oh_paired,
        "disk_guard_gb": DISK_GUARD_GB, "free_gb": free_gb, "disk_ok": ok_disk,
        "gpu_uuids": gpu_uuids,
        "lock_probe": _lock_selftest(out_root),
        "restart_safety": {
            "atomic_lock": "O_CREAT|O_EXCL per-cell .lock",
            "completion_rule": "terminal marker + recorded-checksum match on disk",
            "identical_cell_retry": "crashed cell reruns with identical frozen seeds",
            "no_reserve_seed_replacement": True,
        },
        "dry_run_pass": bool(
            len(rows) == 400 and len(duplicate_ids) == 0
            and oh_missing_fold == 0 and oh_fold_hash_mismatch == 0
            and oh_historical_ref == 0 and seeds_frozen
            and cifar10_3d == 50 and cifar100_3d == 50 and oh_paired == 50
            and classified.get("pending", 0) == 400
        ),
    }
    pre = os.path.join(CAMP, "prelaunch")
    os.makedirs(pre, exist_ok=True)
    plan_csv = os.path.join(pre, "dispatcher_dryrun_plan_bound.csv")
    with open(plan_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(plan[0].keys()))
        w.writeheader(); w.writerows(plan)
    out = os.path.join(pre, "dispatcher_dryrun_bound.json")
    tmp = f"{out}.tmp"
    with open(tmp, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    os.replace(tmp, out)
    print(json.dumps({k: summary[k] for k in (
        "mode", "submitted", "total_cells", "classified", "n_duplicate_semantic_ids",
        "officehome_missing_fold_files", "officehome_fold_hash_mismatches",
        "officehome_historical_fold_references", "cifar10_three_d_blocks",
        "cifar100_three_d_blocks", "officehome_paired_blocks", "dry_run_pass")}, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=os.path.join(CAMP, "runs"))
    p.add_argument("--gpu-uuids", default="")
    args = p.parse_args(argv)
    uuids = [u for u in args.gpu_uuids.split(",") if u]
    dry_run(args.out_root, uuids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
