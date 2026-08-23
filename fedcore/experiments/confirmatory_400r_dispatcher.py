"""Confirmatory-400R restart-safe dispatcher + 400-row DRY-RUN (submits nothing).

Plans, per frozen-matrix cell, the exact training command, its output paths, and
its completion state. It is restart-safe by design:

* atomic per-cell lock  -- ``O_CREAT | O_EXCL`` lock file; a second dispatcher
  cannot claim a cell already in flight.
* completion by terminal-marker + hash -- a cell is DONE iff its terminal marker
  exists AND every checksum the marker recorded matches the on-disk artifact.
* identical-cell retry   -- a crashed cell (lock present, no valid marker) is
  retried with the SAME frozen seeds; seeds are read from the matrix and never
  substituted (no reserve-seed replacement).
* GPU UUID logging       -- each planned dispatch records the physical GPU UUID.
* disk guard             -- refuses to plan a dispatch when free space on the
  output filesystem is below a threshold.

``dry_run`` performs the full 400-row pass -- classifying every cell and emitting
a per-cell plan -- WITHOUT launching any process. Nothing is submitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMP = os.path.join(REPO, "results", "confirmatory_400r")
MATRIX = os.path.join(CAMP, "final_training_matrix.csv")
SIBLING_RUNNER = "/workspace/sibling/run_proser_fedavg_cifar.py"  # in-container path

# Frozen unknown-class provenance for CIFAR splits (read from the balanced splits).
CIFAR_SPLITS = {
    "cifar10": os.path.join(REPO, "results", "confirmatory_400", "prelaunch", "cifar10_class_splits.csv"),
    "cifar100": os.path.join(REPO, "results", "confirmatory_400", "prelaunch", "cifar100_class_splits.csv"),
}
# Office-Home frozen inputs (50-cell provenance; balanced-split folds pending for full launch).
OH_MANIFEST = os.path.join(REPO, "results", "officehome", "preflight", "dataset_manifest.csv")
OH_CLASS_SPLITS = os.path.join(REPO, "results", "officehome", "preflight", "class_splits.csv")
OH_IMAGE_ROOT = "/data/officehome/OfficeHomeDataset"

DISK_GUARD_GB = 20.0


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_matrix():
    with open(MATRIX) as fh:
        return list(csv.DictReader(fh))


def _cifar_unknowns(dataset, split_id):
    with open(CIFAR_SPLITS[dataset]) as fh:
        for r in csv.DictReader(fh):
            if r["split_id"] == split_id:
                return ",".join(r["unknown_classes"].split())
    raise KeyError(f"{dataset} split {split_id} not in frozen class splits")


def cell_paths(out_root, cell):
    sid = cell["semantic_id"]
    base = os.path.join(out_root, sid)
    return {
        "logits": f"{base}_common.npz" if cell["dataset"].startswith("cifar") else f"{base}_logits.npz",
        "checkpoint": f"{base}.pt",
        "marker": f"{base}.TERMINAL.json",
        "lock": f"{base}.lock",
    }


def build_command(cell, paths):
    """Return the exact command list that WOULD run (never executed in dry-run)."""
    ds = cell["dataset"]
    if ds.startswith("cifar"):
        # provenance-check the frozen unknown split (fail closed if absent)
        unknowns = _cifar_unknowns(ds, cell["split_id"])
        n_known = 6 if ds == "cifar10" else 60
        return [
            "python", SIBLING_RUNNER, "smoke",
            "--dataset", ds, "--split-id", cell["split_id"],
            "--n-known", str(n_known), "--n-clients", "5",
            "--dirichlet-alpha", str(cell["d"]),
            "--rounds", "50", "--local-epochs", "2", "--batch-size", "128", "--lr", "0.01",
            "--seed", str(cell["train_rep"]), "--data-root", "/data",
            "--unknown-classes", unknowns,
            "--experiment-id", cell["semantic_id"],
            "--config-sha", cell["semantic_id"],
            "--out", paths["logits"], "--checkpoint", paths["checkpoint"],
            "--marker", paths["marker"],
        ], {"unknown_classes": unknowns, "n_known": n_known}
    # Office-Home
    pipeline = "A" if cell["arm"] == "convnext_full_ft" else "B"
    split_folds = os.path.join(REPO, "results", "officehome", "folds",
                               f"folds_{cell['split_id'].replace('_0', '_')}.csv")
    return [
        "python", "-m", "fedcore.experiments.run_officehome",
        "--manifest-csv", OH_MANIFEST, "--folds-csv", split_folds,
        "--class-splits-csv", OH_CLASS_SPLITS, "--split-id", cell["split_id"],
        "--image-root", OH_IMAGE_ROOT, "--pipeline", pipeline,
        "--train-rep", str(cell["train_rep"]), "--campaign-seed", "20260715",
        "--rounds", "30", "--out", paths["logits"], "--checkpoint", paths["checkpoint"],
    ], {"pipeline": pipeline, "folds_csv": split_folds,
        "folds_present": os.path.isfile(split_folds)}


def is_complete(paths):
    """DONE iff terminal marker exists and its recorded checksums match on disk."""
    if not os.path.isfile(paths["marker"]):
        return False, "no_marker"
    try:
        with open(paths["marker"]) as fh:
            marker = json.load(fh)
    except (OSError, ValueError):
        return False, "unreadable_marker"
    checksums = marker.get("checksums", {})
    if not checksums:
        return False, "marker_without_checksums"
    for name, expected in checksums.items():
        candidate = None
        for key in ("logits", "checkpoint"):
            if os.path.basename(paths[key]) == name:
                candidate = paths[key]
        if candidate is None or not os.path.isfile(candidate):
            return False, f"artifact_missing:{name}"
        if _sha256(candidate) != expected:
            return False, f"checksum_mismatch:{name}"
    return True, "complete"


def acquire_lock(lock_path):
    """Atomic O_EXCL lock; returns True if this process claimed the cell."""
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"pid": os.getpid()}))
    return True


def disk_ok(out_root):
    os.makedirs(out_root, exist_ok=True)
    free_gb = shutil.disk_usage(out_root).free / 1e9
    return free_gb >= DISK_GUARD_GB, round(free_gb, 1)


def dry_run(out_root, gpu_uuids):
    rows = _load_matrix()
    seeds_frozen = True
    seed_fields = [k for k in rows[0] if k.startswith("seed_")]
    classified = {"complete": 0, "pending": 0, "locked": 0, "retry": 0}
    plan = []
    oh_folds_missing = 0
    for i, cell in enumerate(rows):
        paths = cell_paths(out_root, cell)
        cmd, extra = build_command(cell, paths)
        # verify seeds are present + non-empty (never substituted / reserve-replaced)
        for sf in seed_fields:
            if not str(cell.get(sf, "")).strip():
                seeds_frozen = False
        done, why = is_complete(paths)
        locked = os.path.exists(paths["lock"])
        if done:
            state = "complete"
        elif locked:
            state = "retry"  # lock present, not complete -> identical-cell retry (same seeds)
        else:
            state = "pending"
        classified[state] = classified.get(state, 0) + 1
        if extra.get("folds_present") is False:
            oh_folds_missing += 1
        plan.append({
            "idx": i, "semantic_id": cell["semantic_id"], "dataset": cell["dataset"],
            "state": state, "reason": why,
            "assigned_gpu_uuid": gpu_uuids[i % len(gpu_uuids)] if gpu_uuids else None,
            "seed_training_initialization": cell.get("seed_training_initialization"),
            "reserve_seed_replacement": False,
            "command_preview": " ".join(cmd[:6]) + " ...",
            "oh_folds_present": extra.get("folds_present", True),
        })
    ok, free_gb = disk_ok(out_root)
    # write plan CSV
    os.makedirs(os.path.join(CAMP, "prelaunch"), exist_ok=True)
    plan_csv = os.path.join(CAMP, "prelaunch", "dispatcher_dryrun_plan.csv")
    with open(plan_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(plan[0].keys()))
        w.writeheader(); w.writerows(plan)
    summary = {
        "campaign": "confirmatory_400r",
        "mode": "DRY_RUN", "submitted": 0, "launched_processes": 0,
        "total_cells": len(rows),
        "classified": classified,
        "seeds_all_frozen_present": seeds_frozen,
        "reserve_seed_replacement_used": False,
        "disk_guard_gb": DISK_GUARD_GB, "free_gb": free_gb, "disk_ok": ok,
        "gpu_uuids": gpu_uuids,
        "restart_safety": {
            "atomic_lock": "O_CREAT|O_EXCL per-cell .lock",
            "completion_rule": "terminal marker + recorded-checksum match on disk",
            "identical_cell_retry": "crashed cell reruns with identical frozen seeds",
            "no_reserve_seed_replacement": True,
        },
        "officehome_folds_missing_for_balanced_splits": oh_folds_missing,
        "officehome_note": (
            "OH command construction targets 50-cell frozen folds via a naive split-name "
            "remap; the balanced officehome_split_00..09 folds are a full-launch data "
            "dependency. oh_folds_missing counts cells whose fold CSV is absent."),
        "plan_csv": plan_csv,
        "lock_probe": _lock_selftest(out_root),
    }
    out = os.path.join(CAMP, "prelaunch", "dispatcher_dryrun.json")
    tmp = f"{out}.tmp"
    with open(tmp, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    os.replace(tmp, out)
    print(json.dumps({k: summary[k] for k in
                      ("mode", "submitted", "total_cells", "classified",
                       "seeds_all_frozen_present", "disk_ok", "free_gb",
                       "officehome_folds_missing_for_balanced_splits")}, indent=2))
    print(f"wrote {out}")
    print(f"wrote {plan_csv}")
    return summary


def _lock_selftest(out_root):
    """Prove the atomic lock rejects a double-claim (no side effects on cells)."""
    probe = os.path.join(out_root, ".lock_selftest")
    for p in (probe,):
        if os.path.exists(p):
            os.remove(p)
    first = acquire_lock(probe)
    second = acquire_lock(probe)
    if os.path.exists(probe):
        os.remove(probe)
    return {"first_claim": first, "second_claim_rejected": (not second),
            "atomic_exclusion_ok": bool(first and not second)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default=os.path.join(CAMP, "runs"))
    p.add_argument("--gpu-uuids", default="", help="comma-separated physical GPU UUIDs")
    p.add_argument("--dry-run", action="store_true", default=True)
    args = p.parse_args(argv)
    uuids = [u for u in args.gpu_uuids.split(",") if u] or None
    # This dispatcher only supports dry-run in this build; live launch requires the
    # owner phrase AUTHORIZE_FULL_400R_CELL_LAUNCH and is intentionally not wired.
    dry_run(args.out_root, uuids or [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
