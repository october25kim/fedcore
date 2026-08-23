"""Confirmatory-400R measured resource projection -- OPTION B (400 FRESH cells).

Recomputes ``measured_resource_projection.json`` so the Office-Home rows are
extrapolated from the FRESH-confirmatory smoke markers (split_00 x rep0 x
{full, frozen}), NOT from any reused historical Office-Home artifact.  All 400
cells are fresh_training (verified from the matrix), so there is zero OH reuse
credit.  Measured per-round train time is scaled to the canonical production
round counts (CIFAR 50, Office-Home 30) x matrix cardinality.
"""

from __future__ import annotations

import collections
import csv
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRE = os.path.join(REPO, "results", "confirmatory_400r", "prelaunch")
MATRIX = os.path.join(REPO, "results", "confirmatory_400r", "final_training_matrix.csv")
CIFAR_ART = os.path.join(PRE, "smoke_artifacts")
OH_ART = os.path.join(PRE, "smoke_artifacts_c400r")

# path key -> (marker file, matrix pipeline_id, production rounds)
SOURCES = {
    "cifar10": (os.path.join(CIFAR_ART, "S1_cifar10.TERMINAL.json"), "cifar10_proser_fedavg", 50),
    "cifar100": (os.path.join(CIFAR_ART, "S2_cifar100.TERMINAL.json"), "cifar100_proser_fedavg", 50),
    "officehome_full": (os.path.join(OH_ART, "C400R_full.TERMINAL.json"), "officehome_convnext_full", 30),
    "officehome_frozen": (os.path.join(OH_ART, "C400R_frozen.TERMINAL.json"), "officehome_convnext_frozen", 30),
}


def _load(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def build():
    rows = list(csv.DictReader(open(MATRIX)))
    counts = collections.Counter(r["pipeline_id"] for r in rows)
    all_fresh = all(r["reuse_class"] == "fresh_training" for r in rows if r["dataset"] == "officehome")

    per_path = {}
    total_cell_min = 0.0
    total_ckpt = 0
    total_logits = 0
    all_measured = True
    for key, (marker_path, pipeline_id, prod_r) in SOURCES.items():
        m = _load(marker_path)
        n_cells = counts.get(pipeline_id, 0)
        if m is None:
            per_path[key] = {"status": "marker_missing", "expected": os.path.relpath(marker_path, REPO),
                             "n_cells_in_matrix": n_cells}
            all_measured = False
            continue
        smoke_rounds = int(m.get("rounds", 2))
        train_s = float(m.get("train_seconds") or 0.0)
        export_s = float(m.get("export_seconds") or 0.0)
        wall_s = float(m.get("wall_seconds") or (train_s + export_s))
        per_round = train_s / max(smoke_rounds, 1)
        cell_s = per_round * prod_r + export_s
        ckpt = int(m.get("checkpoint_bytes") or 0)
        logits = int(m.get("logits_npz_bytes") or 0)
        per_path[key] = {
            "status": "measured", "pipeline_id": pipeline_id,
            "source_marker": os.path.relpath(marker_path, REPO),
            "fresh_confirmatory": key.startswith("officehome"),
            "smoke_rounds": smoke_rounds,
            "measured_train_seconds": round(train_s, 3),
            "measured_export_seconds": round(export_s, 3),
            "measured_wall_seconds": round(wall_s, 3),
            "measured_peak_vram_gb": m.get("peak_vram_gb"),
            "gpu_name": m.get("gpu_name"),
            "production_rounds": prod_r,
            "extrapolated_cell_gpu_min": round(cell_s / 60.0, 3),
            "n_cells_in_matrix": n_cells,
            "extrapolated_pipeline_gpu_hours": round(cell_s * n_cells / 3600.0, 3),
        }
        total_cell_min += (cell_s / 60.0) * n_cells
        total_ckpt += ckpt * n_cells
        total_logits += logits * n_cells

    total_gpu_h = total_cell_min / 60.0
    proj = {
        "campaign": "confirmatory_400r",
        "option": "B_fresh_confirmatory_officehome",
        "measured_from": "CIFAR: prior 2-round smokes S1/S2; Office-Home: FRESH split_00 rep0 2-round smokes (TITAN RTX)",
        "all_fresh_training_no_oh_reuse": bool(all_fresh),
        "officehome_reuse_credit": 0,
        "cells_by_pipeline": dict(counts),
        "per_path": per_path,
        "all_paths_measured": all_measured,
        "extrapolated_total_gpu_hours_400": round(total_gpu_h, 2),
        "extrapolated_wall_hours": {
            "1_gpu": round(total_gpu_h, 2), "2_gpu": round(total_gpu_h / 2, 2),
            "3_gpu": round(total_gpu_h / 3, 2), "4_gpu": round(total_gpu_h / 4, 2)},
        "extrapolated_total_checkpoint_gb": round(total_ckpt / 1e9, 2),
        "extrapolated_total_logits_gb": round(total_logits / 1e9, 3),
        "extrapolated_total_storage_gb": round((total_ckpt + total_logits) / 1e9, 2),
        "caveat": ("Per-round measured train time scaled to canonical production rounds "
                   "(CIFAR 50, Office-Home 30) x matrix cardinality; smokes used 2 rounds. "
                   "Order-of-magnitude projection."),
    }
    out = os.path.join(PRE, "measured_resource_projection.json")
    tmp = f"{out}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(proj, fh, indent=2, default=str)
    os.replace(tmp, out)
    print(json.dumps({"all_fresh_training_no_oh_reuse": proj["all_fresh_training_no_oh_reuse"],
                      "all_paths_measured": all_measured,
                      "extrapolated_total_gpu_hours_400": proj["extrapolated_total_gpu_hours_400"]}, indent=2))
    return proj


if __name__ == "__main__":
    build()
