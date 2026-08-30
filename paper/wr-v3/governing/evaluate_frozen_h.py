#!/usr/bin/env python3
"""Evaluate the already frozen WR-v3 H selections on disjoint diagnostic folds.

This script is deliberately post-certification. It consumes the frozen H
selection table and never changes a selector, threshold, certificate, or
headline. The output is descriptive and is not a second certificate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ALPHA = 0.20
CONTRACT_ID = "fedcore-wr-v3-postcert-evaluation-v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AssertionError(f"refusing to write empty output: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def office_scores(logits: np.ndarray) -> dict[str, np.ndarray]:
    probabilities = softmax(np.asarray(logits, dtype=float))
    ordered = np.sort(probabilities, axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    energy = logits.max(axis=1) + np.log(np.exp(shifted).sum(axis=1))
    return {
        "native": probabilities.max(axis=1),
        "energy": energy,
        "margin": ordered[:, -1] - ordered[:, -2],
    }


def common_scores(z: Any, role: str) -> dict[str, np.ndarray]:
    return {
        "native": np.asarray(z[f"{role}__native_score"], dtype=float),
        "energy": np.asarray(z[f"{role}__energy_score"], dtype=float),
        "margin": np.asarray(z[f"{role}__known_margin_score"], dtype=float),
    }


def load_evaluation(path: Path, kind: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        if kind == "officehome":
            logits = np.asarray(z["eval_logits"], dtype=float)
            ids = {
                "proposal": np.asarray(z["prop_sample_id"], dtype=str),
                "certification": np.asarray(z["cert_sample_id"], dtype=str),
                "evaluation": np.asarray(z["eval_sample_id"], dtype=str),
            }
            return {
                "scores": office_scores(logits),
                "pred": logits.argmax(axis=1).astype(np.int64),
                "y": np.asarray(z["eval_y_open"], dtype=np.int64),
                "client": np.asarray(z["eval_client"], dtype=np.int64),
                "ids": ids,
            }
        ids = {
            role: np.asarray(z[f"{role}__immutable_source_id"], dtype=str)
            for role in ("proposal", "certification", "test")
        }
        return {
            "scores": common_scores(z, "test"),
            "pred": np.asarray(z["test__predicted_known_index"], dtype=np.int64),
            "y": np.asarray(z["test__true_known_class_index_or_neg1"], dtype=np.int64),
            "client": np.asarray(z["test__client_id"], dtype=np.int64),
            "ids": {
                "proposal": ids["proposal"],
                "certification": ids["certification"],
                "evaluation": ids["test"],
            },
        }


def disjointness(ids: dict[str, np.ndarray]) -> tuple[int, int, int]:
    sets = {name: set(values.tolist()) for name, values in ids.items()}
    for name, values in ids.items():
        if len(values) != len(sets[name]):
            raise AssertionError(f"duplicate source ID within {name} fold")
    return (
        len(sets["proposal"] & sets["certification"]),
        len(sets["proposal"] & sets["evaluation"]),
        len(sets["certification"] & sets["evaluation"]),
    )


def terminal_path(npz_path: Path, kind: str) -> Path:
    suffix = "_logits.npz" if kind == "officehome" else "_common.npz"
    if not npz_path.name.endswith(suffix):
        raise ValueError(npz_path)
    return npz_path.with_name(npz_path.name[: -len(suffix)] + ".TERMINAL.json")


def metrics(view: dict[str, Any], slot: str, threshold: float, J: int) -> dict[str, Any]:
    accept = np.asarray(view["scores"][slot]) >= threshold
    pred = np.asarray(view["pred"])
    y = np.asarray(view["y"])
    client = np.asarray(view["client"])
    error = (y < 0) | (pred != y)
    A = np.asarray([(accept & (client == j)).sum() for j in range(J)], dtype=int)
    K = np.asarray([(accept & error & (client == j)).sum() for j in range(J)], dtype=int)
    n = np.asarray([(client == j).sum() for j in range(J)], dtype=int)
    if np.any(n <= 0) or np.any(K > A) or np.any(A > n):
        raise AssertionError("invalid evaluation counts")
    total_A, total_K = int(A.sum()), int(K.sum())
    risks = np.divide(K, A, out=np.full(J, np.nan), where=A > 0)
    finite = risks[np.isfinite(risks)]
    pooled = total_K / total_A if total_A else math.nan
    worst = float(finite.max()) if len(finite) else math.nan
    known = y >= 0
    unknown = ~known
    known_accepted = int((accept & known).sum())
    known_errors = int((accept & known & (pred != y)).sum())
    unknown_total = int(unknown.sum())
    unknown_accepts = int((accept & unknown).sum())
    if total_K != known_errors + unknown_accepts:
        raise AssertionError("error decomposition does not sum")
    return {
        "evaluation_n": int(n.sum()),
        "evaluation_A": total_A,
        "evaluation_K": total_K,
        "evaluation_coverage": total_A / int(n.sum()),
        "pooled_evaluation_risk": pooled,
        "worst_client_evaluation_risk": worst,
        "clients_with_zero_acceptance": int((A == 0).sum()),
        "accepted_known": known_accepted,
        "known_accepted_errors": known_errors,
        "unknown_total": unknown_total,
        "unknown_false_acceptances": unknown_accepts,
        "unknown_share_of_accepted_errors": unknown_accepts / total_K if total_K else math.nan,
        "A_by_client": json.dumps(A.tolist(), separators=(",", ":")),
        "K_by_client": json.dumps(K.tolist(), separators=(",", ":")),
        "n_by_client": json.dumps(n.tolist(), separators=(",", ":")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--raw-input-pins", required=True)
    parser.add_argument("--primary-per-cell", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    analysis = read_csv(Path(args.analysis_manifest))
    meta_map = {r["semantic_id"]: r for r in analysis}
    pins = read_csv(Path(args.raw_input_pins))
    pin_map = {r["semantic_id"]: r for r in pins}
    if len(meta_map) != 450 or len(pin_map) != 450 or set(meta_map) != set(pin_map):
        raise AssertionError("input roster mismatch")

    primary = read_csv(Path(args.primary_per_cell))
    h_rows = [r for r in primary if r["procedure"] == "H" and abs(float(r["alpha"]) - ALPHA) < 1e-12]
    if len(h_rows) != 450 or len({r["semantic_id"] for r in h_rows}) != 450:
        raise AssertionError("primary H roster mismatch")
    selected = [r for r in h_rows if int(r["certified"]) == 1]
    if len(selected) != 177:
        raise AssertionError(f"expected 177 frozen H selections, found {len(selected)}")

    cell_rows: list[dict[str, Any]] = []
    for index, frozen in enumerate(sorted(selected, key=lambda r: r["semantic_id"]), start=1):
        sid = frozen["semantic_id"]
        meta = meta_map[sid]
        pin = pin_map[sid]
        artifact = repo_root / meta["path"]
        sidecar = terminal_path(artifact, meta["kind"])
        if sha256_file(artifact) != pin["sha256"] or sha256_file(sidecar) != pin["terminal_sha256"]:
            raise AssertionError(f"byte pin mismatch for {sid}")
        view = load_evaluation(artifact, meta["kind"])
        overlaps = disjointness(view["ids"])
        if overlaps != (0, 0, 0):
            raise AssertionError(f"fold source-ID overlap for {sid}: {overlaps}")
        result = metrics(
            view, frozen["selected_slot"], float(frozen["selected_threshold"]), int(frozen["J"])
        )
        cell_rows.append({
            "semantic_id": sid,
            "dataset": meta["dataset"],
            "condition": meta["condition"],
            "selected_candidate_index": frozen["selected_candidate_index"],
            "selected_score": frozen["selected_score"],
            "selected_slot": frozen["selected_slot"],
            "selected_gamma": frozen["selected_gamma"],
            "selected_threshold": frozen["selected_threshold"],
            "selected_selector_sha256": frozen["selected_selector_sha256"],
            **result,
            "pooled_risk_le_alpha": int(result["pooled_evaluation_risk"] <= ALPHA),
            "worst_client_risk_le_alpha": int(result["worst_client_evaluation_risk"] <= ALPHA),
            "proposal_certification_id_overlap": overlaps[0],
            "proposal_evaluation_id_overlap": overlaps[1],
            "certification_evaluation_id_overlap": overlaps[2],
        })
        print(f"[{index:03d}/177] {sid}", flush=True)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        groups[(row["dataset"], row["condition"])].append(row)
    groups[("ALL", "ALL")] = cell_rows
    for dataset in sorted({r["dataset"] for r in cell_rows}):
        groups[(dataset, "ALL")] = [r for r in cell_rows if r["dataset"] == dataset]

    aggregate_rows = []
    for (dataset, condition), rows in sorted(groups.items()):
        total_A = sum(int(r["evaluation_A"]) for r in rows)
        total_K = sum(int(r["evaluation_K"]) for r in rows)
        known_errors = sum(int(r["known_accepted_errors"]) for r in rows)
        unknown_accepts = sum(int(r["unknown_false_acceptances"]) for r in rows)
        aggregate_rows.append({
            "dataset": dataset,
            "condition": condition,
            "certified_cells_evaluated": len(rows),
            "total_evaluation_accepted": total_A,
            "total_evaluation_accepted_errors": total_K,
            "aggregate_pooled_evaluation_risk": total_K / total_A if total_A else math.nan,
            "known_accepted_errors": known_errors,
            "unknown_false_acceptances": unknown_accepts,
            "unknown_share_of_accepted_errors": unknown_accepts / total_K if total_K else math.nan,
            "cells_pooled_risk_above_alpha": sum(not int(r["pooled_risk_le_alpha"]) for r in rows),
            "cells_worst_client_risk_above_alpha": sum(not int(r["worst_client_risk_le_alpha"]) for r in rows),
            "max_worst_client_evaluation_risk": max(float(r["worst_client_evaluation_risk"]) for r in rows),
        })

    write_csv(output / "evaluation_per_certified_cell.csv", cell_rows)
    write_csv(output / "evaluation_aggregate.csv", aggregate_rows)
    validation = {
        "status": "PASS",
        "contract_id": CONTRACT_ID,
        "primary_H_cells": 450,
        "primary_H_certified_evaluated": len(cell_rows),
        "raw_npz_pin_checks": len(cell_rows),
        "terminal_sidecar_pin_checks": len(cell_rows),
        "fold_overlap_failures": 0,
        "certificate_or_selection_modified": False,
        "evaluation_is_a_certificate": False,
    }
    (output / "VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in output.iterdir() if p.is_file())
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.name}\n")


if __name__ == "__main__":
    main()
