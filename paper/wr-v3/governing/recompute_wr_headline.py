#!/usr/bin/env python3
"""Recompute the Fed-CORE H/S/B headline under a sealed WR audit contract.

The script never proposes or tunes a selector.  It consumes the already frozen
proposal-defined family, verifies that its record-level marks reproduce the
archived as-is certification counts, and only then draws one primary audit from
each fixed client certification reservoir with replacement.

The primary draw is the scientific headline.  Cell-level bootstrap intervals
describe variation across frozen split/repetition blocks; they are not theorem
confidence statements and they do not turn the 450 cells into a simultaneous
confidence family.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import beta as scipy_beta
from scipy.stats import binom as scipy_binom


ALPHA_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
PRIMARY_ALPHA = 0.20
DELTA_R = 0.05
DELTA_C = 0.05
M_EXPECTED = 12
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260825
CONTRACT_ID = "fedcore-headline-wr-v3"
RNG_CONTRACT = "numpy-PCG64-SeedSequence([frozen_primary_audit_seed, client_id, 0])"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def tensor_hash(A: np.ndarray, K: np.ndarray, n: np.ndarray) -> str:
    payload = {
        "A": np.asarray(A, dtype=int).tolist(),
        "K": np.asarray(K, dtype=int).tolist(),
        "n": np.asarray(n, dtype=int).tolist(),
    }
    return canonical_json_hash(payload)


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with opener(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def alpha_key(x: float | str) -> str:
    return f"{float(x):.2f}"


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def office_scores(logits: np.ndarray) -> dict[str, np.ndarray]:
    logits = np.asarray(logits, dtype=float)
    probs = softmax(logits)
    ordered = np.sort(probs, axis=1)
    zmax = logits.max(axis=1, keepdims=True)
    energy = zmax[:, 0] + np.log(np.exp(logits - zmax).sum(axis=1))
    return {
        "native": probs.max(axis=1),
        "energy": energy,
        "margin": ordered[:, -1] - ordered[:, -2],
    }


def common_scores(z: Any, role: str) -> dict[str, np.ndarray]:
    return {
        "native": np.asarray(z[f"{role}__native_score"], dtype=float),
        "energy": np.asarray(z[f"{role}__energy_score"], dtype=float),
        "margin": np.asarray(z[f"{role}__known_margin_score"], dtype=float),
    }


def load_cell(path: Path, kind: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        if kind == "officehome":
            cert_logits = np.asarray(z["cert_logits"], dtype=float)
            return {
                "cert_scores": office_scores(cert_logits),
                "cert_pred": cert_logits.argmax(axis=1).astype(np.int64),
                "cert_y": np.asarray(z["cert_y_open"], dtype=np.int64),
                "cert_client": np.asarray(z["cert_client"], dtype=np.int64),
                "cert_source_id": np.asarray(z["cert_sample_id"], dtype=str),
                "embedded_audit_seed": int(z["audit_draw_seed"]),
            }
        return {
            "cert_scores": common_scores(z, "certification"),
            "cert_pred": np.asarray(z["certification__predicted_known_index"], dtype=np.int64),
            "cert_y": np.asarray(z["certification__true_known_class_index_or_neg1"], dtype=np.int64),
            "cert_client": np.asarray(z["certification__client_id"], dtype=np.int64),
            "cert_source_id": np.asarray(z["certification__immutable_source_id"], dtype=str),
            "embedded_audit_seed": None,
        }


def error_marks(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.asarray(y) < 0) | (np.asarray(pred) != np.asarray(y))


def cp_upper(k: int, n: int, eps: float) -> float:
    if n <= 0 or k >= n or eps <= 0.0:
        return 1.0
    return float(scipy_beta.isf(eps, k + 1, n - k))


def cp_lower(k: int, n: int, eps: float) -> float:
    if n <= 0 or k <= 0:
        return 0.0
    return float(scipy_beta.ppf(eps, k, n - k + 1))


def holm_adjusted(pvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    ranks = np.empty(p.shape, dtype=int)
    running = 0.0
    M = len(p)
    for rank, m in enumerate(order, start=1):
        running = max(running, (M - rank + 1) * float(p[m]))
        adjusted[m] = min(1.0, running)
        ranks[m] = rank
    return adjusted, ranks


def candidate_iut_pvalue(A: np.ndarray, K: np.ndarray, alpha: float) -> float:
    vals = []
    for a, k in zip(np.asarray(A, dtype=int), np.asarray(K, dtype=int)):
        vals.append(1.0 if a <= 0 else float(scipy_binom.cdf(int(k), int(a), alpha)))
    return float(max(vals))


def family_tie_key(row: dict[str, str]) -> tuple[float, str, int]:
    # The theorem-facing family names the native slot "msp" for tie-breaking,
    # including CIFAR where the displayed native score is PROSER.
    tie_score = "msp" if row["slot"] == "native" else row["slot"]
    return (float(row["gamma"]), tie_score, int(row["candidate_index"]))


def select_candidate(certified: np.ndarray, C: np.ndarray, family: list[dict[str, str]]) -> int | None:
    eligible = [m for m in range(len(family)) if bool(certified[m])]
    if not eligible:
        return None
    return min(eligible, key=lambda m: (-float(C[m]),) + family_tie_key(family[m]))


def procedure_results(
    A: np.ndarray,
    K: np.ndarray,
    n: np.ndarray,
    family: list[dict[str, str]],
    alpha: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    M, J = A.shape
    proposal_ok = np.asarray([int(r["proposal_feasible"]) == 1 for r in family], dtype=bool)

    # H: full-simplex IUT p-value, Holm over M, and delta_c/M coverage.
    p_raw = np.asarray([candidate_iut_pvalue(A[m], K[m], alpha) for m in range(M)])
    p_adj, p_rank = holm_adjusted(p_raw)
    C_h = np.asarray([
        min(cp_lower(int(A[m, j]), int(n[j]), DELTA_C / M) for j in range(J))
        for m in range(M)
    ])
    h_cert = proposal_ok & (p_adj <= DELTA_R) & (C_h > 0.0)

    # S: theorem-aligned member allocation with no client divisor.
    U_s = np.asarray([
        max(cp_upper(int(K[m, j]), int(A[m, j]), DELTA_R / M) for j in range(J))
        for m in range(M)
    ])
    C_s = np.asarray([
        min(cp_lower(int(A[m, j]), int(n[j]), DELTA_C / M) for j in range(J))
        for m in range(M)
    ])
    s_cert = proposal_ok & (U_s <= alpha) & (C_s > 0.0)

    # B: deliberately conservative clientwise confidence-allocation ablation.
    U_b = np.asarray([
        max(cp_upper(int(K[m, j]), int(A[m, j]), DELTA_R / (M * J)) for j in range(J))
        for m in range(M)
    ])
    C_b = np.asarray([
        min(cp_lower(int(A[m, j]), int(n[j]), DELTA_C / (M * J)) for j in range(J))
        for m in range(M)
    ])
    b_cert = proposal_ok & (U_b <= alpha) & (C_b > 0.0)

    if np.any(U_s > U_b + 1e-14):
        raise AssertionError("candidatewise S UCB exceeded B UCB")
    if np.any(C_s + 1e-14 < C_b):
        raise AssertionError("candidatewise S coverage LCB fell below B")

    arrays = {
        "H_cert": h_cert, "H_C": C_h, "H_p_raw": p_raw, "H_p_adj": p_adj,
        "H_rank": p_rank, "S_cert": s_cert, "S_C": C_s, "S_U": U_s,
        "B_cert": b_cert, "B_C": C_b, "B_U": U_b,
    }
    out: dict[str, dict[str, Any]] = {}
    for proc, cert, C in (("H", h_cert, C_h), ("S", s_cert, C_s), ("B", b_cert, C_b)):
        pick = select_candidate(cert, C, family)
        if pick is None:
            out[proc] = {
                "certified": 0, "coverage_lcb": 0.0, "effective_certified_coverage": 0.0,
                "risk_ucb": None, "holm_raw_pvalue": None,
                "holm_adjusted_pvalue": None, "selected_candidate_index": None,
            }
            continue
        out[proc] = {
            "certified": 1,
            "coverage_lcb": float(C[pick]),
            "effective_certified_coverage": float(C[pick]),
            "risk_ucb": None if proc == "H" else float(arrays[f"{proc}_U"][pick]),
            "holm_raw_pvalue": float(p_raw[pick]) if proc == "H" else None,
            "holm_adjusted_pvalue": float(p_adj[pick]) if proc == "H" else None,
            "selected_candidate_index": int(pick),
        }
    if out["B"]["certified"] and not out["S"]["certified"]:
        raise AssertionError("B-certified cell was not S-certified")
    return out, arrays


def parse_split_rep(semantic_id: str) -> tuple[int, int]:
    m = re.search(r"(?:__split|__pathmnist_split_)(\d+)__seed(\d+)", semantic_id)
    if not m:
        raise ValueError(f"cannot parse split/repetition from {semantic_id}")
    return int(m.group(1)), int(m.group(2))


def load_seed_maps(c400_matrix: Path, path_matrix: Path) -> tuple[dict[str, int], dict[str, dict[str, str]]]:
    primary: dict[str, int] = {}
    meta: dict[str, dict[str, str]] = {}

    def add(row: dict[str, str], seed_field: str) -> None:
        sid = row["semantic_id"]
        if sid in primary:
            raise AssertionError(f"duplicate seed-map semantic_id: {sid}")
        primary[sid] = int(row[seed_field])
        meta[sid] = row

    for row in read_csv(c400_matrix):
        add(row, "seed_primary_audit_draw")
    for row in read_csv(path_matrix):
        if row["backbone"] != "resnext29_8x64d":
            continue
        add(row, "seed_audit")
    return primary, meta


def expected_terminal_path(npz_path: Path, kind: str) -> Path:
    suffix = "_logits.npz" if kind == "officehome" else "_common.npz"
    if not npz_path.name.endswith(suffix):
        raise ValueError(f"unexpected artifact suffix: {npz_path}")
    stem = npz_path.name[: -len(suffix)]
    return npz_path.with_name(stem + ".TERMINAL.json")


def verify_terminal_hash(npz_path: Path, kind: str, digest: str) -> bool:
    terminal = expected_terminal_path(npz_path, kind)
    if not terminal.exists():
        raise FileNotFoundError(terminal)
    obj = json.loads(terminal.read_text(encoding="utf-8"))
    checks = obj.get("checksums", {})
    expected = checks.get(npz_path.name)
    if expected is None:
        # Office-Home terminal records store the same checksum under logits.
        expected = obj.get("logits_sha256") or obj.get("npz_sha256")
    if expected is None:
        raise RuntimeError(f"no terminal checksum for {npz_path}")
    if str(expected) != digest:
        raise RuntimeError(f"terminal checksum mismatch for {npz_path}")
    return True


def derive_primary_positions(seed: int, client_id: int, reservoir_positions: np.ndarray) -> np.ndarray:
    ss = np.random.SeedSequence([int(seed), int(client_id), 0])
    rng = np.random.Generator(np.random.PCG64(ss))
    local = rng.integers(0, len(reservoir_positions), size=len(reservoir_positions), endpoint=False)
    return np.asarray(reservoir_positions, dtype=np.int64)[local]


def bootstrap_condition_effects(
    alpha_rows: list[dict[str, Any]], out_path: Path,
) -> list[dict[str, Any]]:
    by_cell: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in alpha_rows:
        by_cell[row["semantic_id"]][row["procedure"]] = row
    cell_effects: list[dict[str, Any]] = []
    for sid, procs in by_cell.items():
        if set(procs) != {"H", "S", "B"}:
            raise AssertionError(f"incomplete H/S/B cell {sid}")
        split, rep = parse_split_rep(sid)
        base = procs["H"]
        cell_effects.append({
            "semantic_id": sid,
            "dataset": base["dataset"],
            "condition": base["condition"],
            "split": split,
            "repetition": rep,
            "S_minus_B": float(procs["S"]["effective_certified_coverage"]) - float(procs["B"]["effective_certified_coverage"]),
            "H_minus_S": float(procs["H"]["effective_certified_coverage"]) - float(procs["S"]["effective_certified_coverage"]),
            "H_minus_B": float(procs["H"]["effective_certified_coverage"]) - float(procs["B"]["effective_certified_coverage"]),
        })
    write_csv(out_path.parent / "paired_cell_effects_alpha020.csv", cell_effects)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_effects:
        grouped[(row["dataset"], row["condition"])].append(row)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    for dataset, condition in sorted(grouped):
        rows = grouped[(dataset, condition)]
        by_split: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_split[int(row["split"])].append(row)
        split_ids = sorted(by_split)
        for effect_name in ("S_minus_B", "H_minus_S", "H_minus_B"):
            observed = float(np.mean([float(r[effect_name]) for r in rows]))
            boots = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
            for b in range(BOOTSTRAP_REPLICATES):
                sampled_splits = rng.choice(split_ids, size=len(split_ids), replace=True)
                vals: list[float] = []
                for s in sampled_splits:
                    block = by_split[int(s)]
                    chosen = rng.integers(0, len(block), size=len(block))
                    vals.extend(float(block[int(i)][effect_name]) for i in chosen)
                boots[b] = float(np.mean(vals))
            lo, hi = np.quantile(boots, [0.025, 0.975])
            summary.append({
                "dataset": dataset,
                "condition": condition,
                "effect": effect_name,
                "N_cells": len(rows),
                "N_splits": len(split_ids),
                "point_difference": observed,
                "hierarchical_bootstrap_ci_low": float(lo),
                "hierarchical_bootstrap_ci_high": float(hi),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            })
            replicate_rows.extend({
                "dataset": dataset, "condition": condition, "effect": effect_name,
                "replicate": b, "difference": float(v),
            } for b, v in enumerate(boots))
    write_csv(out_path, summary)
    write_csv(out_path.parent / "paired_bootstrap_replicates_alpha020.csv.gz", replicate_rows)
    return summary


def summarize(per_cell: list[dict[str, Any]], output_dir: Path) -> None:
    summary_rows: list[dict[str, Any]] = []
    keys = sorted({(float(r["alpha"]), r["dataset"], r["condition"], r["procedure"]) for r in per_cell})
    for alpha, dataset, condition, proc in keys:
        rows = [r for r in per_cell if float(r["alpha"]) == alpha and r["dataset"] == dataset and r["condition"] == condition and r["procedure"] == proc]
        certified = sum(int(r["certified"]) for r in rows)
        eff = float(np.mean([float(r["effective_certified_coverage"]) for r in rows]))
        cond = (sum(float(r["coverage_lcb"]) for r in rows if int(r["certified"])) / certified) if certified else None
        summary_rows.append({
            "alpha": alpha, "dataset": dataset, "condition": condition,
            "procedure": proc, "N_cells": len(rows), "certified_cells": certified,
            "certification_rate": certified / len(rows), "EffectiveCertCov": eff,
            "CondCertCov": cond,
        })
    write_csv(output_dir / "full_sweep_summary.csv", summary_rows)

    primary = [r for r in per_cell if abs(float(r["alpha"]) - PRIMARY_ALPHA) < 1e-12]
    headline: list[dict[str, Any]] = []
    for dataset in ("cifar10", "cifar100", "officehome", "pathmnist", "ALL"):
        for proc in ("H", "S", "B"):
            rows = [r for r in primary if r["procedure"] == proc and (dataset == "ALL" or r["dataset"] == dataset)]
            certified = sum(int(r["certified"]) for r in rows)
            eff = float(np.mean([float(r["effective_certified_coverage"]) for r in rows]))
            cond = (sum(float(r["coverage_lcb"]) for r in rows if int(r["certified"])) / certified) if certified else None
            headline.append({
                "dataset": dataset, "procedure": proc, "alpha": PRIMARY_ALPHA,
                "delta_r": DELTA_R, "delta_c": DELTA_C, "N_cells": len(rows),
                "certified_cells": certified, "certification_rate": certified / len(rows),
                "EffectiveCertCov": eff, "CondCertCov": cond,
            })
    write_csv(output_dir / "primary_headline_alpha020.csv", headline)

    overlaps: list[dict[str, Any]] = []
    by_sid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in primary:
        by_sid[r["semantic_id"]][r["procedure"]] = r
    for dataset in ("cifar10", "cifar100", "officehome", "pathmnist", "ALL"):
        cells = [v for v in by_sid.values() if dataset == "ALL" or next(iter(v.values()))["dataset"] == dataset]
        for p, q in (("H", "S"), ("H", "B"), ("S", "B")):
            pset = [int(c[p]["certified"]) for c in cells]
            qset = [int(c[q]["certified"]) for c in cells]
            overlaps.append({
                "dataset": dataset, "procedure_1": p, "procedure_2": q,
                "N_cells": len(cells),
                "both": sum(a and b for a, b in zip(pset, qset)),
                "procedure_1_only": sum(a and not b for a, b in zip(pset, qset)),
                "procedure_2_only": sum(b and not a for a, b in zip(pset, qset)),
                "neither": sum((not a) and (not b) for a, b in zip(pset, qset)),
            })
    write_csv(output_dir / "primary_overlaps_alpha020.csv", overlaps)
    bootstrap_condition_effects(primary, output_dir / "paired_hierarchical_bootstrap_alpha020.csv")


def run(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    analysis_rows = read_csv(Path(args.analysis_manifest))
    if len(analysis_rows) != 450:
        raise AssertionError(f"expected 450 cells, found {len(analysis_rows)}")
    if len({r["semantic_id"] for r in analysis_rows}) != 450:
        raise AssertionError("semantic_id roster is not unique")

    semantic_ids = {r["semantic_id"] for r in analysis_rows}
    expected_family_keys = {
        (sid, alpha_key(alpha)) for sid in semantic_ids for alpha in ALPHA_GRID
    }

    raw_pin_rows = read_csv(Path(args.raw_input_pins))
    raw_pin_map: dict[str, dict[str, str]] = {}
    for row in raw_pin_rows:
        sid = row["semantic_id"]
        if sid in raw_pin_map:
            raise AssertionError(f"duplicate raw-input pin semantic_id: {sid}")
        raw_pin_map[sid] = row
    if set(raw_pin_map) != semantic_ids:
        missing = sorted(semantic_ids - set(raw_pin_map))
        extra = sorted(set(raw_pin_map) - semantic_ids)
        raise AssertionError(f"raw-input pin roster mismatch: missing={missing[:3]}, extra={extra[:3]}")

    family_rows = read_csv(Path(args.family_manifest))
    family_map: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seen_family_rows: set[tuple[str, str, int]] = set()
    for row in family_rows:
        group_key = (row["semantic_id"], alpha_key(row["alpha"]))
        row_key = group_key + (int(row["candidate_index"]),)
        if row_key in seen_family_rows:
            raise AssertionError(f"duplicate family row: {row_key}")
        seen_family_rows.add(row_key)
        family_map[group_key].append(row)
    if set(family_map) != expected_family_keys:
        raise AssertionError("family manifest key set does not equal the sealed 450-cell by six-alpha roster")
    for key in family_map:
        family_map[key].sort(key=lambda r: int(r["candidate_index"]))
        if [int(r["candidate_index"]) for r in family_map[key]] != list(range(M_EXPECTED)):
            raise AssertionError(f"bad family ordering for {key}")

    archived = read_csv(Path(args.archived_counts))
    archived_map: dict[tuple[str, str, int, int], tuple[int, int, int]] = {}
    for row in archived:
        key = (
            row["semantic_id"], alpha_key(row["alpha"]),
            int(row["candidate_index"]), int(row["client"]),
        )
        if key in archived_map:
            raise AssertionError(f"duplicate archived-count key: {key}")
        archived_map[key] = (int(row["A"]), int(row["K"]), int(row["n"]))
    expected_archived_keys = {
        (row["semantic_id"], alpha_key(alpha), candidate, client)
        for row in analysis_rows
        for alpha in ALPHA_GRID
        for candidate in range(M_EXPECTED)
        for client in range(int(row["J"]))
    }
    if set(archived_map) != expected_archived_keys:
        raise AssertionError(
            "archived-count key set does not exactly equal the sealed cell/alpha/candidate/client roster"
        )
    seed_map, seed_meta = load_seed_maps(Path(args.c400_matrix), Path(args.path_matrix))
    if set(seed_map) != semantic_ids:
        raise AssertionError("seed-map roster does not exactly equal the sealed 450-cell roster")

    per_cell: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    accounting_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    reproduction_mismatches: list[dict[str, Any]] = []
    total_expected_checks = 0

    for cell_idx, meta in enumerate(sorted(analysis_rows, key=lambda r: r["semantic_id"]), start=1):
        sid = meta["semantic_id"]
        kind = meta["kind"]
        path = repo_root / meta["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if int(meta["bytes"]) != path.stat().st_size:
            raise RuntimeError(f"input size drift for {sid}")
        pin = raw_pin_map[sid]
        if pin["kind"] != kind or pin["path"] != meta["path"]:
            raise RuntimeError(f"raw-input pin identity mismatch for {sid}")
        if int(pin["bytes"]) != path.stat().st_size or pin["sha256"] != digest:
            raise RuntimeError(f"raw-input byte pin mismatch for {sid}")
        terminal = expected_terminal_path(path, kind)
        terminal_rel = str(terminal.relative_to(repo_root))
        terminal_digest = sha256_file(terminal)
        if (
            pin["terminal_path"] != terminal_rel
            or int(pin["terminal_bytes"]) != terminal.stat().st_size
            or pin["terminal_sha256"] != terminal_digest
        ):
            raise RuntimeError(f"terminal sidecar byte pin mismatch for {sid}")
        verify_terminal_hash(path, kind, digest)
        cell = load_cell(path, kind)
        J = int(meta["J"])
        client = cell["cert_client"]
        source = cell["cert_source_id"]
        if len(source) != len(set(source.tolist())):
            raise RuntimeError(f"certification reservoir source IDs are not unique for {sid}")
        if set(np.unique(client).tolist()) != set(range(J)):
            raise RuntimeError(f"declared clients missing from reservoir for {sid}")
        seed = seed_map.get(sid)
        if seed is None:
            raise RuntimeError(f"no frozen primary audit seed for {sid}")

        split, repetition = parse_split_rep(sid)
        input_rows.append({
            "semantic_id": sid, "kind": kind, "dataset": meta["dataset"],
            "condition": meta["condition"], "path": meta["path"],
            "bytes": path.stat().st_size, "sha256": digest,
            "terminal_path": terminal_rel, "terminal_bytes": terminal.stat().st_size,
            "terminal_sha256": terminal_digest,
            "terminal_checksum_verified": 1,
        })
        seed_rows.append({
            "semantic_id": sid, "dataset": meta["dataset"], "condition": meta["condition"],
            "split": split, "repetition": repetition, "frozen_primary_audit_seed": seed,
            "seed_source": "confirmatory_400r_matrix.seed_primary_audit_draw" if kind != "pathmnist_resnext" else "medmnist_matrix.seed_audit",
            "embedded_artifact_audit_seed": cell["embedded_audit_seed"],
            "rng_contract": RNG_CONTRACT,
        })

        err = error_marks(cell["cert_pred"], cell["cert_y"])
        reservoir_positions = [np.flatnonzero(client == j) for j in range(J)]
        n = np.asarray([len(x) for x in reservoir_positions], dtype=int)
        draws = [derive_primary_positions(seed, j, reservoir_positions[j]) for j in range(J)]
        for j in range(J):
            local_map = {int(pos): i for i, pos in enumerate(reservoir_positions[j].tolist())}
            local_draw = np.asarray([local_map[int(pos)] for pos in draws[j]], dtype="<i8")
            uniq, multiplicity = np.unique(local_draw, return_counts=True)
            accounting_rows.append({
                "semantic_id": sid, "dataset": meta["dataset"], "condition": meta["condition"],
                "client": j, "reservoir_n": int(n[j]), "draw_n": int(n[j]),
                "unique_draw_indices": int(len(uniq)),
                "duplication_rate": 1.0 - len(uniq) / int(n[j]),
                "max_multiplicity": int(multiplicity.max()),
                "draw_index_sha256": hashlib.sha256(local_draw.tobytes()).hexdigest(),
            })

        for alpha in ALPHA_GRID:
            akey = alpha_key(alpha)
            family = family_map.get((sid, akey))
            if family is None or len(family) != M_EXPECTED:
                raise RuntimeError(f"missing frozen family for {sid} alpha={alpha}")
            accept = np.zeros((M_EXPECTED, len(client)), dtype=bool)
            for m, cand in enumerate(family):
                threshold = float(cand["threshold"])
                slot = cand["slot"]
                accept[m] = (int(cand["threshold_feasible"]) == 1) & (cell["cert_scores"][slot] >= threshold)

            # Verify exact reconstruction of the archived as-is count tensor.
            for m in range(M_EXPECTED):
                for j in range(J):
                    mask = client == j
                    A0 = int((accept[m] & mask).sum())
                    K0 = int((accept[m] & err & mask).sum())
                    expected = archived_map.get((sid, akey, m, j))
                    total_expected_checks += 1
                    if expected != (A0, K0, int(n[j])):
                        reproduction_mismatches.append({
                            "semantic_id": sid, "alpha": alpha, "candidate_index": m,
                            "client": j, "expected": expected, "observed": [A0, K0, int(n[j])],
                        })

            A = np.zeros((M_EXPECTED, J), dtype=int)
            K = np.zeros((M_EXPECTED, J), dtype=int)
            for m in range(M_EXPECTED):
                for j in range(J):
                    pos = draws[j]
                    marks = accept[m, pos]
                    A[m, j] = int(marks.sum())
                    K[m, j] = int((marks & err[pos]).sum())
            if np.any(K < 0) or np.any(K > A) or np.any(A > n[None, :]):
                raise AssertionError(f"invalid WR counts for {sid} alpha={alpha}")

            thash = tensor_hash(A, K, n)
            results, arrays = procedure_results(A, K, n, family, alpha)
            for m, cand in enumerate(family):
                for j in range(J):
                    count_rows.append({
                        "semantic_id": sid, "dataset": meta["dataset"], "condition": meta["condition"],
                        "alpha": alpha, "candidate_index": m, "score": cand["score"],
                        "slot": cand["slot"], "gamma": float(cand["gamma"]),
                        "threshold": float(cand["threshold"]),
                        "proposal_feasible": int(cand["proposal_feasible"]),
                        "selector_sha256": cand["selector_sha256"], "client": j,
                        "A": int(A[m, j]), "K": int(K[m, j]), "n": int(n[j]),
                        "counts_tensor_sha256": thash,
                    })
            for proc in ("H", "S", "B"):
                result = results[proc]
                pick = result["selected_candidate_index"]
                cand = family[pick] if pick is not None else None
                per_cell.append({
                    "semantic_id": sid, "dataset": meta["dataset"], "condition": meta["condition"],
                    "arm": meta.get("arm", ""), "split": split, "repetition": repetition,
                    "alpha": alpha, "delta_r": DELTA_R, "delta_c": DELTA_C,
                    "J": J, "M": M_EXPECTED, "procedure": proc,
                    "certified": result["certified"],
                    "coverage_lcb": result["coverage_lcb"],
                    "effective_certified_coverage": result["effective_certified_coverage"],
                    "risk_ucb": result["risk_ucb"],
                    "holm_raw_pvalue": result["holm_raw_pvalue"],
                    "holm_adjusted_pvalue": result["holm_adjusted_pvalue"],
                    "selected_candidate_index": pick,
                    "selected_score": cand["score"] if cand else None,
                    "selected_slot": cand["slot"] if cand else None,
                    "selected_gamma": float(cand["gamma"]) if cand else None,
                    "selected_threshold": float(cand["threshold"]) if cand else None,
                    "selected_selector_sha256": cand["selector_sha256"] if cand else None,
                    "counts_tensor_sha256": thash,
                    "frozen_primary_audit_seed": seed,
                    "audit_draw_law": "iid uniform with replacement within client certification reservoir",
                })
        print(f"[{cell_idx:03d}/450] {sid}", flush=True)

    if reproduction_mismatches:
        write_csv(output_dir / "archive_reproduction_mismatches.csv", reproduction_mismatches)
        raise AssertionError(f"archive reproduction failed: {len(reproduction_mismatches)} mismatches")

    write_csv(output_dir / "input_artifact_manifest.csv", input_rows)
    write_csv(output_dir / "primary_seed_manifest.csv", seed_rows)
    write_csv(output_dir / "primary_reservoir_accounting.csv", accounting_rows)
    write_csv(output_dir / "primary_candidate_counts.csv.gz", count_rows)
    write_csv(output_dir / "primary_per_cell_procedures.csv", per_cell)
    summarize(per_cell, output_dir)

    primary_rows = [r for r in per_cell if abs(float(r["alpha"]) - PRIMARY_ALPHA) < 1e-12]
    by_sid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in primary_rows:
        by_sid[row["semantic_id"]][row["procedure"]] = row
    b_not_s = [sid for sid, v in by_sid.items() if int(v["B"]["certified"]) and not int(v["S"]["certified"])]
    shared_tensor_failures = [sid for sid, v in by_sid.items() if len({v[p]["counts_tensor_sha256"] for p in ("H", "S", "B")}) != 1]
    validation = {
        "status": "PASS" if not b_not_s and not shared_tensor_failures else "FAIL",
        "contract_id": CONTRACT_ID,
        "cells": len(analysis_rows),
        "alphas": list(ALPHA_GRID),
        "primary_alpha": PRIMARY_ALPHA,
        "procedures": ["H", "S", "B"],
        "archive_count_checks": total_expected_checks,
        "raw_input_pin_checks": len(raw_pin_map),
        "terminal_sidecar_pin_checks": len(raw_pin_map),
        "archive_reproduction_mismatches": 0,
        "B_certified_not_S": b_not_s,
        "shared_HSB_tensor_failures": shared_tensor_failures,
        "test_or_evaluation_fold_accessed": False,
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "rng_contract": RNG_CONTRACT,
    }
    (output_dir / "VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if validation["status"] != "PASS":
        raise AssertionError("final validation failed")

    files = sorted(p for p in output_dir.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    with (output_dir / "SHA256SUMS").open("w", encoding="utf-8") as f:
        for p in files:
            f.write(f"{sha256_file(p)}  {p.name}\n")


def self_test() -> None:
    assert parse_split_rep("cifar10_proser_fedavg__split03__seed2__d0.5") == (3, 2)
    assert parse_split_rep("resnext29_8x64d__pathmnist_split_03__seed2__d0.5") == (3, 2)
    family = []
    for m in range(12):
        family.append({
            "candidate_index": str(m), "slot": ("native", "energy", "margin")[m // 4],
            "score": ("msp", "energy", "margin")[m // 4],
            "gamma": str((0.3, 0.5, 0.7, 1.0)[m % 4]), "proposal_feasible": "1",
        })
    A = np.full((12, 3), 40, dtype=int)
    K = np.zeros((12, 3), dtype=int)
    n = np.full(3, 100, dtype=int)
    out, arr = procedure_results(A, K, n, family, 0.20)
    assert out["H"]["certified"] == 1
    assert out["S"]["certified"] == 1
    assert out["B"]["certified"] == 1
    assert np.all(arr["S_U"] <= arr["B_U"])
    assert np.all(arr["S_C"] >= arr["B_C"])
    A[:, 0] = 0
    K[:, 0] = 0
    out, _ = procedure_results(A, K, n, family, 0.20)
    assert all(out[p]["certified"] == 0 for p in ("H", "S", "B"))
    pos = np.arange(100, dtype=np.int64)
    a = derive_primary_positions(1234, 2, pos)
    b = derive_primary_positions(1234, 2, pos)
    c = derive_primary_positions(1234, 3, pos)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and len(a) == len(pos)
    print("SELF_TEST_PASS")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--repo-root")
    p.add_argument("--output-dir")
    p.add_argument("--analysis-manifest")
    p.add_argument("--raw-input-pins")
    p.add_argument("--family-manifest")
    p.add_argument("--archived-counts")
    p.add_argument("--c400-matrix")
    p.add_argument("--path-matrix")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return
    required = ("repo_root", "output_dir", "analysis_manifest", "raw_input_pins", "family_manifest", "archived_counts", "c400_matrix", "path_matrix")
    missing = [x for x in required if getattr(args, x) is None]
    if missing:
        raise SystemExit(f"missing arguments: {', '.join(missing)}")
    run(args)


if __name__ == "__main__":
    main()
