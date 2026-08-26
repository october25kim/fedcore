#!/usr/bin/env python3
"""Fail-closed verifier for the Fed-CORE v0.2.0 numerical release."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

try:
    import numpy as np
    from scipy.stats import beta, binom
except Exception as exc:  # pragma: no cover - environment failure is intentional
    raise SystemExit(f"FAIL_CLOSED: NumPy and SciPy are required: {exc}")


ROOT = Path(__file__).resolve().parents[1]
SHA_MANIFEST = ROOT / "SHA256SUMS"
TOL = 2e-11
ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
M = 12
DELTA_R = 0.05
DELTA_C = 0.05
HEADLINE_ALPHA = 0.20
COUNT_ROWS = 154_800
CELL_COUNT = 450
FAMILY_ROWS = CELL_COUNT * len(ALPHAS) * M
FAMILY_HASH_ROWS = CELL_COUNT * len(ALPHAS)
RAW_COUNT_SHA256 = "3074c2d8039b64707bbf27524ebd8baa60548825758c89c77454efc9f0cb3f45"
SELECTOR_HASH_SALT = "officehome-selector-rescue-v1"
KAPPA = 0.01
N_GRID = 300

EXPECTED_DATASET_CELLS = {
    "cifar10": 150,
    "cifar100": 150,
    "officehome": 100,
    "pathmnist": 50,
}
EXPECTED_CERTIFIED = {"H": 198, "S": 198, "B": 148}
EXPECTED_EFFECTIVE = {
    "H": 0.10049009894485754,
    "S": 0.09694642128834831,
    "B": 0.07182488653286874,
}
EXPECTED_DATASET_CERTIFIED = {
    "cifar10": {"H": 66, "S": 66, "B": 54},
    "cifar100": {"H": 11, "S": 11, "B": 1},
    "officehome": {"H": 88, "S": 88, "B": 69},
    "pathmnist": {"H": 33, "S": 33, "B": 24},
}
EXPECTED_DATASET_EFFECTIVE = {
    "cifar10": {
        "H": 0.12932656935056092,
        "S": 0.12411285837161237,
        "B": 0.10124230218667599,
    },
    "cifar100": {
        "H": 0.006988744617308427,
        "S": 0.006988744617308427,
        "B": 0.0005750932761137753,
    },
    "officehome": {
        "H": 0.11917599700677936,
        "S": 0.11449311830334913,
        "B": 0.08197487529326512,
    },
    "pathmnist": {
        "H": 0.257112954586551,
        "S": 0.25022674602167394,
        "B": 0.17702204182091916,
    },
}


class ReleaseError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ReleaseError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def close(actual: float, expected: float, message: str, tol: float = TOL) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
        fail(f"{message}: got {actual!r}, expected {expected!r}")


def optional_float(value: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def optional_int(value: str) -> int | None:
    number = optional_float(value)
    return None if number is None else int(number)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_uncompressed_gzip(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> Any:
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def csv_rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_hash_manifest() -> dict[str, str]:
    require(SHA_MANIFEST.is_file(), "SHA256SUMS is missing")
    listed: dict[str, str] = {}
    for line_number, raw in enumerate(SHA_MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        require(raw.strip() != "", f"blank SHA256SUMS line {line_number}")
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\x00]+)", raw)
        require(match is not None, f"malformed SHA256SUMS line {line_number}")
        digest, name = match.groups()
        posix = PurePosixPath(name)
        require(not posix.is_absolute(), f"absolute manifest path: {name}")
        require(".." not in posix.parts, f"parent traversal in manifest: {name}")
        require(name not in listed, f"duplicate manifest path: {name}")
        listed[name] = digest

    actual: set[str] = set()
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            fail(f"symlink is not permitted: {path.relative_to(ROOT)}")
        if path.is_file() and path != SHA_MANIFEST:
            actual.add(path.relative_to(ROOT).as_posix())
    require(set(listed) == actual,
            f"manifest file-set mismatch: missing={sorted(actual-set(listed))}, "
            f"extra={sorted(set(listed)-actual)}")
    for name, expected in listed.items():
        actual_digest = sha256_file(ROOT / name)
        require(actual_digest == expected, f"hash mismatch: {name}")
    return listed


def check_release_metadata(manifest_hashes: dict[str, str]) -> None:
    release = load_json("RELEASE.json")
    require(release.get("release_id") == "fedcore-v0.2.0-theorem-aligned",
            "unexpected release_id")
    require(release.get("status") == "versioned-release", "unexpected release status")
    require(release.get("source_binding") == {
        "tag": "v0.2.0", "resolution": "git rev-list -n 1 v0.2.0"
    }, "unexpected source tag binding")
    experimental = release.get("experimental_provenance", {})
    require(experimental.get("analysis_checkout_head") ==
            "ee3c646d6190c86d96c3bcf0c9ddbe260a09a8e6",
            "analysis checkout head mismatch")
    require(experimental.get("analysis_checkout_clean") is False and
            experimental.get("analysis_checkout_status_entries") == 6165 and
            experimental.get("source_closure_claimed") is False,
            "analysis checkout scope was overstated")
    require(experimental.get("governing_count_tensor_uncompressed_sha256") ==
            RAW_COUNT_SHA256, "governing count tensor binding mismatch")
    headline = release.get("headline_contract", {})
    require(headline.get("cells") == CELL_COUNT, "RELEASE cell count mismatch")
    require(headline.get("count_rows") == COUNT_ROWS, "RELEASE count-row mismatch")
    require(headline.get("certified_cells") == EXPECTED_CERTIFIED,
            "RELEASE certified totals mismatch")
    for procedure, expected in EXPECTED_EFFECTIVE.items():
        close(headline["effective_acceptance"][procedure], expected,
              f"RELEASE effective acceptance {procedure}")

    provenance = load_json("PROVENANCE_MAP.json")
    require(provenance.get("release_id") == release["release_id"],
            "provenance release_id mismatch")
    original = provenance.get("original_analysis", {})
    require(original.get("checkout_head") == experimental["analysis_checkout_head"] and
            original.get("checkout_clean") is False and
            original.get("status_entry_count") == 6165,
            "original-analysis provenance mismatch")
    entries = provenance.get("entries")
    require(isinstance(entries, list), "provenance entries must be a list")
    paths = [entry.get("release_path") for entry in entries]
    require(all(isinstance(path, str) for path in paths), "invalid provenance path")
    require(len(paths) == len(set(paths)), "duplicate provenance path")
    governed = {
        path.relative_to(ROOT).as_posix()
        for prefix in (ROOT / "artifacts", ROOT / "reference")
        for path in prefix.rglob("*") if path.is_file()
    }
    require(set(paths) == governed,
            f"provenance file-set mismatch: missing={sorted(governed-set(paths))}, "
            f"extra={sorted(set(paths)-governed)}")
    for entry in entries:
        path = entry["release_path"]
        digest = manifest_hashes[path]
        if "origin_sha256" in entry:
            require(entry["origin_sha256"] == digest,
                    f"verbatim provenance hash mismatch: {path}")
        if "release_sha256" in entry:
            require(entry["release_sha256"] == digest,
                    f"derived provenance hash mismatch: {path}")

    contract = load_json("contract/theorem_contract.json")
    point = contract.get("headline_operating_point", {})
    require(point == {"alpha": 0.2, "delta_r": 0.05, "delta_c": 0.05, "M": 12},
            "headline theorem contract mismatch")
    full = contract.get("full_simplex", {})
    require(full["single_fixed_selector"]["client_divisor"] is False,
            "single-selector contract reintroduced a client divisor")
    require(full["simple_simultaneous_family"]["client_divisor"] is False,
            "simple-family contract reintroduced a client divisor")
    require(full["holm_iut_family"]["client_divisor"] is False,
            "Holm/IUT contract reintroduced a client divisor")
    require(full["holm_iut_family"]["reports_numeric_risk_ucb"] is False,
            "Holm/IUT must not report a numerical risk UCB")
    bounded = contract.get("strict_bounded_mixture", {})
    require(bounded.get("single_selector_risk_allocation") ==
            "delta_r/(3S) for each of the three simultaneous endpoint families",
            "bounded-mixture single risk allocation mismatch")
    require(bounded.get("single_selector_coverage_allocation") == "delta_c/S",
            "bounded-mixture single coverage allocation mismatch")
    require(bounded.get("simple_family_risk_allocation") ==
            "delta_r/(3SM) for each selector, stratum, and endpoint family",
            "bounded-mixture family risk allocation mismatch")
    require(bounded.get("simple_family_coverage_allocation") == "delta_c/(SM)",
            "bounded-mixture family coverage allocation mismatch")
    require(bounded.get("solver") == "verified conservative positive-denominator robust solver",
            "bounded-mixture solver contract mismatch")
    require(bounded.get("solver_failure_policy") == "fail closed",
            "bounded-mixture solver failure policy mismatch")


FAMILY_FIELDS = (
    "semantic_id", "alpha", "candidate_index", "score", "slot", "gamma",
    "threshold", "threshold_feasible", "proposal_feasible",
)
FAMILY_MANIFEST_FIELDS = FAMILY_FIELDS + ("selector_sha256",)


def cp_lower(k: np.ndarray, n: np.ndarray, eps: float) -> np.ndarray:
    kk, nn = np.broadcast_arrays(np.asarray(k, dtype=int), np.asarray(n, dtype=int))
    out = np.zeros(kk.shape, dtype=float)
    mask = (nn > 0) & (kk > 0) & (eps > 0)
    out[mask] = beta.ppf(float(eps), kk[mask], nn[mask] - kk[mask] + 1)
    return out


def cp_upper(k: np.ndarray, n: np.ndarray, eps: float) -> np.ndarray:
    kk, nn = np.broadcast_arrays(np.asarray(k, dtype=int), np.asarray(n, dtype=int))
    out = np.ones(kk.shape, dtype=float)
    mask = (nn > 0) & (kk < nn) & (eps > 0)
    out[mask] = beta.isf(float(eps), kk[mask] + 1, nn[mask] - kk[mask])
    return out


def holm_reject(pvalues: np.ndarray, fwer: float) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    rejected = np.zeros(len(values), dtype=bool)
    order = np.argsort(values, kind="stable")
    for rank, index in enumerate(order):
        if float(values[index]) <= float(fwer) / (len(values) - rank):
            rejected[index] = True
        else:
            break
    return rejected


def holm_adjusted(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.ones(len(values), dtype=float)
    order = np.argsort(values, kind="stable")
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def family_tie_name(slot: str) -> str:
    return "msp" if slot == "native" else slot


def selector_hash(slot: str, gamma: float, alpha: float) -> str:
    score = family_tie_name(slot)
    payload = (
        f"{SELECTOR_HASH_SALT}|score={score}|gamma={float(gamma):.6f}"
        f"|alpha={float(alpha):.6f}|kappa={KAPPA:.6f}"
        f"|ngrid={N_GRID}|buffer=gamma*alpha"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def select_candidate(candidates: dict[int, dict[str, Any]], mask: np.ndarray,
                     coverage: np.ndarray, alpha: float) -> int | None:
    eligible = np.flatnonzero(np.asarray(mask, dtype=bool)).tolist()
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda index: (
            -float(coverage[index]),
            float(candidates[index]["meta"]["gamma"]),
            family_tie_name(candidates[index]["meta"]["slot"]),
            selector_hash(candidates[index]["meta"]["slot"],
                          float(candidates[index]["meta"]["gamma"]), alpha),
        ),
    )


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_selector_hash(raw_meta: dict[str, str]) -> str:
    payload = {
        "semantic_id": raw_meta["semantic_id"],
        "alpha": raw_meta["alpha"],
        "candidate_index": int(raw_meta["candidate_index"]),
        "score": raw_meta["score"],
        "slot": raw_meta["slot"],
        "gamma": raw_meta["gamma"],
        "threshold": raw_meta["threshold"],
        "threshold_feasible": int(raw_meta["threshold_feasible"]),
        "proposal_feasible": int(raw_meta["proposal_feasible"]),
    }
    return canonical_hash(payload)


def canonical_family_hash(candidates: dict[int, dict[str, Any]]) -> str:
    ordered_selector_hashes = [
        canonical_selector_hash(candidates[candidate_index]["raw_meta"])
        for candidate_index in range(M)
    ]
    return canonical_hash(ordered_selector_hashes)


def load_and_verify_counts() -> tuple[dict[tuple[str, float], dict[str, Any]], set[str]]:
    source = ROOT / "artifacts/counts/candidate_counts_A_K_n.csv.gz"
    require(sha256_uncompressed_gzip(source) == RAW_COUNT_SHA256,
            "uncompressed count tensor hash mismatch")
    required = set(FAMILY_FIELDS) | {
        "dataset", "condition", "d", "arm", "client", "A", "K", "n",
        "unique_source_n", "unique_accepted_source_n", "client_risk_pvalue",
        "candidate_iut_pvalue", "holm_risk_reject",
        "legacy_coverage_lcb_client", "theorem_coverage_lcb_client",
        "legacy_candidate_coverage_lcb", "theorem_candidate_coverage_lcb",
    }
    groups: dict[tuple[str, float], dict[str, Any]] = {}
    row_count = 0
    semantic_ids: set[str] = set()
    with gzip.open(source, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and required <= set(reader.fieldnames),
                "count tensor schema mismatch")
        for row in reader:
            row_count += 1
            semantic_id = row["semantic_id"]
            alpha = float(row["alpha"])
            require(any(math.isclose(alpha, value, abs_tol=1e-12) for value in ALPHAS),
                    f"unexpected alpha for {semantic_id}: {alpha}")
            key = (semantic_id, alpha)
            group = groups.setdefault(key, {
                "semantic_id": semantic_id,
                "alpha_raw": row["alpha"],
                "dataset": row["dataset"],
                "condition": row["condition"],
                "d": row["d"],
                "arm": row["arm"],
                "candidates": {},
            })
            for field in ("alpha_raw", "dataset", "condition", "d", "arm"):
                current = row["alpha"] if field == "alpha_raw" else row[field]
                require(group[field] == current,
                        f"cell metadata mismatch for {key}: {field}")
            candidate_index = int(row["candidate_index"])
            require(0 <= candidate_index < M, f"invalid candidate index for {key}")
            raw_meta = {field: row[field] for field in FAMILY_FIELDS}
            meta = {
                "semantic_id": semantic_id,
                "alpha": row["alpha"],
                "candidate_index": row["candidate_index"],
                "score": row["score"],
                "slot": row["slot"],
                "gamma": row["gamma"],
                "threshold": row["threshold"],
                "threshold_feasible": row["threshold_feasible"],
                "proposal_feasible": row["proposal_feasible"],
            }
            candidate = group["candidates"].setdefault(candidate_index, {
                "raw_meta": raw_meta,
                "meta": meta,
                "clients": {},
            })
            require(candidate["raw_meta"] == raw_meta,
                    f"selector metadata differs across client rows for {key}, candidate {candidate_index}")
            client = int(row["client"])
            require(client not in candidate["clients"],
                    f"duplicate client row for {key}, candidate {candidate_index}, client {client}")
            A, K, n = int(row["A"]), int(row["K"]), int(row["n"])
            unique_n = int(row["unique_source_n"])
            unique_A = int(row["unique_accepted_source_n"])
            require(0 <= K <= A <= n and n > 0,
                    f"count invariant failed for {key}, candidate {candidate_index}, client {client}")
            require(0 <= unique_A <= A and 0 < unique_n <= n,
                    f"uniqueness invariant failed for {key}, candidate {candidate_index}, client {client}")
            candidate["clients"][client] = {
                "A": A, "K": K, "n": n,
                "client_p": float(row["client_risk_pvalue"]),
                "candidate_p": float(row["candidate_iut_pvalue"]),
                "holm": int(row["holm_risk_reject"]),
                "old_l": float(row["legacy_coverage_lcb_client"]),
                "new_l": float(row["theorem_coverage_lcb_client"]),
                "old_c": float(row["legacy_candidate_coverage_lcb"]),
                "new_c": float(row["theorem_candidate_coverage_lcb"]),
            }
            semantic_ids.add(semantic_id)

    require(row_count == COUNT_ROWS, f"count tensor rows: {row_count}")
    require(len(semantic_ids) == CELL_COUNT, f"count tensor cells: {len(semantic_ids)}")
    require(len(groups) == CELL_COUNT * len(ALPHAS), f"count tensor cell-alpha groups: {len(groups)}")
    dataset_ids: dict[str, set[str]] = defaultdict(set)
    for group in groups.values():
        dataset_ids[group["dataset"]].add(group["semantic_id"])
    require({key: len(value) for key, value in dataset_ids.items()} == EXPECTED_DATASET_CELLS,
            "dataset cell counts mismatch")

    expected_slots = ("native",) * 4 + ("energy",) * 4 + ("margin",) * 4
    expected_gammas = (0.3, 0.5, 0.7, 1.0) * 3
    for key, group in groups.items():
        candidates = group["candidates"]
        require(set(candidates) == set(range(M)), f"incomplete family for {key}")
        expected_J = 4 if group["dataset"] == "officehome" else 5
        reference_n: tuple[int, ...] | None = None
        for candidate_index in range(M):
            candidate = candidates[candidate_index]
            require(candidate["meta"]["slot"] == expected_slots[candidate_index],
                    f"slot order mismatch for {key}, candidate {candidate_index}")
            close(float(candidate["meta"]["gamma"]), expected_gammas[candidate_index],
                  f"gamma order mismatch for {key}, candidate {candidate_index}", 1e-12)
            feasible = int(candidate["meta"]["threshold_feasible"])
            threshold = float(candidate["meta"]["threshold"])
            require(feasible in (0, 1), f"invalid threshold feasibility for {key}")
            require((feasible == 1 and math.isfinite(threshold)) or
                    (feasible == 0 and math.isinf(threshold) and threshold > 0),
                    f"threshold and feasibility disagree for {key}, candidate {candidate_index}")
            require(int(candidate["meta"]["proposal_feasible"]) in (0, 1),
                    f"invalid proposal feasibility for {key}")
            clients = candidate["clients"]
            require(set(clients) == set(range(expected_J)),
                    f"client support mismatch for {key}, candidate {candidate_index}")
            n_vector = tuple(clients[j]["n"] for j in range(expected_J))
            if reference_n is None:
                reference_n = n_vector
            require(n_vector == reference_n, f"audit sizes differ by candidate for {key}")
    return groups, semantic_ids


def verify_family_freeze(groups: dict[tuple[str, float], dict[str, Any]]) -> None:
    manifest_path = ROOT / "artifacts/counts/family_manifest.csv.gz"
    expected_rows: list[dict[str, str]] = []
    expected_hashes: dict[tuple[str, str], str] = {}
    ordered_keys = sorted(groups, key=lambda key: (key[0], key[1]))
    for key in ordered_keys:
        group = groups[key]
        for candidate_index in range(M):
            raw_meta = group["candidates"][candidate_index]["raw_meta"]
            expected_rows.append({
                **raw_meta,
                "selector_sha256": canonical_selector_hash(raw_meta),
            })
        expected_hashes[(key[0], group["alpha_raw"])] = canonical_family_hash(group["candidates"])

    with gzip.open(manifest_path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == FAMILY_MANIFEST_FIELDS,
                "family manifest schema mismatch")
        observed_rows = list(reader)
    require(len(observed_rows) == FAMILY_ROWS, f"family manifest rows: {len(observed_rows)}")
    require(observed_rows == expected_rows, "family manifest differs from count-tensor deduplication")

    rows = csv_rows("artifacts/counts/family_hashes.csv")
    require(len(rows) == FAMILY_HASH_ROWS, f"family hash rows: {len(rows)}")
    observed_hashes: dict[tuple[str, str], str] = {}
    for row in rows:
        require(int(row["M"]) == M, "family hash M mismatch")
        key = (row["semantic_id"], row["alpha"])
        require(key not in observed_hashes, f"duplicate family hash row: {key}")
        observed_hashes[key] = row["ordered_family_sha256"]
    require(observed_hashes == expected_hashes, "ordered family hash mismatch")


def compute_decisions(groups: dict[tuple[str, float], dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
    results: dict[tuple[str, float], dict[str, Any]] = {}
    for key, group in groups.items():
        candidates = group["candidates"]
        J = 4 if group["dataset"] == "officehome" else 5
        A = np.asarray([[candidates[m]["clients"][j]["A"] for j in range(J)] for m in range(M)])
        K = np.asarray([[candidates[m]["clients"][j]["K"] for j in range(J)] for m in range(M)])
        n = np.asarray([candidates[0]["clients"][j]["n"] for j in range(J)])
        p_client = np.ones((M, J), dtype=float)
        positive = A > 0
        p_client[positive] = binom.cdf(K[positive], A[positive], key[1])
        p_candidate = p_client.max(axis=1)
        rejected = holm_reject(p_candidate, DELTA_R)
        adjusted = holm_adjusted(p_candidate)
        old_l = cp_lower(A, n[None, :], DELTA_C / (M * J))
        new_l = cp_lower(A, n[None, :], DELTA_C / M)
        old_c = old_l.min(axis=1)
        new_c = new_l.min(axis=1)
        simple_ucb = cp_upper(K, A, DELTA_R / M).max(axis=1)
        legacy_ucb = cp_upper(K, A, DELTA_R / (M * J)).max(axis=1)
        masks = {
            "H": rejected & (new_c > 0.0),
            "S": (simple_ucb <= key[1]) & (new_c > 0.0),
            "B": (legacy_ucb <= key[1]) & (old_c > 0.0),
            "legacy_H": rejected & (old_c > 0.0),
        }
        coverage = {"H": new_c, "S": new_c, "B": old_c, "legacy_H": old_c}
        selections = {
            name: select_candidate(candidates, mask, coverage[name], key[1])
            for name, mask in masks.items()
        }

        for m in range(M):
            for j in range(J):
                embedded = candidates[m]["clients"][j]
                close(embedded["client_p"], p_client[m, j], f"client p-value {key}/{m}/{j}")
                close(embedded["candidate_p"], p_candidate[m], f"candidate p-value {key}/{m}")
                close(embedded["old_l"], old_l[m, j], f"legacy coverage endpoint {key}/{m}/{j}")
                close(embedded["new_l"], new_l[m, j], f"theorem coverage endpoint {key}/{m}/{j}")
                close(embedded["old_c"], old_c[m], f"legacy candidate coverage {key}/{m}")
                close(embedded["new_c"], new_c[m], f"theorem candidate coverage {key}/{m}")
                require(embedded["holm"] == int(rejected[m]), f"Holm flag mismatch {key}/{m}")

        procedure_results: dict[str, dict[str, Any]] = {}
        for name in ("H", "S", "B", "legacy_H"):
            selected = selections[name]
            procedure_results[name] = {
                "certified": int(selected is not None),
                "selected": selected,
                "coverage": 0.0 if selected is None else float(coverage[name][selected]),
                "risk_ucb": None if name in ("H", "legacy_H") or selected is None else
                    float((simple_ucb if name == "S" else legacy_ucb)[selected]),
                "raw_p": None if selected is None else float(p_candidate[selected]),
                "adjusted_p": None if selected is None else float(adjusted[selected]),
                "candidate_count": int(masks[name].sum()),
            }
        results[key] = {
            "dataset": group["dataset"],
            "condition": group["condition"],
            "d": group["d"],
            "arm": group["arm"],
            "J": J,
            "candidates": candidates,
            "rejected_count": int(rejected.sum()),
            "procedures": procedure_results,
        }
    return results


def compare_recertification(results: dict[tuple[str, float], dict[str, Any]]) -> None:
    rows = csv_rows("artifacts/counts/recertification_per_cell.csv")
    require(len(rows) == CELL_COUNT * len(ALPHAS), f"recertification rows: {len(rows)}")
    seen: set[tuple[str, float]] = set()
    for row in rows:
        key = (row["semantic_id"], float(row["alpha"]))
        require(key in results and key not in seen, f"recertification key mismatch: {key}")
        seen.add(key)
        result = results[key]
        require(int(row["J"]) == result["J"] and int(row["M"]) == M,
                f"recertification J/M mismatch: {key}")
        require(row["coverage_allocation_change"] == "delta_c/(M J) -> delta_c/M",
                f"recertification allocation label mismatch: {key}")
        require(int(row["holm_n_risk_rejected"]) == result["rejected_count"],
                f"recertification Holm count mismatch: {key}")
        for prefix, name in (("legacy", "legacy_H"), ("theorem", "H")):
            expected = result["procedures"][name]
            require(int(row[f"{prefix}_certified"]) == expected["certified"],
                    f"recertification {prefix} decision mismatch: {key}")
            require(optional_int(row[f"{prefix}_selected_index"]) == expected["selected"],
                    f"recertification {prefix} selection mismatch: {key}")
            close(float(row[f"{prefix}_coverage_lcb"]), expected["coverage"],
                  f"recertification {prefix} coverage mismatch: {key}")
            require(int(row[f"{prefix}_n_certified_candidates"]) == expected["candidate_count"],
                    f"recertification {prefix} candidate count mismatch: {key}")
        expected_transition = (
            f"{result['procedures']['legacy_H']['certified']}->"
            f"{result['procedures']['H']['certified']}"
        )
        require(row["certification_transition"] == expected_transition,
                f"recertification transition mismatch: {key}")
        changed = int(result["procedures"]["legacy_H"]["selected"] !=
                      result["procedures"]["H"]["selected"])
        require(int(row["selection_changed"]) == changed,
                f"recertification selection-change mismatch: {key}")
    require(seen == set(results), "recertification coverage is incomplete")


def compare_headline(results: dict[tuple[str, float], dict[str, Any]]) -> dict[str, Any]:
    rows = csv_rows("artifacts/noJ/per_cell_three_procedures_alpha020.csv")
    require(len(rows) == CELL_COUNT * 3, f"headline rows: {len(rows)}")
    seen: set[tuple[str, str]] = set()
    totals = {name: {"certified": 0, "effective": 0.0} for name in ("H", "S", "B")}
    by_dataset = {
        dataset: {name: {"certified": 0, "effective": 0.0, "n": 0}
                  for name in ("H", "S", "B")}
        for dataset in EXPECTED_DATASET_CELLS
    }
    headline_keys = {key for key in results if math.isclose(key[1], HEADLINE_ALPHA, abs_tol=1e-12)}
    for row in rows:
        require(math.isclose(float(row["alpha"]), HEADLINE_ALPHA, abs_tol=1e-12),
                "non-headline alpha in procedure table")
        key = (row["semantic_id"], HEADLINE_ALPHA)
        procedure = row["procedure"]
        require(procedure in ("H", "S", "B"), f"unknown procedure: {procedure}")
        pair = (row["semantic_id"], procedure)
        require(pair not in seen and key in results, f"duplicate or unknown headline row: {pair}")
        seen.add(pair)
        result = results[key]
        expected = result["procedures"][procedure]
        require(row["dataset"] == result["dataset"], f"headline dataset mismatch: {pair}")
        require(int(row["certified"]) == expected["certified"],
                f"headline decision mismatch: {pair}")
        require(optional_int(row["selected_candidate_index"]) == expected["selected"],
                f"headline selection mismatch: {pair}")
        close(float(row["coverage_lcb"]), expected["coverage"], f"headline coverage: {pair}")
        close(float(row["effective_certified_coverage"]), expected["coverage"],
              f"headline effective coverage: {pair}")
        if procedure == "H":
            require(row["risk_ucb"].strip() == "", "Holm/IUT row reports a numerical risk UCB")
            if expected["selected"] is not None:
                close(float(row["holm_raw_pvalue"]), expected["raw_p"], f"headline raw p: {pair}")
                close(float(row["holm_adjusted_pvalue"]), expected["adjusted_p"],
                      f"headline adjusted p: {pair}")
        elif expected["selected"] is not None:
            close(float(row["risk_ucb"]), expected["risk_ucb"], f"headline risk UCB: {pair}")
        if expected["selected"] is not None:
            candidate = result["candidates"][expected["selected"]]["meta"]
            require(row["selected_score"] == candidate["score"], f"headline score mismatch: {pair}")
            require(row["selected_slot"] == candidate["slot"], f"headline slot mismatch: {pair}")
            close(float(row["selected_gamma"]), float(candidate["gamma"]),
                  f"headline gamma mismatch: {pair}", 1e-12)
            close(float(row["selected_threshold"]), float(candidate["threshold"]),
                  f"headline threshold mismatch: {pair}")
        totals[procedure]["certified"] += expected["certified"]
        totals[procedure]["effective"] += expected["coverage"]
        dataset = result["dataset"]
        by_dataset[dataset][procedure]["certified"] += expected["certified"]
        by_dataset[dataset][procedure]["effective"] += expected["coverage"]
        by_dataset[dataset][procedure]["n"] += 1

    require(len(seen) == CELL_COUNT * 3, "headline procedure coverage is incomplete")
    for procedure in ("H", "S", "B"):
        require(totals[procedure]["certified"] == EXPECTED_CERTIFIED[procedure],
                f"headline certified total {procedure}")
        totals[procedure]["effective"] /= CELL_COUNT
        close(totals[procedure]["effective"], EXPECTED_EFFECTIVE[procedure],
              f"headline effective total {procedure}")
    for dataset, expected_procedures in EXPECTED_DATASET_CERTIFIED.items():
        for procedure, expected in expected_procedures.items():
            value = by_dataset[dataset][procedure]
            require(value["certified"] == expected,
                    f"dataset certified mismatch {dataset}/{procedure}")
            require(value["n"] == EXPECTED_DATASET_CELLS[dataset],
                    f"dataset denominator mismatch {dataset}/{procedure}")
            value["effective"] /= value["n"]
            close(value["effective"], EXPECTED_DATASET_EFFECTIVE[dataset][procedure],
                  f"dataset effective mismatch {dataset}/{procedure}")

    for key in headline_keys:
        if results[key]["procedures"]["B"]["certified"]:
            require(results[key]["procedures"]["H"]["certified"] == 1,
                    f"H did not retain a B-certified cell: {key[0]}")
    return {"totals": totals, "by_dataset": by_dataset}


def check_analysis_registry(semantic_ids: set[str]) -> None:
    rows = csv_rows("artifacts/registry/analysis_input_manifest.csv")
    require(len(rows) == CELL_COUNT, f"analysis input registry rows: {len(rows)}")
    ids = [row["semantic_id"] for row in rows]
    require(len(ids) == len(set(ids)) and set(ids) == semantic_ids,
            "analysis input registry does not bind exactly the 450 count cells")
    kinds: dict[str, int] = defaultdict(int)
    datasets: dict[str, int] = defaultdict(int)
    for row in rows:
        kinds[row["kind"]] += 1
        datasets[row["dataset"]] += 1
        path = PurePosixPath(row["path"])
        require(not path.is_absolute() and ".." not in path.parts,
                f"non-relative analysis input path: {row['semantic_id']}")
        expected_J = 4 if row["dataset"] == "officehome" else 5
        require(int(row["J"]) == expected_J, f"analysis registry J mismatch: {row['semantic_id']}")
    require(dict(datasets) == EXPECTED_DATASET_CELLS, "analysis registry dataset counts mismatch")
    require(kinds == {"cifar": 300, "officehome": 100, "pathmnist_resnext": 50},
            f"analysis registry kind counts mismatch: {dict(kinds)}")
    path_rows = [row for row in rows if row["dataset"] == "pathmnist"]
    require(len(path_rows) == 50, "PathMNIST registry does not contain 50 cells")
    for row in path_rows:
        require(row["kind"] == "pathmnist_resnext", "PathMNIST input kind is not ResNeXt")
        require(row["semantic_id"].startswith("resnext29_8x64d__pathmnist_"),
                f"PathMNIST semantic_id is not ResNeXt-29: {row['semantic_id']}")
        require("resnext29_8x64d" in row["path"],
                f"PathMNIST input path is not ResNeXt-29: {row['semantic_id']}")

    training = csv_rows("artifacts/registry/cifar_officehome_training_matrix.csv")
    require(len(training) == 400, f"CIFAR/Office-Home training registry rows: {len(training)}")
    training_ids = {row["semantic_id"] for row in training}
    non_path_ids = {row["semantic_id"] for row in rows if row["dataset"] != "pathmnist"}
    require(training_ids == non_path_ids, "training registry does not bind the 400 non-Path cells")
    require(not (ROOT / "artifacts/registry/pathmnist_training_matrix.csv").exists(),
            "obsolete PathMNIST WRN registry must not be released")


def check_validity() -> dict[str, Any]:
    summary_rows = csv_rows("artifacts/validity/validity_B1000_per_cell.csv")
    methods = {"fedcore_noJ", "bonferroni_delta_over_J", "pooled_CP_intentionally_invalid"}
    require(len(summary_rows) == 1_350, f"validity summary rows: {len(summary_rows)}")
    summary: dict[tuple[str, str], dict[str, str]] = {}
    for row in summary_rows:
        key = (row["semantic_id"], row["method"])
        require(row["method"] in methods and key not in summary, f"invalid validity key: {key}")
        summary[key] = row
        require(int(row["B"]) == 1_000, f"validity B mismatch: {key}")
    require(len({key[0] for key in summary}) == CELL_COUNT, "validity cells mismatch")
    require(all(sum(key[1] == method for key in summary) == CELL_COUNT for method in methods),
            "validity method coverage mismatch")

    acc: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "bits": 0, "covers": 0, "certified": 0, "unsafe": 0,
                 "effective": 0.0, "conditional": 0.0, "sum_ucb": 0.0}
    )
    path = ROOT / "artifacts/validity/validity_B1000_replicates.csv.gz"
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["semantic_id"], row["method"])
            require(key in summary, f"unknown validity replicate key: {key}")
            replicate = int(row["replicate"])
            require(0 <= replicate < 1_000, f"validity replicate out of range: {key}")
            bit = 1 << replicate
            require(acc[key]["bits"] & bit == 0, f"duplicate validity replicate: {key}/{replicate}")
            acc[key]["bits"] |= bit
            acc[key]["rows"] += 1
            ucb = float(row["UCB"])
            true_risk = float(row["true_worst_client_risk"])
            covers = int(row["ucb_covers_true_worst"])
            certified = int(row["certified"])
            unsafe = int(row["unsafe_certified"])
            coverage = float(row["coverage_LCB"])
            require(covers == int(ucb + 1e-14 >= true_risk), f"validity cover flag mismatch: {key}")
            require(certified in (0, 1) and unsafe in (0, 1), f"validity binary flag mismatch: {key}")
            require(unsafe <= certified, f"unsafe without certification: {key}")
            acc[key]["covers"] += covers
            acc[key]["certified"] += certified
            acc[key]["unsafe"] += unsafe
            acc[key]["effective"] += coverage if certified else 0.0
            acc[key]["conditional"] += coverage if certified else 0.0
            acc[key]["sum_ucb"] += ucb
    require(sum(value["rows"] for value in acc.values()) == 1_350_000,
            "validity replicate row count mismatch")
    for key, row in summary.items():
        value = acc[key]
        require(value["rows"] == 1_000 and value["bits"].bit_count() == 1_000,
                f"validity replicate completeness mismatch: {key}")
        require(int(row["ucb_coverage_count"]) == value["covers"], f"validity cover count: {key}")
        require(int(row["certified_count"]) == value["certified"], f"validity cert count: {key}")
        require(int(row["unsafe_certified_count"]) == value["unsafe"], f"validity unsafe count: {key}")
        close(float(row["empirical_ucb_coverage"]), value["covers"] / 1_000,
              f"validity empirical coverage: {key}")
        close(float(row["cert_rate"]), value["certified"] / 1_000, f"validity cert rate: {key}")
        close(float(row["EffectiveCertCov"]), value["effective"] / 1_000,
              f"validity effective coverage: {key}")
        expected_conditional = (value["conditional"] / value["certified"]
                                if value["certified"] else None)
        observed_conditional = optional_float(row["CondCertCov"])
        if expected_conditional is None:
            require(observed_conditional is None, f"validity conditional coverage should be blank: {key}")
        else:
            close(observed_conditional, expected_conditional, f"validity conditional coverage: {key}")
        close(float(row["mean_ucb"]), value["sum_ucb"] / 1_000, f"validity mean UCB: {key}")

    empirical = {
        method: np.asarray([float(row["empirical_ucb_coverage"])
                            for row in summary_rows if row["method"] == method])
        for method in methods
    }
    fed = empirical["fedcore_noJ"]
    pooled = empirical["pooled_CP_intentionally_invalid"]
    close(float(fed.min()), 0.953, "Fed-CORE minimum B1000 validity", 1e-12)
    close(float(np.quantile(fed, 0.05)), 0.994, "Fed-CORE q05 B1000 validity", 1e-12)
    require(int((fed < 0.95).sum()) == 0, "Fed-CORE has a B1000 cell below 0.95")
    require(int((pooled < 0.95).sum()) == 443, "pooled comparator below-0.95 count mismatch")
    return {"fed_min": float(fed.min()), "fed_q05": float(np.quantile(fed, 0.05)),
            "pooled_below_095": int((pooled < 0.95).sum())}


def check_precision() -> dict[str, Any]:
    summary_rows = csv_rows("artifacts/validity/precision_B10000_summary_20_cells.csv")
    require(len(summary_rows) == 20, f"precision summary rows: {len(summary_rows)}")
    summary = {row["semantic_id"]: row for row in summary_rows}
    require(len(summary) == 20, "duplicate precision summary cell")
    selected = csv_rows("artifacts/validity/precision_selected_20_from_B1000.csv")
    require(len(selected) == 20, "precision selection rows mismatch")
    require([row["semantic_id"] for row in selected] ==
            [row["semantic_id"] for row in summary_rows],
            "precision selected-cell order mismatch")
    acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "bits": 0, "covers": 0, "sum_ucb": 0.0}
    )
    path = ROOT / "artifacts/validity/precision_B10000_replicates_20_cells.csv.gz"
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            semantic_id = row["semantic_id"]
            require(semantic_id in summary, f"unknown precision replicate cell: {semantic_id}")
            replicate = int(row["replicate"])
            require(0 <= replicate < 10_000, f"precision replicate out of range: {semantic_id}")
            bit = 1 << replicate
            require(acc[semantic_id]["bits"] & bit == 0,
                    f"duplicate precision replicate: {semantic_id}/{replicate}")
            acc[semantic_id]["bits"] |= bit
            acc[semantic_id]["rows"] += 1
            ucb = float(row["UCB"])
            true_risk = float(row["true_worst_client_risk"])
            covers = int(row["ucb_covers_true_worst"])
            require(covers == int(ucb + 1e-14 >= true_risk),
                    f"precision cover flag mismatch: {semantic_id}/{replicate}")
            acc[semantic_id]["covers"] += covers
            acc[semantic_id]["sum_ucb"] += ucb
    require(sum(value["rows"] for value in acc.values()) == 200_000,
            "precision replicate row count mismatch")
    for semantic_id, row in summary.items():
        value = acc[semantic_id]
        require(value["rows"] == 10_000 and value["bits"].bit_count() == 10_000,
                f"precision replicate completeness: {semantic_id}")
        require(int(row["B10000_cover_count"]) == value["covers"],
                f"precision cover count: {semantic_id}")
        close(float(row["B10000_empirical_bound_validity"]), value["covers"] / 10_000,
              f"precision empirical validity: {semantic_id}")
        close(float(row["mean_ucb"]), value["sum_ucb"] / 10_000,
              f"precision mean UCB: {semantic_id}")
        x = value["covers"]
        exact_low = 0.0 if x == 0 else float(beta.ppf(0.025, x, 10_000 - x + 1))
        close(float(row["exact_CP95_low"]), exact_low, f"precision exact lower: {semantic_id}")
    point_values = [float(row["B10000_empirical_bound_validity"]) for row in summary_rows]
    exact_lows = [float(row["exact_CP95_low"]) for row in summary_rows]
    close(min(point_values), 0.9543, "precision minimum empirical validity", 1e-12)
    require(sum(value < 0.95 for value in point_values) == 0,
            "precision point validity below 0.95")
    require(sum(value < 0.95 for value in exact_lows) == 0,
            "precision exact lower bound below 0.95")
    return {"rows": 200_000, "minimum": min(point_values), "minimum_exact_low": min(exact_lows)}


def compare_group_summary(observed_rows: list[dict[str, str]],
                          computed: dict[tuple[Any, ...], dict[str, float]],
                          key_fields: tuple[str, ...], label: str) -> None:
    require(len(observed_rows) == len(computed), f"{label} row count mismatch")
    seen: set[tuple[Any, ...]] = set()
    for row in observed_rows:
        key: list[Any] = []
        for field in key_fields:
            if field in ("audit_n_per_client",):
                key.append(int(row[field]))
            elif field == "d":
                key.append(float(row[field]))
            else:
                key.append(row[field])
        key_tuple = tuple(key)
        require(key_tuple in computed and key_tuple not in seen, f"{label} key mismatch: {key_tuple}")
        seen.add(key_tuple)
        expected = computed[key_tuple]
        require(int(float(row["n_cells"])) == int(expected["n_cells"]), f"{label} n_cells: {key_tuple}")
        require(int(float(row["n_cell_replicates"])) == int(expected["rows"]),
                f"{label} replicate count: {key_tuple}")
        require(int(float(row["certified_count"])) == int(expected["certified"]),
                f"{label} certified count: {key_tuple}")
        close(float(row["certification_rate"]), expected["certified"] / expected["rows"],
              f"{label} certification rate: {key_tuple}")
        close(float(row["effective_certified_acceptance"]), expected["effective"] / expected["rows"],
              f"{label} effective acceptance: {key_tuple}")
        expected_conditional = (expected["conditional"] / expected["certified"]
                                if expected["certified"] else None)
        observed_conditional = optional_float(row["conditional_certified_acceptance"])
        if expected_conditional is None:
            require(observed_conditional is None, f"{label} conditional should be blank: {key_tuple}")
        else:
            close(observed_conditional, expected_conditional,
                  f"{label} conditional acceptance: {key_tuple}")


def check_audit_size() -> dict[str, Any]:
    group_dataset: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {"rows": 0, "certified": 0, "effective": 0.0, "conditional": 0.0,
                 "cells": set()}
    )
    group_dataset_d: dict[tuple[str, float, int], dict[str, float]] = defaultdict(
        lambda: {"rows": 0, "certified": 0, "effective": 0.0, "conditional": 0.0,
                 "cells": set()}
    )
    group_cell: dict[tuple[str, str, float, int], dict[str, float]] = defaultdict(
        lambda: {"rows": 0, "certified": 0, "effective": 0.0, "conditional": 0.0,
                 "cells": set(), "bits": 0}
    )
    path = ROOT / "artifacts/audit_size/audit_size_sensitivity_replicates.csv.gz"
    total = 0
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total += 1
            semantic_id = row["semantic_id"]
            dataset = row["dataset"]
            d = float(row["d"])
            audit_n = int(row["audit_n_per_client"])
            replicate = int(row["replicate"])
            require(dataset in ("cifar10", "cifar100"), f"audit-size dataset: {dataset}")
            require(audit_n in (128, 256, 514, 1028), f"audit size: {audit_n}")
            require(0 <= replicate < 200, f"audit replicate out of range: {semantic_id}")
            require(int(row["J"]) == 5 and int(row["M"]) == M,
                    f"audit J/M mismatch: {semantic_id}")
            close(float(row["alpha"]), HEADLINE_ALPHA, f"audit alpha: {semantic_id}", 1e-12)
            require(row["acceptance_allocation"] == "delta_c/M",
                    f"audit allocation mismatch: {semantic_id}")
            require(row["risk_procedure"] ==
                    "full-simplex IUT per member followed by Holm over M",
                    f"audit risk procedure mismatch: {semantic_id}")
            certified = int(row["certified"])
            effective = float(row["effective_certified_acceptance"])
            require(certified in (0, 1), f"audit certified flag: {semantic_id}")
            if certified:
                require(optional_int(row["selected_index"]) is not None and
                        0 <= optional_int(row["selected_index"]) < M,
                        f"audit selected index: {semantic_id}")
                selected_lcb = float(row["selected_acceptance_lcb"])
                require(selected_lcb > 0.0, f"audit nonpositive selected LCB: {semantic_id}")
                close(effective, selected_lcb, f"audit effective LCB: {semantic_id}")
            else:
                close(effective, 0.0, f"uncertified audit has effective acceptance: {semantic_id}")
                require(row["selected_index"].strip() == "", f"uncertified audit selected index: {semantic_id}")
            key_cell = (semantic_id, dataset, d, audit_n)
            cell_value = group_cell[key_cell]
            bit = 1 << replicate
            require(int(cell_value["bits"]) & bit == 0,
                    f"duplicate audit replicate: {semantic_id}/{audit_n}/{replicate}")
            cell_value["bits"] = int(cell_value["bits"]) | bit
            for value in (
                group_dataset[(dataset, audit_n)],
                group_dataset_d[(dataset, d, audit_n)],
                cell_value,
            ):
                value["rows"] += 1
                value["certified"] += certified
                value["effective"] += effective
                value["conditional"] += effective if certified else 0.0
                value["cells"].add(semantic_id)
    require(total == 240_000, f"audit-size replicate rows: {total}")
    require(len(group_cell) == 1_200, f"audit-size cell groups: {len(group_cell)}")
    for key, value in group_cell.items():
        require(value["rows"] == 200 and int(value["bits"]).bit_count() == 200,
                f"audit-size replicate completeness: {key}")
    for mapping in (group_dataset, group_dataset_d, group_cell):
        for value in mapping.values():
            value["n_cells"] = len(value["cells"])

    compare_group_summary(
        csv_rows("artifacts/audit_size/audit_size_sensitivity_by_dataset.csv"),
        group_dataset, ("dataset", "audit_n_per_client"), "audit dataset summary")
    compare_group_summary(
        csv_rows("artifacts/audit_size/audit_size_sensitivity_by_dataset_d.csv"),
        group_dataset_d, ("dataset", "d", "audit_n_per_client"), "audit condition summary")
    compare_group_summary(
        csv_rows("artifacts/audit_size/audit_size_sensitivity_by_cell.csv"),
        group_cell, ("semantic_id", "dataset", "d", "audit_n_per_client"),
        "audit cell summary")

    invariants = load_json("artifacts/audit_size/audit_size_invariants.json")
    require(invariants == {
        "A_K_n_invariant": True,
        "count_elements_checked": 14_400_000,
        "expected_replicate_rows": 240_000,
        "nested_prefixes": True,
        "observed_replicate_rows": 240_000,
    }, "audit-size invariant report mismatch")
    local = load_json("artifacts/audit_size/LOCAL_VALIDATION.json")
    audit = local.get("audit_size", {})
    require(audit.get("A_K_n_invariant") is True and audit.get("nested_prefixes") is True,
            "local audit-size validation failed")
    require(audit.get("duplicate_cell_size_replicate_keys") == 0 and
            audit.get("unique_cell_size_replicate_keys") == 240_000,
            "local audit-size key validation failed")
    endpoints = {
        ("cifar10", 128): (0.0064333333333333334, 0.0014334575824051907),
        ("cifar10", 1028): (0.5992, 0.1896486930877579),
        ("cifar100", 128): (0.0, 0.0),
        ("cifar100", 1028): (0.4344, 0.046464997548653623),
    }
    for key, (rate, effective) in endpoints.items():
        value = group_dataset[key]
        close(value["certified"] / value["rows"], rate, f"audit endpoint rate {key}")
        close(value["effective"] / value["rows"], effective, f"audit endpoint effective {key}")
    return {"rows": total, "groups": len(group_cell)}


def aggregate_results(results: dict[tuple[str, float], dict[str, Any]]) -> dict[tuple[str, str, float], dict[str, float]]:
    grouped: dict[tuple[str, str, float], dict[str, float]] = defaultdict(
        lambda: {"n": 0, "certified": 0, "effective": 0.0, "conditional": 0.0}
    )
    for key, result in results.items():
        group_key = (result["dataset"], result["condition"], key[1])
        procedure = result["procedures"]["H"]
        grouped[group_key]["n"] += 1
        grouped[group_key]["certified"] += procedure["certified"]
        grouped[group_key]["effective"] += procedure["coverage"]
        grouped[group_key]["conditional"] += procedure["coverage"] if procedure["certified"] else 0.0
    return grouped


def check_figure_sources(results: dict[tuple[str, float], dict[str, Any]]) -> dict[str, Any]:
    grouped = aggregate_results(results)
    phase_rows = csv_rows("artifacts/phase_map/recertification_summary_by_dataset_condition_alpha.csv")
    require(len(phase_rows) == 60, f"phase-map summary rows: {len(phase_rows)}")
    for row in phase_rows:
        key = (row["dataset"], row["condition"], float(row["alpha"]))
        require(key in grouped, f"phase-map key mismatch: {key}")
        expected = grouped[key]
        require(int(float(row["n_cells"])) == expected["n"], f"phase-map n: {key}")
        require(int(float(row["theorem_certified"])) == expected["certified"],
                f"phase-map certified: {key}")
        close(float(row["theorem_EffectiveCertCov"]), expected["effective"] / expected["n"],
              f"phase-map effective: {key}")

    failure_rows = csv_rows("artifacts/phase_map/failure_transitions.csv")
    require(len(failure_rows) == 48, f"failure-anatomy rows: {len(failure_rows)}")
    condition_map = {
        ("cifar10", "d5.0"): "d5",
        ("cifar100", "d5.0"): "d5",
        ("officehome", "convnext_full_ft"): "full",
        ("officehome", "convnext_frozen_linear"): "frozen",
    }
    anatomy_fields = (
        "fc_certified", "fc_support", "fc_zero_acceptance", "fc_empirical_risk",
        "fc_count", "fc_width",
    )
    for row in failure_rows:
        total = sum(int(row[field]) for field in anatomy_fields)
        require(total == int(row["n_cells"]) == 50,
                f"failure anatomy does not account for 50 cells: {row['dataset']}/{row['condition']}/{row['alpha']}")
        condition = condition_map.get((row["dataset"], row["condition"]), row["condition"])
        key = (row["dataset"], condition, float(row["alpha"]))
        require(key in grouped, f"failure-anatomy key mismatch: {key}")
        require(int(row["fc_certified"]) == grouped[key]["certified"],
                f"failure certified count mismatch: {key}")

    headline_cifar = [row for row in failure_rows
                      if row["dataset"] in ("cifar10", "cifar100") and
                      math.isclose(float(row["alpha"]), HEADLINE_ALPHA, abs_tol=1e-12)]
    require(len(headline_cifar) == 6, "Figure 5 headline CIFAR grid is incomplete")
    certified = sum(int(row["fc_certified"]) for row in headline_cifar)
    refusals = sum(int(row["n_cells"]) - int(row["fc_certified"]) for row in headline_cifar)
    width_count = sum(int(row["fc_width"]) + int(row["fc_count"]) for row in headline_cifar)
    require((certified, refusals, width_count) == (77, 223, 217),
            f"Figure 5 headline invariants: {(certified, refusals, width_count)}")

    path_rows = csv_rows("artifacts/pathmnist/Figure7_PathMNIST_Data.csv")
    split_rows = [row for row in path_rows if row["record_type"] == "split_primary"]
    frontier_rows = [row for row in path_rows if row["record_type"] == "alpha_frontier"]
    require(len(split_rows) == 10 and len(frontier_rows) == 12 and len(path_rows) == 22,
            "PathMNIST Figure source row counts mismatch")
    for row in frontier_rows:
        key = ("pathmnist", row["condition"], float(row["alpha"]))
        expected = grouped[key]
        require(int(float(row["n_cells"])) == expected["n"] == 25,
                f"PathMNIST frontier n mismatch: {key}")
        require(int(float(row["certified"])) == expected["certified"],
                f"PathMNIST frontier certified mismatch: {key}")
        close(float(row["effective_certified_coverage"]), expected["effective"] / expected["n"],
              f"PathMNIST frontier effective mismatch: {key}")
        expected_conditional = (expected["conditional"] / expected["certified"]
                                if expected["certified"] else None)
        observed_conditional = optional_float(row["conditional_certified_coverage"])
        if expected_conditional is None:
            require(observed_conditional is None, f"PathMNIST conditional should be blank: {key}")
        else:
            close(observed_conditional, expected_conditional,
                  f"PathMNIST frontier conditional mismatch: {key}")

    split_grouped: dict[tuple[int, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0, "certified": 0, "effective": 0.0}
    )
    for key, result in results.items():
        if result["dataset"] != "pathmnist" or not math.isclose(key[1], HEADLINE_ALPHA, abs_tol=1e-12):
            continue
        match = re.search(r"pathmnist_split_(\d+)", key[0])
        require(match is not None, f"PathMNIST split missing from semantic_id: {key[0]}")
        split = int(match.group(1))
        value = split_grouped[(split, result["condition"])]
        procedure = result["procedures"]["H"]
        value["n"] += 1
        value["certified"] += procedure["certified"]
        value["effective"] += procedure["coverage"]
    for row in split_rows:
        key = (int(float(row["split"])), row["condition"])
        expected = split_grouped[key]
        require(int(float(row["n_cells"])) == expected["n"] == 5,
                f"PathMNIST split n mismatch: {key}")
        require(int(float(row["certified"])) == expected["certified"],
                f"PathMNIST split certified mismatch: {key}")
        close(float(row["effective_certified_coverage"]), expected["effective"] / expected["n"],
              f"PathMNIST split effective mismatch: {key}")

    alpha020 = {row["condition"]: row for row in frontier_rows
                if math.isclose(float(row["alpha"]), HEADLINE_ALPHA, abs_tol=1e-12)}
    require(int(float(alpha020["d0.5"]["certified"])) == 16 and
            int(float(alpha020["d5"]["certified"])) == 17,
            "PathMNIST alpha=0.20 certified counts mismatch")
    close(float(alpha020["d0.5"]["effective_certified_coverage"]), 0.2296048700521668,
          "PathMNIST d0.5 effective at alpha=0.20")
    close(float(alpha020["d5"]["effective_certified_coverage"]), 0.2846210391209351,
          "PathMNIST d5 effective at alpha=0.20")
    return {"figure5": [certified, refusals, width_count], "path_alpha020": [16, 17]}


def png_info(path: Path) -> tuple[int, int, float | None]:
    with path.open("rb") as handle:
        require(handle.read(8) == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path.name}")
        width = height = None
        dpi = None
        while True:
            length_raw = handle.read(4)
            if not length_raw:
                break
            length = struct.unpack(">I", length_raw)[0]
            chunk_type = handle.read(4)
            payload = handle.read(length)
            handle.read(4)
            if chunk_type == b"IHDR":
                width, height = struct.unpack(">II", payload[:8])
            elif chunk_type == b"pHYs" and len(payload) == 9 and payload[8] == 1:
                x_ppm, y_ppm = struct.unpack(">II", payload[:8])
                require(x_ppm == y_ppm, f"non-square PNG pixels: {path.name}")
                dpi = x_ppm * 0.0254
            elif chunk_type == b"IEND":
                break
    require(width is not None and height is not None, f"PNG dimensions missing: {path.name}")
    return int(width), int(height), dpi


def check_reference_tables_and_figures() -> None:
    for number in (1, 2, 3):
        contract = ROOT / f"contract/table{number}_{['contribution_map','validity_gates','experiment_registry'][number-1]}.csv"
        reference = ROOT / f"reference/tables/Table{number}.csv"
        require(contract.read_bytes() == reference.read_bytes(), f"Table {number} contract/reference mismatch")
    table3 = csv_rows("reference/tables/Table3.csv")
    path_rows = [row for row in table3 if "PathMNIST" in row["Evidence tier / condition"]]
    require(len(path_rows) == 1 and "ResNeXt-29 8x64d" in path_rows[0]["Frozen predictor training and checkpoint"],
            "Table 3 PathMNIST predictor is not ResNeXt-29 8x64d")
    table4_expected = [
        ["CIFAR-10 N=150", "66 / 0.129; 66 / 0.124", "54 / 0.101", "+12 / +2.81 pp"],
        ["CIFAR-100 N=150", "11 / 0.007; 11 / 0.007", "1 / 0.001", "+10 / +0.64 pp"],
        ["Office-Home N=100", "88 / 0.119; 88 / 0.114", "69 / 0.082", "+19 / +3.72 pp"],
        ["PathMNIST N=50", "33 / 0.257; 33 / 0.250", "24 / 0.177", "+9 / +8.01 pp"],
        ["Overall N=450", "198 / 0.100; 198 / 0.097", "148 / 0.072", "+50 / +2.87 pp"],
    ]
    with (ROOT / "reference/tables/Table4.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        observed = list(reader)
    require(observed == table4_expected, "Table 4 transcription mismatch")

    expected_dimensions = {
        "Figure1.png": (1872, 950),
        "Figure2.png": (1872, 930),
        "Figure3.png": (3600, 1302),
        "Figure4.png": (3600, 1425),
        "Figure5.png": (1872, 1935),
        "Figure6.png": (3600, 1302),
        "Figure7.png": (3600, 2175),
    }
    for name, expected in expected_dimensions.items():
        width, height, dpi = png_info(ROOT / "reference/figures" / name)
        require((width, height) == expected, f"{name} dimensions mismatch")
        require(dpi is not None and 299.0 <= dpi <= 301.0, f"{name} is not 300 dpi")


def scan_no_absolute_paths() -> None:
    text_suffixes = {".md", ".json", ".csv", ".py", ".txt"}
    root_names = ("Users", "home", "data", "tmp", "private", "mnt", "var", "opt", "srv")
    forbidden = ["/" + name + "/" for name in root_names]
    forbidden.append("file" + "://")
    windows = re.compile(r"[A-Za-z]:[\\/]")

    def inspect(name: str, chunks: Iterable[str]) -> None:
        for line_number, line in enumerate(chunks, 1):
            if any(token in line for token in forbidden) or windows.search(line):
                fail(f"local absolute path in {name}:{line_number}")

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path == SHA_MANIFEST or path.suffix in text_suffixes:
            with path.open(encoding="utf-8", errors="strict") as handle:
                inspect(relative, handle)
        elif path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="strict") as handle:
                inspect(relative, handle)


def main() -> int:
    try:
        manifest_hashes = check_hash_manifest()
        check_release_metadata(manifest_hashes)
        groups, semantic_ids = load_and_verify_counts()
        verify_family_freeze(groups)
        results = compute_decisions(groups)
        compare_recertification(results)
        headline = compare_headline(results)
        check_analysis_registry(semantic_ids)
        validity = check_validity()
        precision = check_precision()
        audit_size = check_audit_size()
        figures = check_figure_sources(results)
        check_reference_tables_and_figures()
        scan_no_absolute_paths()
    except (ReleaseError, KeyError, ValueError, OSError, csv.Error, json.JSONDecodeError) as exc:
        print(f"FEDCORE V18 THEOREM-ALIGNED RELEASE: FAIL_CLOSED: {exc}", file=sys.stderr)
        return 1

    report = {
        "release_id": "fedcore-v0.2.0-theorem-aligned",
        "count_rows": COUNT_ROWS,
        "cells": CELL_COUNT,
        "family_rows": FAMILY_ROWS,
        "family_hashes": FAMILY_HASH_ROWS,
        "certified_cells": {name: headline["totals"][name]["certified"] for name in ("H", "S", "B")},
        "effective_acceptance": {name: headline["totals"][name]["effective"] for name in ("H", "S", "B")},
        "validity": validity,
        "precision": precision,
        "audit_size": audit_size,
        "figure_invariants": figures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("FEDCORE V18 THEOREM-ALIGNED RELEASE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
