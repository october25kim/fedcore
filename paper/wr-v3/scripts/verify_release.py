#!/usr/bin/env python3
"""Fail-closed verifier for the Fed-CORE WR-v3 numerical release."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import numpy as np
    from scipy.stats import beta, binom
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"FAIL_CLOSED: NumPy and SciPy are required: {exc}")


ROOT = Path(__file__).resolve().parents[1]
SHA_MANIFEST = ROOT / "SHA256SUMS"
ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
PRIMARY_ALPHA = 0.20
DELTA_R = 0.05
DELTA_C = 0.05
M_EXPECTED = 12
CELL_COUNT = 450
COUNT_ROWS = 154_800
TOL = 2e-11

EXPECTED_HEADLINE = {
    "cifar10": {
        "H": (63, 0.11310162900873891),
        "S": (63, 0.10918855528565553),
        "B": (53, 0.09066079034643129),
    },
    "cifar100": {
        "H": (9, 0.0044082234449503875),
        "S": (9, 0.0043971385498148),
        "B": (3, 0.0015108023084971077),
    },
    "officehome": {
        "H": (79, 0.10316349495866468),
        "S": (79, 0.10244740788665777),
        "B": (55, 0.06350311108333068),
    },
    "pathmnist": {
        "H": (26, 0.1919596752230305),
        "S": (26, 0.18928141440124335),
        "B": (19, 0.1381206700852299),
    },
    "ALL": {
        "H": (177, 0.08342402472238086),
        "S": (177, 0.0816592568534411),
        "B": (130, 0.06018240780185294),
    },
}


class ReleaseError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def close(actual: float, expected: float, message: str, tol: float = TOL) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
        raise ReleaseError(f"{message}: got {actual!r}, expected {expected!r}")


def optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def optional_int(value: Any) -> int | None:
    number = optional_float(value)
    return None if number is None else int(number)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def csv_rows(relative: str) -> list[dict[str, str]]:
    path = ROOT / relative
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def json_value(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def check_hash_manifest() -> None:
    require(SHA_MANIFEST.is_file(), "SHA256SUMS is missing")
    listed: dict[str, str] = {}
    for line_number, raw in enumerate(SHA_MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
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
        require(not path.is_symlink(), f"symlink is not permitted: {path.relative_to(ROOT)}")
        if path.is_file() and path != SHA_MANIFEST:
            actual.add(path.relative_to(ROOT).as_posix())
    require(set(listed) == actual,
            f"manifest file-set mismatch: missing={sorted(actual-set(listed))}, "
            f"extra={sorted(set(listed)-actual)}")
    for name, expected in listed.items():
        require(sha256_file(ROOT / name) == expected, f"hash mismatch: {name}")


def cp_upper(k: int, n: int, eps: float) -> float:
    if n <= 0 or k >= n or eps <= 0.0:
        return 1.0
    return float(beta.isf(eps, k + 1, n - k))


def cp_lower(k: int, n: int, eps: float) -> float:
    if n <= 0 or k <= 0:
        return 0.0
    return float(beta.ppf(eps, k, n - k + 1))


def holm_adjusted(pvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    ranks = np.empty(values.shape, dtype=int)
    running = 0.0
    M = len(values)
    for rank, index in enumerate(order, start=1):
        running = max(running, (M - rank + 1) * float(values[index]))
        adjusted[index] = min(1.0, running)
        ranks[index] = rank
    return adjusted, ranks


def family_tie_key(row: dict[str, str]) -> tuple[float, str, int]:
    tie_score = "msp" if row["slot"] == "native" else row["slot"]
    return float(row["gamma"]), tie_score, int(row["candidate_index"])


def select_candidate(certified: np.ndarray, coverage: np.ndarray,
                     family: list[dict[str, str]]) -> int | None:
    eligible = [m for m in range(len(family)) if bool(certified[m])]
    if not eligible:
        return None
    return min(eligible, key=lambda m: (-float(coverage[m]),) + family_tie_key(family[m]))


def procedure_results(A: np.ndarray, K: np.ndarray, n: np.ndarray,
                      family: list[dict[str, str]], alpha: float) -> dict[str, dict[str, Any]]:
    M, J = A.shape
    proposal_ok = np.asarray([int(row["proposal_feasible"]) == 1 for row in family], dtype=bool)

    raw_p = np.asarray([
        max(1.0 if int(A[m, j]) <= 0 else float(binom.cdf(int(K[m, j]), int(A[m, j]), alpha))
            for j in range(J))
        for m in range(M)
    ])
    adjusted_p, _ = holm_adjusted(raw_p)
    C_h = np.asarray([
        min(cp_lower(int(A[m, j]), int(n[j]), DELTA_C / M) for j in range(J))
        for m in range(M)
    ])
    H = proposal_ok & (adjusted_p <= DELTA_R) & (C_h > 0.0)

    U_s = np.asarray([
        max(cp_upper(int(K[m, j]), int(A[m, j]), DELTA_R / M) for j in range(J))
        for m in range(M)
    ])
    C_s = np.asarray([
        min(cp_lower(int(A[m, j]), int(n[j]), DELTA_C / M) for j in range(J))
        for m in range(M)
    ])
    S = proposal_ok & (U_s <= alpha) & (C_s > 0.0)

    U_b = np.asarray([
        max(cp_upper(int(K[m, j]), int(A[m, j]), DELTA_R / (M * J)) for j in range(J))
        for m in range(M)
    ])
    C_b = np.asarray([
        min(cp_lower(int(A[m, j]), int(n[j]), DELTA_C / (M * J)) for j in range(J))
        for m in range(M)
    ])
    B = proposal_ok & (U_b <= alpha) & (C_b > 0.0)

    require(np.all(U_s <= U_b + 1e-14), "candidatewise S UCB exceeded B UCB")
    require(np.all(C_s + 1e-14 >= C_b), "candidatewise S coverage LCB fell below B")

    output: dict[str, dict[str, Any]] = {}
    for name, passed, coverage, risk in (
        ("H", H, C_h, None), ("S", S, C_s, U_s), ("B", B, C_b, U_b)
    ):
        pick = select_candidate(passed, coverage, family)
        if pick is None:
            output[name] = {
                "certified": 0, "coverage_lcb": 0.0, "risk_ucb": None,
                "holm_raw_pvalue": None, "holm_adjusted_pvalue": None,
                "selected_candidate_index": None,
            }
        else:
            output[name] = {
                "certified": 1,
                "coverage_lcb": float(coverage[pick]),
                "risk_ucb": None if risk is None else float(risk[pick]),
                "holm_raw_pvalue": float(raw_p[pick]) if name == "H" else None,
                "holm_adjusted_pvalue": float(adjusted_p[pick]) if name == "H" else None,
                "selected_candidate_index": int(pick),
            }
    require(not (output["B"]["certified"] and not output["S"]["certified"]),
            "B-certified cell was not S-certified")
    return output


def compare_optional(actual: float | int | None, expected_text: str,
                     message: str, integer: bool = False) -> None:
    expected = optional_int(expected_text) if integer else optional_float(expected_text)
    require((actual is None) == (expected is None), f"{message}: null mismatch")
    if actual is not None and expected is not None:
        if integer:
            require(int(actual) == int(expected), f"{message}: got {actual}, expected {expected}")
        else:
            close(float(actual), float(expected), message)


def check_primary() -> list[dict[str, Any]]:
    counts = csv_rows("artifacts/primary/primary_candidate_counts.csv.gz")
    require(len(counts) == COUNT_ROWS, f"count-row mismatch: {len(counts)}")
    groups: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    semantic_ids: set[str] = set()
    for row in counts:
        sid = row["semantic_id"]
        alpha = float(row["alpha"])
        require(alpha in ALPHAS, f"undeclared alpha: {alpha}")
        groups[(sid, alpha)].append(row)
        semantic_ids.add(sid)
    require(len(semantic_ids) == CELL_COUNT, f"cell-count mismatch: {len(semantic_ids)}")
    require(len(groups) == CELL_COUNT * len(ALPHAS), "cell-alpha group count mismatch")

    official_rows = csv_rows("artifacts/primary/primary_per_cell_procedures.csv")
    require(len(official_rows) == CELL_COUNT * len(ALPHAS) * 3,
            "per-cell procedure row count mismatch")
    official = {(row["semantic_id"], float(row["alpha"]), row["procedure"]): row
                for row in official_rows}
    require(len(official) == len(official_rows), "duplicate per-cell procedure key")

    recomputed: list[dict[str, Any]] = []
    for sid, alpha in sorted(groups):
        rows = groups[(sid, alpha)]
        candidate_ids = sorted({int(row["candidate_index"]) for row in rows})
        clients = sorted({int(row["client"]) for row in rows})
        require(candidate_ids == list(range(M_EXPECTED)), f"candidate roster mismatch: {sid}, {alpha}")
        require(clients == list(range(len(clients))), f"client roster mismatch: {sid}, {alpha}")
        J = len(clients)
        require(len(rows) == M_EXPECTED * J, f"count block size mismatch: {sid}, {alpha}")

        by_key = {(int(row["candidate_index"]), int(row["client"])): row for row in rows}
        require(len(by_key) == len(rows), f"duplicate count key: {sid}, {alpha}")
        family: list[dict[str, str]] = []
        A = np.zeros((M_EXPECTED, J), dtype=int)
        K = np.zeros((M_EXPECTED, J), dtype=int)
        n = np.zeros(J, dtype=int)
        for m in range(M_EXPECTED):
            first = by_key[(m, 0)]
            family.append(first)
            for j in range(J):
                row = by_key[(m, j)]
                for field in ("score", "slot", "gamma", "threshold", "proposal_feasible",
                              "selector_sha256", "counts_tensor_sha256"):
                    require(row[field] == first[field], f"candidate metadata mismatch: {sid}, {alpha}, {m}, {field}")
                A[m, j] = int(row["A"])
                K[m, j] = int(row["K"])
                if m == 0:
                    n[j] = int(row["n"])
                else:
                    require(int(row["n"]) == int(n[j]), f"n changed across candidates: {sid}, {alpha}, {j}")
        require(np.all((0 <= K) & (K <= A) & (A <= n[None, :])),
                f"invalid 0 <= K <= A <= n: {sid}, {alpha}")
        tensor_hash = canonical_json_hash({"A": A.tolist(), "K": K.tolist(), "n": n.tolist()})
        require(all(row["counts_tensor_sha256"] == tensor_hash for row in rows),
                f"count tensor hash mismatch: {sid}, {alpha}")

        results = procedure_results(A, K, n, family, alpha)
        for procedure in ("H", "S", "B"):
            result = results[procedure]
            expected = official.get((sid, alpha, procedure))
            require(expected is not None, f"missing official result: {sid}, {alpha}, {procedure}")
            require(int(expected["J"]) == J and int(expected["M"]) == M_EXPECTED,
                    f"J/M mismatch: {sid}, {alpha}, {procedure}")
            close(float(expected["delta_r"]), DELTA_R, "delta_r mismatch")
            close(float(expected["delta_c"]), DELTA_C, "delta_c mismatch")
            require(int(expected["certified"]) == int(result["certified"]),
                    f"certification mismatch: {sid}, {alpha}, {procedure}")
            close(float(expected["coverage_lcb"]), float(result["coverage_lcb"]),
                  f"coverage mismatch: {sid}, {alpha}, {procedure}")
            close(float(expected["effective_certified_coverage"]), float(result["coverage_lcb"]),
                  f"effective coverage mismatch: {sid}, {alpha}, {procedure}")
            compare_optional(result["risk_ucb"], expected["risk_ucb"],
                             f"risk UCB mismatch: {sid}, {alpha}, {procedure}")
            compare_optional(result["holm_raw_pvalue"], expected["holm_raw_pvalue"],
                             f"raw p-value mismatch: {sid}, {alpha}, {procedure}")
            compare_optional(result["holm_adjusted_pvalue"], expected["holm_adjusted_pvalue"],
                             f"adjusted p-value mismatch: {sid}, {alpha}, {procedure}")
            compare_optional(result["selected_candidate_index"], expected["selected_candidate_index"],
                             f"selected member mismatch: {sid}, {alpha}, {procedure}", integer=True)
            require(expected["counts_tensor_sha256"] == tensor_hash,
                    f"official tensor binding mismatch: {sid}, {alpha}, {procedure}")
            pick = result["selected_candidate_index"]
            if pick is None:
                for field in ("selected_score", "selected_slot", "selected_gamma",
                              "selected_threshold", "selected_selector_sha256"):
                    require(expected[field] == "", f"unexpected selected field: {sid}, {alpha}, {procedure}, {field}")
            else:
                chosen = family[int(pick)]
                require(expected["selected_score"] == chosen["score"], "selected score mismatch")
                require(expected["selected_slot"] == chosen["slot"], "selected slot mismatch")
                close(float(expected["selected_gamma"]), float(chosen["gamma"]), "selected gamma mismatch")
                close(float(expected["selected_threshold"]), float(chosen["threshold"]), "selected threshold mismatch")
                require(expected["selected_selector_sha256"] == chosen["selector_sha256"],
                        "selected selector hash mismatch")
            recomputed.append({
                "semantic_id": sid,
                "dataset": expected["dataset"],
                "condition": expected["condition"],
                "alpha": alpha,
                "procedure": procedure,
                "certified": int(result["certified"]),
                "coverage_lcb": float(result["coverage_lcb"]),
            })
    return recomputed


def summary_key(row: dict[str, str]) -> tuple[float, str, str, str]:
    return float(row["alpha"]), row["dataset"], row["condition"], row["procedure"]


def check_summaries(rows: list[dict[str, Any]]) -> None:
    official_summary = {summary_key(row): row for row in csv_rows("artifacts/primary/full_sweep_summary.csv")}
    keys = sorted({(row["alpha"], row["dataset"], row["condition"], row["procedure"]) for row in rows})
    require(len(official_summary) == len(keys), "full-sweep summary key count mismatch")
    for key in keys:
        subset = [row for row in rows if (row["alpha"], row["dataset"], row["condition"], row["procedure"]) == key]
        certified = sum(row["certified"] for row in subset)
        effective = float(np.mean([row["coverage_lcb"] for row in subset]))
        conditional = (sum(row["coverage_lcb"] for row in subset if row["certified"]) / certified) if certified else None
        expected = official_summary[key]
        require(int(expected["N_cells"]) == len(subset), f"summary N mismatch: {key}")
        require(int(expected["certified_cells"]) == certified, f"summary certified mismatch: {key}")
        close(float(expected["certification_rate"]), certified / len(subset), f"summary rate mismatch: {key}")
        close(float(expected["EffectiveCertCov"]), effective, f"summary effective mismatch: {key}")
        compare_optional(conditional, expected["CondCertCov"], f"summary conditional mismatch: {key}")

    headline_rows = csv_rows("artifacts/primary/primary_headline_alpha020.csv")
    headline = {(row["dataset"], row["procedure"]): row for row in headline_rows}
    require(len(headline) == 15, "headline row count mismatch")
    for dataset, procedures in EXPECTED_HEADLINE.items():
        for procedure, (expected_certified, expected_effective) in procedures.items():
            subset = [row for row in rows if row["alpha"] == PRIMARY_ALPHA and row["procedure"] == procedure
                      and (dataset == "ALL" or row["dataset"] == dataset)]
            certified = sum(row["certified"] for row in subset)
            effective = float(np.mean([row["coverage_lcb"] for row in subset]))
            official = headline[(dataset, procedure)]
            require(certified == expected_certified == int(official["certified_cells"]),
                    f"headline certified mismatch: {dataset}, {procedure}")
            close(effective, expected_effective, f"headline effective mismatch: {dataset}, {procedure}")
            close(float(official["EffectiveCertCov"]), effective,
                  f"headline file mismatch: {dataset}, {procedure}")


def check_accounting() -> None:
    rows = csv_rows("artifacts/primary/primary_reservoir_accounting.csv")
    require(len(rows) == 2_150, f"reservoir accounting row count mismatch: {len(rows)}")
    require(all(int(row["reservoir_n"]) == int(row["draw_n"]) for row in rows),
            "draw size differed from reservoir size")
    total_draws = sum(int(row["draw_n"]) for row in rows)
    total_unique = sum(int(row["unique_draw_indices"]) for row in rows)
    require(total_draws == 1_229_400, f"total draw count mismatch: {total_draws}")
    require(total_unique == 778_702, f"unique draw count mismatch: {total_unique}")


def check_postcert(primary_rows: list[dict[str, Any]]) -> None:
    validation = json_value("artifacts/postcert/VALIDATION.json")
    require(validation.get("status") == "PASS", "post-certification validation is not PASS")
    require(validation.get("certificate_or_selection_modified") is False,
            "post-certification evaluation modified certification")
    require(validation.get("evaluation_is_a_certificate") is False,
            "evaluation was mislabelled as a certificate")
    require(validation.get("fold_overlap_failures") == 0, "post-certification fold overlap detected")

    per_cell = csv_rows("artifacts/postcert/evaluation_per_certified_cell.csv")
    require(len(per_cell) == 177, f"post-certification row count mismatch: {len(per_cell)}")
    selected = {
        row["semantic_id"]: row for row in csv_rows("artifacts/primary/primary_per_cell_procedures.csv")
        if row["procedure"] == "H" and float(row["alpha"]) == PRIMARY_ALPHA and int(row["certified"]) == 1
    }
    require(len(selected) == 177, "primary H-selected set does not contain 177 cells")
    require({row["semantic_id"] for row in per_cell} == set(selected),
            "post-certification cells differ from the frozen H-selected set")
    for row in per_cell:
        source = selected[row["semantic_id"]]
        require(int(row["selected_candidate_index"]) == int(source["selected_candidate_index"]),
                "post-certification selected member mismatch")
        require(row["selected_selector_sha256"] == source["selected_selector_sha256"],
                "post-certification selector hash mismatch")
        require(all(int(row[field]) == 0 for field in (
            "proposal_certification_id_overlap", "proposal_evaluation_id_overlap",
            "certification_evaluation_id_overlap")), "source-ID overlap detected")

    accepted = sum(int(row["evaluation_A"]) for row in per_cell)
    errors = sum(int(row["evaluation_K"]) for row in per_cell)
    known_errors = sum(int(row["known_accepted_errors"]) for row in per_cell)
    unknown_errors = sum(int(row["unknown_false_acceptances"]) for row in per_cell)
    require((accepted, errors, known_errors, unknown_errors) == (130_574, 12_978, 1_598, 11_380),
            "post-certification component totals mismatch")
    close(errors / accepted, 0.09939191569531454, "post-certification pooled risk mismatch")
    close(unknown_errors / errors, 0.8768685467714594, "unknown-error share mismatch")
    require(sum(float(row["pooled_evaluation_risk"]) > PRIMARY_ALPHA for row in per_cell) == 0,
            "pooled-risk exceedance count mismatch")
    require(sum(float(row["worst_client_evaluation_risk"]) > PRIMARY_ALPHA for row in per_cell) == 3,
            "worst-client exceedance count mismatch")

    aggregate = csv_rows("artifacts/postcert/evaluation_aggregate.csv")
    all_row = next(row for row in aggregate if row["dataset"] == "ALL" and row["condition"] == "ALL")
    require(int(all_row["certified_cells_evaluated"]) == 177, "aggregate certified count mismatch")
    require(int(all_row["total_evaluation_accepted"]) == accepted, "aggregate accepted mismatch")
    require(int(all_row["total_evaluation_accepted_errors"]) == errors, "aggregate errors mismatch")
    require(int(all_row["known_accepted_errors"]) == known_errors, "aggregate known errors mismatch")
    require(int(all_row["unknown_false_acceptances"]) == unknown_errors, "aggregate unknown errors mismatch")


def check_metadata() -> None:
    release = json_value("RELEASE.json")
    require(release.get("release_id") == "fedcore-wr-v3", "release_id mismatch")
    require(release.get("contract_id") == "fedcore-headline-wr-v3", "contract_id mismatch")
    require(release.get("source_binding", {}).get("tag") == "v0.3.0", "source tag mismatch")
    require(release.get("headline", {}).get("certified_cells") == {"H": 177, "S": 177, "B": 130},
            "release headline mismatch")
    primary = json_value("artifacts/primary/VALIDATION.json")
    require(primary.get("status") == "PASS", "primary validation is not PASS")
    require(primary.get("contract_id") == "fedcore-headline-wr-v3", "primary contract mismatch")
    require(primary.get("test_or_evaluation_fold_accessed") is False,
            "primary run accessed test or evaluation data")
    prereg = json_value("governing/PREREGISTRATION.json")
    require(prereg.get("state") == "SEALED_BEFORE_PRIMARY_RESULT_COMPUTATION",
            "primary preregistration state mismatch")
    require(prereg.get("contract_id") == "fedcore-headline-wr-v3", "preregistration contract mismatch")


def main() -> int:
    try:
        check_hash_manifest()
        check_metadata()
        rows = check_primary()
        check_summaries(rows)
        check_accounting()
        check_postcert(rows)
    except ReleaseError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 1
    print("PASS: Fed-CORE WR-v3 numerical release")
    print("primary: 450 cells, 154,800 count rows, H/S/B = 177/177/130")
    print("post-certification: 177 frozen H policies, 12,978 accepted errors, 87.7% unknown false acceptances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
