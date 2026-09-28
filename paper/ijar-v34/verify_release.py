#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_MANUSCRIPT = "3fe147f9670c1958da928cd6a45d8b03506098d83f613160c30f57b2e3964ded"
EXPECTED_PDF = "745552f6927743a100b8ed66e8e28c79e8da276b431a5cac65ab1efe7d1b0ba5"
EXPECTED_TAG = "v0.6.4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expect_close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise SystemExit(f"FAIL arithmetic parity {label}: {actual} != {expected}")


def main() -> None:
    required = {
        "README.md", "RELEASE.json", "REPRODUCE.md", "SAMPLING_CONTRACT.md",
        "MANUSCRIPT_BINDING.json", "CLAIM_ARTIFACT.csv",
        "PROCEDURE_THEOREM_LEDGER.csv", "THEORY_CONTRACT.md", "SUPERSESSION.json", "SHA256SUMS.txt",
    }
    missing = sorted(name for name in required if not (ROOT / name).is_file())
    if missing:
        raise SystemExit(f"FAIL missing release files: {missing}")

    expected: dict[str, str] = {}
    for raw in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if raw.strip():
            digest, name = raw.split(maxsplit=1)
            expected[name.strip()] = digest
    actual_files = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt" and "__pycache__" not in path.parts
    }
    if set(expected) != actual_files:
        raise SystemExit(
            f"FAIL checksum file set: missing={sorted(actual_files - set(expected))}, "
            f"stale={sorted(set(expected) - actual_files)}"
        )
    checksum_errors = [
        name for name, digest in expected.items()
        if not (ROOT / name).is_file() or sha256(ROOT / name) != digest
    ]
    if checksum_errors:
        raise SystemExit(f"FAIL checksum mismatch: {checksum_errors}")

    binding = json.loads((ROOT / "MANUSCRIPT_BINDING.json").read_text(encoding="utf-8"))
    if binding.get("manuscript_sha256") != EXPECTED_MANUSCRIPT:
        raise SystemExit("FAIL v34 manuscript binding")
    if binding.get("rendered_pdf_sha256") != EXPECTED_PDF:
        raise SystemExit("FAIL v34 PDF binding")
    if binding.get("release_tag") != EXPECTED_TAG:
        raise SystemExit("FAIL release tag binding")
    if binding.get("manuscript_included") is not False:
        raise SystemExit("FAIL manuscript inclusion boundary")

    rows = read_rows("CLAIM_ARTIFACT.csv")
    ids = [row["LedgerID"] for row in rows]
    required_ids = (
        {f"F{i:02d}" for i in range(1, 8)}
        | {f"T{i:02d}" for i in range(1, 8)}
        | {"H01", "H02", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"}
    )
    if len(rows) != 24 or len(set(ids)) != 24 or set(ids) != required_ids:
        raise SystemExit(f"FAIL ledger IDs: {ids}")

    ledger_errors: list[str] = []
    for row in rows:
        if row["Status"] != "GOVERNING_V34":
            ledger_errors.append(f"{row['LedgerID']}:status")
        object_path = row["ManuscriptObject"]
        if object_path:
            path = ROOT / object_path
            if not path.is_file() or sha256(path) != row["ManuscriptObjectSHA256"]:
                ledger_errors.append(f"{row['LedgerID']}:manuscript-object")
        elif row["ManuscriptObjectSHA256"] != "NOT_APPLICABLE_TEXT_CLAIM":
            ledger_errors.append(f"{row['LedgerID']}:text-object-marker")
        artifacts = row["GoverningArtifact"].split(";")
        hashes = row["ArtifactSHA256"].split(";")
        if len(artifacts) != len(hashes):
            ledger_errors.append(f"{row['LedgerID']}:artifact-hash-cardinality")
            continue
        for artifact, digest in zip(artifacts, hashes):
            path = ROOT / artifact
            if not path.is_file() or sha256(path) != digest:
                ledger_errors.append(f"{row['LedgerID']}:{artifact}")
    if ledger_errors:
        raise SystemExit(f"FAIL ledger mapping: {ledger_errors}")

    conceptual = read_rows("artifacts/tables/closest_method_comparison.csv")
    if len(conceptual) != 5 or conceptual[-1]["method_family"] != "FedCORE":
        raise SystemExit("FAIL Table 1 literature-synthesis pin")

    release = json.loads((ROOT / "RELEASE.json").read_text(encoding="utf-8"))
    if (
        release.get("status") != "VERSIONED_RELEASE"
        or release.get("release_tag") != EXPECTED_TAG
        or release.get("release_asset") != "FedCORE_IJAR_v34_evidence_release.zip"
        or release.get("ledger_composition")
        != {"figures": 7, "tables": 7, "abstract_claims": 2, "prose_claims": 8}
    ):
        raise SystemExit("FAIL v34 release metadata")

    theory = (ROOT / "THEORY_CONTRACT.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    ledger_by_id = {row["LedgerID"]: row for row in rows}
    if (
        "if and only if the credal set is that singleton" not in theory
        or "mechanism diagnostics only" not in theory
        or "not a method ranking" not in readme
        or "Mechanism diagnostic only" not in ledger_by_id["F03"]["ScopeBoundary"]
        or "not a valid comparator" not in ledger_by_id["F02"]["ScopeBoundary"]
    ):
        raise SystemExit("FAIL v34 theory or pooled-mechanism framing")

    procedure_rows = read_rows("PROCEDURE_THEOREM_LEDGER.csv")
    procedure_arms = {row["Arm"] for row in procedure_rows}
    if len(procedure_rows) != 7 or not {
        "H: finite family Holm",
        "S: finite family simple",
        "B: finite family client-divided reference",
    }.issubset(procedure_arms):
        raise SystemExit("FAIL H/S/B procedure-contract ledger")

    supersession = json.loads((ROOT / "SUPERSESSION.json").read_text(encoding="utf-8"))
    if (
        supersession.get("status") != "SUPERSEDED_NOT_GOVERNING"
        or supersession.get("historical_sha256")
        != "e00fc0367c58c966bc37250438866ba6ffc8a7bb33f48f2def5daa90b1cae175"
        or supersession.get("v34_alpha_0_20", {}).get("total") != "26/50"
    ):
        raise SystemExit("FAIL v34 PathMNIST supersession binding")

    # Figure 2 and Abstract: positive-acceptance target-matched validity.
    validity = read_rows("artifacts/figures/validity_B1000_per_cell.csv")
    methods = ("fedcore_noJ", "bonferroni_delta_over_J", "pooled_CP_intentionally_invalid")
    grouped_validity = {
        method: [row for row in validity if row["method"] == method]
        for method in methods
    }
    if any(len(grouped_validity[method]) != 450 for method in methods):
        raise SystemExit("FAIL raw validity roster parity")
    feasible = {
        method: [row for row in grouped_validity[method] if row["selector_proposal_feasible"] == "1"]
        for method in methods
    }
    if any(len(feasible[method]) != 443 for method in methods):
        raise SystemExit("FAIL 443 proposal-feasible validity denominator")
    feasible_ids = [{row["semantic_id"] for row in feasible[method]} for method in methods]
    if any(ids != feasible_ids[0] for ids in feasible_ids[1:]):
        raise SystemExit("FAIL method-specific proposal-feasible identities")
    all_ids = {row["semantic_id"] for row in grouped_validity["fedcore_noJ"]}
    excluded_ids = all_ids - feasible_ids[0]
    placeholders = [row for row in validity if row["semantic_id"] in excluded_ids]
    if len(all_ids) != 450 or len(excluded_ids) != 7 or len(placeholders) != 21:
        raise SystemExit("FAIL seven-placeholder validity exclusion")
    if any(
        row["selector_proposal_feasible"] != "0"
        or row["selector_threshold"] != "inf"
        or float(row["selector_prop_coverage"]) != 0.0
        or int(row["certified_count"]) != 0
        or float(row["empirical_ucb_coverage"]) != 1.0
        for row in placeholders
    ):
        raise SystemExit("FAIL reject-all placeholder semantics")
    if min(float(row["empirical_ucb_coverage"]) for row in feasible["fedcore_noJ"]) < 0.95:
        raise SystemExit("FAIL FedCORE nominal-coverage headline")
    pooled_below = sum(
        float(row["empirical_ucb_coverage"]) < 0.95
        for row in feasible["pooled_CP_intentionally_invalid"]
    )
    if pooled_below != 443:
        raise SystemExit(f"FAIL pooled-coverage headline: {pooled_below} != 443")

    def linear_quantile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])

    validity_expected = {
        "fedcore_noJ": (0.953, 0.994, 1.000, 0),
        "bonferroni_delta_over_J": (0.987, 1.000, 1.000, 0),
        "pooled_CP_intentionally_invalid": (0.000, 0.0702, 0.566, 443),
    }
    for method, (minimum, fifth, median, failures) in validity_expected.items():
        values = [float(row["empirical_ucb_coverage"]) for row in feasible[method]]
        expect_close(min(values), minimum, f"{method} minimum")
        expect_close(linear_quantile(values, 0.05), fifth, f"{method} fifth percentile")
        expect_close(statistics.median(values), median, f"{method} median")
        if sum(value < 0.95 for value in values) != failures:
            raise SystemExit(f"FAIL {method} below-nominal count")

    headline = read_rows("artifacts/headline/primary_headline_alpha020.csv")
    headline_all = {
        row["procedure"]: row for row in headline if row["dataset"] == "ALL"
    }
    expected_headline = {
        "H": (177, 0.08342402472238086),
        "S": (177, 0.0816592568534411),
        "B": (130, 0.06018240780185294),
    }
    if set(headline_all) != set(expected_headline):
        raise SystemExit("FAIL H/S/B headline identities")
    for procedure, (certified, eca) in expected_headline.items():
        row = headline_all[procedure]
        if int(row["N_cells"]) != 450 or int(row["certified_cells"]) != certified:
            raise SystemExit(f"FAIL H/S/B certified-count parity: {procedure}")
        expect_close(float(row["EffectiveCertCov"]), eca, f"H/S/B ECA {procedure}")

    h_eca = float(headline_all["H"]["EffectiveCertCov"])
    s_eca = float(headline_all["S"]["EffectiveCertCov"])
    b_eca = float(headline_all["B"]["EffectiveCertCov"])
    expect_close((h_eca - b_eca) * 100.0, 2.324161692052792, "H-minus-B percentage points")
    expect_close((s_eca - b_eca) * 100.0, 2.147684905158816, "S-minus-B percentage points")
    expect_close((h_eca - s_eca) * 100.0, 0.176476786893975, "H-minus-S percentage points")
    expect_close(h_eca - b_eca, (s_eca - b_eca) + (h_eca - s_eca), "ECA decomposition")

    anatomy_all = read_rows("artifacts/figures/failure_anatomy_per_cell_alpha020.csv")
    decisions = {
        procedure: {
            row["semantic_id"] for row in anatomy_all
            if row["procedure"] == procedure and int(row["certified"]) == 1
        }
        for procedure in ("H", "S", "B")
    }
    if len(decisions["H"]) != 177 or decisions["H"] != decisions["S"]:
        raise SystemExit("FAIL H/S decision-set identity")
    if len(decisions["B"]) != 130 or not decisions["B"].issubset(decisions["H"]):
        raise SystemExit("FAIL B subset relation")
    if len(decisions["H"] - decisions["B"]) != 47:
        raise SystemExit("FAIL 47 additional H/S certifications")

    anatomy = [
        row for row in read_rows("artifacts/figures/failure_anatomy_per_cell_alpha020.csv")
        if row["procedure"] == "H"
    ]
    certified_h = sum(int(row["certified"]) for row in anatomy)
    if len(anatomy) != 450 or certified_h != 177 or len(anatomy) - certified_h != 273:
        raise SystemExit("FAIL 273-refusal anatomy parity")

    # Table 3: all full-simplex confidence-sensitivity cells.
    delta = read_rows("artifacts/tables/delta_sensitivity.csv")
    delta_expected = {
        ("0.90", "H"): (177, 0.0834), ("0.90", "S"): (177, 0.0817), ("0.90", "B"): (130, 0.0602),
        ("0.95", "H"): (157, 0.0742), ("0.95", "S"): (157, 0.0726), ("0.95", "B"): (114, 0.0510),
        ("0.98", "H"): (127, 0.0598), ("0.98", "S"): (127, 0.0588), ("0.98", "B"): (87, 0.0401),
        ("0.99", "H"): (113, 0.0520), ("0.99", "S"): (113, 0.0507), ("0.99", "B"): (70, 0.0326),
    }
    indexed_delta = {(row["joint_confidence"], row["procedure"]): row for row in delta}
    if set(indexed_delta) != set(delta_expected):
        raise SystemExit("FAIL Table 3 cell identities")
    for key, (certified, eca) in delta_expected.items():
        row = indexed_delta[key]
        if int(row["certified_cells"]) != certified:
            raise SystemExit(f"FAIL Table 3 certified parity: {key}")
        expect_close(float(row["effective_certified_acceptance"]), eca, f"Table 3 ECA {key}")

    # Table 4 and Figure 6: CIFAR grid and PathMNIST primary target.
    sweep = read_rows("artifacts/tables/full_sweep_summary.csv")
    cifar_expected = {
        ("cifar10", "d0.1"): [0, 0, 1, 5, 9, 21],
        ("cifar10", "d0.5"): [0, 1, 12, 25, 33, 41],
        ("cifar10", "d5"): [0, 7, 19, 33, 40, 49],
        ("cifar100", "d0.1"): [0, 0, 0, 0, 2, 8],
        ("cifar100", "d0.5"): [0, 0, 0, 3, 13, 37],
        ("cifar100", "d5"): [0, 0, 0, 6, 28, 47],
    }
    alpha_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for (dataset, condition), expected_counts in cifar_expected.items():
        rows_h = {
            float(row["alpha"]): int(row["certified_cells"])
            for row in sweep
            if row["dataset"] == dataset and row["condition"] == condition and row["procedure"] == "H"
        }
        if [rows_h.get(alpha) for alpha in alpha_grid] != expected_counts:
            raise SystemExit(f"FAIL Table 4 count parity: {(dataset, condition)}")

    path_frontier = read_rows("artifacts/figures/pathmnist_alpha_frontier.csv")
    path_primary = {row["condition"]: row for row in path_frontier if float(row["alpha"]) == 0.20}
    if int(path_primary["d0.5"]["certified"]) != 14 or int(path_primary["d5"]["certified"]) != 12:
        raise SystemExit("FAIL PathMNIST 26/50 parity")
    expect_close(float(path_primary["d0.5"]["effective_certified_coverage"]), 0.19469046071792906, "PathMNIST d0.5 ECA")
    expect_close(float(path_primary["d5"]["effective_certified_coverage"]), 0.1892288897281319, "PathMNIST d5 ECA")

    # Table 5: four displayed procedures across four datasets.
    procedures = read_rows("artifacts/tables/procedure_primary_table_source.csv")
    procedure_expected = {
        ("cifar10", "RATIO"): (150, 52, 9.63, 14.27),
        ("cifar10", "CP_IUT_H"): (150, 63, 11.49, 15.23),
        ("cifar10", "DIRECT_BETTING_H"): (150, 76, 16.14, 17.97),
        ("cifar10", "SCALE_H"): (150, 75, 16.25, 18.07),
        ("cifar100", "RATIO"): (150, 2, 0.09, 0.81),
        ("cifar100", "CP_IUT_H"): (150, 9, 0.47, 2.06),
        ("cifar100", "DIRECT_BETTING_H"): (150, 40, 2.53, 4.59),
        ("cifar100", "SCALE_H"): (150, 40, 2.61, 4.75),
        ("officehome", "RATIO"): (100, 69, 10.98, 8.11),
        ("officehome", "CP_IUT_H"): (100, 79, 13.35, 7.96),
        ("officehome", "DIRECT_BETTING_H"): (100, 95, 17.34, 5.83),
        ("officehome", "SCALE_H"): (100, 95, 17.79, 5.96),
        ("pathmnist", "RATIO"): (50, 23, 17.27, 20.75),
        ("pathmnist", "CP_IUT_H"): (50, 26, 19.38, 21.29),
        ("pathmnist", "DIRECT_BETTING_H"): (50, 36, 28.18, 21.52),
        ("pathmnist", "SCALE_H"): (50, 38, 29.98, 20.86),
    }
    proc_index = {(row["dataset"], row["method"]): row for row in procedures}
    for key, (n, certified, mean, sd) in procedure_expected.items():
        row = proc_index.get(key)
        if row is None or int(row["N"]) != n or int(row["certified"]) != certified:
            raise SystemExit(f"FAIL Table 5 count parity: {key}")
        if round(float(row["eca_mean_percent"]), 2) != mean or round(float(row["eca_sd_percent"]), 2) != sd:
            raise SystemExit(f"FAIL Table 5 ECA parity: {key}")

    # Table 6: all Office-Home traffic/protection cells.
    office = read_rows("artifacts/tables/officehome_traffic_summary.csv")
    office_index = {(row["procedure"], float(row["alpha"])): row for row in office}
    office_expected = {
        "FS_headline": [8, 36, 79, 92, 100],
        "FS_matched": [4, 35, 74, 92, 100],
        "BOX250": [2, 39, 92, 98, 100],
        "BOX500": [2, 43, 92, 99, 100],
        "BOX1000": [2, 46, 92, 99, 100],
        "BOX2000": [3, 46, 92, 99, 100],
    }
    office_alphas = [0.10, 0.15, 0.20, 0.25, 0.30]
    for procedure, counts in office_expected.items():
        if [int(office_index[(procedure, alpha)]["certified"]) for alpha in office_alphas] != counts:
            raise SystemExit(f"FAIL Table 6 count parity: {procedure}")

    # Table 7, Figure 7, and Abstract allocation statements.
    central = read_rows("artifacts/figures/central_finding_source.csv")
    def deltas(comparator: str) -> list[float]:
        values = [
            float(row["delta_pp"]) for row in central
            if row["cohort"] == "Legacy exports" and row["comparator"] == comparator
            and int(row["budget"]) == 1024 and float(row["protection_t"]) == 1.0
        ]
        if len(values) != 9:
            raise SystemExit(f"FAIL allocation denominator: {comparator}")
        return values
    bottleneck = deltas("bottleneck")
    deficit = deltas("deficit_greedy")
    expect_close(statistics.mean(bottleneck), 11.86986597492434, "Joint minus Bottleneck")
    expect_close(statistics.mean(deficit), 0.5188067444876777, "Joint minus deficit greedy")
    se = statistics.stdev(deficit) / math.sqrt(len(deficit))
    lower = statistics.mean(deficit) - 2.306004135204166 * se
    upper = statistics.mean(deficit) + 2.306004135204166 * se
    if round(lower, 2) != -0.13 or round(upper, 2) != 1.17:
        raise SystemExit(f"FAIL allocation interval parity: {(lower, upper)}")
    if (sum(value > 0 for value in deficit), sum(value == 0 for value in deficit), sum(value < 0 for value in deficit)) != (3, 5, 1):
        raise SystemExit("FAIL allocation sign parity")

    # Retained quantitative prose: source-disjoint and full-census analyses.
    source_disjoint = read_rows("artifacts/tables/cifar10_1_aggregates.csv")
    strict_rows = [row for row in source_disjoint if float(row["protection_t"]) == 1.0]
    if not strict_rows or any(float(row["mean"]) != 0.0 for row in strict_rows):
        raise SystemExit("FAIL source-disjoint t=1 zero-ECA parity")
    census = read_rows("artifacts/tables/census_all_alpha_summary.csv")
    census_020 = [row for row in census if float(row["alpha"]) == 0.20]
    if len(census_020) != 1 or int(census_020[0]["N"]) != 27 or int(census_020[0]["census_feasible"]) != 26:
        raise SystemExit("FAIL census 26/27 parity")
    expect_close(float(census_020[0]["census_exact_acceptance"]), 0.4088485372532065, "census exact acceptance")

    # Retained AUROC diagnostic: association is not certificate determination.
    auroc = read_rows("artifacts/analysis/auroc_vs_cert.csv")
    if len(auroc) != 450 or len({row["semantic_id"] for row in auroc}) != 450:
        raise SystemExit("FAIL AUROC fixed-roster denominator")

    def average_ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=values.__getitem__)
        ranks = [0.0] * len(values)
        begin = 0
        while begin < len(values):
            end = begin + 1
            while end < len(values) and values[order[end]] == values[order[begin]]:
                end += 1
            rank = (begin + end - 1) / 2.0 + 1.0
            for position in range(begin, end):
                ranks[order[position]] = rank
            begin = end
        return ranks

    def correlation(left: list[float], right: list[float]) -> float:
        left_mean = statistics.mean(left)
        right_mean = statistics.mean(right)
        numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
        denominator = math.sqrt(
            sum((x - left_mean) ** 2 for x in left)
            * sum((y - right_mean) ** 2 for y in right)
        )
        return numerator / denominator

    auroc_values = [float(row["best_auroc"]) for row in auroc]
    eca_values = [float(row["H_ecc"]) for row in auroc]
    expect_close(correlation(auroc_values, eca_values), 0.570477852534833, "AUROC Pearson")
    expect_close(
        correlation(average_ranks(auroc_values), average_ranks(eca_values)),
        0.6615282312525805,
        "AUROC Spearman",
    )
    ordered = sorted(range(len(auroc)), key=lambda index: auroc_values[index], reverse=True)
    if sum(int(auroc[index]["H_certified"]) == 0 for index in ordered[:45]) != 7:
        raise SystemExit("FAIL AUROC top-decile refusal parity")
    if sum(int(auroc[index]["H_certified"]) for index in ordered[225:]) != 13:
        raise SystemExit("FAIL AUROC at-or-below-median certification parity")

    # Fixed-roster hierarchical bootstrap evidence.
    bootstrap = read_rows("artifacts/analysis/paired_hierarchical_bootstrap_alpha020.csv")
    h_minus_b = [row for row in bootstrap if row["effect"] == "H_minus_B"]
    if len(bootstrap) != 30 or len(h_minus_b) != 10:
        raise SystemExit("FAIL bootstrap stratum/effect denominator")
    if any(
        int(row["bootstrap_replicates"]) != 20000 or int(row["bootstrap_seed"]) != 20260825
        for row in bootstrap
    ):
        raise SystemExit("FAIL bootstrap registration parity")
    if any(float(row["point_difference"]) < 0.0 for row in h_minus_b):
        raise SystemExit("FAIL H-minus-B nonnegative point estimates")
    positive_low = sum(float(row["hierarchical_bootstrap_ci_low"]) > 0.0 for row in h_minus_b)
    zero_low = sum(float(row["hierarchical_bootstrap_ci_low"]) == 0.0 for row in h_minus_b)
    if (positive_low, zero_low) != (7, 3):
        raise SystemExit("FAIL H-minus-B interval 7/3 parity")
    cell_effects = read_rows("artifacts/analysis/paired_cell_effects_alpha020.csv")
    if len(cell_effects) != 450 or len({row["semantic_id"] for row in cell_effects}) != 450:
        raise SystemExit("FAIL bootstrap cell-effect denominator")
    replicate_counts: dict[tuple[str, str, str], int] = {}
    replicate_rows = 0
    with gzip.open(
        ROOT / "artifacts/analysis/paired_bootstrap_replicates_alpha020.csv.gz",
        "rt",
        newline="",
        encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["condition"], row["effect"])
            replicate_counts[key] = replicate_counts.get(key, 0) + 1
            replicate_rows += 1
    if replicate_rows != 600000 or len(replicate_counts) != 30 or set(replicate_counts.values()) != {20000}:
        raise SystemExit("FAIL bootstrap replicate-file parity")

    # Retrospective evidence-margin audit planning.
    planning = read_rows("artifacts/analysis/h_evidence_margin_audit_frontier_per_cell.csv")
    planning_status = {row["frontier_status"] for row in planning}
    if len(planning) != 223 or planning_status != {"finite", "structural-infinity-at-fixed-rates"}:
        raise SystemExit("FAIL planning cell denominator/status")
    if sum(row["frontier_status"] == "finite" for row in planning) != 222:
        raise SystemExit("FAIL planning 222/223 finite frontier")
    by_dataset = read_rows("artifacts/analysis/h_evidence_margin_audit_frontier_by_dataset.csv")
    all_planning = [row for row in by_dataset if row["dataset"] == "ALL"]
    if len(all_planning) != 1:
        raise SystemExit("FAIL planning ALL summary")
    all_row = all_planning[0]
    if (
        int(all_row["N_evidence_margin_cells"]),
        int(all_row["N_reachable_at_fixed_rates"]),
        int(all_row["N_unreachable_at_fixed_rates"]),
        float(all_row["audit_multiplier_p25"]),
        float(all_row["audit_multiplier_median"]),
        float(all_row["audit_multiplier_p75"]),
    ) != (223, 222, 1, 2.0, 3.0, 5.0):
        raise SystemExit("FAIL planning 223/222/1 and IQR parity")
    planning_validation = json.loads(
        (ROOT / "artifacts/analysis/H_AUDIT_FRONTIER_VALIDATION.json").read_text(encoding="utf-8")
    )
    if (
        planning_validation.get("status") != "PASS"
        or planning_validation.get("target_cells") != 223
        or planning_validation.get("finite_frontier_cells") != 222
        or planning_validation.get("structural_infinity_cells") != 1
        or planning_validation.get("new_sampling_performed") is not False
    ):
        raise SystemExit("FAIL planning validation contract")

    # Closest established Office-Home comparator.
    comparator = read_rows("artifacts/analysis/comparator_primary_source.csv")
    comparator_index = {(row["A"], row["B"]): row for row in comparator}
    comparison = comparator_index.get(("SCALE_H", "DIRECT_BETTING_H"))
    if comparison is None or len(comparator) != 3:
        raise SystemExit("FAIL Office-Home comparator identities")
    if (
        float(comparison["alpha"]),
        int(comparison["m"]),
        int(comparison["N"]),
        int(comparison["wins"]),
        int(comparison["losses"]),
        int(comparison["ties"]),
    ) != (0.2, 1000, 50, 1, 0, 49):
        raise SystemExit("FAIL Office-Home comparator denominator/sign parity")
    expect_close(float(comparison["delta_ECA"]), 0.0006147069234777442, "SCALE-H minus BETTING-H")
    expect_close(float(comparison["CI95_low"]), -0.0007758567465234415, "Office-Home CI low")
    expect_close(float(comparison["CI95_high"]), 0.00200527059347893, "Office-Home CI high")
    completed_audit = json.loads(
        (ROOT / "artifacts/analysis/COMPLETED_COMPARISON_AUDIT.json").read_text(encoding="utf-8")
    )
    primary_contrast = [
        row for row in completed_audit.get("new_direct_contrasts", [])
        if row.get("A") == "SCALE_H"
        and row.get("B") == "DIRECT_BETTING_H"
        and float(row.get("alpha")) == 0.2
        and int(row.get("m")) == 1000
    ]
    if (
        completed_audit.get("status") != "PASS"
        or len(completed_audit.get("cells", [])) != 50
        or completed_audit.get("new_training") is not False
        or completed_audit.get("new_model_inference") is not False
        or completed_audit.get("new_audit_draws") != 0
        or len(primary_contrast) != 1
        or primary_contrast[0].get("certified_A") != 50
        or primary_contrast[0].get("certified_B") != 50
    ):
        raise SystemExit("FAIL completed Office-Home comparator audit")

    # CIFAR-10.1 source identity and exact-pixel audit.
    source_audit = json.loads(
        (ROOT / "artifacts/source_audit/SOURCE_AUDIT.json").read_text(encoding="utf-8")
    )
    source_ledger = json.loads(
        (ROOT / "artifacts/source_audit/SOURCE_LEDGER.json").read_text(encoding="utf-8")
    )
    upstream_head = json.loads(
        (ROOT / "artifacts/source_audit/CIFAR10_1_HEAD.json").read_text(encoding="utf-8")
    )
    if (
        source_audit.get("status") != "PASS"
        or source_audit.get("new_rows") != 2000
        or source_audit.get("retained_rows") != 2000
        or source_audit.get("unique_upstream_source_ids") != 2000
        or source_audit.get("exact_reference_overlaps") != 0
        or source_audit.get("within_new_pixel_duplicates") != 0
        or source_audit.get("reference_rows") != 60000
        or source_audit.get("fresh_label_array_loaded") is not False
        or source_audit.get("model_outputs_loaded") is not False
        or len(source_ledger) != 2000
        or source_audit.get("source_ledger_sha256")
        != sha256(ROOT / "artifacts/source_audit/SOURCE_LEDGER.json")
        or upstream_head.get("sha") != "d9982abb0bfc4846b8d13a11e66b887d946205d0"
    ):
        raise SystemExit("FAIL CIFAR-10.1 source-audit parity")

    print(
        f"PASS IJAR v34 release alignment: {len(expected)} checksums, {len(rows)} ledger entries, "
        "7 figures, 7 tables, 2 abstract claims, 8 retained prose claims, 18 parity groups"
    )


if __name__ == "__main__":
    main()
