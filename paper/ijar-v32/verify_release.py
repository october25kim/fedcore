#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_MANUSCRIPT = "d8087bd4b3d422d52a9d8475c374dfc973be8bd053ce4bf52a2bbf87cd9824c8"
EXPECTED_PDF = "11628ff72104e44f894d4cdb412ed93eb673628b40f861fbf641ba00de23aad5"
EXPECTED_TAG = "v0.6.2"


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
        "PROCEDURE_THEOREM_LEDGER.csv", "SUPERSESSION.json", "SHA256SUMS.txt",
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
        raise SystemExit("FAIL v32 manuscript binding")
    if binding.get("rendered_pdf_sha256") != EXPECTED_PDF:
        raise SystemExit("FAIL v32 PDF binding")
    if binding.get("release_tag") != EXPECTED_TAG:
        raise SystemExit("FAIL release tag binding")
    if binding.get("manuscript_included") is not False:
        raise SystemExit("FAIL manuscript inclusion boundary")

    rows = read_rows("CLAIM_ARTIFACT.csv")
    ids = [row["LedgerID"] for row in rows]
    required_ids = (
        {f"F{i:02d}" for i in range(1, 8)}
        | {f"T{i:02d}" for i in range(1, 8)}
        | {"H01", "H02", "H03", "C01", "C02"}
    )
    if len(rows) != 19 or len(set(ids)) != 19 or set(ids) != required_ids:
        raise SystemExit(f"FAIL ledger IDs: {ids}")

    ledger_errors: list[str] = []
    for row in rows:
        if row["Status"] != "GOVERNING_V32":
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

    supersession = json.loads((ROOT / "SUPERSESSION.json").read_text(encoding="utf-8"))
    historical = (ROOT / supersession["historical_artifact"]).resolve()
    marker = historical.parent / "SUPERSEDED_NOT_GOVERNING.md"
    if not historical.is_file() or sha256(historical) != supersession["historical_sha256"]:
        raise SystemExit("FAIL stale PathMNIST pin")
    if not marker.is_file():
        raise SystemExit("FAIL missing stale-artifact marker")
    if supersession["v32_alpha_0_20"]["total"] != "26/50":
        raise SystemExit("FAIL v32 PathMNIST binding")

    # Figure 2 and Abstract: target-matched implementation validity.
    validity = read_rows("artifacts/figures/validity_B1000_per_cell.csv")
    grouped_validity = {
        method: [row for row in validity if row["method"] == method]
        for method in {row["method"] for row in validity}
    }
    for method in ("fedcore_noJ", "bonferroni_delta_over_J", "pooled_CP_intentionally_invalid"):
        if len(grouped_validity.get(method, [])) != 450:
            raise SystemExit(f"FAIL validity denominator parity: {method}")
    if min(float(row["empirical_ucb_coverage"]) for row in grouped_validity["fedcore_noJ"]) < 0.95:
        raise SystemExit("FAIL FedCORE nominal-coverage headline")
    pooled_below = sum(
        float(row["empirical_ucb_coverage"]) < 0.95
        for row in grouped_validity["pooled_CP_intentionally_invalid"]
    )
    if pooled_below != 443:
        raise SystemExit(f"FAIL pooled-coverage headline: {pooled_below} != 443")

    headline = read_rows("artifacts/headline/primary_headline_alpha020.csv")
    all_h = [row for row in headline if row["dataset"] == "ALL" and row["procedure"] == "H"]
    if len(all_h) != 1 or int(all_h[0]["N_cells"]) != 450 or int(all_h[0]["certified_cells"]) != 177:
        raise SystemExit("FAIL 177/450 Holm headline")

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

    print(
        f"PASS IJAR v32 release alignment: {len(expected)} checksums, {len(rows)} ledger entries, "
        "7 figures, 7 tables, 3 abstract claims, 2 retained prose claims, 11 parity groups"
    )


if __name__ == "__main__":
    main()
