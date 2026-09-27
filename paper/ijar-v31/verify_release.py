#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_MANUSCRIPT = "e36c9cf9bfe30f4feb887ffb3ba2f6165d628404a99619ab2cf53f3ceafffdcb"

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

    expected = {}
    for raw in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if raw.strip():
            digest, name = raw.split(maxsplit=1)
            expected[name.strip()] = digest
    checksum_errors = [name for name, digest in expected.items()
                       if not (ROOT / name).is_file() or sha256(ROOT / name) != digest]
    if checksum_errors:
        raise SystemExit(f"FAIL checksum mismatch: {checksum_errors}")

    binding = json.loads((ROOT / "MANUSCRIPT_BINDING.json").read_text(encoding="utf-8"))
    if binding.get("manuscript_sha256") != EXPECTED_MANUSCRIPT:
        raise SystemExit("FAIL v31 manuscript binding")
    if binding.get("release_tag") != "v0.6.1":
        raise SystemExit("FAIL release tag binding")

    with (ROOT / "CLAIM_ARTIFACT.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["LedgerID"] for row in rows]
    required_ids = {f"F{i:02d}" for i in range(1, 9)} | {f"T{i:02d}" for i in range(1, 11)} | {"H01", "H02"}
    if len(rows) != 20 or len(set(ids)) != 20 or set(ids) != required_ids:
        raise SystemExit(f"FAIL ledger IDs: {ids}")
    ledger_errors = []
    for row in rows:
        object_path = row["ManuscriptObject"]
        if object_path:
            path = ROOT / object_path
            if not path.is_file() or sha256(path) != row["ManuscriptObjectSHA256"]:
                ledger_errors.append(f"{row['LedgerID']}:manuscript-object")
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

    supersession = json.loads((ROOT / "SUPERSESSION.json").read_text(encoding="utf-8"))
    historical = (ROOT / supersession["historical_artifact"]).resolve()
    marker = historical.parent / "SUPERSEDED_NOT_GOVERNING.md"
    if not historical.is_file() or sha256(historical) != supersession["historical_sha256"]:
        raise SystemExit("FAIL stale PathMNIST pin")
    if not marker.is_file():
        raise SystemExit("FAIL missing stale-artifact marker")
    if supersession["v31_alpha_0_20"]["total"] != "26/50":
        raise SystemExit("FAIL v31 PathMNIST binding")

    # Scientific parity gates: recompute the headline values from the released CSVs.
    validity = read_rows("artifacts/figures/validity_B1000_per_cell.csv")
    grouped_validity = {
        method: [row for row in validity if row["method"] == method]
        for method in {row["method"] for row in validity}
    }
    if any(len(grouped_validity.get(method, [])) != 450 for method in (
        "fedcore_noJ", "bonferroni_delta_over_J", "pooled_CP_intentionally_invalid"
    )):
        raise SystemExit("FAIL validity denominator parity")
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

    anatomy = read_rows("artifacts/figures/failure_anatomy_per_cell_alpha020.csv")
    anatomy_h = [row for row in anatomy if row["procedure"] == "H"]
    certified_h = sum(int(row["certified"]) for row in anatomy_h)
    if len(anatomy_h) != 450 or certified_h != 177 or len(anatomy_h) - certified_h != 273:
        raise SystemExit("FAIL 273-refusal anatomy parity")

    path_frontier = read_rows("artifacts/figures/pathmnist_alpha_frontier.csv")
    path_primary = {
        row["condition"]: row for row in path_frontier if float(row["alpha"]) == 0.20
    }
    if int(path_primary["d0.5"]["certified"]) != 14 or int(path_primary["d5"]["certified"]) != 12:
        raise SystemExit("FAIL PathMNIST 26/50 parity")
    expect_close(float(path_primary["d0.5"]["effective_certified_coverage"]), 0.19469046071792906, "PathMNIST d0.5 ECA")
    expect_close(float(path_primary["d5"]["effective_certified_coverage"]), 0.1892288897281319, "PathMNIST d5 ECA")

    t3 = read_rows("artifacts/tables/t3_primary_comparison.csv")
    t3_all = [row for row in t3 if row["dataset"] == "ALL"]
    if len(t3_all) != 1 or any(int(t3_all[0][key]) != value for key, value in {
        "N": 450, "simplex": 177, "bounded": 199, "cert_wins": 26, "cert_losses": 4
    }.items()):
        raise SystemExit("FAIL corrected T3 parity")

    n2 = read_rows("artifacts/tables/tinyimagenet_n2_cells.csv")
    for procedure in "HSB":
        arm = [row for row in n2 if float(row["alpha"]) == 0.30 and row["procedure"] == procedure]
        if len(arm) != 25 or sum(int(row["certified"]) for row in arm) != 0:
            raise SystemExit(f"FAIL Tiny-ImageNet s0 {procedure} parity")

    def check_n3(relative: str, certified: dict[str, int], eca: dict[str, float]) -> None:
        arm = read_rows(relative)
        rows_030 = {row["procedure"]: row for row in arm if float(row["alpha"]) == 0.30}
        for procedure in "HSB":
            row = rows_030[procedure]
            n_value = int(row.get("N") or row.get("cells") or 0)
            eca_value = float(row.get("ECA") or row.get("effective_certified_coverage") or 0.0)
            if n_value != 25 or int(row["certified"]) != certified[procedure]:
                raise SystemExit(f"FAIL Tiny-ImageNet {relative} {procedure} count parity")
            expect_close(eca_value, eca[procedure], f"Tiny-ImageNet {relative} {procedure} ECA")

    check_n3(
        "artifacts/tables/tinyimagenet_n3_s050_summary.csv",
        {"H": 7, "S": 7, "B": 3},
        {"H": 0.027658613508130453, "S": 0.022913233866353036, "B": 0.006311043892659527},
    )
    check_n3(
        "artifacts/tables/tinyimagenet_n3_s075_summary.csv",
        {"H": 1, "S": 1, "B": 1},
        {"H": 0.001985446798563787, "S": 0.001985446798563787, "B": 0.0008683188131508164},
    )

    central = read_rows("artifacts/figures/central_finding_source.csv")
    def mean_delta(comparator: str) -> float:
        values = [
            float(row["delta_pp"]) for row in central
            if row["cohort"] == "Legacy exports" and row["comparator"] == comparator
            and int(row["budget"]) == 1024 and float(row["protection_t"]) == 1.0
        ]
        if len(values) != 9:
            raise SystemExit(f"FAIL allocation denominator {comparator}")
        return sum(values) / len(values)
    expect_close(mean_delta("bottleneck"), 11.86986597492434, "Joint minus Bottleneck")
    expect_close(mean_delta("deficit_greedy"), 0.5188067444876777, "Joint minus deficit greedy")

    print(f"PASS IJAR v31 release alignment: {len(expected)} checksums, {len(rows)} ledger entries, 8 figures, 10 tables, 2 headline claims, 8 arithmetic parity gates")

if __name__ == "__main__":
    main()
