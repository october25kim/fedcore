#!/usr/bin/env python3
"""Verify the FedCORE IJAR v37 manuscript and evidence release."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PACS = ROOT / "artifacts" / "pacs"
DOCX_SHA256 = "e0e8934121136fcc09ae49571f50082d22cc82c9a0a752b65eb01cdfb9197acf"
PDF_SHA256 = "f4b768c9aaba7f7305e3282f417fe8fd706f713dfcedfcdf587273d461650f99"
RELEASE_TAG = "v0.7.0"
SOURCE_TAG = "pacs-local-v1-fa016c1"
SOURCE_COMMIT = "fa016c199a7c846bbe20b64d54ab34164adf7647"
SOURCE_TREE = "8adbb6fea6b5f7eef4ea8e7f78b484103bf08024"
IMAGE_ID = "sha256:62c35527e56f5a05c026598ffda355e2b2b80256d966e5d9202ba639a5bfde8a"
PROTOCOL_ID = "FEDCORE-IJAR-V34-PACS-FRESH-SEED-REPLICATION-HSB-v3"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DC_NS = "http://purl.org/dc/elements/1.1/"


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv(path: Path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_gzip_csv(path: Path) -> int:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def normalized_text(value: str) -> str:
    return " ".join(value.split())


def docx_objects(path: Path):
    with zipfile.ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        core = ET.fromstring(archive.read("docProps/core.xml"))
    paragraphs = []
    tables = []

    def paragraph_text(element) -> str:
        pieces = []
        for node in element.iter():
            if node.tag == f"{{{W_NS}}}t":
                pieces.append(node.text or "")
            elif node.tag in {f"{{{W_NS}}}br", f"{{{W_NS}}}cr"}:
                pieces.append("\n")
            elif node.tag == f"{{{W_NS}}}tab":
                pieces.append("\t")
        return "".join(pieces)

    for child in document.find(f"{{{W_NS}}}body"):
        if child.tag == f"{{{W_NS}}}p":
            paragraphs.append(paragraph_text(child))
        elif child.tag == f"{{{W_NS}}}tbl":
            rows = []
            for row in child.findall(f"{{{W_NS}}}tr"):
                cells = []
                for cell in row.findall(f"{{{W_NS}}}tc"):
                    cell_paragraphs = [
                        paragraph_text(paragraph)
                        for paragraph in cell.iter(f"{{{W_NS}}}p")
                    ]
                    cells.append("\n".join(cell_paragraphs))
                rows.append(cells)
            tables.append(rows)
    description = core.find(f"{{{DC_NS}}}description")
    return paragraphs, tables, "" if description is None else description.text or ""


def correlation(left, right) -> float:
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - mean_left) ** 2 for a in left)
        * sum((b - mean_right) ** 2 for b in right)
    )
    return numerator / denominator


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def verify_manifest() -> None:
    manifest = ROOT / "SHA256SUMS.txt"
    require(manifest.is_file(), "SHA256SUMS.txt is missing")
    listed = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, rel = raw.split("  ", 1)
        require(rel not in listed, f"duplicate checksum entry: {rel}")
        listed[rel] = digest
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != manifest
    }
    require(set(listed) == actual, "checksum manifest file set differs from release file set")
    for rel, expected in listed.items():
        require(sha256(ROOT / rel) == expected, f"checksum mismatch: {rel}")


def verify_binding() -> None:
    binding = read_json(ROOT / "MANUSCRIPT_BINDING.json")
    release = read_json(ROOT / "RELEASE.json")
    docx = ROOT / binding["manuscript_filename"]
    pdf = ROOT / binding["rendered_pdf_filename"]
    require(binding["manuscript_version"] == "v37", "wrong manuscript version")
    require(binding["release_tag"] == RELEASE_TAG, "wrong release tag in binding")
    require(binding["pacs_source_tag"] == SOURCE_TAG, "wrong source tag in binding")
    require(binding["manuscript_included"] is True, "manuscript must be included")
    require(sha256(docx) == DOCX_SHA256 == binding["manuscript_sha256"], "DOCX binding mismatch")
    require(sha256(pdf) == PDF_SHA256 == binding["rendered_pdf_sha256"], "PDF binding mismatch")
    _, _, description = docx_objects(docx)
    require(
        description == "IJAR v37 public-release-bound manuscript with complete PACS evidence",
        "DOCX core metadata is stale",
    )
    require(release["release_tag"] == RELEASE_TAG, "wrong tag in RELEASE.json")
    require(release["pacs_source_tag"] == SOURCE_TAG, "wrong PACS source tag")
    require(release["status"] == "VERSIONED_RELEASE", "release is not final")
    require(release["ledger_composition"] == {"figures": 7, "tables": 6, "abstract_claims": 3, "prose_claims": 8}, "wrong ledger composition")
    retained = (ROOT / "RETAINED_NON_GOVERNING.md").read_text(encoding="utf-8")
    require("The sole claim-to-evidence authority for v37 is `CLAIM_ARTIFACT.csv`" in retained, "non-governing boundary is missing")

    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        if inside.returncode == 0 and inside.stdout.strip() == "true":
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
            tagged = subprocess.check_output(
                ["git", "rev-parse", f"{RELEASE_TAG}^{{commit}}"], cwd=REPO, text=True
            ).strip()
            require(head == tagged, f"checkout HEAD is not release tag {RELEASE_TAG}")
    except subprocess.CalledProcessError as exc:
        raise VerificationError(f"release tag cannot be resolved: {RELEASE_TAG}") from exc


def verify_ledger() -> None:
    rows = read_csv(ROOT / "CLAIM_ARTIFACT.csv")
    expected = {*(f"F{i:02d}" for i in range(1, 8)), *(f"T{i:02d}" for i in range(1, 7)), *(f"H{i:02d}" for i in range(1, 4)), *(f"C{i:02d}" for i in range(1, 9))}
    ids = [row["LedgerID"] for row in rows]
    require(len(rows) == 24 and len(set(ids)) == 24 and set(ids) == expected, "ledger ID composition mismatch")
    require(all(row["Status"] == "GOVERNING_V37" for row in rows), "non-v37 governing ledger row")
    docx_path = ROOT / read_json(ROOT / "MANUSCRIPT_BINDING.json")["manuscript_filename"]
    paragraphs, tables, _ = docx_objects(docx_path)
    normalized_paragraphs = {normalized_text(text) for text in paragraphs}
    for row in rows:
        obj = row["ManuscriptObject"]
        require(obj, f"missing manuscript object: {row['LedgerID']}")
        require(sha256(ROOT / obj) == row["ManuscriptObjectSHA256"], f"manuscript object mismatch: {row['LedgerID']}")
        if obj.startswith("manuscript_objects/"):
            payload = read_json(ROOT / obj)
            require(payload["ledger_id"] == row["LedgerID"], f"manuscript object ID mismatch: {row['LedgerID']}")
            if payload["object_type"] == "paragraph":
                require(normalized_text(payload["text"]) in normalized_paragraphs, f"paragraph object absent from DOCX: {row['LedgerID']}")
            elif payload["object_type"] == "numbered_table":
                index = int(payload["table_index"])
                require(index < len(tables) and payload["rows"] == tables[index], f"table object differs from DOCX: {row['LedgerID']}")
                require(normalized_text(payload["caption"]) in normalized_paragraphs, f"table caption absent from DOCX: {row['LedgerID']}")
            else:
                raise VerificationError(f"unknown manuscript object type: {row['LedgerID']}")
        paths = row["GoverningArtifact"].split(";")
        digests = row["ArtifactSHA256"].split(";")
        require(len(paths) == len(digests), f"artifact/hash arity mismatch: {row['LedgerID']}")
        for rel, digest in zip(paths, digests):
            require(sha256(ROOT / rel) == digest, f"governing artifact mismatch: {row['LedgerID']} {rel}")
    table1 = read_csv(ROOT / "artifacts" / "tables" / "closest_method_comparison.csv")
    require(len(table1) == 6, "Table 1 must contain six component rows")
    require(set(table1[0]) == {"established_component", "standard_object", "target_specific_gap", "fedcore_result", "validity_scope", "evidence_scope"}, "Table 1 schema mismatch")


def verify_legacy_governing_claims() -> None:
    validity = read_csv(ROOT / "artifacts" / "figures" / "validity_B1000_per_cell.csv")
    fedcore = [row for row in validity if row["method"] == "fedcore_noJ" and row["selector_proposal_feasible"] == "1"]
    require(len(fedcore) == 443, "target-matched validity denominator must be 443")
    coverages = [float(row["empirical_ucb_coverage"]) for row in fedcore]
    require(min(coverages) >= 0.95 and close(min(coverages), 0.953, 5e-4), "target-matched validity claim mismatch")
    headline = read_csv(ROOT / "artifacts" / "headline" / "primary_headline_alpha020.csv")
    expected_cert = {"H": 177, "S": 177, "B": 130}
    expected_eca = {"H": 0.08342402472238086, "S": 0.0816592568534411, "B": 0.06018240780185294}
    for procedure in ("H", "S", "B"):
        rows = [row for row in headline if row["dataset"] == "ALL" and row["procedure"] == procedure]
        require(len(rows) == 1, f"missing ALL headline row for {procedure}")
        total = int(rows[0]["N_cells"])
        certified = int(rows[0]["certified_cells"])
        eca = float(rows[0]["EffectiveCertCov"])
        require(total == 450 and certified == expected_cert[procedure], f"headline denominator mismatch for {procedure}")
        require(close(eca, expected_eca[procedure]), f"headline ECA mismatch for {procedure}")


def verify_remaining_claim_estimators() -> None:
    overview = read_json(ROOT / "artifacts" / "figures" / "deployment_contract_overview_source.json")
    require(overview["new_id"] == 1 and len(overview["old_ids"]) == 2, "Figure 1 source contract mismatch")
    require("logical dependencies" in overview["scope"], "Figure 1 scope boundary is missing")

    mixture = read_csv(ROOT / "artifacts" / "figures" / "mixture_shift.csv")
    mixture_by_weight = {float(row["bad_client_weight"]): row for row in mixture}
    require(len(mixture) == 9, "Figure 3 traffic-shift grid mismatch")
    require(close(float(mixture_by_weight[0.2]["naive_pooled_coverage"]), 0.913), "Figure 3 traffic value mismatch")
    require(close(float(mixture_by_weight[0.3]["naive_pooled_coverage"]), 0.061), "Figure 3 traffic value mismatch")
    require(close(float(mixture_by_weight[0.4]["naive_pooled_coverage"]), 0.0), "Figure 3 traffic value mismatch")
    require(min(float(row["full_simplex_coverage"]) for row in mixture) >= 0.915, "Figure 3 full-simplex coverage mismatch")
    prevalence = {float(row["audit_unknown_prevalence"]): row for row in read_csv(ROOT / "artifacts" / "figures" / "prevalence_mismatch.csv")}
    require(close(float(prevalence[0.1]["deployment_target_coverage"]), 0.008666666666666666), "Figure 3 prevalence mismatch")
    require(close(float(prevalence[0.3]["deployment_target_coverage"]), 0.929), "Figure 3 matched-prevalence mismatch")
    composition = {float(row["hard_unknown_share_within_unknown"]): row for row in read_csv(ROOT / "artifacts" / "figures" / "composition_mismatch.csv")}
    require(close(float(composition[0.0]["deployment_target_coverage"]), 0.06333333333333334), "Figure 3 composition mismatch")
    require(close(float(composition[0.5]["deployment_target_coverage"]), 0.929), "Figure 3 matched-composition mismatch")

    anatomy = read_csv(ROOT / "artifacts" / "figures" / "failure_anatomy_per_cell_alpha020.csv")
    h_refusals = [row for row in anatomy if row["procedure"] == "H" and row["certified"] == "0"]
    causes = Counter(row["diagnostic_cause"] for row in h_refusals)
    require(len(h_refusals) == 273, "Figure 4 refusal denominator mismatch")
    require(causes == Counter({
        "finite-sample-evidence-margin": 223,
        "selector-tradeoff": 26,
        "empirical-error-only": 9,
        "joint-count-and-empirical": 7,
        "proposal-infeasible": 7,
        "count-only": 1,
    }), "Figure 4 refusal decomposition mismatch")

    betting = read_csv(ROOT / "artifacts" / "figures" / "all_targets_betting_source.csv")
    require(len(betting) == 120, "Figure 5 row denominator mismatch")
    for row in betting:
        require(
            close(float(row["delta_eca"]), float(row["mean_method_eca"]) - float(row["mean_reference_eca"])),
            "Figure 5 paired difference mismatch",
        )
        require(int(row["wins"]) + int(row["losses"]) + int(row["ties"]) == int(row["n_cells"]), "Figure 5 win/loss/tie mismatch")

    split_rows = read_csv(ROOT / "artifacts" / "figures" / "pathmnist_split_primary.csv")
    split_certified = defaultdict(int)
    for row in split_rows:
        split_certified[row["condition"]] += int(row["certified"])
    frontier = read_csv(ROOT / "artifacts" / "figures" / "pathmnist_alpha_frontier.csv")
    frontier_primary = {row["condition"]: int(row["certified"]) for row in frontier if close(float(row["alpha"]), 0.20)}
    require(split_certified == frontier_primary == {"d0.5": 14, "d5": 12}, "Figure 6 PathMNIST aggregation mismatch")

    manifest = read_csv(ROOT / "artifacts" / "governing" / "analysis_input_manifest.csv")
    accounting = read_csv(ROOT / "artifacts" / "governing" / "primary_reservoir_accounting.csv")
    require(len(manifest) == 450 and len({row["semantic_id"] for row in manifest}) == 450, "Table 2 condition roster mismatch")
    require(len(accounting) == 2150 and {row["semantic_id"] for row in accounting} == {row["semantic_id"] for row in manifest}, "Table 2 reservoir accounting mismatch")

    delta = read_csv(ROOT / "artifacts" / "tables" / "delta_sensitivity.csv")
    require(len(delta) == 12, "Table 3 confidence grid mismatch")
    primary_delta = {row["procedure"]: (int(row["certified_cells"]), float(row["effective_certified_acceptance"])) for row in delta if row["joint_confidence"] == "0.90"}
    require(primary_delta == {"H": (177, 0.0834), "S": (177, 0.0817), "B": (130, 0.0602)}, "Table 3 primary row mismatch")

    sweep = read_csv(ROOT / "artifacts" / "tables" / "full_sweep_summary.csv")
    require(len(sweep) == 180 and {float(row["alpha"]) for row in sweep} == {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}, "Table 4 sweep grid mismatch")
    require(all(int(row["N_cells"]) in {25, 50} for row in sweep), "Table 4 cell denominator mismatch")

    procedure = read_csv(ROOT / "artifacts" / "tables" / "procedure_primary_table_source.csv")
    representative = {"RATIO", "CP_IUT_H", "DIRECT_BETTING_H", "SCALE_H"}
    table5 = [row for row in procedure if row["method"] in representative]
    require(len(table5) == 20 and {row["dataset"] for row in table5} == {"cifar10", "cifar100", "officehome", "pathmnist", "ALL"}, "Table 5 representative roster mismatch")

    traffic = read_csv(ROOT / "artifacts" / "tables" / "officehome_traffic_summary.csv")
    traffic_map = {(row["procedure"], float(row["alpha"])): row for row in traffic}
    matched = traffic_map[("FS_matched", 0.20)]
    box = traffic_map[("BOX1000", 0.20)]
    require(int(matched["certified"]) == 74 and close(float(matched["eca"]), 0.0957), "Table 6 matched-simplex row mismatch")
    require(int(box["certified"]) == 92 and close(float(box["eca"]), 0.193), "Table 6 traffic-box row mismatch")

    auroc = read_csv(ROOT / "artifacts" / "analysis" / "auroc_vs_cert.csv")
    x = [float(row["best_auroc"]) for row in auroc]
    y = [float(row["H_ecc"]) for row in auroc]
    require(len(auroc) == 450, "AUROC diagnostic denominator mismatch")
    require(close(correlation(average_ranks(x), average_ranks(y)), 0.6615282312525805), "Spearman diagnostic mismatch")
    require(close(correlation(x, y), 0.570477852534833), "Pearson diagnostic mismatch")
    descending = sorted(range(len(x)), key=x.__getitem__, reverse=True)
    ascending = sorted(range(len(x)), key=x.__getitem__)
    require(sum(auroc[index]["H_certified"] == "0" for index in descending[:45]) == 7, "top-decile refusal mismatch")
    require(sum(auroc[index]["H_certified"] == "1" for index in ascending[:225]) == 13, "lower-half certification mismatch")

    bootstrap = [row for row in read_csv(ROOT / "artifacts" / "analysis" / "paired_hierarchical_bootstrap_alpha020.csv") if row["effect"] == "H_minus_B"]
    require(len(bootstrap) == 10 and all(int(row["bootstrap_replicates"]) == 20000 for row in bootstrap), "bootstrap roster mismatch")
    require(all(float(row["point_difference"]) >= 0.0 for row in bootstrap), "negative H-minus-B point estimate")
    require(sum(float(row["hierarchical_bootstrap_ci_low"]) > 0.0 for row in bootstrap) == 7, "bootstrap positive-interval count mismatch")
    require(sum(close(float(row["hierarchical_bootstrap_ci_low"]), 0.0) for row in bootstrap) == 3, "bootstrap zero-touch count mismatch")

    frontier_rows = read_csv(ROOT / "artifacts" / "analysis" / "h_evidence_margin_audit_frontier_by_dataset.csv")
    all_frontier = next(row for row in frontier_rows if row["dataset"] == "ALL")
    require((int(all_frontier["N_evidence_margin_cells"]), int(all_frontier["N_reachable_at_fixed_rates"]), int(all_frontier["N_unreachable_at_fixed_rates"])) == (223, 222, 1), "audit-frontier denominator mismatch")
    require(close(float(all_frontier["audit_multiplier_median"]), 3.0) and close(float(all_frontier["audit_multiplier_p25"]), 2.0) and close(float(all_frontier["audit_multiplier_p75"]), 5.0), "audit-frontier summary mismatch")
    frontier_validation = read_json(ROOT / "artifacts" / "analysis" / "H_AUDIT_FRONTIER_VALIDATION.json")
    require(frontier_validation["status"] == "PASS" and frontier_validation["minimum_frontier_checks_passed"] == 222, "audit-frontier validation failed")

    comparator = read_csv(ROOT / "artifacts" / "analysis" / "comparator_primary_source.csv")
    scale_vs_betting = next(row for row in comparator if row["A"] == "SCALE_H" and row["B"] == "DIRECT_BETTING_H")
    require(int(scale_vs_betting["N"]) == 50 and (int(scale_vs_betting["wins"]), int(scale_vs_betting["losses"]), int(scale_vs_betting["ties"])) == (1, 0, 49), "closest-comparator roster mismatch")
    require(close(100 * float(scale_vs_betting["delta_ECA"]), 0.06147069234777442), "closest-comparator effect mismatch")
    require(close(100 * float(scale_vs_betting["CI95_low"]), -0.07758567465234415) and close(100 * float(scale_vs_betting["CI95_high"]), 0.200527059347893), "closest-comparator interval mismatch")


def verify_pacs_denominator_and_contract():
    models = read_csv(PACS / "MODEL_STATUS.csv")
    failures = read_csv(PACS / "FAILURE_LEDGER.csv")
    primary = read_csv(PACS / "PRIMARY_512_REP000.csv")
    require(len(models) == 30 and not failures and len(primary) == 90, "PACS complete denominator mismatch")
    require({row["architecture"] for row in models} == {"resnet18", "convnext_tiny"}, "PACS architecture roster mismatch")
    require({int(row["split"]) for row in models} == set(range(5)), "PACS split roster mismatch")
    require({int(row["nominal_training_seed"]) for row in models} == {0, 1, 2}, "PACS seed roster mismatch")
    require(all(row["training_status"] == "terminal_success" and row["model_evaluable"] == "true" for row in models), "non-terminal or unevaluable PACS model")
    require(len({(row["model_id"], row["procedure"]) for row in primary}) == 90, "duplicate PACS procedure row")
    for row in primary:
        require(row["protocol_id"] == PROTOCOL_ID, "PACS protocol ID drift")
        require(close(float(row["alpha"]), 0.20) and close(float(row["delta_r"]), 0.05) and close(float(row["delta_c"]), 0.05), "PACS confidence contract drift")
        require(row["mixture_set"] == "Delta^3" and int(row["budget_total"]) == 512 and int(row["replicate_id"]) == 0, "PACS target or budget drift")
        require(int(row["proposal_feasible_members"]) == 12, "PACS family size drift")
    return models, primary


def verify_pacs_results(models, primary) -> None:
    procedures = ("H", "S", "B")
    selected_expected = {
        "pacs__convnext_tiny__split02__seed0",
        "pacs__convnext_tiny__split02__seed1",
        "pacs__convnext_tiny__split02__seed2",
    }
    expected_eca = {"H": 0.03866772986037437, "S": 0.03866772986037437, "B": 0.03556827642844303}
    expected_cond = {"H": 0.38667729860374367, "S": 0.38667729860374367, "B": 0.35568276428443024}
    for procedure in procedures:
        rows = [row for row in primary if row["procedure"] == procedure]
        certified = [row for row in rows if row["certified"] == "true"]
        require(len(rows) == 30 and len(certified) == 3, f"PACS certified count mismatch for {procedure}")
        require({row["model_id"] for row in certified} == selected_expected, f"PACS selected model set mismatch for {procedure}")
        require(close(sum(float(row["ECA"]) for row in rows) / 30, expected_eca[procedure]), f"PACS ECA mismatch for {procedure}")
        require(close(sum(float(row["coverage_lcb"]) for row in certified) / 3, expected_cond[procedure]), f"PACS conditional coverage mismatch for {procedure}")
    require(close(expected_eca["S"] - expected_eca["B"], 0.003099453431931335), "PACS S-minus-B mismatch")
    summary = read_json(PACS / "PACS_SUMMARY.json")
    require(summary["planned_models"] == summary["evaluable_models"] == summary["accounted_models"] == 30, "PACS summary denominator mismatch")
    require(summary["selected_model_set"] == sorted(selected_expected), "PACS summary model-set mismatch")


def verify_pacs_candidate_and_figure(primary) -> None:
    candidates = read_gzip_csv(PACS / "PRIMARY_CANDIDATE_DECISIONS.csv.gz")
    proposal = read_gzip_csv(PACS / "PROPOSAL_FAMILY_MANIFEST.csv.gz")
    counts = read_gzip_csv(PACS / "PRIMARY_COUNTS.csv.gz")
    require(len(candidates) == 1080 and len(proposal) == 360 and len(counts) == 1440, "PACS primary row counts mismatch")
    require(len({(row["model_id"], row["candidate_index"], row["procedure"]) for row in candidates}) == 1080, "duplicate PACS candidate row")
    expected = {
        "H": (360, 360, 321, 11, 11, 3),
        "S": (360, 360, 321, 11, 11, 3),
        "B": (360, 360, 321, 8, 8, 3),
    }
    derived = {}
    for procedure in expected:
        rows = [row for row in candidates if row["procedure"] == procedure]
        derived[procedure] = (
            len(rows),
            sum(row["proposal_feasible"] == "true" for row in rows),
            sum(row["coverage_positive"] == "true" for row in rows),
            sum(row["risk_pass"] == "true" for row in rows),
            sum(row["candidate_certified"] == "true" for row in rows),
            sum(row["selected"] == "true" for row in rows),
        )
        require(derived[procedure] == expected[procedure], f"candidate gate mismatch for {procedure}")
    displayed = {row["procedure"]: tuple(int(row[key]) for key in ("total_candidates", "proposal_feasible", "positive_coverage_lcb", "risk_gate_passed", "candidate_certified", "selected_and_certified")) for row in read_csv(PACS / "PACS_CANDIDATE_GATE_SUMMARY.csv")}
    require(displayed == expected, "Figure 7 candidate summary mismatch")
    matrix = read_csv(PACS / "PACS_MODEL_CERTIFICATION_MATRIX.csv")
    require(len(matrix) == 10 and all(int(row["total_models"]) == 3 for row in matrix), "Figure 7 model matrix denominator mismatch")
    for row in matrix:
        expected_value = 3 if row["architecture"] == "convnext_tiny" and int(row["split"]) == 2 else 0
        require(all(int(row[f"{procedure}_certified"]) == expected_value for procedure in ("H", "S", "B")), "Figure 7 model matrix value mismatch")


def verify_truth_replay_and_gates(primary) -> None:
    truth = read_csv(PACS / "POST_DECISION_TRUTH.csv")
    require(len(truth) == 90, "PACS truth denominator mismatch")
    reasons = Counter(row["truth_unavailable_reason"] for row in truth)
    require(reasons == Counter({"certificate_refusal": 81, "none": 9}), "PACS truth reason mismatch")
    certified_keys = {(row["model_id"], row["procedure"]) for row in primary if row["certified"] == "true"}
    truth_keys = {(row["model_id"], row["procedure"]) for row in truth if row["truth_unavailable_reason"] == "none"}
    require(certified_keys == truth_keys, "truth availability does not match certified decisions")
    replay = read_json(PACS / "INDEPENDENT_REPLAY.json")
    require(replay["status"] == "PASS_INDEPENDENT_REPLAY" and replay["candidate_rows_replayed"] == 1080 and replay["mismatch_count"] == 0, "independent replay failed")
    require(replay["production_module_imported"] is False, "independent replay imported production module")
    inputs = {
        "candidate_decisions": PACS / "PRIMARY_CANDIDATE_DECISIONS.csv.gz",
        "counts": PACS / "PRIMARY_COUNTS.csv.gz",
        "proposal": PACS / "PROPOSAL_FAMILY_MANIFEST.csv.gz",
    }
    for key, path in inputs.items():
        require(sha256(path) == replay["input_sha256"][key], f"independent replay input mismatch: {key}")
    validation = read_json(PACS / "FINAL_VALIDATION.json")
    require(validation["status"] == "VALID_NEGATIVE", "PACS final classification mismatch")
    require(all(value is False for value in validation["registered_metrics_and_gates"]["H_absolute_gates"].values()), "an absolute H gate unexpectedly passed")
    gain = validation["registered_metrics_and_gates"]["gain_gates"]
    require(gain == {"architecture_means_nonnegative": True, "mean_S_minus_B_at_least_0_01": False, "strict_positive_cell_each_architecture": False}, "PACS gain gates mismatch")


def verify_pacs_chain() -> None:
    completion = read_json(PACS / "EXECUTION_COMPLETION_SEAL.json")
    readiness = read_json(PACS / "provenance" / "EXECUTION_READINESS.json")
    authorization = read_json(PACS / "provenance" / "RUN_AUTHORIZATION.json")
    runtime = read_json(PACS / "provenance" / "RUNTIME_INSPECT_RECEIPT.json")
    terminal = read_json(PACS / "provenance" / "HOST_TERMINAL_RECEIPT.json")
    current = read_json(PACS / "CURRENT_STATE.json")
    final_state = read_json(PACS / current["state_path"])
    lease = read_json(PACS / "CAMPAIGN_LEASE.json")
    require(completion["status"] == "PASS_AUTHORIZED_CAMPAIGN_COMPLETION_BINDING" and completion["campaign_finalized"] is True, "completion seal mismatch")
    require(completion["source_commit"] == SOURCE_COMMIT and completion["image_id"] == IMAGE_ID and completion["planned_cells"] == 30, "completion source/image/denominator mismatch")
    require(completion["readiness_sha256"] == readiness["readiness_sha256"] == "e522ac49cbf081f985ac944d7c34ed831fbb3364bb8e76bd73b22ad6993ee234", "readiness chain mismatch")
    require(completion["run_authorization_sha256"] == authorization["authorization_sha256"] == lease["run_authorization_sha256"], "authorization chain mismatch")
    require(completion["runtime_receipt_sha256"] == runtime["runtime_receipt_sha256"] == lease["runtime_receipt_sha256"], "runtime chain mismatch")
    require(completion["authorization_id"] == authorization["authorization_id"] == runtime["authorization_id"] == lease["authorization_id"], "authorization ID mismatch")
    require(completion["current_state_file_sha256"] == sha256(PACS / "CURRENT_STATE.json"), "CURRENT_STATE file hash mismatch")
    require(current["state_sha256"] == final_state["state_sha256"] == completion["final_state_sha256"], "final state internal hash mismatch")
    require(final_state["campaign_phase"] == "FINALIZED" and final_state["sequence_number"] == 213, "final campaign state mismatch")
    require(terminal["status"] == "PROCESS_EXITED_ZERO" and terminal["returncode"] == 0 and terminal["timed_out"] is False, "terminal receipt does not show a clean process exit")
    require(terminal["container_confirmed_stopped"] is False, "release scope assumes terminal receipt does not prove stopped state")


def verify_source_tag() -> None:
    binding = read_json(PACS / "provenance" / "SOURCE_BINDING.json")
    manifest_path = PACS / "provenance" / "SOURCE_MANIFEST.json"
    manifest = read_json(manifest_path)
    require(binding["source_commit"] == manifest["source_commit"] == SOURCE_COMMIT, "source commit mismatch")
    require(binding["source_tree"] == manifest["source_tree"] == SOURCE_TREE, "source tree mismatch")
    require(binding["runtime_file_count"] == len(manifest["runtime_files"]) == 217, "runtime manifest count mismatch")
    require(binding["source_manifest_sha256"] == sha256(manifest_path), "source manifest file hash mismatch")
    require(binding["execution_image_id"] == IMAGE_ID, "source binding image mismatch")
    temporary = None
    command = ["git"]
    cwd = REPO
    local_probe = subprocess.run(
        ["git", "rev-parse", f"{SOURCE_TAG}^{{commit}}"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    if local_probe.returncode != 0:
        bundle = PACS / "provenance" / "pacs_source.bundle"
        require(bundle.is_file(), "standalone source bundle is missing")
        temporary = tempfile.TemporaryDirectory(prefix="fedcore-v37-source-")
        git_dir = Path(temporary.name) / "source.git"
        subprocess.run(["git", "init", "--bare", "--quiet", str(git_dir)], check=True)
        subprocess.run(
            [
                "git",
                f"--git-dir={git_dir}",
                "fetch",
                "--quiet",
                str(bundle),
                f"refs/tags/{SOURCE_TAG}:refs/tags/{SOURCE_TAG}",
            ],
            check=True,
        )
        command = ["git", f"--git-dir={git_dir}"]
        cwd = ROOT
    try:
        commit = subprocess.check_output([*command, "rev-parse", f"{SOURCE_TAG}^{{commit}}"], cwd=cwd, text=True).strip()
        tree = subprocess.check_output([*command, "rev-parse", f"{SOURCE_TAG}^{{tree}}"], cwd=cwd, text=True).strip()
        require(commit == SOURCE_COMMIT and tree == SOURCE_TREE, "source tag does not resolve to execution source")
        for item in manifest["runtime_files"]:
            try:
                blob = subprocess.check_output([*command, "show", f"{SOURCE_TAG}:{item['path']}"], cwd=cwd)
            except subprocess.CalledProcessError as exc:
                raise VerificationError(f"source-tag file missing: {item['path']}") from exc
            require(len(blob) == item["size_bytes"], f"source-tag size mismatch: {item['path']}")
            require(hashlib.sha256(blob).hexdigest() == item["sha256"], f"source-tag hash mismatch: {item['path']}")
    finally:
        if temporary is not None:
            temporary.cleanup()


def main() -> int:
    checks = [
        ("checksum manifest", verify_manifest),
        ("manuscript binding", verify_binding),
        ("claim-artifact ledger", verify_ledger),
        ("legacy governing claims", verify_legacy_governing_claims),
        ("remaining claim estimators", verify_remaining_claim_estimators),
    ]
    for label, check in checks:
        check()
        print(f"PASS: {label}")
    models, primary = verify_pacs_denominator_and_contract()
    print("PASS: PACS complete denominator and fixed contract")
    verify_pacs_results(models, primary)
    print("PASS: PACS certification and ECA")
    verify_pacs_candidate_and_figure(primary)
    print("PASS: PACS candidate gates and Figure 7")
    verify_truth_replay_and_gates(primary)
    print("PASS: PACS truth, independent replay, and outcome gates")
    verify_pacs_chain()
    print("PASS: PACS readiness-to-completion provenance chain")
    verify_source_tag()
    print("PASS: exact PACS source tag and 217-file runtime manifest")
    print("PASS: FedCORE IJAR v37 release; 7 figures, 6 tables, 3 abstract claims, 8 prose claims")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
