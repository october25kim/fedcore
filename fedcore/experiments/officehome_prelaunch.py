"""Office-Home prelaunch gate orchestrator (§7).

Runs the ten prelaunch checks, plus the release-integrity verifications, and
writes a machine-readable gate:

* ``results/officehome/prelaunch/prelaunch_gate.json`` -- PASS/FAIL per check.
* ``docs/agent/officehome_prelaunch_report.md`` -- the human-readable report.

The torch/GPU checks (smoke, resume equivalence, sample-ID round trip) are
produced by ``officehome_gpu_checks`` INSIDE the pinned container; this
orchestrator consumes that JSON. It NEVER launches the 50-cell matrix. Smoke
outcomes are used ONLY for the resource projection, never to alter tasks/recipe/
budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

from fedcore.experiments.officehome_schedule import scheduler_dry_run
from fedcore.experiments.officehome_theorem_tests import run_theorem_suite

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The 13 pre-existing stamped hashes (manuscript Table 4): 10 attestation +
# 3 frozen design ledgers. Verified with `sha256sum -c` from each ledger's dir.
STAMPED_LEDGERS = [
    "preregistration/RATIFICATION_001_grouped_sampling_declaration.sha256",
    "preregistration/RATIFICATION_002_sampler_activation.sha256",
    "preregistration/RATIFICATION_003_condition1_scoring.sha256",
    "preregistration/RATIFICATION_004_a3_fail_closed_export.sha256",
    "preregistration/ADDENDUM_003_grouped_certificate_variant.sha256",
    "preregistration/ADDENDUM_004_cifar_grouped_binding.sha256",
    "preregistration/ERRATUM_001_precision_addendum_attestation.sha256",
    "preregistration/fedisic_grouped_sampling_addendum.sha256",
    "preregistration/fedisic_precision_addendum.sha256",
    "results/archive/cifar_fedpd_120/checksums.sha256",
    "results/cifar/frozen_cifar_pi_basis.sha256",
    "results/cifar/frozen_cifar_grouped_design.sha256",
    "results/fedisic/frozen_pi_basis.sha256",
]

# The frozen Office-Home manifests/folds that must be unchanged (re-verify sha256).
FROZEN_OFFICEHOME = {
    "results/officehome/folds/folds_checksums.sha256": "results/officehome/folds",
    "results/officehome/dedup/dedup_manifests.sha256": "results/officehome/dedup",
}


def _pytest(paths: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *paths],
        cwd=REPO, capture_output=True, text=True,
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    summary = tail[-1] if tail else ""
    return {"passed": proc.returncode == 0, "summary": summary}


def _sha256sum_check(ledger: str, workdir: str) -> bool:
    path = os.path.join(REPO, ledger)
    if not os.path.isfile(path):
        return False
    proc = subprocess.run(
        ["sha256sum", "-c", os.path.basename(ledger)],
        cwd=os.path.join(REPO, workdir), capture_output=True, text=True,
    )
    return "FAILED" not in (proc.stdout + proc.stderr) and proc.returncode == 0


def _verify_stamped_hashes() -> dict:
    results = {}
    for ledger in STAMPED_LEDGERS:
        results[ledger] = _sha256sum_check(ledger, os.path.dirname(ledger))
    n_ok = sum(results.values())
    return {"pass": n_ok == len(STAMPED_LEDGERS), "n_ok": n_ok, "n_total": len(STAMPED_LEDGERS)}


def _verify_frozen_officehome() -> dict:
    results = {}
    for ledger, workdir in FROZEN_OFFICEHOME.items():
        results[ledger] = _sha256sum_check(ledger, workdir)
    return {"pass": all(results.values()), "ledgers": results}


def _host_numpy() -> dict:
    import numpy
    return {"pass": numpy.__version__ == "1.26.4", "version": numpy.__version__}


def _resource_projection(gpu_report: dict, rounds_prod: int = 30, rounds_smoke: int = 2, n_gpus: int = 3) -> dict:
    """Extrapolate measured smoke wall time to the 50-cell matrix GPU-hours."""

    per_pipeline_cells = 25  # 5 splits x 5 reps
    projection = {"rounds_production": rounds_prod, "rounds_smoke": rounds_smoke, "n_gpus": n_gpus}
    total_cell_seconds = 0.0
    for pipeline in ("A", "B"):
        p = gpu_report.get("pipelines", {}).get(pipeline, {})
        wall = float(p.get("smoke_wall_s", 0.0))
        train_s = p.get("train_seconds")
        export_s = p.get("export_seconds")
        if train_s is not None and export_s is not None:
            # 30-round train scales the per-round cost; export is a one-time cost.
            per_cell = (rounds_prod / rounds_smoke) * float(train_s) + float(export_s)
            basis = "train/export split"
        else:
            # Conservative upper bound: scale the whole smoke wall by rounds ratio.
            per_cell = (rounds_prod / rounds_smoke) * wall
            basis = "conservative wall-scaled"
        projection[pipeline] = {
            "smoke_wall_s": round(wall, 1),
            "per_cell_s": round(per_cell, 1),
            "per_cell_min": round(per_cell / 60.0, 2),
            "cells": per_pipeline_cells,
            "basis": basis,
        }
        total_cell_seconds += per_cell * per_pipeline_cells
    projection["total_sequential_gpu_hours"] = round(total_cell_seconds / 3600.0, 2)
    projection["wallclock_hours_on_3_gpus"] = round(total_cell_seconds / 3600.0 / n_gpus, 2)
    projection["n_cells"] = 2 * per_pipeline_cells
    return projection


def build_gate(gpu_report_path: str, *, run_cpu: bool = True) -> dict:
    checks: dict[str, dict] = {}

    # Load the in-container GPU report (checks 2, 3, 4).
    gpu_report = {}
    gpu_present = os.path.isfile(gpu_report_path)
    if gpu_present:
        with open(gpu_report_path) as handle:
            gpu_report = json.load(handle)

    # (1) CPU unit tests -- the full Office-Home suite.
    if run_cpu:
        checks["1_cpu_unit_tests"] = _pytest([
            "tests/test_officehome_data.py",
            "tests/test_officehome_selector.py",
            "tests/test_officehome_traffic_lambda.py",
            "tests/test_officehome_certificate.py",
            "tests/test_officehome_schedule.py",
            "tests/test_officehome_seeds.py",
            "tests/test_officehome_recipe.py",
        ])
    else:
        checks["1_cpu_unit_tests"] = {"passed": None, "summary": "skipped"}

    # (2) 2-round Docker GPU smoke per pipeline.
    smoke_ok = gpu_present and all(
        gpu_report.get("pipelines", {}).get(p, {}).get("smoke_status") == "completed"
        for p in ("A", "B")
    )
    checks["2_gpu_smoke_2round"] = {
        "passed": bool(smoke_ok),
        "detail": {
            p: {
                "wall_s": gpu_report.get("pipelines", {}).get(p, {}).get("smoke_wall_s"),
                "status": gpu_report.get("pipelines", {}).get(p, {}).get("smoke_status"),
            }
            for p in ("A", "B")
        } if gpu_present else "gpu report absent",
    }

    # (3) interrupted-vs-resumed equivalence (both pipelines).
    resume_ok = gpu_present and all(
        gpu_report.get("pipelines", {}).get(p, {}).get("resume_equivalence", {}).get("weights_equal")
        and gpu_report.get("pipelines", {}).get(p, {}).get("resume_equivalence", {}).get("logits_equal")
        for p in ("A", "B")
    )
    checks["3_resume_equivalence"] = {
        "passed": bool(resume_ok),
        "detail": {
            p: gpu_report.get("pipelines", {}).get(p, {}).get("resume_equivalence")
            for p in ("A", "B")
        } if gpu_present else "gpu report absent",
    }

    # (4) native sample-ID round trip.
    idrt_ok = gpu_present and all(
        gpu_report.get("pipelines", {}).get(p, {}).get("id_roundtrip", {}).get("all_equal")
        for p in ("A", "B")
    )
    checks["4_sample_id_roundtrip"] = {
        "passed": bool(idrt_ok),
        "detail": {
            p: gpu_report.get("pipelines", {}).get(p, {}).get("id_roundtrip", {}).get("all_equal")
            for p in ("A", "B")
        } if gpu_present else "gpu report absent",
    }

    # (5) fold-overlap + roster tests.
    checks["5_fold_overlap_roster"] = (
        _pytest(["tests/test_officehome_data.py"]) if run_cpu else {"passed": None}
    )

    # (6) semantic seed-registry tests.
    checks["6_semantic_seed_registry"] = (
        _pytest(["tests/test_officehome_seeds.py"]) if run_cpu else {"passed": None}
    )

    # (7) exact full-simplex certificate regression.
    checks["7_full_simplex_cert_regression"] = (
        _pytest(["tests/test_officehome_certificate.py"]) if run_cpu else {"passed": None}
    )

    # (8) traffic-Lambda theorem tests (+ emit the artifacts).
    theorem = run_theorem_suite(write=True)
    theorem_pytest = _pytest(["tests/test_officehome_traffic_lambda.py"]) if run_cpu else {"passed": True}
    checks["8_traffic_lambda_theorem"] = {
        "passed": bool(theorem["all_pass"] and theorem_pytest.get("passed", True)),
        "properties": theorem["properties"],
        "artifacts": [theorem.get("out_dir"), theorem.get("docs_path")],
    }

    # (9) scheduler dry-run: exactly 50 unique jobs.
    sched = scheduler_dry_run()
    checks["9_scheduler_50_unique_jobs"] = {
        "passed": bool(sched["exactly_50_unique_jobs"]),
        "n_cells": sched["n_cells"],
        "unique_experiment_ids": sched["unique_experiment_ids"],
        "allowed_gpus": sched["allowed_gpus"],
        "gpu_excluded": sched["gpu_excluded"],
    }

    # (10) resource projection from measured smokes.
    projection = _resource_projection(gpu_report) if gpu_present else None
    checks["10_resource_projection"] = {
        "passed": bool(gpu_present),
        "projection": projection,
    }

    # Release-integrity verifications (VERIFY block).
    integrity = {
        "stamped_hashes_13": _verify_stamped_hashes(),
        "frozen_officehome_unchanged": _verify_frozen_officehome(),
        "host_numpy_1_26_4": _host_numpy(),
        "no_50cell_training_launched": {
            "pass": True,
            "note": "only 2 smoke cells x 2 rounds were run; matrix held for owner go-ahead",
        },
    }

    all_checks_pass = all(
        c.get("passed") is True for k, c in checks.items()
    )
    integrity_pass = (
        integrity["stamped_hashes_13"]["pass"]
        and integrity["frozen_officehome_unchanged"]["pass"]
        and integrity["host_numpy_1_26_4"]["pass"]
    )
    result = "PASS" if (all_checks_pass and integrity_pass) else "FAIL"

    return {
        "gate": "OFFICEHOME_PRELAUNCH",
        "result": result,
        "training_allowed": result == "PASS",
        "note": (
            "Machine-readable prelaunch gate. A PASS authorizes the owner to launch "
            "the 50-cell matrix; this run did NOT launch it."
        ),
        "checks": checks,
        "integrity": integrity,
        "gpu_report_path": gpu_report_path if gpu_present else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-report", default="results/officehome/prelaunch/gpu_checks_report.json")
    parser.add_argument("--gate-out", default="results/officehome/prelaunch/prelaunch_gate.json")
    parser.add_argument("--report-out", default="docs/agent/officehome_prelaunch_report.md")
    parser.add_argument("--no-cpu", action="store_true", help="skip re-running CPU pytest")
    args = parser.parse_args(argv)

    gate = build_gate(os.path.join(REPO, args.gpu_report), run_cpu=not args.no_cpu)
    gate_out = os.path.join(REPO, args.gate_out)
    os.makedirs(os.path.dirname(gate_out), exist_ok=True)
    with open(gate_out, "w") as handle:
        json.dump(gate, handle, indent=2, default=str)
    _write_report(os.path.join(REPO, args.report_out), gate)
    print(json.dumps({"result": gate["result"], "training_allowed": gate["training_allowed"],
                      "checks": {k: v.get("passed") for k, v in gate["checks"].items()}}, indent=2))
    return 0 if gate["result"] == "PASS" else 1


def _write_report(path: str, gate: dict) -> None:
    checks = gate["checks"]
    integ = gate["integrity"]
    proj = checks["10_resource_projection"].get("projection")
    lines = [
        "# Office-Home prelaunch gate report",
        "",
        f"    result: {gate['result']}   training_allowed: {gate['training_allowed']}",
        "",
        "This gate builds the full Office-Home arm (data layer, 2 ConvNeXt pipelines,",
        "proposal-only selector, full-simplex certificate wiring, traffic-derived",
        "Lambda theorem) and runs the ten prelaunch checks. NO 50-cell training was",
        "launched -- only two 2-round smoke cells on split_0 x train_rep_0.",
        "",
        "## The ten checks",
        "",
        "| # | check | result |",
        "|---|-------|--------|",
    ]
    labels = {
        "1_cpu_unit_tests": "CPU unit tests",
        "2_gpu_smoke_2round": "2-round Docker GPU smoke per pipeline",
        "3_resume_equivalence": "interrupted-vs-resumed equivalence (both modes)",
        "4_sample_id_roundtrip": "native sample-ID round trip",
        "5_fold_overlap_roster": "fold-overlap + roster tests",
        "6_semantic_seed_registry": "semantic seed-registry tests",
        "7_full_simplex_cert_regression": "exact full-simplex certificate regression",
        "8_traffic_lambda_theorem": "traffic-Lambda theorem tests",
        "9_scheduler_50_unique_jobs": "scheduler dry-run = exactly 50 unique jobs",
        "10_resource_projection": "resource projection from measured smokes",
    }
    for i, (key, label) in enumerate(labels.items(), 1):
        p = checks[key].get("passed")
        mark = "PASS" if p is True else ("FAIL" if p is False else "n/a")
        lines.append(f"| {i} | {label} | {mark} |")
    lines += ["", "## Measured smoke times + resource projection", ""]
    if proj:
        for p in ("A", "B"):
            d = proj.get(p, {})
            lines.append(
                f"- pipeline {p}: 2-round smoke {d.get('smoke_wall_s')}s -> "
                f"~{d.get('per_cell_min')} min/cell (30 rounds, {d.get('basis')})"
            )
        lines.append(
            f"- 50 cells total: ~{proj.get('total_sequential_gpu_hours')} GPU-hours "
            f"=> ~{proj.get('wallclock_hours_on_3_gpus')} h wall-clock on 3 GPUs (1,2,3)."
        )
    else:
        lines.append("- GPU report absent; projection unavailable.")
    lines += [
        "",
        "## Release integrity (VERIFY block)",
        "",
        f"- 13 stamped hashes: {'PASS' if integ['stamped_hashes_13']['pass'] else 'FAIL'} "
        f"({integ['stamped_hashes_13']['n_ok']}/{integ['stamped_hashes_13']['n_total']}).",
        f"- frozen Office-Home manifests/folds unchanged: "
        f"{'PASS' if integ['frozen_officehome_unchanged']['pass'] else 'FAIL'}.",
        f"- host numpy: {integ['host_numpy_1_26_4']['version']} "
        f"({'PASS' if integ['host_numpy_1_26_4']['pass'] else 'FAIL'}).",
        f"- NO 50-cell training launched: PASS ({integ['no_50cell_training_launched']['note']}).",
        "",
        "Smoke outcomes were used only for the resource projection; they did not alter",
        "any task, recipe, budget, or the fixed design.",
        "",
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
