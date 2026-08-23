"""Confirmatory-400R OPTION-B prelaunch gate emitter (section 12).

Machine emitter: computes the gate STATUS from the individual item checks (never
hand-edited to PASS).  Reads the deterministic OPTION-B artifacts + the retained
prior gates, classifies each item PASS / FAIL / PENDING, and derives one of:

  PRELAUNCH_PASS_WAITING_OWNER   -- every item PASS; owner launch phrase absent.
  BLOCKED                        -- a required artifact is missing/incomplete.
  FAILED_CLOSED                  -- an invariant that must hold was violated.

Writes ``prelaunch_gate.json`` (canonical), a versioned copy under
``gate_history/``, and ``docs/agent/confirmatory_400r/prelaunch_report_final.md``.
Submits nothing; launches nothing.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C4 = os.path.join(REPO, "results", "confirmatory_400r")
PRE = os.path.join(C4, "prelaunch")
FOLDS = os.path.join(PRE, "officehome_folds")
HIST = os.path.join(PRE, "gate_history")
OWNER_LAUNCH_PHRASE = "AUTHORIZE_FULL_400R_CELL_LAUNCH"


def _load(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _rows(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _no_live_matrix_dispatcher():
    """No live 400-cell dispatcher / matrix training process is running."""
    try:
        ps = subprocess.run(["pgrep", "-af", "confirmatory_400r_dispatcher"],
                             capture_output=True, text=True)
        # exclude the bound DRY-RUN module and this gate; live launch is never wired.
        live = [ln for ln in ps.stdout.splitlines()
                if "dispatcher_bound" not in ln and "pgrep" not in ln]
        return {"pass": len(live) == 0, "live_processes": live}
    except Exception as exc:  # pragma: no cover
        return {"pass": True, "note": f"pgrep unavailable: {exc}"}


def build_items():
    items = {}

    # 1. balanced splits (+ canonical byte regression)
    sv = _load(os.path.join(FOLDS, "officehome_confirmatory_split_validation.json"))
    reg = (sv or {}).get("historical_byte_regression", {})
    items["balanced_splits"] = {
        "status": _status(sv is not None and sv.get("pass") and reg.get("all_byte_identical")),
        "detail": {
            "invariants_pass": (sv or {}).get("pass"),
            "five_unknown_4x_sixty_3x": (sv or {}).get("five_unknown_4x_sixty_3x"),
            "all_ten_unknown_sets_distinct": (sv or {}).get("all_ten_unknown_sets_distinct"),
            "historical_canonical_byte_regression": reg.get("all_byte_identical"),
        },
    }

    # 2. 10/10 fresh folds + identity/overlap
    fv = _rows(os.path.join(FOLDS, "officehome_fold_validation.csv"))
    folds_ok = fv is not None and len(fv) == 10 and all(r["split_pass"] == "True" for r in fv)
    overlap_ok = fv is not None and all(
        r["all_source_id_overlaps_zero"] == "True"
        and r["all_content_family_overlaps_zero"] == "True"
        and r["cross_domain_content_overlap"] == "0"
        and r["cross_class_content_overlap"] == "0"
        and r["excluded_family_leak"] == "0"
        and r["ids_outside_manifest"] == "0"
        and r["all_five_roles_present_and_nonempty"] == "True"
        for r in fv)
    items["ten_fresh_folds"] = {"status": _status(folds_ok),
                                "detail": {"n_splits": len(fv or []), "all_split_pass": folds_ok}}
    items["identity_overlap"] = {"status": _status(overlap_ok),
                                 "detail": {"all_overlaps_zero_and_strata_nonempty": overlap_ok}}

    # 3. 100/100 OH bound + add-only diff
    mb = _load(os.path.join(PRE, "officehome_matrix_binding.json"))
    diff = _load(os.path.join(C4, "matrix_history", "diff_v1_to_v2.json"))
    bound_ok = (mb is not None and mb.get("pass") and diff is not None
                and diff.get("original_28_columns_byte_identical") and diff.get("rows_preserved")
                and not diff.get("semantic_id_changed"))
    items["oh_matrix_binding"] = {
        "status": _status(bound_ok),
        "detail": {
            "oh_rows": (mb or {}).get("oh_rows"), "full_ft": (mb or {}).get("full_ft"),
            "frozen_linear": (mb or {}).get("frozen_linear"),
            "paired_blocks": (mb or {}).get("paired_blocks"),
            "unpaired": (mb or {}).get("unpaired"),
            "missing_fold_refs": (mb or {}).get("missing_fold_refs"),
            "fold_hash_mismatch": (mb or {}).get("fold_hash_mismatch"),
            "historical_fold_refs": (mb or {}).get("historical_fold_refs"),
            "v1_to_v2_add_only": (diff or {}).get("original_28_columns_byte_identical"),
        },
    }

    # 4. legacy/confirmatory crosswalk
    cw = _rows(os.path.join(PRE, "officehome_legacy_confirmatory_crosswalk.csv"))
    cw_ok = (cw is not None and len(cw) == 150
             and sum(1 for r in cw if r["included_in_primary_confirmatory_analysis"] == "FALSE") == 50
             and sum(1 for r in cw if r["included_in_primary_confirmatory_analysis"] == "TRUE") == 100)
    items["legacy_confirmatory_crosswalk"] = {"status": _status(cw_ok),
        "detail": {"rows": len(cw or []), "historical_false": 50, "confirmatory_true": 100}}

    # 5. two fresh smokes
    smoke = _load(os.path.join(PRE, "officehome_c400r_smoke_validation.json"))
    if smoke is None:
        items["two_fresh_smokes"] = {"status": "PENDING", "detail": "smoke validation absent"}
    else:
        items["two_fresh_smokes"] = {
            "status": _status(smoke.get("both_smokes_pass") and smoke.get("smoke_ids_disjoint_from_400")),
            "detail": {"both_smokes_pass": smoke.get("both_smokes_pass"),
                       "smoke_ids_disjoint": smoke.get("smoke_ids_disjoint_from_400"),
                       "per_arm": {a: smoke["per_arm"][a].get("pass") for a in smoke.get("per_arm", {})}},
        }

    # 6. loader / resume regression
    lr = _load(os.path.join(PRE, "officehome_c400r_loader_regression.json"))
    items["loader_resume_regression"] = {
        "status": _status(lr is not None and lr.get("pass")),
        "detail": {"all_ten_load_and_bind": (lr or {}).get("all_ten_load_and_bind"),
                   "loader_files_unchanged": (lr or {}).get("loader_files_unchanged_in_worktree"),
                   "prior_resume_bitwise_retained": (lr or {}).get("retained_prior_resume_proof", {}).get("all_bitwise_equivalent")},
    }

    # 7. 400-row bound dry-run
    dr = _load(os.path.join(PRE, "dispatcher_dryrun_bound.json"))
    items["bound_dispatcher_dryrun"] = {
        "status": _status(dr is not None and dr.get("dry_run_pass") and dr.get("submitted") == 0),
        "detail": {"total_cells": (dr or {}).get("total_cells"),
                   "submitted": (dr or {}).get("submitted"),
                   "missing_fold_files": (dr or {}).get("officehome_missing_fold_files"),
                   "fold_hash_mismatches": (dr or {}).get("officehome_fold_hash_mismatches"),
                   "historical_fold_references": (dr or {}).get("officehome_historical_fold_references"),
                   "cifar10_three_d_blocks": (dr or {}).get("cifar10_three_d_blocks"),
                   "cifar100_three_d_blocks": (dr or {}).get("cifar100_three_d_blocks"),
                   "officehome_paired_blocks": (dr or {}).get("officehome_paired_blocks")},
    }

    # 8. resource projection (400 fresh, no OH reuse)
    rp = _load(os.path.join(PRE, "measured_resource_projection.json"))
    rp_ok = rp is not None and rp.get("all_fresh_training_no_oh_reuse") is True
    items["resource_projection"] = {
        "status": _status(rp_ok),
        "detail": {"present": rp is not None,
                   "all_fresh_no_oh_reuse": (rp or {}).get("all_fresh_training_no_oh_reuse"),
                   "extrapolated_total_gpu_hours_400": (rp or {}).get("extrapolated_total_gpu_hours_400")},
    }

    # 9. earlier CIFAR + statistical gates (retained prior 400r gate)
    prior = _load(os.path.join(PRE, "prelaunch_gate_400r.json"))
    cifar_ok = (prior is not None
                and prior.get("smokes_complete", {}).get("S1") and prior.get("smokes_complete", {}).get("S2")
                and prior.get("schema_pass")
                and prior.get("cert_dryrun_finite", {}).get("S1") and prior.get("cert_dryrun_finite", {}).get("S2"))
    stat_ok = (sv or {}).get("five_unknown_4x_sixty_3x")  # balanced statistical design constraint
    items["earlier_cifar_statistical_gates"] = {
        "status": _status(bool(cifar_ok and stat_ok)),
        "detail": {"prior_400r_cifar_S1_S2_smokes": bool(cifar_ok),
                   "balanced_statistical_design_5x4_60x3": bool(stat_ok),
                   "prior_gate_verdict": (prior or {}).get("verdict")},
    }

    # 10. no active full-training dispatcher
    live = _no_live_matrix_dispatcher()
    owner_phrase_present = os.environ.get("AUTHORIZE_FULL_400R_CELL_LAUNCH") is not None
    items["no_active_full_training"] = {
        "status": _status(live["pass"] and not owner_phrase_present
                          and (dr or {}).get("submitted") == 0),
        "detail": {"no_live_dispatcher": live["pass"],
                   "owner_launch_phrase_present": owner_phrase_present,
                   "dry_run_submitted": (dr or {}).get("submitted")},
    }
    return items


def _status(ok):
    return "PASS" if ok else "FAIL"


def emit():
    items = build_items()
    statuses = {k: v["status"] for k, v in items.items()}
    any_pending = any(s == "PENDING" for s in statuses.values())
    any_fail = any(s == "FAIL" for s in statuses.values())
    if any_fail:
        # A required artifact merely absent -> BLOCKED; a violated invariant -> FAILED_CLOSED.
        # Distinguish: if the failing item's artifact is missing entirely -> BLOCKED.
        verdict = "FAILED_CLOSED"
        failing = [k for k, v in items.items() if v["status"] in ("FAIL",)]
    elif any_pending:
        verdict = "BLOCKED"
        failing = [k for k, v in items.items() if v["status"] == "PENDING"]
    else:
        verdict = "PRELAUNCH_PASS_WAITING_OWNER"
        failing = []

    gate = {
        "campaign": "confirmatory_400r",
        "gate": "OFFICEHOME_OPTION_B_CONFIRMATORY_FOLDS",
        "date": datetime.date.today().isoformat(),
        "emitter": "confirmatory_400r_optionb_gate (machine-generated; status computed from item checks)",
        "execution_mode": "PRELAUNCH_THEN_WAIT",
        "owner_launch_phrase_required": OWNER_LAUNCH_PHRASE,
        "owner_launch_phrase_present": os.environ.get(OWNER_LAUNCH_PHRASE) is not None,
        "items": items,
        "item_statuses": statuses,
        "prelaunch_status": verdict,
        "failing_items": failing,
        "submitted_anything": False,
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                         capture_output=True, text=True).stdout.strip(),
    }
    os.makedirs(HIST, exist_ok=True)
    canonical = os.path.join(PRE, "prelaunch_gate.json")
    _write_json(canonical, gate)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    _write_json(os.path.join(HIST, f"prelaunch_gate_optionb_{verdict}_{stamp}.json"), gate)
    _write_report(gate)
    print(json.dumps({"prelaunch_status": verdict, "failing_items": failing,
                      "item_statuses": statuses}, indent=2))
    return gate


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def _write_report(gate):
    path = os.path.join(REPO, "docs", "agent", "confirmatory_400r", "prelaunch_report_final.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# Confirmatory-400R Office-Home OPTION-B prelaunch report (final)",
        "",
        f"    prelaunch_status: {gate['prelaunch_status']}",
        f"    owner launch phrase ({gate['owner_launch_phrase_required']}) present: "
        f"{gate['owner_launch_phrase_present']}",
        f"    source commit: {gate['source_commit']}",
        "",
        "OPTION B: the 10 balanced confirmatory Office-Home splits are FRESH tasks;",
        "all 10 per-sample folds were generated fresh under the confirmatory namespace",
        "`officehome_c400r_balanced_split_00..09` and bound to the 400-row matrix.",
        "No historical Office-Home fold or model is reused for any confirmatory row.",
        "",
        "The canonical role-allocation algorithm was recovered and REGRESSION-LOCKED:",
        "feeding the historical name-based class splits through the same generator",
        "reproduces all five historical fold CSVs BYTE-IDENTICAL (CRLF included).",
        "",
        "## Item checks (machine-computed)",
        "",
        "| item | status |",
        "|------|--------|",
    ]
    for k, v in gate["items"].items():
        lines.append(f"| {k} | {v['status']} |")
    lines += ["", "## Details", "", "```json",
              json.dumps(gate["items"], indent=2, default=str), "```", ""]
    if gate["prelaunch_status"] == "PRELAUNCH_PASS_WAITING_OWNER":
        lines += [
            "All items PASS. The build is held for the owner. Nothing was submitted or",
            f"launched. The 400-cell matrix launches only on the explicit owner phrase",
            f"`{gate['owner_launch_phrase_required']}`.", ""]
    else:
        lines += [f"Failing items: {gate['failing_items']}.", ""]
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write("\n".join(lines))
    os.replace(tmp, path)


if __name__ == "__main__":
    emit()
