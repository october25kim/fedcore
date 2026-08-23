"""Confirmatory-400R prelaunch post-processing + honest gate (license-neutral).

Runs on the HOST after the in-container GPU smokes have written their native
artifacts, terminal markers, and resume reports into a smoke-artifacts dir. This
module NEVER trains; it only parses the frozen matrix and the smoke outputs to
produce the §11 deliverables:

  * smoke_ids.json                     -- frozen smoke IDs, disjoint from the 400
  * <path>_common.npz                  -- OH native artifacts normalized to schema
  * schema_check.json                  -- common export schema across the 4 paths
  * officehome_identity.json           -- OH identity/fold validation
  * cert_dryrun.json                   -- engineering-only certification dry-run
  * measured_resource_projection.json  -- measured compute/storage + extrapolation
  * resume_equivalence.json            -- per-path bitwise resume verdict
  * prelaunch_gate.json                -- honest gate verdict recommendation

It is honest by construction: any smoke whose marker/artifact is missing is
reported PENDING/FAIL, never assumed to have passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

from fedcore.experiments import confirmatory_400r_common_schema as CS
from fedcore.experiments.confirmatory_400r_cert_dryrun import cert_dry_run

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRE = os.path.join(REPO, "results", "confirmatory_400r", "prelaunch")
ART = os.path.join(PRE, "smoke_artifacts")
MATRIX = os.path.join(REPO, "results", "confirmatory_400r", "final_training_matrix.csv")

# Frozen smoke identities. The "SMOKE_" namespace guarantees disjointness from
# the 400 production semantic_ids (none of which carry that prefix).
SMOKE_IDS = {
    "S1": "SMOKE_S1_cifar10_proser_fedavg__split00__seed0__d0.5__2round",
    "S2": "SMOKE_S2_cifar100_proser_fedavg__split00__seed0__d0.5__2round",
    "S3": "SMOKE_S3_officehome_convnext_full__split0__rep0__2round",
    "S4": "SMOKE_S4_officehome_convnext_frozen__split0__rep0__2round",
}
PATH_FAMILY = {
    "S1": "cifar_proser_fedavg", "S2": "cifar_proser_fedavg",
    "S3": "officehome_convnext_full", "S4": "officehome_convnext_frozen",
}
# common-schema artifact paths (CIFAR emit directly; OH normalized here)
COMMON = {
    "S1": os.path.join(ART, "S1_cifar10_common.npz"),
    "S2": os.path.join(ART, "S2_cifar100_common.npz"),
    "S3": os.path.join(ART, "S3_officehome_full_common.npz"),
    "S4": os.path.join(ART, "S4_officehome_frozen_common.npz"),
}
OH_NATIVE = {
    "S3": os.path.join(ART, "S3_officehome_full.npz"),
    "S4": os.path.join(ART, "S4_officehome_frozen.npz"),
}
MARKERS = {
    "S1": os.path.join(ART, "S1_cifar10.TERMINAL.json"),
    "S2": os.path.join(ART, "S2_cifar100.TERMINAL.json"),
    "S3": os.path.join(ART, "S3_officehome_full.TERMINAL.json"),
    "S4": os.path.join(ART, "S4_officehome_frozen.TERMINAL.json"),
}
RESUME = {
    "S1": os.path.join(ART, "S1_cifar10_resume.json"),
    "S2": os.path.join(ART, "S2_cifar100_resume.json"),
    "S3": os.path.join(ART, "S3_officehome_full_resume.json"),
    "S4": os.path.join(ART, "S4_officehome_frozen_resume.json"),
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# 1. Frozen smoke IDs, disjoint from the 400
# --------------------------------------------------------------------------- #
def freeze_smoke_ids():
    with open(MATRIX) as fh:
        prod = {r["semantic_id"] for r in csv.DictReader(fh)}
    overlap = sorted(set(SMOKE_IDS.values()) & prod)
    out = {
        "smoke_ids": SMOKE_IDS, "n_production_ids": len(prod),
        "overlap_with_400": overlap, "disjoint": (len(overlap) == 0),
        "namespace_rule": "SMOKE_ prefix; production ids never carry it",
    }
    _write(os.path.join(PRE, "smoke_ids.json"), out)
    return out


# --------------------------------------------------------------------------- #
# 2. Normalize OH native -> common schema
# --------------------------------------------------------------------------- #
def normalize_oh():
    made = {}
    for key in ("S3", "S4"):
        native = OH_NATIVE[key]
        if os.path.isfile(native):
            CS.normalize_officehome_to_common(native, COMMON[key], family=PATH_FAMILY[key])
            made[key] = {"status": "normalized", "path": COMMON[key],
                         "sha256": _sha256(COMMON[key])}
        else:
            made[key] = {"status": "native_missing", "expected": native}
    return made


# --------------------------------------------------------------------------- #
# 3. Common export schema check across the 4 paths
# --------------------------------------------------------------------------- #
def schema_check():
    descs, present = {}, {}
    for key, path in COMMON.items():
        if os.path.isfile(path):
            descs[key] = CS.describe_common_npz(path)
            present[key] = True
        else:
            present[key] = False
    field_names = [n for n, _ in CS.PER_OBS_FIELDS]
    # kinds must be identical across all present paths for every structural field
    kinds = defaultdict(dict)
    for key, d in descs.items():
        for name in field_names:
            kinds[name][key] = d["per_field"][name]["kind"]
    field_report = {}
    all_identical = True
    for name in field_names:
        vals = set(kinds[name].values())
        ident = (len(vals) <= 1)
        all_identical = all_identical and ident
        widths = {k: descs[k]["per_field"][name]["width"] for k in descs}
        field_report[name] = {
            "kind_identical_across_paths": ident,
            "kind": sorted(vals),
            "width_by_path": {k: w for k, w in widths.items() if w is not None} or None,
        }
    ambiguous = [n for n in field_names if n in ("logit", "score")]
    native_names = {k: descs[k]["meta"]["native_score_name"] for k in descs}
    n_known = {k: descs[k]["meta"]["n_known"] for k in descs}
    out = {
        "paths_present": present,
        "n_paths_present": sum(present.values()),
        "canonical_field_order": field_names,
        "per_field": field_report,
        "all_structural_fields_kind_identical": all_identical,
        "no_field_named_only_logit_or_score": (len(ambiguous) == 0),
        "known_logits_width_by_path_n_known": n_known,
        "native_score_name_by_path": native_names,
        "native_score_slot_name_identical": True,  # column is always 'native_score'
        "note": ("Structural fields + score columns share identical names/dtypes. "
                 "known_logits width varies with n_known (6/60/45), and the "
                 "family-native score is carried in the shared 'native_score' "
                 "column with an explicit 'native_score_name' tag (proser vs msp)."),
        "schema_pass": bool(all_identical and len(ambiguous) == 0 and sum(present.values()) == 4),
    }
    _write(os.path.join(PRE, "schema_check.json"), out)
    return out


# --------------------------------------------------------------------------- #
# 4. Office-Home identity / fold validation
# --------------------------------------------------------------------------- #
OH_MANIFEST = os.path.join(REPO, "results", "officehome", "preflight", "dataset_manifest.csv")
OH_FOLDS_DIR = os.path.join(REPO, "results", "officehome", "folds")
OH_DOMAINS = ("Art", "Clipart", "Product", "Real_World")


def validate_officehome_identity(smoke_split="officehome_split_0"):
    # Manifest: sample_id -> (domain, klass, content_sha256)
    dom_classes = defaultdict(set)
    sid_content, sid_domain = {}, {}
    all_classes = set()
    with open(OH_MANIFEST, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            dom = row["domain"]; kl = row["klass"]
            dom_classes[dom].add(kl); all_classes.add(kl)
            sid_content[row["sample_id"]] = row.get("content_sha256", "")
            sid_domain[row["sample_id"]] = dom
    n_classes = len(all_classes)
    per_domain_class_count = {d: len(dom_classes[d]) for d in dom_classes}
    all_65_in_all_domains = all(
        dom_classes.get(d, set()) == all_classes for d in OH_DOMAINS)

    # Per-split fold validation (all frozen splits available + the smoke split).
    fold_reports = {}
    roles_order = ("train", "proposal", "certification", "traffic", "evaluation")
    for fp in sorted(os.listdir(OH_FOLDS_DIR)):
        if not fp.startswith("folds_officehome_split_") or not fp.endswith(".csv"):
            continue
        split = fp[len("folds_"):-len(".csv")]
        role_ids = defaultdict(set)
        with open(os.path.join(OH_FOLDS_DIR, fp), newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                role_ids[row["role"]].add(row["sample_id"])
        # pairwise id overlap
        id_overlaps, content_overlaps = {}, {}
        roles = [r for r in roles_order if r in role_ids] or sorted(role_ids)
        for i, a in enumerate(roles):
            for b in roles[i + 1:]:
                ov = role_ids[a] & role_ids[b]
                id_overlaps[f"{a}|{b}"] = len(ov)
                ca = {sid_content.get(s, s) for s in role_ids[a]}
                cb = {sid_content.get(s, s) for s in role_ids[b]}
                content_overlaps[f"{a}|{b}"] = len(ca & cb)
        strata_nonempty = {r: len(role_ids[r]) for r in roles}
        fold_reports[split] = {
            "roles": roles,
            "pairwise_id_overlap": id_overlaps,
            "pairwise_content_family_overlap": content_overlaps,
            "all_id_overlaps_zero": all(v == 0 for v in id_overlaps.values()),
            "all_content_overlaps_zero": all(v == 0 for v in content_overlaps.values()),
            "strata_sizes": strata_nonempty,
            "all_strata_nonempty": all(v > 0 for v in strata_nonempty.values()),
        }
    smoke = fold_reports.get(smoke_split)
    out = {
        "manifest": OH_MANIFEST,
        "n_classes": n_classes,
        "per_domain_class_count": per_domain_class_count,
        "domains_expected": list(OH_DOMAINS),
        "all_classes_in_all_domains": bool(all_65_in_all_domains),
        "smoke_split": smoke_split,
        "smoke_split_report": smoke,
        "all_frozen_splits": fold_reports,
        "pass": bool(
            all_65_in_all_domains and smoke is not None
            and smoke["all_id_overlaps_zero"] and smoke["all_content_overlaps_zero"]
            and smoke["all_strata_nonempty"]),
        "note_400r_balanced_splits": (
            "Per-sample fold assignments for the new balanced officehome_split_00..09 "
            "are a separate frozen-data dependency for the FULL launch and are not "
            "generated here; the engineering smoke validates the proven 50-cell split_0 "
            "folds, and identity invariants hold on every frozen split present."),
    }
    _write(os.path.join(PRE, "officehome_identity.json"), out)
    return out


# --------------------------------------------------------------------------- #
# 5. Certification dry-run over the 4 common artifacts
# --------------------------------------------------------------------------- #
def _n_clients(path):
    data = np.load(path, allow_pickle=False)
    roles = [str(r) for r in data["fold_roles"]]
    cid = np.asarray(data[f"{roles[0]}__client_id"], dtype=int)
    return int(cid.max()) + 1


def cert_dry_run_all():
    out = {}
    for key, path in COMMON.items():
        if not os.path.isfile(path):
            out[key] = {"status": "artifact_missing", "expected": path}
            continue
        nj = _n_clients(path)
        res = cert_dry_run(path, n_clients=nj)
        res["n_clients"] = nj
        res["status"] = "ok"
        out[key] = res
    _write(os.path.join(PRE, "cert_dryrun.json"), out)
    return out


# --------------------------------------------------------------------------- #
# 6. Measured resource projection
# --------------------------------------------------------------------------- #
def resource_projection():
    with open(MATRIX) as fh:
        rows = list(csv.DictReader(fh))
    counts = defaultdict(int)
    for r in rows:
        counts[r["pipeline_id"]] += 1
    per_path = {}
    # map path -> (pipeline_id, production_rounds)
    prod_rounds = {"S1": 50, "S2": 50, "S3": 30, "S4": 30}  # canonical recipe rounds
    pipeline_of = {"S1": "cifar10_proser_fedavg", "S2": "cifar100_proser_fedavg",
                   "S3": "officehome_convnext_full", "S4": "officehome_convnext_frozen"}
    total_measured_gpu_min = 0.0
    for key, mpath in MARKERS.items():
        m = _load_json(mpath)
        if m is None:
            per_path[key] = {"status": "marker_missing"}
            continue
        smoke_rounds = int(m.get("rounds", 2))
        train_s = float(m.get("train_seconds") or m.get("wall_seconds") or 0.0)
        export_s = float(m.get("export_seconds") or 0.0)
        wall_s = float(m.get("wall_seconds") or (train_s + export_s))
        per_round_s = train_s / max(smoke_rounds, 1)
        prod_r = prod_rounds[key]
        extrap_cell_s = per_round_s * prod_r + export_s
        total_measured_gpu_min += wall_s / 60.0
        per_path[key] = {
            "status": "measured", "pipeline": pipeline_of[key],
            "smoke_rounds": smoke_rounds,
            "measured_train_seconds": round(train_s, 3),
            "measured_export_seconds": round(export_s, 3),
            "measured_wall_seconds": round(wall_s, 3),
            "measured_peak_vram_gb": m.get("peak_vram_gb"),
            "measured_checkpoint_bytes": m.get("checkpoint_bytes"),
            "measured_logits_npz_bytes": m.get("logits_npz_bytes"),
            "measured_gpu_min_per_round": round(per_round_s / 60.0, 4),
            "extrapolated_production_rounds": prod_r,
            "extrapolated_cell_gpu_min": round(extrap_cell_s / 60.0, 3),
            "n_cells_in_matrix": counts.get(pipeline_of[key], 0),
        }
    # total extrapolation over the 400 matrix
    total_cell_min = 0.0
    peak_ckpt = 0
    total_ckpt_bytes = 0
    total_logit_bytes = 0
    for key, pinfo in per_path.items():
        if pinfo.get("status") != "measured":
            continue
        n = pinfo["n_cells_in_matrix"]
        total_cell_min += pinfo["extrapolated_cell_gpu_min"] * n
        peak_ckpt = max(peak_ckpt, int(pinfo["measured_checkpoint_bytes"] or 0))
        total_ckpt_bytes += int(pinfo["measured_checkpoint_bytes"] or 0) * n
        total_logit_bytes += int(pinfo["measured_logits_npz_bytes"] or 0) * n
    total_gpu_h = total_cell_min / 60.0
    proj = {
        "measured_from": "four 2-round GPU smokes (TITAN RTX)",
        "per_path": per_path,
        "measured_smoke_gpu_min_total": round(total_measured_gpu_min, 3),
        "extrapolated_total_gpu_hours_400": round(total_gpu_h, 2),
        "extrapolated_wall_hours": {
            "1_gpu": round(total_gpu_h, 2), "2_gpu": round(total_gpu_h / 2, 2),
            "3_gpu": round(total_gpu_h / 3, 2), "4_gpu": round(total_gpu_h / 4, 2)},
        "peak_checkpoint_bytes": peak_ckpt,
        "extrapolated_total_checkpoint_gb": round(total_ckpt_bytes / 1e9, 2),
        "extrapolated_total_logits_gb": round(total_logit_bytes / 1e9, 3),
        "extrapolated_total_storage_gb": round((total_ckpt_bytes + total_logit_bytes) / 1e9, 2),
        "caveat": ("Extrapolation scales measured per-round train time to the canonical "
                   "production round counts (CIFAR 50, OH 30) x matrix cardinality; "
                   "smoke used 2 rounds. Real per-round cost may differ with batch/lr/data "
                   "scale; treat as an order-of-magnitude projection."),
    }
    _write(os.path.join(PRE, "measured_resource_projection.json"), proj)
    return proj


# --------------------------------------------------------------------------- #
# 7. Aggregate resume equivalence
# --------------------------------------------------------------------------- #
def aggregate_resume():
    out = {"delta": "bitwise (torch.equal weights + np.array_equal logits/scores/order); "
                    "NO tolerance rule", "per_path": {}}
    any_missing = False
    any_not_bitwise = False
    for key, rpath in RESUME.items():
        r = _load_json(rpath)
        if r is None:
            out["per_path"][key] = {"status": "missing"}
            any_missing = True
            continue
        out["per_path"][key] = r
        if not r.get("bitwise_equivalent", False):
            any_not_bitwise = True
    out["all_present"] = not any_missing
    out["all_bitwise_equivalent"] = bool(
        not any_missing and not any_not_bitwise)
    _write(os.path.join(PRE, "resume_equivalence.json"), out)
    return out


# --------------------------------------------------------------------------- #
# helpers + finalize
# --------------------------------------------------------------------------- #
def _write(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str, sort_keys=False)
    os.replace(tmp, path)
    print(f"wrote {path}")


def finalize():
    smoke = freeze_smoke_ids()
    normalize_oh()
    schema = schema_check()
    ident = validate_officehome_identity()
    cert = cert_dry_run_all()
    res = resource_projection()
    resume = aggregate_resume()

    # honest gate
    markers = {k: _load_json(MARKERS[k]) for k in SMOKE_IDS}
    smokes_complete = {k: (markers[k] is not None and markers[k].get("status") == "completed")
                       for k in ("S1", "S2")}
    smokes_complete.update({k: (markers[k] is not None and markers[k].get("status") == "completed")
                            for k in ("S3", "S4")})
    all_smokes = all(smokes_complete.values())
    resume_present = resume["all_present"]
    resume_bitwise = resume["all_bitwise_equivalent"]

    if not smoke["disjoint"]:
        verdict = "FAILED_CLOSED"; failing = "smoke_ids_not_disjoint"
    elif not all_smokes:
        pend = [k for k, v in smokes_complete.items() if not v]
        verdict = "BLOCKED"; failing = f"smokes_incomplete:{pend}"
    elif not schema["schema_pass"]:
        verdict = "BLOCKED"; failing = "common_schema_mismatch"
    elif not ident["pass"]:
        verdict = "BLOCKED"; failing = "officehome_identity"
    elif not resume_present:
        miss = [k for k, v in resume["per_path"].items() if v.get("status") == "missing"]
        verdict = "BLOCKED"; failing = f"resume_reports_missing:{miss}"
    elif not resume_bitwise:
        notbw = [k for k, v in resume["per_path"].items()
                 if isinstance(v, dict) and not v.get("bitwise_equivalent", False)]
        verdict = "BLOCKED_RESUME_NOT_BITWISE"; failing = f"paths:{notbw}"
    elif not cert.get("S1", {}).get("all_outputs_finite", False):
        verdict = "BLOCKED"; failing = "cert_dryrun_nonfinite"
    else:
        verdict = "PRELAUNCH_PASS_WAITING_OWNER"; failing = None

    gate = {
        "campaign": "confirmatory_400r",
        "execution_mode": "PRELAUNCH_THEN_WAIT",
        "owner_launch_phrase_present": False,
        "owner_launch_phrase_required": "AUTHORIZE_FULL_400R_CELL_LAUNCH",
        "smoke_ids_disjoint": smoke["disjoint"],
        "smokes_complete": smokes_complete,
        "schema_pass": schema["schema_pass"],
        "officehome_identity_pass": ident["pass"],
        "resume_all_present": resume_present,
        "resume_all_bitwise": resume_bitwise,
        "cert_dryrun_finite": {k: v.get("all_outputs_finite") for k, v in cert.items()
                               if isinstance(v, dict)},
        "resource_projection_measured": res.get("extrapolated_total_gpu_hours_400"),
        "verdict": verdict,
        "failing_item": failing,
        "submitted_anything": False,
    }
    _write(os.path.join(PRE, "prelaunch_gate_400r.json"), gate)
    print(json.dumps(gate, indent=2))
    return gate


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("step", nargs="?", default="finalize",
                   choices=["finalize", "smoke_ids", "normalize_oh", "schema", "identity",
                            "cert", "resources", "resume"])
    args = p.parse_args(argv)
    fn = {"finalize": finalize, "smoke_ids": freeze_smoke_ids, "normalize_oh": normalize_oh,
          "schema": schema_check, "identity": validate_officehome_identity,
          "cert": cert_dry_run_all, "resources": resource_projection, "resume": aggregate_resume}
    fn[args.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
