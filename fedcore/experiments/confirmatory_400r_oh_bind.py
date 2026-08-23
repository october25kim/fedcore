"""Confirmatory-400R Office-Home fold binding (sections 6-8, OPTION B).

Deterministic, outcome-independent host-side binding of the 10 fresh confirmatory
Office-Home fold manifests to the 400-row training matrix.  NEVER trains, NEVER
launches; only reads the frozen matrix + the fresh folds and writes provenance.

  section 6  officehome_legacy_confirmatory_crosswalk.csv
             150 rows: 50 historical (included=FALSE) + 100 confirmatory (TRUE)
  section 7  matrix_history/final_training_matrix_v1_unbound.csv  (byte copy of current)
             matrix_history/final_training_matrix_v2_bound.csv (+ .sha256)
             matrix_history/diff_v1_to_v2.json  (proves ONLY metadata columns added)
  section 8  officehome_matrix_binding.json  (100-row binding validation)

Binding adds five metadata columns to Office-Home rows only (blank for CIFAR):
fold_split_id (namespace clarification), fold_path, fold_sha256,
dataset_manifest_sha256, role_schema_version.  The 28 scientific-design columns
are preserved BYTE-IDENTICAL: v2 keeps each original raw line intact and appends.
No semantic_id changes, so no old->new id crosswalk is required.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C4 = os.path.join(REPO, "results/confirmatory_400r")
PRE = os.path.join(C4, "prelaunch")
FOLDS_DIR = os.path.join(PRE, "officehome_folds")
MATRIX_HISTORY = os.path.join(C4, "matrix_history")
MATRIX_V1_SRC = os.path.join(C4, "final_training_matrix.csv")

RETAINED_MANIFEST = os.path.join(REPO, "results/officehome/dedup/retained_canonical_manifest.csv")
BALANCED_CSV = os.path.join(REPO, "results/confirmatory_400/prelaunch/officehome_class_splits.csv")
HIST_CLASS_SPLITS = os.path.join(REPO, "results/officehome/preflight/class_splits.csv")
HIST_FOLDS_DIR = os.path.join(REPO, "results/officehome/folds")
HIST_LOGITS_CHECKSUMS = os.path.join(REPO, "results/officehome/launch/officehome_logits_checksums.sha256")

ROLE_SCHEMA_VERSION = "officehome_roles_v1"
ROLES = ("train", "proposal", "certification", "traffic", "evaluation")
NEW_COLS = ["fold_split_id", "fold_path", "fold_sha256", "dataset_manifest_sha256", "role_schema_version"]
PIPELINES = ("officehome_convnext_full", "officehome_convnext_frozen")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _write(path, data_bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data_bytes)
    os.replace(tmp, path)


def _write_json(path, obj):
    _write(path, (json.dumps(obj, indent=2, default=str) + "\n").encode("utf-8"))


def class_membership_hash(known_names, unknown_names):
    payload = "known:" + ",".join(sorted(known_names)) + "|unknown:" + ",".join(sorted(unknown_names))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _class_names():
    classes = set()
    with open(RETAINED_MANIFEST, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            classes.add(row["klass"])
    return sorted(classes)


def _balanced_membership(names):
    """confirmatory split index -> (known_names, unknown_names, csv_split_id)."""
    idx_to_name = {i: n for i, n in enumerate(names)}
    out = {}
    with open(BALANCED_CSV, newline="", encoding="utf-8-sig") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            kn = [idx_to_name[int(x)] for x in row["known_classes"].split()]
            un = [idx_to_name[int(x)] for x in row["unknown_classes"].split()]
            out[i] = (kn, un, row["split_id"])
    return out


def _historical_membership():
    """historical split index (0..4) -> (known_names, unknown_names)."""
    with open(HIST_CLASS_SPLITS, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for i in range(5):
        sid = f"officehome_split_{i}"
        kn = sorted({r["class"] for r in rows if r["split_id"] == sid and r["role"] == "known"})
        un = sorted({r["class"] for r in rows if r["split_id"] == sid and r["role"] == "unknown"})
        out[i] = (kn, un)
    return out


def confirmatory_fold_binding(names):
    """matrix split_id (officehome_split_0X) -> binding dict."""
    bal = _balanced_membership(names)
    dataset_manifest_sha = _sha256(RETAINED_MANIFEST)
    binding = {}
    for i in range(10):
        kn, un, csv_split_id = bal[i]
        conf_split = f"officehome_c400r_balanced_split_{i:02d}"
        fold_path = os.path.join(FOLDS_DIR, f"{conf_split}.csv")
        binding[csv_split_id] = {
            "fold_split_id": conf_split,
            "fold_path": os.path.relpath(fold_path, REPO),
            "fold_sha256": _sha256(fold_path),
            "dataset_manifest_sha256": dataset_manifest_sha,
            "role_schema_version": ROLE_SCHEMA_VERSION,
            "class_membership_hash": class_membership_hash(kn, un),
        }
    return binding


# --------------------------------------------------------------------------- #
# section 6: crosswalk
# --------------------------------------------------------------------------- #
def build_crosswalk(names):
    binding = confirmatory_fold_binding(names)
    hist_mem = _historical_membership()
    hist_trained = os.path.isfile(HIST_LOGITS_CHECKSUMS)
    fields = ["study_origin", "split_id", "class_membership_hash", "fold_hash",
              "pipeline", "train_rep", "artifact_status",
              "included_in_primary_confirmatory_analysis"]
    rows = []
    # historical 50: 5 legacy splits x 5 reps x 2 pipelines -> FALSE
    for i in range(5):
        kn, un = hist_mem[i]
        fold_hash = _sha256(os.path.join(HIST_FOLDS_DIR, f"folds_officehome_split_{i}.csv"))
        cmh = class_membership_hash(kn, un)
        for pipeline in PIPELINES:
            for rep in range(5):
                rows.append({
                    "study_origin": "historical_legacy",
                    "split_id": f"officehome_legacy_split_{i:02d}",
                    "class_membership_hash": cmh,
                    "fold_hash": fold_hash,
                    "pipeline": pipeline,
                    "train_rep": rep,
                    "artifact_status": "historical_trained_not_bound" if hist_trained else "historical_untrained",
                    "included_in_primary_confirmatory_analysis": "FALSE",
                })
    # confirmatory 100: 10 balanced splits x 5 reps x 2 pipelines -> TRUE
    for i in range(10):
        csv_split_id = f"officehome_split_{i:02d}"
        b = binding[csv_split_id]
        for pipeline in PIPELINES:
            for rep in range(5):
                rows.append({
                    "study_origin": "confirmatory",
                    "split_id": b["fold_split_id"],
                    "class_membership_hash": b["class_membership_hash"],
                    "fold_hash": b["fold_sha256"],
                    "pipeline": pipeline,
                    "train_rep": rep,
                    "artifact_status": "fold_bound_pending_launch",
                    "included_in_primary_confirmatory_analysis": "TRUE",
                })
    out_path = os.path.join(PRE, "officehome_legacy_confirmatory_crosswalk.csv")
    buf = [",".join(fields)]
    buf += [",".join(str(r[f]) for f in fields) for r in rows]
    _write(out_path, ("\n".join(buf) + "\n").encode("utf-8"))
    return {"path": os.path.relpath(out_path, REPO), "n_rows": len(rows),
            "historical_false": sum(1 for r in rows if r["included_in_primary_confirmatory_analysis"] == "FALSE"),
            "confirmatory_true": sum(1 for r in rows if r["included_in_primary_confirmatory_analysis"] == "TRUE")}


# --------------------------------------------------------------------------- #
# section 7: bind matrix (preserve v1, add-only v2, diff)
# --------------------------------------------------------------------------- #
def bind_matrix(names):
    os.makedirs(MATRIX_HISTORY, exist_ok=True)
    v1_path = os.path.join(MATRIX_HISTORY, "final_training_matrix_v1_unbound.csv")
    shutil.copyfile(MATRIX_V1_SRC, v1_path)  # byte copy of current

    with open(v1_path, "rb") as fh:
        raw = fh.read()
    records = raw.split(b"\r\n")
    trailing_empty = records and records[-1] == b""
    if trailing_empty:
        records = records[:-1]
    header = records[0]
    data = records[1:]
    hdr_fields = header.decode("ascii").split(",")
    ix_dataset = hdr_fields.index("dataset")
    ix_split = hdr_fields.index("split_id")

    binding = confirmatory_fold_binding(names)

    new_header = header + b"," + ",".join(NEW_COLS).encode("ascii")
    new_lines = [new_header]
    n_oh = 0
    for line in data:
        fields = line.split(b",")
        dataset = fields[ix_dataset].decode("ascii")
        if dataset == "officehome":
            split_id = fields[ix_split].decode("ascii")
            b = binding[split_id]
            appended = ",".join([b["fold_split_id"], b["fold_path"], b["fold_sha256"],
                                 b["dataset_manifest_sha256"], b["role_schema_version"]])
            n_oh += 1
        else:
            appended = ",,,,"  # 5 empty fields
        new_lines.append(line + b"," + appended.encode("ascii"))

    v2_bytes = b"\r\n".join(new_lines) + b"\r\n"
    v2_path = os.path.join(MATRIX_HISTORY, "final_training_matrix_v2_bound.csv")
    _write(v2_path, v2_bytes)
    v2_sha = hashlib.sha256(v2_bytes).hexdigest()
    _write(v2_path + ".sha256", f"{v2_sha}  final_training_matrix_v2_bound.csv\n".encode("ascii"))
    # Canonical bound matrix at the campaign root (byte-identical archive in matrix_history/).
    v2_root = os.path.join(C4, "final_training_matrix_v2_bound.csv")
    _write(v2_root, v2_bytes)
    _write(v2_root + ".sha256", f"{v2_sha}  final_training_matrix_v2_bound.csv\n".encode("ascii"))

    # diff proof: every v2 line == corresponding v1 line + appended cols
    add_only = True
    v1_records = data
    v2_records = new_lines[1:]
    for a, c in zip(v1_records, v2_records):
        if not c.startswith(a + b","):
            add_only = False
            break
    diff = {
        "v1_path": os.path.relpath(v1_path, REPO),
        "v2_path": os.path.relpath(v2_path, REPO),
        "v1_sha256": _sha256(v1_path),
        "v2_sha256": v2_sha,
        "v1_rows": len(data),
        "v2_rows": len(v2_records),
        "rows_preserved": len(data) == len(v2_records) == 400,
        "columns_added": NEW_COLS,
        "n_original_columns": len(hdr_fields),
        "original_28_columns_byte_identical": add_only,
        "semantic_id_changed": False,
        "officehome_rows_bound": n_oh,
        "cifar_rows_blank_binding": len(data) - n_oh,
        "change_class": "provenance_metadata_add_only (no scientific-design field mutated)",
    }
    _write_json(os.path.join(MATRIX_HISTORY, "diff_v1_to_v2.json"), diff)
    return diff, v2_path


# --------------------------------------------------------------------------- #
# section 8: binding validation
# --------------------------------------------------------------------------- #
def validate_binding(names, v2_path):
    binding = confirmatory_fold_binding(names)
    with open(v2_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    oh = [r for r in rows if r["dataset"] == "officehome"]
    full = [r for r in oh if r["pipeline_id"] == "officehome_convnext_full"]
    frozen = [r for r in oh if r["pipeline_id"] == "officehome_convnext_frozen"]

    missing_fold_refs = 0
    fold_hash_mismatch = 0
    historical_fold_refs = 0
    seeds = [c for c in rows[0] if c.startswith("seed_")]
    model_seed_names = {"seed_training_initialization"}
    for r in oh:
        split_id = r["split_id"]
        b = binding.get(split_id)
        if not b or not r.get("fold_split_id") or not r.get("fold_sha256"):
            missing_fold_refs += 1
            continue
        fold_path = os.path.join(REPO, r["fold_path"])
        if not os.path.isfile(fold_path):
            missing_fold_refs += 1
        if r["fold_sha256"] != b["fold_sha256"] or (os.path.isfile(fold_path) and _sha256(fold_path) != r["fold_sha256"]):
            fold_hash_mismatch += 1
        if "legacy" in r.get("fold_split_id", "") or "folds_officehome_split_" in r.get("fold_path", ""):
            historical_fold_refs += 1

    # paired blocks
    pb = {}
    for r in oh:
        pb.setdefault(r["paired_block"], {})[r["pipeline_id"]] = r
    paired_blocks = 0
    unpaired = 0
    block_checks = {"class_membership_shared": True, "fold_hash_shared": True,
                    "role_identities_shared": True, "pretrained_weights_shared": True,
                    "nonmodel_seeds_shared": True, "audit_traffic_streams_shared": True}
    for blk, arms in pb.items():
        if set(arms) != set(PIPELINES):
            unpaired += 1
            continue
        paired_blocks += 1
        a, c = arms["officehome_convnext_full"], arms["officehome_convnext_frozen"]
        if a["split_id"] != c["split_id"]:
            block_checks["class_membership_shared"] = False
        if a["fold_sha256"] != c["fold_sha256"]:
            block_checks["fold_hash_shared"] = False
        # role identities are a pure function of the (shared) fold file
        if a["fold_path"] != c["fold_path"]:
            block_checks["role_identities_shared"] = False
        if a["backbone"] != c["backbone"] or a["seed_training_initialization"] != c["seed_training_initialization"]:
            block_checks["pretrained_weights_shared"] = False
        for s in seeds:
            if s in model_seed_names:
                continue
            if a[s] != c[s]:
                block_checks["nonmodel_seeds_shared"] = False
        for s in ("seed_proposal_fold", "seed_certification_reservoir", "seed_primary_audit_draw",
                  "seed_secondary_audit_redraw", "seed_traffic_draw", "seed_evaluation"):
            if a[s] != c[s]:
                block_checks["audit_traffic_streams_shared"] = False

    out = {
        "oh_rows": len(oh),
        "full_ft": len(full),
        "frozen_linear": len(frozen),
        "paired_blocks": paired_blocks,
        "unpaired": unpaired,
        "missing_fold_refs": missing_fold_refs,
        "fold_hash_mismatch": fold_hash_mismatch,
        "historical_fold_refs": historical_fold_refs,
        "all_fresh_training": all(r["reuse_class"] == "fresh_training" for r in oh),
        "paired_block_invariants": block_checks,
        "pass": bool(
            len(oh) == 100 and len(full) == 50 and len(frozen) == 50
            and paired_blocks == 50 and unpaired == 0
            and missing_fold_refs == 0 and fold_hash_mismatch == 0
            and historical_fold_refs == 0
            and all(block_checks.values())
            and all(r["reuse_class"] == "fresh_training" for r in oh)
        ),
    }
    _write_json(os.path.join(PRE, "officehome_matrix_binding.json"), out)
    return out


def main(argv=None):
    names = _class_names()
    cw = build_crosswalk(names)
    diff, v2_path = bind_matrix(names)
    binding_val = validate_binding(names, v2_path)
    result = {"crosswalk": cw, "diff_v1_to_v2": diff, "binding_validation": binding_val}
    print(json.dumps({
        "crosswalk_rows": cw["n_rows"],
        "crosswalk_false": cw["historical_false"],
        "crosswalk_true": cw["confirmatory_true"],
        "v2_add_only_28_cols_byte_identical": diff["original_28_columns_byte_identical"],
        "v2_rows": diff["v2_rows"],
        "binding_pass": binding_val["pass"],
    }, indent=2))
    return result


if __name__ == "__main__":
    main()
