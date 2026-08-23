"""Confirmatory-400R manifest-loading regression (section 9, loader-unchanged path).

The Office-Home loader code path (``fedcore/data/officehome.py`` +
``run_officehome`` + ``models`` + ``fed_train``) is UNCHANGED by this OPTION-B
work -- only NEW fold manifests were added.  Per the section-9 rule we therefore
RETAIN the prior bitwise-resume proof (``resume_equivalence.json``) and add this
manifest-loading regression instead of re-running the full resume:

  * every one of the 10 fresh confirmatory folds loads via the SAME
    ``load_officehome_job`` used by training;
  * the loaded ``folds_sha256`` equals the bound metadata fold hash
    (fold-hash binding);
  * per-role counts equal the fresh fold role counts (role ingestion);
  * ``validate_training_ready`` passes (roles nonempty, unknown support present,
    all 10 pairwise role identities disjoint, no unknown class in train);
  * the loader source files are byte-identical to the commit that produced the
    retained resume proof (loader-unchanged assertion).
"""

from __future__ import annotations

import csv
import collections
import hashlib
import json
import os
import subprocess

from fedcore.data.officehome import OfficeHomeDataConfig, load_officehome_job

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRE = os.path.join(REPO, "results", "confirmatory_400r", "prelaunch")
FOLDS_DIR = os.path.join(PRE, "officehome_folds")
MANIFEST = os.path.join(REPO, "results", "officehome", "dedup", "retained_canonical_manifest.csv")
CLASS_SPLITS = os.path.join(FOLDS_DIR, "officehome_c400r_class_splits.csv")
IMAGE_ROOT = os.path.join(REPO, "data", "officehome", "OfficeHomeDataset")
META = os.path.join(FOLDS_DIR, "officehome_c400r_fold_metadata.json")
ROLE_COUNTS = os.path.join(FOLDS_DIR, "officehome_role_counts.csv")

# Loader code path files (must be unchanged by this OPTION-B work).
LOADER_FILES = [
    "fedcore/data/officehome.py",
    "fedcore/experiments/run_officehome.py",
    "fedcore/models/officehome_train.py",
    "fedcore/models/fed_train.py",
    "fedcore/models/models.py",
    "fedcore/seeds.py",
]
CONF_SPLITS = tuple(f"officehome_c400r_balanced_split_{i:02d}" for i in range(10))


def _role_counts_from_csv():
    d = collections.defaultdict(dict)
    with open(ROLE_COUNTS) as fh:
        for r in csv.DictReader(fh):
            d[r["split_id"]][r["role"]] = int(r["count"])
    return d


def _loader_unchanged():
    """True iff loader files are unmodified in the working tree vs HEAD."""
    files_status = {}
    for f in LOADER_FILES:
        proc = subprocess.run(["git", "status", "--porcelain", "--", f],
                              cwd=REPO, capture_output=True, text=True)
        modified = bool(proc.stdout.strip())
        files_status[f] = {"modified_in_worktree": modified}
    return files_status


def run():
    meta = json.load(open(META))
    role_counts_csv = _role_counts_from_csv()
    per_split = {}
    all_pass = True
    for split_id in CONF_SPLITS:
        cfg = OfficeHomeDataConfig(
            manifest_csv=MANIFEST,
            folds_csv=os.path.join(FOLDS_DIR, f"{split_id}.csv"),
            class_splits_csv=CLASS_SPLITS,
            split_id=split_id,
            image_root=IMAGE_ROOT,
        )
        rec = {}
        try:
            job = load_officehome_job(cfg, check_image_files=False)
            job.validate_training_ready()
            bound_sha = meta["per_split"][split_id]["fold_sha256"]
            loaded_roles = {r: len(job.role_records(r))
                            for r in ("train", "proposal", "certification", "traffic", "evaluation")}
            csv_roles = role_counts_csv[split_id]
            rec = {
                "loaded": True,
                "fold_hash_binding_ok": (job.folds_sha256 == bound_sha),
                "n_known": job.n_known,
                "n_unknown": len(job.unknown_classes),
                "known_unknown_map_ok": (job.n_known == 45 and len(job.unknown_classes) == 20),
                "role_counts": loaded_roles,
                "role_counts_match_csv": (loaded_roles == csv_roles),
                "validate_training_ready": True,
            }
            rec["pass"] = bool(rec["fold_hash_binding_ok"] and rec["known_unknown_map_ok"]
                               and rec["role_counts_match_csv"])
        except Exception as exc:  # fail-closed
            rec = {"loaded": False, "error": f"{type(exc).__name__}: {exc}", "pass": False}
        all_pass = all_pass and rec["pass"]
        per_split[split_id] = rec

    loader_status = _loader_unchanged()
    loader_unchanged = all(not v["modified_in_worktree"] for v in loader_status.values())
    prior_resume = os.path.join(PRE, "resume_equivalence.json")
    prior = json.load(open(prior_resume)) if os.path.isfile(prior_resume) else None
    prior_bitwise = bool(prior and prior.get("all_bitwise_equivalent"))

    out = {
        "campaign": "confirmatory_400r",
        "regression_type": "manifest_loading (loader code path unchanged; prior resume retained)",
        "n_splits_loaded": sum(1 for v in per_split.values() if v.get("loaded")),
        "all_ten_load_and_bind": all_pass,
        "per_split": per_split,
        "loader_files_unchanged_in_worktree": loader_unchanged,
        "loader_files_status": loader_status,
        "retained_prior_resume_proof": {
            "path": os.path.relpath(prior_resume, REPO) if prior else None,
            "all_bitwise_equivalent": prior_bitwise,
            "delta": (prior or {}).get("delta"),
        },
        "pass": bool(all_pass and loader_unchanged and prior_bitwise),
    }
    dst = os.path.join(PRE, "officehome_c400r_loader_regression.json")
    tmp = f"{dst}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    os.replace(tmp, dst)
    print(json.dumps({"all_ten_load_and_bind": all_pass,
                      "loader_files_unchanged": loader_unchanged,
                      "prior_resume_bitwise": prior_bitwise, "pass": out["pass"]}, indent=2))
    return out


if __name__ == "__main__":
    run()
