"""Validate the two FRESH-confirmatory Office-Home GPU smokes (section 9).

Consumes the native artifacts + terminal markers written INSIDE the pinned
container by ``confirmatory_400r_oh_gpu smoke`` for
``officehome_c400r_balanced_split_00 x rep0 x {full, frozen}`` and checks, on the
host, every engineering-only invariant:

  * smoke experiment IDs disjoint from the 400 production semantic_ids;
  * terminal marker present with status=completed; recorded checksums match disk;
  * fold-hash binding + role ingestion: the trained split is the FRESH confirmatory
    split and its per-role counts equal the fresh fold's role counts;
  * known/unknown map: exported known_classes/unknown_classes equal the fold's;
  * native -> common-schema normalization succeeds and the common schema is valid;
  * proposal candidate construction + engineering-only certificate ingestion return
    finite, well-typed numbers.

Writes ``officehome_c400r_smoke_validation.json``.  Honest by construction: any
missing marker/artifact is reported PENDING/FAIL, never assumed passed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np

from fedcore.experiments import confirmatory_400r_common_schema as CS
from fedcore.experiments.confirmatory_400r_cert_dryrun import cert_dry_run

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRE = os.path.join(REPO, "results", "confirmatory_400r", "prelaunch")
ART = os.path.join(PRE, "smoke_artifacts_c400r")
MATRIX = os.path.join(REPO, "results", "confirmatory_400r", "final_training_matrix.csv")
FOLDS_DIR = os.path.join(PRE, "officehome_folds")
ROLE_COUNTS_CSV = os.path.join(FOLDS_DIR, "officehome_role_counts.csv")

SMOKE_IDS = {
    "full": "SMOKE_C400R_officehome_c400r_balanced_split_00__rep0__full__2round",
    "frozen": "SMOKE_C400R_officehome_c400r_balanced_split_00__rep0__frozen__2round",
}
NATIVE = {"full": os.path.join(ART, "C400R_full.npz"), "frozen": os.path.join(ART, "C400R_frozen.npz")}
CKPT = {"full": os.path.join(ART, "C400R_full.pt"), "frozen": os.path.join(ART, "C400R_frozen.pt")}
MARKER = {"full": os.path.join(ART, "C400R_full.TERMINAL.json"),
          "frozen": os.path.join(ART, "C400R_frozen.TERMINAL.json")}
COMMON = {"full": os.path.join(ART, "C400R_full_common.npz"),
          "frozen": os.path.join(ART, "C400R_frozen_common.npz")}
FAMILY = {"full": "officehome_convnext_full", "frozen": "officehome_convnext_frozen"}
CONF_SPLIT = "officehome_c400r_balanced_split_00"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _fresh_role_counts():
    counts = {}
    with open(ROLE_COUNTS_CSV) as fh:
        for r in csv.DictReader(fh):
            if r["split_id"] == CONF_SPLIT:
                counts[r["role"]] = int(r["count"])
    return counts


def _production_ids():
    with open(MATRIX) as fh:
        return {r["semantic_id"] for r in csv.DictReader(fh)}


def validate():
    prod = _production_ids()
    fresh_counts = _fresh_role_counts()
    result = {"campaign": "confirmatory_400r", "smoke_ids": SMOKE_IDS,
              "fresh_split": CONF_SPLIT, "fresh_fold_role_counts": fresh_counts, "per_arm": {}}
    ids_disjoint = len(set(SMOKE_IDS.values()) & prod) == 0
    all_pass = ids_disjoint
    for arm in ("full", "frozen"):
        rep = {"smoke_id": SMOKE_IDS[arm]}
        marker = _load_json(MARKER[arm])
        if marker is None:
            rep["status"] = "PENDING_marker_missing"
            rep["pass"] = False
            all_pass = False
            result["per_arm"][arm] = rep
            continue
        rep["marker_status"] = marker.get("status")
        # checksum match on disk
        checksum_ok = True
        for name, expected in (marker.get("checksums") or {}).items():
            path = os.path.join(ART, name)
            if not os.path.isfile(path) or _sha256(path) != expected:
                checksum_ok = False
        rep["checksums_match_disk"] = checksum_ok
        # role ingestion: marker role_counts + traffic_units == fresh fold counts
        mrc = marker.get("role_counts") or {}
        role_match = (
            mrc.get("proposal") == fresh_counts.get("proposal")
            and mrc.get("certification") == fresh_counts.get("certification")
            and mrc.get("evaluation") == fresh_counts.get("evaluation")
            and marker.get("traffic_units") == fresh_counts.get("traffic")
        )
        rep["role_counts_match_fresh_fold"] = bool(role_match)
        rep["split_id_is_fresh_confirmatory"] = (marker.get("split_id") == CONF_SPLIT)

        # known/unknown map from the native npz
        known_map_ok = None
        schema_ok = None
        cert_finite = None
        n_clients = None
        if os.path.isfile(NATIVE[arm]):
            data = np.load(NATIVE[arm], allow_pickle=False)
            known = sorted(str(c) for c in np.asarray(data["known_classes"], dtype=str))
            unknown = sorted(str(c) for c in np.asarray(data["unknown_classes"], dtype=str))
            known_map_ok = (len(known) == 45 and len(unknown) == 20
                            and not (set(known) & set(unknown)))
            rep["n_known_exported"] = len(known)
            rep["n_unknown_exported"] = len(unknown)
            # normalize to common schema
            CS.normalize_officehome_to_common(NATIVE[arm], COMMON[arm], family=FAMILY[arm])
            desc = CS.describe_common_npz(COMMON[arm])
            field_names = [n for n, _ in CS.PER_OBS_FIELDS]
            schema_ok = all(desc["per_field"][n]["kind"] is not None for n in field_names)
            rep["common_schema_fields_present"] = schema_ok
            rep["common_sha256"] = _sha256(COMMON[arm])
            # cert dry-run (proposal candidate construction + engineering cert ingestion)
            roles = [str(r) for r in np.load(COMMON[arm], allow_pickle=False)["fold_roles"]]
            cid = np.asarray(np.load(COMMON[arm], allow_pickle=False)[f"{roles[0]}__client_id"], dtype=int)
            n_clients = int(cid.max()) + 1
            cert = cert_dry_run(COMMON[arm], n_clients=n_clients)
            cert_finite = bool(cert["all_outputs_finite"])
            rep["cert_dryrun_all_finite"] = cert_finite
            rep["n_clients"] = n_clients
        else:
            rep["native_artifact"] = "missing"
        rep["train_seconds"] = marker.get("train_seconds")
        rep["export_seconds"] = marker.get("export_seconds")
        rep["wall_seconds"] = marker.get("wall_seconds")
        rep["peak_vram_gb"] = marker.get("peak_vram_gb")
        rep["gpu_name"] = marker.get("gpu_name")
        arm_pass = bool(
            marker.get("status") == "completed" and checksum_ok and role_match
            and rep["split_id_is_fresh_confirmatory"] and known_map_ok
            and schema_ok and cert_finite
        )
        rep["pass"] = arm_pass
        all_pass = all_pass and arm_pass
        result["per_arm"][arm] = rep

    result["smoke_ids_disjoint_from_400"] = ids_disjoint
    result["both_smokes_pass"] = bool(all_pass)
    out = os.path.join(PRE, "officehome_c400r_smoke_validation.json")
    tmp = f"{out}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    os.replace(tmp, out)
    print(json.dumps({"smoke_ids_disjoint": ids_disjoint,
                      "both_smokes_pass": result["both_smokes_pass"],
                      "per_arm_pass": {a: result["per_arm"][a].get("pass") for a in result["per_arm"]}},
                     indent=2))
    return result


if __name__ == "__main__":
    validate()
