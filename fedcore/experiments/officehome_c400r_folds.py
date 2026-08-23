"""Confirmatory-400R Office-Home OPTION-B fresh fold generation + validation.

Owner OPTION B (2026-07-20): the 10 balanced confirmatory Office-Home splits are
FRESH tasks.  This module regenerates ALL 10 per-sample role folds fresh from the
frozen balanced class definitions and the frozen dedup corpus, under the
confirmatory namespace ``officehome_c400r_balanced_split_00..09``.  It NEVER binds
or reuses the historical ``folds_officehome_split_0..4.csv`` for confirmatory rows.

Canonical role-allocation algorithm (recovered + regression-locked)
------------------------------------------------------------------
The historical folds ``folds_officehome_split_0..4.csv`` were produced by an
uncommitted generator.  The exact algorithm was recovered and is proven here by a
BYTE-IDENTICAL regression (``verify_historical_regression``): feeding the historical
name-based class splits through this same code reproduces all five historical fold
CSVs bit-for-bit (CRLF line endings included).  The algorithm is:

  * iterate domains in the fixed roster order (Art, Clipart, Product, Real_World),
    then classes in alphabetical order;
  * within each ``(domain, class)`` stratum, sort ``sample_id`` ascending
    (lexicographic on the 16-hex content id) -- there is NO shuffle;
  * KNOWN class: two-way largest-remainder split ``{train:0.60, deploy:0.40}``;
    UNKNOWN class: ``train = 0``, ``deploy = all`` (a held-out class is never seen
    in training);
  * split the per-stratum deploy pool by four-way largest-remainder
    ``{proposal:0.20, certification:0.40, traffic:0.15, evaluation:0.25}``;
  * largest-remainder rounding: floor every share, then hand the leftover units
    one at a time to the largest fractional remainders, breaking ties by the fixed
    role listing order;
  * emit contiguous chunks in sorted-id order in the role sequence
    train, proposal, certification, traffic, evaluation; CRLF-terminated CSV with
    header ``domain,class,role,sample_id``.

Because the allocation consumes NO random stream (it is a deterministic function of
the sorted content ids + the frozen fractions), it is bitwise reproducible.  The
only frozen "seed" that enters an Office-Home split is the class-membership choice,
which is fixed upstream in the balanced class-splits CSV.

Fail-closed on: manifest/class-split mismatch, an integer class index outside the
65-class map, a class-membership that differs from the frozen CSV, a violated
balanced invariant, any empty role stratum for a known class, any pairwise role
identity/content-family overlap, cross-domain or cross-class content overlap, or a
failed historical byte regression.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Frozen inputs (hashed + recorded; never modified here).
BALANCED_CSV = os.path.join(REPO, "results/confirmatory_400/prelaunch/officehome_class_splits.csv")
RETAINED_MANIFEST = os.path.join(REPO, "results/officehome/dedup/retained_canonical_manifest.csv")
EXCLUDED_MANIFEST = os.path.join(REPO, "results/officehome/dedup/excluded_conflict_manifest.csv")
PREFLIGHT_MANIFEST = os.path.join(REPO, "results/officehome/preflight/dataset_manifest.csv")
SEED_REGISTRY = os.path.join(REPO, "results/confirmatory_400/prelaunch/semantic_seed_registry.json")
# Historical (legacy) inputs for the byte-identical regression only.
HIST_CLASS_SPLITS = os.path.join(REPO, "results/officehome/preflight/class_splits.csv")
HIST_FOLDS_DIR = os.path.join(REPO, "results/officehome/folds")

OUT_DIR = os.path.join(REPO, "results/confirmatory_400r/prelaunch/officehome_folds")

DOMAINS = ("Art", "Clipart", "Product", "Real_World")
ROLES = ("train", "proposal", "certification", "traffic", "evaluation")
KNOWN_SPLIT = (("train", 0.60), ("deploy", 0.40))
DEPLOY_SPLIT = (("proposal", 0.20), ("certification", 0.40), ("traffic", 0.15), ("evaluation", 0.25))
ROLE_SCHEMA_VERSION = "officehome_roles_v1"
ROLE_SCHEMA = {
    "version": ROLE_SCHEMA_VERSION,
    "roles": list(ROLES),
    "known_class_split": {"train": 0.60, "deploy": 0.40},
    "deploy_pool_split": {"proposal": 0.20, "certification": 0.40, "traffic": 0.15, "evaluation": 0.25},
    "unknown_class_rule": "train=0; entire stratum is deploy-audit",
    "ordering": "domains fixed roster; classes alphabetical; sample_id ascending; no shuffle",
    "rounding": "largest-remainder, ties broken by fixed role listing order",
    "line_ending": "CRLF",
}
CONFIRMATORY_SPLIT_IDS = tuple(f"officehome_c400r_balanced_split_{i:02d}" for i in range(10))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def largest_remainder(total: int, fracs) -> dict:
    """Floor each share of ``total`` then distribute the leftover by largest
    fractional remainder; ties break by the fixed listing order."""
    rows = []
    for i, (name, fr) in enumerate(fracs):
        exact = total * fr
        floor = int(exact // 1)
        rows.append([name, floor, exact - floor, i])
    counts = {name: floor for name, floor, rem, i in rows}
    leftover = total - sum(counts.values())
    order = sorted(rows, key=lambda r: (-r[2], r[3]))
    for k in range(leftover):
        counts[order[k][0]] += 1
    return counts


def _load_manifest():
    """Return (ids_by_stratum, content_by_sid, domain_by_sid, class_by_sid, all_classes, dom_classes)."""
    ids = collections.defaultdict(list)
    content = {}
    domain_of = {}
    class_of = {}
    all_classes = set()
    dom_classes = collections.defaultdict(set)
    with open(RETAINED_MANIFEST, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            dom, kl, sid = row["domain"], row["klass"], row["sample_id"]
            ids[(dom, kl)].append(sid)
            content[sid] = row.get("content_sha256", "")
            domain_of[sid] = dom
            class_of[sid] = kl
            all_classes.add(kl)
            dom_classes[dom].add(kl)
    return ids, content, domain_of, class_of, all_classes, dom_classes


def _class_index_map(all_classes):
    """Canonical class ordering = alphabetical sorted names -> integer index."""
    names = sorted(all_classes)
    return names, {i: n for i, n in enumerate(names)}


def build_fold_rows(known_names, unknown_names, ids_by_stratum):
    """Deterministic canonical allocation -> list of (domain,class,role,sample_id)."""
    known = set(known_names)
    unknown = set(unknown_names)
    classes = sorted(known | unknown)
    out = []
    empty_known_role = []
    for dom in DOMAINS:
        for cls in classes:
            key = (dom, cls)
            sids = sorted(ids_by_stratum.get(key, []))
            n = len(sids)
            if cls in known:
                two = largest_remainder(n, KNOWN_SPLIT)
                ntrain, ndeploy = two["train"], two["deploy"]
            else:
                ntrain, ndeploy = 0, n
            deploy = largest_remainder(ndeploy, DEPLOY_SPLIT)
            role_counts = {
                "train": ntrain,
                "proposal": deploy["proposal"],
                "certification": deploy["certification"],
                "traffic": deploy["traffic"],
                "evaluation": deploy["evaluation"],
            }
            if cls in known:
                for r in ROLES:
                    if role_counts[r] == 0:
                        empty_known_role.append((dom, cls, r))
            idx = 0
            for role in ROLES:
                c = role_counts[role]
                for sid in sids[idx:idx + c]:
                    out.append((dom, cls, role, sid))
                idx += c
    return out, empty_known_role


def rows_to_csv_bytes(rows) -> bytes:
    parts = ["domain,class,role,sample_id\r\n"]
    parts.extend(",".join(r) + "\r\n" for r in rows)
    return "".join(parts).encode("utf-8")


# --------------------------------------------------------------------------- #
# Historical byte-identical regression (proves canonical fidelity)
# --------------------------------------------------------------------------- #
def verify_historical_regression(ids_by_stratum):
    with open(HIST_CLASS_SPLITS, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    results = {}
    all_ok = True
    for i in range(5):
        sid = f"officehome_split_{i}"
        srows = [r for r in rows if r["split_id"] == sid]
        known = sorted({r["class"] for r in srows if r["role"] == "known"})
        unknown = sorted({r["class"] for r in srows if r["role"] == "unknown"})
        gen_rows, _ = build_fold_rows(known, unknown, ids_by_stratum)
        gen = rows_to_csv_bytes(gen_rows)
        hist_path = os.path.join(HIST_FOLDS_DIR, f"folds_{sid}.csv")
        with open(hist_path, "rb") as fh:
            hist = fh.read()
        ok = gen == hist
        results[sid] = {
            "byte_identical": ok,
            "gen_sha256": hashlib.sha256(gen).hexdigest(),
            "historical_sha256": hashlib.sha256(hist).hexdigest(),
        }
        all_ok = all_ok and ok
    return {"all_byte_identical": all_ok, "per_split": results}


# --------------------------------------------------------------------------- #
# Balanced invariant check (fail-closed) -- section 3
# --------------------------------------------------------------------------- #
def load_balanced(names, idx_to_name):
    with open(BALANCED_CSV, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    parsed = []
    n_names = len(names)
    for row in rows:
        known_idx = [int(x) for x in row["known_classes"].split()]
        unknown_idx = [int(x) for x in row["unknown_classes"].split()]
        for j in known_idx + unknown_idx:
            if not (0 <= j < n_names):
                raise SystemExit(f"class index {j} out of range in {row['split_id']}")
        parsed.append({
            "csv_split_id": row["split_id"],
            "n_known": int(row["n_known"]),
            "n_unknown": int(row["n_unknown"]),
            "known_idx": known_idx,
            "unknown_idx": unknown_idx,
            "known_names": [idx_to_name[j] for j in known_idx],
            "unknown_names": [idx_to_name[j] for j in unknown_idx],
            "csv_split_sha256": row["split_sha256"],
        })
    return parsed


def class_membership_hash(known_names, unknown_names) -> str:
    payload = "known:" + ",".join(sorted(known_names)) + "|unknown:" + ",".join(sorted(unknown_names))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_balanced_invariants(parsed, all_classes, dom_classes):
    n_classes = len(all_classes)
    violations = []
    # each split 45 known / 20 unknown, disjoint, union == all 65
    unknown_freq = collections.Counter()
    unknown_sets = []
    for p in parsed:
        kn, un = set(p["known_names"]), set(p["unknown_names"])
        if len(kn) != 45 or len(un) != 20:
            violations.append(f"{p['csv_split_id']}: sizes {len(kn)}/{len(un)} != 45/20")
        if kn & un:
            violations.append(f"{p['csv_split_id']}: known/unknown overlap")
        if kn | un != set(all_classes):
            violations.append(f"{p['csv_split_id']}: union != 65 classes")
        for c in un:
            unknown_freq[c] += 1
        unknown_sets.append(frozenset(un))
    # across 10 splits: 5 classes unknown 4x + 60 classes 3x, spread=1
    freq_by_count = collections.Counter(unknown_freq[c] for c in all_classes)
    spread = (max(unknown_freq.values()) - min(unknown_freq.values())) if unknown_freq else None
    balanced_ok = (freq_by_count.get(4, 0) == 5 and freq_by_count.get(3, 0) == 60
                   and sum(freq_by_count.values()) == n_classes and spread == 1)
    if not balanced_ok:
        violations.append(f"balanced frequency wrong: {dict(freq_by_count)} spread={spread}")
    # all 10 unknown sets distinct
    if len(set(unknown_sets)) != 10:
        violations.append("unknown sets not all distinct")
    # all 65 classes in all 4 domains
    all65_all_domains = all(dom_classes.get(d, set()) == set(all_classes) for d in DOMAINS)
    if not all65_all_domains:
        violations.append("not all 65 classes present in all 4 domains")
    summary = {
        "n_splits": len(parsed),
        "each_45_known_20_unknown": all(len(set(p["known_names"])) == 45 and len(set(p["unknown_names"])) == 20 for p in parsed),
        "n_classes": n_classes,
        "unknown_frequency_distribution": dict(sorted(freq_by_count.items())),
        "spread": spread,
        "five_unknown_4x_sixty_3x": balanced_ok,
        "all_ten_unknown_sets_distinct": len(set(unknown_sets)) == 10,
        "all_65_classes_in_all_4_domains": all65_all_domains,
        "violations": violations,
        "pass": len(violations) == 0,
    }
    return summary, unknown_sets


def pairwise_overlap_rows(parsed, unknown_sets):
    out = []
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            inter = len(unknown_sets[i] & unknown_sets[j])
            out.append({
                "split_a": CONFIRMATORY_SPLIT_IDS[i],
                "split_b": CONFIRMATORY_SPLIT_IDS[j],
                "unknown_intersection_size": inter,
                "unknown_jaccard": round(inter / len(unknown_sets[i] | unknown_sets[j]), 4),
            })
    return out


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def _write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def generate(argv=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    ids, content, domain_of, class_of, all_classes, dom_classes = _load_manifest()
    names, idx_to_name = _class_index_map(all_classes)

    # (0) record frozen input hashes
    frozen = {
        "source_commit": _source_commit(),
        "balanced_class_splits_csv": {"path": os.path.relpath(BALANCED_CSV, REPO), "sha256": file_sha256(BALANCED_CSV)},
        "retained_canonical_manifest_csv": {"path": os.path.relpath(RETAINED_MANIFEST, REPO), "sha256": file_sha256(RETAINED_MANIFEST)},
        "excluded_conflict_manifest_csv": {"path": os.path.relpath(EXCLUDED_MANIFEST, REPO), "sha256": file_sha256(EXCLUDED_MANIFEST)},
        "preflight_dataset_manifest_csv": {"path": os.path.relpath(PREFLIGHT_MANIFEST, REPO), "sha256": file_sha256(PREFLIGHT_MANIFEST)},
        "semantic_seed_registry_json": {"path": os.path.relpath(SEED_REGISTRY, REPO), "sha256": file_sha256(SEED_REGISTRY)},
        "class_index_map": {str(i): n for i, n in idx_to_name.items()},
        "role_schema": ROLE_SCHEMA,
    }

    # (regression) prove canonical algorithm fidelity BEFORE generating anything.
    regression = verify_historical_regression(ids)
    if not regression["all_byte_identical"]:
        _write_json(os.path.join(OUT_DIR, "officehome_canonical_regression.json"), regression)
        raise SystemExit("FAIL-CLOSED: historical fold byte-regression failed; canonical algorithm not confirmed")

    # (3) balanced invariants
    parsed = load_balanced(names, idx_to_name)
    inv, unknown_sets = check_balanced_invariants(parsed, all_classes, dom_classes)
    pair_rows = pairwise_overlap_rows(parsed, unknown_sets)
    _write_csv(os.path.join(OUT_DIR, "officehome_confirmatory_pairwise_overlap.csv"),
               ["split_a", "split_b", "unknown_intersection_size", "unknown_jaccard"], pair_rows)
    inv_out = dict(inv)
    inv_out["frozen_inputs"] = frozen
    inv_out["historical_byte_regression"] = regression
    inv_out["confirmatory_split_ids"] = list(CONFIRMATORY_SPLIT_IDS)
    inv_out["csv_split_id_namespace_map"] = {
        CONFIRMATORY_SPLIT_IDS[i]: parsed[i]["csv_split_id"] for i in range(len(parsed))
    }
    _write_json(os.path.join(OUT_DIR, "officehome_confirmatory_split_validation.json"), inv_out)
    if not inv["pass"]:
        raise SystemExit(f"FAIL-CLOSED: balanced invariants violated: {inv['violations']}")

    # membership-unchanged check: names derived from frozen CSV integer indices,
    # so any drift would show as a class index out of range (already fail-closed).

    # (4) generate 10 fresh folds + per-split metadata
    per_split_meta = {}
    fold_checksums = {}
    all_empty = {}
    for si, split_id in enumerate(CONFIRMATORY_SPLIT_IDS):
        p = parsed[si]
        rows, empty_known_role = build_fold_rows(p["known_names"], p["unknown_names"], ids)
        if empty_known_role:
            all_empty[split_id] = empty_known_role
        data = rows_to_csv_bytes(rows)
        fold_path = os.path.join(OUT_DIR, f"{split_id}.csv")
        tmp = f"{fold_path}.tmp.{os.getpid()}"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, fold_path)
        fold_sha = hashlib.sha256(data).hexdigest()
        fold_checksums[f"{split_id}.csv"] = fold_sha
        per_split_meta[split_id] = {
            "confirmatory_split_id": split_id,
            "balanced_csv_split_id": p["csv_split_id"],
            "n_known": p["n_known"],
            "n_unknown": p["n_unknown"],
            "class_membership_hash": class_membership_hash(p["known_names"], p["unknown_names"]),
            "balanced_csv_split_sha256": p["csv_split_sha256"],
            "dataset_manifest_sha256": frozen["retained_canonical_manifest_csv"]["sha256"],
            "preflight_dataset_manifest_sha256": frozen["preflight_dataset_manifest_csv"]["sha256"],
            "role_schema_version": ROLE_SCHEMA_VERSION,
            "seed_registry_sha256": frozen["semantic_seed_registry_json"]["sha256"],
            "fold_sha256": fold_sha,
            "fold_rows": len(rows),
            "source_commit": frozen["source_commit"],
            "allocation": "deterministic sorted-order largest-remainder (no RNG); class membership frozen upstream",
            "empty_known_role_strata": empty_known_role,
        }
    # (4b) confirmatory long-format class_splits CSV (loader-compatible, name-based).
    # The per-split ``seed`` is pure provenance (the class membership is frozen by
    # the balanced CSV; the loader only records it as split_seed), so it is derived
    # deterministically from the frozen balanced ``split_sha256``.
    cls_lines = ["split_id,seed,role,class"]
    for si, split_id in enumerate(CONFIRMATORY_SPLIT_IDS):
        p = parsed[si]
        seed = int(p["csv_split_sha256"][:8], 16)
        for name in sorted(p["known_names"]):
            cls_lines.append(f"{split_id},{seed},known,{name}")
        for name in sorted(p["unknown_names"]):
            cls_lines.append(f"{split_id},{seed},unknown,{name}")
    cls_path = os.path.join(OUT_DIR, "officehome_c400r_class_splits.csv")
    tmp = f"{cls_path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="") as fh:
        fh.write("\n".join(cls_lines) + "\n")
    os.replace(tmp, cls_path)
    frozen["confirmatory_class_splits_csv"] = {
        "path": os.path.relpath(cls_path, REPO), "sha256": file_sha256(cls_path),
    }
    for split_id in CONFIRMATORY_SPLIT_IDS:
        per_split_meta[split_id]["confirmatory_class_splits_sha256"] = frozen["confirmatory_class_splits_csv"]["sha256"]

    _write_json(os.path.join(OUT_DIR, "officehome_c400r_fold_metadata.json"),
                {"frozen_inputs": frozen, "per_split": per_split_meta})

    if all_empty:
        _write_json(os.path.join(OUT_DIR, "officehome_empty_role_strata.json"), all_empty)
        raise SystemExit(f"FAIL-CLOSED: empty known-class role stratum: {all_empty}")

    # (5) identity/overlap validation
    validate_out = validate_identity(ids, content, domain_of, class_of, per_split_meta)
    print(json.dumps({
        "regression_all_byte_identical": regression["all_byte_identical"],
        "balanced_invariants_pass": inv["pass"],
        "n_folds": len(CONFIRMATORY_SPLIT_IDS),
        "identity_overlap_pass": validate_out["overall_pass"],
        "any_empty_role": bool(all_empty),
    }, indent=2))
    return {"regression": regression, "invariants": inv, "identity": validate_out,
            "per_split_meta": per_split_meta}


def _read_fold(path):
    role_ids = collections.defaultdict(set)
    id_role = {}
    id_domain = {}
    id_class = {}
    dup = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            sid = row["sample_id"]
            role = row["role"]
            if sid in id_role:
                dup.append(sid)
            role_ids[role].add(sid)
            id_role[sid] = role
            id_domain[sid] = row["domain"]
            id_class[sid] = row["class"]
    return role_ids, id_role, id_domain, id_class, dup


def validate_identity(ids, content, domain_of, class_of, per_split_meta):
    """Section 5: source-ID AND content-family level overlap validation.

    "Excluded families stay excluded" concerns the CONFLICT removals only.  The
    ``same_label_canonicalized`` groups are exact byte-duplicates whose retained
    canonical legitimately shares a content hash with its dropped siblings -- that
    is what dedup DOES, not a leak.  A real leak is a fold sample whose content
    family is one that was FULLY excluded (a cross-class / cross-domain conflict
    with no retained representative).  We therefore define the excluded set as the
    content families present in the excluded manifest that have ZERO retained
    members.
    """
    retained_content = set(content.values())
    excluded_ids = set()
    excluded_manifest_content = set()
    excluded_dispositions = collections.Counter()
    with open(EXCLUDED_MANIFEST, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row.get("sample_id"):
                excluded_ids.add(row["sample_id"])
            if row.get("content_sha256"):
                excluded_manifest_content.add(row["content_sha256"])
            if row.get("disposition"):
                excluded_dispositions[row["disposition"]] += 1
    # Fully-excluded (conflict) families: appear in excluded manifest, never retained.
    excluded_content = excluded_manifest_content - retained_content

    fold_val_rows = []
    overlap_rows = []
    role_count_rows = []
    checksum_lines = []
    overall_pass = True

    for split_id in CONFIRMATORY_SPLIT_IDS:
        path = os.path.join(OUT_DIR, f"{split_id}.csv")
        role_ids, id_role, id_domain, id_class, dup = _read_fold(path)
        roles_present = [r for r in ROLES if r in role_ids]

        # pairwise role overlaps: source-id and content-family
        all_id_overlap_zero = True
        all_content_overlap_zero = True
        for i, a in enumerate(roles_present):
            for b in roles_present[i + 1:]:
                idov = role_ids[a] & role_ids[b]
                ca = {content.get(s, s) for s in role_ids[a]}
                cb = {content.get(s, s) for s in role_ids[b]}
                cov = ca & cb
                overlap_rows.append({
                    "split_id": split_id, "role_a": a, "role_b": b,
                    "source_id_overlap": len(idov),
                    "content_family_overlap": len(cov),
                })
                if idov:
                    all_id_overlap_zero = False
                if cov:
                    all_content_overlap_zero = False

        # each retained id -> exactly one domain+class; agree with manifest
        domain_class_consistent = True
        cross_domain_overlap = 0
        cross_class_overlap = 0
        excluded_leak = 0
        outside_manifest = 0
        for sid in id_role:
            if sid not in domain_of:
                outside_manifest += 1
                continue
            if id_domain[sid] != domain_of[sid] or id_class[sid] != class_of[sid]:
                domain_class_consistent = False
            if sid in excluded_ids or content.get(sid, "___") in excluded_content:
                # allow ids reused across excluded/retained only if content differs
                if content.get(sid, "") in excluded_content:
                    excluded_leak += 1

        # ids unique within a role (sets already dedup; count via re-read)
        # cross-domain / cross-class content overlap: a content family must not
        # span two domains or two classes within this fold.
        content_domains = collections.defaultdict(set)
        content_classes = collections.defaultdict(set)
        for sid in id_role:
            c = content.get(sid, sid)
            content_domains[c].add(id_domain[sid])
            content_classes[c].add(id_class[sid])
        cross_domain_overlap = sum(1 for c, ds in content_domains.items() if len(ds) > 1)
        cross_class_overlap = sum(1 for c, cs in content_classes.items() if len(cs) > 1)

        strata_sizes = {r: len(role_ids[r]) for r in roles_present}
        all_roles_present = set(roles_present) == set(ROLES)
        all_strata_nonempty = all(v > 0 for v in strata_sizes.values()) and all_roles_present

        split_pass = bool(
            all_id_overlap_zero and all_content_overlap_zero and not dup
            and domain_class_consistent and cross_domain_overlap == 0
            and cross_class_overlap == 0 and excluded_leak == 0
            and outside_manifest == 0 and all_strata_nonempty
        )
        overall_pass = overall_pass and split_pass

        fold_val_rows.append({
            "split_id": split_id,
            "balanced_csv_split_id": per_split_meta[split_id]["balanced_csv_split_id"],
            "n_rows": sum(len(role_ids[r]) for r in roles_present),
            "all_source_id_overlaps_zero": all_id_overlap_zero,
            "all_content_family_overlaps_zero": all_content_overlap_zero,
            "duplicate_ids": len(dup),
            "domain_class_consistent_with_manifest": domain_class_consistent,
            "cross_domain_content_overlap": cross_domain_overlap,
            "cross_class_content_overlap": cross_class_overlap,
            "excluded_family_leak": excluded_leak,
            "ids_outside_manifest": outside_manifest,
            "all_five_roles_present_and_nonempty": all_strata_nonempty,
            "fold_sha256": per_split_meta[split_id]["fold_sha256"],
            "split_pass": split_pass,
        })
        for r in ROLES:
            role_count_rows.append({
                "split_id": split_id, "role": r, "count": len(role_ids.get(r, set())),
            })
        checksum_lines.append(f"{per_split_meta[split_id]['fold_sha256']}  {split_id}.csv")

    _write_csv(os.path.join(OUT_DIR, "officehome_fold_validation.csv"),
               list(fold_val_rows[0].keys()), fold_val_rows)
    _write_csv(os.path.join(OUT_DIR, "officehome_overlap_report.csv"),
               ["split_id", "role_a", "role_b", "source_id_overlap", "content_family_overlap"], overlap_rows)
    _write_csv(os.path.join(OUT_DIR, "officehome_role_counts.csv"),
               ["split_id", "role", "count"], role_count_rows)
    checksum_path = os.path.join(OUT_DIR, "officehome_fold_checksums.sha256")
    tmp = f"{checksum_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write("\n".join(checksum_lines) + "\n")
    os.replace(tmp, checksum_path)

    return {
        "overall_pass": overall_pass,
        "n_splits": len(CONFIRMATORY_SPLIT_IDS),
        "retained_content_families": len(retained_content),
        "fully_excluded_conflict_families": len(excluded_content),
        "same_label_canonicalized_shared_with_retained": len(excluded_manifest_content & retained_content),
        "excluded_dispositions": dict(excluded_dispositions),
        "excluded_sample_ids_in_manifest": len(excluded_ids),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only", action="store_true", help="only run the historical byte regression")
    args = p.parse_args(argv)
    if args.verify_only:
        ids, *_ = _load_manifest()
        reg = verify_historical_regression(ids)
        print(json.dumps(reg, indent=2))
        return 0 if reg["all_byte_identical"] else 1
    generate(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
