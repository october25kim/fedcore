"""Gate-0 artifact inventory with semantic reuse classification."""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import numpy as np

from fedcore.accounting.provenance import load_manifest_index, resolve_run
from fedcore.campaign.artifacts import canonical_json, file_sha256, semantic_hash


FIELDS = (
    "path",
    "basename",
    "size_bytes",
    "sha256",
    "duplicate_of",
    "readable",
    "artifact_kind",
    "keys_json",
    "has_logits_schema",
    "has_native_sample_ids",
    "has_split_fingerprints",
    "has_training_config",
    "dataset",
    "backbone",
    "n_known",
    "n_clients",
    "dirichlet_alpha",
    "noise_type",
    "noise_rate",
    "legacy_seed",
    "unknown_classes_json",
    "semantic_config_hash",
    "provenance_source",
    "manifest_file",
    "reuse_class",
    "reason",
)


def _scalar(z, key: str, default: Any = None) -> Any:
    if key not in z.files:
        return default
    value = np.asarray(z[key])
    return value.item() if value.ndim == 0 else value.tolist()


def _inspect_npz(path: str, manifest_index) -> dict[str, Any]:
    base = {
        "readable": False,
        "artifact_kind": "npz",
        "keys_json": "[]",
        "has_logits_schema": False,
        "has_native_sample_ids": False,
        "has_split_fingerprints": False,
        "has_training_config": False,
        "dataset": "",
        "backbone": "",
        "n_known": "",
        "n_clients": "",
        "dirichlet_alpha": "",
        "noise_type": "",
        "noise_rate": "",
        "legacy_seed": "",
        "unknown_classes_json": "[]",
        "semantic_config_hash": "",
        "provenance_source": "",
        "manifest_file": "",
        "reuse_class": "invalid",
        "reason": "unreadable npz",
    }
    try:
        with np.load(path, allow_pickle=False) as z:
            keys = sorted(z.files)
            required = {
                f"{fold}_{name}"
                for fold in ("prop", "cert", "test")
                for name in ("logits", "y_open", "client")
            }
            native_ids = all(
                f"{fold}_sample_id" in keys or f"{fold}_sample_idx" in keys
                for fold in ("prop", "cert", "test")
            )
            base.update(
                {
                    "readable": True,
                    "keys_json": canonical_json(keys),
                    "has_logits_schema": required.issubset(keys),
                    "has_native_sample_ids": native_ids,
                    "has_split_fingerprints": all(
                        f"{f}_fp" in keys for f in ("prop", "cert", "test")
                    ),
                    "has_training_config": "training_config_json" in keys,
                }
            )
            if "training_config_json" in keys:
                config = json.loads(str(_scalar(z, "training_config_json")))
                base.update(
                    {
                        "dataset": config.get("dataset", ""),
                        "backbone": config.get("backbone", ""),
                        "n_known": config.get("n_known", ""),
                        "n_clients": config.get("n_clients", ""),
                        "dirichlet_alpha": config.get("dirichlet_alpha", ""),
                        "noise_type": config.get("noise_type", ""),
                        "noise_rate": config.get("noise_rate", ""),
                        "unknown_classes_json": canonical_json(
                            config.get("unknown_classes") or []
                        ),
                        "semantic_config_hash": semantic_hash(config),
                        "provenance_source": "native",
                    }
                )
    except Exception as exc:
        base["reason"] = f"{type(exc).__name__}: {exc}"
        return base

    resolved_legacy = False
    if base["has_logits_schema"]:
        try:
            spec = resolve_run(path, manifest_index)
        except Exception as exc:
            spec = None
            if not base["reason"]:
                base["reason"] = f"legacy provenance error: {exc}"
        if spec is not None and not base["has_training_config"]:
            resolved_legacy = True
            config = {
                "dataset": spec.dataset,
                "backbone": spec.backbone,
                "n_known": spec.n_known,
                "n_clients": spec.n_clients,
                "dirichlet_alpha": spec.dirichlet_alpha,
                "noise_type": spec.noise_type,
                "noise_rate": spec.noise_rate,
                "legacy_seed": spec.seed,
                "unknown_classes": list(spec.unknown_classes or ()),
            }
            base.update(
                {
                    "dataset": spec.dataset,
                    "backbone": spec.backbone,
                    "n_known": spec.n_known,
                    "n_clients": spec.n_clients,
                    "dirichlet_alpha": spec.dirichlet_alpha,
                    "noise_type": spec.noise_type,
                    "noise_rate": spec.noise_rate,
                    "legacy_seed": spec.seed,
                    "unknown_classes_json": canonical_json(
                        list(spec.unknown_classes or ())
                    ),
                    "semantic_config_hash": semantic_hash(config),
                    "provenance_source": spec.provenance_source,
                    "manifest_file": spec.manifest_file,
                }
            )
        if base["has_native_sample_ids"] and base["has_training_config"]:
            base["reuse_class"] = "native_candidate"
            base["reason"] = (
                "native IDs/config present; terminal manifest still required"
            )
        elif not base["has_training_config"] and not resolved_legacy:
            base["reuse_class"] = "out_of_scope_or_unresolved"
            base["reason"] = (
                "no verified semantic configuration or supported index space"
            )
        else:
            base["reuse_class"] = "legacy_reconstruction_candidate"
            base["reason"] = (
                "requires verified ID reconstruction and semantic seed audit"
            )
    elif base["readable"]:
        base["reuse_class"] = "out_of_scope_or_nonstandard"
        base["reason"] = "does not implement the canonical three-fold logits schema"
    return base


def build_inventory(root: str = ".") -> list[dict[str, Any]]:
    manifest_index = load_manifest_index(root)
    paths = sorted(
        set(
            glob.glob(os.path.join(root, "runs", "*.npz"))
            + glob.glob(os.path.join(root, "runs", ".*.npz"))
        )
    )
    rows: list[dict[str, Any]] = []
    first_hash: dict[str, str] = {}
    for path in paths:
        rel = os.path.relpath(path, root)
        sha = file_sha256(path)
        row = {
            "path": rel,
            "basename": os.path.basename(path),
            "size_bytes": os.path.getsize(path),
            "sha256": sha,
            "duplicate_of": first_hash.get(sha, ""),
        }
        first_hash.setdefault(sha, rel)
        if os.path.basename(path).startswith("._"):
            row.update({field: "" for field in FIELDS if field not in row})
            row.update(
                {
                    "readable": False,
                    "artifact_kind": "appledouble",
                    "reuse_class": "invalid",
                    "reason": "AppleDouble sidecar, not an NPZ artifact",
                }
            )
        else:
            row.update(_inspect_npz(path, manifest_index))
        rows.append({field: row.get(field, "") for field in FIELDS})
    return rows


def write_inventory(rows: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to write the mandated Parquet inventory"
        ) from exc
    bool_fields = {
        "readable",
        "has_logits_schema",
        "has_native_sample_ids",
        "has_split_fingerprints",
        "has_training_config",
    }
    int_fields = {"size_bytes", "n_known", "n_clients", "legacy_seed"}
    float_fields = {"dirichlet_alpha", "noise_rate"}
    normalized = []
    for row in rows:
        clean = {}
        for field in FIELDS:
            value = row.get(field)
            if value == "" or value is None:
                clean[field] = None
            elif field in bool_fields:
                clean[field] = bool(value)
            elif field in int_fields:
                clean[field] = int(value)
            elif field in float_fields:
                clean[field] = float(value)
            else:
                clean[field] = str(value)
        normalized.append(clean)
    schema = pa.schema(
        [
            pa.field(
                field,
                (
                    pa.bool_()
                    if field in bool_fields
                    else (
                        pa.int64()
                        if field in int_fields
                        else pa.float64() if field in float_fields else pa.string()
                    )
                ),
            )
            for field in FIELDS
        ]
    )
    table = pa.Table.from_pylist(normalized, schema=schema)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="docs/agent/experiment_inventory.parquet")
    args = ap.parse_args()
    rows = build_inventory(args.root)
    write_inventory(rows, os.path.join(args.root, args.out))
    from collections import Counter

    print(f"wrote {args.out}: {len(rows)} artifacts")
    for name, count in sorted(Counter(r["reuse_class"] for r in rows).items()):
        print(f"  {name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
