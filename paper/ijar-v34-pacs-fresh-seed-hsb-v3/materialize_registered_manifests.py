#!/usr/bin/env python3
"""Materialize the fully registered PACS model and RNG manifests."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "PROSPECTIVE_CONFIRMATION_PROTOCOL.json").read_text())
PID = PROTOCOL["training"]["seed_namespace"]["seed_namespace_id"]


def seed_for(label: str, parts: list[object]) -> int:
    compact = json.dumps(parts, separators=(",", ":"), ensure_ascii=True)
    payload = f"{PID}|{label}|{compact}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def model_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in PROTOCOL["dataset"]["class_splits"]:
        known = ";".join(split["known"])
        unknown = ";".join(split["unknown"])
        split_id = int(split["split"])
        for nominal_seed in PROTOCOL["training"]["nominal_seeds"]:
            for architecture in PROTOCOL["training"]["architectures"]:
                model_id = f"pacs__{architecture}__split{split_id:02d}__seed{nominal_seed}"
                rows.append(
                    {
                        "model_id": model_id,
                        "architecture": architecture,
                        "split": split_id,
                        "nominal_seed": nominal_seed,
                        "model_init_seed_uint64": seed_for(
                            "model_init", [architecture, split_id, nominal_seed]
                        ),
                        "known_classes": known,
                        "unknown_classes": unknown,
                        "status": "NOT_RUN",
                    }
                )
    return rows


def audit_seed_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    clients = PROTOCOL["dataset"]["clients"]
    for split_id in PROTOCOL["training"]["class_splits"]:
        for client in clients:
            for replicate_id in range(100):
                rows.append(
                    {
                        "split": split_id,
                        "client": client,
                        "replicate_id": replicate_id,
                        "pcg64_seed_uint64": seed_for(
                            "audit_draw", [split_id, client, replicate_id]
                        ),
                        "maximum_prefix_draws": 256,
                        "primary_prefix_draws": 128,
                        "secondary_lower_prefix_draws": 64,
                        "secondary_upper_prefix_draws": 256,
                        "primary": str(replicate_id == 0).lower(),
                    }
                )
    return rows


def main() -> None:
    models = model_rows()
    audit_seeds = audit_seed_rows()
    write_csv(
        ROOT / "PACS_MODEL_MATRIX.csv",
        [
            "model_id",
            "architecture",
            "split",
            "nominal_seed",
            "model_init_seed_uint64",
            "known_classes",
            "unknown_classes",
            "status",
        ],
        models,
    )
    write_csv(
        ROOT / "TRAINING_SEED_MANIFEST.csv",
        ["model_id", "architecture", "split", "nominal_seed", "model_init_seed_uint64"],
        [
            {key: row[key] for key in ["model_id", "architecture", "split", "nominal_seed", "model_init_seed_uint64"]}
            for row in models
        ],
    )
    write_csv(
        ROOT / "AUDIT_SEED_MANIFEST.csv",
        [
            "split",
            "client",
            "replicate_id",
            "pcg64_seed_uint64",
            "maximum_prefix_draws",
            "primary_prefix_draws",
            "secondary_lower_prefix_draws",
            "secondary_upper_prefix_draws",
            "primary",
        ],
        audit_seeds,
    )
    assert len(models) == 30
    assert len({row["model_id"] for row in models}) == 30
    assert all(row["status"] == "NOT_RUN" for row in models)
    assert len(audit_seeds) == 5 * 4 * 100
    print(json.dumps({"models": len(models), "audit_seed_streams": len(audit_seeds), "status": "PASS"}))


if __name__ == "__main__":
    main()
