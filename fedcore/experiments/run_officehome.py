"""Train one resumable Office-Home open-set Fed-CORE model.

One invocation is exactly one immutable training cell:

``class_split x train_rep x pipeline``.

Risk targets, confidence levels, mixture radii, thresholds, scores, and
certificate variants are intentionally ABSENT from this CLI. They are post-hoc
analyses over the immutable per-role logit artifact this runner emits.

Two pipelines share the runner:

* ``A`` / ``convnext_tiny_full_fedavg`` -- torchvision convnext_tiny ImageNet-1k
  pretrained, full FedAvg fine-tune (lr 1e-4).
* ``B`` / ``convnext_tiny_frozen_linear`` -- frozen encoder, federated linear
  known-class head (lr 1e-3).

Recipe (both): 224px, 4 clients/round, 30 rounds, 1 local epoch, batch 32,
AdamW, wd 0.05, cosine schedule with a 2-round warmup, FedAvg weighted by
known-class train count.

The ``--dry-run`` path validates metadata, frozen folds, class split, role
disjointness, label mapping, and semantic provenance without importing torch,
torchvision, or PIL and without opening an image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from fedcore.campaign.artifacts import (
    canonical_json,
    semantic_experiment_id,
    semantic_hash,
)
from fedcore.data.officehome import (
    AUDIT_ROLES,
    DOMAINS,
    OUTPUT_FOLD_NAMES,
    TRAFFIC_ROLE,
    OfficeHomeDataConfig,
    audit_role_arrays,
    domain_client_roster_hash,
    load_officehome_job,
)
from fedcore.seeds import SeedBundle


# Pipeline identity -> (canonical name, frozen encoder, learning rate).
PIPELINES: Mapping[str, dict] = {
    "A": {"name": "convnext_tiny_full_fedavg", "frozen_encoder": False, "lr": 1e-4,
          "backbone": "convnext_tiny"},
    "B": {"name": "convnext_tiny_frozen_linear", "frozen_encoder": True, "lr": 1e-3,
          "backbone": "convnext_tiny"},
    # Pipeline C (backbone-sweep addition): DINOv2 ViT-L/14 FROZEN linear-probe --
    # only the linear head is federated; runs in the fedcore-c400r-dino image.
    "C": {"name": "dinov2_vitl14_frozen_linear", "frozen_encoder": True, "lr": 1e-3,
          "backbone": "dinov2_vitl14"},
}
_PIPELINE_ALIASES = {
    "A": "A",
    "B": "B",
    "C": "C",
    "convnext_tiny_full_fedavg": "A",
    "convnext_tiny_frozen_linear": "B",
    "dinov2_vitl14_frozen_linear": "C",
    "full": "A",
    "frozen_linear": "B",
    "frozen": "B",
    "dinov2": "C",
    "dinov2_frozen": "C",
}

WARMUP_ROUNDS = 2


def resolve_pipeline(spec: str) -> str:
    key = _PIPELINE_ALIASES.get(str(spec))
    if key is None:
        raise ValueError(
            f"unknown pipeline {spec!r}; expected one of {sorted(_PIPELINE_ALIASES)}"
        )
    return key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--folds-csv", required=True)
    parser.add_argument("--class-splits-csv", required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--pipeline", required=True, help="A|B or the canonical name")
    parser.add_argument("--train-rep", "--model-replicate", dest="train_rep", type=int, default=0)
    parser.add_argument("--campaign-seed", type=int, default=0)
    parser.add_argument("--dataset-name", default="officehome")
    parser.add_argument("--dataset-version", default="officehome_dedup_v1")

    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-rounds", type=int, default=WARMUP_ROUNDS)
    parser.add_argument("--lr", type=float, default=None, help="override the pipeline lr")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--out", default=None, help="immutable per-role *_logits.npz")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="metadata/fold validation only; no torch/image import or file access",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.rounds <= 0 or args.local_epochs <= 0:
        raise ValueError("rounds and local_epochs must be positive")
    if args.batch_size <= 0 or args.image_size <= 0:
        raise ValueError("batch_size and image_size must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if args.warmup_rounds < 0:
        raise ValueError("warmup_rounds must be non-negative")
    if args.train_rep < 0 or args.campaign_seed < 0:
        raise ValueError("train_rep and campaign_seed must be non-negative")


def build_seed_bundle(
    job, *, pipeline_name: str, campaign_seed: int, train_rep: int
) -> SeedBundle:
    """Derive all streams from immutable job identity, never post-hoc knobs.

    ``cell_id`` includes split/train_rep/pipeline, so every neural-training cell
    and its audit/traffic draws have their own reproducible streams.
    """

    cell_id = (
        f"{job.config.dataset_name}/{job.split_id}/rep={train_rep}/pipeline={pipeline_name}"
    )
    immutable = {
        "manifest_sha256": job.manifest_sha256,
        "folds_sha256": job.folds_sha256,
        "class_splits_sha256": job.class_splits_sha256,
        "split_id": job.split_id,
        "pipeline": pipeline_name,
        "train_rep": int(train_rep),
    }
    return SeedBundle.derive(
        campaign_seed,
        common_context={
            "campaign": "fedcore-officehome",
            "dataset": job.config.dataset_name,
            "dataset_version": job.config.dataset_version,
        },
        namespace_contexts={
            "class_split": {
                "split_id": job.split_id,
                "split_seed": int(job.split_seed),
                "class_splits_sha256": job.class_splits_sha256,
            },
            "partition": {
                "domain_roster_sha256": domain_client_roster_hash(job.domains),
                "manifest_sha256": job.manifest_sha256,
            },
            "fold": {"folds_sha256": job.folds_sha256},
            "model_init": immutable,
            "loader": immutable,
            "label_noise": {**immutable, "mode": "none"},
            "audit_draw": {
                "experiment_id": cell_id,
                "folds_sha256": job.folds_sha256,
                "draw_index": 0,
            },
            "traffic_draw": {
                "experiment_id": cell_id,
                "folds_sha256": job.folds_sha256,
                "draw_index": 0,
            },
            "solver": {"experiment_id": cell_id},
            "stability": {"experiment_id": cell_id, "redraw_index": 0},
        },
    )


def build_training_config(args: argparse.Namespace, job, pipeline_key: str, seed_bundle: SeedBundle) -> dict:
    pipeline = PIPELINES[pipeline_key]
    lr = float(args.lr) if args.lr is not None else float(pipeline["lr"])
    return {
        "schema_version": 1,
        "family": "officehome",
        "dataset": job.config.dataset_name,
        "dataset_version": job.config.dataset_version,
        "manifest_sha256": job.manifest_sha256,
        "folds_sha256": job.folds_sha256,
        "class_splits_sha256": job.class_splits_sha256,
        "split_id": job.split_id,
        "split_seed": int(job.split_seed),
        "known_classes": list(job.known_classes),
        "unknown_classes": list(job.unknown_classes),
        "domains": list(job.domains),
        "n_clients": job.n_clients,
        "n_known": job.n_known,
        "pipeline": pipeline_key,
        "pipeline_name": pipeline["name"],
        "backbone": pipeline.get("backbone", "convnext_tiny"),
        "frozen_encoder": bool(pipeline["frozen_encoder"]),
        "pretrained": bool(args.pretrained),
        "train_rep": int(args.train_rep),
        "campaign_seed": int(args.campaign_seed),
        "training_seeds": {
            "model_init": seed_bundle.model_init,
            "loader": seed_bundle.loader,
        },
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "batch_size": int(args.batch_size),
        "image_size": int(args.image_size),
        "lr": lr,
        "weight_decay": float(args.weight_decay),
        "warmup_rounds": int(args.warmup_rounds),
        "optimizer": "adamw",
        "lr_schedule": "cosine-warmup-v1",
        "image_transform": "randomresizedcrop-hflip-imagenet-v1",
        "federated_optimizer": "size-weighted-fedavg-known-count",
        "deterministic_replay": True,
        "device_request": args.device,
    }


def prepare_job(args: argparse.Namespace):
    _validate_args(args)
    pipeline_key = resolve_pipeline(args.pipeline)
    data_config = OfficeHomeDataConfig(
        manifest_csv=args.manifest_csv,
        folds_csv=args.folds_csv,
        class_splits_csv=args.class_splits_csv,
        split_id=args.split_id,
        image_root=args.image_root,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
    )
    job = load_officehome_job(data_config, check_image_files=not args.dry_run)
    seed_bundle = build_seed_bundle(
        job,
        pipeline_name=PIPELINES[pipeline_key]["name"],
        campaign_seed=args.campaign_seed,
        train_rep=args.train_rep,
    )
    training_config = build_training_config(args, job, pipeline_key, seed_bundle)
    training_config_sha256 = semantic_hash(training_config)
    prefix = f"officehome-{job.split_id}-rep{args.train_rep}-{pipeline_key}"
    generated_id = semantic_experiment_id(prefix, training_config)
    experiment_id = args.experiment_id or generated_id
    out_path = args.out or os.path.join("runs", f"{experiment_id}_logits.npz")
    checkpoint_path = args.checkpoint or os.path.join(
        "runs", "checkpoints", f"{experiment_id}.pt"
    )
    output_abs = os.path.abspath(out_path)
    checkpoint_abs = os.path.abspath(checkpoint_path)
    inputs = {
        os.path.abspath(args.manifest_csv),
        os.path.abspath(args.folds_csv),
        os.path.abspath(args.class_splits_csv),
    }
    if output_abs == checkpoint_abs:
        raise ValueError("output artifact and checkpoint paths must be distinct")
    if {output_abs, checkpoint_abs} & inputs:
        raise ValueError("output/checkpoint paths must not overwrite frozen inputs")
    return (
        job,
        pipeline_key,
        seed_bundle,
        training_config,
        training_config_sha256,
        experiment_id,
        out_path,
        checkpoint_path,
    )


def dry_run_report(job, pipeline_key, seed_bundle, training_config, training_config_sha256, experiment_id, out_path, checkpoint_path) -> dict:
    report = job.summary()
    report.update(
        {
            "schema_version": 1,
            "status": "metadata_validated",
            "dry_run": True,
            "image_files_checked": False,
            "pipeline": pipeline_key,
            "pipeline_name": PIPELINES[pipeline_key]["name"],
            "experiment_id": experiment_id,
            "train_rep": training_config["train_rep"],
            "training_config_sha256": training_config_sha256,
            "training_config": dict(training_config),
            "seed_bundle": seed_bundle.to_dict(),
            "output_path": os.path.abspath(out_path),
            "checkpoint_path": os.path.abspath(checkpoint_path),
        }
    )
    return report


def _atomic_savez_compressed(path: str, arrays: Mapping[str, np.ndarray]) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".officehome.", suffix=".npz", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        # mkstemp creates 0600; make the immutable artifact world-readable so a
        # post-hoc consumer on any account (in- or out-of-container) can read it.
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _identity_fingerprint(sample_ids: np.ndarray, clients: np.ndarray) -> str:
    h = hashlib.sha256(b"fedcore.officehome.role.v1\x00")
    for sample_id in np.asarray(sample_ids, dtype=str):
        encoded = sample_id.encode("utf-8")
        h.update(len(encoded).to_bytes(8, "big"))
        h.update(encoded)
    h.update(np.asarray(clients, dtype="<i8").tobytes(order="C"))
    return h.hexdigest()


def build_training_components(args, job, pipeline_key, seed_bundle):
    """Return the torch training pieces shared by the runner and the GPU checks.

    Factoring these out lets the prelaunch GPU driver run the exact same model
    factory, datasets, cosine LR schedule, and AdamW client update as production.
    """

    import torch

    from fedcore.data.officehome import OfficeHomeImageDataset, build_transforms
    from fedcore.models.models import make_model
    from fedcore.models.officehome_train import (
        cosine_warmup_lr,
        make_officehome_local_train_fn,
    )

    pipeline = PIPELINES[pipeline_key]
    lr = float(args.lr) if args.lr is not None else float(pipeline["lr"])
    frozen = bool(pipeline["frozen_encoder"])

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")

    train_transform, eval_transform = build_transforms(args.image_size)
    client_datasets = [
        OfficeHomeImageDataset(records, transform=train_transform)
        for records in job.training_by_client
    ]

    backbone = pipeline.get("backbone", "convnext_tiny")

    def make() -> "torch.nn.Module":
        return make_model(
            job.n_known,
            backbone=backbone,
            pretrained=bool(args.pretrained),
            frozen_encoder=frozen,
        )

    def lr_schedule(round_index: int) -> float:
        return cosine_warmup_lr(round_index, lr, int(args.rounds), int(args.warmup_rounds))

    local_train_fn = make_officehome_local_train_fn(
        weight_decay=float(args.weight_decay), frozen_encoder=frozen
    )
    return {
        "device": device,
        "lr": lr,
        "frozen": frozen,
        "make": make,
        "client_datasets": client_datasets,
        "eval_transform": eval_transform,
        "lr_schedule": lr_schedule,
        "local_train_fn": local_train_fn,
    }


def _seed_torch(seed_bundle) -> None:
    import torch

    torch.manual_seed(seed_bundle.model_init)
    np.random.seed(seed_bundle.model_init)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_bundle.model_init)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_and_export(
    args, job, pipeline_key, seed_bundle, training_config, training_config_sha256, experiment_id, out_path, checkpoint_path
) -> dict:
    if os.path.exists(out_path):
        raise FileExistsError(f"refusing to replace immutable logit artifact {out_path}")
    export_only = bool(getattr(args, "export_only", False))
    if export_only:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"--export-only requires an existing final checkpoint: {checkpoint_path}"
            )
    else:
        if args.resume and not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"--resume requested but checkpoint does not exist: {checkpoint_path}"
            )
        if not args.resume and os.path.exists(checkpoint_path):
            raise FileExistsError(
                f"checkpoint already exists; use --resume or a new cell: {checkpoint_path}"
            )

    import torch

    from fedcore.data.officehome import OfficeHomeImageDataset
    from fedcore.models.fed_train import export_logits, fedavg
    from fedcore.models.models import make_model

    pipeline = PIPELINES[pipeline_key]
    _seed_torch(seed_bundle)
    components = build_training_components(args, job, pipeline_key, seed_bundle)
    device = components["device"]
    lr = components["lr"]
    frozen = components["frozen"]
    make = components["make"]
    client_datasets = components["client_datasets"]
    eval_transform = components["eval_transform"]
    lr_schedule = components["lr_schedule"]
    local_train_fn = components["local_train_fn"]

    import time as _time

    train_seconds = 0.0
    if export_only:
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        meta = dict(payload.get("metadata", {}))
        if meta.get("experiment_id") != experiment_id:
            raise ValueError("export-only checkpoint experiment_id mismatch")
        if meta.get("training_config_sha256") != training_config_sha256:
            raise ValueError("export-only checkpoint training_config_sha256 mismatch")
        round_completed = int(payload.get("round_completed", -1))
        rounds_requested = int(payload.get("rounds_requested", -1))
        if round_completed != rounds_requested - 1:
            raise ValueError("export-only refuses a non-final checkpoint")
        model = make()
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
    else:
        _t_train = _time.time()
        model = fedavg(
            make,
            client_datasets,
            args.rounds,
            args.local_epochs,
            lr,
            args.batch_size,
            device,
            loader_seed=seed_bundle.loader,
            local_train_fn=local_train_fn,
            lr_schedule=lr_schedule,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            checkpoint_metadata={
                "experiment_id": experiment_id,
                "training_config_sha256": training_config_sha256,
                "manifest_sha256": job.manifest_sha256,
                "folds_sha256": job.folds_sha256,
                "seed_ledger_json": seed_bundle.to_json(),
                "resolved_device": str(device),
            },
        )
        train_seconds = _time.time() - _t_train

    _t_export = _time.time()
    raw: dict[str, np.ndarray] = {}
    all_ids: dict[str, set] = {}
    for role in AUDIT_ROLES:
        output_role = OUTPUT_FOLD_NAMES[role]
        records = job.role_records(role)
        dataset = OfficeHomeImageDataset(records, transform=eval_transform)
        logits = export_logits(
            model, dataset, np.arange(len(dataset), dtype=np.int64), device, args.batch_size
        )
        arrays = audit_role_arrays(records, logits, job.known_classes, job.unknown_classes)
        for name, values in arrays.items():
            raw[f"{output_role}_{name}"] = values
        raw[f"{output_role}_fp"] = np.asarray(
            _identity_fingerprint(arrays["sample_id"], arrays["client"])
        )
        all_ids[output_role] = set(arrays["sample_id"].tolist())

    from fedcore.data.officehome import traffic_identity_arrays

    traffic = traffic_identity_arrays(job.role_records(TRAFFIC_ROLE))
    for name, values in traffic.items():
        raw[f"traffic_{name}"] = values
    raw["traffic_fp"] = np.asarray(_identity_fingerprint(traffic["sample_id"], traffic["client"]))
    all_ids["traffic"] = set(traffic["sample_id"].tolist())

    names = tuple(all_ids)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = all_ids[left] & all_ids[right]
            if overlap:
                raise RuntimeError(
                    f"refusing overlapping output roles {left}/{right}: {sorted(overlap)[:3]}"
                )

    raw.update(
        {
            "schema_version": np.asarray(1, dtype=np.int64),
            "dataset": np.asarray(job.config.dataset_name),
            "dataset_version": np.asarray(job.config.dataset_version),
            "experiment_id": np.asarray(experiment_id),
            "split_id": np.asarray(job.split_id),
            "split_seed": np.asarray(job.split_seed, dtype=np.int64),
            "pipeline": np.asarray(pipeline_key),
            "pipeline_name": np.asarray(pipeline["name"]),
            "train_rep": np.asarray(args.train_rep, dtype=np.int64),
            "traffic_identity_only": np.asarray(True),
            "domains": np.asarray(job.domains, dtype=str),
            "known_classes": np.asarray(job.known_classes, dtype=str),
            "unknown_classes": np.asarray(job.unknown_classes, dtype=str),
            "class_to_label_json": np.asarray(canonical_json(dict(job.class_to_label))),
            "manifest_sha256": np.asarray(job.manifest_sha256),
            "folds_sha256": np.asarray(job.folds_sha256),
            "class_splits_sha256": np.asarray(job.class_splits_sha256),
            "training_config_json": np.asarray(canonical_json(dict(training_config))),
            "training_config_sha256": np.asarray(training_config_sha256),
            "seed_ledger_json": np.asarray(seed_bundle.to_json()),
            "model_init_seed": np.asarray(seed_bundle.model_init, dtype=np.int64),
            "loader_seed": np.asarray(seed_bundle.loader, dtype=np.int64),
            "audit_draw_seed": np.asarray(seed_bundle.audit_draw, dtype=np.int64),
            "traffic_draw_seed": np.asarray(seed_bundle.traffic_draw, dtype=np.int64),
            "sample_id_schema": np.asarray("officehome-manifest-content-hash-v1"),
        }
    )
    export_seconds = _time.time() - _t_export
    _atomic_savez_compressed(out_path, raw)
    return {
        "schema_version": 1,
        "status": "completed",
        "dry_run": False,
        "experiment_id": experiment_id,
        "pipeline": pipeline_key,
        "device": str(device),
        "manifest_sha256": job.manifest_sha256,
        "folds_sha256": job.folds_sha256,
        "training_config_sha256": training_config_sha256,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "output_path": os.path.abspath(out_path),
        "role_counts": {role: len(job.role_records(role)) for role in AUDIT_ROLES},
        "traffic_units": len(job.role_records(TRAFFIC_ROLE)),
        "rounds": int(args.rounds),
        "train_seconds": round(float(train_seconds), 3),
        "export_seconds": round(float(export_seconds), 3),
    }


def run(args: argparse.Namespace) -> dict:
    prepared = prepare_job(args)
    if args.dry_run:
        return dry_run_report(*prepared)
    return train_and_export(args, *prepared)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
