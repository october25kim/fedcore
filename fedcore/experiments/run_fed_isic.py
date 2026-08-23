"""Train one resumable Fed-ISIC2019 leave-one-diagnosis-out model.

One invocation is exactly one immutable training cell:

``held-out diagnosis x model replicate``.

Risk targets, confidence levels, mixture radii, threshold/allocation policies,
scores, and certificate variants are intentionally absent from this CLI.  They
are post-hoc analyses over the immutable unit-level logit artifact.

Production training is bound to one exact flat ``TrainingCell.config`` from
``fedcore.campaign.plan`` via ``--plan-cell-config``.  The plan-cell ID/hash and
the runtime-expanded training configuration ID/hash are retained separately.

The ``--dry-run`` path validates metadata, frozen folds, unit separation, label
mapping, and semantic provenance without importing torch, torchvision, or PIL
and without opening an image file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import re
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fedcore.campaign.artifacts import (
    canonical_json,
    semantic_experiment_id,
    semantic_hash,
)
from fedcore.campaign.plan import (
    POSTHOC_ONLY_KEYS,
    load_training_cell_config,
    training_cell_experiment_id,
    validate_training_cell_binding,
)
from fedcore.medical.data import (
    AUDIT_FOLDS,
    OUTPUT_FOLD_NAMES,
    MedicalDataConfig,
    MedicalImageDataset,
    audit_artifact_arrays,
    flatten_unit_images,
    load_fed_isic_job,
    traffic_identity_arrays,
)
from fedcore.seeds import SeedBundle


@dataclass(frozen=True)
class PlanCellBinding:
    """Canonical authoritative plan identity, separate from runtime expansion."""

    config: Mapping[str, Any]
    config_json: str
    config_sha256: str
    experiment_id: str


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Positional and named spellings are both accepted to make the runner usable
    # with scheduler templates and direct shell invocations.
    parser.add_argument("metadata_csv_pos", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("folds_csv_pos", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--metadata-csv", dest="metadata_csv_flag")
    parser.add_argument("--folds-csv", dest="folds_csv_flag")
    # The pre-registered roster holds out a PAIR (2 of 8) per split, so this takes a
    # comma-separated set. The singular spellings remain for leave-one-out callers.
    parser.add_argument(
        "--held-out-diagnoses",
        "--heldout-diagnoses",
        "--held-out-diagnosis",
        "--heldout-diagnosis",
        dest="heldout_diagnosis",
        required=True,
        help="comma-separated unknown-class set, e.g. 'MEL,BCC'",
    )
    parser.add_argument(
        "--model-replicate", "--model-seed", dest="model_replicate", type=int, default=0
    )
    parser.add_argument("--campaign-seed", type=int, default=0)
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="optional assertion; a bound run must match the ID derived from its plan cell",
    )
    parser.add_argument(
        "--plan-cell-config",
        default=None,
        help="canonical flat TrainingCell.config JSON from fedcore.campaign.plan",
    )
    parser.add_argument(
        "--allow-unbound-legacy",
        action="store_true",
        help="permit non-dry legacy execution without an authoritative plan cell",
    )

    parser.add_argument("--center-col", default="center")
    parser.add_argument("--diagnosis-col", default="diagnosis")
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--lesion-col", default="lesion_id")
    parser.add_argument("--image-col", default="image_id")
    parser.add_argument("--unit-col", required=True)
    parser.add_argument("--image-path-col", default=None)
    parser.add_argument(
        "--image-root",
        default=None,
        help="root for relative image paths (default: metadata CSV directory)",
    )
    parser.add_argument(
        "--image-extension",
        default="",
        help="optional suffix, e.g. .jpg, when image IDs are used as paths",
    )
    parser.add_argument("--dataset-name", default="fed-isic2019")

    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--backbone",
        choices=("simplecnn", "resnet18", "efficientnet-b0", "convnext_tiny", "dinov2_vitl14"),
        default="resnet18",
    )
    parser.add_argument(
        "--norm",
        choices=("bn", "gn"),
        default=None,
        help="default: gn for scratch ResNet-18, bn otherwise",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    # The Fed-ISIC arm follows FLamby: Adam + weighted focal + albumentations.
    # The legacy defaults (sgd/ce/torchvision) keep leave-one-out callers unchanged.
    parser.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="sgd")
    parser.add_argument("--loss", choices=("ce", "focal"), default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--transform",
        choices=(
            "torchvision",
            "albumentations-flamby",
            "convnext-imagenet",
            "convnext-imagenet-mildcrop",
            "dinov2-imagenet",
        ),
        default="torchvision",
        help="albumentations-flamby: the pre-registered RandomCrop/CenterCrop+Normalize; "
        "convnext-imagenet: RandomResizedCrop(224)+HFlip / Resize(256)+CenterCrop(224), ImageNet norm; "
        "convnext-imagenet-mildcrop: lesion-preserving Resize(256)+RandomCrop(224)+HFlip for TRAIN, "
        "SAME Resize(256)+CenterCrop(224) eval as convnext-imagenet (corrected intervention)",
    )
    # ConvNeXt (post-primary model intervention) arm: AdamW + cosine warmup.
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-rounds", type=int, default=2)

    parser.add_argument("--out", default=None, help="immutable unit-level *_logits.npz")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    # RATIFICATION_004 §8: export-only re-runs ONLY the evaluation/export stage from a
    # final training checkpoint (round_completed == final). It loads the model weights
    # WITHOUT restoring the training RNG state (an eval forward pass needs none), which
    # also sidesteps the separate broken CUDA-RNG resume path. It never trains and never
    # advances a round; it is refused unless the checkpoint is at the final round.
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="metadata/fold validation only; no torch/image import or file access",
    )
    return parser


def _normalize_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    metadata = args.metadata_csv_flag or args.metadata_csv_pos
    folds = args.folds_csv_flag or args.folds_csv_pos
    if args.metadata_csv_flag and args.metadata_csv_pos:
        parser.error(
            "give metadata CSV either positionally or with --metadata-csv, not both"
        )
    if args.folds_csv_flag and args.folds_csv_pos:
        parser.error("give folds CSV either positionally or with --folds-csv, not both")
    if not metadata:
        parser.error("metadata CSV is required (positional or --metadata-csv)")
    if not folds:
        parser.error("frozen fold CSV is required (positional or --folds-csv)")
    args.metadata_csv = metadata
    args.folds_csv = folds


def _validate_training_arguments(args: argparse.Namespace) -> None:
    if args.norm is None:
        args.norm = (
            "bn"
            if args.backbone in (
                "simplecnn",
                "efficientnet-b0",
                "convnext_tiny",
                "dinov2_vitl14",
            )
            or args.pretrained
            else "gn"
        )
    if args.backbone == "convnext_tiny":
        # Post-primary ConvNeXt-Tiny intervention arm: torchvision ImageNet
        # weights + AdamW + cosine-warmup + ImageNet transforms. It reuses the
        # Office-Home client update / schedule verbatim and never touches the
        # fold/role/export logic. The ORIGINAL intervention used plain CE +
        # convnext-imagenet (RandomResizedCrop default scale). The CORRECTED
        # intervention additionally permits the FLamby weighted-focal loss
        # (--loss focal, imbalance-aware) and the lesion-preserving
        # convnext-imagenet-mildcrop transform, changing ONLY the training recipe
        # (loss + train crop) while every certification-facing element (eval
        # transform, folds, export) stays identical.
        if args.optimizer != "adamw":
            raise ValueError("convnext_tiny requires --optimizer adamw")
        if args.transform not in ("convnext-imagenet", "convnext-imagenet-mildcrop"):
            raise ValueError(
                "convnext_tiny requires --transform convnext-imagenet or "
                "convnext-imagenet-mildcrop"
            )
        if args.loss not in ("ce", "focal"):
            raise ValueError(
                "convnext_tiny supports --loss ce (original) or focal (corrected)"
            )
        if args.warmup_rounds < 0 or args.warmup_rounds > args.rounds:
            raise ValueError("warmup_rounds must be in [0, rounds]")
        if args.weight_decay < 0.0 or not np.isfinite(args.weight_decay):
            raise ValueError("weight_decay must be a finite non-negative value")
    if args.backbone == "dinov2_vitl14":
        # Fed-ISIC DINOv2 ViT-L/14 arm: a FROZEN self-supervised encoder + a fresh
        # linear known-class head, federated with AdamW + cosine-warmup (the
        # linear-probe recipe). It reuses the Office-Home frozen-encoder client
        # update verbatim and never touches the fold/role/export logic. Input is
        # 224x224 (patch-14 -> 16x16 tokens); only 224-compatible ImageNet
        # transforms are permitted. Loss may be plain CE or FLamby weighted focal
        # (imbalance-aware) over the known classes.
        if args.optimizer != "adamw":
            raise ValueError("dinov2_vitl14 requires --optimizer adamw")
        if args.image_size != 224:
            raise ValueError("dinov2_vitl14 requires --image-size 224 (patch-14 grid)")
        if args.transform not in ("dinov2-imagenet", "convnext-imagenet-mildcrop"):
            raise ValueError(
                "dinov2_vitl14 requires --transform dinov2-imagenet or "
                "convnext-imagenet-mildcrop (both 224, ImageNet-norm)"
            )
        if args.loss not in ("ce", "focal"):
            raise ValueError("dinov2_vitl14 supports --loss ce or focal")
        if args.warmup_rounds < 0 or args.warmup_rounds > args.rounds:
            raise ValueError("warmup_rounds must be in [0, rounds]")
        if args.weight_decay < 0.0 or not np.isfinite(args.weight_decay):
            raise ValueError("weight_decay must be a finite non-negative value")
    if args.backbone == "simplecnn" and args.pretrained:
        raise ValueError("simplecnn has no pretrained implementation")
    if args.backbone == "simplecnn" and args.norm != "bn":
        raise ValueError("simplecnn uses BatchNorm; --norm gn would be ignored")
    if args.backbone == "resnet18" and args.pretrained and args.norm != "bn":
        raise ValueError(
            "pretrained resnet18 uses torchvision BatchNorm; --norm gn is unsupported"
        )
    if args.backbone == "efficientnet-b0" and args.norm != "bn":
        raise ValueError(
            "efficientnet-b0 carries its own BatchNorm (FLamby recipe); "
            "--norm gn is unsupported"
        )
    if args.optimizer == "adam" and args.loss != "focal":
        # FLamby's Fed-ISIC baseline pairs Adam with the weighted focal loss; the
        # runner refuses a half-applied recipe rather than quietly mixing arms.
        raise ValueError(
            "the FLamby Fed-ISIC recipe pairs --optimizer adam with --loss focal"
        )
    if args.loss == "focal" and args.backbone not in (
        "efficientnet-b0",
        "convnext_tiny",
        "dinov2_vitl14",
    ):
        raise ValueError(
            "weighted focal loss is the FLamby Fed-ISIC recipe and is wired for "
            "--backbone efficientnet-b0 (canonical) or convnext_tiny (corrected)"
        )
    if args.model_replicate < 0:
        raise ValueError("model_replicate must be non-negative")
    if args.campaign_seed < 0:
        raise ValueError("campaign_seed must be non-negative")
    if args.rounds <= 0 or args.local_epochs <= 0:
        raise ValueError("rounds and local_epochs must be positive")
    if args.lr <= 0.0 or not np.isfinite(args.lr):
        raise ValueError("lr must be a finite positive value")
    if args.batch_size <= 0 or args.image_size <= 0:
        raise ValueError("batch_size and image_size must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if args.experiment_id is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(args.experiment_id)
    ):
        raise ValueError(
            "experiment_id must use only alphanumerics, dot, underscore, and hyphen"
        )


def _declared_runner_values(args: argparse.Namespace) -> dict[str, Any]:
    """Plan-comparable CLI values; execution paths are deliberately excluded."""

    from fedcore.medical.data import normalize_heldout

    heldout = normalize_heldout(args.heldout_diagnosis)
    return {
        "family": "fed_isic2019",
        # Singular key stays the split's canonical identity string ("MEL+BCC" for a
        # pair) so plan binding and experiment IDs keep one spelling per split.
        "heldout_diagnosis": "+".join(heldout),
        "heldout_diagnoses": list(heldout),
        "model_seed": int(args.model_replicate),
        "model_replicate": int(args.model_replicate),
        "campaign_seed": int(args.campaign_seed),
        "dataset": args.dataset_name,
        "dataset_name": args.dataset_name,
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "backbone": args.backbone,
        "norm": args.norm,
        "pretrained": bool(args.pretrained),
        "image_size": int(args.image_size),
        "optimizer_name": args.optimizer,
        "loss_name": args.loss,
        "transform_name": args.transform,
        "weight_decay": float(args.weight_decay),
        "warmup_rounds": int(args.warmup_rounds),
        "device": args.device,
        "device_request": args.device,
        "checkpoint_every": int(args.checkpoint_every),
        "center_col": args.center_col,
        "diagnosis_col": args.diagnosis_col,
        "patient_col": args.patient_col,
        "lesion_col": args.lesion_col,
        "image_col": args.image_col,
        "unit_col": args.unit_col,
        "image_path_col": args.image_path_col,
        "image_extension": args.image_extension,
    }


def _read_strict_json_object(path: str) -> dict[str, Any]:
    """Reject duplicate keys and non-standard numeric constants at every depth."""

    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"plan-cell config contains duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError(f"plan-cell config contains non-finite constant {value!r}")

    with open(path, encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    if not isinstance(value, dict):
        raise ValueError("plan-cell config must be a flat JSON object")
    return value


def load_plan_cell_binding(
    args: argparse.Namespace,
) -> PlanCellBinding | None:
    """Load and bind exactly one flat authoritative medical TrainingCell config."""

    if args.plan_cell_config is None:
        if not args.dry_run and not args.allow_unbound_legacy:
            raise ValueError(
                "non-dry Fed-ISIC training requires --plan-cell-config; "
                "use --allow-unbound-legacy only for an existing legacy workflow"
            )
        return None

    strict_source = _read_strict_json_object(args.plan_cell_config)
    config = load_training_cell_config(args.plan_cell_config)
    if canonical_json(strict_source) != canonical_json(config):
        raise RuntimeError("plan-cell config changed while it was being loaded")
    forbidden = (set(POSTHOC_ONLY_KEYS) | {"gamma", "gammas"}) & set(config)
    if forbidden:
        raise ValueError(
            f"plan-cell training config contains post-hoc-only keys: {sorted(forbidden)}"
        )
    declared = _declared_runner_values(args)
    required = {
        "family": declared["family"],
        "heldout_diagnosis": declared["heldout_diagnosis"],
        "model_seed": declared["model_seed"],
        "campaign_seed": declared["campaign_seed"],
    }
    # Every additional plan key that names a runner training argument is an
    # assertion, never an instruction that silently overrides the CLI.
    required.update({key: value for key, value in declared.items() if key in config})
    validate_training_cell_binding(config, required)

    try:
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "plan-cell config must contain finite canonical JSON values"
        ) from exc
    config_json = canonical_json(config)
    config_sha256 = semantic_hash(config)
    experiment_id = training_cell_experiment_id(config)
    if args.experiment_id is not None and args.experiment_id != experiment_id:
        raise ValueError(
            "--experiment-id disagrees with the authoritative plan-cell ID: "
            f"{args.experiment_id!r} != {experiment_id!r}"
        )
    return PlanCellBinding(
        config=_freeze_json(config),
        config_json=config_json,
        config_sha256=config_sha256,
        experiment_id=experiment_id,
    )


def _validate_dynamic_plan_overlap(
    binding: PlanCellBinding | None, training_config: Mapping[str, Any]
) -> None:
    if binding is None:
        return
    overlap = set(binding.config) & set(training_config)
    validate_training_cell_binding(
        binding.config,
        {key: training_config[key] for key in sorted(overlap)},
    )


def _center_roster_hash(centers: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json(list(centers)).encode("utf-8")).hexdigest()


def build_seed_bundle(job, *, campaign_seed: int, model_replicate: int) -> SeedBundle:
    """Derive all streams from immutable job identity, never post-hoc knobs."""

    cell_id = (
        f"{job.config.dataset_name}/heldout={job.heldout_label}/"
        f"model={model_replicate}"
    )
    immutable = {
        "dataset_sha256": job.dataset_sha256,
        "fold_sha256": job.fold_sha256,
        "heldout_diagnosis": job.heldout_label,
        "model_replicate": int(model_replicate),
    }
    return SeedBundle.derive(
        campaign_seed,
        common_context={
            "campaign": "fedcore-oneshot",
            "dataset": job.config.dataset_name,
        },
        namespace_contexts={
            "class_split": {
                "heldout_diagnosis": job.heldout_label,
                "dataset_sha256": job.dataset_sha256,
            },
            "partition": {
                "center_roster_sha256": _center_roster_hash(job.centers),
                "dataset_sha256": job.dataset_sha256,
            },
            "fold": {"fold_sha256": job.fold_sha256},
            "model_init": immutable,
            "loader": immutable,
            "label_noise": {**immutable, "mode": "none"},
            "audit_draw": {
                "experiment_id": cell_id,
                "fold_sha256": job.fold_sha256,
                "draw_index": 0,
            },
            "traffic_draw": {
                "experiment_id": cell_id,
                "fold_sha256": job.fold_sha256,
                "draw_index": 0,
            },
            "solver": {"experiment_id": cell_id},
            "stability": {"experiment_id": cell_id, "redraw_index": 0},
        },
    )


def build_training_config(
    args: argparse.Namespace, job, seed_bundle: SeedBundle
) -> dict:
    """Canonical training-only configuration used for reuse and resume checks."""

    return {
        "schema_version": 1,
        "family": "fed_isic2019",
        "dataset": job.config.dataset_name,
        "dataset_sha256": job.dataset_sha256,
        "dataset_hash_scope": "metadata-csv-bytes",
        "fold_sha256": job.fold_sha256,
        "unit_kind": job.config.unit_kind,
        "metadata_columns": {
            "center": job.config.center_col,
            "diagnosis": job.config.diagnosis_col,
            "patient": job.config.patient_col,
            "lesion": job.config.lesion_col,
            "image": job.config.image_col,
            "unit": job.config.unit_col,
            "image_path": job.config.image_path_col,
        },
        # The metadata hash binds image IDs and any declared relative path
        # column.  A mount/root is execution location, not dataset semantics;
        # excluding it keeps the same frozen dataset reusable across hosts.
        "image_source": {"extension": job.config.image_extension},
        "heldout_diagnosis": job.heldout_label,
        "heldout_diagnoses": list(job.heldout_diagnoses),
        "known_diagnoses": list(job.known_diagnoses),
        "centers": list(job.centers),
        "n_clients": job.n_clients,
        "n_known": job.n_known,
        "model_replicate": int(args.model_replicate),
        "campaign_seed": int(args.campaign_seed),
        # Only streams that can alter learned weights belong to the semantic
        # training identity.  The complete bundle (including audit/traffic and
        # solver streams) is persisted separately, so post-hoc replay cannot
        # accidentally multiply or invalidate a trained model.
        "training_seeds": {
            "model_init": seed_bundle.model_init,
            "loader": seed_bundle.loader,
        },
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "backbone": args.backbone,
        "norm": args.norm,
        "pretrained": bool(args.pretrained),
        "image_size": int(args.image_size),
        "image_transform": {
            "albumentations-flamby": "flamby-albumentations-randomcrop200-normalize-v1",
            "convnext-imagenet": "convnext-randomresizedcrop224-hflip-centercrop224-imagenet-v1",
            "convnext-imagenet-mildcrop": "convnext-resize256-randomcrop224-hflip-centercrop224-imagenet-v1",
            "dinov2-imagenet": "dinov2-resize256-randomcrop224-hflip-centercrop224-imagenet-v1",
        }.get(args.transform, "resize-random-horizontal-flip-imagenet-v1"),
        "optimizer": {
            "adam": f"adam-lr-{float(args.lr):g}",
            "adamw": f"adamw-lr-{float(args.lr):g}-wd-{float(args.weight_decay):g}",
        }.get(args.optimizer, "sgd-momentum-0.9-weight-decay-5e-4"),
        "lr_schedule": (
            f"cosine-warmup-v1-warmup-{int(args.warmup_rounds)}"
            if args.backbone in ("convnext_tiny", "dinov2_vitl14")
            else "constant"
        ),
        "loss": (
            f"flamby-weighted-focal-gamma-{float(args.focal_gamma):g}"
            if args.loss == "focal"
            else "cross-entropy"
        ),
        "federated_optimizer": "size-weighted-fedavg",
        "deterministic_replay": True,
        "device_request": args.device,
    }


def prepare_job(
    args: argparse.Namespace,
) -> tuple[object, SeedBundle, dict, str, str, str, str, PlanCellBinding | None]:
    """Validate every metadata input and resolve immutable output identities."""

    _validate_training_arguments(args)
    plan_binding = load_plan_cell_binding(args)
    data_config = MedicalDataConfig(
        metadata_csv=args.metadata_csv,
        folds_csv=args.folds_csv,
        center_col=args.center_col,
        diagnosis_col=args.diagnosis_col,
        patient_col=args.patient_col,
        lesion_col=args.lesion_col,
        image_col=args.image_col,
        unit_col=args.unit_col,
        image_path_col=args.image_path_col,
        image_root=args.image_root,
        image_extension=args.image_extension,
        dataset_name=args.dataset_name,
    )
    # The non-dry path validates every image pathname before torch is imported.
    job = load_fed_isic_job(
        data_config,
        args.heldout_diagnosis,
        check_image_files=not args.dry_run,
    )
    seed_bundle = build_seed_bundle(
        job,
        campaign_seed=args.campaign_seed,
        model_replicate=args.model_replicate,
    )
    training_config = build_training_config(args, job, seed_bundle)
    _validate_dynamic_plan_overlap(plan_binding, training_config)
    training_config_sha256 = semantic_hash(training_config)
    prefix = f"fed-isic-{job.heldout_label}-model{args.model_replicate}"
    generated_experiment_id = semantic_experiment_id(prefix, training_config)
    experiment_id = (
        plan_binding.experiment_id
        if plan_binding is not None
        else args.experiment_id or generated_experiment_id
    )
    out_path = args.out or os.path.join("runs", f"{experiment_id}_logits.npz")
    checkpoint_path = args.checkpoint or os.path.join(
        "runs", "checkpoints", f"{experiment_id}.pt"
    )
    output_abs = os.path.abspath(out_path)
    checkpoint_abs = os.path.abspath(checkpoint_path)
    input_paths = {
        os.path.abspath(args.metadata_csv),
        os.path.abspath(args.folds_csv),
    }
    if args.plan_cell_config is not None:
        input_paths.add(os.path.abspath(args.plan_cell_config))
    if output_abs == checkpoint_abs:
        raise ValueError("output artifact and FedAvg checkpoint paths must be distinct")
    if {output_abs, checkpoint_abs} & input_paths:
        raise ValueError(
            "output/checkpoint paths must not overwrite metadata or frozen folds"
        )
    return (
        job,
        seed_bundle,
        training_config,
        training_config_sha256,
        experiment_id,
        out_path,
        checkpoint_path,
        plan_binding,
    )


def dry_run_report(
    job,
    seed_bundle: SeedBundle,
    training_config: Mapping,
    training_config_sha256: str,
    experiment_id: str,
    out_path: str,
    checkpoint_path: str,
    plan_binding: PlanCellBinding | None,
) -> dict:
    report = job.summary()
    report.update(
        {
            "schema_version": 1,
            "status": "metadata_validated",
            "dry_run": True,
            "image_files_checked": False,
            "pixel_provenance": "not-checked-metadata-only",
            "experiment_id": experiment_id,
            "model_replicate": training_config["model_replicate"],
            "training_config_sha256": training_config_sha256,
            "training_config_json": canonical_json(dict(training_config)),
            "training_config": dict(training_config),
            "plan_cell_bound": plan_binding is not None,
            "plan_cell_config_json": (
                plan_binding.config_json if plan_binding is not None else None
            ),
            "plan_cell_config_sha256": (
                plan_binding.config_sha256 if plan_binding is not None else None
            ),
            "seed_bundle": seed_bundle.to_dict(),
            "output_path": os.path.abspath(out_path),
            "checkpoint_path": os.path.abspath(checkpoint_path),
        }
    )
    return report


def _build_transforms(image_size: int, transform: str = "torchvision"):
    """Return ``(train_transform, eval_transform)``.

    ``albumentations-flamby`` is the pre-registered Fed-ISIC pipeline
    (``RandomCrop(200,200)+Normalize`` / ``CenterCrop(200,200)+Normalize``). See
    :mod:`fedcore.medical.flamby` for the two declared deviations from FLamby main
    (dropped train augmentations; undeclared resize/colour-constancy preprocessing).
    """
    if transform == "albumentations-flamby":
        from fedcore.medical.flamby import eval_transform, train_transform

        return train_transform(image_size), eval_transform(image_size)

    if transform == "convnext-imagenet":
        # Reuse the audited Office-Home ConvNeXt ImageNet pipeline verbatim:
        #   train: RandomResizedCrop(224) + HFlip + ImageNet norm
        #   eval:  Resize(256) + CenterCrop(224) + ImageNet norm
        from fedcore.data.officehome import build_transforms

        return build_transforms(image_size)

    if transform in ("convnext-imagenet-mildcrop", "dinov2-imagenet"):
        # Lesion-preserving 224 ImageNet pipeline, shared by the CORRECTED ConvNeXt
        # intervention (audit confound #2) and the DINOv2 ViT-L/14 linear-probe arm.
        # It replaces the aggressive RandomResizedCrop(scale=(0.08,1.0)) -- which can
        # crop the dermoscopic lesion entirely out of a training image -- with a
        # lesion-preserving Resize(shorter->256)+RandomCrop(224). The EVAL pipeline
        # is Resize(256)+CenterCrop(224)+ImageNet norm (byte-identical to the
        # ``convnext-imagenet`` eval), so the deterministic logits that drive the
        # certificate come from a fixed 224-crop eval; only the stochastic TRAIN crop
        # differs. 224 is patch-14 compatible for DINOv2 (16x16 tokens). ImageNet
        # mean/std are applied exactly once (ToTensor->Normalize).
        import torchvision.transforms as transforms

        from fedcore.data.officehome import IMAGENET_MEAN, IMAGENET_STD

        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        resize = int(round(image_size * 256 / 224))
        train = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        evaluate = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return train, evaluate

    # Imported only after all metadata and image paths have validated.
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluate = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train, evaluate


def _atomic_savez_compressed(path: str, arrays: Mapping[str, np.ndarray]) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".fed_isic.", suffix=".npz", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _identity_fingerprint(
    sample_ids: np.ndarray,
    clients: np.ndarray,
    labels: np.ndarray | None = None,
    multiplicities: np.ndarray | None = None,
) -> str:
    """Portable hash over ordered, distinct unit rows."""

    h = hashlib.sha256(b"fedcore.medical.unit-fold.v1\x00")
    for sample_id in np.asarray(sample_ids, dtype=str):
        encoded = sample_id.encode("utf-8")
        h.update(len(encoded).to_bytes(8, "big"))
        h.update(encoded)
    h.update(np.asarray(clients, dtype="<i8").tobytes(order="C"))
    if labels is not None:
        h.update(np.asarray(labels, dtype="<i8").tobytes(order="C"))
    if multiplicities is not None:
        h.update(np.asarray(multiplicities, dtype="<i8").tobytes(order="C"))
    return h.hexdigest()


def train_and_export(
    args: argparse.Namespace,
    job,
    seed_bundle: SeedBundle,
    training_config: Mapping,
    training_config_sha256: str,
    experiment_id: str,
    out_path: str,
    checkpoint_path: str,
    plan_binding: PlanCellBinding | None,
) -> dict:
    """Run FedAvg and atomically persist distinct-unit outputs."""

    if os.path.exists(out_path):
        raise FileExistsError(
            f"refusing to replace immutable logit artifact {out_path}"
        )
    export_only = getattr(args, "export_only", False)
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
                f"checkpoint already exists; use --resume or a new training cell: {checkpoint_path}"
            )

    # Heavy imports happen only after prepare_job has validated all inputs.
    import torch

    from fedcore.models.fed_train import (
        export_logits,
        export_logits_and_penultimate,
        fedavg,
    )
    from fedcore.medical.data import aggregate_unit_logits
    from fedcore.models.models import make_model

    capture_embedding = args.backbone in ("convnext_tiny", "dinov2_vitl14")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    if args.resume:
        # FedAvg validates the semantic config/loader/round count.  The runner
        # additionally binds a checkpoint to the resolved CPU/GPU backend so an
        # ``auto`` resume cannot silently cross numerical implementations.
        try:
            checkpoint_payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:  # torch versions predating the weights_only keyword
            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        stored_device = dict(checkpoint_payload.get("metadata", {})).get(
            "resolved_device"
        )
        if stored_device != str(device):
            raise ValueError(
                "checkpoint resolved_device does not match this run: "
                f"{stored_device!r} != {str(device)!r}"
            )
        checkpoint_metadata = dict(checkpoint_payload.get("metadata", {}))
        expected_plan_hash = (
            plan_binding.config_sha256 if plan_binding is not None else ""
        )
        if checkpoint_metadata.get("plan_cell_config_sha256", "") != expected_plan_hash:
            raise ValueError(
                "checkpoint plan-cell binding does not match the requested run"
            )
        if checkpoint_metadata.get("experiment_id") != experiment_id:
            raise ValueError(
                "checkpoint experiment_id does not match the requested run"
            )

    torch.manual_seed(seed_bundle.model_init)
    np.random.seed(seed_bundle.model_init)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_bundle.model_init)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_transform, eval_transform = _build_transforms(args.image_size, args.transform)
    client_datasets = [
        MedicalImageDataset(records, transform=train_transform)
        for records in job.training_by_client
    ]
    # FLamby's client update: Adam (lr from common.py) + weighted focal loss, with
    # alpha restricted BY INDEX to this split's known classes and not renormalized.
    # The alpha order follows job.known_diagnoses, which is exactly the head's
    # column order (diagnosis_to_label enumerates known_diagnoses).
    local_train_fn = None
    lr_schedule = None
    if args.loss == "focal":
        from fedcore.medical.flamby import make_local_train_fn

        local_train_fn = make_local_train_fn(
            job.known_diagnoses, gamma=args.focal_gamma
        )
    if args.backbone == "convnext_tiny":
        # Post-primary ConvNeXt arm: reuse the audited Office-Home AdamW client
        # update and the pure-function cosine-warmup LR schedule (both restore
        # bit-exactly on --resume because they carry no cross-round optimizer
        # state and depend only on the round index). The CORRECTED intervention
        # swaps in FLamby's imbalance-aware weighted focal criterion when
        # --loss focal (restricted BY INDEX to this split's known classes, same
        # policy as the efficientnet arm), keeping the AdamW+cosine schedule and
        # every certification-facing element unchanged. --loss ce reproduces the
        # ORIGINAL intervention (plain cross-entropy) byte-for-byte.
        from fedcore.models.officehome_train import (
            cosine_warmup_lr,
            make_officehome_local_train_fn,
        )

        convnext_criterion = None
        if args.loss == "focal":
            from fedcore.medical.flamby import make_focal_loss

            convnext_criterion = make_focal_loss(
                job.known_diagnoses, gamma=args.focal_gamma
            )
        local_train_fn = make_officehome_local_train_fn(
            weight_decay=float(args.weight_decay),
            frozen_encoder=False,
            criterion=convnext_criterion,
        )

        def lr_schedule(round_index: int) -> float:
            return cosine_warmup_lr(
                round_index, float(args.lr), int(args.rounds), int(args.warmup_rounds)
            )
    if args.backbone == "dinov2_vitl14":
        # Fed-ISIC DINOv2 ViT-L/14 LINEAR-PROBE arm: the frozen self-supervised
        # encoder runs in eval mode (deterministic, no drop-path) and ONLY the
        # linear known-class head is federated with AdamW + cosine-warmup. Reuses
        # the audited Office-Home frozen-encoder client update verbatim; the FLamby
        # weighted-focal criterion (imbalance-aware, known-class-restricted) is used
        # when --loss focal, else plain cross-entropy. frozen_encoder=True makes the
        # per-round update depend only on the round's head weights + minibatch order,
        # so --resume restores bit-exactly.
        from fedcore.models.officehome_train import (
            cosine_warmup_lr,
            make_officehome_local_train_fn,
        )

        dinov2_criterion = None
        if args.loss == "focal":
            from fedcore.medical.flamby import make_focal_loss

            dinov2_criterion = make_focal_loss(
                job.known_diagnoses, gamma=args.focal_gamma
            )
        local_train_fn = make_officehome_local_train_fn(
            weight_decay=float(args.weight_decay),
            frozen_encoder=True,
            criterion=dinov2_criterion,
        )

        def lr_schedule(round_index: int) -> float:
            return cosine_warmup_lr(
                round_index, float(args.lr), int(args.rounds), int(args.warmup_rounds)
            )
    if export_only:
        # RATIFICATION_004 §8: load the final model weights ONLY (no RNG restore, no
        # training, no round advance). Verify the checkpoint is at the final round and
        # that its identity bindings match this cell, so an export cannot silently pull
        # a wrong or unfinished model.
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        meta = dict(payload.get("metadata", {}))
        if meta.get("experiment_id") != experiment_id:
            raise ValueError("export-only checkpoint experiment_id does not match this run")
        if meta.get("training_config_sha256") != training_config_sha256:
            raise ValueError("export-only checkpoint training_config_sha256 mismatch")
        round_completed = int(payload.get("round_completed", -1))
        rounds_requested = int(payload.get("rounds_requested", -1))
        if round_completed != rounds_requested - 1:
            raise ValueError(
                "export-only refuses a non-final checkpoint: "
                f"round_completed={round_completed}, rounds_requested={rounds_requested}"
            )
        model = make_model(
            job.n_known,
            backbone=args.backbone,
            norm=args.norm,
            pretrained=args.pretrained,
        )
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
    else:
        model = fedavg(
            lambda: make_model(
                job.n_known,
                backbone=args.backbone,
                norm=args.norm,
                pretrained=args.pretrained,
            ),
            client_datasets,
            args.rounds,
            args.local_epochs,
            args.lr,
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
                "dataset_sha256": job.dataset_sha256,
                "fold_sha256": job.fold_sha256,
                "seed_ledger_json": seed_bundle.to_json(),
                "resolved_device": str(device),
                "plan_cell_config_sha256": (
                    plan_binding.config_sha256 if plan_binding is not None else ""
                ),
            },
        )

    raw: dict[str, np.ndarray] = {}
    all_output_ids: dict[str, set[str]] = {}
    for source_fold in AUDIT_FOLDS:
        output_fold = OUTPUT_FOLD_NAMES[source_fold]
        units = job.fold_units(source_fold)
        images, parent_ids = flatten_unit_images(units)
        dataset = MedicalImageDataset(images, transform=eval_transform)
        indices = np.arange(len(dataset), dtype=np.int64)
        if capture_embedding:
            # ConvNeXt (768-d) / DINOv2 ViT-L/14 (1024-d) arms: also export the
            # penultimate embedding (the pre-classifier feature), aggregated to unit
            # level (mean over a unit's images) with the SAME grouping used for the
            # unit-level logits, so `{fold}_embedding[i]` aligns row-for-row with
            # `{fold}_logits[i]` / `{fold}_sample_id[i]`. The classifier head module
            # differs by backbone: convnext -> classifier[2]; timm ViT -> head.
            head_module = (
                model.get_classifier()
                if args.backbone == "dinov2_vitl14"
                else model.classifier[2]
            )
            image_logits, image_emb = export_logits_and_penultimate(
                model, dataset, indices, device, head_module, args.batch_size
            )
            unit_order = [unit.sample_id for unit in units]
            unit_emb, _ = aggregate_unit_logits(image_emb, parent_ids, unit_order)
            raw[f"{output_fold}_embedding"] = np.asarray(unit_emb, dtype=np.float32)
        else:
            image_logits = export_logits(
                model, dataset, indices, device, args.batch_size
            )
        arrays = audit_artifact_arrays(units, image_logits, job.heldout_diagnoses)
        for name, values in arrays.items():
            raw[f"{output_fold}_{name}"] = values
        raw[f"{output_fold}_fp"] = np.asarray(
            _identity_fingerprint(
                arrays["sample_id"],
                arrays["client"],
                arrays["y_open"],
                arrays["image_multiplicity"],
            )
        )
        all_output_ids[output_fold] = set(arrays["sample_id"].tolist())

    # RATIFICATION_004: three INDEPENDENT support statuses, one per fold. A missing
    # declared unknown in a fold sets exactly one status false with its own reason;
    # the statuses are never collapsed into a single verdict. `support_complete` was
    # threaded per fold by audit_artifact_arrays above (prop/cert/test).
    def _support(fold: str) -> bool:
        return bool(np.asarray(raw[f"{fold}_support_complete"]).item())

    proposal_ok = _support("prop")
    certification_a3_ok = _support("cert")
    evaluation_defined = _support("test")
    raw["proposal_support_valid"] = np.asarray(proposal_ok, dtype=bool)
    raw["certification_a3_valid"] = np.asarray(certification_a3_ok, dtype=bool)
    raw["evaluation_unknown_metrics_defined"] = np.asarray(evaluation_defined, dtype=bool)
    raw["proposal_support_reason"] = np.asarray(
        "" if proposal_ok else "proposal_missing_declared_unknown_support"
    )
    raw["certification_a3_reason"] = np.asarray(
        "" if certification_a3_ok else "infeasible_a3_missing_unknown_support"
    )
    raw["evaluation_unknown_reason"] = np.asarray(
        "" if evaluation_defined else "undefined_no_test_unknown_support"
    )

    # A traffic fold is optional: the sealed prereg declares none for Fed-ISIC
    # (data.folds spends the whole audit pool on prop/cert/test), and the
    # pre-registered targets [simplex, box] never enter the traffic mixture mode.
    # When a frozen fold artifact does carry traffic units they are exported exactly
    # as before -- identity only, never a label or a model output.
    traffic_units = job.fold_units("traffic")
    if traffic_units:
        traffic = traffic_identity_arrays(traffic_units)
        for name, values in traffic.items():
            raw[f"traffic_{name}"] = values
        raw["traffic_fp"] = np.asarray(
            _identity_fingerprint(traffic["sample_id"], traffic["client"])
        )
        all_output_ids["traffic"] = set(traffic["sample_id"].tolist())
    names = tuple(all_output_ids)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = all_output_ids[left] & all_output_ids[right]
            if overlap:
                raise RuntimeError(
                    f"refusing overlapping output units in {left}/{right}: "
                    f"{sorted(overlap)[:3]}"
                )

    raw.update(
        {
            "schema_version": np.asarray(1, dtype=np.int64),
            "dataset": np.asarray(job.config.dataset_name),
            "experiment_id": np.asarray(experiment_id),
            "heldout_diagnosis": np.asarray(job.heldout_label),
            "heldout_diagnoses": np.asarray(job.heldout_diagnoses, dtype=str),
            "model_replicate": np.asarray(args.model_replicate, dtype=np.int64),
            "unit_kind": np.asarray(job.config.unit_kind),
            "unit_sample_id_schema": np.asarray("opaque-sha256-v1"),
            "unit_logit_aggregation": np.asarray("mean-image-logits-v1"),
            "penultimate_embedding_included": np.asarray(bool(capture_embedding)),
            "penultimate_embedding_schema": np.asarray(
                {
                    "convnext_tiny": "convnext-tiny-768d-mean-image-preclassifier-v1",
                    "dinov2_vitl14": "dinov2-vitl14-1024d-mean-image-preclassifier-v1",
                }.get(args.backbone, "")
                if capture_embedding
                else ""
            ),
            "backbone": np.asarray(args.backbone),
            "traffic_identity_only": np.asarray(True),
            "center_labels": np.asarray(job.centers, dtype=str),
            "known_diagnoses": np.asarray(job.known_diagnoses, dtype=str),
            "diagnosis_to_label_json": np.asarray(
                canonical_json(dict(job.diagnosis_to_label))
            ),
            "dataset_sha256": np.asarray(job.dataset_sha256),
            "dataset_hash_scope": np.asarray("metadata-csv-bytes"),
            "pixel_provenance": np.asarray("paths-checked-content-not-hashed"),
            "metadata_sha256": np.asarray(job.metadata_sha256),
            "fold_sha256": np.asarray(job.fold_sha256),
            "training_config_json": np.asarray(canonical_json(dict(training_config))),
            "training_config_sha256": np.asarray(training_config_sha256),
            "plan_cell_bound": np.asarray(plan_binding is not None),
            "plan_cell_config_json": np.asarray(
                plan_binding.config_json if plan_binding is not None else ""
            ),
            "plan_cell_config_sha256": np.asarray(
                plan_binding.config_sha256 if plan_binding is not None else ""
            ),
            "seed_ledger_json": np.asarray(seed_bundle.to_json()),
            "model_init_seed": np.asarray(seed_bundle.model_init, dtype=np.int64),
            "loader_seed": np.asarray(seed_bundle.loader, dtype=np.int64),
        }
    )
    _atomic_savez_compressed(out_path, raw)
    return {
        "schema_version": 1,
        "status": "completed",
        "dry_run": False,
        "experiment_id": experiment_id,
        "device": str(device),
        "dataset_sha256": job.dataset_sha256,
        "dataset_hash_scope": "metadata-csv-bytes",
        "pixel_provenance": "paths-checked-content-not-hashed",
        "fold_sha256": job.fold_sha256,
        "training_config_sha256": training_config_sha256,
        "plan_cell_bound": plan_binding is not None,
        "plan_cell_config_sha256": (
            plan_binding.config_sha256 if plan_binding is not None else None
        ),
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "output_path": os.path.abspath(out_path),
        "unit_counts": {
            output: len(job.fold_units(source))
            for source, output in OUTPUT_FOLD_NAMES.items()
        },
        "traffic_units": len(job.fold_units("traffic")),
    }


def run(args: argparse.Namespace) -> dict:
    prepared = prepare_job(args)
    (
        job,
        seed_bundle,
        config,
        config_hash,
        experiment_id,
        out_path,
        checkpoint,
        plan_binding,
    ) = prepared
    if args.dry_run:
        return dry_run_report(
            job,
            seed_bundle,
            config,
            config_hash,
            experiment_id,
            out_path,
            checkpoint,
            plan_binding,
        )
    return train_and_export(
        args,
        job,
        seed_bundle,
        config,
        config_hash,
        experiment_id,
        out_path,
        checkpoint,
        plan_binding,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _normalize_paths(args, parser)
    result = run(args)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
