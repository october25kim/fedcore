"""Real CIFAR-10/100 FedOSR run (torch + torchvision, GPU).

End-to-end: open-set split -> Dirichlet non-IID client partition -> client-side
TRAIN-label corruption -> FedAvg training -> trusted calibration from the CLEAN
test set -> the IDENTICAL certification path as ``run_smoke.py`` (scored_views ->
certify_grid for ``Lambda in {simplex, box}``).

Example::

    python -m fedcore.experiments.run_cifar --dataset cifar10 --n_known 6 \
        --n_clients 5 --dirichlet_alpha 0.1 --rounds 50 --local_epochs 2 \
        --alpha 0.10 --delta 0.10 --noise_type symmetric --noise_rate 0.35
"""

from __future__ import annotations

import argparse
import hashlib
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T

from fedcore.campaign.artifacts import canonical_json, semantic_hash
from fedcore.campaign.plan import (
    load_training_cell_config,
    training_cell_experiment_id,
    validate_training_cell_binding,
)
from fedcore.certify import certify_best_gamma, certify_grid
from fedcore.config import FedOSRConfig
from fedcore.models.fed_train import export_logits, fedavg
from fedcore.data.fedosr_split import (
    add_split_fingerprint,
    build_calibration,
    build_identity_only_traffic,
    dirichlet_partition,
    open_set_split,
)
from fedcore.models.models import make_model
from fedcore.data.noise import make_label_noise
from fedcore.experiments.run_smoke import print_metric_table, save_csv
from fedcore.scores import scored_views
from fedcore.seeds import SeedBundle


# --------------------------------------------------------------------------- #
# label-remapping subset (applies known-class remap + optional noise override)
# --------------------------------------------------------------------------- #
class _LabelRemapSubset(Dataset):
    """Subset of ``base`` exposing remapped (and optionally corrupted) labels."""

    def __init__(self, base, indices, remap: Dict[int, int], label_override=None):
        self.base = base
        self.indices = list(indices)
        self.remap = remap
        self.label_override = label_override or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x, y = self.base[idx]
        if idx in self.label_override:
            label = self.label_override[idx]
        else:
            label = self.remap[int(y)]
        return x, label


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
_NORM = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def _load_cifar(dataset: str, root: str):
    mean, std = _NORM[dataset]
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    cls = (
        torchvision.datasets.CIFAR10
        if dataset == "cifar10"
        else torchvision.datasets.CIFAR100
    )
    train = cls(root=root, train=True, download=True, transform=tf)
    test = cls(root=root, train=False, download=True, transform=tf)
    return train, test


def _cifar_dataset_sha256(dataset: Dataset) -> str:
    """Hash the exact extracted torchvision CIFAR train/test/meta bytes."""

    root = os.path.join(str(dataset.root), str(dataset.base_folder))
    file_names = {
        str(file_name)
        for file_name, _ in tuple(dataset.train_list) + tuple(dataset.test_list)
    }
    meta_file = str(dataset.meta.get("filename", ""))
    if meta_file:
        file_names.add(meta_file)
    digest = hashlib.sha256(b"fedcore.cifar-dataset.v1\x00")
    for file_name in sorted(file_names):
        encoded = file_name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        path = os.path.join(root, file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CIFAR dataset component is missing: {path}")
        size = os.path.getsize(path)
        digest.update(size.to_bytes(8, "big"))
        with open(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _gather_fold(calib, fold: str):
    """Concatenate (idx, y_open, client) across clients for a fold."""
    idx, y_open, client = [], [], []
    for j, cf in enumerate(calib):
        f = cf[fold]
        idx.append(np.asarray(f["idx"]))
        y_open.append(np.asarray(f["y_open"]))
        client.append(np.full(len(f["idx"]), j))
    return (
        np.concatenate(idx),
        np.concatenate(y_open),
        np.concatenate(client),
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--n_known", type=int, default=6)
    ap.add_argument("--n_clients", type=int, default=5)
    ap.add_argument("--dirichlet_alpha", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument(
        "--noise_type", choices=["none", "symmetric", "asymmetric"], default="none"
    )
    ap.add_argument("--noise_rate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--campaign_seed",
        type=int,
        default=None,
        help="opt in to stable semantic seed namespaces",
    )
    ap.add_argument(
        "--split_id",
        default="split0",
        help="predeclared class-split identity used in semantic seeds",
    )
    ap.add_argument(
        "--model_seed",
        type=int,
        default=None,
        help="model replicate identity (default: legacy --seed)",
    )
    ap.add_argument(
        "--experiment_id",
        default=None,
        help="semantic training-cell ID recorded in seed/artifact ledgers",
    )
    ap.add_argument(
        "--plan_cell_config",
        default=None,
        help="exact flat TrainingCell.config JSON for a primary campaign job",
    )
    ap.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="class-split RNG seed (default: legacy --seed)",
    )
    ap.add_argument(
        "--partition_seed",
        type=int,
        default=None,
        help="client-partition RNG seed (default: legacy --seed)",
    )
    ap.add_argument(
        "--fold_seed",
        type=int,
        default=None,
        help="proposal/certification/test split seed (default: legacy --seed)",
    )
    ap.add_argument(
        "--train_seed",
        type=int,
        default=None,
        help="model initialization/training seed (default: legacy --seed)",
    )
    ap.add_argument(
        "--loader_seed",
        type=int,
        default=None,
        help="minibatch-order root seed (default: legacy --seed)",
    )
    ap.add_argument(
        "--noise_seed",
        type=int,
        default=None,
        help="training-label corruption root seed (default: legacy --seed)",
    )
    ap.add_argument(
        "--solver_seed",
        type=int,
        default=None,
        help="legacy sampled-box solver seed (default: legacy --seed)",
    )
    ap.add_argument(
        "--traffic_seed",
        type=int,
        default=None,
        help="identity-only traffic draw seed (default: semantic namespace or --seed)",
    )
    ap.add_argument(
        "--traffic_size",
        type=int,
        default=0,
        help="number of test identities reserved before labeled audit folds",
    )
    ap.add_argument(
        "--traffic_client_probs",
        default=None,
        help="required with --traffic_size: comma-separated predeclared client mixture",
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="atomic round checkpoint path for restart-safe training",
    )
    ap.add_argument("--checkpoint_every", type=int, default=1)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume from --checkpoint without changing seeds/config",
    )
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default="runs/cifar_results.csv")
    ap.add_argument(
        "--alpha_frontier",
        action="store_true",
        help="also report CertifiedCoverage@alpha for alpha in {.10,.15,.20,.25}",
    )
    ap.add_argument(
        "--proxy_margin",
        type=float,
        default=0.01,
        help="proposal-proxy safety margin for best-gamma (frontier monotonicity)",
    )
    ap.add_argument(
        "--backbone", choices=["simplecnn", "resnet18"], default="simplecnn"
    )
    ap.add_argument(
        "--norm",
        choices=["bn", "gn"],
        default="bn",
        help="resnet18 normalization: bn (BatchNorm) or gn (GroupNorm, FL-appropriate)",
    )
    ap.add_argument(
        "--pretrained",
        action="store_true",
        help="resnet18: load torchvision ImageNet weights",
    )
    ap.add_argument(
        "--unknown_classes",
        default=None,
        help="comma-separated class ids to FIX as unknown (open-set split held "
        "constant across seeds); default None = seed-driven random split",
    )
    args = ap.parse_args()
    fixed_unknown = (
        [int(c) for c in args.unknown_classes.split(",")]
        if args.unknown_classes
        else None
    )
    model_replicate = int(args.model_seed if args.model_seed is not None else args.seed)
    plan_cell_config = None
    plan_cell_config_sha256 = ""
    if args.plan_cell_config is not None:
        plan_cell_config = load_training_cell_config(args.plan_cell_config)
        if args.campaign_seed is None:
            ap.error("--plan_cell_config requires --campaign_seed")
        if fixed_unknown is None:
            ap.error("--plan_cell_config requires explicit --unknown_classes")
        validate_training_cell_binding(
            plan_cell_config,
            {
                "family": "cifar",
                "split_id": args.split_id,
                "unknown_classes": sorted(fixed_unknown),
                "model_seed": model_replicate,
                "dirichlet_alpha": float(args.dirichlet_alpha),
                "campaign_seed": int(args.campaign_seed),
            },
        )
        overlapping = {
            "dataset": args.dataset,
            "n_known": args.n_known,
            "n_clients": args.n_clients,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "noise_type": args.noise_type,
            "noise_rate": args.noise_rate,
            "backbone": args.backbone,
            "norm": args.norm,
            "pretrained": bool(args.pretrained),
            "traffic_size": args.traffic_size,
        }
        validate_training_cell_binding(
            plan_cell_config,
            {
                key: value
                for key, value in overlapping.items()
                if key in plan_cell_config
            },
        )
        plan_cell_config_sha256 = semantic_hash(plan_cell_config)
        planned_experiment_id = training_cell_experiment_id(plan_cell_config)
        if (
            args.experiment_id is not None
            and args.experiment_id != planned_experiment_id
        ):
            ap.error("--experiment_id disagrees with --plan_cell_config")
        experiment_id = planned_experiment_id
    else:
        experiment_id = args.experiment_id or (
            f"{args.dataset}/{args.split_id}/model{model_replicate}/"
            f"d{args.dirichlet_alpha}/{args.noise_type}{args.noise_rate}"
        )
    seed_bundle = None
    if args.campaign_seed is not None:
        seed_bundle = SeedBundle.derive(
            args.campaign_seed,
            common_context={"campaign": "fedcore-oneshot", "dataset": args.dataset},
            namespace_contexts={
                "class_split": {"split_id": args.split_id},
                "partition": {"split_id": args.split_id},
                "fold": {"split_id": args.split_id},
                "model_init": {
                    "split_id": args.split_id,
                    "model_replicate": model_replicate,
                },
                "loader": {
                    "split_id": args.split_id,
                    "model_replicate": model_replicate,
                },
                "label_noise": {
                    "split_id": args.split_id,
                    "model_replicate": model_replicate,
                },
                "audit_draw": {"experiment_id": experiment_id, "draw_index": 0},
                "traffic_draw": {"experiment_id": experiment_id, "draw_index": 0},
                "solver": {"experiment_id": experiment_id},
                "stability": {"experiment_id": experiment_id, "redraw_index": 0},
            },
        )
        derived = {
            "split_seed": seed_bundle.class_split,
            "partition_seed": seed_bundle.partition,
            "fold_seed": seed_bundle.fold,
            "train_seed": seed_bundle.model_init,
            "loader_seed": seed_bundle.loader,
            "noise_seed": seed_bundle.label_noise,
            "solver_seed": seed_bundle.solver,
            "traffic_seed": seed_bundle.traffic_draw,
        }
    else:
        derived = {
            name: int(args.seed)
            for name in (
                "split_seed",
                "partition_seed",
                "fold_seed",
                "train_seed",
                "loader_seed",
                "noise_seed",
                "solver_seed",
                "traffic_seed",
            )
        }
    # Explicit namespace flags are supported for exact replay/migration and win
    # over either semantic derivation or the legacy alias.
    seeds = {
        name: int(
            getattr(args, name) if getattr(args, name) is not None else derived[name]
        )
        for name in derived
    }
    if args.traffic_size < 0:
        ap.error("--traffic_size must be non-negative")
    if (args.traffic_size > 0) != (args.traffic_client_probs is not None):
        ap.error(
            "--traffic_size > 0 and --traffic_client_probs must be supplied together"
        )
    traffic_client_probabilities = None
    if args.traffic_client_probs is not None:
        try:
            traffic_client_probabilities = np.asarray(
                [float(value) for value in args.traffic_client_probs.split(",")],
                dtype=float,
            )
        except ValueError:
            ap.error("--traffic_client_probs must be comma-separated numbers")

    cfg = FedOSRConfig(
        dataset=args.dataset,
        n_known=args.n_known,
        n_clients=args.n_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        alpha=args.alpha,
        delta=args.delta,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        seed=args.seed,
    )
    torch.manual_seed(seeds["train_seed"])
    np.random.seed(seeds["train_seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  cfg={cfg}  semantic_seeds={seeds}")

    train, test = _load_cifar(cfg.dataset, args.data_root)
    dataset_sha256 = _cifar_dataset_sha256(train)
    train_labels = np.array(train.targets)
    test_labels = np.array(test.targets)

    known_classes, unknown_classes, remap = open_set_split(
        train_labels, cfg.n_known, seeds["split_seed"], unknown_classes=fixed_unknown
    )
    print(
        f"known={known_classes.tolist()} unknown={unknown_classes.tolist()}"
        f"{' (FIXED split)' if fixed_unknown is not None else ''}"
    )

    # ---- TRAIN: known points only, Dirichlet-partitioned across clients -----
    known_mask = np.isin(train_labels, known_classes)
    known_train_idx = np.where(known_mask)[0]
    known_train_remapped = np.array(
        [remap[int(c)] for c in train_labels[known_train_idx]]
    )
    client_train_idx = dirichlet_partition(
        known_train_idx,
        known_train_remapped,
        cfg.n_clients,
        cfg.dirichlet_alpha,
        seeds["partition_seed"],
    )

    # global map dataset_index -> clean remapped label (for noise generation)
    remap_by_dsidx = {int(i): int(remap[int(train_labels[i])]) for i in known_train_idx}

    client_datasets: List[Dataset] = []
    for j, idx_j in enumerate(client_train_idx):
        override = make_label_noise(
            remap_by_dsidx,
            idx_j,
            cfg.noise_type,
            cfg.noise_rate,
            cfg.n_known,
            seeds["noise_seed"] + j,
        )
        client_datasets.append(
            _LabelRemapSubset(train, idx_j, remap, label_override=override)
        )
        print(f"client {j}: {len(idx_j)} train pts, {len(override)} corrupted")

    # ---- identity-only traffic, then trusted audit folds --------------------
    # Traffic is reserved before reading labels into the fold constructor. The
    # artifact exports only its stable identity and simulated deployment client.
    traffic = build_identity_only_traffic(
        np.arange(len(test_labels), dtype=np.int64),
        n_traffic=args.traffic_size,
        n_clients=cfg.n_clients,
        client_probabilities=(
            traffic_client_probabilities
            if traffic_client_probabilities is not None
            else np.full(cfg.n_clients, 1.0 / cfg.n_clients)
        ),
        seed=seeds["traffic_seed"],
    )
    traffic_mask = np.zeros(len(test_labels), dtype=bool)
    traffic_mask[traffic["idx"]] = True
    test_known_mask = np.isin(test_labels, known_classes)
    test_known_idx = np.where(test_known_mask & ~traffic_mask)[0]
    test_known_remapped = np.array([remap[int(c)] for c in test_labels[test_known_idx]])
    test_unknown_idx = np.where(np.isin(test_labels, unknown_classes) & ~traffic_mask)[
        0
    ]

    calib = build_calibration(
        test_known_idx,
        test_known_remapped,
        test_unknown_idx,
        cfg.n_clients,
        cfg.folds(),
        cfg.unknown_contamination,
        seeds["fold_seed"],
    )

    # ---- FedAvg training ----------------------------------------------------
    print(f"backbone={args.backbone} norm={args.norm} pretrained={args.pretrained}")
    training_config = {
        "dataset": cfg.dataset,
        "dataset_sha256": dataset_sha256,
        "dataset_hash_scope": "torchvision-extracted-batch-bytes-v1",
        "n_known": cfg.n_known,
        "n_clients": cfg.n_clients,
        "dirichlet_alpha": cfg.dirichlet_alpha,
        "noise_type": cfg.noise_type,
        "noise_rate": cfg.noise_rate,
        "rounds": cfg.rounds,
        "local_epochs": cfg.local_epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "backbone": args.backbone,
        "norm": args.norm,
        "pretrained": bool(args.pretrained),
        "unknown_classes": fixed_unknown,
        "traffic_size": int(args.traffic_size),
        "traffic_client_probabilities": (
            traffic_client_probabilities.tolist()
            if traffic_client_probabilities is not None
            else []
        ),
        "plan_cell_config_sha256": plan_cell_config_sha256,
        "seeds": seeds,
    }
    training_config_sha256 = semantic_hash(training_config)
    model = fedavg(
        lambda: make_model(
            cfg.n_known,
            backbone=args.backbone,
            norm=args.norm,
            pretrained=args.pretrained,
        ),
        client_datasets,
        cfg.rounds,
        cfg.local_epochs,
        cfg.lr,
        cfg.batch_size,
        device,
        loader_seed=seeds["loader_seed"],
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        checkpoint_metadata={
            "training_config_sha256": training_config_sha256,
            "seeds": seeds,
        },
    )

    # ---- per-fold logits -> scored views -> certify -------------------------
    views: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    raw_npz: Dict[str, np.ndarray] = {}
    for fold in ("prop", "cert", "test"):
        idx, y_open, client = _gather_fold(calib, fold)
        logits = export_logits(model, test, idx, device, cfg.batch_size)
        views[fold] = scored_views(logits, y_open, client, list(cfg.scores))
        raw_npz[f"{fold}_logits"] = logits
        raw_npz[f"{fold}_y_open"] = y_open
        raw_npz[f"{fold}_client"] = client
        raw_npz[f"{fold}_sample_idx"] = np.asarray(idx, dtype=np.int64)
        raw_npz[f"{fold}_sample_id"] = np.asarray(
            [f"{cfg.dataset}:test:{int(i)}" for i in idx], dtype="U32"
        )

    raw_npz["dataset"] = np.asarray(cfg.dataset)
    raw_npz["dataset_sha256"] = np.asarray(dataset_sha256)
    raw_npz["dataset_hash_scope"] = np.asarray("torchvision-extracted-batch-bytes-v1")
    raw_npz["experiment_id"] = np.asarray(experiment_id)
    raw_npz["split_id"] = np.asarray(args.split_id)
    raw_npz["model_seed"] = np.asarray(model_replicate, dtype=np.int64)
    raw_npz["seed_ledger_json"] = np.asarray(
        seed_bundle.to_json() if seed_bundle is not None else ""
    )
    raw_npz["training_config_json"] = np.asarray(canonical_json(training_config))
    raw_npz["training_config_sha256"] = np.asarray(training_config_sha256)
    if plan_cell_config is not None:
        raw_npz["plan_cell_config_json"] = np.asarray(canonical_json(plan_cell_config))
        raw_npz["plan_cell_config_sha256"] = np.asarray(plan_cell_config_sha256)
    raw_npz["seed_legacy"] = np.asarray(cfg.seed, dtype=np.int64)
    for name, value in seeds.items():
        raw_npz[name] = np.asarray(value, dtype=np.int64)
    raw_npz["known_classes"] = np.asarray(known_classes, dtype=np.int64)
    raw_npz["unknown_classes"] = np.asarray(unknown_classes, dtype=np.int64)
    if args.traffic_size > 0:
        raw_npz["traffic_sample_id"] = np.asarray(
            [f"{cfg.dataset}:test:{int(i)}" for i in traffic["idx"]], dtype="U32"
        )
        raw_npz["traffic_client"] = np.asarray(traffic["client"], dtype=np.int64)

    # save raw logits so downstream analyses (e.g. exp_necessity_real) can reuse
    add_split_fingerprint(raw_npz, seeds["fold_seed"])
    npz_path = os.path.splitext(args.out)[0] + "_logits.npz"
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    np.savez_compressed(npz_path, **raw_npz)
    print(
        f"saved {npz_path} (split_fp prop={raw_npz['prop_fp']} "
        f"cert={raw_npz['cert_fp']} test={raw_npz['test_fp']}, "
        f"numpy={raw_npz['numpy_version']})"
    )

    rows = certify_grid(
        views["prop"],
        views["cert"],
        views["test"],
        scores=cfg.scores,
        gammas=cfg.gammas,
        alpha=cfg.alpha,
        delta=cfg.delta,
        Lambdas=("simplex", "box"),
        n_clients=cfg.n_clients,
        dirichlet_alpha=cfg.dirichlet_alpha,
        box=cfg.box_radius,
        seed=seeds["solver_seed"],
    )

    print_metric_table(rows)

    # Legacy diagnostic grid. Gamma is proposal-selected, but choosing score/Lambda
    # below from certification coverage is an oracle comparison, not a primary claim.
    def best_gamma_rows(alpha: float):
        out = []
        for L in ("simplex", "box"):
            for s in cfg.scores:
                out.append(
                    certify_best_gamma(
                        views["prop"][s],
                        views["cert"][s],
                        views["test"][s],
                        score_name=s,
                        gammas=cfg.gammas,
                        alpha=alpha,
                        delta=cfg.delta,
                        n_clients=cfg.n_clients,
                        dirichlet_alpha=cfg.dirichlet_alpha,
                        Lambda=L,
                        box=cfg.box_radius,
                        seed=seeds["solver_seed"],
                        margin=args.proxy_margin,
                    )
                )
        return out

    bg = best_gamma_rows(cfg.alpha)
    bg_cert = [r for r in bg if r["certified"]]
    best = max(bg_cert, key=lambda r: r["cert_coverage_lcb"], default=None)
    print(f"\n[best-gamma] grid={cfg.gammas}")
    if best:
        print(
            f"DIAGNOSTIC ORACLE (certification-selected; not a primary claim): "
            f"CertifiedCoverage@alpha={cfg.alpha}: {best['cert_coverage_lcb']:.4f} "
            f"(score={best['score_name']}, gamma*={best['gamma_star']}, "
            f"Lambda={best['Lambda']}, cert_ucb={best['cert_risk_ucb']:.3f}, "
            f"test_risk={best['test_risk']:.3f})"
        )
    else:
        # honest diagnosis when nothing certifies even with the most conservative gamma
        tight = min(bg, key=lambda r: r["cert_risk_ucb"])
        thm2 = np.log(cfg.n_clients / cfg.delta) / (-np.log(1 - cfg.alpha))
        print(
            f"CertifiedCoverage@alpha={cfg.alpha}: 0 (best-gamma). "
            f"min cert_ucb={tight['cert_risk_ucb']:.3f} at gamma*={tight['gamma_star']} "
            f"(cert_n={tight['cert_n']}); Theorem-2 floor per client ~ {thm2:.0f}. "
            f"Lever is calibration size / fewer clients / backbone, not gamma."
        )

    if args.alpha_frontier:
        print("\n[alpha-frontier diagnostic oracle] (same logits, no retraining)")
        print(
            f"{'alpha':>7} {'cov_lcb':>9} {'gamma*':>7} {'score/L':>16} {'cert_ucb':>9}"
        )
        for a in (0.10, 0.15, 0.20, 0.25):
            cert_a = [r for r in best_gamma_rows(a) if r["certified"]]
            b = max(cert_a, key=lambda r: r["cert_coverage_lcb"], default=None)
            if b:
                print(
                    f"{a:>7.2f} {b['cert_coverage_lcb']:>9.4f} {b['gamma_star']:>7} "
                    f"{b['score_name']+'/'+b['Lambda']:>16} {b['cert_risk_ucb']:>9.3f}"
                )
            else:
                print(f"{a:>7.2f} {0.0:>9.4f} {'-':>7} {'(none)':>16} {'-':>9}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_csv(rows, args.out)
    save_csv(bg, os.path.splitext(args.out)[0] + "_bestgamma.csv")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
