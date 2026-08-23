"""Plain-FedAvg CIFAR open-set runner emitting the confirmatory COMMON per-obs schema.

A license-neutral, PROSER-free sibling of the confirmatory CIFAR runner: it reuses
the fedcore split core (``open_set_split`` / ``dirichlet_partition`` /
``build_calibration``) and the ``run_cifar`` data helpers, trains an arbitrary
``make_model`` backbone with PLAIN size-weighted FedAvg (default SGD+momentum
client update -- no PROSER, no LPD, no GDCA/OT), and exports the SAME common
per-obs schema (``confirmatory_400r_common_schema.pack_common_npz``) so the new
CIFAR backbone arms certify through the identical post-hoc path as the
WRN-28-10 PROSER-FedAvg baseline.

Backbones (from-scratch, native 32x32): ``resnext29_8x64d`` (ResNeXt-29 8x64d),
``wrn28_10`` (WideResNet-28-10). The native accept score is MSP (max softmax over
the known-class logits; higher => more likely a known class), matching the
Office-Home normalization convention. Every element that feeds the certificate
(splits, calibration folds, source IDs, logits) is byte-compatible with the 400R
common schema; only the backbone + the (absent) PROSER dummy differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

import fedcore.experiments.confirmatory_400r_common_schema as CS
from fedcore.config import FedOSRConfig
from fedcore.data.fedosr_split import (
    build_calibration,
    dirichlet_partition,
    open_set_split,
)
from fedcore.experiments.run_cifar import _LabelRemapSubset, _gather_fold, _load_cifar
from fedcore.models.fed_train import export_logits, fedavg, make_fedprox_local_train_fn
from fedcore.models.models import make_model

BACKBONES = ("resnext29_8x64d", "wrn28_10")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_deterministic(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def msp_score(known_logits: np.ndarray) -> np.ndarray:
    """MSP accept score: max softmax prob over known classes (higher => ID/accept)."""
    z = np.asarray(known_logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    sm = e / e.sum(axis=1, keepdims=True)
    return sm.max(axis=1).astype(np.float32)


def build_job(dataset, data_root, n_known, n_clients, dirichlet_alpha, seed, unknown_classes):
    """Reuse the license-neutral fedcore split core (identical to the confirmatory arm)."""
    cfg = FedOSRConfig(dataset=dataset, n_known=n_known, n_clients=n_clients,
                       dirichlet_alpha=dirichlet_alpha, seed=seed)
    train, test = _load_cifar(dataset, data_root)
    train_labels = np.array(train.targets)
    test_labels = np.array(test.targets)
    fixed_unknown = ([int(c) for c in unknown_classes] if unknown_classes else None)
    known_classes, unknown, remap = open_set_split(
        train_labels, cfg.n_known, cfg.seed, unknown_classes=fixed_unknown)
    known_train_idx = np.where(np.isin(train_labels, known_classes))[0]
    known_train_remapped = np.array([remap[int(c)] for c in train_labels[known_train_idx]])
    client_train_idx = dirichlet_partition(
        known_train_idx, known_train_remapped, cfg.n_clients, cfg.dirichlet_alpha, cfg.seed)
    client_datasets = [_LabelRemapSubset(train, idx_j, remap) for idx_j in client_train_idx]
    test_known_idx = np.where(np.isin(test_labels, known_classes))[0]
    test_known_remapped = np.array([remap[int(c)] for c in test_labels[test_known_idx]])
    test_unknown_idx = np.where(np.isin(test_labels, unknown))[0]
    calib = build_calibration(test_known_idx, test_known_remapped, test_unknown_idx,
                              cfg.n_clients, cfg.folds(), cfg.unknown_contamination, cfg.seed)
    return dict(cfg=cfg, test=test, test_labels=test_labels, client_datasets=client_datasets,
                calib=calib, known_classes=known_classes, unknown_classes=unknown, remap=remap)


def export_common(model, job, device, dataset, split_id, out_path, backbone, batch_size=256,
                  family=None):
    n_known = int(job["cfg"].n_known)
    test = job["test"]
    test_labels = job["test_labels"]
    role_map = {"prop": "proposal", "cert": "certification", "test": "test"}
    folds = {}
    for fold, role in role_map.items():
        idx, y_open, client = _gather_fold(job["calib"], fold)
        idx = np.asarray(idx, dtype=np.int64)
        known = export_logits(model, test, idx, device, batch_size)
        known = np.asarray(known, dtype=np.float32)
        native = msp_score(known)
        src = np.asarray([f"{dataset}:test:{int(i)}" for i in idx], dtype=str)
        folds[role] = CS.assemble_fold(
            immutable_source_id=src,
            client_id=np.asarray(client, dtype=np.int64),
            fold_role=role,
            true_global_class=test_labels[idx].astype(np.int64),
            true_known_class_index_or_neg1=np.asarray(y_open, dtype=np.int64),
            known_logits=known,
            native_score=native,
        )
    if family is None:
        family = f"cifar_{backbone}_plain_fedavg"
    payload = CS.pack_common_npz(
        folds, native_score_name="msp", n_known=n_known, family=family)
    payload["dataset"] = np.asarray(str(dataset))
    payload["split_id"] = np.asarray(str(split_id))
    payload["backbone"] = np.asarray(str(backbone))
    payload["known_classes"] = np.asarray(job["known_classes"], dtype=np.int64)
    payload["unknown_classes"] = np.asarray(job["unknown_classes"], dtype=np.int64)
    tmp = f"{out_path}.tmp.{os.getpid()}.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, out_path)
    os.chmod(out_path, 0o644)
    return out_path


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_deterministic(args.seed)
    job = build_job(args.dataset, args.data_root, args.n_known, args.n_clients,
                    args.dirichlet_alpha, args.seed, args.unknown_classes)
    make = lambda: make_model(args.n_known, backbone=args.backbone, norm=args.norm)
    # FedProx methodology arm: mu>0 adds the (mu/2)||w-w_t||^2 proximal term to each
    # client update; mu==0 (default) is plain FedAvg, byte-identical to Arms B/C.
    mu = float(getattr(args, "fedprox_mu", 0.0) or 0.0)
    local_train_fn = make_fedprox_local_train_fn(mu) if mu > 0 else None
    family = (f"cifar_{args.backbone}_plain_fedavg" if mu == 0
              else f"cifar_{args.backbone}_fedprox_mu{mu:g}")
    meta = {"training_config_sha256": args.config_sha, "experiment_id": args.experiment_id}
    t0 = time.time()
    # PLAIN FedAvg (mu==0): no local_train_fn (default SGD+momentum CE), constant lr.
    model = fedavg(make, job["client_datasets"], args.rounds, args.local_epochs,
                   args.lr, args.batch_size, device, loader_seed=args.seed,
                   checkpoint_path=args.checkpoint, resume=args.resume,
                   checkpoint_every=1, checkpoint_metadata=meta,
                   local_train_fn=local_train_fn)
    train_s = time.time() - t0
    peak_vram = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    t1 = time.time()
    export_common(model, job, device, args.dataset, args.split_id, args.out,
                  args.backbone, args.batch_size, family=family)
    export_s = time.time() - t1
    marker = {
        "status": "completed", "experiment_id": args.experiment_id,
        "family": family, "backbone": args.backbone,
        "dataset": args.dataset, "split_id": args.split_id, "n_known": args.n_known,
        "n_clients": args.n_clients, "dirichlet_alpha": args.dirichlet_alpha,
        "rounds": args.rounds, "local_epochs": args.local_epochs, "batch_size": args.batch_size,
        "lr": args.lr, "norm": args.norm, "device": device,
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "train_seconds": round(train_s, 3), "export_seconds": round(export_s, 3),
        "peak_vram_gb": round(peak_vram, 4),
        "checkpoint_bytes": os.path.getsize(args.checkpoint),
        "logits_npz_bytes": os.path.getsize(args.out),
        "checksums": {os.path.basename(args.out): _sha256(args.out),
                      os.path.basename(args.checkpoint): _sha256(args.checkpoint)},
        "proser": False, "lpd": False, "gdca_ot": False,
        "aggregation": "weighted_fedavg" if mu == 0 else "weighted_fedavg_fedprox",
    }
    if mu > 0:
        marker["fedprox_mu"] = mu
    tmp = f"{args.marker}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(marker, fh, indent=2, default=str)
    os.replace(tmp, args.marker)
    print(json.dumps(marker, indent=2, default=str))
    return marker


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train one CIFAR plain-FedAvg cell + export common schema")
    r.add_argument("--dataset", required=True, choices=("cifar10", "cifar100"))
    r.add_argument("--backbone", required=True, choices=BACKBONES)
    r.add_argument("--norm", default="bn", choices=("bn", "gn"))
    r.add_argument("--split-id", dest="split_id", required=True)
    r.add_argument("--n-known", dest="n_known", type=int, required=True)
    r.add_argument("--n-clients", dest="n_clients", type=int, default=5)
    r.add_argument("--dirichlet-alpha", dest="dirichlet_alpha", type=float, default=0.5)
    r.add_argument("--rounds", type=int, default=50)
    r.add_argument("--local-epochs", dest="local_epochs", type=int, default=2)
    r.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    r.add_argument("--lr", type=float, default=0.01)
    r.add_argument("--fedprox-mu", dest="fedprox_mu", type=float, default=0.0,
                   help="FedProx proximal coefficient mu; 0.0 (default) = plain FedAvg, "
                        "byte-identical to Arms B/C.")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--data-root", dest="data_root", default="/workspace/data")
    r.add_argument("--unknown-classes", dest="unknown_classes", default=None,
                   type=lambda s: [int(x) for x in s.split(",")] if s else None)
    r.add_argument("--experiment-id", dest="experiment_id", required=True)
    r.add_argument("--config-sha", dest="config_sha", default="cifar-common")
    r.add_argument("--out", required=True)
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--marker", required=True)
    r.add_argument("--resume", action="store_true")
    r.set_defaults(func=run)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 0)
