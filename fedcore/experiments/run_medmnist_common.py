"""Plain-FedAvg MedMNIST open-set runner emitting the confirmatory COMMON per-obs schema.

The medical counterpart of run_cifar_common: a large-reservoir federated open-set
MEDICAL benchmark (MedMNIST, 28x28) that -- unlike Fed-ISIC -- satisfies the
Theorem-2 feasibility floor with room to spare and can certify NON-TRIVIALLY under
the distribution-free client full-simplex certificate.

Reuses the exact fedcore split core (open_set_split / dirichlet_partition /
build_calibration) and the run_cifar label-remap subset + common-schema export.
A separate module (does NOT import or edit run_cifar_common) so it is isolated from
any concurrently-running CIFAR campaign. Native accept score = MSP.

Datasets: pathmnist (9 classes, RGB). Held-out classes form the open-set unknown;
known = the rest. Backbones (from scratch, 28x28-native): wrn28_10, resnext29_8x64d.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset

import fedcore.experiments.confirmatory_400r_common_schema as CS
from fedcore.config import FedOSRConfig
from fedcore.data.fedosr_split import build_calibration, dirichlet_partition, open_set_split
from fedcore.experiments.run_cifar import _LabelRemapSubset, _gather_fold
from fedcore.models.fed_train import export_logits, fedavg, make_fedprox_local_train_fn
from fedcore.models.models import make_model

BACKBONES = ("wrn28_10", "resnext29_8x64d")
DATASETS = {
    # dataset: (npz_relpath, n_classes, default_unknown_classes)
    "pathmnist": ("data/medmnist/pathmnist.npz", 9, (6, 7, 8)),       # RGB 28x28, colon pathology
    "tissuemnist": ("data/medmnist/tissuemnist.npz", 8, (5, 6, 7)),   # GRAYSCALE 28x28, kidney tissue
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_deterministic(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def msp_score(known_logits):
    z = np.asarray(known_logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).max(axis=1).astype(np.float32)


class _MedDS(Dataset):
    def __init__(self, X, y, mean, std):
        self.X = X
        self.targets = [int(v) for v in y]
        self._m = torch.tensor(mean).view(-1, 1, 1)
        self._s = torch.tensor(std).view(-1, 1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = torch.from_numpy(self.X[i]).float() / 255.0
        if img.ndim == 2:
            img = img.unsqueeze(-1)
        img = img.permute(2, 0, 1)
        if img.shape[0] == 1:            # grayscale (e.g. TissueMNIST) -> 3 channels
            img = img.repeat(3, 1, 1)
        return (img - self._m) / self._s, self.targets[i]


def _load_medmnist(dataset, data_root):
    rel, n_cls, _ = DATASETS[dataset]
    d = np.load(os.path.join(data_root, os.path.basename(rel)) if os.path.isdir(data_root)
                else rel)
    Xtr, ytr = d["train_images"], d["train_labels"].ravel().astype(int)
    Xte, yte = d["test_images"], d["test_labels"].ravel().astype(int)
    ch = 3 if Xtr.ndim == 4 else 1
    flat = Xtr.reshape(-1, ch) if ch == 3 else Xtr.reshape(-1, 1)
    mean = (flat.mean(0) / 255.0).astype(np.float32)
    std = (flat.std(0) / 255.0 + 1e-6).astype(np.float32)
    return _MedDS(Xtr, ytr, mean, std), _MedDS(Xte, yte, mean, std), ytr, yte, n_cls


def build_job(dataset, data_root, n_known, n_clients, dirichlet_alpha, seed, unknown_classes):
    cfg = FedOSRConfig(dataset="cifar10", n_known=n_known, n_clients=n_clients,
                       dirichlet_alpha=dirichlet_alpha, seed=seed)
    train, test, ytr, yte, n_cls = _load_medmnist(dataset, data_root)
    known, unknown, remap = open_set_split(ytr, n_known, seed, unknown_classes=list(unknown_classes))
    ktr = np.where(np.isin(ytr, known))[0]
    ktr_rm = np.array([remap[int(c)] for c in ytr[ktr]])
    parts = dirichlet_partition(ktr, ktr_rm, n_clients, dirichlet_alpha, seed)
    client_ds = [_LabelRemapSubset(train, idx, remap) for idx in parts]
    tk = np.where(np.isin(yte, known))[0]
    tk_rm = np.array([remap[int(c)] for c in yte[tk]])
    tu = np.where(np.isin(yte, unknown))[0]
    calib = build_calibration(tk, tk_rm, tu, n_clients, cfg.folds(), cfg.unknown_contamination, seed)
    return dict(cfg=cfg, test=test, test_labels=yte, client_datasets=client_ds,
                calib=calib, known_classes=known, unknown_classes=unknown, remap=remap)


def export_common(model, job, device, dataset, split_id, out_path, backbone, batch_size=256,
                  family=None):
    n_known = int(job["cfg"].n_known)
    test, test_labels = job["test"], job["test_labels"]
    role_map = {"prop": "proposal", "cert": "certification", "test": "test"}
    folds = {}
    for fold, role in role_map.items():
        idx, y_open, client = _gather_fold(job["calib"], fold)
        idx = np.asarray(idx, dtype=np.int64)
        known = np.asarray(export_logits(model, test, idx, device, batch_size), dtype=np.float32)
        src = np.asarray([f"{dataset}:test:{int(i)}" for i in idx], dtype=str)
        folds[role] = CS.assemble_fold(
            immutable_source_id=src, client_id=np.asarray(client, dtype=np.int64),
            fold_role=role, true_global_class=test_labels[idx].astype(np.int64),
            true_known_class_index_or_neg1=np.asarray(y_open, dtype=np.int64),
            known_logits=known, native_score=msp_score(known))
    if family is None:
        family = f"medmnist_{dataset}_{backbone}_plain_fedavg"
    payload = CS.pack_common_npz(folds, native_score_name="msp", n_known=n_known,
                                 family=family)
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
    unknown = args.unknown_classes if args.unknown_classes else list(DATASETS[args.dataset][2])
    job = build_job(args.dataset, args.data_root, args.n_known, args.n_clients,
                    args.dirichlet_alpha, args.seed, unknown)
    make = lambda: make_model(args.n_known, backbone=args.backbone, norm=args.norm)
    # FedProx methodology arm: mu>0 adds the (mu/2)||w-w_t||^2 proximal term; mu==0
    # (default) is plain FedAvg, byte-identical to the existing MedMNIST plain arms.
    mu = float(getattr(args, "fedprox_mu", 0.0) or 0.0)
    local_train_fn = make_fedprox_local_train_fn(mu) if mu > 0 else None
    family = (f"medmnist_{args.dataset}_{args.backbone}_plain_fedavg" if mu == 0
              else f"medmnist_{args.dataset}_{args.backbone}_fedprox_mu{mu:g}")
    meta = {"training_config_sha256": args.config_sha, "experiment_id": args.experiment_id}
    t0 = time.time()
    model = fedavg(make, job["client_datasets"], args.rounds, args.local_epochs,
                   args.lr, args.batch_size, device, loader_seed=args.seed,
                   checkpoint_path=args.checkpoint, resume=args.resume,
                   checkpoint_every=1, checkpoint_metadata=meta,
                   local_train_fn=local_train_fn)
    train_s = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    export_common(model, job, device, args.dataset, args.split_id, args.out, args.backbone,
                  args.batch_size, family=family)
    marker = {
        "status": "completed", "experiment_id": args.experiment_id,
        "family": family, "backbone": args.backbone,
        "dataset": args.dataset, "split_id": args.split_id, "n_known": args.n_known,
        "n_clients": args.n_clients, "dirichlet_alpha": args.dirichlet_alpha,
        "rounds": args.rounds, "local_epochs": args.local_epochs, "batch_size": args.batch_size,
        "lr": args.lr, "norm": args.norm, "device": device,
        "unknown_classes": list(unknown),
        "train_seconds": round(train_s, 3), "peak_vram_gb": round(peak, 4),
        "checkpoint_bytes": os.path.getsize(args.checkpoint),
        "logits_npz_bytes": os.path.getsize(args.out),
        "checksums": {os.path.basename(args.out): _sha256(args.out),
                      os.path.basename(args.checkpoint): _sha256(args.checkpoint)},
        "proser": False,
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
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    r.add_argument("--backbone", required=True, choices=BACKBONES)
    r.add_argument("--norm", default="bn", choices=("bn", "gn"))
    r.add_argument("--split-id", dest="split_id", required=True)
    r.add_argument("--n-known", dest="n_known", type=int, required=True)
    r.add_argument("--n-clients", dest="n_clients", type=int, default=5)
    r.add_argument("--dirichlet-alpha", dest="dirichlet_alpha", type=float, default=1.0)
    r.add_argument("--rounds", type=int, default=25)
    r.add_argument("--local-epochs", dest="local_epochs", type=int, default=2)
    r.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    r.add_argument("--lr", type=float, default=0.01)
    r.add_argument("--fedprox-mu", dest="fedprox_mu", type=float, default=0.0,
                   help="FedProx proximal coefficient mu; 0.0 (default) = plain FedAvg, "
                        "byte-identical to the existing MedMNIST plain arms.")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--data-root", dest="data_root", default="data/medmnist")
    r.add_argument("--unknown-classes", dest="unknown_classes", default=None,
                   type=lambda s: [int(x) for x in s.split(",")] if s else None)
    r.add_argument("--experiment-id", dest="experiment_id", required=True)
    r.add_argument("--config-sha", dest="config_sha", default="medmnist")
    r.add_argument("--out", required=True)
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--marker", required=True)
    r.add_argument("--resume", action="store_true")
    r.set_defaults(func=run)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
