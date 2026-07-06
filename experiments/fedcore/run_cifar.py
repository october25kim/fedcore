"""Real CIFAR FedOSR run: FedAvg over non-IID known classes, then certify.

Run inside the project's Docker (torch + torchvision required); not executed in
the certificate sandbox. Mirrors ``run_smoke.py`` exactly from the logit stage
onward, so the certification path is identical for synthetic and real logits.

Example:
    python run_cifar.py --dataset cifar10 --n_known 6 --dirichlet_alpha 0.1 \
        --rounds 50 --local_epochs 2 --alpha 0.10 --delta 0.10
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as T

from certify import certify_grid
from config import FedOSRConfig
from fed_train import export_logits, fedavg
from fedosr_split import build_calibration, dirichlet_partition, open_set_split
from models import make_model
from noise import make_label_noise
from scores import scored_views

CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


class _LabelRemapSubset(Dataset):
    """Subset of a base dataset that returns (image, remapped_known_label)."""

    def __init__(
        self,
        base: Dataset,
        indices: np.ndarray,
        remap: dict[int, int],
        label_override: dict[int, int] | None = None,
    ):
        self.base = base
        self.indices = list(indices)
        self.remap = remap
        self.label_override = label_override or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x, y = self.base[idx]
        if idx in self.label_override:        # corrupted TRAIN label
            return x, self.label_override[idx]
        return x, self.remap[int(y)]


def _load(dataset: str):
    mean, std = CIFAR_STATS[dataset]
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    train = cls(root="./data", train=True, download=True, transform=tf)
    test = cls(root="./data", train=False, download=True, transform=tf)
    return train, test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--n_known", type=int, default=6)
    p.add_argument("--n_clients", type=int, default=5)
    p.add_argument("--dirichlet_alpha", type=float, default=0.1)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--noise_type", default="none", choices=["none", "symmetric", "asymmetric"])
    p.add_argument("--noise_rate", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="cifar_results.csv")
    args = p.parse_args()

    cfg = FedOSRConfig(
        dataset=args.dataset, n_known=args.n_known, n_clients=args.n_clients,
        dirichlet_alpha=args.dirichlet_alpha, rounds=args.rounds,
        local_epochs=args.local_epochs, alpha=args.alpha, delta=args.delta,
        noise_type=args.noise_type, noise_rate=args.noise_rate, seed=args.seed,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    train, test = _load(cfg.dataset)
    y_train = np.array(train.targets)
    y_test = np.array(test.targets)

    split = open_set_split(y_train, cfg.n_known, cfg.seed)
    known_set = set(split.known_classes.tolist())

    # --- federated known-class training data (non-IID) ---
    known_train_idx = np.where(np.isin(y_train, list(known_set)))[0]
    known_train_y = np.array([split.remap[int(c)] for c in y_train[known_train_idx]])
    client_idx = dirichlet_partition(
        known_train_idx, known_train_y, cfg.n_clients, cfg.dirichlet_alpha, cfg.seed
    )
    # client-side TRAIN-label corruption (calibration/test stay clean)
    override = make_label_noise(
        known_train_y, known_train_idx, cfg.noise_type, cfg.noise_rate, cfg.n_known, cfg.seed
    )
    print(f"[noise] type={cfg.noise_type} rate={cfg.noise_rate} "
          f"flipped={len(override)}/{len(known_train_idx)} train labels")
    client_datasets = [
        _LabelRemapSubset(train, idx, split.remap, label_override=override)
        for idx in client_idx
    ]

    # --- trusted calibration pool from the TEST set (known + unknown) ---
    known_test_idx = np.where(np.isin(y_test, list(known_set)))[0]
    known_test_y = np.array([split.remap[int(c)] for c in y_test[known_test_idx]])
    unknown_test_idx = np.where(~np.isin(y_test, list(known_set)))[0]
    per_client_folds = build_calibration(
        known_test_idx, known_test_y, unknown_test_idx,
        cfg.n_clients, cfg.folds(), cfg.unknown_contamination, cfg.seed,
    )

    # --- train ---
    model = fedavg(
        lambda: make_model(cfg.n_known), client_datasets,
        rounds=cfg.rounds, local_epochs=cfg.local_epochs, lr=cfg.lr,
        batch_size=cfg.batch_size, device=device,
    )

    # --- export logits per fold (concatenated across clients) and certify ---
    def fold_views(fold: str) -> dict:
        idx_all, y_all, client_all = [], [], []
        for j, folds in enumerate(per_client_folds):
            f = folds[fold]
            idx_all.append(f["idx"]); y_all.append(f["y_open"])
            client_all.append(np.full(len(f["idx"]), j))
        idx_all = np.concatenate(idx_all)
        y_open = np.concatenate(y_all)
        client = np.concatenate(client_all)
        logits = export_logits(model, test, idx_all, device, cfg.batch_size)
        return scored_views(logits, y_open, client, cfg.scores)

    prop, cert, test_v = fold_views("prop"), fold_views("cert"), fold_views("test")

    base = np.full(cfg.n_clients, 1.0 / cfg.n_clients)
    box = (np.clip(base - cfg.box_radius, 0, 1), np.clip(base + cfg.box_radius, 0, 1))

    all_rows = []
    for Lambda in ("simplex", "box"):
        rows = certify_grid(
            prop=prop, cert=cert, test=test_v,
            score_names=cfg.scores, gammas=cfg.gammas,
            alpha=cfg.alpha, delta=cfg.delta,
            n_clients=cfg.n_clients, dirichlet_alpha=cfg.dirichlet_alpha,
            Lambda=Lambda, box=box if Lambda == "box" else None,
        )
        for r in rows:
            r["Lambda"] = Lambda
        all_rows.extend(rows)
        n_cert = sum(r["certified"] for r in rows)
        print(f"[Lambda={Lambda}] {n_cert}/{len(rows)} certified at alpha={cfg.alpha}")
        for r in rows:
            print(r)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
