"""Real CIFAR-10/100 FedOSR run (torch + torchvision, GPU).

End-to-end: open-set split -> Dirichlet non-IID client partition -> client-side
TRAIN-label corruption -> FedAvg training -> trusted calibration from the CLEAN
test set -> the IDENTICAL certification path as ``run_smoke.py`` (scored_views ->
certify_grid for ``Lambda in {simplex, box}``).

Example::

    python experiments/fedcore/run_cifar.py --dataset cifar10 --n_known 6 \
        --n_clients 5 --dirichlet_alpha 0.1 --rounds 50 --local_epochs 2 \
        --alpha 0.10 --delta 0.10 --noise_type symmetric --noise_rate 0.35
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T

from certify import certify_best_gamma, certify_grid
from config import FedOSRConfig
from fed_train import export_logits, fedavg
from fedosr_split import build_calibration, dirichlet_partition, open_set_split
from models import make_model
from noise import make_label_noise
from run_smoke import print_metric_table, save_csv
from scores import scored_views


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
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    train = cls(root=root, train=True, download=True, transform=tf)
    test = cls(root=root, train=False, download=True, transform=tf)
    return train, test


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
    ap.add_argument("--noise_type", choices=["none", "symmetric", "asymmetric"], default="none")
    ap.add_argument("--noise_rate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default="runs/cifar_results.csv")
    ap.add_argument("--alpha_frontier", action="store_true",
                    help="also report CertifiedCoverage@alpha for alpha in {.10,.15,.20,.25}")
    ap.add_argument("--proxy_margin", type=float, default=0.01,
                    help="proposal-proxy safety margin for best-gamma (frontier monotonicity)")
    ap.add_argument("--backbone", choices=["simplecnn", "resnet18"], default="simplecnn")
    ap.add_argument("--norm", choices=["bn", "gn"], default="bn",
                    help="resnet18 normalization: bn (BatchNorm) or gn (GroupNorm, FL-appropriate)")
    ap.add_argument("--pretrained", action="store_true",
                    help="resnet18: load torchvision ImageNet weights")
    args = ap.parse_args()

    cfg = FedOSRConfig(
        dataset=args.dataset, n_known=args.n_known, n_clients=args.n_clients,
        dirichlet_alpha=args.dirichlet_alpha, rounds=args.rounds,
        local_epochs=args.local_epochs, alpha=args.alpha, delta=args.delta,
        noise_type=args.noise_type, noise_rate=args.noise_rate, seed=args.seed,
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  cfg={cfg}")

    train, test = _load_cifar(cfg.dataset, args.data_root)
    train_labels = np.array(train.targets)
    test_labels = np.array(test.targets)

    known_classes, unknown_classes, remap = open_set_split(
        train_labels, cfg.n_known, cfg.seed
    )
    print(f"known={known_classes.tolist()} unknown={unknown_classes.tolist()}")

    # ---- TRAIN: known points only, Dirichlet-partitioned across clients -----
    known_mask = np.isin(train_labels, known_classes)
    known_train_idx = np.where(known_mask)[0]
    known_train_remapped = np.array([remap[int(c)] for c in train_labels[known_train_idx]])
    client_train_idx = dirichlet_partition(
        known_train_idx, known_train_remapped, cfg.n_clients,
        cfg.dirichlet_alpha, cfg.seed,
    )

    # global map dataset_index -> clean remapped label (for noise generation)
    remap_by_dsidx = {int(i): int(remap[int(train_labels[i])]) for i in known_train_idx}

    client_datasets: List[Dataset] = []
    for j, idx_j in enumerate(client_train_idx):
        override = make_label_noise(
            remap_by_dsidx, idx_j, cfg.noise_type, cfg.noise_rate,
            cfg.n_known, cfg.seed + j,
        )
        client_datasets.append(
            _LabelRemapSubset(train, idx_j, remap, label_override=override)
        )
        print(f"client {j}: {len(idx_j)} train pts, {len(override)} corrupted")

    # ---- trusted calibration from the CLEAN test set ------------------------
    test_known_mask = np.isin(test_labels, known_classes)
    test_known_idx = np.where(test_known_mask)[0]
    test_known_remapped = np.array([remap[int(c)] for c in test_labels[test_known_idx]])
    test_unknown_idx = np.where(np.isin(test_labels, unknown_classes))[0]

    calib = build_calibration(
        test_known_idx, test_known_remapped, test_unknown_idx,
        cfg.n_clients, cfg.folds(), cfg.unknown_contamination, cfg.seed,
    )

    # ---- FedAvg training ----------------------------------------------------
    print(f"backbone={args.backbone} norm={args.norm} pretrained={args.pretrained}")
    model = fedavg(
        lambda: make_model(cfg.n_known, backbone=args.backbone, norm=args.norm,
                           pretrained=args.pretrained),
        client_datasets, cfg.rounds,
        cfg.local_epochs, cfg.lr, cfg.batch_size, device,
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

    # save raw logits so downstream analyses (e.g. exp_necessity_real) can reuse
    npz_path = os.path.splitext(args.out)[0] + "_logits.npz"
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    np.savez_compressed(npz_path, **raw_npz)
    print(f"saved {npz_path}")

    rows = certify_grid(
        views["prop"], views["cert"], views["test"],
        scores=cfg.scores, gammas=cfg.gammas, alpha=cfg.alpha, delta=cfg.delta,
        Lambdas=("simplex", "box"), n_clients=cfg.n_clients,
        dirichlet_alpha=cfg.dirichlet_alpha, box=cfg.box_radius, seed=cfg.seed,
    )

    print_metric_table(rows)

    # PRIMARY headline: validity-preserving certified-coverage-maximizing selector
    # (gamma chosen on the proposal fold; single certification at full delta).
    def best_gamma_rows(alpha: float):
        out = []
        for L in ("simplex", "box"):
            for s in cfg.scores:
                out.append(certify_best_gamma(
                    views["prop"][s], views["cert"][s], views["test"][s],
                    score_name=s, gammas=cfg.gammas, alpha=alpha, delta=cfg.delta,
                    n_clients=cfg.n_clients, dirichlet_alpha=cfg.dirichlet_alpha,
                    Lambda=L, box=cfg.box_radius, seed=cfg.seed,
                    margin=args.proxy_margin))
        return out

    bg = best_gamma_rows(cfg.alpha)
    bg_cert = [r for r in bg if r["certified"]]
    best = max(bg_cert, key=lambda r: r["cert_coverage_lcb"], default=None)
    print(f"\n[best-gamma] grid={cfg.gammas}")
    if best:
        print(f"CertifiedCoverage@alpha={cfg.alpha} headline: {best['cert_coverage_lcb']:.4f} "
              f"(score={best['score_name']}, gamma*={best['gamma_star']}, "
              f"Lambda={best['Lambda']}, cert_ucb={best['cert_risk_ucb']:.3f}, "
              f"test_risk={best['test_risk']:.3f})")
    else:
        # honest diagnosis when nothing certifies even with the most conservative gamma
        tight = min(bg, key=lambda r: r["cert_risk_ucb"])
        thm2 = np.log(cfg.n_clients / cfg.delta) / (-np.log(1 - cfg.alpha))
        print(f"CertifiedCoverage@alpha={cfg.alpha}: 0 (best-gamma). "
              f"min cert_ucb={tight['cert_risk_ucb']:.3f} at gamma*={tight['gamma_star']} "
              f"(cert_n={tight['cert_n']}); Theorem-2 floor per client ~ {thm2:.0f}. "
              f"Lever is calibration size / fewer clients / backbone, not gamma.")

    if args.alpha_frontier:
        print("\n[alpha-frontier] (same logits, no retraining)")
        print(f"{'alpha':>7} {'cov_lcb':>9} {'gamma*':>7} {'score/L':>16} {'cert_ucb':>9}")
        for a in (0.10, 0.15, 0.20, 0.25):
            cert_a = [r for r in best_gamma_rows(a) if r["certified"]]
            b = max(cert_a, key=lambda r: r["cert_coverage_lcb"], default=None)
            if b:
                print(f"{a:>7.2f} {b['cert_coverage_lcb']:>9.4f} {b['gamma_star']:>7} "
                      f"{b['score_name']+'/'+b['Lambda']:>16} {b['cert_risk_ucb']:>9.3f}")
            else:
                print(f"{a:>7.2f} {0.0:>9.4f} {'-':>7} {'(none)':>16} {'-':>9}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_csv(rows, args.out)
    save_csv(bg, os.path.splitext(args.out)[0] + "_bestgamma.csv")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
