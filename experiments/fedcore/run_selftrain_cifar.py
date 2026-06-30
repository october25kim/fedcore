"""Real CIFAR certified federated self-training (Proposition 4) -- torch, GPU.

Runs the certified vs naive vs no-self-training trajectories on CIFAR-10/100 under
non-IID Dirichlet partition + client-side TRAIN-label corruption, reusing
``fedosr_split``, ``models``, ``fed_train``, ``noise`` and the ``selftrain`` loop.

Smoke-size by default (few rounds/epochs) per the acceptance gate; scale up via
flags once it runs end-to-end.

Example::

    python experiments/fedcore/run_selftrain_cifar.py --dataset cifar10 \
        --T 4 --fedavg_rounds 5 --local_epochs 1 --dirichlet_alpha 0.1 \
        --noise_type symmetric --noise_rate 0.35
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import torch
import torchvision
import torchvision.transforms as T_tf

from fedosr_split import dirichlet_partition, open_set_split
from models import make_model
from noise import make_label_noise
from run_cifar import _NORM, _load_cifar
from selftrain import partition_selftrain, run_self_training

REC_KEYS = ["mode", "round", "cert_risk_ucb", "cert_coverage_lcb", "realized_contam",
            "n_pseudo", "test_acc", "test_coverage", "test_risk", "admitted",
            "infeasible_round"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--n_known", type=int, default=6)
    ap.add_argument("--n_clients", type=int, default=5)
    ap.add_argument("--dirichlet_alpha", type=float, default=0.1)
    ap.add_argument("--T", type=int, default=4, help="self-training rounds / audit folds")
    ap.add_argument("--fedavg_rounds", type=int, default=5)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--score", default="energy")
    ap.add_argument("--labeled_frac", type=float, default=0.5,
                    help="fraction of known-train used as LABELED; rest is unlabeled pool U")
    ap.add_argument("--noise_type", choices=["none", "symmetric", "asymmetric"], default="none")
    ap.add_argument("--noise_rate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", choices=["simplecnn", "resnet18"], default="simplecnn")
    ap.add_argument("--norm", choices=["bn", "gn"], default="bn")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default="runs/selftrain.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} dataset={args.dataset} T={args.T} d={args.dirichlet_alpha} "
          f"noise={args.noise_type}{args.noise_rate}")

    train, test = _load_cifar(args.dataset, args.data_root)
    train_labels = np.array(train.targets)
    test_labels = np.array(test.targets)
    known_classes, unknown_classes, remap = open_set_split(train_labels, args.n_known, args.seed)
    print(f"known={known_classes.tolist()} unknown={unknown_classes.tolist()}")

    rng = np.random.default_rng(args.seed)

    # ---- labeled vs unlabeled split of the known-train set -------------------
    known_train_idx = np.where(np.isin(train_labels, known_classes))[0]
    rng.shuffle(known_train_idx)
    n_lab = int(len(known_train_idx) * args.labeled_frac)
    labeled_idx = known_train_idx[:n_lab]
    unlab_known_idx = known_train_idx[n_lab:]
    unlab_unknown_idx = np.where(np.isin(train_labels, unknown_classes))[0]

    # labeled: Dirichlet partition + client-side corruption
    labeled_remapped = np.array([remap[int(c)] for c in train_labels[labeled_idx]])
    labeled_client_idx = dirichlet_partition(labeled_idx, labeled_remapped,
                                             args.n_clients, args.dirichlet_alpha, args.seed)
    remap_by_dsidx = {int(i): int(remap[int(train_labels[i])]) for i in labeled_idx}
    labeled_overrides = [
        make_label_noise(remap_by_dsidx, idx_j, args.noise_type, args.noise_rate,
                         args.n_known, args.seed + j)
        for j, idx_j in enumerate(labeled_client_idx)
    ]

    # unlabeled pool U (known leftovers + unknown classes), partitioned per client
    pool = np.concatenate([unlab_known_idx, unlab_unknown_idx])
    rng.shuffle(pool)
    unlabeled_client_idx = [np.asarray(c) for c in np.array_split(pool, args.n_clients)]
    for j in range(args.n_clients):
        print(f"client {j}: labeled={len(labeled_client_idx[j])} "
              f"corrupt={len(labeled_overrides[j])} unlabeled={len(unlabeled_client_idx[j])}")

    # ---- trusted calibration from the CLEAN test set: P + C^(1..T) + test ----
    test_known_idx = np.where(np.isin(test_labels, known_classes))[0]
    test_known_remapped = np.array([remap[int(c)] for c in test_labels[test_known_idx]])
    test_unknown_idx = np.where(np.isin(test_labels, unknown_classes))[0]
    parts = partition_selftrain(test_known_idx, test_known_remapped, test_unknown_idx,
                                args.n_clients, args.T, prop_frac=0.4, test_frac=0.3,
                                unknown_contamination=0.30, seed=args.seed)

    # ---- run the three trajectories -----------------------------------------
    all_records: List[Dict] = []
    for mode in ("certified", "naive", "none"):
        print(f"\n=== mode={mode} ===")
        recs = run_self_training(
            mode, make_model_fn=lambda: make_model(args.n_known, backbone=args.backbone, norm=args.norm),
            train_dataset=train, test_dataset=test, train_targets=train_labels,
            remap=remap, known_classes=known_classes,
            labeled_client_idx=labeled_client_idx, labeled_overrides=labeled_overrides,
            unlabeled_client_idx=unlabeled_client_idx, parts=parts,
            n_known=args.n_known, n_clients=args.n_clients, T=args.T,
            alpha=args.alpha, delta=args.delta,
            gammas=(0.2, 0.3, 0.5, 0.7, 1.0), score_name=args.score,
            fedavg_rounds=args.fedavg_rounds, local_epochs=args.local_epochs,
            lr=0.01, batch_size=64, device=device,
        )
        for r in recs:
            print(f"  round {r['round']}: acc={r['test_acc']:.4f} "
                  f"contam={r['realized_contam']} ucb={r['cert_risk_ucb']} "
                  f"n_pseudo={r['n_pseudo']} admit={r['admitted']} "
                  f"infeasible={r['infeasible_round']}")
        all_records.extend(recs)

    # ---- save + headline summary --------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REC_KEYS)
        w.writeheader()
        for r in all_records:
            w.writerow({k: r.get(k) for k in REC_KEYS})
    print(f"\nsaved {args.out}")

    def final_acc(mode):
        rs = [r for r in all_records if r["mode"] == mode]
        return rs[-1]["test_acc"] if rs else float("nan")

    print(f"final test acc: certified={final_acc('certified'):.4f} "
          f"naive={final_acc('naive'):.4f} none={final_acc('none'):.4f}")
    cert_contam = [r["realized_contam"] for r in all_records
                   if r["mode"] == "certified" and r["admitted"]]
    naive_contam = [r["realized_contam"] for r in all_records
                    if r["mode"] == "naive" and r["admitted"]]
    print(f"certified realized contamination per admitted round: "
          f"{[round(c, 3) for c in cert_contam if c == c]}")
    print(f"naive realized contamination per admitted round:     "
          f"{[round(c, 3) for c in naive_contam if c == c]}")


if __name__ == "__main__":
    main()
