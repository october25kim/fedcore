"""Real FedOSR base model #1: FOOGD SM3D semantic-shift score on a shared backbone.

Closes reviewer Risk 2 for FOOGD: Fed-CORE certifies the NATIVE FOOGD open-set score
(``sm = ||score_model(latents)||``), not just MSP/energy. Uses the IDENTICAL CIFAR-10
FedOSR split as the FedAvg baseline (same seed/params => identical audit folds), trains
a FedAvg backbone, exports penultimate FEATURES on the audit folds, federated-trains the
FOOGD SM3D score model on per-client ID features (see foogd_score.py, faithful to
FOOGD-main), and writes a scored npz consumed by aggregate_T8.py.

Split contract (matches run_cifar.py exactly): TRAIN = CIFAR-10 train split, n_known=6
known classes, Dirichlet-partitioned over n_clients; AUDIT (prop/cert/test) = clean CIFAR
test split + injected unknowns (the 4 held-out classes). TRAIN is disjoint from AUDIT, so
no audit label ever touches training or score selection.

Output: runs/foogd_<dataset>_d<dirichlet>_seed<seed>.npz with, per fold, keys
{<fold>_logits, <fold>_feat, <fold>_sm, <fold>_y_open, <fold>_client}. The native FOOGD
accept-score used downstream is ``-<fold>_sm`` (higher => more ID).

Run (inside the torch container, see scripts/docker_foogd.sh):
  python experiments/fedcore/run_foogd_cifar.py --dataset cifar10 --dirichlet_alpha 5 \
      --seed 0 --rounds 50 --score_rounds 30
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as T

from config import FedOSRConfig
from fed_train import fedavg, export_logits
from fedosr_split import build_calibration, dirichlet_partition, open_set_split
from foogd_score import sm_score, train_federated_score_model
from models import make_model
from run_cifar import _LabelRemapSubset, _gather_fold, _load_cifar


@torch.no_grad()
def export_features(model, base_dataset, indices, device, bs=256):
    """Penultimate features (model.features(x)) for dataset indices, shape (N, D)."""
    model.to(device).eval()
    loader = DataLoader(Subset(base_dataset, list(indices)), batch_size=bs, shuffle=False)
    out = []
    for xb, _ in loader:
        out.append(model.features(xb.to(device)).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, 0), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--n_known", type=int, default=6)
    ap.add_argument("--n_clients", type=int, default=5)
    ap.add_argument("--dirichlet_alpha", type=float, default=5.0)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm", choices=["bn", "gn"], default="gn")
    ap.add_argument("--score_rounds", type=int, default=40)
    ap.add_argument("--score_local_steps", type=int, default=80)
    ap.add_argument("--score_sigma", type=float, default=0.5,
                    help="DSM noise scale on STANDARDIZED features (unit variance => 0.5 = half a std; "
                         "principled, not tuned on test labels)")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate wiring")
    args = ap.parse_args()

    if args.smoke:
        args.rounds, args.score_rounds, args.score_local_steps = 2, 3, 10

    cfg = FedOSRConfig(dataset=args.dataset, n_known=args.n_known, n_clients=args.n_clients,
                       dirichlet_alpha=args.dirichlet_alpha, rounds=args.rounds,
                       local_epochs=args.local_epochs, seed=args.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} dataset={cfg.dataset} d={cfg.dirichlet_alpha} seed={cfg.seed} "
          f"norm={args.norm} rounds={cfg.rounds} score_rounds={args.score_rounds}")

    train, test = _load_cifar(cfg.dataset, args.data_root)
    train_labels = np.array(train.targets)
    test_labels = np.array(test.targets)

    known_classes, unknown_classes, remap = open_set_split(train_labels, cfg.n_known, cfg.seed)
    print(f"known={known_classes.tolist()} unknown={unknown_classes.tolist()}")

    # ---- TRAIN partition (identical recipe to run_cifar) --------------------
    known_train_idx = np.where(np.isin(train_labels, known_classes))[0]
    known_train_remapped = np.array([remap[int(c)] for c in train_labels[known_train_idx]])
    client_train_idx = dirichlet_partition(known_train_idx, known_train_remapped,
                                           cfg.n_clients, cfg.dirichlet_alpha, cfg.seed)
    client_datasets: List[Dataset] = [
        _LabelRemapSubset(train, idx_j, remap) for idx_j in client_train_idx]
    for j, idx_j in enumerate(client_train_idx):
        print(f"client {j}: {len(idx_j)} train pts")

    # ---- AUDIT folds from CLEAN test set (identical to baseline) -------------
    test_known_idx = np.where(np.isin(test_labels, known_classes))[0]
    test_known_remapped = np.array([remap[int(c)] for c in test_labels[test_known_idx]])
    test_unknown_idx = np.where(np.isin(test_labels, unknown_classes))[0]
    calib = build_calibration(test_known_idx, test_known_remapped, test_unknown_idx,
                              cfg.n_clients, cfg.folds(), cfg.unknown_contamination, cfg.seed)

    # ---- FedAvg backbone -----------------------------------------------------
    print("training FedAvg backbone (resnet18)...")
    model = fedavg(lambda: make_model(cfg.n_known, backbone="resnet18", norm=args.norm),
                   client_datasets, cfg.rounds, cfg.local_epochs, cfg.lr, cfg.batch_size, device)
    feat_dim = make_model(cfg.n_known, backbone="resnet18", norm=args.norm).classifier.in_features

    # ---- per-client TRAIN features for the federated score model ------------
    print("exporting per-client TRAIN features...")
    client_feats = [export_features(model, train, idx_j, device, cfg.batch_size)
                    for idx_j in client_train_idx]

    # ---- federated SM3D score model (FOOGD native) --------------------------
    print(f"federated DSM score-model training ({args.score_rounds} rounds)...")
    score_model, feat_mu, feat_sd = train_federated_score_model(
        client_feats, feat_dim, rounds=args.score_rounds, local_steps=args.score_local_steps,
        batch_size=256, lr=1e-3, sigma=args.score_sigma, device=device, seed=cfg.seed)

    # ---- export audit folds: logits, features, native sm score --------------
    raw: Dict[str, np.ndarray] = {}
    for fold in ("prop", "cert", "test"):
        idx, y_open, client = _gather_fold(calib, fold)
        logits = export_logits(model, test, idx, device, cfg.batch_size)
        feats = export_features(model, test, idx, device, cfg.batch_size)
        sm = sm_score(score_model, feats, feat_mu, feat_sd, device=device)  # higher => more OOD
        raw[f"{fold}_logits"] = logits
        raw[f"{fold}_feat"] = feats.astype(np.float32)
        raw[f"{fold}_sm"] = sm.astype(np.float32)
        raw[f"{fold}_y_open"] = y_open
        raw[f"{fold}_client"] = client

    # quick context: native-score AUROC (unknown vs known) on the test fold
    yo = raw["test_y_open"]; sm = raw["test_sm"]
    is_unk = yo < 0
    if is_unk.any() and (~is_unk).any():
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(is_unk.astype(int), sm)  # sm high => OOD
        print(f"[context] FOOGD sm-score AUROC (unknown detection) = {auroc:.3f}")

    out = args.out or f"runs/foogd_{cfg.dataset}_d{cfg.dirichlet_alpha:g}_seed{cfg.seed}.npz"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **raw)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
