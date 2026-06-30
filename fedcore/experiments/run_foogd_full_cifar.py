"""FULL FOOGD (SAG) base model: drives FOOGD's REAL ODGClient training on our split.

Unlike run_foogd_cifar.py (representative: FedAvg backbone + post-hoc DSM score head),
this runs FOOGD's actual training code from third_party/FOOGD-main verbatim:
  - backbone = FOOGD WideResNet-40-2 (128-dim intermediate_forward);
  - score model = Energy(MLPScore) (128-dim);
  - per-client local update = ODGClient.train() == CE + lambda1*KSD-Stein on backbone
    latents (SAG generalization) + anneal-DSM score matching + lambda2*MMD with Langevin
    samples (SAG diversity);
  - FedAvg of BOTH backbone and score model each round (FOOGD ODGServer.fit logic).

The repo ships lambda1=lambda2=0 (SAG OFF -> reduces to FedAvg+score-matching, i.e. the
representative run). We set them >0 (CLI, default 0.1) to engage the real SAG terms, so
this run earns base_model_kind=full. Native score is FOOGD's sm detector exactly:
``sm = ||score_model(intermediate_forward(x))||`` (higher => more OOD); Fed-CORE accept
score = -sm.

Split is IDENTICAL to run_cifar.py / run_foogd_cifar.py (same seed/params => same audit
folds): TRAIN = CIFAR train knowns (Dirichlet); AUDIT = clean CIFAR test + injected
unknowns. Requires FOOGD-main mounted at /foogd (see scripts/docker_foogd_full.sh).

Output: runs/foogdfull_<dataset>_d<dirichlet>_seed<seed>.npz with per-fold
{<fold>_logits, <fold>_sm, <fold>_y_open, <fold>_client}.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fedcore.config import FedOSRConfig
from fedcore.data.fedosr_split import build_calibration, dirichlet_partition, open_set_split
from fedcore.experiments.run_cifar import _LabelRemapSubset, _gather_fold, _load_cifar

# FOOGD repo (mounted at /foogd) provides the real training code.
sys.path.insert(0, "/foogd")
from src.models.wideresnet import WideResNet           # noqa: E402
from src.models.score import Energy, MLPScore          # noqa: E402
from src.algorithms.FOOGD.client_foogd import ODGClient  # noqa: E402


def _model_average(states, weights):
    w = [x / sum(weights) for x in weights]
    avg = copy.deepcopy(states[0])
    for k in avg.keys():
        avg[k] = sum(states[i][k] * w[i] for i in range(len(states)))
    return avg


def _client_args(cid, dataset, backbone, score_model, n_known, cfg, device, lam1, lam2):
    return {
        "cid": cid, "device": device, "epochs": cfg.local_epochs,
        "backbone": backbone, "learning_rate": cfg.lr, "momentum": 0.9, "weight_decay": 5e-4,
        "batch_size": cfg.batch_size, "num_workers": 2, "pin_memory": False,
        "train_id_dataset": dataset,
        "score_model": score_model, "score_learning_rate": 0.01,
        "num_classes": n_known, "lambda1": lam1, "lambda2": lam2,
        # ODG_* (faithful FOOGD CIFAR defaults from config.py)
        "ODG_noise_type": "gaussian", "ODG_loss_types": "anneal_dsm", "ODG_sampler": "ld",
        "ODG_sample_steps": 10, "ODG_sample_eps": 0.01, "ODG_n_slices": 0,
        "ODG_mmd_kernel_num": 2, "ODG_sigma_begin": 0.01, "ODG_sigma_end": 1.0,
        "ODG_anneal_power": 2,
    }


@torch.no_grad()
def _export(backbone, score_model, base, indices, device, bs=256):
    backbone.to(device).eval(); score_model.to(device).eval()
    loader = DataLoader(Subset(base, list(indices)), batch_size=bs, shuffle=False)
    logits, sm = [], []
    for xb, _ in loader:
        xb = xb.to(device)
        lat = backbone.intermediate_forward(xb)
        logits.append(backbone(xb).cpu().numpy())
        sm.append(score_model(lat).norm(dim=-1).cpu().numpy())
    return (np.concatenate(logits, 0) if logits else np.zeros((0, 0)),
            np.concatenate(sm, 0) if sm else np.zeros((0,)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--n_known", type=int, default=6)
    ap.add_argument("--n_clients", type=int, default=5)
    ap.add_argument("--dirichlet_alpha", type=float, default=5.0)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda1", type=float, default=0.1, help="KSD-Stein SAG weight (>0 engages SAG)")
    ap.add_argument("--lambda2", type=float, default=0.1, help="MMD SAG weight (>0 engages SAG)")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.rounds, args.local_epochs = 2, 1

    cfg = FedOSRConfig(dataset=args.dataset, n_known=args.n_known, n_clients=args.n_clients,
                       dirichlet_alpha=args.dirichlet_alpha, rounds=args.rounds,
                       local_epochs=args.local_epochs, seed=args.seed)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FOOGD-full] device={device} d={cfg.dirichlet_alpha} seed={cfg.seed} "
          f"rounds={cfg.rounds} lambda1={args.lambda1} lambda2={args.lambda2}")

    train, test = _load_cifar(cfg.dataset, args.data_root)
    train_labels = np.array(train.targets); test_labels = np.array(test.targets)
    known_classes, unknown_classes, remap = open_set_split(train_labels, cfg.n_known, cfg.seed)
    print(f"known={known_classes.tolist()} unknown={unknown_classes.tolist()}")

    known_train_idx = np.where(np.isin(train_labels, known_classes))[0]
    known_train_remapped = np.array([remap[int(c)] for c in train_labels[known_train_idx]])
    client_train_idx = dirichlet_partition(known_train_idx, known_train_remapped,
                                           cfg.n_clients, cfg.dirichlet_alpha, cfg.seed)
    client_datasets = [_LabelRemapSubset(train, idx_j, remap) for idx_j in client_train_idx]

    test_known_idx = np.where(np.isin(test_labels, known_classes))[0]
    test_known_remapped = np.array([remap[int(c)] for c in test_labels[test_known_idx]])
    test_unknown_idx = np.where(np.isin(test_labels, unknown_classes))[0]
    calib = build_calibration(test_known_idx, test_known_remapped, test_unknown_idx,
                              cfg.n_clients, cfg.folds(), cfg.unknown_contamination, cfg.seed)

    # ---- build FOOGD clients (real ODGClient) -------------------------------
    global_backbone = WideResNet(depth=40, num_classes=cfg.n_known, widen_factor=2, dropRate=0.3)
    global_score = Energy(net=MLPScore())
    clients = []
    for cid, ds in enumerate(client_datasets):
        ca = _client_args(cid, ds, copy.deepcopy(global_backbone), Energy(net=MLPScore()),
                          cfg.n_known, cfg, device, args.lambda1, args.lambda2)
        clients.append(ODGClient(ca))
    # init all clients to the same global weights; force drop_last so FOOGD's
    # latents = split(batch, B//2) always yields exactly two even halves (its
    # train() unpacks two; an odd tail batch -> 3 chunks -> ValueError).
    for c in clients:
        c.backbone.load_state_dict(global_backbone.state_dict())
        c.score_model.load_state_dict(global_score.state_dict())
        c.train_id_dataloader = DataLoader(
            c.train_id_dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=2, pin_memory=False, drop_last=True)

    # ---- federated rounds (ODGServer.fit logic) -----------------------------
    for r in range(cfg.rounds):
        bstates, sstates, weights, accs = [], [], [], []
        for c in clients:
            rep = c.train()
            bstates.append(rep["backbone"]); sstates.append(rep["score_model"])
            accs.append(rep["acc"]); weights.append(len(c.train_id_dataloader))
        gb = _model_average(bstates, weights); gs = _model_average(sstates, weights)
        for c in clients:
            c.backbone.load_state_dict(gb); c.score_model.load_state_dict(gs)
        global_backbone.load_state_dict(gb); global_score.load_state_dict(gs)
        if r % max(1, cfg.rounds // 10) == 0 or r == cfg.rounds - 1:
            print(f"  round {r}: mean client acc={sum(a*w for a,w in zip(accs,weights))/sum(weights):.3f}")

    # ---- export native sm score on audit folds ------------------------------
    raw = {}
    for fold in ("prop", "cert", "test"):
        idx, y_open, client = _gather_fold(calib, fold)
        logits, sm = _export(global_backbone, global_score, test, idx, device, cfg.batch_size)
        raw[f"{fold}_logits"] = logits.astype(np.float32)
        raw[f"{fold}_sm"] = sm.astype(np.float32)
        raw[f"{fold}_y_open"] = y_open
        raw[f"{fold}_client"] = client

    yo = raw["test_y_open"]; sm = raw["test_sm"]; is_unk = yo < 0
    if is_unk.any() and (~is_unk).any():
        from sklearn.metrics import roc_auc_score
        print(f"[context] FOOGD-full sm AUROC (unknown detection) = {roc_auc_score(is_unk.astype(int), sm):.3f}")

    out = args.out or f"runs/foogdfull_{cfg.dataset}_d{cfg.dirichlet_alpha:g}_seed{cfg.seed}.npz"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **raw)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
