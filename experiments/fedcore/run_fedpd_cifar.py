"""Real FedOSR base model #2: FedPD (PROSER) on our CIFAR-10 FedOSR split.

Drives FedPD's actual PROSER open-set training (ICCV 2023; third_party/FedPD) on our
split: WideResNet-28-10 backbone + a dummy/placeholder classifier head (clf2), trained
with PROSER manifold-mixup (traindummy: known CE + dummy detection + mixup->unknown),
FedAvg-aggregated each round. The native open-set score is FedPD's dummy-vs-known
confidence (valdummy): with softmax over [known_logits, max_dummy_logit],
``conf = softmax[:,-1] - max_k softmax[:,k]`` (higher => more unknown). Fed-CORE accept
score = -conf.

The PROSER model surgery (pre2block / latter2blockclf1 / latter2blockclf2 / dummypredict)
and the traindummy/valdummy logic are lifted verbatim from FedPD tools/proser_federated.py
(we bypass FedPD's mmcv-based utils and .mat loader; the model is imported from
third_party/FedPD mounted at /fedpd). Split is IDENTICAL to run_cifar.py (same seeds/params
=> same audit folds). If WideResNet-28-10 fails to train usefully in budget we report it
honestly; the score head is the genuine PROSER detector.

Output: runs/fedpd_<dataset>_d<dirichlet>_seed<seed>.npz with per-fold
{<fold>_logits, <fold>_sm, <fold>_y_open, <fold>_client} where _sm = PROSER conf (high=>OOD).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from config import FedOSRConfig
from fedosr_split import build_calibration, dirichlet_partition, open_set_split
from run_cifar import _LabelRemapSubset, _gather_fold, _load_cifar

# Load FedPD's WideResNet by file path: our experiments/fedcore/models.py already owns
# the top-level name `models` (imported via run_cifar), so a normal import would clash.
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("fedpd_wrn", "/fedpd/models/wide_resnet_embedding.py")
_wrn = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_wrn)
Wide_ResNet = _wrn.Wide_ResNet


# --- PROSER network surgery (verbatim from FedPD tools/proser_federated.py) ---
def pre2block(net, x):
    out = net.conv1(x); out = net.layer1(out); out = net.layer2(out)
    return out

def latter2blockclf1(net, x):
    out = net.layer3(x); out = F.relu(net.bn1(out)); out = F.avg_pool2d(out, 8)
    out = out.view(out.size(0), -1); return net.linear(out)

def latter2blockclf2(net, x):
    out = net.layer3(x); out = F.relu(net.bn1(out)); out = F.avg_pool2d(out, 8)
    out = out.view(out.size(0), -1); return net.clf2(out)

def dummypredict(net, x):
    out = net.conv1(x); out = net.layer1(out); out = net.layer2(out); out = net.layer3(out)
    out = F.relu(net.bn1(out)); out = F.avg_pool2d(out, 8)
    out = out.view(out.size(0), -1); return net.clf2(out)


def make_net(n_known, device):
    net = Wide_ResNet(28, 10, n_known)
    net.clf2 = nn.Linear(640, 1)        # dummy head (nStages[3]=64*10=640)
    return net.to(device)


def train_ce(loader, net, opt, device):
    """Plain closed-set CE pretrain on known classes (one forward/batch -> fast)."""
    ce = nn.CrossEntropyLoss(); net.train(); correct = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        opt.zero_grad()
        out = net(inputs)
        loss = ce(out, targets)
        loss.backward(); opt.step()
        correct += out.max(1)[1].eq(targets).sum().item(); total += len(targets)
    return correct / max(total, 1)


def traindummy(loader, net, opt, n_known, device, alpha=1.0, lamda1=1.0, lamda2=1.0):
    """PROSER manifold-mixup open-set training (FedPD traindummy)."""
    ce = nn.CrossEntropyLoss()
    net.train(); correct = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        opt.zero_grad()
        half = len(inputs) // 2
        if half < 1:
            continue
        beta = float(torch.distributions.beta.Beta(alpha, alpha).sample())
        pre, prelab = inputs[:half], targets[:half]
        later, laterlab = inputs[half:], targets[half:]
        idx = torch.randperm(pre.size(0), device=device)
        pre2 = pre2block(net, pre)
        mixed = beta * pre2 + (1 - beta) * pre2[idx]
        dummylogit = dummypredict(net, later)
        lateroutputs = net(later)
        latterhalfoutput = torch.cat((lateroutputs, dummylogit), 1)
        prehalfoutput = torch.cat((latter2blockclf1(net, mixed), latter2blockclf2(net, mixed)), 1)
        maxdummy, _ = torch.max(dummylogit.clone(), dim=1); maxdummy = maxdummy.view(-1, 1)
        dummyoutputs = torch.cat((lateroutputs.clone(), maxdummy), dim=1)
        for i in range(len(dummyoutputs)):
            dummyoutputs[i][laterlab[i]] = -1e9
        dummytargets = torch.ones_like(laterlab) * n_known
        loss1 = ce(prehalfoutput, (torch.ones_like(prelab) * n_known).long())
        loss2 = ce(latterhalfoutput, laterlab)
        loss3 = ce(dummyoutputs, dummytargets)
        loss = 0.01 * loss1 + lamda1 * loss2 + lamda2 * loss3
        loss.backward(); opt.step()
        pred = lateroutputs.max(1)[1]
        correct += pred.eq(laterlab).sum().item(); total += len(laterlab)
    return correct / max(total, 1)


def _avg(states, weights):
    w = [x / sum(weights) for x in weights]
    avg = copy.deepcopy(states[0])
    for k in avg.keys():
        if torch.is_floating_point(avg[k]):
            avg[k] = sum(states[i][k].float() * w[i] for i in range(len(states))).to(avg[k].dtype)
        else:
            avg[k] = states[0][k]
    return avg


@torch.no_grad()
def export(net, base, indices, n_known, device, bs=256):
    net.eval()
    loader = DataLoader(Subset(base, list(indices)), batch_size=bs, shuffle=False)
    logits_all, conf_all = [], []
    for xb, _ in loader:
        xb = xb.to(device)
        known = net(xb)                                   # (B, n_known)
        dummy = dummypredict(net, xb)                     # (B, 1)
        maxdummy = dummy.max(1, keepdim=True)[0]
        total = torch.cat((known, maxdummy), 1)           # (B, n_known+1)
        sm = F.softmax(total, dim=1)
        dummyconf = sm[:, -1]
        maxknown = sm[:, :n_known].max(1)[0]
        conf = (dummyconf - maxknown)                     # high => unknown
        logits_all.append(known.cpu().numpy()); conf_all.append(conf.cpu().numpy())
    return np.concatenate(logits_all, 0), np.concatenate(conf_all, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10"); ap.add_argument("--n_known", type=int, default=6)
    ap.add_argument("--n_clients", type=int, default=5); ap.add_argument("--dirichlet_alpha", type=float, default=5.0)
    ap.add_argument("--pretrain_rounds", type=int, default=50, help="closed-set CE FedAvg rounds (PROSER fine-tunes a converged net)")
    ap.add_argument("--rounds", type=int, default=15, help="PROSER traindummy fine-tune rounds")
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrain_lr", type=float, default=0.1); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--data_root", default="data"); ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.pretrain_rounds, args.rounds, args.local_epochs = 2, 2, 1

    cfg = FedOSRConfig(dataset=args.dataset, n_known=args.n_known, n_clients=args.n_clients,
                       dirichlet_alpha=args.dirichlet_alpha, rounds=args.rounds,
                       local_epochs=args.local_epochs, seed=args.seed)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FedPD-PROSER] device={device} d={cfg.dirichlet_alpha} seed={cfg.seed} "
          f"rounds={cfg.rounds} local_epochs={cfg.local_epochs} lr={args.lr}")

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

    glob = make_net(cfg.n_known, device)
    loaders = [DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=2)
               for ds in client_datasets]
    weights = [len(ds) for ds in client_datasets]

    def fed_round(local_fn, lr):
        states, accs = [], []
        for ld in loaders:
            local = make_net(cfg.n_known, device)
            local.load_state_dict(copy.deepcopy(glob.state_dict()))
            opt = torch.optim.SGD(local.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            acc = 0.0
            for _ in range(cfg.local_epochs):
                acc = local_fn(ld, local, opt)
            states.append({k: v.cpu() for k, v in local.state_dict().items()}); accs.append(acc)
        glob.load_state_dict(_avg(states, weights))
        return sum(a * w for a, w in zip(accs, weights)) / sum(weights)

    # ---- Phase 1: closed-set CE pretrain (PROSER fine-tunes a CONVERGED net) ----
    print(f"phase 1: closed-set CE pretrain ({args.pretrain_rounds} rounds, lr={args.pretrain_lr})")
    for r in range(args.pretrain_rounds):
        a = fed_round(lambda ld, net, opt: train_ce(ld, net, opt, device), args.pretrain_lr)
        if r % max(1, args.pretrain_rounds // 8) == 0 or r == args.pretrain_rounds - 1:
            print(f"  pretrain round {r}: known-acc={a:.3f}")

    # ---- Phase 2: PROSER dummy fine-tune (manifold mixup) -----------------------
    print(f"phase 2: PROSER fine-tune ({cfg.rounds} rounds, lr={args.lr})")
    for r in range(cfg.rounds):
        a = fed_round(lambda ld, net, opt: traindummy(ld, net, opt, cfg.n_known, device), args.lr)
        if r % max(1, cfg.rounds // 8) == 0 or r == cfg.rounds - 1:
            print(f"  proser round {r}: known-acc={a:.3f}")

    raw = {}
    for fold in ("prop", "cert", "test"):
        idx, y_open, client = _gather_fold(calib, fold)
        logits, conf = export(glob, test, idx, cfg.n_known, device, cfg.batch_size)
        raw[f"{fold}_logits"] = logits.astype(np.float32)
        raw[f"{fold}_sm"] = conf.astype(np.float32)        # high => OOD (PROSER conf)
        raw[f"{fold}_y_open"] = y_open; raw[f"{fold}_client"] = client

    yo = raw["test_y_open"]; sm = raw["test_sm"]; is_unk = yo < 0
    if is_unk.any() and (~is_unk).any():
        from sklearn.metrics import roc_auc_score
        print(f"[context] FedPD PROSER conf AUROC (unknown detection) = {roc_auc_score(is_unk.astype(int), sm):.3f}")

    out = args.out or f"runs/fedpd_{cfg.dataset}_d{cfg.dirichlet_alpha:g}_seed{cfg.seed}.npz"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **raw)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
