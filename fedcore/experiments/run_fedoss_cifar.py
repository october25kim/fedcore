"""Real FedOSR base model #3: FedOSS (Zhu et al., TMI 2024) on our CIFAR-10 FedOSR split.

Drives FedOSS's actual open-set training (third_party/FedOSS, commit 2dd9afab) on OUR
split: the FedOSR ResNet-18 with a (K+1)-way head + a per-client auxiliary head, trained
with the two FedOSS modules lifted VERBATIM from the paper's code --
  * DUSS (Discrete Unknown Sample Synthesis): inter-client-inconsistency boundary-sample
    recognition (peer auxiliary heads) + iterative feature-space adversarial push
    (attack.i_DUS) to synthesize discrete virtual unknowns; and
  * FOSS (Federated Open Space Sampling): per-client class-conditional Gaussians of the
    synthesized unknowns aggregated across clients (communication.compute_global_statistic)
    into per-class MultivariateNormals, then low-density resampled as extra virtual unknowns.
The (K+1)-th logit is the synthetic unknown class. The native open-set score is
``sm = softmax(outputs)[:, K]`` (probability mass on the unknown class; higher => unknown).
Fed-CORE accept-score = -sm (handled downstream by scores.py `provided`), same as FedPD.

Split is IDENTICAL to run_cifar.py / run_fedpd_cifar.py (same seeds/params => same audit
folds; calibration folds stay clean). The FedOSS model class is imported unmodified from the
mounted repo at /fedoss; only the two small mechanism fns (i_DUS, compute_global_statistic)
are lifted here for a self-contained, committable adapter. Recipe: ONE permitted intervention
-- federated closed-set CE pretrain (as FedPD needed) then FedOSS DUSS+FOSS finetune. If the
detector does not train usefully in budget we report the AUROC honestly; this IS the genuine
FedOSS detector, never a representative head.

Output: runs/fedoss_<dataset>_d<dirichlet>_seed<seed>.npz with per-fold
{<fold>_logits, <fold>_sm, <fold>_y_open, <fold>_client}; _sm = FedOSS unknown-prob (high=>OOD).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util as _ilu
import os
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from fedcore.config import FedOSRConfig
from fedcore.data.fedosr_split import build_calibration, dirichlet_partition, open_set_split
from fedcore.experiments.run_cifar import (
    _LabelRemapSubset, _gather_fold, _load_cifar, add_split_fingerprint,
)

# FedOSS model, imported unmodified from the mounted repo (/fedoss). Its module-level code is
# import-safe under modern torch (acsconv is only touched in its __main__).
_spec = _ilu.spec_from_file_location("fedoss_finetune", "/fedoss/models/ResNet_FedOSR_Finetune.py")
_ft = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_ft)
fedoss_resnet18 = _ft.resnet18


# ---- DUSS: iterative feature-space adversarial push (attack/attack.py i_DUS, verbatim) ----
def i_DUS(model, feats, targets, eps, num_steps):
    x_adv = feats.clone().detach().data
    model.eval()
    ce = F.cross_entropy
    for _ in range(num_steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        h_adv = model.discrete_forward(x_adv)["outputs"]
        cost = -ce(h_adv, targets)
        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()
        x_adv = x_adv - eps * x_adv.grad.data
    model.train()
    return x_adv.data, targets.data


# ---- FOSS: aggregate per-client class-conditional Gaussians (communication.compute_global_statistic, verbatim) ----
def compute_global_statistic(known_class, mean_clients, cov_clients, number_clients):
    D = mean_clients.shape[-1]
    number_total = number_clients.sum(0, keepdim=True)
    mean_weights = number_clients / number_total.float()
    mean_clients_weighted = mean_clients * mean_weights.unsqueeze(2).expand([-1, -1, D])
    g_mean = mean_clients_weighted.sum(0)
    if (number_total > 1).all():
        cw1 = (number_clients - 1) / (number_total - 1).float()
        cw2 = (number_clients) / (number_total - 1).float()
        cw3 = number_total / (number_total - 1).float()
    else:
        cw1 = (number_clients) / (number_total + 1e-9).float()
        cw2 = (number_clients) / (number_total + 1e-9).float()
        cw3 = number_total / (number_total + 1e-9).float()
    ct1 = (cov_clients * cw1.unsqueeze(2).unsqueeze(3).expand([-1, -1, D, D])).sum(0)
    ct2 = torch.einsum('abcd, abde->abce', mean_clients.unsqueeze(3), mean_clients.unsqueeze(2))
    ct2 = (ct2 * cw2.unsqueeze(2).unsqueeze(3).expand([-1, -1, D, D])).sum(0)
    ct3 = torch.einsum('abc, acd->abd', g_mean.unsqueeze(2), g_mean.unsqueeze(1))
    ct3 = ct3 * cw3.permute(1, 0).unsqueeze(2).expand([-1, D, D])
    g_cov = ct1 + ct2 - ct3
    eye = torch.eye(g_cov.shape[1]).expand(g_cov.shape[0], g_cov.shape[1], g_cov.shape[1])
    g_cov = g_cov + 0.0001 * eye
    dis = []
    for c in range(known_class):
        if number_total[0][c] > 10:
            try:
                dis.append(torch.distributions.multivariate_normal.MultivariateNormal(
                    g_mean[c], covariance_matrix=g_cov[c]))
            except Exception:
                dis.append(None)
        else:
            dis.append(None)
    return dis


def make_net(n_known, device):
    return fedoss_resnet18(pretrained=False, num_classes=n_known).to(device)


def _avg_non_aux(server, clients, weights):
    """FedAvg every param EXCEPT the auxiliary head (kept per-client for inter-client
    discrepancy), copying the aggregate back into each client. Mirrors communication_Finetune."""
    w = [x / sum(weights) for x in weights]
    with torch.no_grad():
        sd = server.state_dict()
        for k in sd.keys():
            if 'auxiliary' in k:
                continue
            if 'num_batches_tracked' in k:
                sd[k].data.copy_(clients[0].state_dict()[k])
            else:
                temp = torch.zeros_like(sd[k])
                for i in range(len(clients)):
                    temp += w[i] * clients[i].state_dict()[k].to(temp.dtype)
                sd[k].data.copy_(temp)
                for i in range(len(clients)):
                    clients[i].state_dict()[k].data.copy_(sd[k])


def pretrain_local(loader, net, opt, device):
    """FedOSS pretrain step: CE(outputs, y) + CE(aux_out, y) on known classes."""
    ce = nn.CrossEntropyLoss(); net.train(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.long().to(device)
        opt.zero_grad()
        outs = net(x)
        loss = ce(outs["outputs"], y) + ce(outs["aux_out"], y)
        loss.backward(); opt.step()
        correct += outs["outputs"][:, :net.main_cls.out_features - 1].max(1)[1].eq(y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def finetune_local(loader, net, peers, opt, n_known, device, attack_eps, attack_steps,
                   unknown_weight, p_lower, p_upper, sample_from, collect, unknown_dis):
    """One FedOSS DUSS+FOSS finetune pass (ported from Finetune_library.train)."""
    ce = nn.CrossEntropyLoss(); net.train()
    for p in peers:
        p.eval()
    n_peer = len(peers)
    unknown_dict = [None] * n_known
    for x, y in loader:
        x, y = x.to(device), y.long().to(device)
        outs = net(x)
        outputs, aux_outputs = outs["outputs"], outs["aux_out"]
        boundary_feats, discrete_feats = outs["boundary_feats"], outs["discrete_feats"]
        loss = ce(outputs, y) + ce(aux_outputs, y)

        # DUSS: inter-client-inconsistency boundary recognition
        _, aux_pred = aux_outputs.max(1)
        agree = torch.eq(aux_pred, y).float()
        for pn in peers:
            with torch.no_grad():
                ap = pn.aux_forward(boundary_feats.clone().detach())["aux_out"].max(1)[1]
                agree += torch.eq(ap, y).float()
        frac = agree / (n_peer + 1)
        is_boundary = (frac > p_lower) & (frac < p_upper)

        if is_boundary.sum() > 1:
            df = discrete_feats[is_boundary]; dt = y[is_boundary]
            inputs_unknown, targets_unknown = i_DUS(net, df, dt, attack_eps, attack_steps)
            if inputs_unknown is not None:
                ou = net.discrete_forward(inputs_unknown.clone().detach())["outputs"]
                prob_u = torch.softmax(ou, dim=-1)
                PDs = prob_u[:, -1] - prob_u[:, :-1].max(-1)[0]
                gt = torch.ones(ou.shape[0]).long().to(device) * n_known
                for i in range(len(ou)):
                    ou[i][targets_unknown[i]] = -1e9
                loss = loss + ce(ou, gt) * unknown_weight
                if collect:
                    tun = targets_unknown.cpu().numpy()
                    for i in range(len(targets_unknown)):
                        if PDs[i] > -1:
                            k = int(tun[i]); s = inputs_unknown[i].clone().detach().view(1, -1)
                            unknown_dict[k] = s if unknown_dict[k] is None else torch.cat((unknown_dict[k], s), 0)
                # FOSS: resample low-density virtual unknowns from the global Gaussians
                if unknown_dis is not None:
                    sc = torch.randint(0, n_known, (sample_from,))
                    cnt = {i: 0 for i in range(n_known)}
                    for it in sc:
                        cnt[it.item()] += 1
                    ood, oodt = None, None
                    for c in range(n_known):
                        if cnt[c] > 0 and unknown_dis[c] is not None:
                            g = unknown_dis[c].rsample((100,))
                            lp = unknown_dis[c].log_prob(g)
                            _, idx = torch.topk(-lp, cnt[c])
                            g = g[idx].to(device).reshape(cnt[c], 256, 2, 2)
                            gt2 = (torch.ones(cnt[c]) * c).long().to(device)
                            ood = g if ood is None else torch.cat((ood, g), 0)
                            oodt = gt2 if oodt is None else torch.cat((oodt, gt2), 0)
                    if ood is not None and ood.shape[0] > 1:
                        ou2 = net.discrete_forward(ood.clone().detach())["outputs"]
                        gt3 = torch.ones(ou2.shape[0]).long().to(device) * n_known
                        for i in range(len(ou2)):
                            ou2[i][oodt[i]] = -1e9
                        loss = loss + ce(ou2, gt3) * unknown_weight

        opt.zero_grad(); loss.backward(); opt.step()

    for p in peers:
        p.train()

    # per-class Gaussian stats of this client's synthesized unknowns (for FOSS)
    mean_d = [None] * n_known; cov_d = [None] * n_known; num_d = torch.zeros(n_known)
    if collect:
        D = None
        for c in range(n_known):
            if unknown_dict[c] is not None:
                D = unknown_dict[c].shape[1]; break
        if D is None:
            return mean_d, cov_d, num_d, False
        for c in range(n_known):
            if unknown_dict[c] is not None:
                mean_d[c] = unknown_dict[c].mean(0).cpu()
                X = unknown_dict[c] - unknown_dict[c].mean(0)
                cov_d[c] = (torch.mm(X.t(), X) / len(X)).cpu()
                num_d[c] = len(X)
            else:
                mean_d[c] = torch.zeros(D); cov_d[c] = torch.zeros(D, D)
        return torch.stack(mean_d, 0), torch.stack(cov_d, 0), num_d, True
    return mean_d, cov_d, num_d, False


@torch.no_grad()
def export(net, base, indices, n_known, device, bs=256):
    net.eval()
    loader = DataLoader(Subset(base, list(indices)), batch_size=bs, shuffle=False)
    logits_all, sm_all = [], []
    for xb, _ in loader:
        xb = xb.to(device)
        outputs = net(xb)["outputs"]                 # (B, K+1)
        sm = F.softmax(outputs, dim=1)
        known_logits = outputs[:, :n_known]          # (B, K)
        unk = sm[:, n_known]                          # unknown-class prob, high => OOD
        logits_all.append(known_logits.cpu().numpy()); sm_all.append(unk.cpu().numpy())
    return np.concatenate(logits_all, 0), np.concatenate(sm_all, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10"); ap.add_argument("--n_known", type=int, default=6)
    ap.add_argument("--n_clients", type=int, default=5); ap.add_argument("--dirichlet_alpha", type=float, default=5.0)
    ap.add_argument("--pretrain_rounds", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=30, help="FedOSS DUSS+FOSS finetune rounds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrain_lr", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4, help="FedOSS Adam lr (their default)")
    ap.add_argument("--eps", type=float, default=0.1); ap.add_argument("--num_steps", type=int, default=1)
    ap.add_argument("--unknown_weight", type=float, default=1.0)
    ap.add_argument("--sample_from", type=int, default=8)
    ap.add_argument("--p_upper", type=float, default=1.0, help="boundary upper frac (Hyperkvasir default)")
    ap.add_argument("--data_root", default="data"); ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.pretrain_rounds, args.rounds = 2, 3

    cfg = FedOSRConfig(dataset=args.dataset, n_known=args.n_known, n_clients=args.n_clients,
                       dirichlet_alpha=args.dirichlet_alpha, rounds=args.rounds, seed=args.seed)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # FOSS collection schedule: last ~5 evenly-spaced finetune rounds (mirrors [5,10,15,20,25]/30)
    start_epoch = sorted(set(max(1, round(args.rounds * f)) - 1 for f in (0.17, 0.33, 0.5, 0.67, 0.83)))
    print(f"[FedOSS] device={device} d={cfg.dirichlet_alpha} seed={cfg.seed} "
          f"pretrain={args.pretrain_rounds} finetune={cfg.rounds} lr={args.lr} eps={args.eps} "
          f"num_steps={args.num_steps} start_epoch={start_epoch} p_upper={args.p_upper}")

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

    loaders = [DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=2)
               for ds in client_datasets]
    weights = [len(ds) for ds in client_datasets]
    server = make_net(cfg.n_known, device)
    clients = [make_net(cfg.n_known, device) for _ in range(cfg.n_clients)]
    for c in clients:
        c.load_state_dict(copy.deepcopy(server.state_dict()))

    # ---- Phase 1: federated closed-set CE pretrain (the single permitted recipe intervention) ----
    print(f"phase 1: CE pretrain ({args.pretrain_rounds} rounds, lr={args.pretrain_lr})")
    for r in range(args.pretrain_rounds):
        accs = []
        for c, ld in zip(clients, loaders):
            c.load_state_dict(copy.deepcopy(server.state_dict()))
            opt = torch.optim.SGD(c.parameters(), lr=args.pretrain_lr, momentum=0.9, weight_decay=5e-4)
            accs.append(pretrain_local(ld, c, opt, device))
        _avg_non_aux(server, clients, weights)
        if r % max(1, args.pretrain_rounds // 8) == 0 or r == args.pretrain_rounds - 1:
            print(f"  pretrain round {r}: known-acc={sum(a*w for a,w in zip(accs,weights))/sum(weights):.3f}")

    # ---- Phase 2: FedOSS DUSS+FOSS finetune ----
    print(f"phase 2: FedOSS finetune ({cfg.rounds} rounds, lr={args.lr})")
    for c in clients:
        c.load_state_dict(copy.deepcopy(server.state_dict()))
    unknown_dis = None
    for r in range(cfg.rounds):
        collect = r in start_epoch
        means, covs, nums = [], [], []
        for ci in range(cfg.n_clients):
            peers = [clients[j] for j in range(cfg.n_clients) if j != ci]
            opt = torch.optim.Adam(clients[ci].parameters(), lr=args.lr, betas=(0.9, 0.99))
            md, cd, nd, ok = finetune_local(loaders[ci], clients[ci], peers, opt, cfg.n_known,
                                            device, args.eps, args.num_steps, args.unknown_weight,
                                            0.0, args.p_upper, args.sample_from, collect, unknown_dis)
            if collect and ok:
                means.append(md); covs.append(cd); nums.append(nd)
        if len(means) > 0:
            unknown_dis = compute_global_statistic(cfg.n_known, torch.stack(means, 0),
                                                   torch.stack(covs, 0), torch.stack(nums, 0))
            print(f"  finetune round {r}: FOSS stats collected ({len(means)} clients contributed)")
        _avg_non_aux(server, clients, weights)
        for c in clients:
            c.load_state_dict(copy.deepcopy(server.state_dict()))
        if r % max(1, cfg.rounds // 8) == 0 or r == cfg.rounds - 1:
            print(f"  finetune round {r} done (unknown_dis={'set' if unknown_dis else 'none'})")

    raw = {}
    for fold in ("prop", "cert", "test"):
        idx, y_open, client = _gather_fold(calib, fold)
        logits, sm = export(server, test, idx, cfg.n_known, device, cfg.batch_size)
        raw[f"{fold}_logits"] = logits.astype(np.float32)
        raw[f"{fold}_sm"] = sm.astype(np.float32)      # high => OOD (FedOSS unknown prob)
        raw[f"{fold}_y_open"] = y_open; raw[f"{fold}_client"] = client

    yo = raw["test_y_open"]; sm = raw["test_sm"]; is_unk = yo < 0
    if is_unk.any() and (~is_unk).any():
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(is_unk.astype(int), sm)
        print(f"[GATE-1] FedOSS unknown-prob AUROC (test fold) = {auroc:.4f}")

    add_split_fingerprint(raw, cfg.seed)
    out = args.out or f"runs/fedoss_{cfg.dataset}_d{cfg.dirichlet_alpha:g}_seed{cfg.seed}.npz"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **raw)
    print(f"saved {out} (split_fp test={raw['test_fp']}, numpy={raw['numpy_version']})")


if __name__ == "__main__":
    main()
