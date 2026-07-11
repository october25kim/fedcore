"""One-shot certified self-training (P1-P4 package): remove the T-way fold split.

Pipeline (one shot, level delta -- NOT delta/T):
  1. FedAvg-train the base model on LABELED data;
  2. pick the selector on the PROPOSAL fold (risk buffer gamma*alpha);
  3. certify the accepted set ONCE on the FULL certification fold at level delta;
  4. admit certified-accepted pseudo-labels from the unlabeled pool U;
  5. fine-tune ONCE on  L_sup + beta * L_pseudo  (optionally confidence-weighted);
  6. evaluate (known acc, balanced acc, test selective risk, CertifiedCoverage@alpha).

Modes:
  none      -- no pseudo-labels (supervised-only reference);
  naive     -- admit the selector-accepted set with NO certificate (may contaminate);
  certified -- admit only if the one-shot certificate clears alpha (Prop-4 contract);
  oracle    -- admit only TRULY-correct accepted pseudo-labels (clean upper bound).

Split hygiene (asserted upstream in partition_selftrain): proposal / certification / test
folds are disjoint; the selector is chosen on the proposal fold; the admission certificate
uses a fold independent of the model that produced the labels. The unlabeled pool U is the
deployment-mixture source whose accepted-risk the certificate bounds (Prop-4 assumption).

torch required. base="fedavg" uses make_model+FedAvg with MSP/energy; base="fedpd" expects
a pre-trained FedPD-PROSER model + native score callable injected by the caller.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedcore.certificate import conditional_risk_certificate, cp_lower
from fedcore.certify import certify_best_gamma_grouped
from fedcore.models.fed_train import export_logits, fedavg
from fedcore.scores import compute_score, scored_views
from fedcore.experiments.selftrain import MappedSubset, _gather, best_gamma_selector, naive_selector
from fedcore.selector import Selector, counts_per_client, empirical_risk_coverage, open_set_error


# --------------------------------------------------------------------------- #
# weighted FedAvg fine-tune: L_sup + beta * L_pseudo
# --------------------------------------------------------------------------- #
class _WeightedDS(Dataset):
    """Wrap a base dataset slice returning (x, y, w); w is a per-sample loss weight."""

    def __init__(self, base, indices, label_of: Dict[int, int], weight: float):
        self.base, self.indices = base, list(indices)
        self.label_of, self.weight = label_of, weight

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x, _ = self.base[idx]
        return x, self.label_of[int(idx)], self.weight


def _local_train_weighted(model, loaders, epochs, lr, device):
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    ce = nn.CrossEntropyLoss(reduction="none")
    for _ in range(epochs):
        for loader in loaders:
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device).float()
                opt.zero_grad()
                loss = (ce(model(xb), yb) * wb).mean()
                loss.backward()
                opt.step()
    return model


def _fedavg_weighted(init_state, make_model_fn, client_specs, rounds, epochs, lr, bs, device):
    """FedAvg starting from init_state; each client has a list of (_WeightedDS) loaders."""
    from fed_train import _average_state_dicts
    g = make_model_fn().to(device)
    g.load_state_dict(copy.deepcopy(init_state))
    sizes = [sum(len(d) for d in specs) for specs in client_specs]
    for _ in range(rounds):
        states, used = [], []
        for specs, size in zip(client_specs, sizes):
            if size == 0:
                continue
            local = make_model_fn().to(device)
            local.load_state_dict(copy.deepcopy(g.state_dict()))
            loaders = [DataLoader(d, batch_size=bs, shuffle=True, drop_last=False) for d in specs]
            _local_train_weighted(local, loaders, epochs, lr, device)
            states.append(local.state_dict()); used.append(size)
        if states:
            g.load_state_dict(_average_state_dicts(states, used))
    return g


# --------------------------------------------------------------------------- #
# evaluation: known acc, balanced acc, accepted risk, CertCov@alpha
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _eval(model, test_dataset, parts, score_name, n_known, n_clients, alpha, delta, device, bs, G=2):
    idx, yo, cl = _gather(parts["test"])
    log = export_logits(model, test_dataset, idx, device, bs)
    pred = log.argmax(1)
    known = yo >= 0
    acc = float((pred[known] == yo[known]).mean()) if known.any() else float("nan")
    # balanced (class-mean) accuracy over known classes
    pcs = []
    for c in range(n_known):
        m = known & (yo == c)
        if m.any():
            pcs.append(float((pred[m] == yo[m]).mean()))
    bal = float(np.mean(pcs)) if pcs else float("nan")
    # final deployment CertCov@alpha via the standard best-gamma path (prop/cert/test)
    views = {}
    for fold in ("prop", "cert", "test"):
        fi, fyo, fcl = _gather(parts[fold] if fold != "cert" else parts["cert_folds"][0])
        flog = export_logits(model, test_dataset, fi, device, bs)
        views[fold] = scored_views(flog, fyo, fcl, [score_name])[score_name]
    gmap = np.array([c * G // n_clients for c in range(n_clients)])
    r = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"], score_name=score_name,
                                   group_map=gmap, G=G, gammas=(0.2, 0.3, 0.5, 0.7, 1.0),
                                   alpha=alpha, delta=delta, Lambda="box", box=0.15, seed=0, margin=0.01)
    certcov = r["cert_coverage_lcb"] if r["certified"] else 0.0
    return {"known_acc": acc, "balanced_acc": bal, "test_risk": float(r["test_risk"]),
            "test_coverage": float(r["test_coverage"]), "certcov_alpha": float(certcov)}


def train_base(make_model_fn, train_dataset, remap, labeled_client_idx, labeled_overrides,
               n_clients, fedavg_rounds, local_epochs, lr, batch_size, device):
    """FedAvg the base model on LABELED data only (cache + reuse across grid cells)."""
    labeled_ds = [MappedSubset(train_dataset, labeled_client_idx[j], remap, override=labeled_overrides[j])
                  for j in range(n_clients)]
    return fedavg(make_model_fn, labeled_ds, fedavg_rounds, local_epochs, lr, batch_size, device)


# --------------------------------------------------------------------------- #
# the one-shot trajectory
# --------------------------------------------------------------------------- #
def run_oneshot(
    mode: str,                       # none | naive | certified | oracle
    *,
    make_model_fn,
    train_dataset, test_dataset, train_targets, remap, known_classes,
    labeled_client_idx, labeled_overrides, unlabeled_client_idx,
    parts, n_known, n_clients, alpha, delta, gammas, score_name,
    fedavg_rounds, finetune_rounds, local_epochs, lr, batch_size, device,
    beta: float = 0.25, conf_weight: bool = False, audit_mult: int = 1, G: int = 2,
    base=None,
) -> Dict[str, object]:
    known_set = set(int(c) for c in known_classes)
    # worst-group certificate (G public groups), matching the headline/T8 protocol --
    # NOT per-client (G=J), which is needlessly conservative for admission.
    group_map = np.array([c * G // n_clients for c in range(n_clients)])

    # (1) base model on labeled only (train once; the caller may pass a cached base
    # since it is identical across mode/alpha/beta/audit for a given seed)
    if base is None:
        base = train_base(make_model_fn, train_dataset, remap, labeled_client_idx,
                          labeled_overrides, n_clients, fedavg_rounds, local_epochs, lr,
                          batch_size, device)

    # (2) selector on the proposal fold
    p_idx, p_yo, p_cl = _gather(parts["prop"])
    p_log = export_logits(base, test_dataset, p_idx, device, batch_size)
    p_s, p_p = compute_score(score_name, p_log), p_log.argmax(1)
    if mode == "certified":
        sel = best_gamma_selector(p_s, p_p, p_yo, group_map[p_cl], gammas, alpha, delta, G, Lambda="box")
    elif mode in ("naive", "oracle"):
        sel = naive_selector(p_s, p_p, p_yo, alpha, conf_thresh=0.95)
    else:
        sel = Selector(threshold=np.inf, feasible=False)

    # (3) one-shot certificate on the FULL cert fold (audit_mult subsamples the trusted pool)
    cert_ucb, cert_cov_lcb, feasible = float("nan"), float("nan"), False
    if mode in ("certified", "naive") and sel.feasible:
        c_idx, c_yo, c_cl = _gather(parts["cert_folds"][0])
        keep = _audit_subsample(len(c_idx), audit_mult, seed=0)  # 1x=1/4 ... 4x=full block
        c_idx, c_yo, c_cl = c_idx[keep], c_yo[keep], c_cl[keep]
        c_log = export_logits(base, test_dataset, c_idx, device, batch_size)
        c_s, c_p = compute_score(score_name, c_log), c_log.argmax(1)
        A, K, n = counts_per_client(c_s, c_p, c_yo, group_map[c_cl], sel, G)  # worst-group G
        cert = conditional_risk_certificate(A, K, n, delta, Lambda="box", box=0.15, seed=0)
        cert_ucb = float(cert.U)
        thm2 = np.log(G / delta) / (-np.log(1 - alpha))
        feasible = bool(cert.feasible and np.min(A) >= thm2)
        eps = delta / (2 * G)
        cert_cov_lcb = float(np.min([cp_lower(int(A[j]), int(n[j]), eps) for j in range(G)]))

    if mode == "certified":
        admit = bool(sel.feasible and cert.feasible and cert_ucb <= alpha)
        halted = bool(sel.feasible and not feasible)
    elif mode == "naive":
        admit, halted = bool(sel.feasible), False
    elif mode == "oracle":
        admit, halted = True, False
    else:
        admit, halted = False, False

    # (4) admit pseudo-labels from U
    accepted: List[Dict[int, int]] = [dict() for _ in range(n_clients)]
    n_pseudo = wrong = total = 0
    pseudo_conf: Dict[int, float] = {}
    if admit:
        for j in range(n_clients):
            u_idx = unlabeled_client_idx[j]
            if len(u_idx) == 0:
                continue
            u_log = export_logits(base, train_dataset, u_idx, device, batch_size)
            u_s, u_p = compute_score(score_name, u_log), u_log.argmax(1)
            acc_mask = (u_s >= sel.threshold) if mode != "oracle" else np.ones(len(u_idx), bool)
            for k, ds_i in enumerate(u_idx):
                if not acc_mask[k]:
                    continue
                true_c = int(train_targets[ds_i])
                correct = (true_c in known_set) and (remap[true_c] == int(u_p[k]))
                if mode == "oracle" and not correct:
                    continue  # oracle admits only truly-correct labels
                accepted[j][int(ds_i)] = int(u_p[k]); pseudo_conf[int(ds_i)] = float(u_s[k])
                total += 1
                if not correct:
                    wrong += 1
        n_pseudo = total
    realized_contam = (wrong / total) if total else float("nan")

    # (5) fine-tune ONCE on L_sup + beta * L_pseudo (continue from base)
    model = base
    if mode != "none" and n_pseudo > 0:
        cw = pseudo_conf if conf_weight else None
        specs = []
        for j in range(n_clients):
            sup_labels = {int(i): (labeled_overrides[j].get(int(i)) if int(i) in labeled_overrides[j]
                                   else remap[int(train_targets[i])]) for i in labeled_client_idx[j]}
            client = [_WeightedDS(train_dataset, labeled_client_idx[j], sup_labels, 1.0)]
            if accepted[j]:
                if cw is None:
                    client.append(_WeightedDS(train_dataset, list(accepted[j]), accepted[j], beta))
                else:  # confidence-weighted: w = beta * normalized margin
                    for di, lab in accepted[j].items():
                        client.append(_WeightedDS(train_dataset, [di], {di: lab}, beta * cw[di]))
            specs.append(client)
        model = _fedavg_weighted(base.state_dict(), make_model_fn, specs,
                                 finetune_rounds, local_epochs, lr, batch_size, device)

    # (6) evaluate
    ev = _eval(model, test_dataset, parts, score_name, n_known, n_clients, alpha, delta, device, batch_size)
    return {"mode": mode, "alpha": alpha, "beta": beta, "audit_mult": audit_mult,
            "cert_risk_ucb": cert_ucb, "cert_coverage_lcb": cert_cov_lcb,
            "realized_contam": realized_contam, "admitted_count": n_pseudo,
            "admitted": admit, "halted": halted, **ev}


def _audit_subsample(n, mult, seed):
    """Audit-budget ladder on a FIXED trusted pool: keep mult/4 of the cert block.

    1x -> n/4 (small audit budget), 2x -> n/2, 4x -> full block. This holds the model
    fixed and only changes how many trusted certification points are available.
    """
    rng = np.random.default_rng(seed)
    target = max(1, min(n, int(round(n * mult / 4.0))))
    return rng.permutation(n)[:target]
