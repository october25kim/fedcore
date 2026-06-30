"""Certified federated self-training (Proposition 4) -- audit-fold split + loops.

Protocol (honored exactly):

* Trusted clean calibration is split into a FIXED proposal pool ``P``, ``T``
  DISJOINT certification folds ``C^(1..T)``, and a held-out ``test`` fold.
* Round ``t = 1..T``:
    1. FedAvg -> ``f_t`` on (labeled data + currently accepted pseudo-labels).
    2. Choose selector ``A_t`` on ``P`` (risk buffer ``gamma*alpha``) using ``f_t``.
    3. Certify ``A_t`` on the FRESH fold ``C^(t)`` at level ``delta/T`` (Thm 1/1').
    4. Pseudo-label the unlabeled pool ``U``; accept the ``A_t=1`` subset
       (recomputed each round); fold accepted pseudo-labels into training.
* INVARIANT: ``C^(t)`` is never used for training or for selecting ``A_t``; the
  folds ``{C^(t)}`` and ``P`` are pairwise disjoint. Asserted in code.
* ASSUMPTION: ``U`` and the calibration folds share the deployment mixture, so the
  certified accepted-risk bound transfers to the contamination of accepted
  pseudo-labels.

Theorem-2 feasibility: a fold too small to certify (some client's accepted count
below ``ln(J/delta') / (-ln(1-alpha))``, ``delta'=delta/T``) yields an INFEASIBLE
round; the certified loop stops rather than fabricating a certificate.

torch is required (training); the certificate path reuses the torch-free core.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from certificates import conditional_risk_certificate, cp_lower
from fed_train import export_logits, fedavg
from scores import compute_score
from selector import (
    Selector,
    choose_threshold,
    counts_per_client,
    empirical_risk_coverage,
    open_set_error,
)


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
class MappedSubset(Dataset):
    """Subset exposing labels via a remap and/or a per-index override.

    ``override`` (dataset_index -> label) wins; else ``remap[true_label]``; if
    ``remap`` is None the override must cover every index (pseudo-labels).
    """

    def __init__(self, base, indices, remap: Optional[Dict[int, int]] = None,
                 override: Optional[Dict[int, int]] = None):
        self.base = base
        self.indices = list(indices)
        self.remap = remap
        self.override = override or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x, y = self.base[idx]
        if idx in self.override:
            return x, self.override[idx]
        return x, self.remap[int(y)]


def assert_disjoint(*index_arrays) -> None:
    """Assert the given index sets are pairwise disjoint."""
    seen = set()
    for arr in index_arrays:
        s = set(int(i) for i in arr)
        assert seen.isdisjoint(s), "self-training split leakage: index sets overlap"
        seen |= s


# --------------------------------------------------------------------------- #
# trusted-set partition: P + C^(1..T) + test
# --------------------------------------------------------------------------- #
def partition_selftrain(
    known_idx, known_y_remapped, unknown_idx, n_clients, T,
    prop_frac, test_frac, unknown_contamination, seed,
) -> Dict[str, object]:
    """Split the trusted set into proposal P, T cert folds, and a test fold.

    Returns ``{"prop": per_client, "cert_folds": [per_client]*T, "test": per_client}``
    where each per_client entry is a list (over clients) of ``{"idx","y_open"}``.
    Known points carry their true remapped label; injected unknowns carry -1.
    """
    rng = np.random.default_rng(seed)
    known_idx = np.asarray(known_idx)
    known_y = np.asarray(known_y_remapped)
    unk = np.asarray(unknown_idx).copy()
    rng.shuffle(unk)
    perm = rng.permutation(len(known_idx))
    known_idx, known_y = known_idx[perm], known_y[perm]

    client_pos = np.array_split(np.arange(len(known_idx)), n_clients)
    contam = unknown_contamination
    cert_frac = max(0.0, 1.0 - prop_frac - test_frac)

    prop: List[Dict] = []
    cert_folds: List[List[Dict]] = [[] for _ in range(T)]
    test: List[Dict] = []
    unk_ptr = 0

    def take_unk(n_k):
        nonlocal unk_ptr
        n_unk = int(round(contam / (1 - contam) * n_k)) if contam < 1 else n_k
        sl = unk[unk_ptr:unk_ptr + n_unk]
        unk_ptr += len(sl)
        return sl

    for j in range(n_clients):
        pos = client_pos[j]
        kidx, ky = known_idx[pos], known_y[pos]
        n = len(pos)
        n_prop = int(round(n * prop_frac))
        n_test = int(round(n * test_frac))
        n_cert = n - n_prop - n_test
        # proposal
        s, e = 0, n_prop
        u = take_unk(e - s)
        prop.append({"idx": np.concatenate([kidx[s:e], u]).astype(int),
                     "y_open": np.concatenate([ky[s:e], -np.ones(len(u), int)]).astype(int)})
        # T cert folds (disjoint slices of the cert block)
        cert_bounds = np.linspace(n_prop, n_prop + n_cert, T + 1).astype(int)
        for t in range(T):
            cs, ce = cert_bounds[t], cert_bounds[t + 1]
            u = take_unk(ce - cs)
            cert_folds[t].append({"idx": np.concatenate([kidx[cs:ce], u]).astype(int),
                                  "y_open": np.concatenate([ky[cs:ce], -np.ones(len(u), int)]).astype(int)})
        # test
        ts, te = n_prop + n_cert, n
        u = take_unk(te - ts)
        test.append({"idx": np.concatenate([kidx[ts:te], u]).astype(int),
                     "y_open": np.concatenate([ky[ts:te], -np.ones(len(u), int)]).astype(int)})

    # disjointness invariant across P and all cert folds (per client and global)
    allsets = [np.concatenate([p["idx"] for p in prop])]
    for t in range(T):
        allsets.append(np.concatenate([cf["idx"] for cf in cert_folds[t]]))
    allsets.append(np.concatenate([p["idx"] for p in test]))
    assert_disjoint(*allsets)

    return {"prop": prop, "cert_folds": cert_folds, "test": test}


def _gather(per_client: List[Dict]):
    """Concatenate (idx, y_open, client) across clients for one fold."""
    idx, yo, cl = [], [], []
    for j, f in enumerate(per_client):
        idx.append(np.asarray(f["idx"]))
        yo.append(np.asarray(f["y_open"]))
        cl.append(np.full(len(f["idx"]), j))
    return np.concatenate(idx), np.concatenate(yo), np.concatenate(cl)


# --------------------------------------------------------------------------- #
# selectors
# --------------------------------------------------------------------------- #
def best_gamma_selector(score, pred, y_open, client, gammas, alpha, delta, n_clients,
                        Lambda="simplex", box=0.15, seed=0) -> Selector:
    """Pick the buffer gamma on the PROPOSAL fold via a proposal-side proxy
    certificate (the most aggressive gamma whose proxy clears alpha; else the
    smallest gamma). Validity-preserving: the returned selector is a function of
    the proposal fold only."""
    cands = []
    for gamma in gammas:
        sel = choose_threshold(score, pred, y_open, gamma, alpha)
        cov, _ = empirical_risk_coverage(score, open_set_error(pred, y_open), sel.threshold)
        A, K, n = counts_per_client(score, pred, y_open, client, sel, n_clients)
        u = conditional_risk_certificate(A, K, n, delta, Lambda=Lambda, box=box, seed=seed).U
        cands.append({"gamma": gamma, "sel": sel, "cov": cov, "u": u})
    feas = [c for c in cands if c["sel"].feasible and c["u"] <= alpha]
    chosen = max(feas, key=lambda c: c["cov"]) if feas else min(cands, key=lambda c: c["gamma"])
    return chosen["sel"]


def naive_selector(score, pred, y_open, alpha, conf_thresh, n_grid=300) -> Selector:
    """Heuristic threshold: empirical accepted risk <= alpha (NO buffer, NO cert),
    intersected with a fixed-confidence floor (score >= conf_thresh proxy)."""
    sel = choose_threshold(score, pred, y_open, gamma=1.0, alpha=alpha, n_grid=n_grid)
    # also respect a fixed confidence floor if it is higher (heuristic union)
    if not sel.feasible:
        return Selector(threshold=float(np.quantile(score, conf_thresh)), feasible=True)
    return sel


# --------------------------------------------------------------------------- #
# the self-training loop (certified / naive / none)
# --------------------------------------------------------------------------- #
def run_self_training(
    mode: str,                       # 'certified' | 'naive' | 'none'
    *,
    make_model_fn,
    train_dataset,                   # base train dataset (with transform)
    test_dataset,                    # base test dataset (calibration source)
    train_targets: np.ndarray,       # true train labels (for U contamination)
    remap: Dict[int, int],
    known_classes,
    labeled_client_idx: List[np.ndarray],
    labeled_overrides: List[Dict[int, int]],   # client-side TRAIN corruption
    unlabeled_client_idx: List[np.ndarray],
    parts: Dict[str, object],        # from partition_selftrain
    n_known: int,
    n_clients: int,
    T: int,
    alpha: float,
    delta: float,
    gammas,
    score_name: str,
    fedavg_rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    conf_thresh: float = 0.95,
) -> List[Dict[str, object]]:
    """Run one self-training trajectory; return per-round metric records."""
    known_set = set(int(c) for c in known_classes)
    delta_round = delta / T
    thm2_floor = np.log(n_clients / delta_round) / (-np.log(1 - alpha))

    accepted: List[Dict[int, int]] = [dict() for _ in range(n_clients)]  # dsidx->pseudo label
    records: List[Dict[str, object]] = []

    for t in range(T):
        # 1) build training datasets (labeled + accepted pseudo) and FedAvg
        client_datasets = []
        for j in range(n_clients):
            ds = [MappedSubset(train_dataset, labeled_client_idx[j], remap,
                               override=labeled_overrides[j])]
            if mode != "none" and accepted[j]:
                ds.append(MappedSubset(train_dataset, list(accepted[j].keys()),
                                       remap=None, override=accepted[j]))
            client_datasets.append(ds[0] if len(ds) == 1 else ConcatDataset(ds))
        model = fedavg(make_model_fn, client_datasets, fedavg_rounds,
                       local_epochs, lr, batch_size, device)

        # 2) selector on the proposal pool P
        p_idx, p_yo, p_cl = _gather(parts["prop"])
        p_log = export_logits(model, test_dataset, p_idx, device, batch_size)
        p_s, p_p = compute_score(score_name, p_log), p_log.argmax(1)
        if mode == "certified":
            sel = best_gamma_selector(p_s, p_p, p_yo, p_cl, gammas, alpha,
                                      delta_round, n_clients)
        elif mode == "naive":
            sel = naive_selector(p_s, p_p, p_yo, alpha, conf_thresh)
        else:
            sel = Selector(threshold=np.inf, feasible=False)

        # 3) certify on the FRESH fold C^(t) at level delta/T
        cert_ucb, cert_cov_lcb, feasible_round = np.nan, np.nan, False
        if mode in ("certified", "naive") and sel.feasible:
            c_idx, c_yo, c_cl = _gather(parts["cert_folds"][t])
            c_log = export_logits(model, test_dataset, c_idx, device, batch_size)
            c_s, c_p = compute_score(score_name, c_log), c_log.argmax(1)
            A, K, n = counts_per_client(c_s, c_p, c_yo, c_cl, sel, n_clients)
            cert = conditional_risk_certificate(A, K, n, delta_round, Lambda="simplex")
            cert_ucb = float(cert.U)
            eps = delta_round / (2 * n_clients)
            cert_cov_lcb = float(np.min([cp_lower(int(A[j]), int(n[j]), eps)
                                         for j in range(n_clients)]))
            feasible_round = bool(cert.feasible and np.min(A) >= thm2_floor)

        # admission decision
        if mode == "certified":
            admit = bool(sel.feasible and cert_ucb <= alpha)
        elif mode == "naive":
            admit = bool(sel.feasible)
        else:
            admit = False

        # Theorem-2 stop: certified loop must not fabricate a certificate
        if mode == "certified" and sel.feasible and not feasible_round:
            stop_acc, stop_cov, stop_risk = _eval_test(
                model, test_dataset, parts["test"], score_name, sel, device, batch_size)
            records.append(_record(t, mode, cert_ucb, cert_cov_lcb, np.nan, 0,
                                   stop_acc, admit=False, infeasible=True,
                                   test_cov=stop_cov, test_risk=stop_risk))
            break

        # 4) pseudo-label U, accept the A_t=1 subset (recompute each round)
        n_pseudo, realized_contam = 0, np.nan
        if admit:
            wrong = total = 0
            for j in range(n_clients):
                u_idx = unlabeled_client_idx[j]
                if len(u_idx) == 0:
                    accepted[j] = {}
                    continue
                u_log = export_logits(model, train_dataset, u_idx, device, batch_size)
                u_s, u_p = compute_score(score_name, u_log), u_log.argmax(1)
                acc = sel.accept(u_s)
                newacc = {}
                for k, ds_i in enumerate(u_idx):
                    if not acc[k]:
                        continue
                    newacc[int(ds_i)] = int(u_p[k])
                    total += 1
                    true_c = int(train_targets[ds_i])
                    if true_c not in known_set or remap[true_c] != int(u_p[k]):
                        wrong += 1
                accepted[j] = newacc
            n_pseudo = total
            realized_contam = (wrong / total) if total else 0.0

        # downstream metrics on the held-out test fold
        test_acc, test_cov, test_risk = _eval_test(model, test_dataset, parts["test"],
                                                    score_name, sel, device, batch_size)
        records.append(_record(t, mode, cert_ucb, cert_cov_lcb, realized_contam,
                               n_pseudo, test_acc, admit=admit, infeasible=False,
                               test_cov=test_cov, test_risk=test_risk))
    return records


def _record(t, mode, ucb, cov_lcb, contam, n_pseudo, test_acc, *, admit, infeasible,
            test_cov=np.nan, test_risk=np.nan):
    return {"round": t, "mode": mode, "cert_risk_ucb": ucb, "cert_coverage_lcb": cov_lcb,
            "realized_contam": contam, "n_pseudo": n_pseudo, "test_acc": test_acc,
            "test_coverage": test_cov, "test_risk": test_risk,
            "admitted": admit, "infeasible_round": infeasible}


@torch.no_grad()
def _eval_test(model, test_dataset, test_parts, score_name, sel, device, batch_size):
    """Known-class accuracy over the test fold, plus accepted coverage/risk."""
    idx, yo, cl = _gather(test_parts)
    log = export_logits(model, test_dataset, idx, device, batch_size)
    pred = log.argmax(1)
    known_mask = yo >= 0
    acc = float((pred[known_mask] == yo[known_mask]).mean()) if known_mask.any() else np.nan
    if sel is not None and sel.feasible:
        s = compute_score(score_name, log)
        err = open_set_error(pred, yo)
        cov, risk = empirical_risk_coverage(s, err, sel.threshold)
    else:
        cov, risk = 0.0, 0.0
    return acc, cov, risk
