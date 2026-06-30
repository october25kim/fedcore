"""P2: one-shot certified self-training with FedPD-PROSER as the base model.

Strongest detector route. Reuses the FedPD-PROSER training + native score from
run_fedpd_cifar.py and the (score-agnostic) certificate core; the self-training data
setup (labeled/unlabeled split, P/cert/test folds, U matched to the 0.30 deployment
unknown rate) is reused from run_selftrain_pkg.setup_data.

Pipeline per mode in {none, certified, oracle}:
  1. federated FedPD base = closed-set CE pretrain + PROSER dummy fine-tune on LABELED;
  2. native PROSER accept-score = -(dummyconf - maxknownconf) on the proposal/cert folds;
  3. one-shot worst-group G=2 certificate on the FULL cert fold at delta (audit_mult ladder);
  4. admit certified-accepted pseudo-labels from U (oracle: only truly-correct);
  5. fine-tune ONCE (more PROSER rounds) on labeled + admitted pseudo (beta=1.0 => concat);
  6. evaluate known/balanced acc, accepted test risk, CertifiedCoverage@alpha (G=2).

Run (container, /fedpd mounted): see scripts/docker_selftrain_fedpd.sh.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

import fedcore.experiments.run_fedpd_cifar as fp                     # FedPD model + train_ce + traindummy + export + _avg
from fedcore.certificate import conditional_risk_certificate, cp_lower
from fedcore.experiments.run_selftrain_pkg import setup_data
from fedcore.scores import scored_views
from fedcore.experiments.selftrain import MappedSubset, _gather, best_gamma_selector
from fedcore.atomic_io import append_csv_locked
from fedcore.selector import Selector, counts_per_client, empirical_risk_coverage, open_set_error
from fedcore.certify import certify_best_gamma_grouped

FIELDS = ["base_model", "labeled_frac", "alpha", "mode", "audit_mult", "beta", "seed",
          "realized_contam", "admitted_count", "halted", "halt_freq", "known_acc",
          "balanced_acc", "test_risk", "test_coverage", "certcov_alpha", "cert_risk_ucb",
          "dirichlet_alpha"]


def _loader(ds, bs):
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)


def fed_fedpd(client_datasets, n_known, rounds, local_fn, lr, bs, device, init_state=None, tag=""):
    glob = fp.make_net(n_known, device)
    if init_state is not None:
        glob.load_state_dict(init_state)
    weights = [len(d) for d in client_datasets]
    for r in range(rounds):
        states = []
        for ds in client_datasets:
            if len(ds) < bs:
                states.append(None); continue
            local = fp.make_net(n_known, device)
            local.load_state_dict(copy.deepcopy(glob.state_dict()))
            opt = torch.optim.SGD(local.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            ld = _loader(ds, bs)
            for _ in range(1):
                local_fn(ld, local, opt)
            states.append({k: v.cpu() for k, v in local.state_dict().items()})
        valid = [(s, w) for s, w in zip(states, weights) if s is not None]
        glob.load_state_dict(fp._avg([s for s, _ in valid], [w for _, w in valid]))
        if tag and (r % max(1, rounds // 5) == 0 or r == rounds - 1):
            print(f"  {tag} round {r}/{rounds}")
    return glob


def train_base(labeled_ds, n_known, pretrain_rounds, proser_rounds, lr, bs, device):
    ce = lambda ld, net, opt: fp.train_ce(ld, net, opt, device)
    pros = lambda ld, net, opt: fp.traindummy(ld, net, opt, n_known, device)
    net = fed_fedpd(labeled_ds, n_known, pretrain_rounds, ce, lr, bs, device, tag="pretrain")
    net = fed_fedpd(labeled_ds, n_known, proser_rounds, pros, lr, bs, device,
                    init_state=net.state_dict(), tag="proser")
    return net


def _score_pred(net, base_dataset, idx, n_known, device, bs=256):
    """FedPD native: accept-score = -(dummyconf - maxknown) [higher => more ID], pred=argmax."""
    logits, conf = fp.export(net, base_dataset, idx, n_known, device, bs)
    return -conf.astype(float), logits.argmax(1), logits


def _eval(net, test_dataset, parts, n_known, n_clients, alpha, delta, device, bs, G=2):
    ti, tyo, tcl = _gather(parts["test"])
    s, pred, logits = _score_pred(net, test_dataset, ti, n_known, device, bs)
    known = tyo >= 0
    acc = float((pred[known] == tyo[known]).mean()) if known.any() else float("nan")
    pcs = [float((pred[(known) & (tyo == c)] == c).mean()) for c in range(n_known)
           if ((known) & (tyo == c)).any()]
    bal = float(np.mean(pcs)) if pcs else float("nan")
    gmap = np.array([c * G // n_clients for c in range(n_clients)])
    views = {}
    for fold, pc in (("prop", parts["prop"]), ("cert", parts["cert_folds"][0]), ("test", parts["test"])):
        fi, fyo, fcl = _gather(pc)
        fs, fp_, _ = _score_pred(net, test_dataset, fi, n_known, device, bs)
        views[fold] = {"score": fs, "pred": fp_, "y_open": fyo, "client": fcl}
    r = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"], score_name="FedPD-PROSER",
                                   group_map=gmap, G=G, gammas=(0.2, 0.3, 0.5, 0.7, 1.0), alpha=alpha,
                                   delta=delta, Lambda="box", box=0.15, seed=0, margin=0.01)
    return {"known_acc": acc, "balanced_acc": bal, "test_risk": float(r["test_risk"]),
            "test_coverage": float(r["test_coverage"]),
            "certcov_alpha": float(r["cert_coverage_lcb"] if r["certified"] else 0.0)}


def _audit_subsample(n, mult, div=4, seed=0):
    """Audit ladder on the FIXED cert block: keep mult/div of it (div=8 -> 8x = full block)."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n)[:max(1, min(n, int(round(n * mult / float(div)))))]


def run_mode(mode, base, D, n_known, n_clients, alpha, delta, audit_mult, device, bs,
             finetune_rounds, lr, G=2, audit_div=4):
    remap, train_targets = D["remap"], D["train_labels"]
    known_set = set(int(c) for c in D["known_classes"])
    group_map = np.array([c * G // n_clients for c in range(n_clients)])
    parts = D["parts"]

    # selector on proposal
    pi, pyo, pcl = _gather(parts["prop"])
    ps, pp, _ = _score_pred(base, D["test"], pi, n_known, device, bs)
    if mode == "certified":
        sel = best_gamma_selector(ps, pp, pyo, group_map[pcl], (0.2, 0.3, 0.5, 0.7, 1.0),
                                  alpha, delta, G, Lambda="box")
    elif mode == "oracle":
        from selftrain import naive_selector
        sel = naive_selector(ps, pp, pyo, alpha, conf_thresh=0.95)
    else:
        sel = Selector(threshold=np.inf, feasible=False)

    cert_ucb, halted, admit = float("nan"), False, False
    if mode == "certified" and sel.feasible:
        ci, cyo, ccl = _gather(parts["cert_folds"][0])
        keep = _audit_subsample(len(ci), audit_mult, div=audit_div)
        ci, cyo, ccl = ci[keep], cyo[keep], ccl[keep]
        cs, cp, _ = _score_pred(base, D["test"], ci, n_known, device, bs)
        A, K, n = counts_per_client(cs, cp, cyo, group_map[ccl], sel, G)
        cert = conditional_risk_certificate(A, K, n, delta, Lambda="box", box=0.15, seed=0)
        cert_ucb = float(cert.U)
        thm2 = np.log(G / delta) / (-np.log(1 - alpha))
        feasible = bool(cert.feasible and np.min(A) >= thm2)
        admit = bool(cert.feasible and cert_ucb <= alpha)
        halted = bool(not feasible)
    elif mode == "oracle":
        admit = True

    # admit pseudo from U
    accepted = [dict() for _ in range(n_clients)]
    total = wrong = 0
    if admit:
        for j in range(n_clients):
            u_idx = D["unlabeled_client_idx"][j]
            if len(u_idx) == 0:
                continue
            us, up, _ = _score_pred(base, D["train"], u_idx, n_known, device, bs)
            acc_mask = (us >= sel.threshold) if mode != "oracle" else np.ones(len(u_idx), bool)
            for k, ds_i in enumerate(u_idx):
                if not acc_mask[k]:
                    continue
                tc = int(train_targets[ds_i]); correct = (tc in known_set) and (remap[tc] == int(up[k]))
                if mode == "oracle" and not correct:
                    continue
                accepted[j][int(ds_i)] = int(up[k]); total += 1
                if not correct:
                    wrong += 1
    contam = (wrong / total) if total else float("nan")

    # fine-tune (PROSER) on labeled + admitted pseudo (beta=1.0 -> concat)
    net = base
    if mode != "none" and total > 0:
        ft_ds = []
        for j in range(n_clients):
            parts_j = [MappedSubset(D["train"], D["labeled_client_idx"][j], remap,
                                    override=D["labeled_overrides"][j])]
            if accepted[j]:
                parts_j.append(MappedSubset(D["train"], list(accepted[j]), remap=None, override=accepted[j]))
            ft_ds.append(ConcatDataset(parts_j) if len(parts_j) > 1 else parts_j[0])
        pros = lambda ld, m, opt: fp.traindummy(ld, m, opt, n_known, device)
        net = fed_fedpd(ft_ds, n_known, finetune_rounds, pros, lr, bs, device,
                        init_state=base.state_dict(), tag="finetune")

    ev = _eval(net, D["test"], parts, n_known, n_clients, alpha, delta, device, bs)
    return {"base_model": "FedPD-PROSER", "alpha": alpha, "mode": mode, "audit_mult": audit_mult,
            "beta": 1.0, "realized_contam": contam, "admitted_count": total, "halted": halted,
            "cert_risk_ucb": cert_ucb, **ev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10"); ap.add_argument("--n_known", type=int, default=6)
    ap.add_argument("--n_clients", type=int, default=5); ap.add_argument("--dirichlet_alpha", type=float, default=5.0)
    ap.add_argument("--labeled_frac", type=float, default=0.5)
    ap.add_argument("--noise_type", default="none"); ap.add_argument("--noise_rate", type=float, default=0.0)
    ap.add_argument("--pretrain_rounds", type=int, default=40); ap.add_argument("--proser_rounds", type=int, default=15)
    ap.add_argument("--finetune_rounds", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.20); ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--audit", type=int, nargs="+", default=[4]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit_div", type=int, default=4, help="audit ladder divisor (8 => 8x = full cert block)")
    ap.add_argument("--prop_frac", type=float, default=0.4); ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--modes", nargs="+", default=["none", "certified", "oracle"])
    ap.add_argument("--data_root", default="data"); ap.add_argument("--out", default="runs/selftrain_pkg.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.pretrain_rounds, args.proser_rounds, args.finetune_rounds = 2, 2, 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"[selftrain FedPD-PROSER] device={device} alpha={args.alpha} audit={args.audit} "
          f"(div={args.audit_div}) lf={args.labeled_frac} prop/test={args.prop_frac}/{args.test_frac} seed={args.seed}")
    D = setup_data(args, args.seed, prop_frac=args.prop_frac, test_frac=args.test_frac)
    labeled_ds = [MappedSubset(D["train"], D["labeled_client_idx"][j], D["remap"],
                               override=D["labeled_overrides"][j]) for j in range(args.n_clients)]
    print(f"training FedPD base ({args.pretrain_rounds} pretrain + {args.proser_rounds} PROSER)...")
    base = train_base(labeled_ds, args.n_known, args.pretrain_rounds, args.proser_rounds, 0.1, 64, device)

    # base trained ONCE; certified swept over audit budgets; none/oracle are audit-invariant
    jobs = []
    for mode in args.modes:
        if mode == "certified":
            jobs += [("certified", a) for a in args.audit]
        else:
            jobs.append((mode, args.audit[0]))
    rows = []
    for mode, audit in jobs:
        rec = run_mode(mode, base, D, args.n_known, args.n_clients, args.alpha, args.delta,
                       audit, device, 64, args.finetune_rounds, 0.01, audit_div=args.audit_div)
        rec["seed"] = args.seed; rec["dirichlet_alpha"] = args.dirichlet_alpha
        rec["labeled_frac"] = args.labeled_frac
        rec["halt_freq"] = 1.0 if rec.get("halted") else 0.0
        rows.append(rec)
        print(f"  [{mode:9s} a={args.alpha} audit={audit}x] contam={_fmt(rec['realized_contam'])} "
              f"adm={rec['admitted_count']} known_acc={rec['known_acc']:.4f} bal={rec['balanced_acc']:.4f} "
              f"test_risk={rec['test_risk']:.3f} certcov={rec['certcov_alpha']:.3f} halt={rec['halted']}")

    append_csv_locked(args.out, FIELDS, rows, extrasaction="ignore")
    print(f"saved/appended {len(rows)} rows -> {args.out}")
    none = next((r["known_acc"] for r in rows if r["mode"] == "none"), None)
    for r in rows:
        if r["mode"] in ("certified", "oracle") and none is not None:
            print(f"  gain[{r['mode']}] {none:.4f} -> {r['known_acc']:.4f} (Δ={r['known_acc']-none:+.4f})")


def _fmt(x):
    return "nan" if x != x else f"{x:.3f}"


if __name__ == "__main__":
    main()
