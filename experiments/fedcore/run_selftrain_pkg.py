"""One-shot certified self-training package (P1-P4) runner -> runs/selftrain_pkg.csv.

Reuses the run_selftrain_cifar data setup (labeled/unlabeled split of known-train +
client-side corruption; trusted P/cert/test folds from the CLEAN test set) but drives the
ONE-SHOT trajectory (selftrain_oneshot.run_oneshot) over a bounded grid:

  base in {fedavg}   (fedpd-prose base added by run_selftrain_pkg_fedpd.py once P1 pays),
  alpha in {0.10,0.20}, mode in {none,naive,certified,oracle}, audit_mult in {1,2,4},
  beta in {0.1,0.25,0.5,1.0}, seed in {...}.

Start small: --modes certified --alphas 0.20 --betas 0.25 --audit 1 (the smallest informative
slice), then expand. Logs the package metric schema per run.

Run (inside the torch container):
  python experiments/fedcore/run_selftrain_pkg.py --backbone resnet18 --norm gn \
      --modes none naive certified oracle --alphas 0.20 --betas 0.25 --audit 1 --seeds 0
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import torch

from fedosr_split import dirichlet_partition, open_set_split
from models import make_model
from noise import make_label_noise
from run_cifar import _load_cifar
from selftrain import partition_selftrain
from selftrain_oneshot import run_oneshot, train_base

FIELDS = ["base_model", "alpha", "mode", "audit_mult", "beta", "seed",
          "realized_contam", "admitted_count", "halted", "halt_freq",
          "known_acc", "balanced_acc", "test_risk", "test_coverage", "certcov_alpha",
          "cert_risk_ucb", "cert_coverage_lcb", "dirichlet_alpha", "noise"]


def setup_data(args, seed, prop_frac=0.4, test_frac=0.3):
    train, test = _load_cifar(args.dataset, args.data_root)
    train_labels = np.array(train.targets); test_labels = np.array(test.targets)
    known_classes, unknown_classes, remap = open_set_split(train_labels, args.n_known, seed)
    rng = np.random.default_rng(seed)
    known_train_idx = np.where(np.isin(train_labels, known_classes))[0]
    rng.shuffle(known_train_idx)
    n_lab = int(len(known_train_idx) * args.labeled_frac)
    labeled_idx, unlab_known = known_train_idx[:n_lab], known_train_idx[n_lab:]
    unlab_unknown = np.where(np.isin(train_labels, unknown_classes))[0]
    labeled_remapped = np.array([remap[int(c)] for c in train_labels[labeled_idx]])
    labeled_client_idx = dirichlet_partition(labeled_idx, labeled_remapped, args.n_clients,
                                             args.dirichlet_alpha, seed)
    remap_by_dsidx = {int(i): int(remap[int(train_labels[i])]) for i in labeled_idx}
    labeled_overrides = [make_label_noise(remap_by_dsidx, idx_j, args.noise_type, args.noise_rate,
                                          args.n_known, seed + j) for j, idx_j in enumerate(labeled_client_idx)]
    # CRITICAL (Prop-4 contract): U must share the deployment mixture with the cert folds,
    # else the certificate is anti-conservative for pseudo-label contamination (A5 effect).
    # Subsample U's unknowns so its unknown rate == the cert-fold deployment rate (0.30).
    p_deploy = 0.30
    n_u_known = len(unlab_known)
    n_u_unk_target = int(round(p_deploy / (1 - p_deploy) * n_u_known))
    rng.shuffle(unlab_unknown)
    unlab_unknown = unlab_unknown[:min(n_u_unk_target, len(unlab_unknown))]
    pool = np.concatenate([unlab_known, unlab_unknown]); rng.shuffle(pool)
    unlabeled_client_idx = [np.asarray(c) for c in np.array_split(pool, args.n_clients)]
    print(f"  U: known={n_u_known} unknown={len(unlab_unknown)} "
          f"(unknown rate={len(unlab_unknown)/len(pool):.3f}, matched to deployment {p_deploy})")
    test_known_idx = np.where(np.isin(test_labels, known_classes))[0]
    test_known_remapped = np.array([remap[int(c)] for c in test_labels[test_known_idx]])
    test_unknown_idx = np.where(np.isin(test_labels, unknown_classes))[0]
    parts = partition_selftrain(test_known_idx, test_known_remapped, test_unknown_idx,
                                args.n_clients, T=1, prop_frac=prop_frac, test_frac=test_frac,
                                unknown_contamination=0.30, seed=seed)
    return dict(train=train, test=test, train_labels=train_labels, remap=remap,
                known_classes=known_classes, labeled_client_idx=labeled_client_idx,
                labeled_overrides=labeled_overrides, unlabeled_client_idx=unlabeled_client_idx,
                parts=parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10"); ap.add_argument("--n_known", type=int, default=6)
    ap.add_argument("--n_clients", type=int, default=5); ap.add_argument("--dirichlet_alpha", type=float, default=5.0)
    ap.add_argument("--labeled_frac", type=float, default=0.5)
    ap.add_argument("--noise_type", choices=["none", "symmetric", "asymmetric"], default="none")
    ap.add_argument("--noise_rate", type=float, default=0.0)
    ap.add_argument("--backbone", choices=["simplecnn", "resnet18"], default="resnet18")
    ap.add_argument("--norm", choices=["bn", "gn"], default="gn")
    ap.add_argument("--score", default="msp")
    ap.add_argument("--fedavg_rounds", type=int, default=40)
    ap.add_argument("--finetune_rounds", type=int, default=15)
    ap.add_argument("--local_epochs", type=int, default=2)
    ap.add_argument("--modes", nargs="+", default=["none", "naive", "certified", "oracle"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.20])
    ap.add_argument("--betas", type=float, nargs="+", default=[0.25])
    ap.add_argument("--audit", type=int, nargs="+", default=[1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--conf_weight", action="store_true")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default="runs/selftrain_pkg.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.fedavg_rounds, args.finetune_rounds, args.local_epochs = 2, 1, 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = f"FedAvg+{args.score.upper()}"
    print(f"device={device} base={base_model} d={args.dirichlet_alpha} backbone={args.backbone}/{args.norm}")
    print(f"grid: modes={args.modes} alphas={args.alphas} betas={args.betas} audit={args.audit} seeds={args.seeds}")

    rows: List[Dict] = []
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        D = setup_data(args, seed)
        make_fn = lambda: make_model(args.n_known, backbone=args.backbone, norm=args.norm)
        # train the base ONCE per seed (identical across mode/alpha/beta/audit) and reuse
        print(f"  [seed {seed}] training base ({args.fedavg_rounds} FedAvg rounds)...")
        base = train_base(make_fn, D["train"], D["remap"], D["labeled_client_idx"],
                          D["labeled_overrides"], args.n_clients, args.fedavg_rounds,
                          args.local_epochs, 0.01, 64, device)
        for alpha in args.alphas:
            for mode in args.modes:
                # 'none'/'oracle' are beta/audit-invariant for admission; still sweep beta for fine-tune effect
                betas = args.betas if mode in ("naive", "certified", "oracle") else [args.betas[0]]
                audits = args.audit if mode in ("naive", "certified") else [args.audit[0]]
                for audit_mult in audits:
                    for beta in betas:
                        rec = run_oneshot(
                            mode, make_model_fn=make_fn, train_dataset=D["train"], test_dataset=D["test"],
                            train_targets=D["train_labels"], remap=D["remap"], known_classes=D["known_classes"],
                            labeled_client_idx=D["labeled_client_idx"], labeled_overrides=D["labeled_overrides"],
                            unlabeled_client_idx=D["unlabeled_client_idx"], parts=D["parts"],
                            n_known=args.n_known, n_clients=args.n_clients, alpha=alpha, delta=0.10,
                            gammas=(0.2, 0.3, 0.5, 0.7, 1.0), score_name=args.score,
                            fedavg_rounds=args.fedavg_rounds, finetune_rounds=args.finetune_rounds,
                            local_epochs=args.local_epochs, lr=0.01, batch_size=64, device=device,
                            beta=beta, conf_weight=args.conf_weight, audit_mult=audit_mult, base=base)
                        row = {"base_model": base_model, "seed": seed,
                               "halt_freq": 1.0 if rec.get("halted") else 0.0,
                               "dirichlet_alpha": args.dirichlet_alpha,
                               "noise": f"{args.noise_type}{args.noise_rate}", **rec}
                        rows.append(row)
                        print(f"  [{mode:9s} a={alpha} b={beta} audit={audit_mult}x] "
                              f"contam={_fmt(rec['realized_contam'])} adm={rec['admitted_count']} "
                              f"known_acc={rec['known_acc']:.4f} bal={rec['balanced_acc']:.4f} "
                              f"test_risk={rec['test_risk']:.3f} certcov={rec['certcov_alpha']:.3f} "
                              f"halt={rec.get('halted')}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsaved/appended {len(rows)} rows -> {args.out}")
    # headline: gain vs 'none'
    none_acc = {(r["seed"], r["alpha"]): r["known_acc"] for r in rows if r["mode"] == "none"}
    for r in rows:
        if r["mode"] in ("certified", "oracle"):
            base_acc = none_acc.get((r["seed"], r["alpha"]))
            if base_acc is not None:
                print(f"  gain[{r['mode']} a={r['alpha']} b={r['beta']} audit={r['audit_mult']}x] "
                      f"known_acc {base_acc:.4f} -> {r['known_acc']:.4f}  (Δ={r['known_acc']-base_acc:+.4f})")


def _fmt(x):
    return "nan" if x != x else f"{x:.3f}"


if __name__ == "__main__":
    main()
