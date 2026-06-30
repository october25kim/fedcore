"""STAGE 2 breadth: tabular federated open-set certification (CPU, no torch).

Answers the "CIFAR-only" concern and fits the safety-sensitive FL narrative. Runs
the IDENTICAL certification path (open_set_split + dirichlet_partition +
build_calibration + certify) on a TABULAR multi-class dataset, with a federated
linear model (per-client SGD logistic regression, FedAvg-averaged each round).

Dataset: sklearn covtype (forest cover, 7 classes) if fetchable; else a synthetic
multi-class tabular set (make_classification). Some classes are held out as unknown.

Run: ``python experiments/fedcore/run_tabular.py [--dataset covtype|synthetic]``
"""

from __future__ import annotations

import argparse

import numpy as np

from certify import certify_best_gamma, certify_grid
from fedosr_split import build_calibration, dirichlet_partition, open_set_split
from run_smoke import print_metric_table, save_csv
from scores import scored_views


def _load(dataset, seed):
    if dataset == "covtype":
        try:
            from sklearn.datasets import fetch_covtype
            d = fetch_covtype()
            X, y = d.data.astype(np.float32), d.target.astype(int) - 1  # classes 0..6
            # subsample for CPU speed
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(y), size=min(40000, len(y)), replace=False)
            return X[idx], y[idx], "covtype"
        except Exception as e:
            print(f"covtype fetch failed ({e}); falling back to synthetic")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=40000, n_features=40, n_informative=20,
                               n_classes=10, n_clusters_per_class=2, class_sep=1.5,
                               random_state=seed)
    return X.astype(np.float32), y.astype(int), "synthetic"


def _standardize(Xtr, Xte):
    Xtr = Xtr.astype(np.float64); Xte = Xte.astype(np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _fed_logreg(X, client_idx, remap, n_known, rounds, seed):
    """FedAvg over per-client SGD logistic regression; returns (coef, intercept)."""
    from sklearn.linear_model import SGDClassifier
    classes = np.arange(n_known)
    rng = np.random.default_rng(seed)
    g_coef = np.zeros((n_known, X.shape[1])); g_int = np.zeros(n_known)
    for r in range(rounds):
        coefs, ints, ws = [], [], []
        for idx in client_idx:
            if len(idx) == 0:
                continue
            yj = np.array([remap[int(c)] for c in _labels[idx]])
            clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1, tol=None,
                                random_state=seed)
            clf.coef_ = g_coef.copy(); clf.intercept_ = g_int.copy(); clf.classes_ = classes
            for _ in range(3):
                clf.partial_fit(X[idx], yj, classes=classes)
            coefs.append(clf.coef_); ints.append(clf.intercept_); ws.append(len(idx))
        ws = np.array(ws, float); ws /= ws.sum()
        g_coef = np.tensordot(ws, np.array(coefs), axes=1)
        g_int = np.tensordot(ws, np.array(ints), axes=1)
    return g_coef, g_int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["covtype", "synthetic"], default="covtype")
    ap.add_argument("--n_known", type=int, default=4)
    ap.add_argument("--n_clients", type=int, default=5)
    ap.add_argument("--dirichlet_alpha", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/tabular.csv")
    args = ap.parse_args()

    global _labels
    X, y, name = _load(args.dataset, args.seed)
    n_classes = len(np.unique(y))
    n_known = min(args.n_known, n_classes - 1)
    print(f"tabular={name} n={len(y)} classes={n_classes} n_known={n_known}")

    known_c, unknown_c, remap = open_set_split(y, n_known, args.seed)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(y))
    cut = int(0.6 * len(y))
    tr, te = perm[:cut], perm[cut:]
    Xtr, Xte = _standardize(X[tr], X[te])
    X_all = np.zeros(X.shape, dtype=np.float64); X_all[tr] = Xtr; X_all[te] = Xte
    _labels = y

    # labeled known-train -> dirichlet partition
    ktr = tr[np.isin(y[tr], known_c)]
    ktr_remap = np.array([remap[int(c)] for c in y[ktr]])
    client_idx = dirichlet_partition(ktr, ktr_remap, args.n_clients, args.dirichlet_alpha, args.seed)
    coef, intercept = _fed_logreg(X_all, client_idx, remap, n_known, args.rounds, args.seed)

    # calibration from TEST (known + unknown)
    kte = te[np.isin(y[te], known_c)]
    kte_remap = np.array([remap[int(c)] for c in y[kte]])
    ute = te[np.isin(y[te], unknown_c)]
    calib = build_calibration(kte, kte_remap, ute, args.n_clients, (0.4, 0.3, 0.3), 0.30, args.seed)

    def fold_logits(fold):
        idx, yo, cl = [], [], []
        for j, cf in enumerate(calib):
            f = cf[fold]; idx.append(np.asarray(f["idx"])); yo.append(np.asarray(f["y_open"]))
            cl.append(np.full(len(f["idx"]), j))
        idx = np.concatenate(idx); yo = np.concatenate(yo); cl = np.concatenate(cl)
        logits = X_all[idx] @ coef.T + intercept
        return logits, yo, cl

    scores = ("msp", "neg_entropy", "margin", "energy")
    views = {}
    raw = {}
    for fold in ("prop", "cert", "test"):
        lg, yo, cl = fold_logits(fold)
        views[fold] = scored_views(lg, yo, cl, list(scores))
        raw[f"{fold}_logits"] = lg; raw[f"{fold}_y_open"] = yo; raw[f"{fold}_client"] = cl
    import os as _os
    npz = _os.path.splitext(args.out)[0] + "_logits.npz"
    _os.makedirs(_os.path.dirname(_os.path.abspath(npz)), exist_ok=True)
    np.savez_compressed(npz, **raw)
    print(f"saved {npz}")

    rows = certify_grid(views["prop"], views["cert"], views["test"], scores=scores,
                        gammas=(0.2, 0.3, 0.5, 0.7, 1.0), alpha=args.alpha, delta=args.delta,
                        Lambdas=("simplex", "box"), n_clients=args.n_clients,
                        dirichlet_alpha=args.dirichlet_alpha, box=0.15, seed=args.seed)
    print_metric_table(rows)
    bg = [certify_best_gamma(views["prop"][s], views["cert"][s], views["test"][s], score_name=s,
                             gammas=(0.2, 0.3, 0.5, 0.7, 1.0), alpha=args.alpha, delta=args.delta,
                             n_clients=args.n_clients, dirichlet_alpha=args.dirichlet_alpha,
                             Lambda="box", box=0.15, seed=args.seed, margin=0.01) for s in scores]
    cert = [r for r in bg if r["certified"]]
    best = max(cert, key=lambda r: r["cert_coverage_lcb"], default=None)
    if best:
        print(f"\n[tabular {name}] CertifiedCoverage@{args.alpha} = {best['cert_coverage_lcb']:.4f} "
              f"(score={best['score_name']}, gamma*={best['gamma_star']}, test_risk={best['test_risk']:.3f})")
    else:
        finite = [r["cert_risk_ucb"] for r in bg if np.isfinite(r["cert_risk_ucb"])]
        mn = min(finite) if finite else float("inf")
        print(f"\n[tabular {name}] CertifiedCoverage@{args.alpha} = 0 (none certified); "
              f"min finite cert_ucb={mn:.3f} (same feasibility regime as CIFAR)")
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_csv(rows, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
