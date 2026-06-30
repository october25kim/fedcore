"""PRIORITY 4 / A4-real: calibration-budget sweep (mirrors synthetic Fig 9/A4).

Question (from the synthetic A4): does the worst-group alpha=0.10 certificate become
non-vacuous as the AUDIT BUDGET (number of trusted certification points) grows, with
the MODEL HELD FIXED?

Method (post-hoc on exported logits, no retraining). For each model seed we pool the
clean trusted folds (all exchangeable test-set points), hold a FIXED proposal fold
(selector source) and a FIXED held-out test fold, and then GROW the certification
fold from a small audit budget up to the full remaining pool. At each budget we run
the validity-preserving worst-group (G=2) best-gamma certificate (fixed MSP, box-
Lambda) at alpha=0.10 and record per-group accepted count, cert_risk_ucb,
CertifiedCoverage@0.10, and realized test_risk. We aggregate mean +/- std across the
5 model seeds and emit runs/ablation_calib_budget.csv plus a figure.

VALIDITY: model fixed; prop fold fixed and disjoint from cert/test; only the cert-fold
SIZE changes. The selector is a function of the proposal fold alone, so growing the
certification budget never leaks. G=2 is the legitimate worst-group certificate.

Run: python experiments/fedcore/exp_ablation_calib_budget.py
"""

from __future__ import annotations

import csv
import glob
import os

import numpy as np

from fedcore.certify import certify_best_gamma_grouped
from fedcore.scores import scored_views

ALPHA, DELTA = 0.10, 0.10
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
SCORE = "msp"          # fixed score -> no selection bias
MARGIN = 0.01
G = 2                  # worst-group certificate (legitimate)
SEEDS = (0, 1, 2, 3, 4)
TAG = "cifar10_d5_resnet18gn_none0.0"   # headline cell

# fixed fold sizes (out of the ~8570 pooled trusted points); cert grows from the rest
N_PROP = 2000
N_TEST = 1500
CERT_SIZES = (400, 800, 1200, 1800, 2400, 3200, 4000, 5000)


def _pool(npz):
    d = np.load(npz)
    return {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}


def _folds(pool, n_cert, seed):
    """Fixed prop(N_PROP)/test(N_TEST); cert = first n_cert of the remaining pool.

    The permutation is seeded by the MODEL seed only, so prop and test are identical
    across the whole budget sweep -- only the cert slice length changes."""
    rng = np.random.default_rng(1000 + seed)
    n = len(pool["y_open"])
    perm = rng.permutation(n)
    test_ix = perm[:N_TEST]
    prop_ix = perm[N_TEST:N_TEST + N_PROP]
    rest = perm[N_TEST + N_PROP:]
    cert_ix = rest[:n_cert]
    take = lambda ix: {k: pool[k][ix] for k in ("logits", "y_open", "client")}
    return {"prop": take(prop_ix), "cert": take(cert_ix), "test": take(test_ix)}


def _certify(parts, n_clients, alpha=ALPHA):
    views = {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                              parts[fn]["client"], [SCORE])[SCORE]
             for fn in ("prop", "cert", "test")}
    gmap = np.array([c * G // n_clients for c in range(n_clients)])
    return certify_best_gamma_grouped(
        views["prop"], views["cert"], views["test"], score_name=SCORE,
        group_map=gmap, G=G, gammas=GAMMAS, alpha=alpha, delta=DELTA,
        Lambda="box", box=0.15, seed=0, margin=MARGIN)


def main() -> None:
    base = "" if glob.glob("runs/*_logits.npz") else "../../"
    files = [base + f"runs/{TAG}_seed{s}_logits.npz" for s in SEEDS]
    files = [f for f in files if os.path.exists(f)]
    pools = [_pool(f) for f in files]
    n_clients = int(pools[0]["client"].max()) + 1

    rows = []
    print(f"A4-real calibration-budget sweep  (cell={TAG}, G={G}, alpha={ALPHA}, "
          f"delta={DELTA}, score={SCORE}, {len(pools)} model seeds)")
    print(f"{'n_cert':>7} {'cert_n(acc)':>11} {'per-grp':>8} {'cert_ucb_med':>12} "
          f"{'CertCov@.10':>12} {'n_pass':>7} {'test_risk':>10}")
    print("-" * 75)
    for n_cert in CERT_SIZES:
        covs, ucbs, accs, risks, passes = [], [], [], [], []
        for pool in pools:
            if N_PROP + N_TEST + n_cert > len(pool["y_open"]):
                continue
            r = _certify(_folds(pool, n_cert, 0), n_clients)
            cov = r["cert_coverage_lcb"] if r["certified"] else 0.0
            covs.append(cov)
            ucbs.append(r["cert_risk_ucb"] if np.isfinite(r["cert_risk_ucb"]) else np.nan)
            accs.append(r["cert_n"])
            risks.append(r["test_risk"])
            passes.append(int(r["certified"]))
        if not covs:
            continue
        covs = np.array(covs)
        cert_ucb_certified = [u for u, p in zip(ucbs, passes) if p]
        ucb_med = float(np.median(cert_ucb_certified)) if cert_ucb_certified else float("inf")
        row = {
            "n_cert": n_cert,
            "cert_n_acc_mean": round(float(np.mean(accs)), 1),
            "per_group_acc_mean": round(float(np.mean(accs)) / G, 1),
            "cert_ucb_median_certified": round(ucb_med, 4),
            "cert_ucb_mean_all": round(float(np.nanmean(ucbs)), 4),
            "CertCov_mean": round(float(covs.mean()), 4),
            "CertCov_std": round(float(covs.std()), 4),
            "n_pass": int(sum(passes)),
            "n_seeds": len(covs),
            "test_risk_mean": round(float(np.mean(risks)), 4),
        }
        rows.append(row)
        print(f"{n_cert:>7} {np.mean(accs):>11.0f} {np.mean(accs)/G:>8.0f} "
              f"{ucb_med:>12.4f} {covs.mean():>7.3f}+/-{covs.std():<4.3f} "
              f"{sum(passes):>3}/{len(covs)} {np.mean(risks):>10.4f}")

    out = base + "runs/ablation_calib_budget.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out}")
    _plot(rows, base)


def _plot(rows, base):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipping figure: {exc})")
        return
    x = [r["per_group_acc_mean"] for r in rows]
    cov = [r["CertCov_mean"] for r in rows]
    std = [r["CertCov_std"] for r in rows]
    ucb = [r["cert_ucb_mean_all"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.errorbar(x, cov, yerr=std, marker="o", color="C0", label="CertCov@0.10 (G=2)")
    ax1.set_xlabel("per-group accepted count (audit budget)")
    ax1.set_ylabel("CertifiedCoverage@0.10", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_ylim(bottom=0)
    ax2 = ax1.twinx()
    ax2.plot(x, ucb, marker="s", color="C3", label="cert_ucb (mean)")
    ax2.axhline(ALPHA, ls="--", color="gray", lw=1, label=f"alpha={ALPHA}")
    ax2.set_ylabel("cert_risk_ucb", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    fig.suptitle("A4-real: calibration-budget sweep (model fixed)")
    fig.tight_layout()
    path = base + "experiments/fedcore/figs/ablation_calib_budget.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
