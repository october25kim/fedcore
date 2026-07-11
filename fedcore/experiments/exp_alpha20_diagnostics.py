"""P0: full worst-group (G=2) certification diagnostics for Table 5.

Reproduces the EXACT headline protocol (worst-group G=2, fixed MSP,
cert_frac=0.5, box Lambda, margin=0.01, gamma grid) and emits, per cell x seed
x alpha, the per-certificate diagnostics the alpha=0.10 aggregates
(runs/agg_main.csv) do not carry, so the paper table itself (not the code
release) can show cert_n / cert_k / test_coverage / selected gamma without
dashes:

  - cert_risk_ucb_G2   : worst-group conditional risk UCB (Theorem 1/2)
  - cert_n_min_group   : the SMALLER per-group accepted count (Thm-3 feasibility)
  - cert_k_worst_group : accepted-error count K of the binding (worst) group
  - cert_coverage_lcb  : worst-case-over-Lambda accepted coverage LCB
  - test_risk / test_coverage : held-out deployment estimates
  - selected_gamma     : the risk buffer chosen on the proposal fold
  - certified

Cells: the three ResNet headline cells (5 seeds) PLUS the FedPD and FOOGD
detector runs (3 seeds, d in {0.5, 5}). Both alpha in {0.10, 0.20}.

Output: runs/T9_diagnostics.csv  (schema:
  backbone,d,alpha,seed,cert_risk_ucb_G2,cert_n_min_group,cert_k_worst_group,
  cert_coverage_lcb,test_risk,test_coverage,selected_gamma,certified)
Run: python experiments/fedcore/exp_alpha20_diagnostics.py   (CPU, no torch)

Diagnostic regeneration only; does NOT alter the CertifiedCoverage headline
numbers (produced by fedcore/aggregate/main.py).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from fedcore.certificate import cp_upper
from fedcore.certify import certify_best_gamma
from fedcore.grouping import make_group_map, repartition_trusted_pool
from fedcore.io_utils import atomic_write_csv
from fedcore.scores import scored_views
from fedcore.selector import choose_threshold, counts_per_client

# --- protocol constants ---
# R8 reconciliation: each cell is certified on the score head that DEFINES it, at
# that cell's published headline protocol, so this diagnostic table reproduces the
# headline numbers per row (not a uniform-MSP re-scoring that silently swaps score
# heads). Two regimes:
#   - ResNet MSP-backbone cells  -> MSP score, main.py grid {0.2,0.3,0.5,0.7,1.0}
#   - detector cells (FedPD/FOOGD) -> NATIVE open-set score (-sm), T8 grid
#     {0.5,0.7,1.0}. Certifying a detector on MSP measures an MSP-on-detector-
#     backbone baseline (== T8's FedAvg+MSP control), NOT the detector; that
#     conflation is exactly the T9-vs-T8 disagreement R8 was asked to resolve.
DELTA = 0.10
GAMMAS_MSP = (0.2, 0.3, 0.5, 0.7, 1.0)      # main.py headline (ResNet cells)
GAMMAS_NATIVE = (0.5, 0.7, 1.0)             # T8 detector-block protocol
MARGIN, BOX, CERT_FRAC, TEST_FRAC = 0.01, 0.15, 0.5, 0.2
G, SEED = 2, 0
ALPHAS = (0.10, 0.20)

# --- cells: (backbone label, d, filename pattern, seeds, score_mode) ---
# score_mode in {"msp", "native"}; "native" uses -sm from the detector npz.
CELLS = [
    # M2: headline baselines extended to 10 seeds (0..9); seeds 5..9 trained in manifest_M2.
    ("resnet18gn", "5",   "runs/cifar10_d5_resnet18gn_none0.0_seed{s}_logits.npz", tuple(range(10)), "msp"),
    ("resnet18gn", "0.5", "runs/cifar10_d0.5_resnet18gn_none0.0_seed{s}_logits.npz", tuple(range(10)), "msp"),
    ("resnet18",   "5",   "runs/cifar10_d5_resnet18_seed{s}_logits.npz", tuple(range(10)), "msp"),
    ("fedpd",      "5",   "runs/fedpd_cifar10_d5_seed{s}.npz", (0, 1, 2, 3, 4), "native"),
    ("fedpd",      "0.5", "runs/fedpd_cifar10_d0.5_seed{s}.npz", (0, 1, 2, 3, 4), "native"),
    ("foogd",      "5",   "runs/foogd_cifar10_d5_seed{s}.npz", (0, 1, 2, 3, 4), "native"),  # R3 seed extension (d5 only)
    ("foogd",      "0.5", "runs/foogd_cifar10_d0.5_seed{s}.npz", (0, 1, 2), "native"),
]


def _native_views(d, n_clients):
    """Fold views scored by the detector's NATIVE accept score (-sm), using the
    SAME test/cert/prop permutation as repartition_trusted_pool (seed=SEED)."""
    pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client", "sm")}
    n = len(pool["y_open"])
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_test, n_cert = int(round(n * TEST_FRAC)), int(round(n * CERT_FRAC))
    idx = {"test": perm[:n_test], "cert": perm[n_test:n_test + n_cert],
           "prop": perm[n_test + n_cert:]}
    views = {}
    for fn, ix in idx.items():
        logits = np.asarray(pool["logits"][ix], dtype=float)
        views[fn] = {"score": -np.asarray(pool["sm"][ix], dtype=float),
                     "pred": logits.argmax(-1), "y_open": pool["y_open"][ix],
                     "client": pool["client"][ix]}
    return views


def diagnostics_for_file(npz_path: str, alpha: float, score_mode: str = "msp",
                         delta: float = DELTA) -> dict:
    """Run the G=2 protocol on one file and return diagnostics.

    score_mode="msp": MSP on logits + GAMMAS_MSP (ResNet headline).
    score_mode="native": -sm detector score + GAMMAS_NATIVE (T8 detector block).
    """
    d = np.load(npz_path)
    n_clients = int(d["cert_client"].max()) + 1
    if score_mode == "native":
        views = _native_views(d, n_clients)
        score_name, gammas = "native", GAMMAS_NATIVE
    else:
        pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
                for k in ("logits", "y_open", "client")}
        parts = repartition_trusted_pool(pool, CERT_FRAC, TEST_FRAC, seed=SEED)
        views = {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                                  parts[fn]["client"], ["msp"])["msp"]
                 for fn in ("prop", "cert", "test")}
        score_name, gammas = "msp", GAMMAS_MSP

    gmap = make_group_map(n_clients, G)
    gviews = {}
    for fn in ("prop", "cert", "test"):
        v = dict(views[fn]); v["client"] = gmap[np.asarray(views[fn]["client"])]
        gviews[fn] = v

    res = certify_best_gamma(
        gviews["prop"], gviews["cert"], gviews["test"],
        score_name=score_name, gammas=gammas, alpha=alpha, delta=delta,
        n_clients=G, dirichlet_alpha=float("nan"), Lambda="box", box=BOX,
        seed=SEED, margin=MARGIN)

    # per-GROUP (A, K) under the chosen selector; identify the BINDING worst group.
    gamma_star = res["gamma_star"]
    sel = choose_threshold(gviews["prop"]["score"], gviews["prop"]["pred"],
                           gviews["prop"]["y_open"], gamma_star, alpha)
    A, K, _n = counts_per_client(
        gviews["cert"]["score"], gviews["cert"]["pred"], gviews["cert"]["y_open"],
        gviews["cert"]["client"], sel, G)
    eps = delta / (3.0 * G)   # box per-group CP level
    rbar = np.array([cp_upper(int(K[g]), int(A[g]), eps) if A[g] > 0 else np.inf
                     for g in range(G)])
    worst = int(np.argmax(rbar))

    return {
        "cert_risk_ucb_G2": float(res["cert_risk_ucb"]),
        "cert_n_min_group": int(A.min()),
        "cert_k_worst_group": int(K[worst]),
        "cert_coverage_lcb": float(res["cert_coverage_lcb"]),
        "test_risk": float(res["test_risk"]),
        "test_coverage": float(res["test_coverage"]),
        "selected_gamma": float(gamma_star),
        "certified": int(bool(res["certified"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/T9_diagnostics.csv")
    ap.add_argument("--delta", type=float, default=DELTA,
                    help="total budget; pass delta/2 (0.05) for the simultaneous "
                         "risk+coverage split -> runs/T9_diagnostics_simul.csv")
    args = ap.parse_args()
    delta = args.delta

    fields = ["backbone", "d", "alpha", "seed", "cert_risk_ucb_G2", "cert_n_min_group",
              "cert_k_worst_group", "cert_coverage_lcb", "test_risk", "test_coverage",
              "selected_gamma", "certified"]
    rows = []
    print(f"delta={delta} G={G} score=per-cell(msp|native) cert_frac={CERT_FRAC} box={BOX} margin={MARGIN}")
    print(f"{'cell':>16} {'alpha':>5} {'seed':>4} {'ucb_G2':>8} {'n_min':>6} {'k_worst':>7} "
          f"{'cov_lcb':>8} {'t_risk':>7} {'t_cov':>7} {'gamma':>6} {'cert':>4}")
    print("-" * 92)
    for backbone, dd, pat, seeds, score_mode in CELLS:
        for alpha in ALPHAS:
            for s in seeds:
                path = pat.format(s=s)
                if not os.path.exists(path):
                    print(f"{backbone+'/d'+dd:>16} {alpha:>5.2f} {s:>4}   MISSING: {path}")
                    continue
                r = diagnostics_for_file(path, alpha, score_mode=score_mode, delta=delta)
                rows.append({"backbone": backbone, "d": dd, "alpha": alpha, "seed": s, **r})
                ucb = r["cert_risk_ucb_G2"]
                ucb_s = f"{ucb:.4f}" if np.isfinite(ucb) else "inf"
                print(f"{backbone+'/d'+dd:>16} {alpha:>5.2f} {s:>4} {ucb_s:>8} "
                      f"{r['cert_n_min_group']:>6} {r['cert_k_worst_group']:>7} "
                      f"{r['cert_coverage_lcb']:>8.4f} {r['test_risk']:>7.4f} "
                      f"{r['test_coverage']:>7.4f} {r['selected_gamma']:>6.2f} {r['certified']:>4}")

    atomic_write_csv(args.out, fields, rows)
    print(f"\nsaved {args.out}  ({len(rows)} rows)")

    # per-cell x alpha medians (the numbers Table 5 panels (a)/(b) would show)
    print("\n=== per-cell medians (Table 5 diagnostics) ===")
    print(f"{'cell':>16} {'alpha':>5} {'med ucb_G2':>11} {'med n_min':>10} {'med k_worst':>12} "
          f"{'mean cov_lcb(cert)':>18} {'med t_cov':>10} {'n_cert':>7}")
    for backbone, dd, _pat, _seeds, _mode in CELLS:
        for alpha in ALPHAS:
            cell = [r for r in rows if r["backbone"] == backbone and r["d"] == dd and r["alpha"] == alpha]
            if not cell:
                continue
            ucbs = [r["cert_risk_ucb_G2"] for r in cell if np.isfinite(r["cert_risk_ucb_G2"])]
            med_ucb = float(np.median(ucbs)) if ucbs else float("inf")
            med_nmin = float(np.median([r["cert_n_min_group"] for r in cell]))
            med_kw = float(np.median([r["cert_k_worst_group"] for r in cell]))
            cov_cert = [r["cert_coverage_lcb"] for r in cell if r["certified"]]
            mean_cov = float(np.mean(cov_cert)) if cov_cert else 0.0
            med_tcov = float(np.median([r["test_coverage"] for r in cell]))
            n_cert = sum(r["certified"] for r in cell)
            med_ucb_s = f"{med_ucb:.4f}" if np.isfinite(med_ucb) else "inf"
            print(f"{backbone+'/d'+dd:>16} {alpha:>5.2f} {med_ucb_s:>11} {med_nmin:>10.1f} "
                  f"{med_kw:>12.1f} {mean_cov:>18.4f} {med_tcov:>10.4f} {n_cert:>4}/{len(cell)}")


if __name__ == "__main__":
    main()
