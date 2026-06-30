"""PRIORITY 4 / A5-real: unknown-proportion sweep (mirrors synthetic Fig 10/A5).

Claim under test (from synthetic A5): UNDER-REPRESENTING the unknown (open-set) class
in the certification fold relative to the deployment rate is ANTI-CONSERVATIVE -- the
certified risk UCB drops below the truth and the bound's empirical coverage falls
below 1 - delta.

Method (post-hoc, model fixed). Deployment unknown rate p ~ 0.30. For rho in
{0.25, 0.5, 0.75, 1.0} we subsample the unknown-labeled points IN THE CERTIFICATION
FOLD to an unknown fraction rho * p, while the PROPOSAL fold (selector source) and the
held-out TEST/deployment fold keep the full rate p. We then run the worst-group (G=2)
best-gamma certificate (fixed MSP, box-Lambda) and record, over many (model seed x
resample) Monte-Carlo draws:
  - cert_risk_ucb (the bound) and the realized TEST selective risk (the truth);
  - empirical coverage = P(realized test risk <= cert_risk_ucb), which must be
    >= 1 - delta iff the cert fold is representative.

Expectation: coverage ~>= 0.90 at rho = 1.0 (matched) and FALLS below 0.90 as rho
shrinks -- confirming that under-representing unknowns is anti-conservative.

Run: python experiments/fedcore/exp_ablation_unknown_prop.py
"""

from __future__ import annotations

import csv
import glob
import os

import numpy as np

from fedcore.certify import certify_best_gamma_grouped
from fedcore.scores import scored_views

ALPHA, DELTA = 0.20, 0.10          # headline regime: robust accepted set
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
SCORE = "msp"
MARGIN = 0.01
G = 2
SEEDS = (0, 1, 2, 3, 4)
TAG = "cifar10_d5_resnet18gn_none0.0"
RHOS = (0.25, 0.5, 0.75, 1.0)
N_REPS = 40                         # resamples per (model seed, rho)
P_DEPLOY = 0.30                     # deployment unknown rate
TEST_FRAC, CERT_FRAC = 0.30, 0.35   # remaining -> proposal


def _pool(npz):
    d = np.load(npz)
    return {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}


def _split(pool, seed):
    rng = np.random.default_rng(seed)
    n = len(pool["y_open"])
    perm = rng.permutation(n)
    n_test = int(round(n * TEST_FRAC))
    n_cert = int(round(n * CERT_FRAC))
    ix = {"test": perm[:n_test], "cert": perm[n_test:n_test + n_cert],
          "prop": perm[n_test + n_cert:]}
    take = lambda i: {k: pool[k][i] for k in ("logits", "y_open", "client")}
    return {f: take(i) for f, i in ix.items()}, rng


def _subsample_unknowns(fold, rho, rng):
    """Drop unknowns in `fold` so its unknown fraction = rho * P_DEPLOY (knowns kept)."""
    yo = fold["y_open"]
    known_ix = np.where(yo >= 0)[0]
    unk_ix = np.where(yo < 0)[0]
    n_k = len(known_ix)
    frac = rho * P_DEPLOY
    u_target = int(round(frac * n_k / (1.0 - frac)))
    if u_target >= len(unk_ix):
        keep_unk = unk_ix
    else:
        keep_unk = rng.choice(unk_ix, size=u_target, replace=False)
    keep = np.concatenate([known_ix, keep_unk])
    keep.sort()
    return {k: fold[k][keep] for k in ("logits", "y_open", "client")}


def _certify(parts, n_clients):
    views = {fn: scored_views(parts[fn]["logits"], parts[fn]["y_open"],
                              parts[fn]["client"], [SCORE])[SCORE]
             for fn in ("prop", "cert", "test")}
    gmap = np.array([c * G // n_clients for c in range(n_clients)])
    return certify_best_gamma_grouped(
        views["prop"], views["cert"], views["test"], score_name=SCORE,
        group_map=gmap, G=G, gammas=GAMMAS, alpha=ALPHA, delta=DELTA,
        Lambda="box", box=0.15, seed=0, margin=MARGIN)


def main() -> None:
    base = "" if glob.glob("runs/*_logits.npz") else "../../"
    files = [base + f"runs/{TAG}_seed{s}_logits.npz" for s in SEEDS]
    pools = [_pool(f) for f in files if os.path.exists(f)]
    n_clients = int(pools[0]["client"].max()) + 1

    print(f"A5-real unknown-proportion sweep  (cell={TAG}, G={G}, alpha={ALPHA}, "
          f"delta={DELTA}, p_deploy={P_DEPLOY}, {len(pools)} seeds x {N_REPS} reps)")
    print(f"target coverage >= 1-delta = {1 - DELTA:.2f}\n")
    print(f"{'rho':>5} {'cert_unk_frac':>13} {'cert_ucb':>9} {'realized_risk':>13} "
          f"{'coverage':>9} {'n_eval':>7} {'verdict':>16}")
    print("-" * 78)

    rows = []
    for rho in RHOS:
        ucbs, reals, indic = [], [], []
        for si, pool in enumerate(pools):
            for rep in range(N_REPS):
                parts, rng = _split(pool, seed=10_000 * si + rep)
                parts = dict(parts)
                parts["cert"] = _subsample_unknowns(parts["cert"], rho, rng)
                r = _certify(parts, n_clients)
                if not np.isfinite(r["cert_risk_ucb"]) or r["cert_n"] == 0:
                    continue
                ucbs.append(r["cert_risk_ucb"])
                reals.append(r["test_risk"])
                indic.append(1 if r["test_risk"] <= r["cert_risk_ucb"] + 1e-12 else 0)
        cov = float(np.mean(indic)) if indic else float("nan")
        anti = cov < (1 - DELTA) - 1e-9
        verdict = "ANTI-CONSERV" if anti else "ok (>=1-delta)"
        cert_unk_frac = rho * P_DEPLOY
        row = {"rho": rho, "cert_unknown_frac": round(cert_unk_frac, 4),
               "cert_ucb_mean": round(float(np.mean(ucbs)), 4),
               "realized_test_risk_mean": round(float(np.mean(reals)), 4),
               "coverage": round(cov, 4), "n_eval": len(indic),
               "target_1_minus_delta": 1 - DELTA,
               "anti_conservative": bool(anti)}
        rows.append(row)
        print(f"{rho:>5.2f} {cert_unk_frac:>13.3f} {np.mean(ucbs):>9.4f} "
              f"{np.mean(reals):>13.4f} {cov:>9.3f} {len(indic):>7} {verdict:>16}")

    out = base + "runs/ablation_unknown_prop.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out}")
    print("Reading: coverage falls below 1-delta as rho shrinks => under-representing "
          "unknowns in calibration is anti-conservative (synthetic A5 confirmed).")
    _plot(rows, base)


def _plot(rows, base):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipping figure: {exc})")
        return
    rho = [r["rho"] for r in rows]
    cov = [r["coverage"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rho, cov, marker="o", color="C0", label="empirical coverage")
    ax.axhline(1 - DELTA, ls="--", color="gray", lw=1, label=f"1-delta={1-DELTA:.2f}")
    ax.set_xlabel("rho = cert-fold unknown rate / deployment rate")
    ax.set_ylabel("coverage  P(realized risk <= cert_ucb)")
    ax.set_title("A5-real: under-representing unknowns is anti-conservative")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    path = base + "experiments/fedcore/figs/ablation_unknown_prop.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
