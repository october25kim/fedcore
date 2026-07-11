"""Resampling validity experiment (E1): empirical coverage of the certificate
over calibration randomness, on REAL federated logits.

Rationale. The certificate's guarantee is a probability over the draw of the
certification fold, not over training seeds. Five training seeds cannot
estimate a false-certificate rate; resampling the audit fold at a FIXED model
can. This script treats each run's held-out pool (certification + test folds)
as the ground-truth population, re-draws per-client audit folds B times, and
counts violations of the stratified certificate (Theorem 1 / grouped variant)
against the population worst-stratum conditional risk
sup_lambda R_sel(lambda) = max_g r_g.

Grouping rule (fixed a priori, matching A6): clients are merged into G
contiguous groups via np.array_split(range(J), G).

CPU-only: consumes runs/*_logits.npz produced by run_cifar.py.

Usage:
  python exp_resampling_validity.py [shard_start] [shard_end]

Outputs:
  runs/resampling_validity[_s_e].csv   per (run, alpha, G, gamma) tallies
  stdout aggregate with the Clopper-Pearson bound on the violation rate
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

from fedcore.certificate.cp import cp_upper
from fedcore.certify import conditional_risk_certificate
from fedcore.scores import compute_score
from fedcore.selector import choose_threshold, open_set_error

B = 1000                       # resamples per run (shared across configs)
DELTA = 0.10
ALPHAS = (0.10, 0.20)
GROUPS = (5, 2)                # 5 = per-client stratified; 2 = headline grouping
GAMMAS = (0.5, 0.7, 1.0)
SCORE = "msp"                  # fixed-score protocol (headline)
SEED = 0


def load_run(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    out = {}
    for fold in ("prop", "cert", "test"):
        logits = z[f"{fold}_logits"]
        out[fold] = {
            "score": compute_score(SCORE, logits),
            "pred": logits.argmax(axis=1),
            "y_open": z[f"{fold}_y_open"],
            "client": z[f"{fold}_client"],
        }
    return out


def group_of_client(n_clients: int, G: int) -> np.ndarray:
    g = np.empty(n_clients, dtype=int)
    for gi, block in enumerate(np.array_split(np.arange(n_clients), G)):
        g[block] = gi
    return g


def worst_group_risk(err: np.ndarray, acc: np.ndarray, grp: np.ndarray,
                     G: int) -> float:
    worst = 0.0
    for g in range(G):
        m = (grp == g) & acc
        if m.sum() > 0:
            worst = max(worst, float(err[m].mean()))
    return worst


def group_counts(err: np.ndarray, acc: np.ndarray, grp: np.ndarray,
                 G: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.zeros(G, dtype=int)
    K = np.zeros(G, dtype=int)
    n = np.zeros(G, dtype=int)
    for g in range(G):
        m = grp == g
        n[g] = int(m.sum())
        a = m & acc
        A[g] = int(a.sum())
        K[g] = int(err[a].sum())
    return A, K, n


def main() -> None:
    rng = np.random.default_rng(SEED)
    paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..",
                                          "runs", "*_logits.npz")))
    if not paths:
        paths = sorted(glob.glob("runs/*_logits.npz"))
    if len(sys.argv) >= 3:
        paths = paths[int(sys.argv[1]):int(sys.argv[2])]

    rows = []
    tot_dep = tot_viol = 0
    for path in paths:
        run = os.path.basename(path).replace("_logits.npz", "")
        d = load_run(path)
        n_clients = int(max(d["cert"]["client"].max(),
                            d["test"]["client"].max())) + 1
        pool = {k: np.concatenate([d["cert"][k], d["test"][k]])
                for k in ("score", "pred", "y_open", "client")}
        pool_err = open_set_error(pool["pred"], pool["y_open"])
        n_j = np.array([(d["cert"]["client"] == j).sum()
                        for j in range(n_clients)])
        client_idx = [np.flatnonzero(pool["client"] == j)
                      for j in range(n_clients)]
        # one shared set of B bootstrap audit folds per run
        boots = [np.concatenate([rng.choice(client_idx[j], size=n_j[j],
                                            replace=True)
                                 for j in range(n_clients) if n_j[j] > 0])
                 for _ in range(B)]

        for alpha in ALPHAS:
            for gamma in GAMMAS:
                sel = choose_threshold(d["prop"]["score"], d["prop"]["pred"],
                                       d["prop"]["y_open"],
                                       gamma=gamma, alpha=alpha)
                if not sel.feasible:
                    for G in GROUPS:
                        rows.append((run, alpha, G, gamma,
                                     "infeasible-proposal", 0, 0, np.nan))
                    continue
                pool_acc = pool["score"] >= sel.threshold
                for G in GROUPS:
                    grp_c = group_of_client(n_clients, G)
                    pool_grp = grp_c[pool["client"]]
                    true_worst = worst_group_risk(pool_err, pool_acc,
                                                  pool_grp, G)
                    n_dep = n_viol = 0
                    for idx in boots:
                        acc = pool_acc[idx]
                        err = pool_err[idx]
                        grp = pool_grp[idx]
                        A, K, n = group_counts(err, acc, grp, G)
                        res = conditional_risk_certificate(
                            A, K, n, DELTA, Lambda="simplex")
                        if res.U <= alpha:
                            n_dep += 1
                            if true_worst > alpha:
                                n_viol += 1
                    tot_dep += n_dep
                    tot_viol += n_viol
                    rows.append((run, alpha, G, gamma, "ok", n_dep, n_viol,
                                 round(true_worst, 4)))
        print(f"{run}: done")

    suffix = f"_{sys.argv[1]}_{sys.argv[2]}" if len(sys.argv) >= 3 else ""
    out = os.path.join("runs", f"resampling_validity{suffix}.csv")
    with open(out, "w") as f:
        f.write("run,alpha,G,gamma,status,n_deploy_of_B,n_violation,"
                "true_worst_group_risk\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    ucb = cp_upper(tot_viol, max(tot_dep, 1), 0.05)
    print("\n=== aggregate (this shard) ===")
    print(f"deployments: {tot_dep}  violations: {tot_viol}  "
          f"CP95 UCB on violation rate: {ucb:.5f}  (delta = {DELTA})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
