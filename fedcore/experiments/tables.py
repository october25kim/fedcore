"""STAGE 1: generate paper tables T1-T7 as CSV from existing data (CPU).

T1 main results        (from runs/agg_main.csv)
T2 certificate efficiency (conditional vs mass-ratio vs box vs pooled; median U)
T3 necessity           (naive-emp / pooled-CP / Fed-CORE unsafe-deploy at boundary)
T5 score-agnostic      (4 scores: test_risk, CertCov; all valid)
T6 privacy taxonomy    (pooled/stratified/grouped: released stats, leakage, scope)
T7 self-train delta/T  (simultaneous unsafe rate with vs without delta/T)
(T4 superiority is produced by exp_superiority.py -> runs/T4.csv.)

Run: ``python experiments/fedcore/tables.py``  (CPU, no torch)
"""

from __future__ import annotations

import csv
import glob
import os

import numpy as np

from fedcore.certificate import (conditional_risk_certificate, pooled_cp,
                          stratified_certificate, true_selective_risk)
from fedcore.data.clients import ClientPopulation, draw_counts, heterogeneous_population
from fedcore.certify import certify_grid
from fedcore.experiments.run_smoke import SmokeSpec, generate_smoke
from fedcore.experiments.run_selftrain_smoke import simultaneous_unsafe_rate
from fedcore.scores import scored_views

BASE = "" if glob.glob("runs/*.csv") else "../../"
OUT = BASE + "runs/"
DELTA = 0.10


def _write(name, header, rows):
    path = OUT + name
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"saved {path}  ({len(rows)} rows)")


def T1():
    src = OUT + "agg_main.csv"
    if not glob.glob(src):
        print("  (T1 skipped: run aggregate.py first)")
        return
    rows = list(csv.DictReader(open(src)))
    out = [[r["dataset"], r["backbone"], r["d"], r["noise"], r["n_seeds"],
            f'{float(r["CertCovG2_mean"]):.3f}+/-{float(r["CertCovG2_std"]):.3f}',
            r.get("n_pass_G2", "-"),
            f'{float(r["certucbG2_median"]):.3f}',
            f'{float(r["test_risk_mean"]):.3f}'] for r in rows]
    _write("T1_main.csv", ["dataset", "backbone", "d", "noise", "n_seeds",
                           "CertCov@0.1(G=2,fixed-MSP,cf=0.5)", "n_pass_G2",
                           "median_cert_ucb(G=2)", "test_risk"], out)


def T2():
    pop = heterogeneous_population()
    J = pop.J
    n = np.full(J, 300)
    rows = []
    for r_bad in (0.10, 0.20, 0.30):
        pop.r[-1] = r_bad
        rng = np.random.default_rng(0)
        up, ub, us, um = [], [], [], []
        for t in range(1000):
            A, K = draw_counts(pop, n, rng)
            up.append(pooled_cp(A, K, DELTA))
            ub.append(conditional_risk_certificate(A, K, n, DELTA, Lambda="box", box=0.10, n_lam_samples=48, seed=t).U)
            us.append(conditional_risk_certificate(A, K, n, DELTA, Lambda="simplex").U)
            um.append(stratified_certificate(A, K, n, DELTA, Lambda="simplex").U)
        rows.append([f"{r_bad:.2f}", f"{np.median(up):.3f} (matched-only)",
                     f"{np.median(ub):.3f}", f"{np.median(us):.3f}", f"{np.median(um):.3f}"])
    _write("T2_efficiency.csv",
           ["r_bad", "pooled (Prop3)", "cond-box (Thm1')", "cond-simplex (Thm1)", "mass-ratio (AppC)"], rows)


def T3():
    alpha = 0.05
    n = np.array([300, 300, 300, 300, 400]); J = len(n)
    lam = n / n.sum()
    a = np.array([0.7] * 4 + [0.5])
    rows = []
    for r_bad in (0.178, 0.30):  # boundary and clearly-unsafe
        r = np.array([0.02] * 4 + [r_bad])
        pop = ClientPopulation(a=a, r=r)
        R = true_selective_risk(a, r, lam)
        rng = np.random.default_rng(0)
        dn = dp = df = 0
        N = 3000
        for _ in range(N):
            A, K = draw_counts(pop, n, rng)
            tA, tK = int(A.sum()), int(K.sum())
            dn += (tK / tA if tA else 0) <= alpha
            dp += pooled_cp(A, K, DELTA) <= alpha
            df += conditional_risk_certificate(A, K, n, DELTA, Lambda="known", lam=lam).U <= alpha
        rows.append([f"{r_bad:.3f}", f"{R:.4f}", str(R > alpha),
                     f"{dn/N:.3f}", f"{dp/N:.3f}", f"{df/N:.3f}"])
    _write("T3_necessity.csv",
           ["r_bad", "R_sel", "unsafe(R>a)", "naive-emp", "pooled-CP", "Fed-CORE"], rows)


def T5():
    spec = SmokeSpec(); scores = ("msp", "neg_entropy", "margin", "energy")
    data = generate_smoke(spec)
    views = {fn: scored_views(data[fn]["logits"], data[fn]["y_open"], data[fn]["client"], list(scores))
             for fn in ("prop", "cert", "test")}
    grid = certify_grid(views["prop"], views["cert"], views["test"], scores=scores, gammas=(0.5, 0.7, 1.0),
                        alpha=0.10, delta=0.10, Lambdas=("box",), n_clients=spec.n_clients,
                        dirichlet_alpha=float("nan"), box=0.10, seed=0)
    rows = []
    for s in scores:
        cert = [r for r in grid if r["score_name"] == s and r["certified"]]
        if cert:
            b = max(cert, key=lambda r: r["cert_coverage_lcb"])
            rows.append([s, f'{b["cert_coverage_lcb"]:.3f}', f'{b["test_risk"]:.3f}', str(b["test_risk"] <= 0.10)])
        else:
            rows.append([s, "0.000", "-", "n/a"])
    _write("T5_score_agnostic.csv", ["score", "CertCov@0.1", "test_risk", "valid(test<=a)"], rows)


def T6():
    rows = [
        ["pooled (Prop 3)", "sum K, sum A (2 scalars)", "lowest", "matched-mixture i.i.d. ONLY"],
        ["grouped-stratified (sec 4.4)", "per-group (A_g,K_g), G groups >=k clients", "medium (secure-agg within group)", "worst-group, all mixtures"],
        ["stratified (Thm 1/1')", "per-client (A_j,K_j)", "highest", "worst-client, all mixtures"],
    ]
    _write("T6_privacy.csv", ["certificate", "released statistics", "leakage", "validity scope"], rows)


def T7():
    rng = np.random.default_rng(0)
    sim_split, pr_split = simultaneous_unsafe_rate(0.10 / 5, rng)
    rng = np.random.default_rng(0)
    sim_no, pr_no = simultaneous_unsafe_rate(0.10, rng)
    rows = [["with delta/T", f"{0.10/5:.3f}", f"{pr_split:.4f}", f"{sim_split:.4f}", str(sim_split <= 0.10)],
            ["without (/T)", "0.100", f"{pr_no:.4f}", f"{sim_no:.4f}", str(sim_no <= 0.10)]]
    _write("T7_selftrain_validity.csv",
           ["scheme", "delta_round", "per-round fail", "simultaneous unsafe", "<=delta"], rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    T1(); T2(); T3(); T5(); T6(); T7()
    print("\n(T4 superiority: runs/T4.csv from exp_superiority.py)")


if __name__ == "__main__":
    main()
