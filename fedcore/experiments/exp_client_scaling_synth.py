"""M8: synthetic client-scaling law beyond the GPU-feasible range (J in {10,20,50}).

Complements Figure 4d (the real CIFAR client-scaling, J in {10,20}) by pushing J
synthetically to 50 at a FIXED total audit budget, and showing the grouped
certificate's recovery law: the full per-client stratification (G=J) collapses as J
grows (each per-client Clopper-Pearson interval is at Bonferroni level delta/J and the
per-client accepted count A_j = a * N_total/J falls below the Theorem 3 feasibility
floor ln(G/delta)/(-ln(1-alpha))), while contiguous grouping (G in {10,5,2}) pools the
counts back above the floor and restores certification.

Homogeneous clients (a, r fixed, r < alpha) isolate the federation/grouping price from
heterogeneity. Grouping rule (pre-declared, PUBLIC, data-independent): client c ->
group c*G//J (contiguous balanced blocks, fedcore.grouping.make_group_map). The grouped
certificate sums per-client (A_j, K_j) within each group and applies the conditional
Theorem-1 simplex certificate over the G group-units at eps = delta/G.

Writes a NEW file runs/client_scaling_synth.csv (schema below) -- does NOT touch the
real-data runs/client_scaling.csv (different producer/schema).

CSV columns:
  J,G,n_per_client,expected_A_per_group,theorem3_floor_G,alpha,delta,
  cert_coverage_mean,frac_certified,cert_ucb_median,trials
('theorem3_floor_G' = per-group accepted-count feasibility floor ln(G/delta)/(-ln(1-alpha)),
 the Theorem 3 feasibility law in the current manuscript.)

Run: python experiments/fedcore/exp_client_scaling_synth.py   (CPU, no torch)
"""

from __future__ import annotations

import numpy as np

from fedcore.certificate import conditional_risk_certificate, cp_lower, thm2_floor
from fedcore.data.clients import ClientPopulation, draw_counts
from fedcore.grouping import make_group_map
from fedcore.io_utils import atomic_write_csv

ALPHA, DELTA = 0.10, 0.10
A_ACC, R_TRUE = 0.60, 0.05           # per-client accept prob / true accepted-risk (r<alpha)
N_TOTAL = 1200                       # FIXED total audit budget, split equally across J
JS = (10, 20, 50)
G_CANDIDATES = ("J", 10, 5, 2)       # 'J' = full per-client stratification (worst case)
N_MC = 2000
LAMBDA = "simplex"
OUT = "runs/client_scaling_synth.csv"
FIELDS = ["J", "G", "n_per_client", "expected_A_per_group", "theorem3_floor_G", "alpha",
          "delta", "cert_coverage_mean", "frac_certified", "cert_ucb_median", "trials"]


def _groupings(J):
    gs = []
    for g in G_CANDIDATES:
        gv = J if g == "J" else int(g)
        if 1 <= gv <= J and gv not in gs:
            gs.append(gv)
    return gs


def _coverage_lcb(Ag, ng, G):
    """Worst-group accepted coverage LCB = min_g U-(A_g, n_g; delta/2G)."""
    eps = DELTA / (2.0 * G)
    return float(min(cp_lower(int(Ag[g]), int(ng[g]), eps) for g in range(G)))


def _aggregate_to_groups(A, K, n_arr, gmap, G):
    Ag = np.zeros(G, dtype=int); Kg = np.zeros(G, dtype=int); ng = np.zeros(G, dtype=int)
    for j in range(len(A)):
        g = gmap[j]
        Ag[g] += A[j]; Kg[g] += K[j]; ng[g] += n_arr[j]
    return Ag, Kg, ng


def scaling_cell(J, G, rng):
    n_j = N_TOTAL // J
    n_arr = np.full(J, n_j, dtype=int)
    pop = ClientPopulation(a=np.full(J, A_ACC), r=np.full(J, R_TRUE))
    gmap = make_group_map(J, G)
    covs, flags, ucbs = [], [], []
    for _ in range(N_MC):
        A, K = draw_counts(pop, n_arr, rng)
        Ag, Kg, ng = _aggregate_to_groups(A, K, n_arr, gmap, G)
        cert = conditional_risk_certificate(Ag, Kg, ng, DELTA, Lambda=LAMBDA)
        certified = bool(cert.feasible and cert.U <= ALPHA)
        covs.append(_coverage_lcb(Ag, ng, G) if certified else 0.0)
        flags.append(certified)
        ucbs.append(cert.U if np.isfinite(cert.U) else np.nan)
    exp_A_per_group = A_ACC * n_j * (J / G)          # expected accepted points per group
    return {
        "J": J, "G": G, "n_per_client": n_j,
        "expected_A_per_group": round(float(exp_A_per_group), 1),
        "theorem3_floor_G": round(float(thm2_floor(G, DELTA, ALPHA)), 1),
        "alpha": ALPHA, "delta": DELTA,
        "cert_coverage_mean": round(float(np.mean(covs)), 4),
        "frac_certified": round(float(np.mean(flags)), 4),
        "cert_ucb_median": round(float(np.nanmedian(ucbs)), 4),
        "trials": N_MC,
    }


def main():
    rng = np.random.default_rng(0)
    print(f"M8 synthetic client scaling: N_total={N_TOTAL} split across J, a={A_ACC}, "
          f"r={R_TRUE}, alpha={ALPHA}, delta={DELTA}, T={N_MC}\n")
    print(f"{'J':>3} {'G':>3} {'n/cl':>5} {'E[A]/grp':>8} {'T3floor':>7} "
          f"{'CertCov':>8} {'frac_cert':>9} {'ucb_med':>8}")
    print("-" * 60)
    rows = []
    for J in JS:
        for G in _groupings(J):
            r = scaling_cell(J, G, rng)
            rows.append(r)
            print(f"{r['J']:>3} {r['G']:>3} {r['n_per_client']:>5} "
                  f"{r['expected_A_per_group']:>8} {r['theorem3_floor_G']:>7} "
                  f"{r['cert_coverage_mean']:>8.3f} {r['frac_certified']:>9.3f} "
                  f"{r['cert_ucb_median']:>8.3f}")
        print("-" * 60)
    atomic_write_csv(OUT, FIELDS, rows)
    print(f"saved {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
