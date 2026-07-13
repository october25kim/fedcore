"""M1: grouped-certificate validity stress -- QUOTA vs IID-GROUP-MIXTURE audit sampling.

Directly supports the revised Proposition 3: the grouped certificate is EXACT only
under i.i.d. group-mixture audit sampling (Assumption A6); our per-client quota
sampling is a declared caveat. This script quantifies both:

  * IID-GROUP-MIXTURE sampling: each audit point of group g is drawn i.i.d. from the
    group's deployment mixture P_g. Then K_g | A_g ~ Bin(A_g, r_g^dep) EXACTLY, so the
    per-group Clopper-Pearson upper bound covers the group's deployment risk at
    1 - delta/G. Acceptance target: coverage >= 1 - delta everywhere.
  * QUOTA sampling: each client contributes a FIXED n_per_client audit points. The
    summed group count reflects the *quota* (uniform-over-clients) mixture, which can
    differ from the deployment mixture P_g. Under within-group risk spread + a
    non-uniform deployment mixture this biases the pooled binomial low -> under-coverage.
    Its worst cell quantifies the caveat.

============================ PRE-DECLARED POPULATIONS (SYNTHETIC) ============================
Fixed BEFORE running; no post-hoc selection of favorable configs.
  * J = 6 clients, G = 2 contiguous groups: clients {0,1,2} -> group 0, {3,4,5} -> group 1
    (make_group_map(6, 2) = [0,0,0,1,1,1]).
  * Both groups use the SAME within-group risk triple in a given cell (symmetric); we
    report PER-GROUP coverage pooled over both groups x trials (avoids worst-group
    max-masking, isolating the within-group pooling validity question).
  * Per-client acceptance a_j = 0.5 (homogeneous) so the sampling gap is driven purely
    by the risk spread, not by acceptance heterogeneity.
  * Within-group DEPLOYMENT mixture weights w = (0.6, 0.3, 0.1), FIXED across all cells
    ("one dominant client"; realistic FL client-size skew). Risks are assigned to
    clients in DESCENDING order (largest risk -> 0.6-share client): the pre-declared
    "dominant client is riskiest" arrangement. This is the arrangement quota
    under-samples, so it is the honest stress on quota (not cherry-picked after the fact).
  * Group deployment risk (target the certificate must cover):
        r_g^dep = sum_j w_j a_j r_j / sum_j w_j a_j = sum_j w_j r_j   (a_j const)
  * Swept within-group risk spread (the three triples, sorted descending):
        (0.02, 0.02, 0.02)  -> no spread   (deployment risk 0.020, quota mean 0.0200)
        (0.06, 0.02, 0.01)  -> mild spread  (deployment risk 0.043, quota mean 0.0300)
        (0.20, 0.02, 0.01)  -> heavy spread (deployment risk 0.127, quota mean 0.0767)
    Under NO spread the quota and deployment mixtures coincide -> both modes must cover.
  * Audit budget swept: n_per_client in {150, 600}. Larger n shrinks CP slack, so any
    quota bias becomes MORE visible (the caveat sharpens with data, honestly reported).
  * delta = 0.10, per-group CP level eps = delta/G = 0.05. T = 5000 trials per cell.

============================ REAL LOGITS (stored GN d in {0.5, 5}) ============================
Re-run the same two sampling modes on the frozen resnet18gn (GroupNorm) CIFAR-10 logits
for G=2 (contiguous grouping of the 5 clients: [0,0,1,1,1] via make_group_map(5,2)):
  * The FULL cert-fold pool defines each point's accept/error under a fixed MSP selector
    (threshold chosen on the disjoint proposal fold at gamma=1.0, alpha=0.20 -- NEVER on
    test/cert labels). The TRUE per-group deployment risk r_g^dep is the pool's accepted
    error rate. Audit points are resampled (with replacement) from the pool either by
    QUOTA (fixed n_per_client per client) or IID-GROUP (each draw's client ~ the pool's
    within-group client shares). Report the coverage gap between the two modes.

CSV: runs/grouped_validity_stress.csv
  columns: study,sampling,G,within_group_risk_spread,n_per_client,coverage,mean_ucb,trials
  ('within_group_risk_spread' = the risk triple for SYNTHETIC, or "d=<val>" for REAL.)

Run: python -m fedcore.experiments.exp_grouped_validity_stress   (CPU, no torch)
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np

from fedcore.certificate import cp_upper
from fedcore.data.clients import ClientPopulation, draw_counts
from fedcore.grouping import make_group_map
from fedcore.io_utils import atomic_write_csv

OUT = "runs/grouped_validity_stress.csv"
FIELDS = ["study", "sampling", "G", "within_group_risk_spread", "n_per_client",
          "coverage", "mean_ucb", "trials"]

DELTA = 0.10
T_SYNTH = 5000
G_SYNTH = 2
J_SYNTH = 6
A_CONST = 0.5
W_DEP = np.array([0.6, 0.3, 0.1])                    # within-group deployment shares (fixed)
SPREADS = [                                          # per-client risk triples (sorted desc)
    (0.02, 0.02, 0.02),
    (0.06, 0.02, 0.01),
    (0.20, 0.02, 0.01),
]
N_PER_CLIENT_SYNTH = [150, 600]


def _spread_label(triple):
    return "|".join(f"{x:g}" for x in triple)


def _group_dep_risk(triple):
    """Deployment-mixture accepted-error rate for one group (a_j constant => = sum w_j r_j)."""
    r = np.asarray(triple, dtype=float)
    return float(np.sum(W_DEP * A_CONST * r) / np.sum(W_DEP * A_CONST))


# --------------------------------------------------------------------------- #
# SYNTHETIC study
# --------------------------------------------------------------------------- #
def _iid_group_counts(triple, n_group, rng):
    """One group's (A_g, K_g) under IID-GROUP-MIXTURE sampling of n_group audit points."""
    r = np.asarray(triple, dtype=float)
    clients = rng.choice(len(r), size=n_group, p=W_DEP)      # each draw's client ~ P_g
    accept = rng.random(n_group) < A_CONST
    A_g = int(accept.sum())
    # among accepted points, wrong w.p. r_{its client}
    r_acc = r[clients[accept]]
    K_g = int((rng.random(A_g) < r_acc).sum()) if A_g else 0
    return A_g, K_g


def run_synthetic(rng):
    rows = []
    group_map = make_group_map(J_SYNTH, G_SYNTH)             # [0,0,0,1,1,1]
    eps = DELTA / G_SYNTH
    for triple in SPREADS:
        target = _group_dep_risk(triple)                    # r_g^dep (same for both groups)
        # per-client risk vector: both groups identical, clients within a group in desc order
        r_vec = np.array(list(triple) * G_SYNTH, dtype=float)
        a_vec = np.full(J_SYNTH, A_CONST)
        for n_pc in N_PER_CLIENT_SYNTH:
            for mode in ("quota", "iid_group"):
                covered = 0
                ucb_sum = 0.0
                obs = 0
                for _ in range(T_SYNTH):
                    if mode == "quota":
                        pop = ClientPopulation(a=a_vec, r=r_vec)
                        A, K = draw_counts(pop, n_pc, rng)   # fixed n_pc per client
                        for g in range(G_SYNTH):
                            sel = group_map == g
                            Ag, Kg = int(A[sel].sum()), int(K[sel].sum())
                            rbar = cp_upper(Kg, Ag, eps)
                            covered += int(rbar >= target)
                            ucb_sum += rbar
                            obs += 1
                    else:  # iid_group
                        for g in range(G_SYNTH):
                            Ag, Kg = _iid_group_counts(triple, 3 * n_pc, rng)
                            rbar = cp_upper(Kg, Ag, eps)
                            covered += int(rbar >= target)
                            ucb_sum += rbar
                            obs += 1
                rows.append({
                    "study": "SYNTHETIC", "sampling": mode, "G": G_SYNTH,
                    "within_group_risk_spread": _spread_label(triple),
                    "n_per_client": n_pc,
                    "coverage": round(covered / obs, 4),
                    "mean_ucb": round(ucb_sum / obs, 4),
                    "trials": T_SYNTH,
                })
                print(f"  SYNTH spread={_spread_label(triple):>14} (target={target:.3f}) "
                      f"n_pc={n_pc:>3} {mode:>9}: coverage={covered/obs:.4f} "
                      f"mean_ucb={ucb_sum/obs:.4f}")
    return rows


# --------------------------------------------------------------------------- #
# REAL-LOGITS study (frozen resnet18gn CIFAR-10 logits)
# --------------------------------------------------------------------------- #
def _msp(logits):
    """Max-softmax-probability accept score (higher => more in-distribution)."""
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return p.max(axis=1)


def _load_pool(path):
    """Merge prop/cert into an audit pool + keep prop separately for selector choice."""
    z = np.load(path)
    out = {}
    for fold in ("prop", "cert", "test"):
        out[fold] = {
            "score": _msp(z[f"{fold}_logits"]),
            "pred": z[f"{fold}_logits"].argmax(1),
            "y_open": z[f"{fold}_y_open"],
            "client": z[f"{fold}_client"],
        }
    return out


def _accept_error(view, thresh):
    """Return boolean accept mask and error-among-accepted mask.

    Open-set error: an accepted point is wrong if its class is unknown (y_open < 0)
    or (known and mispredicted).
    """
    acc = view["score"] >= thresh
    wrong = (view["y_open"] < 0) | (view["pred"] != view["y_open"])
    return acc, (acc & wrong)


def _choose_threshold_prop(prop, alpha, gamma=1.0):
    """Largest-coverage MSP threshold whose proposal risk <= gamma*alpha (proposal fold only)."""
    acc_all_wrong = (prop["y_open"] < 0) | (prop["pred"] != prop["y_open"])
    order = np.argsort(-prop["score"])                       # descending score
    s_sorted = prop["score"][order]
    wrong_sorted = acc_all_wrong[order].astype(float)
    cum_wrong = np.cumsum(wrong_sorted)
    cum_n = np.arange(1, len(order) + 1)
    risk = cum_wrong / cum_n
    ok = risk <= gamma * alpha
    if not ok.any():
        return float("inf")                                 # infeasible: accept nothing
    k = np.max(np.where(ok)[0])                              # most coverage among feasible
    return float(s_sorted[k])


def run_real(rng, alpha=0.20, n_per_client=400, trials=3000):
    rows = []
    G = 2
    for d in ("0.5", "5"):
        paths = sorted(glob.glob(f"runs/cifar10_d{d}_resnet18gn_none0.0_seed*_logits.npz"))
        if not paths:
            print(f"  [warn] no GN d={d} logits -> skip")
            continue
        # aggregate coverage across the available seeds (each seed: its own pool + selector)
        for mode in ("quota", "iid_group"):
            covered = 0
            obs = 0
            ucb_sum = 0.0
            for path in paths:
                pool = _load_pool(path)
                thr = _choose_threshold_prop(pool["prop"], alpha, gamma=1.0)
                n_clients = int(pool["cert"]["client"].max()) + 1
                gmap = make_group_map(n_clients, G)
                # audit pool = cert fold (disjoint from prop selector + test target)
                audit = pool["cert"]
                acc_a, err_a = _accept_error(audit, thr)
                # TRUE per-group deployment risk from the FULL pool (accepted error rate)
                grp_a = gmap[audit["client"]]
                targets = {}
                for g in range(G):
                    tot = int(((grp_a == g) & acc_a).sum())    # accepted in group g
                    targets[g] = (float(err_a[grp_a == g].sum()) / tot) if tot > 0 else 0.0
                # per-client index lists within the audit pool
                by_client = {c: np.where(audit["client"] == c)[0] for c in range(n_clients)}
                # within-group deployment client shares (pool composition)
                grp_clients = {g: [c for c in range(n_clients) if gmap[c] == g] for g in range(G)}
                eps = DELTA / G
                for _ in range(trials):
                    for g in range(G):
                        cs = grp_clients[g]
                        if mode == "quota":
                            idx = np.concatenate([
                                by_client[c][rng.integers(0, len(by_client[c]), size=n_per_client)]
                                for c in cs if len(by_client[c]) > 0
                            ])
                        else:  # iid_group: draw from the group's pooled points (deployment mixture)
                            gp = np.where(grp_a == g)[0]
                            idx = gp[rng.integers(0, len(gp), size=n_per_client * len(cs))]
                        Ag = int(acc_a[idx].sum())
                        Kg = int(err_a[idx].sum())
                        rbar = cp_upper(Kg, Ag, eps)
                        covered += int(rbar >= targets[g])
                        ucb_sum += rbar
                        obs += 1
            rows.append({
                "study": "REAL", "sampling": mode, "G": G,
                "within_group_risk_spread": f"d={d}", "n_per_client": n_per_client,
                "coverage": round(covered / obs, 4), "mean_ucb": round(ucb_sum / obs, 4),
                "trials": trials,
            })
            print(f"  REAL d={d} {mode:>9}: coverage={covered/obs:.4f} "
                  f"mean_ucb={ucb_sum/obs:.4f} (seeds={len(paths)})")
    return rows


def main():
    print(f"M1 grouped-certificate validity stress (delta={DELTA}, G=2)\n"
          f"  SYNTHETIC: J={J_SYNTH}, dep weights={tuple(W_DEP)}, T={T_SYNTH}")
    rng = np.random.default_rng(0)
    rows = run_synthetic(rng)
    print("  --- real logits ---")
    rows += run_real(np.random.default_rng(1))
    atomic_write_csv(OUT, FIELDS, rows)
    print(f"\nsaved {OUT}  ({len(rows)} rows)")

    # honest summary: worst quota cell vs iid guarantee
    syn = [r for r in rows if r["study"] == "SYNTHETIC"]
    for mode in ("iid_group", "quota"):
        cvs = [r["coverage"] for r in syn if r["sampling"] == mode]
        if cvs:
            print(f"  SYNTHETIC {mode:>9}: coverage range [{min(cvs):.3f}, {max(cvs):.3f}] "
                  f"(target >= {1 - DELTA:.2f})")


if __name__ == "__main__":
    main()
