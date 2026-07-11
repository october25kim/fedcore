"""Task D: procedurally-VALID multi-score covtype cell.

The draft demoted the covtype "best-of-scores" number to a non-guaranteed
diagnostic because it selected the score by CERTIFICATION-fold coverage (a
post-selection over the same fold used to certify -> invalid). This script
produces the two valid alternatives and reports both:

  proposal_select : choose the score on the PROPOSAL fold only (max proposal
                    coverage among proposal-certified scores), then certify that
                    ONE score once on the cert fold at the FULL delta. Valid
                    because the score choice is a function of the proposal fold,
                    independent of the certification labels.

  bonferroni_d4   : certify all four scores, each at delta/4, and report the best
                    certified coverage. Valid by a union bound over the four
                    scores (overall level delta).

Both use the CIFAR headline protocol: worst-group G=2, cert_frac=0.5, box-Lambda
best-gamma, margin=0.01. For reference we also recompute the fixed-MSP honest
number and the OLD selection-optimistic best-of-4x{G1,2,3} diagnostic.

Output: runs/covtype_valid_multiscore.csv
Run: python experiments/fedcore/exp_covtype_valid_multiscore.py   (CPU, no torch)
"""

from __future__ import annotations

import glob
import os

import numpy as np

from fedcore.certify import certify_best_gamma_grouped
from fedcore.grouping import make_group_map, repartition_trusted_pool, views_from_parts
from fedcore.io_utils import atomic_write_csv

SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.20, 0.25, 0.30)
GAMMAS = (0.2, 0.3, 0.5, 0.7, 1.0)
DELTA, MARGIN, CERT_FRAC = 0.10, 0.01, 0.5
SCORES = ("msp", "energy", "neg_entropy", "margin")
G = 2


def _certify(npz, score, alpha, delta):
    """Full worst-group G=2 best-gamma certificate for one (score, alpha, delta)."""
    d = np.load(npz)
    n_clients = int(d["cert_client"].max()) + 1
    pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
            for k in ("logits", "y_open", "client")}
    parts = repartition_trusted_pool(pool, CERT_FRAC, 0.2, seed=0)
    v = views_from_parts(parts, score)
    gmap = make_group_map(n_clients, G)
    return certify_best_gamma_grouped(
        v["prop"], v["cert"], v["test"], score_name=score, group_map=gmap, G=G,
        gammas=GAMMAS, alpha=alpha, delta=delta, Lambda="box", box=0.15,
        seed=0, margin=MARGIN)


def proposal_select(npz, alpha):
    """VALID: pick score by PROPOSAL fold, certify once at full delta.

    Returns (cov, chosen_score). Selection criterion: among scores whose
    proposal-side proxy certifies (u_proxy <= alpha), the one with the largest
    PROPOSAL coverage; if none certifies on the proposal fold, the lowest proxy
    UCB (least infeasible). The chosen score is then read off its FULL-delta cert.
    """
    per = {s: _certify(npz, s, alpha, DELTA) for s in SCORES}
    feas = [s for s in SCORES if per[s]["u_proxy"] <= alpha]
    if feas:
        chosen = max(feas, key=lambda s: per[s]["prop_coverage"])
    else:
        chosen = min(SCORES, key=lambda s: per[s]["u_proxy"])
    r = per[chosen]
    cov = r["cert_coverage_lcb"] if r["certified"] else 0.0
    return cov, chosen


def bonferroni_d4(npz, alpha):
    """VALID: certify each score at delta/4, report best certified coverage."""
    best_cov, best_score = 0.0, None
    for s in SCORES:
        r = _certify(npz, s, alpha, DELTA / len(SCORES))
        cov = r["cert_coverage_lcb"] if r["certified"] else 0.0
        if cov > best_cov:
            best_cov, best_score = cov, s
    return best_cov, (best_score or "none")


def honest_msp(npz, alpha):
    r = _certify(npz, "msp", alpha, DELTA)
    return (r["cert_coverage_lcb"] if r["certified"] else 0.0), "msp"


def selection_optimistic(npz, alpha):
    """OLD invalid diagnostic: best over scores x G in {1,2,3} at full delta."""
    best = 0.0
    for s in SCORES:
        for g in (1, 2, 3):
            d = np.load(npz)
            n_clients = int(d["cert_client"].max()) + 1
            pool = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
                    for k in ("logits", "y_open", "client")}
            parts = repartition_trusted_pool(pool, CERT_FRAC, 0.2, seed=0)
            v = views_from_parts(parts, s)
            gmap = make_group_map(n_clients, g)
            r = certify_best_gamma_grouped(
                v["prop"], v["cert"], v["test"], score_name=s, group_map=gmap, G=g,
                gammas=GAMMAS, alpha=alpha, delta=DELTA, Lambda="box", box=0.15,
                seed=0, margin=MARGIN)
            best = max(best, r["cert_coverage_lcb"] if r["certified"] else 0.0)
    return best, "best"


PROTOCOLS = {
    "honest_msp_G2":        (honest_msp,           "valid (fixed MSP)"),
    "proposal_select_G2":   (proposal_select,      "valid (proposal-fold score select)"),
    "bonferroni_d4_G2":     (bonferroni_d4,        "valid (Bonferroni delta/4)"),
    "selection_G123":       (selection_optimistic, "INVALID diagnostic (cert-fold select)"),
}


def main() -> None:
    base = "" if glob.glob("runs/*_logits.npz") else "../../"
    files = [base + f"runs/covtype_seed{s}_logits.npz" for s in SEEDS]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("covtype logits NOT found -> skipping (rerun run_tabular per seed first).")
        return

    print(f"covtype VALID multi-score ({len(files)} seeds), G={G}, cert_frac={CERT_FRAC}, "
          f"box best-gamma, margin={MARGIN}, delta={DELTA}\n")
    print(f"{'protocol':>20} {'alpha':>5} {'CertCov mean+/-std':>20} {'n_pass':>7}  scores")
    print("-" * 96)
    rows = []
    for alpha in ALPHAS:
        for name, (fn, note) in PROTOCOLS.items():
            covs, chosen = [], []
            for f in files:
                c, s = fn(f, alpha)
                covs.append(c)
                chosen.append(s)
            a = np.array(covs)
            rows.append({
                "protocol": name, "note": note, "alpha": alpha, "n_seeds": len(a),
                "CertCov_mean": round(float(a.mean()), 4),
                "CertCov_std": round(float(a.std()), 4),
                "n_pass": int((a > 0).sum()),
                "per_seed": "|".join(f"{x:.3f}" for x in a),
                "chosen_scores": "|".join(chosen),
            })
            print(f"{name:>20} {alpha:>5.2f} {a.mean():>9.3f}+/-{a.std():<9.3f} "
                  f"{int((a>0).sum()):>3}/{len(a)}  [{', '.join(chosen)}]")
        print()

    out = base + "runs/covtype_valid_multiscore.csv"
    atomic_write_csv(out, list(rows[0].keys()), rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
