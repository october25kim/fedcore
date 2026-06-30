"""T8: Fed-CORE certification on real FedOSR base models (deliverable for Risk 2).

Reads scored npz produced by the base-model runners (run_foogd_cifar.py; later FedPD /
FedOSS) and certifies each base model's NATIVE open-set score with the IDENTICAL
Fed-CORE protocol used for the CIFAR headline: worst-group G=2 (and per-client G=J),
fixed native score (no best-of-scores selection), cert_frac=0.5 on the pooled audit
points, box-Lambda best-gamma over gammas in {0.5,0.7,1.0}, delta=0.10, seeds {0,1,2},
alpha in {0.10, 0.20}.

For FOOGD the npz stores logits + the native sm score (||score_model(latents)||, higher
=> more OOD); the Fed-CORE accept-score is -sm. The same npz yields a controlled
FedAvg+MSP baseline row (same backbone features, MSP score) so the table isolates the
SCORE HEAD. Each row carries base_model + base_model_kind in {full, representative} and
context columns (native-score AUROC, closed-set accuracy, accepted r_hat).

Emits runs/T8_fedosr_bases.csv (per-seed canonical schema) and
runs/T8_fedosr_bases_agg.csv (mean+/-std, n_pass/seeds). CPU, no torch.

Run: python experiments/fedcore/aggregate_T8.py
"""

from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict

import numpy as np

from certify import certify_best_gamma, certify_best_gamma_grouped
from scores import msp as msp_score
from atomic_io import atomic_write_csv

GAMMAS = (0.5, 0.7, 1.0)
DELTA, MARGIN, CERT_FRAC = 0.10, 0.01, 0.5
ALPHAS = (0.10, 0.20)
SEEDS = (0, 1, 2)

# base model registry: name -> (kind, score extractor on a pooled fold dict)
def _accept_from_sm(pool):
    return -np.asarray(pool["sm"], dtype=float)          # higher => more ID
def _accept_from_msp(pool):
    return msp_score(pool["logits"])                     # higher => more ID

BASES = {
    "FedAvg+MSP":     {"kind": "full",           "needs": ("logits",),     "score": _accept_from_msp},
    "FOOGD-SM3D":     {"kind": "representative", "needs": ("logits", "sm"), "score": _accept_from_sm},
    "FOOGD-SM3D-SAG": {"kind": "full",           "needs": ("logits", "sm"), "score": _accept_from_sm},
    "FedPD-PROSER":   {"kind": "full",           "needs": ("logits", "sm"), "score": _accept_from_sm},
}


def _pool(npz, keys):
    d = np.load(npz)
    out = {k: np.concatenate([d[f"{f}_{k}"] for f in ("prop", "cert", "test")])
           for k in keys}
    out["y_open"] = np.concatenate([d[f"{f}_y_open"] for f in ("prop", "cert", "test")])
    out["client"] = np.concatenate([d[f"{f}_client"] for f in ("prop", "cert", "test")])
    return out


def _repartition(pool, keys, cert_frac, test_frac, seed):
    rng = np.random.default_rng(seed)
    n = len(pool["y_open"]); perm = rng.permutation(n)
    nt, nc = int(round(n * test_frac)), int(round(n * cert_frac))
    sl = {"test": perm[:nt], "cert": perm[nt:nt + nc], "prop": perm[nt + nc:]}
    allk = tuple(keys) + ("y_open", "client")
    return {f: {k: pool[k][ix] for k in allk} for f, ix in sl.items()}


def _view(part, accept_score, score_name):
    """Build a Fed-CORE scored view {score, pred, y_open, client} for one fold."""
    logits = np.asarray(part["logits"], dtype=float)
    return {"score": accept_score, "pred": logits.argmax(-1),
            "y_open": part["y_open"], "client": part["client"], "_score_name": score_name}


def _auroc_acc_rhat(pool, accept):
    """Native-score AUROC (unknown detection), closed-set acc on knowns (context)."""
    yo = pool["y_open"]; is_unk = yo < 0
    pred = np.asarray(pool["logits"]).argmax(-1)
    auroc = float("nan")
    if is_unk.any() and (~is_unk).any():
        try:
            from sklearn.metrics import roc_auc_score
            # accept high => ID; unknown should get LOW accept => use -accept as OOD score
            auroc = float(roc_auc_score(is_unk.astype(int), -accept))
        except Exception:
            pass
    known = ~is_unk
    acc = float((pred[known] == yo[known]).mean()) if known.any() else float("nan")
    return auroc, acc


def _certify_one(npz, base, alpha, G):
    spec = BASES[base]
    pool = _pool(npz, spec["needs"])
    accept_full = spec["score"](pool)
    pool = dict(pool); pool["_accept"] = accept_full
    parts = _repartition(pool, tuple(spec["needs"]) + ("_accept",), CERT_FRAC, 0.2, seed=0)
    views = {f: _view(parts[f], parts[f]["_accept"], base) for f in ("prop", "cert", "test")}
    n_clients = int(pool["client"].max()) + 1
    if G is None or G >= n_clients:
        r = certify_best_gamma(views["prop"], views["cert"], views["test"], score_name=base,
                               gammas=GAMMAS, alpha=alpha, delta=DELTA, n_clients=n_clients,
                               dirichlet_alpha=float("nan"), Lambda="box", box=0.15, seed=0, margin=MARGIN)
    else:
        gmap = np.array([c * G // n_clients for c in range(n_clients)])
        r = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"], score_name=base,
                                       group_map=gmap, G=G, gammas=GAMMAS, alpha=alpha, delta=DELTA,
                                       Lambda="box", box=0.15, seed=0, margin=MARGIN)
    return r, pool, accept_full


def main() -> None:
    base_dir = "" if glob.glob("runs/*.npz") else "../../"
    rows = []
    import re
    jobs = []  # (base, dtag, seed, npz)
    # representative FOOGD npz: carry FOOGD-SM3D (native score) + controlled FedAvg+MSP
    for f in sorted(glob.glob(base_dir + "runs/foogd_cifar10_d*_seed*.npz")):
        m = re.search(r"_d([0-9.]+)_seed(\d+)", os.path.basename(f))
        if not m or int(m.group(2)) not in SEEDS:
            continue
        d, seed = m.group(1), int(m.group(2))
        jobs.append(("FOOGD-SM3D", d, seed, f))
        jobs.append(("FedAvg+MSP", d, seed, f))
    # full FOOGD-SAG npz (real ODGClient WideResNet+SAG): native score only
    for f in sorted(glob.glob(base_dir + "runs/foogdfull_cifar10_d*_seed*.npz")):
        m = re.search(r"_d([0-9.]+)_seed(\d+)", os.path.basename(f))
        if not m or int(m.group(2)) not in SEEDS:
            continue
        jobs.append(("FOOGD-SM3D-SAG", m.group(1), int(m.group(2)), f))
    # FedPD PROSER npz (real PROSER dummy-class training): native score only
    for f in sorted(glob.glob(base_dir + "runs/fedpd_cifar10_d*_seed*.npz")):
        m = re.search(r"_d([0-9.]+)_seed(\d+)", os.path.basename(f))
        if not m or int(m.group(2)) not in SEEDS:
            continue
        jobs.append(("FedPD-PROSER", m.group(1), int(m.group(2)), f))

    print(f"T8: {len(jobs)} (base,d,seed) jobs")
    for base, d, seed, npz in jobs:
        for alpha in ALPHAS:
            rG2, pool, accept = _certify_one(npz, base, alpha, 2)
            rGJ, _, _ = _certify_one(npz, base, alpha, None)
            auroc, acc = _auroc_acc_rhat(pool, accept)
            rows.append({
                "base_model": base, "base_model_kind": BASES[base]["kind"],
                "dirichlet_alpha": d, "seed": seed, "alpha": alpha, "delta": DELTA,
                "n_clients": 5, "score_name": base, "gamma": rG2.get("gamma_star"),
                "Lambda": "box",
                "certified": rG2["certified"], "cert_risk_ucb": round(rG2["cert_risk_ucb"], 4),
                "cert_coverage_lcb": round(rG2["cert_coverage_lcb"], 4),
                "cert_n": rG2["cert_n"], "cert_k": rG2["cert_k"],
                "prop_coverage": round(rG2["prop_coverage"], 4), "prop_risk": round(rG2["prop_risk"], 4),
                "test_coverage": round(rG2["test_coverage"], 4), "test_risk": round(rG2["test_risk"], 4),
                "certifiedGJ": rGJ["certified"],
                "cert_coverage_lcb_GJ": round(rGJ["cert_coverage_lcb"], 4),
                "auroc": round(auroc, 4), "closed_acc": round(acc, 4),
            })

    out = base_dir + "runs/T8_fedosr_bases.csv"
    atomic_write_csv(out, list(rows[0].keys()), rows)
    print(f"saved {out}  ({len(rows)} rows)")

    # ---- aggregate over seeds ----
    agg = defaultdict(list)
    for r in rows:
        agg[(r["base_model"], r["base_model_kind"], r["dirichlet_alpha"], r["alpha"])].append(r)
    aout = base_dir + "runs/T8_fedosr_bases_agg.csv"
    fields = ["base_model", "base_model_kind", "dirichlet_alpha", "alpha", "n_seeds",
              "CertCovG2_mean", "CertCovG2_std", "n_pass_G2", "certucbG2_median",
              "test_risk_mean", "auroc_mean", "closed_acc_mean", "rhat_mean"]
    print(f"\n{'base_model':>14} {'kind':>14} {'d':>4} {'alpha':>5} "
          f"{'CertCov@a(G2)':>16} {'pass':>5} {'AUROC':>6} {'r_hat':>6}")
    print("-" * 92)
    agg_rows = []
    for key, lst in sorted(agg.items()):
        bm, kind, d, alpha = key
        cov = np.array([x["cert_coverage_lcb"] if x["certified"] else 0.0 for x in lst])
        ucb_cert = [x["cert_risk_ucb"] for x in lst if x["certified"]]
        rhat = np.mean([x["test_risk"] for x in lst])
        row = {"base_model": bm, "base_model_kind": kind, "dirichlet_alpha": d, "alpha": alpha,
               "n_seeds": len(lst), "CertCovG2_mean": round(cov.mean(), 4),
               "CertCovG2_std": round(cov.std(), 4), "n_pass_G2": int((cov > 0).sum()),
               "certucbG2_median": round(float(np.median(ucb_cert)), 4) if ucb_cert else float("inf"),
               "test_risk_mean": round(float(rhat), 4),
               "auroc_mean": round(float(np.nanmean([x["auroc"] for x in lst])), 4),
               "closed_acc_mean": round(float(np.nanmean([x["closed_acc"] for x in lst])), 4),
               "rhat_mean": round(float(rhat), 4)}
        agg_rows.append(row)
        print(f"{bm:>14} {kind:>14} {d:>4} {alpha:>5.2f} "
              f"{cov.mean():>9.3f}+/-{cov.std():<5.3f} {int((cov>0).sum())}/{len(lst)} "
              f"{row['auroc_mean']:>6.3f} {rhat:>6.3f}")
    atomic_write_csv(aout, fields, agg_rows)
    print(f"\nsaved {aout}")


if __name__ == "__main__":
    main()
