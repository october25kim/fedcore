"""M6: certify FedPD-PROSER CIFAR-100 (native -sm detector) and APPEND to
runs/cifar100_multimodel.csv with backbone=fedpd_proser -- CPU, no retraining.

Tests whether the strong-detector effect transfers to CIFAR-100 (the MSP baselines there
are thin, CertCov ~0.05-0.09). Reads runs/fedpd_cifar100_d5_seed{0,1,2}.npz (produced by
scripts/run_fedpd_cifar100_m6.sh), certifies with the IDENTICAL grouped protocol used for
the CIFAR-100 multimodel table but on the detector's NATIVE accept score (-sm, gammas
{0.5,0.7,1.0}) -- the score head that DEFINES FedPD-PROSER -- for G in {2, J} and
alpha in {0.10, 0.20}.

Preserves the existing MSP rows: reads the current CSV, drops any prior fedpd_proser rows
(idempotent), appends the fresh fedpd_proser rows, and rewrites comment+header+all rows.

Run: python -m fedcore.experiments.exp_m6_cifar100_fedpd   (CPU, no torch)
"""
from __future__ import annotations

import csv
import glob
import os
import re

import numpy as np

from fedcore.certify import certify_best_gamma_grouped
from fedcore.selector import choose_threshold, counts_per_client
from fedcore.grouping import make_group_map
from fedcore.aggregate.t8 import (
    CERT_FRAC, DELTA, GAMMAS, MARGIN,
    _accept_from_sm, _pool, _repartition_score_pool, _view,
)

CSV = "runs/cifar100_multimodel.csv"
COMMENT = ("# grouping=contiguous c*G//J; G in {2,J=5}; protocol=msp,cert_frac=0.5,box=0.15,delta=0.10 "
           "(+ fedpd_proser rows: NATIVE -sm, gammas{0.5,0.7,1.0})")
FIELDS = ["backbone", "d", "seed", "alpha", "G", "cert_risk_ucb", "cert_coverage_lcb",
          "cert_n_min_group", "certified", "test_risk", "test_coverage"]
BASE, BB = "FedPD-PROSER", "fedpd_proser"
TEST_FRAC, SEED, BOX = 0.2, 0, 0.15
ALPHAS = (0.10, 0.20)
GLOB = "runs/fedpd_cifar100_d{d}_seed[0-9].npz"
DS = ("5",)


def certify_native(npz_path, G, alpha):
    pool = _pool(npz_path, ("logits", "sm"))
    pool["_accept"] = _accept_from_sm(pool)
    parts = _repartition_score_pool(pool, ("logits", "sm", "_accept"), CERT_FRAC, TEST_FRAC, seed=SEED)
    J = int(np.asarray(parts["cert"]["client"]).max()) + 1
    views = {fn: _view(parts[fn], np.asarray(parts[fn]["_accept"], float), BASE)
             for fn in ("prop", "cert", "test")}
    gmap = make_group_map(J, G)
    res = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"],
                                     score_name=BASE, group_map=gmap, G=G, gammas=GAMMAS,
                                     alpha=alpha, delta=DELTA, Lambda="box", box=BOX,
                                     seed=SEED, margin=MARGIN)
    # grouped per-group counts -> cert_n_min_group
    gv = {}
    for fn in ("prop", "cert"):
        v = dict(views[fn]); v["client"] = gmap[np.asarray(views[fn]["client"])]; gv[fn] = v
    sel = choose_threshold(gv["prop"]["score"], gv["prop"]["pred"], gv["prop"]["y_open"],
                           res["gamma_star"], alpha)
    A, _K, _n = counts_per_client(gv["cert"]["score"], gv["cert"]["pred"],
                                  gv["cert"]["y_open"], np.asarray(gv["cert"]["client"]), sel, G)
    return res, int(A.min()), J


def _read_existing():
    rows, comment = [], COMMENT
    if not os.path.exists(CSV):
        return rows, comment
    with open(CSV) as f:
        lines = f.read().splitlines()
    if lines and lines[0].startswith("#"):
        comment = lines[0]
    body = [l for l in lines if l and not l.startswith("#")]
    if not body:
        return rows, comment
    reader = csv.DictReader(body)
    for r in reader:
        if r.get("backbone") != BB:              # drop prior fedpd_proser rows (idempotent)
            rows.append(r)
    return rows, comment


def main():
    keep, comment = _read_existing()
    new_rows = []
    for d in DS:
        files = sorted(glob.glob(GLOB.format(d=d)))
        if not files:
            print(f"[warn] no FedPD cifar100 d={d} npz yet ({GLOB.format(d=d)}); run scripts/run_fedpd_cifar100_m6.sh")
            continue
        for f in files:
            s = int(re.search(r"seed(\d+)", os.path.basename(f)).group(1))
            for alpha in ALPHAS:
                for G in (2, 5):
                    res, nmin, J = certify_native(f, G, alpha)
                    new_rows.append({"backbone": BB, "d": d, "seed": s, "alpha": alpha, "G": G,
                                     "cert_risk_ucb": round(float(res["cert_risk_ucb"]), 4),
                                     "cert_coverage_lcb": round(float(res["cert_coverage_lcb"]), 4),
                                     "cert_n_min_group": nmin,
                                     "certified": int(bool(res["certified"])),
                                     "test_risk": round(float(res["test_risk"]), 4),
                                     "test_coverage": round(float(res["test_coverage"]), 4)})
            print(f"[ok] fedpd_proser cifar100 d={d} seed={s}")
    if not new_rows:
        print("[warn] no FedPD cifar100 logits found; nothing appended."); return
    all_rows = keep + new_rows
    with open(CSV, "w", newline="") as f:
        f.write(comment + "\n")
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"\nwrote {CSV}  ({len(all_rows)} rows; +{len(new_rows)} fedpd_proser)")
    for alpha in ALPHAS:
        for G in (2, 5):
            sub = [r for r in new_rows if abs(float(r["alpha"]) - alpha) < 1e-9 and int(r["G"]) == G]
            if sub:
                cov = np.mean([r["cert_coverage_lcb"] if r["certified"] else 0.0 for r in sub])
                print(f"  fedpd_proser alpha={alpha:.2f} G={G}: CertCov(cert else 0)={cov:.4f} "
                      f"(n={len(sub)}, cert={sum(r['certified'] for r in sub)})")


if __name__ == "__main__":
    main()
