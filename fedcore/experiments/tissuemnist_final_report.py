"""TissueMNIST final certification report (owner spec 2026-08-21).

Run at TissueMNIST 50/50. Emits, for the ResNeXt-plain TissueMNIST arm:
  - final certified count (Holm/IUT full-simplex, alpha=0.20)
  - zero-imputed EffectiveCertCov (mean coverage over ALL cells, 0 for uncertified)
  - CondCertCov (mean coverage conditional on certification)
  - split/rep variability
  - single-policy vs Holm/IUT
  - complete mutually exclusive failure taxonomy

No selector tuning, no seed/matrix changes: reads the frozen common-schema npz and
applies the SAME family machinery as every other Fed-CORE arm.
"""
from __future__ import annotations

import glob
import math
import os
from collections import Counter, defaultdict

import numpy as np

from fedcore.experiments.recertify_cifar_sweep import ALPHA, DELTA_C, DELTA_R, N_CLIENTS, certify_cell
from fedcore.officehome_rescue import holm_family_certificate

CELLS = "runs/tissuemnist_sweep/cells"
FLOOR = math.log(N_CLIENTS / DELTA_C) / (-math.log(1 - ALPHA))   # Theorem-2 per-client A_j floor


def _split_rep(base):
    # resnext29_8x64d__tissuemnist_split_02__seed3__d0.5
    parts = base.split("__")                       # [arch, tissuemnist_split_NN, seedR, dX]
    sp = parts[1].split("_")[-1]                    # "02"
    rep = parts[2].replace("seed", "")             # "3"
    d = "0.1" if "d0.1" in base else ("0.5" if "d0.5" in base else "5.0")
    return f"split{sp}", f"seed{rep}", d


def analyze():
    npzs = sorted(glob.glob(f"{CELLS}/resnext29_8x64d__*_common.npz"))
    # expected 50 cells; cells with training completed have an npz
    done_ids = {os.path.basename(p).replace("_common.npz", "") for p in npzs}
    # full expected set from the matrix
    import csv
    matrix = [r["semantic_id"] for r in csv.DictReader(open("results/tissuemnist_sweep/final_training_matrix.csv"))]

    rows = []
    modal = Counter()
    for sid in matrix:
        base = sid
        npz = f"{CELLS}/{base}_common.npz"
        if base not in done_ids or not os.path.isfile(npz):
            # distinguish a structural calibration barrier from a generic training crash
            cat = "TRAIN_FAILED"
            errlog = f"{CELLS}/{base}.err.log"
            if os.path.isfile(errlog) and "insufficient unknown observations" in open(errlog, errors="ignore").read():
                cat = "CALIBRATION_INFEASIBLE"
            sp, rep, d = _split_rep(base)
            rows.append(dict(sid=base, cat=cat, certified=False, cov=0.0, win=None,
                             split=sp, rep=rep, d=d))
            continue
        res = certify_cell(npz)
        A, K, n, fc, sel, keys = res["A"], res["K"], res["n"], res["fc"], res["sel_idx"], res["keys"]
        certified = sel is not None
        cov = float(fc.C[sel]) if certified else 0.0
        win = f"{keys[sel].score_name}/g{keys[sel].gamma:g}" if certified else None
        if certified:
            modal[win] += 1
        # feasibility: best member's min-client accepted count
        best_min_A = int(A.min(axis=1).max())
        feasible = best_min_A >= FLOOR
        any_cov = bool((fc.C > 0).any())
        # taxonomy (mutually exclusive)
        if certified:
            cat = "CERTIFIED"
        elif not feasible:
            cat = "INFEASIBLE"
        elif not any_cov:
            cat = "ZERO_COVERAGE"
        else:
            cat = "RISK_BARRIER"
        sp, rep, d = _split_rep(base)
        rows.append(dict(sid=base, cat=cat, certified=certified, cov=cov, win=win,
                         A=A, K=K, n=n, keys=keys, split=sp, rep=rep, d=d, best_min_A=best_min_A))
    return rows, modal


def single_policy(rows, member_key):
    """Certify each done cell with ONE fixed family member (no Holm correction over M)."""
    out = []
    for r in rows:
        if "keys" not in r:                      # TRAIN_FAILED / CALIBRATION_INFEASIBLE (no npz)
            out.append((False, 0.0)); continue
        keys = r["keys"]
        mi = next((i for i, k in enumerate(keys) if f"{k.score_name}/g{k.gamma:g}" == member_key), None)
        if mi is None:
            out.append((False, 0.0)); continue
        fc1 = holm_family_certificate(r["A"][mi:mi + 1], r["K"][mi:mi + 1], r["n"],
                                      alpha=ALPHA, delta_r=DELTA_R, delta_c=DELTA_C)
        cert = bool(fc1.certified[0]); cov = float(fc1.C[0]) if cert else 0.0
        out.append((cert, cov))
    return out


def report():
    rows, modal = analyze()
    N = len(rows)
    NOTRUN = {"TRAIN_FAILED", "CALIBRATION_INFEASIBLE"}
    done = [r for r in rows if r["cat"] not in NOTRUN]        # trained (feasible) cells only
    cert = [r for r in rows if r["certified"]]
    covs_all = [r["cov"] for r in rows]                       # zero-imputed over ALL 50
    covs_feasible = [r["cov"] for r in done]                  # zero-imputed over trained/feasible
    covs_cond = [r["cov"] for r in cert]
    print(f"{'='*70}\nTissueMNIST ResNeXt-plain FINAL REPORT  (alpha={ALPHA}, {N} cells, {len(done)} trained)\n{'='*70}")
    print(f"   [{len(done)} feasible/trained cells; {N-len(done)} not run (see taxonomy)]")
    print(f"1. Certified (Holm/IUT full-simplex): {len(cert)}/{N} (= {len(cert)}/{len(done)} of feasible)")
    print(f"2. EffectiveCertCov (zero-imputed): /{N} all = {np.mean(covs_all):.4f} | /{len(done)} feasible = {np.mean(covs_feasible):.4f}")
    print(f"3. CondCertCov (| certified): mean {np.mean(covs_cond):.4f}" if covs_cond else "3. CondCertCov: n/a")

    print("\n4. split/rep variability (certified count):")
    for axis in ("split", "rep", "d"):
        agg = defaultdict(lambda: [0, 0])
        for r in done:
            agg[r[axis]][0] += int(r["certified"]); agg[r[axis]][1] += 1
        print(f"   by {axis}: " + " ".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(agg.items())))

    print("\n5. single-policy vs Holm/IUT:")
    print(f"   Holm/IUT: {len(cert)}/{N} certified, EffCertCov {np.mean(covs_all):.4f}")
    mp = modal.most_common(1)[0][0] if modal else None
    if mp:
        sp = single_policy(rows, mp)
        sp_cert = sum(1 for c, _ in sp if c); sp_effcov = np.mean([cov if c else 0.0 for c, cov in sp])
        print(f"   single-policy [{mp}] (modal winner, no family correction): {sp_cert}/{N} certified, EffCertCov {sp_effcov:.4f}")

    print("\n6. failure taxonomy (mutually exclusive):")
    tax = Counter(r["cat"] for r in rows)
    for cat in ("CERTIFIED", "RISK_BARRIER", "ZERO_COVERAGE", "INFEASIBLE", "CALIBRATION_INFEASIBLE", "TRAIN_FAILED"):
        print(f"   {cat:22}: {tax.get(cat,0)}")
    assert sum(tax.values()) == N, "taxonomy not exhaustive!"
    print(f"   (sum {sum(tax.values())} == {N} cells: mutually exclusive + exhaustive)")
    print(f"\n  Feasibility floor A_j >= {FLOOR:.1f} (Theorem 2); winners: {dict(modal)}")


if __name__ == "__main__":
    report()
