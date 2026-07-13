"""M5: A4 composition stress -- mismatch in WHICH unknowns appear (audit vs deployment)
at a MATCHED unknown fraction, on the stored GN (resnet18gn) CIFAR-10 logits.

Section 5.3 stresses the unknown FRACTION; A4 also needs COMPOSITION. Here the audit
unknown fraction is held FIXED at 0.30 while we reweight WHICH held-out unknowns appear
in the audit fold versus deployment, and measure whether the worst-group risk certificate
built on the audit fold still covers the TRUE deployment accepted-error risk.

DATA LIMITATION (stated openly): the frozen logits encode every unknown as y_open=-1 --
the original held-out CLASS id was not persisted. We therefore use a PRE-DECLARED,
transparent proxy for "held-out class composition": the unknown pool is split into four
MSP-score quartiles Q1<Q2<Q3<Q4 (Q4 = highest MSP = hardest to reject = most often
wrongly accepted). Reweighting which quartiles appear in the audit vs deployment fold is
a direct, and if anything sharper, realization of composition mismatch: it is exactly the
acceptance-relevant axis of an unknown sub-population. This is honest about the mechanism
A4 warns of; it does not claim to be the CIFAR class partition.

Per seed (GN d in {0.5, 5}, seeds 0..4):
  * pool known points (with labels) + unknown points; quartile the unknowns by MSP.
  * fix an MSP selector on a matched-composition proposal draw (gamma=1.0, alpha=0.20).
  * TRUE deployment risk = analytic accepted-error rate of the deployment composition
    (known + unknowns drawn from deploy quartiles at unknown fraction 0.30).
  * T resamples of the AUDIT fold (known + unknowns from audit quartiles at frac 0.30):
    worst-group (G=2) risk UCB rbar = max_g U+(K_g, A_g; delta/G); covered = UCB >= true_risk.
  * coverage = mean covered over trials x seeds.

Expected: composition mismatch (audit=easy-to-reject unknowns, deploy=hard-to-reject) can
break validity even at matched fraction; the reverse is conservative. This sharpens A4.

CSV: runs/a4_composition_stress.csv
  columns: d,audit_classes,deploy_classes,coverage,mean_ucb,true_risk,trials

Run: python -m fedcore.experiments.exp_a4_composition_stress   (CPU, no torch)
"""
from __future__ import annotations

import csv
import glob
import os

import numpy as np

from fedcore.certificate import cp_upper
from fedcore.grouping import make_group_map

DELTA, ALPHA, G = 0.10, 0.20, 2
UNK_FRAC = 0.30
N_KNOWN_FOLD = 800
T = 2000
DS = ("0.5", "5")
# pre-declared audit/deploy quartile compositions (Q1=lowest MSP ... Q4=highest MSP)
SETTINGS = [
    ((0, 1, 2, 3), (0, 1, 2, 3), "Q1-Q4", "Q1-Q4"),   # matched control
    ((0, 1),       (2, 3),       "Q1-Q2", "Q3-Q4"),   # audit easy / deploy hard  -> expect break
    ((2, 3),       (0, 1),       "Q3-Q4", "Q1-Q2"),   # audit hard / deploy easy  -> conservative
    ((0, 1),       (0, 1, 2, 3), "Q1-Q2", "Q1-Q4"),   # audit easy / deploy all
    ((2, 3),       (0, 1, 2, 3), "Q3-Q4", "Q1-Q4"),   # audit hard / deploy all
]
OUT = "runs/a4_composition_stress.csv"
FIELDS = ["d", "audit_classes", "deploy_classes", "coverage", "mean_ucb", "true_risk", "trials"]


def _msp(logits):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return p.max(1), p.argmax(1)


def _load(npz):
    z = np.load(npz)
    logits = np.concatenate([z[f"{f}_logits"] for f in ("prop", "cert", "test")])
    y = np.concatenate([z[f"{f}_y_open"] for f in ("prop", "cert", "test")])
    cl = np.concatenate([z[f"{f}_client"] for f in ("prop", "cert", "test")])
    score, pred = _msp(logits)
    return score, pred, y, cl


def _choose_threshold(score, err, rng, n_known, known_idx, unk_idx):
    """Matched-composition proposal draw -> largest-coverage MSP with risk <= ALPHA."""
    nk = min(n_known, len(known_idx))
    ksamp = known_idx[rng.integers(0, len(known_idx), size=nk)]
    n_unk = int(round(UNK_FRAC / (1 - UNK_FRAC) * nk))
    usamp = unk_idx[rng.integers(0, len(unk_idx), size=n_unk)]
    idx = np.concatenate([ksamp, usamp])
    s, e = score[idx], err[idx]
    order = np.argsort(-s)
    risk = np.cumsum(e[order]) / np.arange(1, len(order) + 1)
    ok = risk <= ALPHA
    if not ok.any():
        return np.inf
    return float(s[order][int(np.max(np.where(ok)[0]))])


def _true_risk(score, err, acc_known, acc_err_known, quart_idx, deploy_q, thr):
    """Analytic deployment accepted-error rate at UNK_FRAC unknown composition deploy_q."""
    du = np.concatenate([quart_idx[q] for q in deploy_q])
    p_acc_unk = float((score[du] >= thr).mean())          # unknowns: accept prob (all errors)
    f = UNK_FRAC
    num = (1 - f) * acc_err_known + f * p_acc_unk
    den = (1 - f) * acc_known + f * p_acc_unk
    return num / den if den > 0 else 0.0


def run_d(d, rng):
    files = sorted(glob.glob(f"runs/cifar10_d{d}_resnet18gn_none0.0_seed[0-9]_logits.npz"))
    if not files:
        return []
    # accumulate coverage per setting over seeds x trials
    agg = {i: {"cov": 0, "obs": 0, "ucb": 0.0, "true": []} for i in range(len(SETTINGS))}
    for f in files:
        score, pred, y, cl = _load(f)
        n_clients = int(cl.max()) + 1
        gmap = make_group_map(n_clients, G)
        err = (y < 0) | (pred != y)                       # open-set error
        known_idx = np.where(y >= 0)[0]
        unk_idx = np.where(y < 0)[0]
        # quartile the unknowns by MSP score (Q0=lowest ... Q3=highest)
        us = score[unk_idx]
        order = unk_idx[np.argsort(us)]
        quart_idx = {q: order[q * len(order) // 4:(q + 1) * len(order) // 4] for q in range(4)}
        thr = _choose_threshold(score, err, rng, N_KNOWN_FOLD, known_idx, unk_idx)
        # known-pool accept / accept-error rates (composition-independent)
        acc_known = float((score[known_idx] >= thr).mean())
        acc_err_known = float(((score[known_idx] >= thr) & err[known_idx]).mean())
        n_unk = int(round(UNK_FRAC / (1 - UNK_FRAC) * N_KNOWN_FOLD))
        for i, (audit_q, deploy_q, _al, _dl) in enumerate(SETTINGS):
            tr = _true_risk(score, err, acc_known, acc_err_known, quart_idx, deploy_q, thr)
            agg[i]["true"].append(tr)
            au = np.concatenate([quart_idx[q] for q in audit_q])
            for _ in range(T):
                ksamp = known_idx[rng.integers(0, len(known_idx), size=N_KNOWN_FOLD)]
                usamp = au[rng.integers(0, len(au), size=n_unk)]
                idx = np.concatenate([ksamp, usamp])
                acc = score[idx] >= thr
                grp = gmap[cl[idx]]
                werr = acc & err[idx]
                rbar = 0.0
                for g in range(G):
                    Ag = int((acc & (grp == g)).sum())
                    Kg = int((werr & (grp == g)).sum())
                    rbar = max(rbar, cp_upper(Kg, Ag, DELTA / G))
                agg[i]["cov"] += int(rbar >= tr)
                agg[i]["ucb"] += rbar
                agg[i]["obs"] += 1
    rows = []
    for i, (_aq, _dq, al, dl) in enumerate(SETTINGS):
        a = agg[i]
        rows.append({"d": d, "audit_classes": al, "deploy_classes": dl,
                     "coverage": round(a["cov"] / a["obs"], 4),
                     "mean_ucb": round(a["ucb"] / a["obs"], 4),
                     "true_risk": round(float(np.mean(a["true"])), 4), "trials": T})
        print(f"  d={d} audit={al:>6} deploy={dl:>6}: coverage={a['cov']/a['obs']:.4f} "
              f"mean_ucb={a['ucb']/a['obs']:.4f} true_risk={np.mean(a['true']):.4f}")
    return rows


def main():
    print(f"M5 A4 composition stress (unknown frac fixed={UNK_FRAC}, G={G}, delta={DELTA}, "
          f"MSP quartile proxy, T={T})")
    rng = np.random.default_rng(0)
    rows = []
    for d in DS:
        print(f"--- GN d={d} ---")
        rows += run_d(d, rng)
    if not rows:
        print("[warn] no GN logits found"); return
    comment = ("# M5 A4 composition stress. PROXY: held-out unknown CLASS id is NOT persisted in the "
               "frozen logits (all unknowns y_open=-1), so the unknown pool is split into MSP-score "
               "quartiles Q1<Q2<Q3<Q4 as a pre-declared stand-in for held-out-class composition -- "
               "NOT the CIFAR class partition. audit/deploy = which quartiles appear; unknown fraction "
               "fixed=0.30; grouped G=2; worst-group risk UCB; delta=0.10; MSP head.")
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        fh.write(comment + "\n")
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"\nsaved {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
