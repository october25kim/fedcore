"""PHASE-0 golden snapshot capture for the structure-only refactor (regression oracle).

Captures the DETERMINISTIC outputs that MUST stay bit-for-bit identical (abs diff <= 1e-9)
after the refactor:
  1. certificate math on FIXED integer inputs (cp_upper/lower, Thm1 simplex, Thm1' box,
     Thm3 pooled, Thm2 floor) -- the purest invariant;
  2. scores + selector on a FIXED logit fixture (all four scores; threshold + accepted mask);
  3. split-index determinism: calibration folds from a FIXED seed (disjointness + index hashes);
  4. certify path on FROZEN runs/*_logits.npz (full canonical metric schema).

Writes JSON snapshots (full double precision) under tests/golden/. Re-run after the refactor
and diff with tests/golden_check.py. NO source is modified by this file.

Run: python tests/golden_capture.py            (CPU, host; numpy+scipy only)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# ROOT itself so `import fedcore` resolves to the hoisted project-root package without relying
# on `pip install -e .`; experiments/fedcore so the flat backward-compat shims still import.
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "fedcore"))
GOLD = os.environ.get("GOLDEN_OUT", os.path.join(HERE, "golden"))
os.makedirs(GOLD, exist_ok=True)

from certificates import (  # noqa: E402
    conditional_risk_certificate, cp_lower, cp_upper, pooled_cp, stratified_certificate,
)
from scores import compute_score, scored_views  # noqa: E402
from selector import choose_threshold, counts_per_client, open_set_error  # noqa: E402


def _dump(name, obj):
    path = os.path.join(GOLD, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


def _f(x):
    """Full-precision JSON-safe float (or list)."""
    a = np.asarray(x)
    return a.item() if a.ndim == 0 else a.tolist()


# --------------------------------------------------------------------------- #
# 1. certificate math on FIXED inputs
# --------------------------------------------------------------------------- #
def snap_certificate_math():
    out = {}
    # CP UCB/LCB on fixed (k, n, eps)
    out["cp_upper"] = {f"k{k}_n{n}_eps{eps}": cp_upper(k, n, eps)
                       for k, n, eps in [(3, 50, 0.05), (0, 30, 0.1), (12, 100, 0.025), (50, 200, 0.01)]}
    out["cp_lower"] = {f"A{A}_n{n}_eps{eps}": cp_lower(A, n, eps)
                       for A, n, eps in [(40, 50, 0.05), (10, 30, 0.1), (180, 200, 0.01)]}
    # conditional certificate (Thm 1 simplex / Thm 1' box / known-lambda) on fixed A,K,n
    A = [120, 95, 140, 60, 110]; K = [8, 6, 11, 9, 7]; n = [400, 380, 420, 300, 360]
    delta = 0.10
    for Lambda in ("simplex", "box", "known"):
        lam = [0.2, 0.2, 0.2, 0.2, 0.2] if Lambda == "known" else None
        c = conditional_risk_certificate(A, K, n, delta, Lambda=Lambda, lam=lam, box=0.15, seed=0)
        out[f"conditional_{Lambda}"] = {"U": _f(c.U), "feasible": bool(c.feasible)}
    # stratified certificate
    st = stratified_certificate(A, K, n, delta)
    out["stratified"] = {k: _f(getattr(st, k)) for k in ("U", "feasible") if hasattr(st, k)}
    # pooled (Prop 3 / Thm 3)
    out["pooled_cp"] = pooled_cp([sum(A)], [sum(K)], delta)
    out["pooled_cp_perclient_sum"] = pooled_cp(A, K, delta)
    # Theorem-2 floor (per-group), J in {2,3,5}, alpha 0.10/0.20
    out["thm2_floor"] = {f"J{J}_a{a}_d{delta}": float(np.log(J / delta) / (-np.log(1 - a)))
                         for J in (2, 3, 5) for a in (0.10, 0.20)}
    _dump("certificate_math.json", out)


# --------------------------------------------------------------------------- #
# 2. scores + selector on a FIXED logit fixture
# --------------------------------------------------------------------------- #
def snap_scores_selector():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(40, 6)).astype(np.float64)        # fixed fixture
    y_open = np.array(([0, 1, 2, 3, 4, 5] * 7)[:40]); y_open[::5] = -1
    pred = logits.argmax(1)
    out = {"logits_sha": int(np.abs(logits).sum() * 1e6) % 10**9}
    out["scores"] = {s: _f(compute_score(s, logits)) for s in ("msp", "neg_entropy", "margin", "energy")}
    sel = choose_threshold(logits.max(1) - logits.mean(1), pred, y_open, gamma=0.7, alpha=0.20)
    out["selector"] = {"threshold": _f(sel.threshold), "feasible": bool(sel.feasible),
                       "accepted_mask_sum": int((sel.accept(logits.max(1) - logits.mean(1))).sum())}
    out["open_set_error_sum"] = int(open_set_error(pred, y_open).sum())
    _dump("scores_selector.json", out)


# --------------------------------------------------------------------------- #
# 3. split-index determinism (calibration folds from a FIXED seed)
# --------------------------------------------------------------------------- #
def snap_split_determinism():
    from fedosr_split import build_calibration, dirichlet_partition, open_set_split
    rng = np.random.default_rng(0)
    labels = np.array(([c for c in range(10)] * 600))           # 6000 fixture labels
    known, unknown, remap = open_set_split(labels, 6, 0)
    kidx = np.where(np.isin(labels, known))[0]
    ky = np.array([remap[int(c)] for c in labels[kidx]])
    parts = dirichlet_partition(kidx[:3000], ky[:3000], 5, 5.0, 0)
    uidx = np.where(np.isin(labels, unknown))[0]
    calib = build_calibration(kidx[3000:], ky[3000:], uidx, 5, (0.4, 0.3, 0.3), 0.30, 0)
    folds = {}
    seen = []
    for fold in ("prop", "cert", "test"):
        idx = np.concatenate([np.asarray(c[fold]["idx"]) for c in calib])
        folds[fold] = {"n": int(len(idx)), "idx_sum": int(idx.sum()), "idx_min": int(idx.min()),
                       "idx_max": int(idx.max())}
        seen.append(set(int(i) for i in idx))
    disjoint = (len(seen[0] & seen[1]) == 0 and len(seen[0] & seen[2]) == 0 and len(seen[1] & seen[2]) == 0)
    out = {"known_classes": known.tolist(), "unknown_classes": unknown.tolist(),
           "dirichlet_client_sizes": [int(len(c)) for c in parts],
           "folds": folds, "prop_cert_test_disjoint": bool(disjoint)}
    _dump("split_determinism.json", out)


# --------------------------------------------------------------------------- #
# 4. certify path on FROZEN logits (full canonical schema)
# --------------------------------------------------------------------------- #
CANON = ["score_name", "gamma", "alpha", "delta", "Lambda", "dirichlet_alpha", "n_clients",
         "certified", "cert_risk_ucb", "cert_coverage_lcb", "cert_n", "cert_k",
         "prop_coverage", "prop_risk", "test_coverage", "test_risk"]


def snap_certify_frozen():
    from certify import certify_best_gamma, certify_best_gamma_grouped, certify_for_score
    npzs = ["runs/cifar10_d5_resnet18_seed0_logits.npz", "runs/cifar100_d5_none0.0_seed0_logits.npz"]
    out = {}
    for rel in npzs:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            out[rel] = "MISSING"; continue
        d = np.load(p)
        nclient = int(d["cert_client"].max()) + 1
        views = {fn: scored_views(d[f"{fn}_logits"], d[f"{fn}_y_open"], d[f"{fn}_client"], ["msp"])["msp"]
                 for fn in ("prop", "cert", "test")}
        cell = {}
        for Lambda in ("simplex", "box"):
            r = certify_for_score("msp", views["prop"], views["cert"], views["test"], gamma=0.5,
                                  alpha=0.10, delta=0.10, Lambda=Lambda, n_clients=nclient,
                                  dirichlet_alpha=5.0, box=0.15, seed=0)
            cell[f"for_score_{Lambda}_g0.5_a0.10"] = {k: _f(r[k]) for k in CANON}
        bg = certify_best_gamma(views["prop"], views["cert"], views["test"], score_name="msp",
                                gammas=(0.2, 0.3, 0.5, 0.7, 1.0), alpha=0.20, delta=0.10,
                                n_clients=nclient, dirichlet_alpha=5.0, Lambda="box", box=0.15,
                                seed=0, margin=0.01)
        cell["best_gamma_box_a0.20"] = {k: _f(bg[k]) for k in CANON} | {"gamma_star": _f(bg["gamma_star"])}
        gmap = np.array([c * 2 // nclient for c in range(nclient)])
        gg = certify_best_gamma_grouped(views["prop"], views["cert"], views["test"], score_name="msp",
                                        group_map=gmap, G=2, gammas=(0.2, 0.3, 0.5, 0.7, 1.0),
                                        alpha=0.20, delta=0.10, Lambda="box", box=0.15, seed=0, margin=0.01)
        cell["best_gamma_grouped_G2_a0.20"] = {k: _f(gg[k]) for k in CANON}
        out[rel] = cell
    _dump("certify_frozen.json", out)


if __name__ == "__main__":
    print("golden capture ->", os.path.relpath(GOLD, ROOT))
    snap_certificate_math()
    snap_scores_selector()
    snap_split_determinism()
    snap_certify_frozen()
    print("done.")
