"""F3b stage 2 -- embedding selector-family certification (Fed-ISIC, pure numpy).

Reads the per-cell embedding-score cache written by ``fedisic_f3b_extract --stage
embed`` (which only runs after the fidelity gate passes) and certifies exactly as
F3a did, swapping the four logit scores for the four TRAIN-referenced embedding
scores (cos_proto, cos_margin, neg_maha, neg_knn).  It reuses the audited fedcore
core verbatim: selector.choose_threshold / counts_per_client / open_set_error,
officehome_rescue.holm_family_certificate / simultaneous_family_certificate /
candidate_null_pvalue, certificate.cp.cp_lower.

Per-client FULL SIMPLEX, J=6 centers.  alpha in {0.10,0.15,0.20,0.25,0.30} (primary
0.20).  delta_r = delta_c = 0.05.  Route A = one proposal-only frozen policy,
certified on the actual 1x cert fold plus fresh-draw bootstrap 1x/2x/4x (seed =
blake2b("fedisic-fresh-v1|{cell}"), independent of alpha/score/gamma/outcome).
Route B = Holm + simultaneous family certificate over the frozen family.  No
cert/eval label enters candidate construction (references TRAIN only; thresholds
PROPOSAL only).  No canonical input is modified.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os

import numpy as np

from fedcore.selector import Selector, choose_threshold, counts_per_client, open_set_error
from fedcore.certificate.cp import cp_lower
from fedcore.officehome_rescue import (
    candidate_null_pvalue,
    holm_family_certificate,
    simultaneous_family_certificate,
)

OUT = os.environ.get("F3B_OUT", "/out")
EMB_DIR = os.path.join(OUT, "_emb")
SCORES = ["cos_proto", "cos_margin", "neg_maha", "neg_knn"]
GAMMAS = [0.2, 0.3, 0.5, 0.7]
ALPHAS = [0.10, 0.15, 0.20, 0.25, 0.30]
DELTA_R = 0.05
DELTA_C = 0.05
MULTS = [1, 2, 4]
NCLIENTS = 6


def fresh_seed(cell: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"fedisic-fresh-v1|{cell}".encode(), digest_size=8).digest(),
        "big",
    ) % (2**32)


def cand_hash(score, gamma, thr) -> str:
    return hashlib.sha256(f"{score}|{gamma:.6f}|{thr:.8g}".encode()).hexdigest()[:16]


def load_cell(path):
    d = np.load(path, allow_pickle=True)
    cell = dict(
        cell=str(d["cell"]),
        n_known=int(d["n_known"]),
        heldout=str(d["heldout"]),
        prop_logits=d["prop_logits"],
        prop_y=d["prop_y"].astype(int),
        prop_client=d["prop_client"].astype(int),
        cert_logits=d["cert_logits"],
        cert_y=d["cert_y"].astype(int),
        cert_client=d["cert_client"].astype(int),
        train_n=int(d["train_n"]),
    )
    cell["prop_score"] = {sc: d[f"prop_score__{sc}"] for sc in SCORES}
    cell["cert_score"] = {sc: d[f"cert_score__{sc}"] for sc in SCORES}
    return cell


def build_candidates(cell, alpha):
    """One prospectively frozen family (dedup by hash). Thresholds from PROPOSAL only."""
    prop_pred = cell["prop_logits"].argmax(1)
    prop_err = open_set_error(prop_pred, cell["prop_y"])
    cands, seen = [], set()
    for sc in SCORES:
        s_prop = cell["prop_score"][sc]
        for g in GAMMAS:
            sel = choose_threshold(s_prop, prop_pred, cell["prop_y"], g, alpha)
            h = cand_hash(sc, g, sel.threshold)
            if h in seen:
                continue
            seen.add(h)
            acc = sel.accept(s_prop)
            per_center_ok, per_center_acc, per_center_risk = True, [], []
            for j in range(NCLIENTS):
                m = cell["prop_client"] == j
                if m.sum() == 0:
                    continue
                aj = int((m & acc).sum())
                per_center_acc.append(aj / max(1, int(m.sum())))
                if aj > 0:
                    rj = float(prop_err[m & acc].mean())
                    per_center_risk.append(rj)
                    if rj > g * alpha:
                        per_center_ok = False
                else:
                    per_center_ok = False
            cands.append(
                dict(
                    score=sc,
                    gamma=g,
                    threshold=float(sel.threshold),
                    feasible=bool(sel.feasible),
                    hash=h,
                    prop_min_center_accept=min(per_center_acc) if per_center_acc else 0.0,
                    prop_max_center_risk=max(per_center_risk) if per_center_risk else 1.0,
                    prop_buffer_ok_all_centers=per_center_ok,
                )
            )
    return cands


def cert_counts(cell, cand, idx=None):
    """Per-client (A,K,n) on the cert fold (idx=None) or a bootstrap resample."""
    sel = Selector(threshold=cand["threshold"], feasible=cand["feasible"])
    cl = cell["cert_client"]
    yv = cell["cert_y"]
    pred = cell["cert_logits"].argmax(1)
    s = cell["cert_score"][cand["score"]]
    if idx is not None:
        cl, yv, pred, s = cl[idx], yv[idx], pred[idx], s[idx]
    A, K, n = counts_per_client(s, pred, yv, cl, sel, NCLIENTS)
    return A, K, n


def supported(n):
    return np.where(np.asarray(n) > 0)[0]


def route_a_select(cands):
    feas = [c for c in cands if c["feasible"] and c["prop_buffer_ok_all_centers"]]
    pool = feas if feas else [c for c in cands if c["feasible"]]
    if not pool:
        return None
    return min(
        pool,
        key=lambda c: (-c["prop_min_center_accept"], c["prop_max_center_risk"], c["gamma"], c["score"], c["hash"]),
    )


def main():
    paths = sorted(glob.glob(os.path.join(EMB_DIR, "*isic*.npz")))
    manifest, single_rows, holm_rows, acct_rows = [], [], [], []
    for p in paths:
        cell = load_cell(p)
        for alpha in ALPHAS:
            cands = build_candidates(cell, alpha)
            for c in cands:
                manifest.append(
                    dict(cell=cell["cell"], alpha=alpha, **{k: c[k] for k in (
                        "score", "gamma", "threshold", "feasible", "hash",
                        "prop_min_center_accept", "prop_max_center_risk",
                        "prop_buffer_ok_all_centers")})
                )
            M = len(cands)
            _, _, n0 = cert_counts(cell, cands[0])
            sup = supported(n0)
            if len(sup) == 0:
                continue
            A = np.zeros((M, len(sup)), int)
            K = np.zeros((M, len(sup)), int)
            nvec = n0[sup]
            for mi, c in enumerate(cands):
                a, k, n = cert_counts(cell, c)
                A[mi] = a[sup]
                K[mi] = k[sup]
            # Route B: Holm + simultaneous over the frozen family (J = supported)
            hc = holm_family_certificate(A, K, nvec, alpha=alpha, delta_r=DELTA_R, delta_c=DELTA_C)
            fc = simultaneous_family_certificate(A, K, nvec, alpha=alpha, delta_r=DELTA_R, delta_c=DELTA_C)
            n_holm = int(hc.certified.sum())
            n_sim = int(fc.certified.sum())
            holm_sel = None
            if n_holm > 0:
                idxs = np.where(hc.certified)[0]
                holm_sel = int(idxs[np.argmax(hc.C[idxs])])
            holm_rows.append(
                dict(
                    cell=cell["cell"], alpha=alpha, J_supported=len(sup), M=M,
                    holm_certified=n_holm, simultaneous_certified=n_sim,
                    holm_selected_hash=(cands[holm_sel]["hash"] if holm_sel is not None else ""),
                    holm_selected_score=(cands[holm_sel]["score"] if holm_sel is not None else ""),
                    holm_selected_gamma=(cands[holm_sel]["gamma"] if holm_sel is not None else ""),
                    holm_selected_cov_lcb=(float(hc.C[holm_sel]) if holm_sel is not None else ""),
                )
            )
            # Route A: single proposal-frozen policy, IUT on cert fold + bootstrap 1x/2x/4x
            ra = route_a_select(cands)
            if ra is not None:
                mi = next(i for i, c in enumerate(cands) if c["hash"] == ra["hash"])
                p_iut = candidate_null_pvalue(A[mi], K[mi], alpha)
                cov = min(cp_lower(int(A[mi][j]), int(nvec[j]), DELTA_C / len(sup)) for j in range(len(sup)))
                cert_1x = (p_iut <= DELTA_R) and cov > 0
                row = dict(
                    cell=cell["cell"], alpha=alpha, route="A_single_frozen",
                    score=ra["score"], gamma=ra["gamma"], J_supported=len(sup),
                    p_iut_1x=p_iut, cov_lcb_1x=cov, certified_1x=cert_1x,
                )
                seed = fresh_seed(cell["cell"])
                cl = cell["cert_client"]
                for mult in MULTS:
                    rng = np.random.default_rng(seed + mult)
                    idx_all = []
                    for j in sup:
                        pool = np.where(cl == j)[0]
                        idx_all.append(rng.choice(pool, size=len(pool) * mult, replace=True))
                    idx = np.concatenate(idx_all)
                    a, k, n = cert_counts(cell, ra, idx=idx)
                    pj = candidate_null_pvalue(a[sup], k[sup], alpha)
                    covb = min(cp_lower(int(a[sup][j]), int(n[sup][j]), DELTA_C / len(sup)) for j in range(len(sup)))
                    row[f"p_iut_{mult}x"] = pj
                    row[f"certified_{mult}x"] = (pj <= DELTA_R) and covb > 0
                    acct_rows.append(
                        dict(cell=cell["cell"], alpha=alpha, mult=mult, seed=seed,
                             draw_total=int(len(idx)), p_iut=pj, certified=row[f"certified_{mult}x"])
                    )
                single_rows.append(row)

    def dump(name, rows):
        path = os.path.join(OUT, name)
        if not rows:
            open(path, "w").write("EMPTY\n")
            return
        keys = list({k for r in rows for k in r})
        order = list(rows[0].keys()) + [k for k in keys if k not in rows[0]]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=order)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    dump("f3b_candidate_manifest.csv", manifest)
    dump("f3b_single_policy.csv", single_rows)
    dump("f3b_holm_family.csv", holm_rows)
    dump("f3b_accounting.csv", acct_rows)

    summ = []
    for alpha in ALPHAS:
        hb = [r for r in holm_rows if r["alpha"] == alpha]
        sa = [r for r in single_rows if r["alpha"] == alpha]
        summ.append(
            dict(
                alpha=alpha, n_cells=len(hb),
                routeB_holm_cells=sum(1 for r in hb if r["holm_certified"] > 0),
                routeB_simultaneous_cells=sum(1 for r in hb if r["simultaneous_certified"] > 0),
                routeA_1x_cells=sum(1 for r in sa if r["certified_1x"]),
                routeA_2x_cells=sum(1 for r in sa if r.get("certified_2x")),
                routeA_4x_cells=sum(1 for r in sa if r.get("certified_4x")),
            )
        )
    with open(os.path.join(OUT, "f3b_transition_matrix.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        for r in summ:
            w.writerow(r)

    print("=== F3b certification summary (cells with a valid positive; canonical eligible was 0/32) ===")
    print(f"{'alpha':>5} | {'cells':>5} | {'B_holm':>6} {'B_sim':>6} | {'A_1x':>5} {'A_2x':>5} {'A_4x':>5}")
    for r in summ:
        print(f"{r['alpha']:>5} | {r['n_cells']:>5} | {r['routeB_holm_cells']:>6} {r['routeB_simultaneous_cells']:>6} | "
              f"{r['routeA_1x_cells']:>5} {r['routeA_2x_cells']:>5} {r['routeA_4x_cells']:>5}")
    print(f"\nembedding cells processed: {len(paths)}")
    json.dump(
        {"n_cells": len(paths), "scores": SCORES, "gammas": GAMMAS, "alphas": ALPHAS,
         "delta_r": DELTA_R, "delta_c": DELTA_C, "mults": MULTS, "n_clients": NCLIENTS,
         "knn_k": 5, "maha_ridge": 1e-3, "summary": summ},
        open(os.path.join(OUT, "f3b_certify_summary.json"), "w"), indent=2,
    )


if __name__ == "__main__":
    main()
