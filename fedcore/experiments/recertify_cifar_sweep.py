"""Common fresh-audit recertification driver for the CIFAR backbone sweep.

Certifies ALL THREE arms (A = WRN PROSER exact-reuse, B = WRN plain, C = ResNeXt
plain) on ONE common selector family, reading each arm's common per-obs schema npz
(arm A from the Confirmatory-400R runs; arms B/C from runs/cifar_backbone_sweep).

Common primary family (owner spec): scores {MSP, energy, known-class margin} x
gammas {0.3, 0.5, 0.7, 1.0} = M=12 candidates; CLIENT-LEVEL FULL SIMPLEX; alpha=0.20;
Holm/IUT primary (fedcore.officehome_rescue.holm_family_certificate). The candidate
threshold is chosen on the PROPOSAL fold with the risk buffer gamma*alpha
(fedcore.selector.choose_threshold); per-client (A,K,n) come from the CERTIFICATION
fold (counts_per_client). MSP is recomputed from known_logits for EVERY arm so the
family is identical across arms; energy/known_margin come from the schema. The PROSER
dummy-vs-known score is reported as an arm-A SECONDARY diagnostic only (never in the
primary family, never cross-arm).

Fresh-audit-stream integrity: arms share the Confirmatory-400R block seeds (CRN), so
the proposal/certification source-ids MUST match across arms for a block; the driver
asserts this and fails closed otherwise.

Emits the canonical metric schema per cell:
  certified, cert_risk_ucb, cert_coverage_lcb, cert_n, cert_k, prop_coverage,
  prop_risk, test_coverage, test_risk, score_name, gamma, alpha, delta, Lambda,
  dirichlet_alpha, n_clients  (+ arm, dataset, split_id, train_rep, semantic_id).
For Holm/IUT, ``cert_risk_ucb`` remains NaN because the procedure returns a
fixed-alpha decision. The row instead records the raw IUT p-value, monotone
Holm-adjusted p-value, one-based Holm rank, and the risk-decision indicator.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from fedcore.experiments.confirmatory_400r_common_schema import load_fold_counts_inputs
from fedcore.officehome_rescue import (
    FAMILY_GAMMAS, FAMILY_SCORES, family_keys, holm_family_certificate,
    select_family_candidate,
)
from fedcore.selector import (
    choose_threshold, counts_per_client, empirical_risk_coverage, open_set_error,
)

ALPHA = 0.20
DELTA_R = 0.05
DELTA_C = 0.05
N_CLIENTS = 5


def _msp(known_logits: np.ndarray) -> np.ndarray:
    z = np.asarray(known_logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).max(axis=1)


def _score_array(fold: dict, name: str) -> np.ndarray:
    """The three common-family scores, uniform across arms (higher => accept/known)."""
    if name == "msp":
        return _msp(fold["known_logits"])
    if name == "energy":
        return fold["energy_score"]
    if name == "margin":
        return fold["known_margin_score"]
    raise ValueError(f"unknown score {name}")


def _cell_npz(cell, arm_a_root, fresh_root):
    if cell["reuse_class"] == "exact_reuse":
        return os.path.join(arm_a_root, f"{cell['ref_semantic_id']}_common.npz")
    return os.path.join(fresh_root, f"{cell['semantic_id']}_common.npz")


def certify_cell(npz_path):
    """Return the M=12 family certificate + selected candidate for one cell."""
    prop = load_fold_counts_inputs(npz_path, "proposal")
    cert = load_fold_counts_inputs(npz_path, "certification")
    test = load_fold_counts_inputs(npz_path, "test")
    keys = family_keys(ALPHA)
    M = len(keys)
    A = np.zeros((M, N_CLIENTS), dtype=int)
    K = np.zeros((M, N_CLIENTS), dtype=int)
    n_vec = None
    prop_cov = np.zeros(M); prop_risk = np.zeros(M)
    test_cov = np.zeros(M); test_risk = np.zeros(M)
    for m, key in enumerate(keys):
        ps = _score_array(prop, key.score_name)
        cs = _score_array(cert, key.score_name)
        ts = _score_array(test, key.score_name)
        sel = choose_threshold(ps, prop["predicted_known_index"], prop["y_open"],
                               key.gamma, ALPHA)
        Aj, Kj, nj = counts_per_client(cs, cert["predicted_known_index"], cert["y_open"],
                                       cert["client"], sel, N_CLIENTS)
        A[m] = Aj; K[m] = Kj
        n_vec = nj if n_vec is None else n_vec
        # descriptive proposal/test risk-coverage at the chosen threshold
        pe = open_set_error(prop["predicted_known_index"], prop["y_open"])
        te = open_set_error(test["predicted_known_index"], test["y_open"])
        prop_cov[m], prop_risk[m] = empirical_risk_coverage(ps, pe, sel.threshold)
        test_cov[m], test_risk[m] = empirical_risk_coverage(ts, te, sel.threshold)
    fc = holm_family_certificate(A, K, n_vec, alpha=ALPHA, delta_r=DELTA_R, delta_c=DELTA_C)
    sel_idx = select_family_candidate(keys, fc.certified, fc.C)
    return dict(keys=keys, A=A, K=K, n=n_vec, fc=fc, sel_idx=sel_idx,
                prop_cov=prop_cov, prop_risk=prop_risk, test_cov=test_cov, test_risk=test_risk,
                prop_src=np.asarray(np.load(npz_path, allow_pickle=False)["proposal__immutable_source_id"]),
                cert_src=np.asarray(np.load(npz_path, allow_pickle=False)["certification__immutable_source_id"]))


def _row(cell, res):
    keys = res["keys"]; fc = res["fc"]; i = res["sel_idx"]
    if i is None:
        certified = False
        cert_risk_ucb = float("nan")       # Holm is a decision, not a numerical UCB
        sim_risk_ucb = float("nan")        # retained column; not reported for Holm
        cov_lcb = 0.0
        sname = ""; gamma = ""
        cert_n = 0; cert_k = 0
        pc = pr = tc = tr = float("nan")
        pval = adjusted_pval = holm_rank = float("nan")
        holm_risk_decision = False
    else:
        certified = True
        cov_lcb = float(fc.C[i])
        # PRIMARY (Holm/IUT): a fixed-alpha decision at FWER delta_r. It does not
        # assign a numerical risk UCB unless the test is explicitly inverted.
        cert_risk_ucb = float("nan")
        sim_risk_ucb = float("nan")
        pval = float(fc.pvalues[i])
        adjusted_pval = float(fc.adjusted_pvalues[i])
        holm_rank = int(fc.holm_rank[i])
        holm_risk_decision = bool(fc.holm_reject[i])
        sname = keys[i].score_name; gamma = keys[i].gamma
        cert_n = int(res["n"].sum()); cert_k = int(res["K"][i].sum())
        pc = float(res["prop_cov"][i]); pr = float(res["prop_risk"][i])
        tc = float(res["test_cov"][i]); tr = float(res["test_risk"][i])
    # coverage@alpha is the zero-imputed selected coverage LCB
    coverage_at_alpha = cov_lcb if certified else 0.0
    return {
        "semantic_id": cell["semantic_id"], "arm": cell["arm"], "dataset": cell["dataset"],
        "split_id": cell["split_id"], "train_rep": cell["train_rep"], "d": cell["d"],
        "certified": int(certified), "cert_risk_ucb": cert_risk_ucb,
        "risk_output_type": "fixed_alpha_decision", "risk_pass": holm_risk_decision,
        "family_procedure": "holm_iut", "selected_candidate_index": i,
        "risk_decision_alpha": ALPHA, "holm_risk_decision": holm_risk_decision,
        "sim_worst_client_risk_ucb": sim_risk_ucb, "iut_raw_pvalue": pval,
        "holm_iu_pvalue": pval,
        "holm_adjusted_pvalue": adjusted_pval, "holm_rank": holm_rank,
        "cert_coverage_lcb": coverage_at_alpha,
        "cert_n": cert_n, "cert_k": cert_k,
        "prop_coverage": pc, "prop_risk": pr, "test_coverage": tc, "test_risk": tr,
        "score_name": sname, "gamma": gamma, "alpha": ALPHA,
        "delta": DELTA_R + DELTA_C, "delta_r": DELTA_R, "delta_c": DELTA_C,
        "Lambda": "client_full_simplex", "dirichlet_alpha": cell["d"], "n_clients": N_CLIENTS,
        "certificate": "holm_iut_full_simplex_common_family",
    }


def _check_stream_integrity(rows_by_block):
    """Assert prop/cert source-ids match across arms within each (dataset,split,rep,d)."""
    mismatches = []
    for block, per_arm in rows_by_block.items():
        refs = list(per_arm.values())
        base_p, base_c = refs[0]
        for (p, c) in refs[1:]:
            if not (np.array_equal(p, base_p) and np.array_equal(c, base_c)):
                mismatches.append(block)
                break
    return mismatches


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default="results/cifar_backbone_sweep/final_training_matrix.csv")
    ap.add_argument("--arm-a-root", required=True,
                    help="dir of Confirmatory-400R arm-A common npz ({ref_semantic_id}_common.npz)")
    ap.add_argument("--fresh-root", default="runs/cifar_backbone_sweep/cells")
    ap.add_argument("--out", default="results/cifar_backbone_sweep/recert/per_cell_certificates.csv")
    ap.add_argument("--require-all", action="store_true",
                    help="fail closed if any cell npz is missing (else skip + record)")
    args = ap.parse_args(argv)

    rows = list(csv.DictReader(open(args.matrix, newline="", encoding="utf-8-sig")))
    out_rows = []
    missing = []
    integrity = {}
    for cell in rows:
        npz = _cell_npz(cell, args.arm_a_root, args.fresh_root)
        if not os.path.isfile(npz):
            missing.append(cell["semantic_id"])
            continue
        res = certify_cell(npz)
        out_rows.append(_row(cell, res))
        block = f"{cell['dataset']}__{cell['split_id']}__seed{cell['train_rep']}__d{cell['d']}"
        integrity.setdefault(block, {})[cell["arm"]] = (res["prop_src"], res["cert_src"])

    if args.require_all and missing:
        raise SystemExit(f"FAIL_CLOSED: {len(missing)} cell npz missing (first: {missing[:3]})")

    mismatches = _check_stream_integrity(integrity)
    if mismatches:
        raise SystemExit(f"FAIL_CLOSED: fresh-audit-stream source-id mismatch across arms "
                         f"in {len(mismatches)} blocks (first: {mismatches[:3]})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if out_rows:
        cols = list(out_rows[0].keys())
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader(); w.writerows(out_rows)
    summary = {
        "certified_cells": len(out_rows), "missing": len(missing),
        "stream_integrity_mismatch_blocks": len(mismatches),
        "family": {"scores": list(FAMILY_SCORES), "gammas": list(FAMILY_GAMMAS),
                   "M": len(FAMILY_SCORES) * len(FAMILY_GAMMAS)},
        "alpha": ALPHA, "certificate": "holm_iut_client_full_simplex",
        "note": "PROSER dummy-vs-known is an arm-A secondary diagnostic; not in this primary family.",
    }
    json.dump(summary, open(os.path.join(os.path.dirname(args.out), "recert_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
