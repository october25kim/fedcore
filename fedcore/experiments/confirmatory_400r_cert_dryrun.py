"""Confirmatory-400R engineering-only certification dry-run (license-neutral).

Given a common-schema per-obs artifact (see ``confirmatory_400r_common_schema``),
this builds the prospectively-frozen selector family

    score in {native, energy, margin}  x  gamma in {0.3, 0.5, 0.7, 1.0}  (M = 12)

on the PROPOSAL fold only, freezes each candidate's threshold, then reads the
CERTIFICATION fold to form per-candidate/per-client conditional counts
``(A_jm, K_jm)`` and ``n_j``, and runs the client FULL-SIMPLEX certificate two
ways over an alpha grid (incl. 0.20) at ``delta_r = delta_c = 0.05``:

* Theorem S1  -- simultaneous union-bound family certificate;
* Theorem S1' -- Holm/IUT risk + simultaneous coverage.

This is ENGINEERING-ONLY: it validates that the certification pipeline ingests
the exported schema and returns finite, well-typed numbers. It NEVER influences
design and its numeric outputs are not a scientific result.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from fedcore.experiments.confirmatory_400r_common_schema import load_fold_counts_inputs
from fedcore.officehome_rescue import (
    holm_family_certificate,
    select_family_candidate,
    family_keys,
    simultaneous_family_certificate,
)
from fedcore.selector import choose_threshold, counts_per_client

FAMILY_GAMMAS: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0)
FAMILY_SCORE_SLOTS: tuple[str, ...] = ("native_score", "energy_score", "known_margin_score")
ALPHA_GRID: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)


def _score_columns(inputs: dict) -> dict[str, np.ndarray]:
    return {slot: inputs[slot] for slot in FAMILY_SCORE_SLOTS}


def cert_dry_run(
    artifact_path: str,
    *,
    n_clients: int,
    alpha_grid: Sequence[float] = ALPHA_GRID,
    delta_r: float = 0.05,
    delta_c: float = 0.05,
    proposal_role: str = "proposal",
    cert_role: str = "certification",
) -> dict:
    prop = load_fold_counts_inputs(artifact_path, proposal_role)
    cert = load_fold_counts_inputs(artifact_path, cert_role)
    prop_scores = _score_columns(prop)
    cert_scores = _score_columns(cert)
    prop_pred = prop["predicted_known_index"]
    prop_y = prop["y_open"]
    cert_pred = cert["predicted_known_index"]
    cert_y = cert["y_open"]
    cert_client = cert["client"]
    n_j = np.array([int((cert_client == j).sum()) for j in range(n_clients)], dtype=int)

    per_alpha = {}
    for alpha in alpha_grid:
        # Freeze one threshold per (score, gamma) on the PROPOSAL fold only.
        A_rows, K_rows = [], []
        cand_meta = []
        for score_slot in FAMILY_SCORE_SLOTS:
            for gamma in FAMILY_GAMMAS:
                sel = choose_threshold(
                    prop_scores[score_slot], prop_pred, prop_y, float(gamma), float(alpha)
                )
                A, K, _n = counts_per_client(
                    cert_scores[score_slot], cert_pred, cert_y, cert_client, sel, n_clients
                )
                A_rows.append(A)
                K_rows.append(K)
                cand_meta.append(
                    {"score_slot": score_slot, "gamma": float(gamma),
                     "threshold_feasible": bool(sel.feasible)}
                )
        A_mat = np.asarray(A_rows, dtype=int)   # (M, J)
        K_mat = np.asarray(K_rows, dtype=int)
        sim = simultaneous_family_certificate(
            A_mat, K_mat, n_j, alpha=float(alpha), delta_r=delta_r, delta_c=delta_c
        )
        holm = holm_family_certificate(
            A_mat, K_mat, n_j, alpha=float(alpha), delta_r=delta_r, delta_c=delta_c
        )
        keys = family_keys(float(alpha))
        sim_pick = select_family_candidate(keys, sim.certified.tolist(), sim.C.tolist())
        holm_pick = select_family_candidate(keys, holm.certified.tolist(), holm.C.tolist())
        finite = bool(
            np.all(np.isfinite(sim.U)) and np.all(np.isfinite(sim.C))
            and np.all(np.isfinite(holm.pvalues)) and np.all(np.isfinite(holm.C))
        )
        per_alpha[f"{alpha:.2f}"] = {
            "M": int(sim.M),
            "J": int(sim.J),
            "cert_totals_per_client": n_j.tolist(),
            "simultaneous_n_certified": int(sim.certified.sum()),
            "simultaneous_best_coverage_lcb": (
                float(sim.C[sim_pick]) if sim_pick is not None else None),
            "simultaneous_min_risk_ucb": float(np.min(sim.U)),
            "holm_n_certified": int(holm.certified.sum()),
            "holm_best_coverage_lcb": (
                float(holm.C[holm_pick]) if holm_pick is not None else None),
            "holm_min_pvalue": float(np.min(holm.pvalues)),
            "n_threshold_feasible_candidates": int(sum(c["threshold_feasible"] for c in cand_meta)),
            "outputs_finite": finite,
        }

    return {
        "engineering_only": True,
        "note": "certification dry-run; never influences design",
        "family": {"scores": list(FAMILY_SCORE_SLOTS), "gammas": list(FAMILY_GAMMAS),
                   "M": len(FAMILY_SCORE_SLOTS) * len(FAMILY_GAMMAS)},
        "delta_r": delta_r,
        "delta_c": delta_c,
        "client_target": "full_simplex_worst_client",
        "per_alpha": per_alpha,
        "all_outputs_finite": bool(all(v["outputs_finite"] for v in per_alpha.values())),
    }
