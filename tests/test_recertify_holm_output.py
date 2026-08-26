"""Release-contract tests for the Holm/IUT CIFAR recertification row."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fedcore.experiments.recertify_cifar_sweep import (
    ALPHA,
    DELTA_C,
    DELTA_R,
    N_CLIENTS,
    _row,
)
from fedcore.officehome_rescue import (
    family_keys,
    holm_family_certificate,
    select_family_candidate,
)


def _cell() -> dict[str, object]:
    return {
        "semantic_id": "fixture",
        "arm": "A",
        "dataset": "cifar10",
        "split_id": "0",
        "train_rep": "0",
        "d": "0.5",
    }


def _result(accepted: int) -> dict[str, object]:
    keys = family_keys(ALPHA)
    M = len(keys)
    A = np.full((M, N_CLIENTS), accepted, dtype=int)
    K = np.zeros_like(A)
    n = np.full(N_CLIENTS, 100, dtype=int)
    certificate = holm_family_certificate(
        A, K, n, alpha=ALPHA, delta_r=DELTA_R, delta_c=DELTA_C
    )
    selected = select_family_candidate(keys, certificate.certified, certificate.C)
    return {
        "keys": keys,
        "A": A,
        "K": K,
        "n": n,
        "fc": certificate,
        "sel_idx": selected,
        "prop_cov": np.zeros(M),
        "prop_risk": np.zeros(M),
        "test_cov": np.zeros(M),
        "test_risk": np.zeros(M),
    }


def test_selected_holm_row_is_fixed_alpha_decision_without_numeric_ucb() -> None:
    result = _result(accepted=100)
    row = _row(_cell(), result)
    selected = int(result["sel_idx"])
    certificate = result["fc"]

    assert row["certified"] == 1
    assert row["risk_output_type"] == "fixed_alpha_decision"
    assert row["family_procedure"] == "holm_iut"
    assert row["risk_pass"] is True
    assert row["selected_candidate_index"] == selected
    assert row["holm_risk_decision"] is True
    assert row["risk_decision_alpha"] == pytest.approx(ALPHA)
    assert math.isnan(row["cert_risk_ucb"])
    assert math.isnan(row["sim_worst_client_risk_ucb"])
    assert row["iut_raw_pvalue"] == pytest.approx(certificate.pvalues[selected])
    assert row["holm_iu_pvalue"] == pytest.approx(certificate.pvalues[selected])
    assert row["holm_adjusted_pvalue"] == pytest.approx(
        certificate.adjusted_pvalues[selected]
    )
    assert row["holm_rank"] == int(certificate.holm_rank[selected])
    assert row["delta"] == pytest.approx(DELTA_R + DELTA_C)
    assert row["delta_r"] == pytest.approx(DELTA_R)
    assert row["delta_c"] == pytest.approx(DELTA_C)


def test_refused_holm_row_keeps_decision_fields_null_and_ucb_nan() -> None:
    row = _row(_cell(), _result(accepted=0))

    assert row["certified"] == 0
    assert row["risk_output_type"] == "fixed_alpha_decision"
    assert row["family_procedure"] == "holm_iut"
    assert row["risk_pass"] is False
    assert row["selected_candidate_index"] is None
    assert row["holm_risk_decision"] is False
    assert math.isnan(row["cert_risk_ucb"])
    assert math.isnan(row["iut_raw_pvalue"])
    assert math.isnan(row["holm_iu_pvalue"])
    assert math.isnan(row["holm_adjusted_pvalue"])
    assert math.isnan(row["holm_rank"])
