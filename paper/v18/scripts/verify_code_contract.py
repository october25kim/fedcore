#!/usr/bin/env python3
"""Artifact-free executable checks for the v18 statistical API contract."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from fedcore.certificate import (
    cp_upper,
    full_simplex_fixed_member_certificate,
    pooled_cp,
    simple_simultaneous_family_certificate,
    zero_error_count_threshold,
)
from fedcore.certificate.holm import (
    holm_adjusted_pvalues,
    holm_family_certificate,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    require(zero_error_count_threshold(0.20, 0.05) == 14, "Theorem 4 floor")
    require(cp_upper(0, 13, 0.05) > 0.20, "A=13 must refuse")
    require(cp_upper(0, 14, 0.05) <= 0.20, "A=14 must pass the risk floor")

    fixed = full_simplex_fixed_member_certificate(
        [14, 30], [0, 0], [50, 60], delta_r=0.05, delta_c=0.05
    )
    require(fixed.risk_tail == 0.05, "fixed risk tail has a stratum divisor")
    require(fixed.coverage_tail == 0.05, "fixed coverage tail has a stratum divisor")

    A = np.array([[80, 90], [70, 85]])
    K = np.array([[0, 0], [1, 0]])
    n = np.array([120, 130])
    simple = simple_simultaneous_family_certificate(
        A, K, n, alpha=0.20, delta_r=0.05, delta_c=0.05
    )
    require(simple.risk_tail == 0.05 / 2, "simple family risk tail is not delta_r/M")
    require(simple.coverage_tail == 0.05 / 2, "simple family coverage tail is not delta_c/M")

    bounded = simple_simultaneous_family_certificate(
        A,
        K,
        n,
        alpha=0.20,
        delta_r=0.06,
        delta_c=0.04,
        mixture_target="bounded",
        lambda_lower=[0.20, 0.20],
        lambda_upper=[0.80, 0.80],
    )
    require(bounded.risk_tail == 0.06 / (3 * 2 * 2), "bounded family risk allocation")
    require(bounded.coverage_tail == 0.04 / (2 * 2), "bounded family coverage allocation")
    require(all(member.solver_certificate_valid for member in bounded.members),
            "bounded family solver did not validate")

    raw = np.array([0.001, 0.020, 0.500, 0.009])
    adjusted, ranks = holm_adjusted_pvalues(raw)
    require(np.allclose(adjusted, [0.004, 0.040, 0.500, 0.027]),
            "Holm adjusted p-values")
    require(np.array_equal(ranks, [1, 3, 4, 2]), "Holm stable ranks")
    holm = holm_family_certificate(
        A, K, n, alpha=0.20, delta_r=0.05, delta_c=0.05
    )
    require(holm.risk_ucb is None, "Holm must not report a numerical risk UCB")
    require(holm.eps_c == 0.05 / 2, "Holm coverage allocation is not delta_c/M")
    try:
        holm_family_certificate(
            A, K, n, alpha=0.20, delta_r=0.05, delta_c=0.05,
            mixture_target="bounded",
        )
    except ValueError:
        pass
    else:
        raise RuntimeError("bounded-mixture Holm/IUT was not rejected")

    try:
        pooled_cp([20, 20], [1, 2], 0.05, matched_mixture_iid=False)
    except ValueError:
        pass
    else:
        raise RuntimeError("pooled CP ran without the matched-mixture i.i.d. contract")
    require(math.isfinite(pooled_cp([20, 20], [1, 2], 0.05,
                                     matched_mixture_iid=True)),
            "matched-mixture pooled CP failed")

    print("FEDCORE V18 CODE CONTRACT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
