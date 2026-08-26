"""Artifact-free tests for the current theorem-facing certificate API."""

from __future__ import annotations

import unittest

import numpy as np

from fedcore.certificate import (
    conditional_risk_certificate,
    cp_lower,
    cp_upper,
    full_simplex_fixed_member_certificate,
    pooled_cp,
)


class CurrentCertificateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.A = [120, 95, 140, 60, 110]
        self.K = [8, 6, 11, 9, 7]
        self.n = [400, 380, 420, 300, 360]

    def test_fixed_member_has_no_stratum_count_penalty(self) -> None:
        result = full_simplex_fixed_member_certificate(
            self.A,
            self.K,
            self.n,
            delta_r=0.05,
            delta_c=0.05,
        )
        self.assertEqual(result.risk_tail, 0.05)
        self.assertEqual(result.coverage_tail, 0.05)
        self.assertAlmostEqual(
            result.risk_ucb,
            max(cp_upper(k, a, 0.05) for a, k in zip(self.A, self.K)),
        )
        self.assertAlmostEqual(
            result.coverage_lcb,
            min(cp_lower(a, n, 0.05) for a, n in zip(self.A, self.n)),
        )

    def test_family_divides_by_members_not_strata(self) -> None:
        result = full_simplex_fixed_member_certificate(
            self.A,
            self.K,
            self.n,
            delta_r=0.05,
            delta_c=0.05,
            family_size=12,
        )
        self.assertAlmostEqual(result.risk_tail, 0.05 / 12)
        self.assertAlmostEqual(result.coverage_tail, 0.05 / 12)
        self.assertNotAlmostEqual(result.risk_tail, 0.05 / (12 * len(self.A)))

    def test_compact_simplex_api_defaults_to_current_theorem(self) -> None:
        current = conditional_risk_certificate(
            self.A, self.K, self.n, 0.05, Lambda="simplex"
        )
        legacy = conditional_risk_certificate(
            self.A,
            self.K,
            self.n,
            0.05,
            Lambda="simplex",
            legacy_simplex_union_bound=True,
        )
        self.assertEqual(current.eps, 0.05)
        self.assertEqual(legacy.eps, 0.05 / len(self.A))
        self.assertLessEqual(current.U, legacy.U)
        self.assertIn("no-S-penalty", current.method)
        self.assertIn("legacy", legacy.method)

    def test_bounded_box_is_conservative_and_seed_independent(self) -> None:
        first = conditional_risk_certificate(
            self.A,
            self.K,
            self.n,
            0.05,
            Lambda="box",
            box=0.15,
            n_lam_samples=1,
            seed=1,
        )
        second = conditional_risk_certificate(
            self.A,
            self.K,
            self.n,
            0.05,
            Lambda="box",
            box=0.15,
            n_lam_samples=10000,
            seed=999,
        )
        self.assertTrue(first.feasible)
        self.assertEqual(first.U, second.U)
        self.assertEqual(first.method, "theorem2-conservative-normalized-box")
        self.assertTrue(first.solver_certificate_valid)
        self.assertEqual(first.solver_status, "converged")
        self.assertGreater(first.solver_iterations, 0)
        self.assertGreaterEqual(first.solver_residual_lower, 0.0)
        self.assertLessEqual(first.solver_residual_upper, 0.0)

    def test_pooled_certificate_requires_iid_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "matched-mixture i.i.d"):
            pooled_cp(self.A, self.K, 0.05, matched_mixture_iid=False)
        got = pooled_cp(self.A, self.K, 0.05, matched_mixture_iid=True)
        self.assertAlmostEqual(got, cp_upper(sum(self.K), sum(self.A), 0.05))
        self.assertEqual(
            pooled_cp([0, 0], [0, 0], 0.05, matched_mixture_iid=True), 1.0
        )
        with self.assertRaisesRegex(ValueError, "K <= A"):
            pooled_cp([1], [2], 0.05, matched_mixture_iid=True)

    def test_invalid_count_order_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "K <= A <= n"):
            full_simplex_fixed_member_certificate(
                [10], [11], [20], delta_r=0.05, delta_c=0.05
            )


if __name__ == "__main__":
    unittest.main()
