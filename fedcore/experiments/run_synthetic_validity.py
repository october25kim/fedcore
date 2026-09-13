"""Plan-driven Monte Carlo validity checks for the one-shot certificate.

This runner intentionally contains no built-in paper grid.  The authoritative
grid belongs in the external one-shot plan; embedding a convenient replacement
here would turn an implementation smoke into an undeclared experiment.

Each sampling specification generates one immutable array of certification
counts.  Any number of analysis specifications (alpha, total delta, mixture
set, and allocation policy) are then evaluated on those same arrays.  The audit
seed depends only on ``campaign_seed`` and ``sampling_id`` -- never on a
post-hoc analysis knob.

Example::

    python -m fedcore.experiments.run_synthetic_validity \
        --plan path/to/predeclared_synthetic_plan.json \
        --output results/source_data/synthetic_validity.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from fedcore.budget import allocate_failure_budget, make_failure_budget
from fedcore.campaign.artifacts import semantic_hash
from fedcore.certificate.joint import joint_conditional_certificate
from fedcore.io_utils import atomic_write_csv, atomic_write_text
from fedcore.mixture import BoundedSimplex, solve_coverage_infimum, solve_robust_ratio
from fedcore.seeds import derive_seed


FIELDS = (
    "sampling_id",
    "analysis_id",
    "repetitions",
    "n_clients",
    "alpha",
    "total_delta",
    "allocation_policy",
    "Lambda",
    "true_risk_sup",
    "true_coverage_inf",
    "joint_bound_misses",
    "risk_bound_misses",
    "coverage_bound_misses",
    "unsafe_certificates",
    "certified_repetitions",
    "empirical_joint_miss_rate",
    "audit_seed",
    "counts_sha256",
    "sampling_config_hash",
    "analysis_config_hash",
)


@dataclass(frozen=True)
class SamplingCounts:
    A: np.ndarray
    K: np.ndarray
    n: np.ndarray
    audit_seed: int
    digest: str


def _probability_vector(value: Any, name: str, *, nonempty: bool = True) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or (nonempty and len(array) == 0):
        raise ValueError(f"{name} must be a non-empty vector")
    if np.any(~np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return array


def _positive_counts(value: Any, dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (dimension,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be an integer vector of length {dimension}")
    array = array.astype(np.int64, copy=False)
    if np.any(array <= 0):
        raise ValueError(f"{name} entries must be positive")
    return array


def _counts_digest(A: np.ndarray, K: np.ndarray, n: np.ndarray) -> str:
    h = hashlib.sha256()
    for name, value in (("A", A), ("K", K), ("n", n)):
        h.update(name.encode("ascii"))
        canonical = np.ascontiguousarray(value.astype("<i8", copy=False))
        h.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        h.update(canonical.tobytes())
    return h.hexdigest()


def validate_plan(plan: Mapping[str, Any]) -> None:
    if not isinstance(plan, Mapping) or plan.get("schema_version") != 1:
        raise ValueError("synthetic plan schema_version must be 1")
    if isinstance(plan.get("campaign_seed"), bool) or not isinstance(
        plan.get("campaign_seed"), int
    ):
        raise ValueError("campaign_seed must be an integer")
    samplings = plan.get("samplings")
    analyses = plan.get("analyses")
    if not isinstance(samplings, list) or not samplings:
        raise ValueError("samplings must be a non-empty list")
    if not isinstance(analyses, list) or not analyses:
        raise ValueError("analyses must be a non-empty list")

    sampling_ids: set[str] = set()
    for sampling in samplings:
        required = {
            "sampling_id",
            "repetitions",
            "n",
            "acceptance_probability",
            "conditional_risk",
            "proposal_A",
            "proposal_K",
        }
        if not isinstance(sampling, Mapping) or set(sampling) != required:
            raise ValueError(
                "each sampling must contain exactly " + ", ".join(sorted(required))
            )
        sid = sampling["sampling_id"]
        if not isinstance(sid, str) or not sid or sid in sampling_ids:
            raise ValueError("sampling_id values must be non-empty and unique")
        sampling_ids.add(sid)
        repetitions = sampling["repetitions"]
        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, int)
            or repetitions <= 0
        ):
            raise ValueError("repetitions must be a positive integer")
        acceptance = _probability_vector(
            sampling["acceptance_probability"], "acceptance_probability"
        )
        risk = _probability_vector(sampling["conditional_risk"], "conditional_risk")
        if risk.shape != acceptance.shape:
            raise ValueError("conditional_risk must align with acceptance_probability")
        _positive_counts(sampling["n"], len(acceptance), "n")
        proposal_A = np.asarray(sampling["proposal_A"])
        proposal_K = np.asarray(sampling["proposal_K"])
        if proposal_A.shape != acceptance.shape or proposal_K.shape != acceptance.shape:
            raise ValueError("proposal_A/proposal_K must align with clients")
        if not (
            np.issubdtype(proposal_A.dtype, np.integer)
            and np.issubdtype(proposal_K.dtype, np.integer)
        ):
            raise ValueError("proposal_A/proposal_K must be integer vectors")
        if (
            np.any(proposal_A < 0)
            or np.any(proposal_K < 0)
            or np.any(proposal_K > proposal_A)
        ):
            raise ValueError("proposal counts must satisfy 0 <= K <= A")

    analysis_ids: set[str] = set()
    for analysis in analyses:
        required = {
            "analysis_id",
            "sampling_id",
            "alpha",
            "total_delta",
            "allocation_policy",
            "Lambda",
        }
        optional = {"lambda_lower", "lambda_upper"}
        if not isinstance(analysis, Mapping) or not required <= set(analysis):
            raise ValueError("analysis is missing a required key")
        if set(analysis) - required - optional:
            raise ValueError("analysis contains an unsupported key")
        aid = analysis["analysis_id"]
        if not isinstance(aid, str) or not aid or aid in analysis_ids:
            raise ValueError("analysis_id values must be non-empty and unique")
        analysis_ids.add(aid)
        if analysis["sampling_id"] not in sampling_ids:
            raise ValueError("analysis references an unknown sampling_id")
        if not 0.0 < float(analysis["alpha"]) < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 < float(analysis["total_delta"]) < 1.0:
            raise ValueError("total_delta must lie in (0, 1)")
        if analysis["allocation_policy"] not in {"uniform", "proposal_informed"}:
            raise ValueError("unsupported allocation_policy")
        if analysis["Lambda"] not in {"simplex", "bounded"}:
            raise ValueError("Lambda must be simplex or bounded")
        bounded_keys = "lambda_lower" in analysis or "lambda_upper" in analysis
        if analysis["Lambda"] == "bounded" and not (
            "lambda_lower" in analysis and "lambda_upper" in analysis
        ):
            raise ValueError("bounded analysis requires both lambda bounds")
        if analysis["Lambda"] == "simplex" and bounded_keys:
            raise ValueError("simplex analysis must not contain lambda bounds")


def draw_sampling_counts(
    sampling: Mapping[str, Any], campaign_seed: int
) -> SamplingCounts:
    acceptance = _probability_vector(
        sampling["acceptance_probability"], "acceptance_probability"
    )
    risk = _probability_vector(sampling["conditional_risk"], "conditional_risk")
    n = _positive_counts(sampling["n"], len(acceptance), "n")
    repetitions = int(sampling["repetitions"])
    seed = derive_seed(
        campaign_seed,
        "audit_draw",
        experiment_id="synthetic_validity",
        sampling_id=sampling["sampling_id"],
        draw_index=0,
    )
    rng = np.random.default_rng(seed)
    A = rng.binomial(n[None, :], acceptance[None, :], size=(repetitions, len(n)))
    K = rng.binomial(A, risk[None, :])
    totals = np.broadcast_to(n, A.shape).copy()
    return SamplingCounts(
        A=A, K=K, n=totals, audit_seed=seed, digest=_counts_digest(A, K, totals)
    )


def _mixture(analysis: Mapping[str, Any], dimension: int) -> BoundedSimplex:
    if analysis["Lambda"] == "simplex":
        return BoundedSimplex(np.zeros(dimension), np.ones(dimension))
    return BoundedSimplex(
        analysis["lambda_lower"], analysis["lambda_upper"]
    ).tightened()


def evaluate_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_plan(plan)
    sampling_by_id = {value["sampling_id"]: value for value in plan["samplings"]}
    draws = {
        sid: draw_sampling_counts(sampling, int(plan["campaign_seed"]))
        for sid, sampling in sampling_by_id.items()
    }
    rows: list[dict[str, Any]] = []
    for analysis in plan["analyses"]:
        sampling = sampling_by_id[analysis["sampling_id"]]
        counts = draws[analysis["sampling_id"]]
        acceptance = np.asarray(sampling["acceptance_probability"], dtype=float)
        risk = np.asarray(sampling["conditional_risk"], dtype=float)
        mixture = _mixture(analysis, len(acceptance))
        true_ratio = solve_robust_ratio(risk, acceptance, acceptance, mixture)
        if not true_ratio.feasible:
            raise ValueError(
                "generative sampling cell has a vanishing robust denominator"
            )
        true_risk = float(true_ratio.value)
        true_coverage = float(solve_coverage_infimum(acceptance, mixture).value)
        bounded = analysis["Lambda"] == "bounded"
        budget = make_failure_budget(
            float(analysis["total_delta"]),
            include_mixture=False,
            include_acceptance_box=bounded,
        )
        allocations = allocate_failure_budget(
            budget,
            sampling["proposal_A"],
            sampling["proposal_K"],
            policy=analysis["allocation_policy"],
        )
        risk_misses = coverage_misses = joint_misses = 0
        unsafe_certificates = certified_repetitions = 0
        for A, K, n in zip(counts.A, counts.K, counts.n, strict=True):
            cert = joint_conditional_certificate(
                A,
                K,
                n,
                alpha=float(analysis["alpha"]),
                risk_eps=allocations["conditional_risk"],
                acceptance_lower_eps=allocations["acceptance_lower"],
                acceptance_upper_eps=allocations.get("acceptance_upper"),
                lambda_lower=(mixture.lower if bounded else None),
                lambda_upper=(mixture.upper if bounded else None),
            )
            risk_miss = cert.risk_ucb + 1e-14 < true_risk
            coverage_miss = cert.coverage_lcb - 1e-14 > true_coverage
            risk_misses += int(risk_miss)
            coverage_misses += int(coverage_miss)
            joint_misses += int(risk_miss or coverage_miss)
            certified_repetitions += int(cert.certified)
            unsafe_certificates += int(
                cert.certified and true_risk > float(analysis["alpha"])
            )
        repetitions = int(sampling["repetitions"])
        rows.append(
            {
                "sampling_id": sampling["sampling_id"],
                "analysis_id": analysis["analysis_id"],
                "repetitions": repetitions,
                "n_clients": len(acceptance),
                "alpha": float(analysis["alpha"]),
                "total_delta": float(analysis["total_delta"]),
                "allocation_policy": analysis["allocation_policy"],
                "Lambda": analysis["Lambda"],
                "true_risk_sup": true_risk,
                "true_coverage_inf": true_coverage,
                "joint_bound_misses": joint_misses,
                "risk_bound_misses": risk_misses,
                "coverage_bound_misses": coverage_misses,
                "unsafe_certificates": unsafe_certificates,
                "certified_repetitions": certified_repetitions,
                "empirical_joint_miss_rate": joint_misses / repetitions,
                "audit_seed": counts.audit_seed,
                "counts_sha256": counts.digest,
                "sampling_config_hash": semantic_hash(sampling),
                "analysis_config_hash": semantic_hash(analysis),
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    with open(args.plan, encoding="utf-8") as handle:
        plan = json.load(handle)
    rows = evaluate_plan(plan)
    atomic_write_csv(args.output, FIELDS, rows)
    manifest = {
        "schema_version": 1,
        "kind": "synthetic_validity",
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "row_count": len(rows),
        "output": os.path.abspath(args.output),
        "status": "succeeded",
    }
    atomic_write_text(
        args.output + ".manifest.json",
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
