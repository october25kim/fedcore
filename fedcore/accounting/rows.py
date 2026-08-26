"""Row builders -- the single source of truth for the accounting CSV schemas.

Every count here is derived from immutable sample IDs, never from logits and never
from a nominal draw counter. The distinction the whole phase turns on:

* ``requested_draw_count_n``            -- nominal draws (what a run's log reports)
* ``unique_sampled_count``              -- distinct labelled examples behind them
* ``unique_labels_used_in_certification_draw`` -- distinct labels the certificate saw
* ``operational_unique_trusted_labels`` -- distinct trusted labels a deployer must buy

For a without-replacement draw the first two coincide; for a with-replacement audit
draw they do not, and reporting the first as if it were the second inflates the
apparent evidence.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from fedcore.accounting.draws import (
    HEADLINE_DELTA,
    WITHOUT_REPLACEMENT_MODES,
    DrawResult,
)
from fedcore.accounting.occupancy import expected_unique_count, unique_and_multiplicity

LONG_FIELDS = [
    # identity / provenance
    "dataset",
    "model",
    "run_id",
    "provenance_source",
    "split_seed",
    "train_seed",
    "partition_seed",
    "audit_seed",
    "seeds_aliased",
    "heterogeneity_d",
    "noise_type",
    "noise_rate",
    # cell
    "target_alpha",
    "certificate_variant",
    "draw_mode",
    "draw_construction",
    "with_replacement",
    "fold",
    "stratum_type",
    "stratum_id",
    # reservoir vs draw
    "reservoir_size_M",
    "requested_draw_count_n",
    "unique_sampled_count",
    "duplicate_draw_count",
    "duplication_rate",
    "maximum_multiplicity",
    "expected_unique_count",
    "observed_minus_expected_unique",
    # accepted counts
    "accepted_draw_count_A",
    "accepted_error_count_K",
    "unique_accepted_sample_count",
    "unique_accepted_error_sample_count",
    "unique_accepted_fraction",
    # fold sizes (run level)
    "proposal_unique_count",
    "certification_reservoir_unique_count",
    "test_unique_count",
    # overlaps (run level, ID-based)
    "overlap_prop_cert",
    "overlap_cert_test",
    "overlap_prop_test",
    "overlap_certdraw_vs_proposal",
    "overlap_certdraw_vs_test",
    "overlap_certdraw_vs_original_proposal",
    "overlap_certdraw_vs_original_test",
    # label economics
    "operational_unique_trusted_labels",
    "research_only_test_labels",
    "nominal_certification_draws",
    "unique_labels_used_in_certification_draw",
    # failure budget (every component logged; see docs/reservoir_accounting_report.md)
    "delta_total_declared",
    "eps_per_stratum_risk",
    "n_cp_bounds_risk",
    "delta_spent_risk_certificate",
    "eps_per_stratum_coverage",
    "n_cp_bounds_coverage",
    "delta_spent_coverage_lcb",
    "delta_spent_joint_headline_claim",
    # certificate outcome (echoed, never recomputed differently)
    "certified",
    "cert_risk_ucb",
    "cert_coverage_lcb",
    "cert_n",
    "cert_k",
    "gamma_star",
]

OVERLAP_FIELDS = [
    "dataset",
    "model",
    "run_id",
    "heterogeneity_d",
    "protocol",
    "fold_a",
    "fold_b",
    "unique_a",
    "unique_b",
    "overlap_count",
    "overlap_is_expected_zero",
    "status",
]

GROUP_FIELDS = [
    "dataset",
    "model",
    "run_id",
    "heterogeneity_d",
    "target_alpha",
    "G",
    "draw_mode",
    "draw_construction",
    "audit_seed",
    "group_id",
    "client_id",
    "pi_j_given_g_declared",
    "pi_j_given_g_realized",
    "expected_client_draws",
    "realized_client_draws",
    "composition_discrepancy_abs",
    "unique_sampled_count_by_client",
    "group_total_draws",
    "declaration_source",
]

SUMMARY_FIELDS = [
    "dataset",
    "model",
    "heterogeneity_d",
    "target_alpha",
    "certificate_variant",
    "draw_mode",
    "n_runs",
    "n_strata",
    "reservoir_size_M_total",
    "requested_draw_count_n_total",
    "unique_sampled_count_total",
    "duplicate_draw_count_total",
    "duplication_rate_mean",
    "maximum_multiplicity_max",
    "accepted_draw_count_A_total",
    "accepted_error_count_K_total",
    "unique_accepted_sample_count_total",
    "unique_accepted_fraction_mean",
    "operational_unique_trusted_labels_total",
    "research_only_test_labels_total",
    "nominal_certification_draws_total",
    "unique_labels_used_in_certification_draw_total",
    "evidence_inflation_ratio",
]


def _budget(
    variant: str, n_strata: int, delta: float = HEADLINE_DELTA
) -> Dict[str, object]:
    """Explicit accounting of every Clopper-Pearson bound the cell pays for.

    The compact current path predeclares ``delta_r = delta_c = delta/2``.  A
    full-simplex member uses each whole member-level tail in every stratum and
    invokes the scalar-extremum argument, so the stratum count is not a union-bound
    multiplier. A strict box uses ``delta_r/(3J)`` for its simultaneous risk and
    acceptance endpoints and ``delta_c/J`` for simultaneous coverage endpoints.
    The joint headline claim spends at most ``delta_r + delta_c = delta``.
    """
    J = max(int(n_strata), 1)
    delta_r = delta / 2.0
    delta_c = delta / 2.0
    if "box" in variant:
        eps_risk, n_risk = delta_r / (3.0 * J), 3 * J
        eps_cov, n_cov = delta_c / J, J
        structure = "strict-mixture-simultaneous"
    else:
        eps_risk, n_risk = delta_r, J
        eps_cov, n_cov = delta_c, J
        structure = "full-simplex-scalar-extremum-no-J-penalty"
    return {
        "delta_total_declared": delta,
        "eps_per_stratum_risk": eps_risk,
        "n_cp_bounds_risk": n_risk,
        "delta_spent_risk_certificate": delta_r,
        "eps_per_stratum_coverage": eps_cov,
        "n_cp_bounds_coverage": n_cov,
        "delta_spent_coverage_lcb": delta_c,
        "delta_spent_joint_headline_claim": delta_r + delta_c,
        "multiplicity_structure": structure,
    }


def _n_unique(a) -> int:
    return int(len(np.unique(np.asarray(a)))) if len(a) else 0


def long_rows(
    spec, res: DrawResult, ids: Dict[str, np.ndarray], alpha: float
) -> List[dict]:
    """One row per stratum for a single (run, alpha, variant, draw_mode) cell."""
    orig = {f: set(ids[f].tolist()) for f in ("prop", "cert", "test")}
    # Fold sets under the protocol that produced THIS draw. The headline repartition
    # rebuilds its own prop/cert/test from the pooled trusted set; the audit draws
    # keep the original folds.
    proto = (
        {f: set(v.tolist()) for f, v in res.fold_ids.items()} if res.fold_ids else orig
    )
    cert_reservoir = proto.get("cert", orig["cert"])

    budget = _budget(res.certificate_variant, len(res.strata))
    row_cert = res.cert_row or {}
    without_replacement = res.draw_mode in WITHOUT_REPLACEMENT_MODES

    # Run-level label economics. Operational = what a deployer must actually label
    # (proposal + certification reservoir). Test is research-only: it never touches
    # the certificate and a deployed system would not buy it.
    operational = len(proto.get("prop", orig["prop"])) + len(cert_reservoir)
    research_only = len(proto.get("test", orig["test"]))

    all_cert_draw_ids = (
        np.concatenate([s.sampled_ids for s in res.strata])
        if res.strata
        else np.array([])
    )
    cert_draw_set = (
        set(np.unique(all_cert_draw_ids).tolist()) if len(all_cert_draw_ids) else set()
    )

    rows = []
    for s in res.strata:
        uniq, max_mult, _ = unique_and_multiplicity(s.sampled_ids)
        n = s.requested_draw_count_n
        dup = n - uniq
        # The occupancy formula is a with-replacement statement. Applying it to a
        # permutation draw would manufacture a nonsense expectation, so it is NaN
        # there -- the honest value, and asserted by the test suite.
        exp_uniq = (
            float("nan")
            if without_replacement
            else expected_unique_count(s.reservoir_size_M, n)
        )
        u_acc = _n_unique(s.accepted_ids)
        rows.append(
            {
                "dataset": spec.dataset,
                "model": spec.model,
                "run_id": spec.run_id,
                "provenance_source": spec.provenance_source,
                "split_seed": spec.split_seed,
                "train_seed": spec.train_seed,
                "partition_seed": spec.partition_seed,
                "audit_seed": res.audit_seed,
                "seeds_aliased": True,  # run_cifar drives all three from one --seed (F3)
                "heterogeneity_d": spec.heterogeneity_d,
                "noise_type": spec.noise_type,
                "noise_rate": spec.noise_rate,
                "target_alpha": alpha,
                "certificate_variant": res.certificate_variant,
                "draw_mode": res.draw_mode,
                "draw_construction": res.draw_construction,
                "with_replacement": res.replacement,
                "fold": "certification_draw",
                "stratum_type": s.stratum_type,
                "stratum_id": s.stratum_id,
                "reservoir_size_M": s.reservoir_size_M,
                "requested_draw_count_n": n,
                "unique_sampled_count": uniq,
                "duplicate_draw_count": dup,
                "duplication_rate": (1.0 - uniq / n) if n else 0.0,
                "maximum_multiplicity": max_mult,
                "expected_unique_count": exp_uniq,
                "observed_minus_expected_unique": (
                    float("nan") if without_replacement else uniq - exp_uniq
                ),
                "accepted_draw_count_A": s.accepted_draw_count_A,
                "accepted_error_count_K": s.accepted_error_count_K,
                "unique_accepted_sample_count": u_acc,
                "unique_accepted_error_sample_count": _n_unique(s.accepted_error_ids),
                "unique_accepted_fraction": (
                    u_acc / s.accepted_draw_count_A if s.accepted_draw_count_A else 0.0
                ),
                "proposal_unique_count": len(proto.get("prop", orig["prop"])),
                "certification_reservoir_unique_count": len(cert_reservoir),
                "test_unique_count": len(proto.get("test", orig["test"])),
                "overlap_prop_cert": len(
                    proto.get("prop", orig["prop"]) & cert_reservoir
                ),
                "overlap_cert_test": len(
                    cert_reservoir & proto.get("test", orig["test"])
                ),
                "overlap_prop_test": len(
                    proto.get("prop", orig["prop"]) & proto.get("test", orig["test"])
                ),
                "overlap_certdraw_vs_proposal": len(
                    cert_draw_set & proto.get("prop", orig["prop"])
                ),
                "overlap_certdraw_vs_test": len(
                    cert_draw_set & proto.get("test", orig["test"])
                ),
                "overlap_certdraw_vs_original_proposal": len(
                    cert_draw_set & orig["prop"]
                ),
                "overlap_certdraw_vs_original_test": len(cert_draw_set & orig["test"]),
                "operational_unique_trusted_labels": operational,
                "research_only_test_labels": research_only,
                "nominal_certification_draws": n,
                "unique_labels_used_in_certification_draw": uniq,
                **budget,
                "certified": row_cert.get("certified", ""),
                "cert_risk_ucb": row_cert.get("cert_risk_ucb", ""),
                "cert_coverage_lcb": row_cert.get("cert_coverage_lcb", ""),
                "cert_n": row_cert.get("cert_n", ""),
                "cert_k": row_cert.get("cert_k", ""),
                "gamma_star": row_cert.get("gamma_star", ""),
            }
        )
    return rows


def overlap_rows(spec, ids: Dict[str, np.ndarray], headline: DrawResult) -> List[dict]:
    """Pairwise fold overlaps by immutable ID, under both protocols.

    ``original``  -- the folds as run_cifar exported them. All three pairs MUST be
    empty; a non-zero entry is a split-hygiene violation.

    ``headline_repartition`` -- the folds aggregate.main._headline actually certifies
    on. These are also mutually disjoint (a permutation partition), but they are NOT
    the original folds: points that were TEST at export become CERTIFICATION points
    here. The ``vs_original`` rows quantify that reuse. It is reuse across protocols,
    not intra-cell leakage -- and the number belongs in the report either way.
    """
    rows = []
    pairs = (("prop", "cert"), ("cert", "test"), ("prop", "test"))
    for proto_name, sets in (
        ("original", {f: set(ids[f].tolist()) for f in ("prop", "cert", "test")}),
        (
            "headline_repartition",
            {f: set(v.tolist()) for f, v in headline.fold_ids.items()},
        ),
    ):
        for a, b in pairs:
            n = len(sets[a] & sets[b])
            rows.append(
                {
                    "dataset": spec.dataset,
                    "model": spec.model,
                    "run_id": spec.run_id,
                    "heterogeneity_d": spec.heterogeneity_d,
                    "protocol": proto_name,
                    "fold_a": a,
                    "fold_b": b,
                    "unique_a": len(sets[a]),
                    "unique_b": len(sets[b]),
                    "overlap_count": n,
                    "overlap_is_expected_zero": True,
                    "status": "OK" if n == 0 else "VIOLATION",
                }
            )
    # Cross-protocol reuse: headline cert fold vs the ORIGINAL folds. Expected non-zero
    # by construction; recorded so no reader assumes the original test fold stayed out.
    orig = {f: set(ids[f].tolist()) for f in ("prop", "cert", "test")}
    hcert = set(headline.fold_ids["cert"].tolist())
    for f in ("prop", "cert", "test"):
        rows.append(
            {
                "dataset": spec.dataset,
                "model": spec.model,
                "run_id": spec.run_id,
                "heterogeneity_d": spec.heterogeneity_d,
                "protocol": "headline_cert_vs_original",
                "fold_a": "headline_cert",
                "fold_b": f"original_{f}",
                "unique_a": len(hcert),
                "unique_b": len(orig[f]),
                "overlap_count": len(hcert & orig[f]),
                "overlap_is_expected_zero": False,
                "status": "EXPECTED_REUSE",
            }
        )
    return rows


def group_composition_rows(
    spec, z, res: DrawResult, ids, alpha: float, G: int
) -> List[dict]:
    """Declared vs realized within-group client composition for a grouped draw.

    ``pi_j_given_g_declared`` is the probability the protocol INTENDS each client to
    contribute inside its group:

    * ``grouped_iid_group`` -- the group's pool composition ``n_j / n_g``. The original
      code never writes this down: it draws single-stage from the pooled points, so the
      probability is implicit. Declaring it here is what makes it checkable.
    * ``grouped_quota``     -- uniform ``1/|g|`` by construction, independent of the
      group's actual composition. Where the pool is unbalanced, the realized composition
      is uniform BY DESIGN and the discrepancy against the deployment mixture is the
      documented Assumption-A6 caveat, not a bug.
    """
    from fedcore.grouping import make_group_map

    n_clients = int(z["cert_client"].max()) + 1
    gmap = make_group_map(n_clients, G)
    cert_client = np.asarray(z["cert_client"])
    cert_ids = ids["cert"]
    id_to_client = dict(zip(cert_ids.tolist(), cert_client.tolist()))

    rows = []
    for s in res.strata:
        g = s.stratum_id
        clients_g = [c for c in range(n_clients) if gmap[c] == g]
        n_g = int((gmap[cert_client] == g).sum())
        drawn_clients = np.array([id_to_client[i] for i in s.sampled_ids.tolist()])
        total = len(drawn_clients)
        for c in clients_g:
            n_c = int((cert_client == c).sum())
            if res.draw_mode == "grouped_quota":
                declared = 1.0 / len(clients_g) if clients_g else 0.0
                source = "uniform_by_quota_construction"
            else:
                declared = n_c / n_g if n_g else 0.0
                source = "pool_composition_implicit_in_single_stage_draw"
            realized = float((drawn_clients == c).sum()) / total if total else 0.0
            drawn_c = s.sampled_ids[drawn_clients == c]
            rows.append(
                {
                    "dataset": spec.dataset,
                    "model": spec.model,
                    "run_id": spec.run_id,
                    "heterogeneity_d": spec.heterogeneity_d,
                    "target_alpha": alpha,
                    "G": G,
                    "draw_mode": res.draw_mode,
                    "draw_construction": res.draw_construction,
                    "audit_seed": res.audit_seed,
                    "group_id": g,
                    "client_id": c,
                    "pi_j_given_g_declared": declared,
                    "pi_j_given_g_realized": realized,
                    "expected_client_draws": declared * total,
                    "realized_client_draws": int((drawn_clients == c).sum()),
                    "composition_discrepancy_abs": abs(realized - declared),
                    "unique_sampled_count_by_client": _n_unique(drawn_c),
                    "group_total_draws": total,
                    "declaration_source": source,
                }
            )
    return rows


def summarize(long: List[dict]) -> List[dict]:
    """Aggregate long rows per (dataset, model, d, alpha, variant, draw_mode)."""
    from collections import defaultdict

    cells = defaultdict(list)
    for r in long:
        cells[
            (
                r["dataset"],
                r["model"],
                r["heterogeneity_d"],
                r["target_alpha"],
                r["certificate_variant"],
                r["draw_mode"],
            )
        ].append(r)

    out = []
    for key, rs in sorted(cells.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        ds, model, d, alpha, variant, mode = key
        runs = {r["run_id"] for r in rs}
        n_tot = sum(r["requested_draw_count_n"] for r in rs)
        u_tot = sum(r["unique_sampled_count"] for r in rs)
        A_tot = sum(r["accepted_draw_count_A"] for r in rs)
        # Per-run figures must not be summed across strata (they are run-level).
        per_run = {r["run_id"]: r for r in rs}
        op_tot = sum(v["operational_unique_trusted_labels"] for v in per_run.values())
        test_tot = sum(v["research_only_test_labels"] for v in per_run.values())
        out.append(
            {
                "dataset": ds,
                "model": model,
                "heterogeneity_d": d,
                "target_alpha": alpha,
                "certificate_variant": variant,
                "draw_mode": mode,
                "n_runs": len(runs),
                "n_strata": len(rs),
                "reservoir_size_M_total": sum(r["reservoir_size_M"] for r in rs),
                "requested_draw_count_n_total": n_tot,
                "unique_sampled_count_total": u_tot,
                "duplicate_draw_count_total": sum(
                    r["duplicate_draw_count"] for r in rs
                ),
                "duplication_rate_mean": round(
                    float(np.mean([r["duplication_rate"] for r in rs])), 6
                ),
                "maximum_multiplicity_max": max(r["maximum_multiplicity"] for r in rs),
                "accepted_draw_count_A_total": A_tot,
                "accepted_error_count_K_total": sum(
                    r["accepted_error_count_K"] for r in rs
                ),
                "unique_accepted_sample_count_total": sum(
                    r["unique_accepted_sample_count"] for r in rs
                ),
                "unique_accepted_fraction_mean": round(
                    float(np.mean([r["unique_accepted_fraction"] for r in rs])), 6
                ),
                "operational_unique_trusted_labels_total": op_tot,
                "research_only_test_labels_total": test_tot,
                "nominal_certification_draws_total": n_tot,
                "unique_labels_used_in_certification_draw_total": u_tot,
                # >1 means nominal draws overstate the distinct labelled evidence.
                "evidence_inflation_ratio": (
                    round(n_tot / u_tot, 6) if u_tot else float("nan")
                ),
            }
        )
    return out
