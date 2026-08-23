"""Deterministic replay of every reservoir draw, carrying immutable sample IDs.

Three draw families exist in this repo and they obey DIFFERENT sampling laws. The
whole point of this module is to keep them apart, because conflating them is exactly
how "18,000 evaluations" gets mistaken for 18,000 labelled examples.

``HEADLINE_REPARTITION``
    ``fedcore.aggregate.main._headline`` pools prop+cert+test from the npz and
    re-splits it with ``repartition_trusted_pool(..., seed=0)``. That is an
    ``rng.permutation`` -- WITHOUT replacement. Unique count == n by construction;
    duplication is structurally zero; the occupancy formula does NOT apply and is
    reported as NaN rather than misapplied. This is the draw behind every published
    ``CertCov@alpha`` cell.

``AUDIT_BOOTSTRAP``
    The protocol of ``exp_resampling_validity`` / ``exp_fcp_recast_resampling``:
    per-client draws WITH replacement from a frozen pool. Duplication is real and
    the occupancy formula applies.

``GROUPED_QUOTA`` / ``GROUPED_IID_GROUP``
    The two modes of ``exp_grouped_validity_stress.run_real``, which differ in how
    the within-group client composition pi_{j|g} arises: a fixed quota, versus a
    single-stage draw from the group's pooled points.

Replay fidelity is not asserted, it is checked: ``headline_draw`` recomputes A/K from
the replayed IDs and cross-checks them against ``certify_best_gamma``'s own
``cert_n``/``cert_k`` on the same view. A mismatch raises.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from fedcore.certify import certify_best_gamma, certify_best_gamma_grouped
from fedcore.grouping import make_group_map, repartition_trusted_pool
from fedcore.scores import scored_views
from fedcore.selector import counts_per_client, open_set_error

# Draw modes.
HEADLINE_REPARTITION = "headline_repartition"
AUDIT_BOOTSTRAP = "audit_bootstrap"
GROUPED_QUOTA = "grouped_quota"
GROUPED_IID_GROUP = "grouped_iid_group"

# Draw constructions (how the sample was actually built).
PERMUTATION_WITHOUT_REPLACEMENT = "permutation_without_replacement"
FIXED_QUOTA_WITH_REPLACEMENT = "fixed_quota_with_replacement"
POOLED_SINGLE_STAGE_WITH_REPLACEMENT = "pooled_single_stage_with_replacement"
CATEGORICAL_THEN_RESERVOIR = "categorical_then_reservoir"

WITHOUT_REPLACEMENT_MODES = frozenset({HEADLINE_REPARTITION})

# Mirrors fedcore.aggregate.main -- the headline cell definition. Imported as
# constants rather than re-derived so accounting cannot silently drift from it.
from fedcore.aggregate.main import (  # noqa: E402
    CERT_FRAC as HEADLINE_CERT_FRAC,
    DELTA as HEADLINE_DELTA,
    GAMMAS as HEADLINE_GAMMAS,
    MARGIN as HEADLINE_MARGIN,
)

HEADLINE_TEST_FRAC = 0.2  # aggregate.main._headline passes 0.2 positionally
HEADLINE_REPARTITION_SEED = 0  # aggregate.main._headline passes seed=0
HEADLINE_SCORE = "msp"  # aggregate.main._headline main path: SCORES=("msp",)
HEADLINE_BOX = 0.15


def derive_audit_seed(*parts: str) -> int:
    """Deterministic, cross-machine-stable audit seed from string parts.

    Python's ``hash()`` is salted per process; blake2b is not. Reproducibility of the
    sampled ID sequence depends on this being stable across interpreters and hosts.
    """
    h = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=4)
    return int(h.hexdigest(), 16)


@dataclass
class StratumDraw:
    """One stratum's draw: which IDs were drawn, how often, and what it certified."""

    stratum_type: str  # 'client' | 'group'
    stratum_id: int
    reservoir_size_M: int
    requested_draw_count_n: int
    sampled_ids: np.ndarray  # with multiplicity, in draw order
    accepted_draw_count_A: int
    accepted_error_count_K: int
    accepted_ids: np.ndarray  # accepted draws, with multiplicity
    accepted_error_ids: np.ndarray  # accepted-and-wrong draws, with multiplicity


@dataclass
class DrawResult:
    """A complete draw over all strata, plus the certificate it fed."""

    draw_mode: str
    draw_construction: str
    replacement: bool
    audit_seed: int
    strata: List[StratumDraw] = field(default_factory=list)
    certificate_variant: str = ""
    cert_row: Optional[dict] = None
    # fold-level ID sets under the protocol that actually produced the draw
    fold_ids: Dict[str, np.ndarray] = field(default_factory=dict)


def pooled_reservoir(z, ids: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The trusted pool exactly as aggregate.main._headline builds it.

    Fold concatenation order (prop, cert, test) is load-bearing: it fixes the
    permutation's meaning, so it must match _headline's comprehension exactly.
    """
    pool = {
        k: np.concatenate([z[f"{f}_{k}"] for f in ("prop", "cert", "test")])
        for k in ("logits", "y_open", "client")
    }
    pool["sample_id"] = np.concatenate([ids[f] for f in ("prop", "cert", "test")])
    return pool


def _strata_from_view(view, sel, ids_view, stratum_type, n_strata, reservoir_sizes):
    """Build per-stratum draw records from a scored view + a fixed selector."""
    err = open_set_error(view["pred"], view["y_open"])
    accept = sel.accept(view["score"])
    client = np.asarray(view["client"])
    out = []
    for j in range(n_strata):
        m = client == j
        acc_j = m & accept
        out.append(
            StratumDraw(
                stratum_type=stratum_type,
                stratum_id=j,
                reservoir_size_M=int(reservoir_sizes[j]),
                requested_draw_count_n=int(m.sum()),
                sampled_ids=ids_view[m],
                accepted_draw_count_A=int(acc_j.sum()),
                accepted_error_count_K=int((acc_j & err).sum()),
                accepted_ids=ids_view[acc_j],
                accepted_error_ids=ids_view[acc_j & err],
            )
        )
    return out


def headline_draw(
    spec, z, ids: Dict[str, np.ndarray], alpha: float, G: Optional[int]
) -> DrawResult:
    """Replay one published headline cell's certification draw, carrying IDs.

    Reproduces ``fedcore.aggregate.main._headline`` for the MAIN (fixed-MSP) path and
    cross-checks the replayed A/K against the certificate's own cert_n/cert_k.

    ``G=None`` -> per-client stratification (the CertCovGJ column);
    ``G=2``   -> grouped-stratified (the CertCovG2 headline column).
    """
    n_clients = int(z["cert_client"].max()) + 1
    pool = pooled_reservoir(z, ids)
    parts = repartition_trusted_pool(
        pool, HEADLINE_CERT_FRAC, HEADLINE_TEST_FRAC, seed=HEADLINE_REPARTITION_SEED
    )
    views = {
        fn: scored_views(
            parts[fn]["logits"],
            parts[fn]["y_open"],
            parts[fn]["client"],
            [HEADLINE_SCORE],
        )[HEADLINE_SCORE]
        for fn in ("prop", "cert", "test")
    }

    if G is None or G >= n_clients:
        row = certify_best_gamma(
            views["prop"],
            views["cert"],
            views["test"],
            score_name=HEADLINE_SCORE,
            gammas=HEADLINE_GAMMAS,
            alpha=alpha,
            delta=HEADLINE_DELTA,
            n_clients=n_clients,
            dirichlet_alpha=float("nan"),
            Lambda="box",
            box=HEADLINE_BOX,
            seed=0,
            margin=HEADLINE_MARGIN,
        )
        stratum_type, n_strata = "client", n_clients
        cert_stratum = np.asarray(parts["cert"]["client"])
        variant = "thm1prime_box_perclient"
    else:
        gmap = make_group_map(n_clients, G)
        row = certify_best_gamma_grouped(
            views["prop"],
            views["cert"],
            views["test"],
            score_name=HEADLINE_SCORE,
            group_map=gmap,
            G=G,
            gammas=HEADLINE_GAMMAS,
            alpha=alpha,
            delta=HEADLINE_DELTA,
            Lambda="box",
            box=HEADLINE_BOX,
            seed=0,
            margin=HEADLINE_MARGIN,
        )
        stratum_type, n_strata = "group", G
        cert_stratum = gmap[np.asarray(parts["cert"]["client"])]
        variant = f"thm1prime_box_grouped_G{G}"

    # Re-derive the selector the certificate actually used: gamma* is chosen on the
    # proposal fold, so replaying choose_threshold at gamma* reproduces it exactly.
    from fedcore.selector import choose_threshold

    prop_view = dict(views["prop"])
    if stratum_type == "group":
        prop_view["client"] = make_group_map(n_clients, G)[
            np.asarray(views["prop"]["client"])
        ]
    sel = choose_threshold(
        prop_view["score"],
        prop_view["pred"],
        prop_view["y_open"],
        row["gamma_star"],
        alpha,
    )

    cert_view = dict(views["cert"])
    cert_view["client"] = cert_stratum
    reservoir_sizes = np.array(
        [
            int((np.asarray(pool_stratum(pool, n_clients, G)) == j).sum())
            for j in range(n_strata)
        ]
    )
    strata = _strata_from_view(
        cert_view,
        sel,
        np.asarray(parts["cert"]["sample_id"]),
        stratum_type,
        n_strata,
        reservoir_sizes,
    )

    # Fail loudly if the replay and the certification pipeline disagree.
    A = np.array([s.accepted_draw_count_A for s in strata])
    K = np.array([s.accepted_error_count_K for s in strata])
    A_ref, K_ref, _ = counts_per_client(
        cert_view["score"],
        cert_view["pred"],
        cert_view["y_open"],
        cert_stratum,
        sel,
        n_strata,
    )
    if not (np.array_equal(A, A_ref) and np.array_equal(K, K_ref)):
        raise ValueError(f"{spec.run_id}: replayed A/K disagree with counts_per_client")
    if int(A.sum()) != int(row["cert_n"]) or int(K.sum()) != int(row["cert_k"]):
        raise ValueError(
            f"{spec.run_id}: accounting A/K ({A.sum()},{K.sum()}) != certificate "
            f"cert_n/cert_k ({row['cert_n']},{row['cert_k']}) -- accounting would "
            f"misreport the published cell"
        )

    return DrawResult(
        draw_mode=HEADLINE_REPARTITION,
        draw_construction=PERMUTATION_WITHOUT_REPLACEMENT,
        replacement=False,
        audit_seed=HEADLINE_REPARTITION_SEED,
        strata=strata,
        certificate_variant=variant,
        cert_row=row,
        fold_ids={
            fn: np.asarray(parts[fn]["sample_id"]) for fn in ("prop", "cert", "test")
        },
    )


def pool_stratum(pool, n_clients: int, G: Optional[int]) -> np.ndarray:
    """Stratum label of every pooled point (client id, or its group id)."""
    client = np.asarray(pool["client"])
    if G is None or G >= n_clients:
        return client
    return make_group_map(n_clients, G)[client]


def bootstrap_draw(
    spec, z, ids, alpha: float, ratio: float, audit_seed: int
) -> DrawResult:
    """Per-client WITH-replacement audit draw from the frozen cert+test pool.

    This replays the PROTOCOL of ``exp_resampling_validity`` (pool = cert + test;
    per-client bootstrap at the original per-client certification sizes), under an
    explicitly declared per-run ``audit_seed``.

    It is NOT a reproduction of that script's historical draw sequence: it uses one
    shared RNG advanced across every run in a sorted glob and optionally sharded by
    argv, so its stream depends on how many runs were processed first. That is a
    reproducibility defect of the study, reported in the accounting report, not
    something this replay can paper over.

    ``ratio`` scales the per-client draw count to sweep n/M for the occupancy figure;
    ``ratio=1.0`` is the study's own operating point.
    """
    n_clients = int(z["cert_client"].max()) + 1
    pool_logits = np.concatenate([z["cert_logits"], z["test_logits"]])
    pool_y = np.concatenate([z["cert_y_open"], z["test_y_open"]])
    pool_client = np.concatenate([z["cert_client"], z["test_client"]])
    pool_ids = np.concatenate([ids["cert"], ids["test"]])

    view = scored_views(pool_logits, pool_y, pool_client, [HEADLINE_SCORE])[
        HEADLINE_SCORE
    ]
    prop_view = scored_views(
        z["prop_logits"], z["prop_y_open"], z["prop_client"], [HEADLINE_SCORE]
    )[HEADLINE_SCORE]
    from fedcore.selector import choose_threshold

    sel = choose_threshold(
        prop_view["score"], prop_view["pred"], prop_view["y_open"], 1.0, alpha
    )

    err = open_set_error(view["pred"], view["y_open"])
    accept = sel.accept(view["score"])
    strata = []
    for j in range(n_clients):
        member = np.flatnonzero(pool_client == j)
        n_j = int(round(int((z["cert_client"] == j).sum()) * ratio))
        if len(member) == 0 or n_j <= 0:
            strata.append(
                StratumDraw(
                    "client",
                    j,
                    len(member),
                    0,
                    np.array([], dtype=object),
                    0,
                    0,
                    np.array([], dtype=object),
                    np.array([], dtype=object),
                )
            )
            continue
        # One immutable per-client stream makes every audit-budget fraction a
        # prefix of the larger draw. The seed is independent of ratio/policy/score.
        client_seed = derive_audit_seed(str(audit_seed), "client", str(j))
        rng = np.random.default_rng(client_seed)
        drawn = rng.choice(member, size=n_j, replace=True)
        acc = accept[drawn]
        strata.append(
            StratumDraw(
                stratum_type="client",
                stratum_id=j,
                reservoir_size_M=len(member),
                requested_draw_count_n=n_j,
                sampled_ids=pool_ids[drawn],
                accepted_draw_count_A=int(acc.sum()),
                accepted_error_count_K=int((acc & err[drawn]).sum()),
                accepted_ids=pool_ids[drawn][acc],
                accepted_error_ids=pool_ids[drawn][acc & err[drawn]],
            )
        )
    return DrawResult(
        draw_mode=AUDIT_BOOTSTRAP,
        draw_construction=FIXED_QUOTA_WITH_REPLACEMENT,
        replacement=True,
        audit_seed=audit_seed,
        strata=strata,
        certificate_variant="thm1_simplex_perclient_resampled",
    )


def grouped_draw(
    spec, z, ids, alpha: float, G: int, mode: str, n_per_client: int, audit_seed: int
) -> DrawResult:
    """Grouped audit draw in QUOTA or IID-GROUP mode, recording pi_{j|g}.

    Mirrors ``exp_grouped_validity_stress.run_real``. The two modes differ precisely
    in how the within-group client composition arises:

    * ``quota``     -- fixed ``n_per_client`` draws from each client's own points, so
      the realized composition is uniform-over-clients by construction, regardless of
      the group's actual composition.
    * ``iid_group`` -- a single-stage draw from the group's POOLED points, so the
      realized composition follows the pool shares ``pi_{j|g} = n_j / n_g`` in
      expectation -- but only implicitly. Nothing in the original code declares those
      probabilities or checks them; that is the gap this record closes.
    """
    n_clients = int(z["cert_client"].max()) + 1
    gmap = make_group_map(n_clients, G)
    view = scored_views(
        z["cert_logits"], z["cert_y_open"], z["cert_client"], [HEADLINE_SCORE]
    )[HEADLINE_SCORE]
    prop_view = scored_views(
        z["prop_logits"], z["prop_y_open"], z["prop_client"], [HEADLINE_SCORE]
    )[HEADLINE_SCORE]
    from fedcore.selector import choose_threshold

    sel = choose_threshold(
        prop_view["score"], prop_view["pred"], prop_view["y_open"], 1.0, alpha
    )
    err = open_set_error(view["pred"], view["y_open"])
    accept = sel.accept(view["score"])
    cert_client = np.asarray(z["cert_client"])
    grp = gmap[cert_client]
    cert_ids = ids["cert"]

    strata = []
    for g in range(G):
        # Group-local streams prevent a construction change in group g from
        # shifting every subsequent group's random numbers.
        rng = np.random.default_rng(derive_audit_seed(str(audit_seed), "group", str(g)))
        members_g = np.flatnonzero(grp == g)
        clients_g = [c for c in range(n_clients) if gmap[c] == g]
        if mode == GROUPED_QUOTA:
            drawn = np.concatenate(
                [
                    np.flatnonzero(cert_client == c)[
                        rng.integers(
                            0, int((cert_client == c).sum()), size=n_per_client
                        )
                    ]
                    for c in clients_g
                    if int((cert_client == c).sum()) > 0
                ]
            )
        else:  # iid_group: single-stage draw from the group's pooled points
            drawn = members_g[
                rng.integers(0, len(members_g), size=n_per_client * len(clients_g))
            ]
        acc = accept[drawn]
        strata.append(
            StratumDraw(
                stratum_type="group",
                stratum_id=g,
                reservoir_size_M=len(members_g),
                requested_draw_count_n=len(drawn),
                sampled_ids=cert_ids[drawn],
                accepted_draw_count_A=int(acc.sum()),
                accepted_error_count_K=int((acc & err[drawn]).sum()),
                accepted_ids=cert_ids[drawn][acc],
                accepted_error_ids=cert_ids[drawn][acc & err[drawn]],
            )
        )
    construction = (
        FIXED_QUOTA_WITH_REPLACEMENT
        if mode == GROUPED_QUOTA
        else POOLED_SINGLE_STAGE_WITH_REPLACEMENT
    )
    return DrawResult(
        draw_mode=mode,
        draw_construction=construction,
        replacement=True,
        audit_seed=audit_seed,
        strata=strata,
        certificate_variant=f"thm1_simplex_grouped_G{G}_{mode}",
    )
