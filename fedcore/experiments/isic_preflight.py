"""Metadata-only Fed-ISIC2019 audit-supply preflight (WP-C question A3).

Answers, WITHOUT training anything and WITHOUT opening an image: *given a 2-of-8
open-set split, can every center -- including the small ones (819 and 439 images)
-- supply enough LABELED UNKNOWNS to certify?*

Torch-free by construction (numpy + pandas only), so it runs on the host.

Run::

    python -m fedcore.experiments.isic_preflight                       # default split
    python -m fedcore.experiments.isic_preflight --unknown-classes DF,VASC
    python -m fedcore.experiments.isic_preflight --all-pairs           # all 28 pairs

Counting unit: LESIONS under the declared design (``--counting-unit``)
---------------------------------------------------------------------
``declared-lesion`` (DEFAULT) implements ``data.fed_isic2019.audit_unit``
(AMENDMENT A-002) and is the only mode that describes the folds this campaign
actually emits.  The audit supply of center ``j`` is its count of AUDIT-ELIGIBLE
LESIONS: on the FLamby TEST side of the declared train/audit boundary, excluding
the 2,302 straddling and 9 cross-center lesions, counted ONE PER LESION because
the prereg draws exactly one image per lesion.

``legacy-image`` reproduces the SUPERSEDED projection that produced the sealed
``a3_preflight_finding``: every image of a center counted as audit supply, with a
``max_known_audit_frac`` knob for the training/audit trade-off.  It is retained
ONLY so the two can be compared; it overstates the supply by ~16x (23,247 images
vs 1,463 audit-eligible lesions) and no longer corresponds to any emitted fold.

Projection model (stated explicitly; every number below is a *projection*, not a
measurement of a fold that exists)
-----------------------------------------------------------------------------
Let center ``j`` hold ``K_j`` units of the 6 known classes and ``U_j`` units of
the 2 held-out unknown classes, and let ``q`` be the audit unknown fraction.

* Under ``declared-lesion``, ``K_j``/``U_j`` count audit-eligible LESIONS, so both
  supplies are already restricted to the audit side of the boundary and no known
  unit is being taken away from training.
* The audit pool is sized so its unknown fraction is exactly ``q``::

      P_j = min( floor(U_j / q),  floor(K_j_avail / (1 - q)) )

  where ``K_j_avail = floor(max_known_audit_frac * K_j)``.  Under the declared
  design ``max_known_audit_frac`` is fixed at 1.0 and is no longer a knob: the
  boundary already decided what may be audited, so nothing is being diverted.
* The pool splits into proposal/certification/test by ``fold_fractions``
  (default 0.40/0.30/0.30).  Knowns and unknowns are each allocated across folds
  by largest remainder, so counts are exact, integral, and RNG-free -- which is
  why the per-center fold SIZES below match the emitted CSVs exactly, even though
  fold MEMBERSHIP is a seeded draw.

Feasibility reference
---------------------
Theorem 2 requires the OBSERVED accepted count ``A_j >= ln(J/delta)/(-ln(1-alpha))``.
The certification-fold size ``n_j`` upper-bounds ``A_j`` (acceptance can only
discard).  So ``n_j < thm2_floor`` means center ``j`` is infeasible *even if the
selector accepts every point* -- a hard, model-independent verdict.  This is a
necessary condition only: it is reported as a floor, never as a guarantee.

Licensing: ISIC-2019 / HAM10000 are CC-BY-NC.  See ``fedcore.data.fed_isic2019``.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from fedcore.certificate.feasibility import thm2_floor
from fedcore.data.fed_isic2019 import (
    CENTER_DESCRIPTIONS,
    CLASSES,
    N_CENTERS,
    attach_official_fold,
    class_counts_by_center,
    cross_center_lesions,
    load_center_table,
    straddling_lesions,
)

DEFAULT_FOLD_FRACTIONS: Tuple[float, float, float] = (0.40, 0.30, 0.30)
DEFAULT_UNKNOWN_FRAC: float = 0.30
FOLD_NAMES: Tuple[str, str, str] = ("proposal", "certification", "test")


def audit_eligible_lesion_counts(root: str | None = None) -> pd.DataFrame:
    """Per-center x class counts of AUDIT-ELIGIBLE LESIONS under A-002.

    The audit supply the pre-registered design actually has: FLamby TEST-side
    lesions, excluding the straddling and cross-center ones, counted once per lesion
    because exactly one image per lesion is drawn.  This is the ``declared-lesion``
    counting unit, and it is what every emitted fold CSV is built from.
    """
    table = attach_official_fold(load_center_table(root), root)
    excluded = set(straddling_lesions(table)) | set(
        cross_center_lesions(table)["lesion_id"].unique()
    )
    pool = table[(table["fold"] == "test") & (~table["lesion_id"].isin(excluded))]
    lesions = pool.drop_duplicates(subset=["lesion_id"])
    counts = pd.crosstab(lesions["center"], lesions["diagnosis"])
    return counts.reindex(index=range(N_CENTERS), columns=list(CLASSES), fill_value=0)


def _largest_remainder(total: int, fractions: Sequence[float]) -> List[int]:
    """Split ``total`` by ``fractions`` into integers summing exactly to ``total``."""
    if total <= 0:
        return [0] * len(fractions)
    exact = np.asarray(fractions, dtype=float) * float(total)
    base = np.floor(exact).astype(int)
    remainder = int(total - base.sum())
    if remainder:
        order = np.argsort(-(exact - base), kind="stable")
        base[order[:remainder]] += 1
    return [int(v) for v in base]


def project_center(
    n_known_images: int,
    n_unknown_images: int,
    *,
    unknown_frac: float = DEFAULT_UNKNOWN_FRAC,
    fold_fractions: Sequence[float] = DEFAULT_FOLD_FRACTIONS,
    max_known_audit_frac: float = 1.0,
) -> Dict:
    """Project one center's audit pool and per-fold composition.

    Pure integer arithmetic on counts; no RNG, no images, no labels beyond the
    class histogram.  See the module docstring for the model.
    """
    if not 0.0 < unknown_frac < 1.0:
        raise ValueError("unknown_frac must lie strictly in (0, 1)")
    known_avail = int(np.floor(max_known_audit_frac * n_known_images))

    pool_by_unknown = int(np.floor(n_unknown_images / unknown_frac))
    pool_by_known = int(np.floor(known_avail / (1.0 - unknown_frac)))
    pool = int(min(pool_by_unknown, pool_by_known))
    binding = "unknown-supply" if pool_by_unknown <= pool_by_known else "known-supply"

    n_unk_pool = int(round(unknown_frac * pool))
    n_unk_pool = min(n_unk_pool, n_unknown_images)
    n_known_pool = pool - n_unk_pool

    unk_by_fold = _largest_remainder(n_unk_pool, fold_fractions)
    known_by_fold = _largest_remainder(n_known_pool, fold_fractions)
    size_by_fold = [u + k for u, k in zip(unk_by_fold, known_by_fold)]

    cert = FOLD_NAMES.index("certification")
    return {
        "known_images": int(n_known_images),
        "unknown_images": int(n_unknown_images),
        "audit_pool": pool,
        "binding_constraint": binding,
        "pool_known": n_known_pool,
        "pool_unknown": n_unk_pool,
        "n_cert": int(size_by_fold[cert]),
        "cert_unknown": int(unk_by_fold[cert]),
        "cert_known": int(known_by_fold[cert]),
        "fold_sizes": {name: int(s) for name, s in zip(FOLD_NAMES, size_by_fold)},
        "fold_unknowns": {name: int(u) for name, u in zip(FOLD_NAMES, unk_by_fold)},
    }


def preflight_split(
    counts: pd.DataFrame,
    unknown_classes: Sequence[str],
    *,
    unknown_frac: float = DEFAULT_UNKNOWN_FRAC,
    fold_fractions: Sequence[float] = DEFAULT_FOLD_FRACTIONS,
    max_known_audit_frac: float = 1.0,
    alpha: float = 0.10,
    delta: float = 0.10,
) -> Dict:
    """Project every center for one 2-of-8 open-set split."""
    unknown_classes = list(unknown_classes)
    bad = set(unknown_classes) - set(CLASSES)
    if bad:
        raise ValueError(
            f"unknown classes {sorted(bad)} are not ISIC classes {CLASSES}"
        )
    known_classes = [c for c in CLASSES if c not in set(unknown_classes)]

    floor = thm2_floor(N_CENTERS, delta, alpha)
    rows = []
    for j in range(N_CENTERS):
        n_unknown = int(counts.loc[j, unknown_classes].sum())
        n_known = int(counts.loc[j, known_classes].sum())
        proj = project_center(
            n_known,
            n_unknown,
            unknown_frac=unknown_frac,
            fold_fractions=fold_fractions,
            max_known_audit_frac=max_known_audit_frac,
        )
        proj["center"] = j
        proj["center_name"] = CENTER_DESCRIPTIONS[j]
        proj["total_images"] = int(counts.loc[j].sum())
        proj["starved"] = proj["cert_unknown"] == 0
        proj["thm2_infeasible"] = proj["n_cert"] < floor
        # Second failure mode, independent of unknown supply: a center may retain
        # too few KNOWN classes to pose a non-trivial local closed-set task.  With
        # only one known class present the local risk is degenerate (every known
        # point shares a label), so per-client error counts carry no signal.
        present = [c for c in known_classes if int(counts.loc[j, c]) > 0]
        proj["known_classes_present"] = present
        proj["n_known_classes_present"] = len(present)
        proj["degenerate_known"] = len(present) < 2
        rows.append(proj)

    return {
        "unknown_classes": unknown_classes,
        "known_classes": known_classes,
        "thm2_floor": float(floor),
        "rows": rows,
        "starved_centers": [r["center"] for r in rows if r["starved"]],
        "thm2_infeasible_centers": [r["center"] for r in rows if r["thm2_infeasible"]],
        "degenerate_known_centers": [
            r["center"] for r in rows if r["degenerate_known"]
        ],
        "min_cert_unknown": min(r["cert_unknown"] for r in rows),
        "min_known_classes_present": min(r["n_known_classes_present"] for r in rows),
    }


def _print_split_table(counts: pd.DataFrame, report: Dict) -> None:
    unk = report["unknown_classes"]
    print(f"\nOpen-set split: unknown = {unk}, known = {report['known_classes']}")
    print(
        f"Theorem 2 floor (J=6): accepted count A_j >= {report['thm2_floor']:.2f} "
        f"-> n_j must be >= {int(np.ceil(report['thm2_floor']))}"
    )

    print("\nPer-center class counts (unknown classes marked *):")
    header = f"{'ctr':>3} {'total':>6} " + " ".join(
        f"{(c + '*') if c in unk else c:>6}" for c in CLASSES
    )
    print(header)
    print("-" * len(header))
    for j in range(N_CENTERS):
        cells = " ".join(f"{int(counts.loc[j, c]):>6}" for c in CLASSES)
        print(f"{j:>3} {int(counts.loc[j].sum()):>6} {cells}")

    print("\nProjected audit supply per center:")
    header2 = (
        f"{'ctr':>3} {'site':<34} {'imgs':>6} {'known':>6} {'unk':>5} "
        f"{'pool':>6} {'bind':>13} {'n_cert':>7} {'cert_unk':>9} {'kcls':>4} {'verdict':>18}"
    )
    print(header2)
    print("-" * len(header2))
    for r in report["rows"]:
        if r["starved"]:
            verdict = "STARVED (0 unk)"
        elif r["thm2_infeasible"]:
            verdict = "THM2-INFEASIBLE"
        elif r["degenerate_known"]:
            verdict = "DEGENERATE-KNOWN"
        else:
            verdict = "ok"
        site = r["center_name"][:34]
        print(
            f"{r['center']:>3} {site:<34} {r['total_images']:>6} {r['known_images']:>6} "
            f"{r['unknown_images']:>5} {r['audit_pool']:>6} {r['binding_constraint']:>13} "
            f"{r['n_cert']:>7} {r['cert_unknown']:>9} {r['n_known_classes_present']:>4} "
            f"{verdict:>18}"
        )

    starved = report["starved_centers"]
    infeas = report["thm2_infeasible_centers"]
    degen = report["degenerate_known_centers"]
    print(
        f"\n  starved centers (0 labeled unknowns in cert fold): "
        f"{starved if starved else 'none'}"
    )
    print(
        f"  Thm2-infeasible centers (n_cert < floor, even at 100% acceptance): "
        f"{infeas if infeas else 'none'}"
    )
    print(
        f"  degenerate-known centers (<2 known classes present locally): "
        f"{degen if degen else 'none'}"
    )
    for r in report["rows"]:
        if r["degenerate_known"]:
            print(
                f"      center {r['center']} retains only known class(es) "
                f"{r['known_classes_present']} -- local closed-set task is trivial"
            )


def _print_all_pairs(counts: pd.DataFrame, **kwargs) -> None:
    reports = []
    for pair in combinations(CLASSES, 2):
        reports.append(preflight_split(counts, list(pair), **kwargs))

    print(f"\nAll {len(reports)} two-of-eight unknown-class pairs")
    header = (
        f"{'unknown pair':<12} {'min cert_unk':>12} {'starved':>18} "
        f"{'thm2-infeasible':>22} {'status':>10}"
    )
    print(header)
    print("-" * len(header))
    for rep in sorted(reports, key=lambda r: r["min_cert_unknown"]):
        pair = "+".join(rep["unknown_classes"])
        starved = rep["starved_centers"]
        infeas = rep["thm2_infeasible_centers"]
        degen = rep["degenerate_known_centers"]
        if starved:
            status = "STARVE"
        elif infeas:
            status = "THM2"
        elif degen:
            status = "DEGEN"
        else:
            status = "ok"
        print(
            f"{pair:<12} {rep['min_cert_unknown']:>12} "
            f"{str(starved) if starved else '-':>18} "
            f"{str(infeas) if infeas else '-':>22} {status:>10}"
        )

    starving = [r for r in reports if r["starved_centers"]]
    infeasible = [
        r for r in reports if r["thm2_infeasible_centers"] and not r["starved_centers"]
    ]
    print(f"\nSUMMARY over {len(reports)} pairs")
    print(
        f"  pairs starving >=1 center (0 labeled unknowns in cert fold): {len(starving)}"
    )
    for r in starving:
        print(
            f"      {'+'.join(r['unknown_classes']):<12} -> centers {r['starved_centers']}"
        )
    print(f"  pairs Thm2-infeasible at >=1 center (but not starved): {len(infeasible)}")
    for r in infeasible:
        print(
            f"      {'+'.join(r['unknown_classes']):<12} -> centers {r['thm2_infeasible_centers']}"
        )
    usable = [
        r
        for r in reports
        if not r["starved_centers"] and not r["thm2_infeasible_centers"]
    ]
    print(f"  pairs usable at all 6 centers (unknown supply + Thm2): {len(usable)}")
    for r in sorted(usable, key=lambda r: -r["min_cert_unknown"]):
        flag = (
            f"  [DEGENERATE-KNOWN at {r['degenerate_known_centers']}]"
            if r["degenerate_known_centers"]
            else ""
        )
        print(
            f"      {'+'.join(r['unknown_classes']):<12} min cert_unk={r['min_cert_unknown']:<4}"
            f"min known classes present={r['min_known_classes_present']}{flag}"
        )
    clean = [r for r in usable if not r["degenerate_known_centers"]]
    print(f"  pairs usable AND non-degenerate at all 6 centers: {len(clean)}")
    for r in sorted(clean, key=lambda r: -r["min_cert_unknown"]):
        print(
            f"      {'+'.join(r['unknown_classes']):<12} min cert_unk={r['min_cert_unknown']}"
        )


#: The sealed prereg's ``data.fed_isic2019.split_roster.drawn`` -- final, never
#: re-drawn. Mirrored here so the preflight can report roster viability without
#: importing the campaign runner. ``tests/test_isic_audit_design.py`` asserts this
#: equals the prereg, so it cannot drift.
PREREG_ROSTER: Tuple[Tuple[int, Tuple[str, str]], ...] = (
    (0, ("MEL", "BCC")),
    (1, ("MEL", "BKL")),
    (2, ("MEL", "DF")),
    (3, ("MEL", "VASC")),
    (4, ("NV", "VASC")),
    (5, ("BCC", "AK")),
    (6, ("BCC", "BKL")),
    (7, ("BCC", "SCC")),
    (8, ("AK", "VASC")),
    (9, ("BKL", "VASC")),
)


def _print_roster(counts: pd.DataFrame, **kwargs) -> None:
    """Viability of the 10 PRE-REGISTERED roster splits, in roster order."""
    floor = None
    print(f"\nPre-registered roster ({len(PREREG_ROSTER)} of 28 pairs), per split:")
    header = (
        f"{'split':>5} {'unknown':<10} {'n_cert per center (j=0..5)':<34} "
        f"{'cert_unk per center':<28} {'verdict':<18}"
    )
    print(header)
    print("-" * len(header))
    viable = []
    for split_id, pair in PREREG_ROSTER:
        rep = preflight_split(counts, list(pair), **kwargs)
        floor = rep["thm2_floor"]
        n_cert = [r["n_cert"] for r in rep["rows"]]
        cert_unk = [r["cert_unknown"] for r in rep["rows"]]
        starved = rep["starved_centers"]
        infeas = rep["thm2_infeasible_centers"]
        degen = rep["degenerate_known_centers"]
        if starved:
            verdict = f"STARVED {starved}"
        elif infeas:
            verdict = f"THM2 {infeas}"
        elif degen:
            verdict = f"DEGEN {degen}"
        else:
            verdict = "VIABLE"
            viable.append((split_id, pair))
        print(
            f"{split_id:>5} {'+'.join(pair):<10} {str(n_cert):<34} "
            f"{str(cert_unk):<28} {verdict:<18}"
        )
    print(f"\n  Theorem-2 floor: n_cert must be >= {int(np.ceil(floor))} at EVERY center")
    print(
        f"  VIABLE AT ALL 6 CENTERS: {len(viable)} of {len(PREREG_ROSTER)} "
        f"-> {[f'{s}:{"+".join(p)}' for s, p in viable] if viable else 'NONE'}"
    )
    print(
        "  The other splits are pre-registered A3 feasibility FINDINGS for the J=6 "
        "client-simplex target, reported with accounting rows -- never dropped, "
        "never re-drawn (prereg: handling / prohibitions)."
    )
    _print_grouped(counts, **kwargs)


#: ``data.fed_isic2019.group_partition_G2``, predeclared from FLamby metadata before
#: training. ``predeclared_mitigations`` names the grouped G=2 target as a remedy for
#: exactly the A3 scarcity the client-simplex target hits.
PREREG_GROUP_PARTITION_G2: Tuple[Tuple[int, ...], ...] = ((1, 2, 3, 5), (0, 4))


def _print_grouped(counts: pd.DataFrame, **kwargs) -> None:
    """Roster viability under the PREDECLARED grouped G=2 target.

    Grouping is the prereg's own predeclared mitigation, not a post-hoc rescue: a
    group is one stratum, so its supply is the SUM over its centers and the union
    bound is spent over G=2 strata rather than J=6.  Both effects raise viability.
    """
    alpha = float(kwargs.get("alpha", 0.10))
    delta = float(kwargs.get("delta", 0.10))
    grouped = pd.DataFrame(
        [counts.loc[list(group)].sum() for group in PREREG_GROUP_PARTITION_G2],
        index=range(len(PREREG_GROUP_PARTITION_G2)),
    )
    sub = dict(kwargs)
    floor = thm2_floor(len(PREREG_GROUP_PARTITION_G2), delta, alpha)
    print(
        f"\nPREDECLARED grouped G=2 target "
        f"{[list(g) for g in PREREG_GROUP_PARTITION_G2]} "
        f"(group_partition_G2: HAM_derived / other):"
    )
    header = f"{'split':>5} {'unknown':<10} {'n_cert per group':<20} {'cert_unk':<14} {'verdict':<12}"
    print(header)
    print("-" * len(header))
    viable = []
    for split_id, pair in PREREG_ROSTER:
        rows = []
        for g in range(len(PREREG_GROUP_PARTITION_G2)):
            known = [c for c in CLASSES if c not in set(pair)]
            proj = project_center(
                int(grouped.loc[g, known].sum()),
                int(grouped.loc[g, list(pair)].sum()),
                unknown_frac=sub.get("unknown_frac", DEFAULT_UNKNOWN_FRAC),
                fold_fractions=sub.get("fold_fractions", DEFAULT_FOLD_FRACTIONS),
                max_known_audit_frac=1.0,
            )
            rows.append(proj)
        n_cert = [r["n_cert"] for r in rows]
        cert_unk = [r["cert_unknown"] for r in rows]
        starved = [g for g, r in enumerate(rows) if r["cert_unknown"] == 0]
        infeas = [g for g, r in enumerate(rows) if r["n_cert"] < floor]
        if starved:
            verdict = f"STARVED {starved}"
        elif infeas:
            verdict = f"THM2 {infeas}"
        else:
            verdict = "VIABLE"
            viable.append((split_id, pair))
        print(f"{split_id:>5} {'+'.join(pair):<10} {str(n_cert):<20} {str(cert_unk):<14} {verdict:<12}")
    print(f"\n  Theorem-2 floor at G=2: n_cert >= {int(np.ceil(floor))} per GROUP")
    print(
        f"  VIABLE AT BOTH GROUPS: {len(viable)} of {len(PREREG_ROSTER)} "
        f"-> {[f'{s}:{"+".join(p)}' for s, p in viable] if viable else 'NONE'}"
    )
    print(
        "  Per the prereg, the simplex-vs-grouped comparison on Fed-ISIC IS a "
        "headline result, not a failure of the campaign."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Metadata-only Fed-ISIC2019 audit-supply preflight (WP-C A3)."
    )
    parser.add_argument("--root", default=None, help="data/isic2019 directory")
    parser.add_argument(
        "--unknown-classes",
        default="DF,VASC",
        help="comma-separated 2 of 8 held-out classes (default: DF,VASC)",
    )
    parser.add_argument("--all-pairs", action="store_true", help="sweep all 28 pairs")
    parser.add_argument(
        "--counting-unit",
        choices=("declared-lesion", "legacy-image"),
        default="declared-lesion",
        help=(
            "declared-lesion (default): the A-002 audit pool, one unit per "
            "audit-eligible lesion. legacy-image: the SUPERSEDED image-count "
            "projection behind the sealed a3_preflight_finding."
        ),
    )
    parser.add_argument(
        "--roster",
        action="store_true",
        help="report viability of the 10 pre-registered roster splits",
    )
    parser.add_argument("--unknown-frac", type=float, default=DEFAULT_UNKNOWN_FRAC)
    parser.add_argument(
        "--fold-fractions",
        default="0.40,0.30,0.30",
        help="proposal,certification,test (default: 0.40,0.30,0.30)",
    )
    parser.add_argument(
        "--max-known-audit-frac",
        type=float,
        default=1.0,
        help="cap on known images divertible from training into the audit pool",
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.10)
    args = parser.parse_args()

    fold_fractions = tuple(float(x) for x in args.fold_fractions.split(","))
    if len(fold_fractions) != 3 or not np.isclose(sum(fold_fractions), 1.0):
        raise SystemExit("--fold-fractions must be 3 numbers summing to 1")

    table = load_center_table(args.root)
    declared = args.counting_unit == "declared-lesion"
    counts = (
        audit_eligible_lesion_counts(args.root)
        if declared
        else class_counts_by_center(table)
    )

    print("=" * 100)
    print("Fed-ISIC2019 preflight -- metadata only, no images, no model, no torch")
    print("=" * 100)
    print(f"images={len(table)}  centers={N_CENTERS}  classes={len(CLASSES)}")
    print(f"counting unit: {args.counting_unit}")
    if declared:
        print(
            "  A-002 audit pool: FLamby TEST-side lesions, minus straddling and "
            "cross-center, ONE image per lesion"
        )
        print(f"  audit-eligible lesions: {int(counts.values.sum())} of 11847")
    else:
        print(
            "  SUPERSEDED image-count projection (the sealed a3_preflight_finding's "
            "basis); does not correspond to any emitted fold"
        )
    print(
        f"fold fractions {dict(zip(FOLD_NAMES, fold_fractions))}, "
        f"audit unknown fraction {args.unknown_frac}, "
        f"max known audit fraction "
        f"{1.0 if declared else args.max_known_audit_frac}"
        f"{' (fixed by the declared boundary; not a knob)' if declared else ''}"
    )

    kwargs = dict(
        unknown_frac=args.unknown_frac,
        fold_fractions=fold_fractions,
        max_known_audit_frac=1.0 if declared else args.max_known_audit_frac,
        alpha=args.alpha,
        delta=args.delta,
    )
    if args.roster:
        _print_roster(counts, **kwargs)
    elif args.all_pairs:
        _print_all_pairs(counts, **kwargs)
    else:
        unknown_classes = [
            c.strip() for c in args.unknown_classes.split(",") if c.strip()
        ]
        if len(unknown_classes) != 2:
            raise SystemExit("--unknown-classes must name exactly 2 classes")
        report = preflight_split(counts, unknown_classes, **kwargs)
        _print_split_table(counts, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
