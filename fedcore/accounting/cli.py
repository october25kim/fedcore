"""One command that regenerates every reservoir-accounting artifact.

    python -m fedcore.accounting.cli          # all outputs, all resolvable runs
    python -m fedcore.accounting.cli --limit 5 --no-figures     # fast smoke

CPU only, no torch, no network. Reads the frozen ``runs/*_logits.npz``, the ws4090
manifests, and the vendored CIFAR label files; writes only under
``results/accounting/``.

Fail-loud policy: a run whose sample identity cannot be RECOVERED AND VERIFIED is
reported as unresolved and excluded -- never accounted with guessed IDs. A replayed
A/K that disagrees with the certification pipeline raises immediately.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import List

import numpy as np

from fedcore.accounting import draws, rows as rowmod
from fedcore.accounting.artifacts import artifact_extension, write_draw_ids
from fedcore.accounting.ids import IdRecoveryError, recover_fold_ids
from fedcore.accounting.provenance import discover_runs
from fedcore.io_utils import atomic_write_csv

OUT_DIR = os.path.join("results", "accounting")
DRAW_ID_DIR = os.path.join(OUT_DIR, "audit_draw_ids")

# The published headline cells: aggregate.main reports CertCov at G=J and G=2; the
# manuscript quotes alpha=0.10 (feasibility edge) and alpha=0.20 (headline).
HEADLINE_ALPHAS = (0.10, 0.20)
HEADLINE_GROUPINGS = (None, 2)
# n/M sweep for the occupancy figure. ratio=1.0 is exp_resampling_validity's own
# operating point; the rest exist only to trace the theoretical curve.
BOOTSTRAP_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)
GROUPED_N_PER_CLIENT = 400
GROUPED_G = 2


def _sorted_rows(rows: List[dict], keys) -> List[dict]:
    return sorted(rows, key=lambda r: tuple(str(r[k]) for k in keys))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=".")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="account only the first N resolvable runs (smoke)",
    )
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--no-draw-ids",
        action="store_true",
        help="skip the per-draw ID artifacts (they dominate runtime)",
    )
    args = ap.parse_args()

    os.chdir(args.root)
    discovered = discover_runs(".")
    print(f"discovered {len(discovered)} reservoirs under runs/")

    long_rows: List[dict] = []
    overlap_rows: List[dict] = []
    group_rows: List[dict] = []
    unresolved: List[tuple] = []
    accounted = 0

    for npz_path, spec in discovered:
        name = os.path.basename(npz_path)
        if spec is None:
            unresolved.append(
                (name, "out-of-scope: no CIFAR index space (tabular/unparsable)")
            )
            continue
        if args.limit is not None and accounted >= args.limit:
            break
        try:
            with np.load(npz_path, allow_pickle=True) as z:
                ids = recover_fold_ids(spec, npz=z)

                headline_by_key = {}
                for alpha in HEADLINE_ALPHAS:
                    for G in HEADLINE_GROUPINGS:
                        res = draws.headline_draw(spec, z, ids, alpha=alpha, G=G)
                        headline_by_key[(alpha, G)] = res
                        long_rows += rowmod.long_rows(spec, res, ids, alpha)

                # Overlap report uses the per-client headline repartition (the
                # repartition is identical across cells: seed=0, fixed fracs).
                overlap_rows += rowmod.overlap_rows(
                    spec, ids, headline_by_key[(HEADLINE_ALPHAS[0], None)]
                )

                # With-replacement audit draws: the only family where duplication is
                # real, and the source of the occupancy figure's observed points.
                # The seed excludes ratio: all budget fractions use prefixes of one
                # frozen per-client stream (common random numbers, nested audits).
                bootstrap_seed = draws.derive_audit_seed(
                    "bootstrap", spec.run_id, "replicate0"
                )
                for ratio in BOOTSTRAP_RATIOS:
                    res = draws.bootstrap_draw(
                        spec, z, ids, alpha=0.20, ratio=ratio, audit_seed=bootstrap_seed
                    )
                    res.certificate_variant += f"_ratio{ratio}"
                    long_rows += rowmod.long_rows(spec, res, ids, 0.20)
                    if not args.no_draw_ids and ratio == 1.0:
                        write_draw_ids(
                            os.path.join(
                                DRAW_ID_DIR, f"{spec.run_id}__audit_bootstrap"
                            ),
                            [
                                {
                                    "stratum_type": s.stratum_type,
                                    "stratum_id": s.stratum_id,
                                    "sample_ids": s.sampled_ids,
                                }
                                for s in res.strata
                            ],
                        )

                # Grouped G=2 composition: quota vs iid-group.
                grouped_seed = draws.derive_audit_seed(
                    "grouped", spec.run_id, "replicate0"
                )
                for mode in (draws.GROUPED_QUOTA, draws.GROUPED_IID_GROUP):
                    res = draws.grouped_draw(
                        spec,
                        z,
                        ids,
                        alpha=0.20,
                        G=GROUPED_G,
                        mode=mode,
                        n_per_client=GROUPED_N_PER_CLIENT,
                        audit_seed=grouped_seed,
                    )
                    long_rows += rowmod.long_rows(spec, res, ids, 0.20)
                    group_rows += rowmod.group_composition_rows(
                        spec, z, res, ids, 0.20, GROUPED_G
                    )

                if not args.no_draw_ids:
                    hres = headline_by_key[(HEADLINE_ALPHAS[0], None)]
                    write_draw_ids(
                        os.path.join(DRAW_ID_DIR, f"{spec.run_id}__headline_cert"),
                        [
                            {
                                "stratum_type": s.stratum_type,
                                "stratum_id": s.stratum_id,
                                "sample_ids": s.sampled_ids,
                            }
                            for s in hres.strata
                        ],
                    )
            accounted += 1
            print(f"  [{accounted}] {spec.run_id} ({spec.provenance_source})")
        except IdRecoveryError as e:
            unresolved.append((name, f"id-recovery failed: {e}"))
        except Exception as e:  # replay/count inconsistency: loud, never swallowed
            traceback.print_exc()
            print(f"FATAL while accounting {name}: {e}", file=sys.stderr)
            return 1

    if not long_rows:
        print("no runs accounted -- nothing to write", file=sys.stderr)
        return 1

    # ---- fail loudly on any fold-overlap violation -------------------------- #
    violations = [r for r in overlap_rows if r["status"] == "VIOLATION"]
    if violations:
        for v in violations[:10]:
            print(
                f"FOLD OVERLAP VIOLATION: {v['run_id']} {v['fold_a']}/{v['fold_b']} "
                f"= {v['overlap_count']}",
                file=sys.stderr,
            )
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    long_sorted = _sorted_rows(
        long_rows,
        [
            "dataset",
            "model",
            "run_id",
            "target_alpha",
            "certificate_variant",
            "draw_mode",
            "stratum_type",
            "stratum_id",
        ],
    )
    atomic_write_csv(
        os.path.join(OUT_DIR, "reservoir_accounting_long.csv"),
        rowmod.LONG_FIELDS,
        long_sorted,
    )
    atomic_write_csv(
        os.path.join(OUT_DIR, "reservoir_accounting_summary.csv"),
        rowmod.SUMMARY_FIELDS,
        rowmod.summarize(long_sorted),
    )
    atomic_write_csv(
        os.path.join(OUT_DIR, "fold_overlap_report.csv"),
        rowmod.OVERLAP_FIELDS,
        _sorted_rows(overlap_rows, ["run_id", "protocol", "fold_a", "fold_b"]),
    )
    atomic_write_csv(
        os.path.join(OUT_DIR, "group_sampling_composition.csv"),
        rowmod.GROUP_FIELDS,
        _sorted_rows(group_rows, ["run_id", "draw_mode", "group_id", "client_id"]),
    )

    from fedcore.accounting.tables import write_tex_table

    write_tex_table(
        rowmod.summarize(long_sorted),
        os.path.join(OUT_DIR, "reservoir_accounting_table.tex"),
    )

    if not args.no_figures:
        from fedcore.accounting.figures import make_all_figures

        make_all_figures(long_sorted, os.path.join(OUT_DIR, "figs"))

    print(f"\naccounted runs      : {accounted}")
    print(f"unresolved runs     : {len(unresolved)}")
    for name, why in unresolved[:12]:
        print(f"  - {name}: {why}")
    if len(unresolved) > 12:
        print(f"  ... and {len(unresolved) - 12} more")
    print(f"long rows           : {len(long_sorted)}")
    print(f"draw-id artifacts   : {DRAW_ID_DIR}/*{artifact_extension()}")
    print(f"wrote               : {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
