"""PRODUCTION entry point for the CIFAR/FedPD grouped-stratified certificate.

This is the module that makes the CIFAR production call graph reach the declared sampler:

    certify_cell_grouped                                  (run_all.py, production caller)
      -> certify_cifar_cell_grouped                       (here)
      -> fedcore.cifar_group_mixture.spec_for_cell        (stamped pi/n_g, fail-closed)
      -> fedcore.certify.certify_best_gamma_grouped(mixture_spec=...)
      -> fedcore.group_draw.draw_group_certification_sample
      -> fedcore.sampling.sample_group_mixture            <-- the declared sampler

It is the CIFAR counterpart of :mod:`fedcore.medical.certify_grouped` and closes GAP-3, which
the prior agent escalated rather than resolved: the ratified declaration (``9a308e5f...``)
section 5 states CIFAR's pi RULE and requires "Freeze and hash **before** grouped
certification", and no frozen, stamped CIFAR pi basis existed. One exists now
(``scripts/freeze_cifar_grouped_design.py``), frozen and stamped BEFORE this path can run.

AUTHORITY
---------
``RATIFICATION_003_condition1_scoring.md`` (``7d6d7096...``) section 5a authorizes, after the
CIFAR/FedPD campaign drains, "a deliberate NO-TRAINING exact-grouped recertification over ALL
120 frozen CIFAR/FedPD logit sets", with the freezes above as mandatory preconditions. The
``grouped`` target itself is declared by ``ADDENDUM_003`` (``416f5dbe...``) for BOTH datasets.

ALL 120, NOT A SUBSET -- the safeguard, stated where the code lives
-------------------------------------------------------------------
Section 5a: "Run every one of the 120 frozen logit sets, so the recertification set is not
outcome-correlated... certifying only the cells that happened to finish after the wiring
would be a subset correlated with completion order, which is the selective-analysis pattern
the brief's prohibitions forbid." The recertification entry point
(``run_all.py --no-train --targets grouped``) enumerates the sealed matrix, not the
filesystem, and refuses to proceed unless all 120 artifacts are present and valid.

PROVENANCE -- carried on every row, never blurred
--------------------------------------------------
``pi``/``n_g`` were **NOT** in the original sealed pre-registration (``945aa2c9...``), which
declares the group PARTITION only. Condition 1's verdict is ``PASS_PREOUTCOME_ADDENDUM``:
**prospectively specified through an owner-authored pre-outcome addendum**.
NEVER "originally preregistered grouped analysis" -- that phrase is forbidden outright.
Weaker provenance than the sealed partition's; never present the two as equivalent. Both
ratifications state plainly that no cryptographic owner->subagent chain exists.

FAIL-CLOSED
-----------
If any positive-``pi`` client has an empty certification reservoir, this raises
:class:`~fedcore.cifar_group_mixture.MissingClientSupportError`. It does **not** drop the
client and does **not** renormalize ``pi`` over the survivors -- either would silently change
the certified target. The cell is reported as a negative feasibility finding.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from fedcore.certify import certify_best_gamma_grouped
from fedcore import cifar_group_mixture as CGM

#: References to the owner's ratifications, carried on every emitted row.
RATIFICATION = (
    "RATIFICATION_001 (af7f4e74...) -- ratifies the pi denominator. RATIFICATION_002 "
    "(02d5f821...) -- authorizes activation of the production grouped path. ADDENDUM_003 "
    "(416f5dbe...) -- declares the 'grouped' target for BOTH Fed-ISIC and CIFAR and declares "
    "NO new design content. RATIFICATION_003 (7d6d7096...) -- rules condition 1 PASS under "
    "the EPISTEMIC reading (status PASS_PREOUTCOME_ADDENDUM) and authorizes this NO-TRAIN "
    "exact-grouped recertification over ALL 120 frozen CIFAR/FedPD logit sets."
)

PROVENANCE_NOTE = (
    "pi/n_g were NOT in the original sealed prereg (945aa2c9..., partition only). CIFAR's pi "
    "RULE was declared 2026-07-16 (9a308e5f... section 5) with zero grouped outcomes in "
    "existence for any dataset, and ratified (af7f4e74...). Condition 1 status: "
    "PASS_PREOUTCOME_ADDENDUM -- prospectively specified through an owner-authored "
    "pre-outcome addendum, NOT part of the original sealed pre-registration. WEAKER "
    "provenance than the sealed partition's -- never present the two as equivalent. No "
    "cryptographic owner->subagent chain exists; all stamped records say so themselves."
)

#: G=2 is manuscript-ELIGIBLE, not established. Stage D is a separate claim.
ELIGIBILITY_NOTE = (
    "ELIGIBLE IS NOT ESTABLISHED (RATIFICATION_003 section 4): eligibility means only that a "
    "grouped certificate, once computed, MAY be reported. It asserts nothing about whether "
    "any will deploy or be non-vacuous. Stage D is reported separately and never merged with "
    "stages A/B/C."
)


def certify_cifar_cell_grouped(
    dataset: str,
    pipeline: str,
    split_id: str,
    train_seed: int,
    audit_seed: int,
    prop_view: Mapping,
    cert_view: Mapping,
    test_view: Mapping,
    *,
    score_name: str,
    gammas: Sequence[float],
    alpha: float,
    delta: float,
    Lambda: str = "box",
    rho: float | None = None,
    margin: float = 0.0,
    cell_label: str = "",
) -> Dict[str, object]:
    """Certify one CIFAR/FedPD cell under the DECLARED group mixture, or FAIL CLOSED.

    ``audit_seed`` must be the pre-registered ``audit``-namespace seed for this cell
    (``run_all.py::seed_ledger(...)['audit']['seed']``). Its sealed scope
    ``(dataset, pipeline, split_id, train_seed, d)`` masks ``alpha`` and
    ``certificate_variant``, and ``delta``/``rho``/policy/solver are not cell_id coordinates
    at all -- so every competing post-hoc analysis shares ONE draw and no certification
    outcome can select it. That is the within-run pairing the brief relies on.

    Raises ``MissingClientSupportError`` when a positive-pi client cannot supply an
    observation. Callers must record that as
    ``grouped_exact_sampler_status='infeasible_missing_client_support'`` and
    ``g2_manuscript_eligible=False`` -- never by renormalizing pi.
    """
    # Fail-closed: raises MissingClientSupportError if a positive-pi client is unsupported.
    spec = CGM.spec_for_cell(
        dataset, pipeline, split_id, train_seed, audit_seed,
        cert_view=cert_view, cell_label=cell_label,
    )

    box = CGM.rho_headline() if rho is None else float(rho)
    row = certify_best_gamma_grouped(
        prop_view,
        cert_view,
        test_view,
        score_name=score_name,
        group_map=spec.client_to_group,
        G=spec.G,
        gammas=gammas,
        alpha=alpha,
        delta=delta,
        Lambda=Lambda,
        box=box,
        seed=int(audit_seed),
        margin=margin,
        mixture_spec=spec,          # <-- activates the declared sampler
    )
    row.update(
        {
            "dataset": str(dataset),
            "pipeline": str(pipeline),
            "split_id": str(split_id),
            "train_seed": int(train_seed),
            "grouped_exact_sampler_status": "satisfiable",
            "g2_manuscript_eligible": True,
            "eligibility_note": ELIGIBILITY_NOTE,
            "group_names": list(spec.group_names),
            "rho": box,
            "lambda_G": f"uniform_box(G={spec.G}, rho={box})",
            "lambda_G_provenance": (
                "DERIVATION from sealed grids.rho_headline + the sealed G=2 partition; not a "
                "new choice. NAMING COLLISION: the sealed lfp_solver.lambda_G_semantics' 'G' "
                "is GAMMA (a rho-box over J CLIENT parts) -- a different object."
            ),
            "audit_seed": int(audit_seed),
            "ratification": RATIFICATION,
            "provenance_note": PROVENANCE_NOTE,
            "pi_semantics": CGM.PI_SEMANTICS,
            "target": CGM.GROUPED_TARGET,
            # ADDENDUM_004 section 3: owner rules this be recorded on EVERY CIFAR grouped
            # output. pi carries NO d-dependence by construction; do not read the non-IID
            # axis into any CIFAR grouped number.
            "cifar_declared_limitation": CGM.CIFAR_DECLARED_LIMITATION,
        }
    )
    row.update(spec.provenance)
    # draw_record holds numpy arrays; the JSON writer would stringify it into an
    # unparseable blob. The replayable facts (n_g, per-group seeds) are kept explicitly.
    record = row.pop("draw_record", None)
    if record is not None:
        row["draw_replay"] = {
            "n_g": {str(g): int(n) for g, n in record.n_g.items()},
            "seed_per_group": {str(g): int(s) for g, s in record.seed_per_group.items()},
            "draw_construction": record.draw_construction,
            "sampler": record.sampler,
            "n_observations": int(len(record.client_id)),
            "realized_client_draws": {
                str(j): int((record.client_id == j).sum())
                for j in sorted(set(int(c) for c in record.client_id))
            },
            "note": (
                "realized_client_draws is a DIAGNOSTIC of the draw, never an input to pi. "
                "pi is a pure function of the stamped basis."
            ),
        }
    return row
