"""PRODUCTION entry point for the Fed-ISIC grouped-stratified certificate.

This is the module that makes the production call graph reach the declared sampler:

    certify_task_grouped
      -> fedcore.medical.group_mixture.spec_for_task      (declared pi/n_g, fail-closed)
      -> fedcore.certify.certify_best_gamma_grouped(mixture_spec=...)
      -> fedcore.group_draw.draw_group_certification_sample
      -> fedcore.sampling.sample_group_mixture            <-- the declared sampler

Before this existed, ``sample_group_mixture`` had ZERO production callers: the grouped
certificate relabelled client->group and certified fixed largest-remainder quotas as if
they were a group sample. Fixed quotas are not draws from the declared mixture, so
``K_g | A_g ~ Bin(A_g, r_g)`` did not hold exactly. It does now.

ACTIVATION
----------
Activation of this path is authorized by
``preregistration/RATIFICATION_002_sampler_activation.md`` (``02d5f821...``), which
authorizes activation and recomputation ONLY. It explicitly does **not** authorize changing
``pi``, grouping, ``n_g``, ``Lambda_G``, folds, tasks, seeds, or failure budgets: activation
switches the production path onto the already-declared design and declares nothing new.

**Reachability, stated honestly:** this function is the grouped-certificate entry point and
it drives the declared sampler. As of activation it has **no production caller** — the
sealed campaign grid (``grids.targets = [simplex, box]``) contains no grouped variant, so
``run_all.py`` never reaches this path. Wiring a grouped arm into the campaign would add a
certificate variant the sealed prereg does not declare, which RATIFICATION_002 section 3.3
forbids. That gap is ESCALATED, not closed here.

DESIGN PROVENANCE — carried on every row, never blurred
-------------------------------------------------------
``pi_{j|g}`` and ``n_g`` were **NOT** in the original sealed pre-registration
(``945aa2c9...``), which declares the group PARTITION only. They were declared by the owner
on 2026-07-16 (``fedisic_grouped_sampling_addendum.md``, ``9a308e5f...``) with zero
full-tier Fed-ISIC outcomes in existence, and RATIFIED in
``RATIFICATION_001_grouped_sampling_declaration.md`` (``af7f4e74...``). **That is weaker
provenance than the sealed partition's, and no downstream text may present the two as
equivalent.** Both ratification records state plainly that no cryptographic owner->subagent
chain exists; neither this module nor any agent can verify one.

FAIL-CLOSED
-----------
If any positive-``pi`` center has an empty eligible certification reservoir, this raises
:class:`~fedcore.medical.group_mixture.MissingCenterSupportError`. It does **not** drop the
center and does **not** renormalize ``pi`` over the survivors — either would silently
change the certified target. The task is reported as a negative feasibility finding.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from fedcore.certify import certify_best_gamma_grouped
from fedcore.medical import group_mixture as GM

#: References to the owner's ratifications, carried on every emitted row.
RATIFICATION = (
    "RATIFICATION_001_grouped_sampling_declaration.md (af7f4e74...) — ratifies the pi "
    "denominator (straddlers OUT: BCN=0.464646, not 0.914871). "
    "RATIFICATION_002_sampler_activation.md (02d5f821...) — authorizes activation of this "
    "production grouped path and the recomputation of every G=2 table under the exact "
    "sampler; authorizes NOTHING else."
)

PROVENANCE_NOTE = (
    "pi/n_g were NOT in the original sealed prereg (partition only); declared 2026-07-16 "
    "with zero full-tier outcomes in existence and ratified by the owner "
    "(RATIFICATION_001, af7f4e74...; activation RATIFICATION_002, 02d5f821...). WEAKER "
    "provenance than the sealed partition -- never present the two as equivalent. Both "
    "ratifications state plainly that NO cryptographic owner->subagent chain exists. "
    "ERRATUM_001 (dbe37622...) disclosure travels with this design: one SMOKE-TIER Fed-ISIC "
    "model exists (master seed 21160715, disjoint from the full-tier 20260715), with no "
    "certification outcome; the owner ruled it does not close the declaration window."
)


def certify_task_grouped(
    split_id: int,
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
) -> Dict[str, object]:
    """Certify one Fed-ISIC task under the DECLARED group mixture, or FAIL CLOSED.

    ``audit_seed`` must be the pre-registered ``audit``-namespace seed for this cell
    (``run_all.py::seed_ledger(...)['audit']['seed']``). Its sealed scope masks ``alpha``
    and ``certificate_variant``, so every competing post-hoc analysis shares ONE draw and
    no certification outcome can select it.

    Raises ``MissingCenterSupportError`` when a positive-pi center cannot supply an
    observation. Callers must record that as
    ``grouped_exact_sampler_status='infeasible_missing_center_support'`` and
    ``g2_manuscript_eligible=False`` — never by renormalizing pi.
    """
    # Fail-closed: raises MissingCenterSupportError if a positive-pi center is unsupported.
    spec = GM.spec_for_task(split_id, train_seed, audit_seed)

    box = GM.rho_headline() if rho is None else float(rho)
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
            "split_id": int(split_id),
            "train_seed": int(train_seed),
            "grouped_exact_sampler_status": "satisfiable",
            "g2_manuscript_eligible": True,
            "group_names": list(spec.group_names),
            "rho": box,
            "lambda_G": f"uniform_box(G={spec.G}, rho={box})",
            "lambda_G_provenance": (
                "DERIVATION from sealed grids.rho_headline + the sealed G=2 partition; not "
                "a new choice. NAMING COLLISION: the sealed lfp_solver.lambda_G_semantics' "
                "'G' is GAMMA (a rho-box over J CLIENT parts) -- a different object."
            ),
            "ratification": RATIFICATION,
            "provenance_note": PROVENANCE_NOTE,
            "pi_semantics": GM.PI_SEMANTICS,
            "target": GM.GROUPED_TARGET,
        }
    )
    row.update(spec.provenance)
    return row


def sampler_status_for_task(split_id: int, train_seed: int = 0) -> dict:
    """STAGE A only: sampler satisfiability. Never merged with count feasibility."""
    status = GM.sampler_status(split_id, train_seed)
    status["ratification"] = RATIFICATION
    status["provenance_note"] = PROVENANCE_NOTE
    return status
