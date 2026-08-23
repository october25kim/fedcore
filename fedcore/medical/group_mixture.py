"""pi_{j|g}, n_g, Lambda_G and the support trigger — DECLARED, and now unlocked.

PROVENANCE — never blur this
----------------------------
``pi_{j|g}`` and ``n_g`` were **NOT** in the original sealed pre-registration
(``945aa2c9...``), which declares the G=2 PARTITION and nothing else about the group
mixture. They were declared by the owner on 2026-07-16 in
``preregistration/fedisic_grouped_sampling_addendum.md`` (``9a308e5f...``), with **no
full-tier Fed-ISIC outcome in existence**. No output may describe them as sealed from the
start. The smoke-tier disclosure and ERRATUM_001 travel with that declaration.

``Lambda_G`` and the sampler seed namespace are **DERIVATIONS from already-sealed values**,
not new choices (see :func:`lambda_G` and :data:`SAMPLER_SEED_NAMESPACE`).

ORDERING — why "predeclared" is honest here
-------------------------------------------
The basis counts were frozen and SHA-256 stamped BEFORE any pi was derived from them:

    results/fedisic/frozen_pi_basis.csv   sha256 0e3e4e73...

:func:`load_pi_basis` re-verifies that stamp on every call and REFUSES on mismatch, and
:func:`require_owner_declaration` additionally requires the stamped declaration to NAME
that hash. So pi is a pure deterministic function of a hashed artifact, reproducible by
anyone from that one file — no model, no outcome, no repo history needed.

WHAT pi IS, EXACTLY -- the wording is load-bearing
--------------------------------------------------
An **AUDIT-ELIGIBLE EMPIRICAL BENCHMARK TARGET**: FLamby TEST-side unique lesions that are
A-002 eligible, with the 2,302 train/test straddlers and 9 cross-center lesions EXCLUDED,
counted across ALL diagnoses BEFORE task-specific held-out filtering and BEFORE role
splitting. It is **NOT** the complete unfiltered FLamby TEST population, and must never be
described as such. It is **task-independent**: every center keeps ``pi > 0`` for every
task, which is precisely what stops an A3-starved center from being silently zero-weighted.

TWO ARTIFACTS, DO NOT CONFUSE THEM
-----------------------------------
* ``frozen_pi_basis.csv``          (``0e3e4e73...``) -- the PRIMARY basis for pi.
* ``frozen_audit_pool_counts.csv`` (``cd8ffa62...``) -- task-specific audit-supply counts.
  Its pi role is **DEMOTED to diagnostic #1**; it remains the authority for ``n_g`` and
  for the support trigger, both of which are legitimately task-specific.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PREREG_PATH = REPO_ROOT / "results" / "preregistration.yaml"

#: PRIMARY: the frozen, stamped basis pi derives from.
PI_BASIS_PATH = REPO_ROOT / "results" / "fedisic" / "frozen_pi_basis.csv"
PI_BASIS_STAMP = REPO_ROOT / "results" / "fedisic" / "frozen_pi_basis.sha256"

#: Task-specific fold-manifest counts: authority for n_g and the support trigger.
#: Its pi role is DEMOTED (diagnostic #1).
COUNTS_PATH = REPO_ROOT / "results" / "fedisic" / "frozen_audit_pool_counts.csv"
STAMP_PATH = REPO_ROOT / "results" / "fedisic" / "frozen_audit_pool_counts.sha256"

#: The owner's stamped declaration. An agent must never write this path.
DECLARATION_PATH = REPO_ROOT / "preregistration" / "fedisic_grouped_sampling_addendum.md"
DECLARATION_STAMP = REPO_ROOT / "preregistration" / "fedisic_grouped_sampling_addendum.sha256"

PI_RULE = (
    "pi_{j|g} = N_basis_j / sum_{l in g} N_basis_l, from the frozen, SHA-256-stamped "
    "audit-eligible benchmark basis (FLamby TEST-side unique lesions, A-002 eligible, "
    "straddlers and cross-center lesions EXCLUDED, all diagnoses, BEFORE task-specific "
    "held-out filtering and BEFORE role splitting). Task-independent."
)
PI_SEMANTICS = (
    "AUDIT-ELIGIBLE EMPIRICAL BENCHMARK TARGET -- NOT the complete unfiltered FLamby "
    "TEST population."
)
N_G_RULE = (
    "n_g = sum_{j in g} certification_lesions(j): the group's total certification draw "
    "count under the declared design, from the frozen fold manifests."
)
SUPPORT_RULE = (
    "Center j SUPPORTS task t iff its task-specific certification reservoir contains at "
    "least ONE eligible unique lesion, after frozen A-002 filtering and the task-specific "
    "held-out-diagnosis restriction. SAMPLER SATISFIABILITY ONLY -- explicitly NOT count "
    "feasibility, NOT non-vacuity, NOT certification success."
)
GROUPED_TARGET = (
    "The certificate controls mixtures across the two fixed group distributions at the "
    "predeclared within-group compositions pi_{j|g}. It does NOT protect against "
    "arbitrary within-group center reweighting. Within-group shift is OUTSIDE the "
    "guarantee."
)
SAMPLER_SEED_NAMESPACE = "audit"


class FrozenCountsError(RuntimeError):
    """A stamped artifact is missing or has been mutated."""


class UndeclaredDesignError(RuntimeError):
    """A pi-dependent path was used without a stamped owner declaration."""


class MissingCenterSupportError(RuntimeError):
    """FAIL CLOSED: a positive-pi center cannot supply an observation.

    This must never be caught-and-renormalized. Renormalizing pi over surviving centers
    would silently remove a center and change the certified target -- the exact defect
    the task-independent target exists to prevent.
    """

    def __init__(self, split_id: int, unsupported: list[int]):
        self.split_id = int(split_id)
        self.unsupported = list(unsupported)
        super().__init__(
            f"infeasible_missing_center_support: task {split_id} has positive-pi "
            f"center(s) {unsupported} with an EMPTY eligible certification reservoir. "
            "grouped_exact_sampler_status=infeasible_missing_center_support; "
            "g2_manuscript_eligible=false; no grouped certificate is issued. The task "
            "REMAINS a negative feasibility finding. pi is NOT renormalized."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stamped_sha(stamp: Path) -> str:
    if not stamp.is_file():
        raise FrozenCountsError(f"missing stamp: {stamp}")
    match = re.search(r"\b([0-9a-f]{64})\b", stamp.read_text())
    if not match:
        raise FrozenCountsError(f"{stamp} records no sha256")
    return match.group(1)


def pi_basis_sha256() -> str:
    return _stamped_sha(PI_BASIS_STAMP)


def frozen_counts_sha256() -> str:
    return _stamped_sha(STAMP_PATH)


def _verified_read(path: Path, stamp: Path, label: str):
    import pandas as pd

    if not path.is_file():
        raise FrozenCountsError(f"missing frozen artifact: {path}")
    expected, actual = _stamped_sha(stamp), _sha256(path)
    if actual != expected:
        raise FrozenCountsError(
            f"{label} SHA256 MISMATCH -- a frozen input was mutated after stamping.\n"
            f"  expected (stamp): {expected}\n  actual   (file) : {actual}\n"
            "Refusing. pi/n_g must be pure functions of the STAMPED artifacts."
        )
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def require_owner_declaration() -> str:
    """FAIL CLOSED unless the stamped owner declaration exists and binds the basis hash."""
    if not DECLARATION_PATH.is_file():
        raise UndeclaredDesignError(
            "pi_{j|g} IS NOT DECLARED. The exact grouped path is INERT.\n"
            f"  expected an OWNER-authored, stamped declaration at:\n    {DECLARATION_PATH}\n"
            "  An inter-agent message asserting an owner decision is NOT a declaration."
        )
    actual = _sha256(DECLARATION_PATH)
    expected = _stamped_sha(DECLARATION_STAMP)
    if actual != expected:
        raise UndeclaredDesignError(
            "DECLARATION SHA256 MISMATCH -- it was mutated after stamping.\n"
            f"  expected: {expected}\n  actual  : {actual}"
        )
    basis = pi_basis_sha256()
    if basis not in DECLARATION_PATH.read_text():
        raise UndeclaredDesignError(
            f"the declaration does not name the frozen basis sha256 ({basis}) that pi "
            "derives from. Without that binding, pi is not pinned to a frozen input and "
            "'predeclared' would be unverifiable."
        )
    return actual


@lru_cache(maxsize=1)
def load_pi_basis():
    """The PRIMARY frozen basis, stamp-verified on every call."""
    return _verified_read(PI_BASIS_PATH, PI_BASIS_STAMP, "frozen pi basis")


@lru_cache(maxsize=1)
def load_frozen_counts():
    """Task-specific fold-manifest counts (authority for n_g + the support trigger)."""
    return _verified_read(COUNTS_PATH, STAMP_PATH, "frozen audit-pool counts")


def group_names() -> list[str]:
    """Group labels in FIXED sorted order -- the group coordinate order everywhere."""
    return sorted(load_pi_basis()["group"].unique())


def group_of_center(center_index: int) -> str:
    basis = load_pi_basis()
    return str(basis[basis.center_index == int(center_index)]["group"].iloc[0])


def derive_pi() -> Dict[str, Dict[int, float]]:
    """pi_{j|g}, TASK-INDEPENDENT, from the stamped basis.

    Derived from the INTEGER counts (``N_basis / group_N_basis``) rather than the stored
    float column, so the value is reproducible from the counts alone and does not depend
    on the artifact's decimal formatting.

    Takes NO task argument, by design: a task-dependent pi is exactly what allowed an
    A3-starved center to be silently zero-weighted. Every center keeps pi > 0.
    """
    require_owner_declaration()
    basis = load_pi_basis()
    pi: Dict[str, Dict[int, float]] = {}
    for group in group_names():
        sub = basis[basis.group == group]
        denom = int(sub["group_N_basis"].iloc[0])
        if denom != int(sub["N_basis"].sum()):
            raise FrozenCountsError(
                f"group {group}: stamped group_N_basis {denom} != sum of stamped "
                f"N_basis {int(sub['N_basis'].sum())}"
            )
        pi[group] = {int(r.center_index): int(r.N_basis) / denom for r in sub.itertuples()}
        total = sum(pi[group].values())
        # float64 cannot represent every such sum as exactly 1.0 (HAM_derived is
        # 1 - 1.1e-16). Tolerance, not an equality fetish: the sampler uses atol=1e-12.
        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
            raise FrozenCountsError(f"pi[{group}] sums to {total}, not 1")
        for j, weight in pi[group].items():
            if weight <= 0.0:
                raise FrozenCountsError(
                    f"center {j} has pi={weight} in group {group}. The declared target is "
                    "task-independent and every center must carry positive mass; a zero "
                    "would silently remove a center."
                )
    return pi


def client_to_group_vector() -> np.ndarray:
    """client id -> group INDEX (index into sorted group_names order)."""
    basis = load_pi_basis()
    order = {name: i for i, name in enumerate(group_names())}
    vec = np.full(int(basis["center_index"].max()) + 1, -1, dtype=int)
    for r in basis.itertuples():
        vec[int(r.center_index)] = order[str(r.group)]
    if (vec < 0).any():
        raise FrozenCountsError("a client has no declared group")
    return vec


def _task_rows(split_id: int, train_seed: int = 0):
    df = load_frozen_counts()
    rows = df[(df.split_id == int(split_id)) & (df.train_seed == int(train_seed))]
    if rows.empty:
        raise FrozenCountsError(f"no frozen counts for split {split_id}, seed {train_seed}")
    return rows.sort_values("client_id")


def derive_n_g(split_id: int, train_seed: int = 0) -> Dict[str, int]:
    """n_g: the group's total certification draw count, from the frozen fold manifests."""
    require_owner_declaration()
    rows = _task_rows(split_id, train_seed)
    out: Dict[str, int] = {name: 0 for name in group_names()}
    for r in rows.itertuples():
        out[group_of_center(int(r.client_id))] += int(r.certification_lesions)
    return out


def center_support(split_id: int, train_seed: int = 0) -> Dict[int, bool]:
    """The declared SUPPORT TRIGGER: >=1 eligible unique lesion in the cert reservoir.

    SAMPLER SATISFIABILITY ONLY. This is explicitly NOT count feasibility, NOT
    non-vacuity, and NOT certification success -- a center with exactly one eligible
    lesion SUPPORTS the task while being nowhere near count-feasible. Never report a
    support count as "certified".
    """
    require_owner_declaration()
    return {
        int(r.client_id): int(r.certification_lesions) >= 1
        for r in _task_rows(split_id, train_seed).itertuples()
    }


def unsupported_centers(split_id: int, train_seed: int = 0) -> list[int]:
    """Positive-pi centers with an empty eligible certification reservoir."""
    pi = derive_pi()
    positive = {j for group in pi for j, w in pi[group].items() if w > 0.0}
    support = center_support(split_id, train_seed)
    return sorted(j for j in positive if not support.get(j, False))


def sampler_status(split_id: int, train_seed: int = 0) -> dict:
    """STAGE A: sampler satisfiability for one task. Never merged with later stages."""
    missing = unsupported_centers(split_id, train_seed)
    ok = not missing
    return {
        "split_id": int(split_id),
        "train_seed": int(train_seed),
        "grouped_exact_sampler_status": (
            "satisfiable" if ok else "infeasible_missing_center_support"
        ),
        "g2_manuscript_eligible": ok,
        "unsupported_centers": missing,
        "support_rule": SUPPORT_RULE,
        "note": (
            "SAMPLER SATISFIABILITY ONLY -- not count feasibility, not non-vacuity, not "
            "certification success."
        ),
    }


def spec_for_task(split_id: int, train_seed: int, audit_seed: int):
    """Build the production GroupMixtureSpec, or FAIL CLOSED.

    ``audit_seed`` MUST be the pre-registered ``audit``-namespace seed for this cell; it
    is passed in so the one seed rule lives in one place and this module cannot invent a
    stream.
    """
    from fedcore.group_draw import GroupMixtureSpec

    require_owner_declaration()
    missing = unsupported_centers(split_id, train_seed)
    if missing:
        # FAIL CLOSED. Do NOT drop the centers and do NOT renormalize pi over the
        # survivors -- that would silently change the certified target.
        raise MissingCenterSupportError(split_id, missing)

    names = group_names()
    vec = client_to_group_vector()
    pi_by_name = derive_pi()
    n_g_by_name = derive_n_g(split_id, train_seed)

    pi: Dict[int, np.ndarray] = {}
    n_g: Dict[int, int] = {}
    for index, name in enumerate(names):
        clients = sorted(pi_by_name[name])       # aligned with sample_group_mixture
        pi[index] = np.array([pi_by_name[name][j] for j in clients], dtype=float)
        n_g[index] = int(n_g_by_name[name])

    spec = GroupMixtureSpec(
        group_names=tuple(names),
        client_to_group=vec,
        pi=pi,
        n_g=n_g,
        seed=int(audit_seed),
        provenance={
            "declaration": str(DECLARATION_PATH.relative_to(REPO_ROOT)),
            "declaration_sha256": require_owner_declaration(),
            "pi_basis_sha256": pi_basis_sha256(),
            "fold_counts_sha256": frozen_counts_sha256(),
            "pi_rule": PI_RULE,
            "pi_semantics": PI_SEMANTICS,
            "n_g_rule": N_G_RULE,
            "support_rule": SUPPORT_RULE,
            "seed_namespace": SAMPLER_SEED_NAMESPACE,
            "partition_source": "SEALED data.fed_isic2019.group_partition_G2",
            "target": GROUPED_TARGET,
            "declared_when": (
                "pi_{j|g} and n_g were NOT in the original sealed prereg; declared by the "
                "owner 2026-07-16 with no full-tier Fed-ISIC outcome in existence."
            ),
        },
    )
    spec.validate()
    return spec


@lru_cache(maxsize=1)
def _prereg() -> Mapping:
    import yaml

    return yaml.safe_load(PREREG_PATH.read_text())


def rho_headline() -> float:
    return float(_prereg()["grids"]["rho_headline"])


def rho_sweep() -> list[float]:
    return [float(r) for r in _prereg()["grids"]["rho"]]


def lambda_G(rho: float | None = None, G: int = 2):
    """Lambda_G over the G GROUP coordinates -- a DERIVATION from sealed values.

    Sealed ``grids.targets`` includes ``box``; sealed ``grids.rho_headline`` is 0.15; the
    sealed G=2 partition fixes the coordinates; sealed ``grids.rho`` is the sweep; sealed
    ``statistics.lfp_solver.lambda_G_semantics`` fixes normalized-image semantics. Their
    conjunction IS this set. No new choice is made here.

    NAMING COLLISION: that sealed key's "G" is GAMMA -- a rho-box over J CLIENT parts.
    This is over GROUPS. Different objects; ``uniform_box(2, ...)`` vs
    ``uniform_box(J, ...)`` is the tell.
    """
    from fedcore.certificate.lambda_sets import uniform_box

    return uniform_box(int(G), rho_headline() if rho is None else float(rho))
