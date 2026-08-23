"""CIFAR pi_{j|g}, n_g, Lambda_G and the support gate — DERIVED FROM A STAMPED BASIS.

This is the CIFAR counterpart of :mod:`fedcore.medical.group_mixture`. It exists because
``RATIFICATION_003_condition1_scoring.md`` section 5a authorizes a NO-TRAIN exact-grouped
recertification over all 120 frozen CIFAR/FedPD logit sets, and makes a frozen, hashed
``pi`` basis a PRECONDITION of it.

PROVENANCE — never blur this
----------------------------
``pi`` and ``n_g`` were **NOT** in the original sealed pre-registration (``945aa2c9...``),
which declares the CIFAR group PARTITION only (``data.cifar.group_partition_G2 =
[[0,1,2],[3,4]]``). CIFAR's pi **rule** was declared in the owner-authored grouped-sampling
addendum (``9a308e5f...``) section 5 and ratified (RATIFICATION_001, ``af7f4e74...``), with
zero grouped certification outcomes in existence for any dataset.

Condition 1's verdict is recorded as, and only as, ``PASS_PREOUTCOME_ADDENDUM``. Every use
must call this **"prospectively specified through an owner-authored pre-outcome addendum"**
and must never call it an "originally preregistered grouped analysis". This is **weaker
provenance than the sealed partition's and must never be presented as equivalent.**

ORDERING — why "predeclared" is honest here
-------------------------------------------
The basis counts were frozen and SHA-256 stamped BEFORE any pi was derived from them:

    results/cifar/frozen_cifar_pi_basis.csv

:func:`load_pi_basis` re-verifies that stamp on every call and REFUSES on mismatch, so pi
is a pure deterministic function of one hashed file, reproducible from that file alone.

A BINDING ASYMMETRY, ESCALATED RATHER THAN ENGINEERED AROUND
-------------------------------------------------------------
Fed-ISIC's binding is MUTUAL: ``fedisic_grouped_sampling_addendum.md`` names the basis hash
``0e3e4e73...``, and :func:`fedcore.medical.group_mixture.require_owner_declaration`
enforces that naming. **CIFAR cannot have that**: the declaration was stamped on 2026-07-16
and cannot name the hash of a table that did not yet exist, and no agent may edit a stamped
file to add it. So the binding here runs ONE WAY ONLY -- this basis names the declaration;
the declaration does not name this basis. What is enforced is that the declaration is
present, unmutated at ``9a308e5f...``, and literally contains CIFAR's pi rule. Closing the
loop requires an OWNER-authored record naming this basis hash. **Flagged, not resolved.**

WHAT THE BASIS IS, EXACTLY -- the wording is load-bearing
---------------------------------------------------------
``n_nominal_cert_j`` is the size of client j's certification fold as the DECLARED design
allocates it, computed by calling the pipeline's own split path with the pre-registered
seeds. **NOMINAL, not realized.** It is not derived from accepted counts, errors, realized
sampled counts, model scores, certification results, or alpha/delta/rho/policy/solver.

A FINDING TRAVELS WITH IT: CIFAR's Dirichlet partition applies to TRAIN only, and
``build_calibration`` array_splits the known TEST pool EQUALLY across clients, so the
declared rule yields UNIFORM within-group pi. That coincides NUMERICALLY with the
"uniform-within-group pi" the declaration's section 6 lists as a SUPERSEDED DIAGNOSTIC, and
differs from it in PROVENANCE. Never conflate the two.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = REPO_ROOT / "results" / "preregistration.yaml"

#: PRIMARY: the frozen, stamped basis pi derives from. Integer counts ONLY -- deliberately
#: no pi column, so "frozen" and "derived" are not the same act.
PI_BASIS_PATH = REPO_ROOT / "results" / "cifar" / "frozen_cifar_pi_basis.csv"
PI_BASIS_STAMP = REPO_ROOT / "results" / "cifar" / "frozen_cifar_pi_basis.sha256"

#: The frozen design: partition, n_g, Lambda_G, sampler seeds, allocation/threshold policy.
DESIGN_PATH = REPO_ROOT / "results" / "cifar" / "frozen_cifar_grouped_design.json"
DESIGN_STAMP = REPO_ROOT / "results" / "cifar" / "frozen_cifar_grouped_design.sha256"

#: The owner's stamped declaration. An agent must NEVER write this path.
DECLARATION_PATH = REPO_ROOT / "preregistration" / "fedisic_grouped_sampling_addendum.md"
DECLARATION_STAMP = REPO_ROOT / "preregistration" / "fedisic_grouped_sampling_addendum.sha256"

#: The exact rule the declaration's section 5 states for CIFAR. Checked literally: if the
#: declaration does not contain it, this module has no authority to derive anything.
CIFAR_PI_RULE_TEXT = "pi_{j|g} = n_nominal_cert_j / sum_{l in g} n_nominal_cert_l"

PI_RULE = (
    "pi_{j|g} = n_nominal_cert_j / sum_{l in g} n_nominal_cert_l, from the frozen NOMINAL "
    "certification allocations (the sealed Dirichlet partition and fold manifests). "
    "Grouped-sampling declaration (9a308e5f...) section 5. NOMINAL, not realized."
)
PI_SEMANTICS = (
    "NOMINAL CERTIFICATION-FOLD ALLOCATION TARGET -- the mixture the declared design "
    "allocates, NOT the realized composition of any drawn or observed sample."
)
N_G_RULE = (
    "n_g = sum_{j in g} n_nominal_cert_j: the group's total certification draw count under "
    "the declared design, from the stamped basis."
)
SUPPORT_RULE = (
    "Client j SUPPORTS a cell iff its certification reservoir contains at least ONE unit. "
    "SAMPLER SATISFIABILITY ONLY -- explicitly NOT count feasibility, NOT non-vacuity, NOT "
    "certification success."
)
GROUPED_TARGET = (
    "The certificate controls mixtures across the two fixed group distributions at the "
    "predeclared within-group compositions pi_{j|g}. It does NOT protect against arbitrary "
    "within-group client reweighting. Within-group shift is OUTSIDE the guarantee."
)

#: ADDENDUM_004 (5d4c1c0f...) DECLARED LIMITATION. Section 3: "The owner rules that this be
#: recorded on EVERY CIFAR grouped output." CIFAR's grouped pi carries NO d-dependence by
#: construction, so no CIFAR grouped number tracks the non-IID (d) axis. Carried verbatim on
#: every CIFAR grouped row (succeeded, infeasible, and blocked).
CIFAR_DECLARED_LIMITATION = {
    "addendum": "preregistration/ADDENDUM_004_cifar_grouped_binding.md",
    "addendum_sha256": "5d4c1c0f2b19da6c30529819026e3065ca72168596cef7f256f690f76ded060f",
    "binds_pi_basis_sha256": "3492d09b287f7b6e362a9956c7e6b9ac2dc8950a90207240765c9661784a2200",
    "binds_frozen_design_sha256": "7068b580792444fc8b96fc0ea61bd040bdc25a2270ba4edc31fea1086033c03b",
    "mechanism": (
        "CIFAR's Dirichlet partition applies to TRAIN only; build_calibration array_splits the "
        "known TEST pool UNIFORMLY across clients, so the declared supply-proportional rule "
        "mechanically yields pi=(1/3,1/3,1/3 | 1/2,1/2), n_g=1542/1028 for EVERY d."
    ),
    "consequences": [
        "within-group pi is UNIFORM BY CONSTRUCTION",
        "pi has NO d-dependence",
        "d changes TRAINING heterogeneity, NOT audit composition",
        "CIFAR grouped results do NOT establish robustness to within-group traffic heterogeneity",
        "numerical coincidence with the superseded uniform-within-group diagnostic does NOT "
        "make them the same object (declared supply-proportional rule on uniform folds vs an "
        "arbitrary alternative rule -- different provenance, never conflate)",
    ],
    "only_fedisic_exercises_nonuniform": (
        "Only Fed-ISIC's real acquisition centres (N_basis 138/790/132/211/159/33) exercise a "
        "non-uniform within-group target; no CIFAR grouped number tracks the non-IID axis."
    ),
}
SAMPLER_SEED_NAMESPACE = "audit"

#: Escalated, not resolved. See :mod:`run_all` GROUPED_KNOB_ESCALATION.
BINDING_ASYMMETRY = (
    "ONE-WAY BINDING, ESCALATED: this basis names the declaration (9a308e5f...), but the "
    "declaration cannot name this basis -- it was stamped before the basis existed and no "
    "agent may edit a stamped file. Fed-ISIC's binding is MUTUAL (the declaration names "
    "0e3e4e73...). Closing the loop requires an OWNER-authored record naming this basis "
    "hash. Until then CIFAR's pi binding is strictly weaker than Fed-ISIC's."
)


class FrozenBasisError(RuntimeError):
    """A stamped artifact is missing or has been mutated."""


class UndeclaredDesignError(RuntimeError):
    """A pi-dependent path was used without a stamped owner declaration."""


class MissingClientSupportError(RuntimeError):
    """FAIL CLOSED: a positive-pi client cannot supply an observation.

    Never caught-and-renormalized. Renormalizing pi over surviving clients would silently
    remove a client and change the certified target.
    """

    def __init__(self, cell_label: str, unsupported: Sequence[int]):
        self.cell_label = str(cell_label)
        self.unsupported = [int(j) for j in unsupported]
        super().__init__(
            f"infeasible_missing_client_support: cell {cell_label} has positive-pi "
            f"client(s) {self.unsupported} with an EMPTY certification reservoir. "
            "grouped_exact_sampler_status=infeasible_missing_client_support; "
            "g2_manuscript_eligible=false; no grouped certificate is issued. The cell "
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
        raise FrozenBasisError(f"missing stamp: {stamp}")
    match = re.search(r"\b([0-9a-f]{64})\b", stamp.read_text())
    if not match:
        raise FrozenBasisError(f"{stamp} records no sha256")
    return match.group(1)


def pi_basis_sha256() -> str:
    return _stamped_sha(PI_BASIS_STAMP)


def design_sha256() -> str:
    return _stamped_sha(DESIGN_STAMP)


def _verified_read(path: Path, stamp: Path, label: str):
    import pandas as pd

    if not path.is_file():
        raise FrozenBasisError(
            f"missing frozen artifact: {path}\n"
            "The CIFAR grouped path is INERT until the basis is frozen and stamped by "
            "scripts/freeze_cifar_grouped_design.py. Deriving pi on the fly would select "
            "design post hoc, which the prereg's prohibitions forbid."
        )
    expected, actual = _stamped_sha(stamp), _sha256(path)
    if actual != expected:
        raise FrozenBasisError(
            f"{label} SHA256 MISMATCH -- a frozen input was mutated after stamping.\n"
            f"  expected (stamp): {expected}\n  actual   (file) : {actual}\n"
            "Refusing. pi/n_g must be pure functions of the STAMPED artifacts."
        )
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def require_owner_declaration() -> str:
    """FAIL CLOSED unless the stamped declaration exists, verifies, and states CIFAR's rule.

    This is deliberately NOT symmetric with the Fed-ISIC check (see BINDING_ASYMMETRY): it
    cannot require the declaration to name this basis's hash, because the declaration
    predates the basis and stamped files are never edited.
    """
    if not DECLARATION_PATH.is_file():
        raise UndeclaredDesignError(
            "CIFAR pi_{j|g} has no stamped declaration. The exact grouped path is INERT.\n"
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
    text = DECLARATION_PATH.read_text()
    if CIFAR_PI_RULE_TEXT not in text:
        raise UndeclaredDesignError(
            "the stamped declaration does not state CIFAR's pi rule "
            f"({CIFAR_PI_RULE_TEXT!r}). Without it there is no declared CIFAR pi to derive."
        )
    return actual


@lru_cache(maxsize=1)
def load_pi_basis():
    """The PRIMARY frozen basis, stamp-verified on every call."""
    return _verified_read(PI_BASIS_PATH, PI_BASIS_STAMP, "frozen CIFAR pi basis")


@lru_cache(maxsize=1)
def load_design() -> Mapping:
    """The frozen design (partition, n_g, Lambda_G, seeds, policies), stamp-verified."""
    import json

    if not DESIGN_PATH.is_file():
        raise FrozenBasisError(f"missing frozen design: {DESIGN_PATH}")
    expected, actual = _stamped_sha(DESIGN_STAMP), _sha256(DESIGN_PATH)
    if actual != expected:
        raise FrozenBasisError(
            "frozen CIFAR design SHA256 MISMATCH -- mutated after stamping.\n"
            f"  expected (stamp): {expected}\n  actual   (file) : {actual}"
        )
    return json.loads(DESIGN_PATH.read_text())


@lru_cache(maxsize=1)
def _prereg() -> Mapping:
    import yaml

    return yaml.safe_load(PREREG_PATH.read_text())


def frozen_keys() -> list[Tuple[str, str, str, int]]:
    """Every (dataset, pipeline, split_id, train_seed) the stamped basis covers."""
    basis = load_pi_basis()
    keys = basis[["dataset", "pipeline", "split_id", "train_seed"]].drop_duplicates()
    return [
        (str(r.dataset), str(r.pipeline), str(r.split_id), int(r.train_seed))
        for r in keys.itertuples()
    ]


def _rows(dataset: str, pipeline: str, split_id: str, train_seed: int):
    basis = load_pi_basis()
    rows = basis[
        (basis.dataset == str(dataset))
        & (basis.pipeline == str(pipeline))
        & (basis.split_id == str(split_id))
        & (basis.train_seed == int(train_seed))
    ]
    if rows.empty:
        raise FrozenBasisError(
            f"the stamped basis has no row for ({dataset}, {pipeline}, {split_id}, "
            f"seed{train_seed}). Refusing to invent one."
        )
    return rows.sort_values("client_index")


def client_to_group_vector() -> np.ndarray:
    """client id -> group index, from the SEALED data.cifar.group_partition_G2.

    Read from the sealed prereg, NOT from the frozen basis: the partition is the one part of
    this design that IS sealed, and it must keep its stronger provenance rather than inherit
    the basis's weaker one.
    """
    partition = _prereg()["data"]["cifar"]["group_partition_G2"]
    size = max(int(c) for members in partition for c in members) + 1
    vec = np.full(size, -1, dtype=int)
    for index, members in enumerate(partition):
        for client in members:
            if vec[int(client)] != -1:
                raise FrozenBasisError(f"client {client} is in two sealed groups")
            vec[int(client)] = index
    if (vec < 0).any():
        raise FrozenBasisError("a client has no sealed group")
    return vec


def derive_pi(dataset: str, pipeline: str, split_id: str, train_seed: int) -> Dict[int, Dict[int, float]]:
    """pi_{j|g} from the stamped basis: group index -> {client -> weight}.

    Derived from the INTEGER counts, so the value is reproducible from the counts alone and
    does not depend on the artifact's decimal formatting. Every client keeps pi > 0 -- a zero
    would silently remove a client from the certified target.
    """
    require_owner_declaration()
    rows = _rows(dataset, pipeline, split_id, train_seed)
    vec = client_to_group_vector()
    pi: Dict[int, Dict[int, float]] = {}
    for group in sorted(set(int(g) for g in rows.group_index)):
        sub = rows[rows.group_index == group]
        denom = int(sub["group_n_nominal_cert"].iloc[0])
        if denom != int(sub["n_nominal_cert"].sum()):
            raise FrozenBasisError(
                f"group {group}: stamped group_n_nominal_cert {denom} != sum of stamped "
                f"n_nominal_cert {int(sub['n_nominal_cert'].sum())}"
            )
        pi[group] = {
            int(r.client_index): int(r.n_nominal_cert) / denom for r in sub.itertuples()
        }
        for client in pi[group]:
            if int(vec[client]) != group:
                raise FrozenBasisError(
                    f"basis puts client {client} in group {group}, but the SEALED partition "
                    f"puts it in group {int(vec[client])}. Refusing."
                )
        total = sum(pi[group].values())
        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
            raise FrozenBasisError(f"pi[{group}] sums to {total}, not 1")
        for client, weight in pi[group].items():
            if weight <= 0.0:
                raise FrozenBasisError(
                    f"client {client} has pi={weight} in group {group}. Every client must "
                    "carry positive mass; a zero would silently remove a client."
                )
    return pi


def derive_n_g(dataset: str, pipeline: str, split_id: str, train_seed: int) -> Dict[int, int]:
    """n_g from the stamped basis, cross-checked against the frozen design copy."""
    require_owner_declaration()
    rows = _rows(dataset, pipeline, split_id, train_seed)
    out: Dict[int, int] = {}
    for group in sorted(set(int(g) for g in rows.group_index)):
        sub = rows[rows.group_index == group]
        out[group] = int(sub["group_n_nominal_cert"].iloc[0])

    # The design JSON froze n_g independently. If the two ever disagree, one of them is
    # wrong and neither may be preferred silently.
    key = "|".join((str(dataset), str(pipeline), str(split_id), str(int(train_seed))))
    frozen = load_design()["n_g"]["by_cell"].get(key)
    if frozen is None:
        raise FrozenBasisError(f"the frozen design has no n_g entry for {key}")
    if {int(k): int(v) for k, v in frozen.items()} != out:
        raise FrozenBasisError(
            f"n_g MISMATCH for {key}: basis-derived {out} != frozen design {frozen}. "
            "Refusing -- a frozen quantity and its basis must not disagree."
        )
    return out


def rho_headline() -> float:
    return float(_prereg()["grids"]["rho_headline"])


def lambda_G(rho: float | None = None, G: int = 2):
    """Lambda_G over the G GROUP coordinates -- a DERIVATION from sealed values.

    NAMING COLLISION, kept visible: the sealed ``lfp_solver.lambda_G_semantics`` uses "G" for
    GAMMA -- a rho-box over J CLIENT parts. This is over GROUPS.
    """
    from fedcore.certificate.lambda_sets import uniform_box

    return uniform_box(int(G), rho_headline() if rho is None else float(rho))


def unsupported_clients(cert_view: Mapping[str, np.ndarray]) -> list[int]:
    """Positive-pi clients with an EMPTY realized certification reservoir.

    The gate is realized-reservoir-based BY NECESSITY and that is a real difference from
    Fed-ISIC's, stated rather than hidden: Fed-ISIC's support trigger reads a frozen
    task-specific fold manifest, while a CIFAR cell's reservoir is the artifact's own
    certification fold. This function reads ONLY the client id vector -- never a label, a
    score, an accepted count or an error -- so it cannot leak an outcome into the design.
    """
    vec = client_to_group_vector()
    present = set(int(j) for j in np.unique(np.asarray(cert_view["client"])))
    return sorted(j for j in range(len(vec)) if j not in present)


def spec_for_cell(
    dataset: str,
    pipeline: str,
    split_id: str,
    train_seed: int,
    audit_seed: int,
    cert_view: Mapping[str, np.ndarray] | None = None,
    cell_label: str = "",
):
    """Build the production GroupMixtureSpec for one CIFAR cell, or FAIL CLOSED.

    ``audit_seed`` MUST be the pre-registered ``audit``-namespace seed for the cell; it is
    passed in so the one seed rule lives in one place and this module cannot invent a stream.
    """
    from fedcore.group_draw import GroupMixtureSpec

    require_owner_declaration()
    if cert_view is not None:
        missing = unsupported_clients(cert_view)
        if missing:
            # FAIL CLOSED. Do NOT drop the clients and do NOT renormalize pi over the
            # survivors -- either would silently change the certified target.
            raise MissingClientSupportError(cell_label or split_id, missing)

    vec = client_to_group_vector()
    pi_by_group = derive_pi(dataset, pipeline, split_id, train_seed)
    n_g_by_group = derive_n_g(dataset, pipeline, split_id, train_seed)
    names = tuple(f"group{g}" for g in sorted(pi_by_group))

    pi: Dict[int, np.ndarray] = {}
    n_g: Dict[int, int] = {}
    for group in sorted(pi_by_group):
        clients = sorted(pi_by_group[group])       # aligned with sample_group_mixture
        pi[group] = np.array([pi_by_group[group][j] for j in clients], dtype=float)
        n_g[group] = int(n_g_by_group[group])

    spec = GroupMixtureSpec(
        group_names=names,
        client_to_group=vec,
        pi=pi,
        n_g=n_g,
        seed=int(audit_seed),
        provenance={
            "declaration": str(DECLARATION_PATH.relative_to(REPO_ROOT)),
            "declaration_sha256": require_owner_declaration(),
            "pi_basis": str(PI_BASIS_PATH.relative_to(REPO_ROOT)),
            "pi_basis_sha256": pi_basis_sha256(),
            "frozen_design_sha256": design_sha256(),
            "pi_rule": PI_RULE,
            "pi_semantics": PI_SEMANTICS,
            "n_g_rule": N_G_RULE,
            "support_rule": SUPPORT_RULE,
            "seed_namespace": SAMPLER_SEED_NAMESPACE,
            "partition_source": "SEALED data.cifar.group_partition_G2",
            "target": GROUPED_TARGET,
            "binding_asymmetry": BINDING_ASYMMETRY,
            "declared_when": (
                "pi_{j|g} and n_g were NOT in the original sealed prereg (partition only); "
                "CIFAR's pi RULE was declared by the owner 2026-07-16 (9a308e5f... section "
                "5) with zero grouped certification outcomes in existence for any dataset."
            ),
        },
    )
    spec.validate()
    return spec
