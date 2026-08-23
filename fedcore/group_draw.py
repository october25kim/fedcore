"""The EXACT grouped certification draw: the declared sampler, in the production path.

What this fixes
---------------
The grouped certificate previously RELABELLED client ids to group ids and certified the
fixed-quota certification fold as if it were a group sample. Fixed quotas are
deterministic largest-remainder allocations, not draws from the declared group mixture,
so ``K_g | A_g ~ Bin(A_g, r_g)`` did not hold exactly and the "grouped" numbers were a
diagnostic, not a certificate.

This module routes the draw through the pre-registered sampler
``fedcore.sampling.sample_group_mixture``, so the conditional binomial law holds EXACTLY.

The declared law (per group g)
------------------------------
Draw ``n_g`` observations. Each observation independently:

    client j ~ Categorical(pi_{.|g})
    one unit ~ Uniform(client j's FROZEN certification reservoir), WITH REPLACEMENT

Each of the n_g observations is therefore i.i.d. from the group-g mixture. Conditional
on the realised accepted count A_g, the accepted units are i.i.d. from the
acceptance-conditioned group-g mixture, so ``K_g | A_g ~ Bin(A_g, r_g)`` EXACTLY, with
``r_g = P(error | accepted, group g)``. That is the law the certificate assumes.

WHICH DECLARED FORM IS IMPLEMENTED, and why
-------------------------------------------
The owner allowed either the per-observation categorical form or the equivalent
pre-outcome multinomial form. **The per-observation categorical form is implemented**,
because it is the one that is deterministically replayable UNIT BY UNIT: it is exactly
what ``sample_group_mixture`` already implements, and its returned
``GroupMixtureDraw`` carries the per-observation ``client_id``, ``reservoir_position``
and immutable ``sample_id``, so a replay reproduces the sampled client ids AND the
source unit ids -- not merely their counts. The multinomial form fixes only the counts
``(N_g1..N_gJ)``, from which the identity of each drawn unit is not recoverable, so it
is strictly weaker for audit. The two forms are distributionally identical.

Group sampling is done PER GROUP with a one-hot group probability vector, so the group
sizes are the declared ``n_g`` rather than random multinomial group counts. Conditioning
on the group is exactly what the grouped-stratified certificate does.

Seeding
-------
The sampler consumes the pre-registered ``audit`` namespace, whose SEALED scope
``(dataset, pipeline, split_id, train_seed, d)`` masks ``alpha`` and
``certificate_variant``. That masking is load-bearing: every competing post-hoc analysis
shares ONE common draw, so no certification outcome can select the draw. A per-group
sub-stream is derived with ``SeedSequence([audit_seed, group_index])`` -- deterministic,
and independent of how many groups happen to be drawn first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from fedcore.sampling import sample_group_mixture


@dataclass(frozen=True)
class GroupMixtureSpec:
    """A fully-declared group mixture: partition + pi_{j|g} + n_g + seed.

    Every field must come from a declaration or a derivation from sealed values --
    never from an outcome, and never from a runtime convenience default. Construct it
    with :func:`fedcore.medical.group_mixture.spec_for_task` for Fed-ISIC.
    """

    group_names: tuple[str, ...]
    client_to_group: np.ndarray          # client id -> group index
    pi: Dict[int, np.ndarray]            # group index -> probs over that group's SORTED clients
    n_g: Dict[int, int]                  # group index -> declared draw count
    seed: int
    provenance: Dict[str, object]

    @property
    def G(self) -> int:
        return len(self.group_names)

    def clients_in(self, group_index: int) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.client_to_group) == int(group_index))

    def validate(self) -> None:
        for index, name in enumerate(self.group_names):
            clients = self.clients_in(index)
            probs = np.asarray(self.pi[index], dtype=float)
            if len(probs) != len(clients):
                raise ValueError(f"group {name}: pi has {len(probs)} entries for {len(clients)} clients")
            if not np.isclose(probs.sum(), 1.0, rtol=0.0, atol=1e-12):
                raise ValueError(f"group {name}: pi sums to {probs.sum()}, not 1")
            if int(self.n_g[index]) < 0:
                raise ValueError(f"group {name}: n_g is negative")


@dataclass(frozen=True)
class GroupDrawRecord:
    """The full, replayable record of one grouped certification draw."""

    group_index: np.ndarray        # per observation
    client_id: np.ndarray          # per observation -- the Categorical(pi) outcome
    source_id: np.ndarray          # per observation -- the immutable drawn unit id
    reservoir_position: np.ndarray
    seed_per_group: Dict[int, int]
    n_g: Dict[int, int]
    draw_construction: str
    sampler: str

    def multiplicities(self) -> Dict[object, int]:
        """How many times each source unit was drawn (with replacement)."""
        out: Dict[object, int] = {}
        for sid in self.source_id.tolist():
            out[sid] = out.get(sid, 0) + 1
        return out


def _group_seed(base_seed: int, group_index: int) -> int:
    """Deterministic per-group sub-stream; order-independent by construction."""
    return int(
        np.random.SeedSequence([int(base_seed), int(group_index)]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def draw_group_certification_sample(
    cert_view: Mapping[str, np.ndarray],
    spec: GroupMixtureSpec,
    *,
    id_key: str | None = None,
) -> tuple[Dict[str, np.ndarray], GroupDrawRecord]:
    """Draw the certification sample from the DECLARED group mixture.

    Returns ``(drawn_view, record)``. ``drawn_view`` has the same keys as ``cert_view``,
    with each row a drawn observation (WITH REPLACEMENT, so rows may repeat), and
    ``client`` REPLACED by the group index -- the stratum the certificate conditions on.

    ``cert_view`` is the FROZEN certification reservoir and is never mutated.
    """
    spec.validate()
    client = np.asarray(cert_view["client"])
    n_units = len(client)
    ids = (
        np.asarray(cert_view[id_key])
        if id_key is not None and id_key in cert_view
        else np.arange(n_units)
    )

    # Reservoir per client: the immutable ids of that client's certification units.
    reservoir_ids: Dict[int, list] = {
        int(j): ids[client == j].tolist() for j in range(len(spec.client_to_group))
    }
    id_to_row = {sid: row for row, sid in enumerate(ids.tolist())}

    group_of_obs: list[int] = []
    client_of_obs: list[int] = []
    source_of_obs: list = []
    position_of_obs: list[int] = []
    seed_per_group: Dict[int, int] = {}

    for index in range(spec.G):
        n = int(spec.n_g[index])
        seed = _group_seed(spec.seed, index)
        seed_per_group[index] = seed
        if n == 0:
            continue
        # One-hot over groups: we draw n_g observations FROM GROUP index, which is what
        # a grouped-stratified certificate conditions on. The client step below is the
        # declared Categorical(pi_{.|g}).
        group_probabilities = np.zeros(spec.G, dtype=float)
        group_probabilities[index] = 1.0
        conditional = {g: np.asarray(spec.pi[g], dtype=float) for g in range(spec.G)}

        draw = sample_group_mixture(
            reservoir_ids=reservoir_ids,
            client_to_group=spec.client_to_group,
            group_probabilities=group_probabilities,
            client_probabilities_given_group=conditional,
            n=n,
            seed=seed,
        )
        group_of_obs.extend(draw.group_id.tolist())
        client_of_obs.extend(draw.client_id.tolist())
        source_of_obs.extend(draw.sample_id.tolist())
        position_of_obs.extend(draw.reservoir_position.tolist())

    rows = np.array([id_to_row[sid] for sid in source_of_obs], dtype=int)
    drawn: Dict[str, np.ndarray] = {
        key: np.asarray(value)[rows] for key, value in cert_view.items()
    }
    # The certificate strata ARE the groups.
    drawn["client"] = np.asarray(group_of_obs, dtype=int)

    record = GroupDrawRecord(
        group_index=np.asarray(group_of_obs, dtype=int),
        client_id=np.asarray(client_of_obs, dtype=int),
        source_id=np.asarray(source_of_obs),
        reservoir_position=np.asarray(position_of_obs, dtype=int),
        seed_per_group=seed_per_group,
        n_g={g: int(spec.n_g[g]) for g in range(spec.G)},
        draw_construction="categorical_then_reservoir_with_replacement",
        sampler="fedcore.sampling.sample_group_mixture",
    )
    return drawn, record
