"""Stable, semantically namespaced seeds for Fed-CORE experiments.

One integer called ``seed`` is not enough provenance for an experiment.  It also
invites accidental coupling: changing a post-hoc certificate option can silently
change the audit sample if that option is included in a seed hash.  This module
gives every stochastic operation an explicit namespace and derives a portable
32-bit seed from a master seed plus JSON-serializable semantic context.

The derivation is deliberately independent of Python's process-salted ``hash``
and of NumPy's implementation.  The complete derivation input is retained in a
serializable :class:`SeedBundle`; loading a bundle recomputes every seed and
rejects a stale or edited ledger.

Audit and traffic draws have an additional fail-closed rule.  Their context may
describe immutable inputs (for example ``experiment_id``, ``fold_hash`` and
``draw_index``), but it may not contain analysis knobs such as alpha, delta,
rho, policy, score, solver, or certificate variant.  Consequently competing
post-hoc analyses must consume common audit and traffic draws.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


class SeedNamespace(str, Enum):
    """Domain separators for every stochastic operation in the campaign."""

    CLASS_SPLIT = "class_split"
    PARTITION = "partition"
    FOLD = "fold"
    MODEL_INIT = "model_init"
    LOADER = "loader"
    LABEL_NOISE = "label_noise"
    AUDIT_DRAW = "audit_draw"
    TRAFFIC_DRAW = "traffic_draw"
    SOLVER = "solver"
    STABILITY = "stability"


SEED_NAMESPACES: Tuple[str, ...] = tuple(namespace.value for namespace in SeedNamespace)
"""Canonical ledger order and the complete set of supported namespaces."""

DRAW_NAMESPACES = frozenset(
    {SeedNamespace.AUDIT_DRAW.value, SeedNamespace.TRAFFIC_DRAW.value}
)

# These are public so schedulers and schema validators can use the same contract.
# Validation also catches tokenized spellings such as ``threshold-policy`` and
# ``score_name``; see ``_forbidden_draw_key`` below.
FORBIDDEN_DRAW_CONTEXT_KEYS = frozenset(
    {
        "alpha",
        "delta",
        "rho",
        "policy",
        "threshold_policy",
        "allocation_policy",
        "confidence_allocation_policy",
        "score",
        "score_name",
        "solver",
        "solver_name",
        "certificate_variant",
        "certificate_type",
        "certificate_method",
        "cert_variant",
        "cert_type",
        "cert_method",
    }
)

SEED_LEDGER_SCHEMA_VERSION = 1
SEED_DERIVATION_ALGORITHM = "blake2b-32-v1"
_DERIVATION_DOMAIN = b"fedcore.semantic-seed.v1\x00"
_MAX_MASTER_SEED = (1 << 64) - 1
_MAX_DERIVED_SEED = (1 << 32) - 1
_TOKEN_FORBIDDEN_DRAW_KEYS = frozenset(
    {"alpha", "delta", "rho", "policy", "score", "solver"}
)


class SeedError(ValueError):
    """Base class for invalid semantic seed inputs or ledgers."""


class UnknownSeedNamespaceError(SeedError):
    """Raised when a caller uses a namespace outside :data:`SEED_NAMESPACES`."""


class ForbiddenSeedContextError(SeedError):
    """Raised when an audit/traffic seed depends on a post-hoc analysis knob."""


class SeedLedgerError(SeedError):
    """Raised when a serialized ledger is malformed or fails replay validation."""


def _master_seed(value: Any) -> int:
    """Return a portable unsigned 64-bit master seed, rejecting bools/floats."""

    if isinstance(value, bool):
        raise SeedError("master_seed must be an integer, not bool")
    try:
        seed = operator.index(value)
    except TypeError as exc:
        raise SeedError("master_seed must be an integer") from exc
    if not 0 <= seed <= _MAX_MASTER_SEED:
        raise SeedError(f"master_seed must be in [0, {_MAX_MASTER_SEED}]")
    return int(seed)


def _namespace(value: Any) -> str:
    if isinstance(value, SeedNamespace):
        return value.value
    if not isinstance(value, str) or value not in SEED_NAMESPACES:
        raise UnknownSeedNamespaceError(
            f"unknown seed namespace {value!r}; expected one of {SEED_NAMESPACES}"
        )
    return value


def _canonicalize(value: Any, path: str = "context") -> Any:
    """Convert a value to a strict, deterministic JSON-compatible structure."""

    if isinstance(value, Enum):
        return _canonicalize(value.value, path)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SeedError(f"{path} contains a non-finite float")
        # JSON keeps 1 and 1.0 distinct, which is useful for a provenance ledger.
        return float(value)
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SeedError(f"{path} contains non-string key {key!r}")
            result[key] = _canonicalize(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _canonicalize(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    raise SeedError(
        f"{path} contains unsupported value {value!r} of type "
        f"{type(value).__name__}; seed context must be JSON-serializable"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")


def _forbidden_draw_key(key: str) -> bool:
    normalized = _normalized_key(key)
    if normalized in FORBIDDEN_DRAW_CONTEXT_KEYS:
        return True
    tokens = frozenset(token for token in normalized.split("_") if token)
    if tokens & _TOKEN_FORBIDDEN_DRAW_KEYS:
        return True
    # Catch ordinary aliases without banning legitimate fold identity keys such
    # as ``certification_fold_hash``.
    return bool(
        ({"certificate", "variant"} <= tokens)
        or ({"cert", "variant"} <= tokens)
        or ({"certificate", "method"} <= tokens)
        or ({"cert", "method"} <= tokens)
        or ({"certificate", "type"} <= tokens)
        or ({"cert", "type"} <= tokens)
    )


def _iter_mapping_keys(value: Any, path: str = "context") -> Iterable[Tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            yield key, child_path
            yield from _iter_mapping_keys(item, child_path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_mapping_keys(item, f"{path}[{index}]")


def _validate_draw_context(namespace: str, context: Mapping[str, Any]) -> None:
    if namespace not in DRAW_NAMESPACES:
        return
    forbidden = [
        path for key, path in _iter_mapping_keys(context) if _forbidden_draw_key(key)
    ]
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise ForbiddenSeedContextError(
            f"{namespace} seed context contains post-hoc analysis knob(s): {names}. "
            "Audit and traffic draws must depend only on immutable run/draw identity."
        )


def canonical_seed_context(
    namespace: str | SeedNamespace,
    context: Optional[Mapping[str, Any]] = None,
    **components: Any,
) -> Dict[str, Any]:
    """Validate and canonicalize context for ``namespace``.

    ``context`` and keyword components may be combined, but duplicate keys are
    rejected instead of silently overriding provenance.
    """

    namespace_value = _namespace(namespace)
    if context is None:
        merged: Dict[str, Any] = {}
    elif not isinstance(context, Mapping):
        raise SeedError("context must be a mapping")
    else:
        merged = dict(context)
    duplicates = sorted(set(merged).intersection(components))
    if duplicates:
        raise SeedError(f"duplicate seed context key(s): {', '.join(duplicates)}")
    merged.update(components)
    canonical = _canonicalize(merged)
    _validate_draw_context(namespace_value, canonical)
    return canonical


def derive_seed(
    master_seed: int,
    namespace: str | SeedNamespace,
    context: Optional[Mapping[str, Any]] = None,
    **components: Any,
) -> int:
    """Derive a stable unsigned 32-bit seed from semantic inputs.

    The same inputs produce the same result across processes and machines.  The
    namespace is part of the hash input, so two operations never intentionally
    share an RNG stream merely because their context is equal.
    """

    root = _master_seed(master_seed)
    namespace_value = _namespace(namespace)
    canonical = canonical_seed_context(namespace_value, context, **components)
    payload = {
        "algorithm": SEED_DERIVATION_ALGORITHM,
        "master_seed": root,
        "namespace": namespace_value,
        "context": canonical,
    }
    encoded = _DERIVATION_DOMAIN + _canonical_json(payload).encode("ascii")
    digest = hashlib.blake2b(encoded, digest_size=4).digest()
    seed = int.from_bytes(digest, byteorder="big", signed=False)
    assert 0 <= seed <= _MAX_DERIVED_SEED  # structural, not input-dependent
    return seed


def _context_digest(context: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(context).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class SeedLedgerEntry:
    """One replayable seed derivation record."""

    namespace: str
    seed: int
    context: Dict[str, Any] = field(repr=True)
    context_sha256: str = ""

    def __post_init__(self) -> None:
        namespace_value = _namespace(self.namespace)
        canonical = canonical_seed_context(namespace_value, self.context)
        try:
            seed_value = operator.index(self.seed)
        except TypeError as exc:
            raise SeedLedgerError("ledger seed must be an integer") from exc
        if isinstance(self.seed, bool) or not 0 <= seed_value <= _MAX_DERIVED_SEED:
            raise SeedLedgerError(f"ledger seed must be in [0, {_MAX_DERIVED_SEED}]")
        digest = _context_digest(canonical)
        if self.context_sha256 and self.context_sha256 != digest:
            raise SeedLedgerError(
                f"context digest mismatch for namespace {namespace_value!r}"
            )
        # Defensive canonical copies make key ordering and tuple/list differences
        # irrelevant in the ledger. ``context`` should be treated as read-only.
        object.__setattr__(self, "namespace", namespace_value)
        object.__setattr__(self, "seed", int(seed_value))
        object.__setattr__(self, "context", canonical)
        object.__setattr__(self, "context_sha256", digest)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace,
            "seed": self.seed,
            "context": json.loads(_canonical_json(self.context)),
            "context_sha256": self.context_sha256,
        }


@dataclass(frozen=True)
class SeedBundle:
    """Complete, serializable ledger of all Fed-CORE semantic seed streams."""

    master_seed: int
    entries: Tuple[SeedLedgerEntry, ...]
    schema_version: int = SEED_LEDGER_SCHEMA_VERSION
    algorithm: str = SEED_DERIVATION_ALGORITHM

    def __post_init__(self) -> None:
        root = _master_seed(self.master_seed)
        if self.schema_version != SEED_LEDGER_SCHEMA_VERSION:
            raise SeedLedgerError(
                f"unsupported seed ledger schema_version {self.schema_version!r}"
            )
        if self.algorithm != SEED_DERIVATION_ALGORITHM:
            raise SeedLedgerError(
                f"unsupported seed derivation algorithm {self.algorithm!r}"
            )
        entries = tuple(
            (
                entry
                if isinstance(entry, SeedLedgerEntry)
                else SeedLedgerEntry(**entry)
            )  # type: ignore[arg-type]
            for entry in self.entries
        )
        namespaces = tuple(entry.namespace for entry in entries)
        if len(set(namespaces)) != len(namespaces):
            raise SeedLedgerError("seed ledger contains duplicate namespaces")
        if set(namespaces) != set(SEED_NAMESPACES):
            missing = sorted(set(SEED_NAMESPACES) - set(namespaces))
            extra = sorted(set(namespaces) - set(SEED_NAMESPACES))
            raise SeedLedgerError(
                f"seed ledger namespace mismatch; missing={missing}, extra={extra}"
            )
        by_namespace = {entry.namespace: entry for entry in entries}
        ordered = tuple(by_namespace[namespace] for namespace in SEED_NAMESPACES)
        for entry in ordered:
            expected = derive_seed(root, entry.namespace, entry.context)
            if entry.seed != expected:
                raise SeedLedgerError(
                    f"seed mismatch for namespace {entry.namespace!r}: "
                    f"ledger={entry.seed}, replay={expected}"
                )
        object.__setattr__(self, "master_seed", root)
        object.__setattr__(self, "entries", ordered)

    @classmethod
    def derive(
        cls,
        master_seed: int,
        *,
        common_context: Optional[Mapping[str, Any]] = None,
        namespace_contexts: Optional[
            Mapping[str | SeedNamespace, Mapping[str, Any]]
        ] = None,
    ) -> "SeedBundle":
        """Derive all namespaces and retain their full semantic contexts.

        ``common_context`` is merged into every namespace.  Namespace-specific
        keys must not overlap common keys, preventing an accidental silent
        override.  Draw-context restrictions apply after the merge.
        """

        root = _master_seed(master_seed)
        common = canonical_seed_context(SeedNamespace.CLASS_SPLIT, common_context)
        provided: Dict[str, Mapping[str, Any]] = {}
        for raw_namespace, raw_context in (namespace_contexts or {}).items():
            namespace_value = _namespace(raw_namespace)
            if namespace_value in provided:
                raise SeedError(f"duplicate namespace context {namespace_value!r}")
            if not isinstance(raw_context, Mapping):
                raise SeedError(
                    f"namespace context for {namespace_value!r} must be a mapping"
                )
            provided[namespace_value] = raw_context

        entries = []
        for namespace in SEED_NAMESPACES:
            specific = provided.get(namespace, {})
            duplicates = sorted(set(common).intersection(specific))
            if duplicates:
                raise SeedError(
                    f"namespace {namespace!r} duplicates common context key(s): "
                    f"{', '.join(duplicates)}"
                )
            merged = dict(common)
            merged.update(specific)
            context = canonical_seed_context(namespace, merged)
            entries.append(
                SeedLedgerEntry(
                    namespace=namespace,
                    seed=derive_seed(root, namespace, context),
                    context=context,
                )
            )
        return cls(master_seed=root, entries=tuple(entries))

    # Naming aliases make the intended call readable in runners and manifests.
    from_master_seed = derive
    from_base_seed = derive

    def seed(self, namespace: str | SeedNamespace) -> int:
        namespace_value = _namespace(namespace)
        for entry in self.entries:
            if entry.namespace == namespace_value:
                return entry.seed
        raise AssertionError("validated bundle is missing a namespace")

    def context(self, namespace: str | SeedNamespace) -> Dict[str, Any]:
        namespace_value = _namespace(namespace)
        for entry in self.entries:
            if entry.namespace == namespace_value:
                return json.loads(_canonical_json(entry.context))
        raise AssertionError("validated bundle is missing a namespace")

    def __getitem__(self, namespace: str | SeedNamespace) -> int:
        return self.seed(namespace)

    @property
    def seeds(self) -> Dict[str, int]:
        return {entry.namespace: entry.seed for entry in self.entries}

    @property
    def ledger(self) -> Tuple[SeedLedgerEntry, ...]:
        return self.entries

    @property
    def class_split(self) -> int:
        return self.seed(SeedNamespace.CLASS_SPLIT)

    @property
    def partition(self) -> int:
        return self.seed(SeedNamespace.PARTITION)

    @property
    def fold(self) -> int:
        return self.seed(SeedNamespace.FOLD)

    @property
    def model_init(self) -> int:
        return self.seed(SeedNamespace.MODEL_INIT)

    @property
    def loader(self) -> int:
        return self.seed(SeedNamespace.LOADER)

    @property
    def label_noise(self) -> int:
        return self.seed(SeedNamespace.LABEL_NOISE)

    @property
    def audit_draw(self) -> int:
        return self.seed(SeedNamespace.AUDIT_DRAW)

    @property
    def traffic_draw(self) -> int:
        return self.seed(SeedNamespace.TRAFFIC_DRAW)

    @property
    def solver(self) -> int:
        return self.seed(SeedNamespace.SOLVER)

    @property
    def stability(self) -> int:
        return self.seed(SeedNamespace.STABILITY)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "master_seed": self.master_seed,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    as_dict = to_dict

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Serialize deterministically; compact by default, pretty when requested."""

        separators = (",", ":") if indent is None else None
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=separators,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SeedBundle":
        if not isinstance(payload, Mapping):
            raise SeedLedgerError("seed ledger payload must be a mapping")
        required = {"schema_version", "algorithm", "master_seed", "entries"}
        missing = sorted(required - set(payload))
        if missing:
            raise SeedLedgerError(
                f"seed ledger is missing key(s): {', '.join(missing)}"
            )
        raw_entries = payload["entries"]
        if not isinstance(raw_entries, list):
            raise SeedLedgerError("seed ledger entries must be a list")
        try:
            entries = tuple(SeedLedgerEntry(**raw_entry) for raw_entry in raw_entries)
        except (TypeError, SeedError) as exc:
            if isinstance(exc, SeedLedgerError):
                raise
            raise SeedLedgerError(f"malformed seed ledger entry: {exc}") from exc
        return cls(
            master_seed=payload["master_seed"],
            entries=entries,
            schema_version=payload["schema_version"],
            algorithm=payload["algorithm"],
        )

    @classmethod
    def from_json(cls, payload: str) -> "SeedBundle":
        if not isinstance(payload, str):
            raise SeedLedgerError("seed ledger JSON payload must be a string")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SeedLedgerError(f"invalid seed ledger JSON: {exc}") from exc
        return cls.from_dict(decoded)


def make_seed_bundle(
    master_seed: int,
    *,
    common_context: Optional[Mapping[str, Any]] = None,
    namespace_contexts: Optional[
        Mapping[str | SeedNamespace, Mapping[str, Any]]
    ] = None,
) -> SeedBundle:
    """Functional alias for :meth:`SeedBundle.derive`."""

    return SeedBundle.derive(
        master_seed,
        common_context=common_context,
        namespace_contexts=namespace_contexts,
    )


__all__ = [
    "DRAW_NAMESPACES",
    "FORBIDDEN_DRAW_CONTEXT_KEYS",
    "SEED_DERIVATION_ALGORITHM",
    "SEED_LEDGER_SCHEMA_VERSION",
    "SEED_NAMESPACES",
    "ForbiddenSeedContextError",
    "SeedBundle",
    "SeedError",
    "SeedLedgerEntry",
    "SeedLedgerError",
    "SeedNamespace",
    "UnknownSeedNamespaceError",
    "canonical_seed_context",
    "derive_seed",
    "make_seed_bundle",
]
