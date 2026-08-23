"""Regression tests for portable, semantically isolated RNG seeds.

The file is both a normal pytest module and a dependency-free standalone gate:

    python tests/test_semantic_seeds.py
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedcore.seeds import (  # noqa: E402
    SEED_DERIVATION_ALGORITHM,
    SEED_NAMESPACES,
    ForbiddenSeedContextError,
    SeedBundle,
    SeedLedgerError,
    SeedNamespace,
    UnknownSeedNamespaceError,
    derive_seed,
)


EXPECTED_NAMESPACES = (
    "class_split",
    "partition",
    "fold",
    "model_init",
    "loader",
    "label_noise",
    "audit_draw",
    "traffic_draw",
    "solver",
    "stability",
)


def _raises(expected, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except expected as exc:
        return exc
    except Exception as exc:  # pragma: no cover - produces a clearer standalone failure
        raise AssertionError(
            f"expected {expected.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {expected.__name__} to be raised")


def _bundle(audit_draw_index=0):
    return SeedBundle.derive(
        7,
        common_context={
            "dataset": "cifar10",
            "experiment_id": "cifar10/split0/model0",
        },
        namespace_contexts={
            "class_split": {"split_index": 0},
            "partition": {"dirichlet_alpha": 0.5},
            "model_init": {"model_replicate": 0},
            "loader": {"epoch": 0},
            "label_noise": {"noise_replicate": 0},
            "audit_draw": {"draw_index": audit_draw_index},
            "traffic_draw": {"draw_index": 0},
            "solver": {"restart": 0},
            "stability": {"redraw_index": 0},
        },
    )


def test_namespace_contract_and_domain_isolation():
    assert SEED_NAMESPACES == EXPECTED_NAMESPACES
    assert tuple(namespace.value for namespace in SeedNamespace) == EXPECTED_NAMESPACES

    seeds = [
        derive_seed(123, namespace, run_id="run-17", replicate=2)
        for namespace in SEED_NAMESPACES
    ]
    assert len(set(seeds)) == len(SEED_NAMESPACES), seeds
    assert all(0 <= seed <= 2**32 - 1 for seed in seeds)

    # Canonical mapping order cannot perturb the stream.
    left = derive_seed(123, "partition", {"dataset": "cifar10", "split": 2})
    right = derive_seed(123, "partition", {"split": 2, "dataset": "cifar10"})
    assert left == right

    # Unknown/free-form namespaces are rejected; every stream is domain-separated.
    _raises(UnknownSeedNamespaceError, derive_seed, 123, "generic", {"run_id": "x"})


def test_namespace_specific_context_changes_only_that_stream():
    before = _bundle(audit_draw_index=0)
    after = _bundle(audit_draw_index=1)
    changed = {
        namespace
        for namespace in SEED_NAMESPACES
        if before[namespace] != after[namespace]
    }
    assert changed == {"audit_draw"}
    assert before.context("audit_draw")["draw_index"] == 0
    assert after.context("audit_draw")["draw_index"] == 1


def test_cross_process_stability_and_golden_value():
    # A pinned value catches accidental changes to canonicalization or byte order.
    expected = 3_587_295_643
    assert derive_seed(123, "class_split", dataset="cifar10", split_index=2) == expected

    code = """
import json
from fedcore.seeds import SEED_NAMESPACES, derive_seed
print(json.dumps({n: derive_seed(123, n, {"tags": {"b": 2, "a": 1},
                                                "run_id": "cross-process"})
                  for n in SEED_NAMESPACES}, sort_keys=True))
"""
    outputs = []
    for hash_seed in ("1", "8675309", "random"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code],
                cwd=str(ROOT),
                env=env,
                text=True,
            ).strip()
        )
    assert outputs[0] == outputs[1] == outputs[2]
    assert set(json.loads(outputs[0])) == set(EXPECTED_NAMESPACES)


def test_audit_and_traffic_reject_every_posthoc_analysis_knob():
    forbidden_examples = (
        "alpha",
        "DELTA",
        "rho",
        "threshold-policy",
        "allocation_policy",
        "score_name",
        "solver",
        "certificate variant",
        "cert_method",
    )
    for namespace in ("audit_draw", "traffic_draw"):
        for key in forbidden_examples:
            exc = _raises(
                ForbiddenSeedContextError,
                derive_seed,
                9,
                namespace,
                {"experiment_id": "fixed-run", "nested": {key: "post-hoc-value"}},
            )
            assert namespace in str(exc)

    # These names are analysis inputs only in draw namespaces.  The solver's own
    # namespaced stream may naturally identify its deterministic restart method.
    derive_seed(9, "solver", solver="enumeration", certificate_variant="box")


def test_audit_and_traffic_are_invariant_across_posthoc_grid():
    immutable_draw_context = {
        "experiment_id": "cifar10/split2/model1",
        "fold_hash": "sha256:abc123",
        "draw_index": 4,
    }
    audit_seed = derive_seed(44, "audit_draw", immutable_draw_context)
    traffic_seed = derive_seed(44, "traffic_draw", immutable_draw_context)

    # The grid is deliberately kept out of draw identity.  Every competing cell
    # therefore consumes common random numbers and identical sample identifiers.
    posthoc_grid = (
        {
            "alpha": 0.05,
            "delta": 0.05,
            "rho": 0.0,
            "policy": "global",
            "score": "msp",
            "solver": "exact",
            "certificate_variant": "simplex",
        },
        {
            "alpha": 0.20,
            "delta": 0.10,
            "rho": 0.3,
            "policy": "client_specific",
            "score": "energy",
            "solver": "enumeration",
            "certificate_variant": "box",
        },
    )
    observed = []
    for analysis_cell in posthoc_grid:
        assert analysis_cell  # analysis metadata exists but is not draw identity
        observed.append(
            (
                derive_seed(44, "audit_draw", immutable_draw_context),
                derive_seed(44, "traffic_draw", immutable_draw_context),
            )
        )
    assert observed == [(audit_seed, traffic_seed), (audit_seed, traffic_seed)]

    # Fail closed if a caller accidentally merges a post-hoc cell into context.
    for namespace in ("audit_draw", "traffic_draw"):
        contaminated = dict(immutable_draw_context)
        contaminated.update(posthoc_grid[0])
        _raises(ForbiddenSeedContextError, derive_seed, 44, namespace, contaminated)


def test_seed_bundle_json_round_trip_and_replay_validation():
    bundle = _bundle()
    assert bundle.algorithm == SEED_DERIVATION_ALGORITHM
    assert tuple(entry.namespace for entry in bundle.ledger) == EXPECTED_NAMESPACES
    assert bundle.class_split == bundle["class_split"]
    assert bundle.audit_draw == bundle[SeedNamespace.AUDIT_DRAW]
    assert bundle.seeds == {entry.namespace: entry.seed for entry in bundle.entries}

    encoded = bundle.to_json()
    restored = SeedBundle.from_json(encoded)
    assert restored == bundle
    assert restored.to_json() == encoded
    assert SeedBundle.from_dict(bundle.to_dict()) == bundle

    # A ledger is replayable evidence, not a bag of trusted integers.
    tampered_seed = copy.deepcopy(bundle.to_dict())
    tampered_seed["entries"][0]["seed"] ^= 1
    _raises(SeedLedgerError, SeedBundle.from_dict, tampered_seed)

    tampered_context = copy.deepcopy(bundle.to_dict())
    tampered_context["entries"][0]["context"]["split_index"] = 99
    _raises(SeedLedgerError, SeedBundle.from_dict, tampered_context)


def test_bundle_rejects_posthoc_knobs_even_when_common_context_is_contaminated():
    _raises(
        ForbiddenSeedContextError,
        SeedBundle.derive,
        1,
        common_context={"experiment_id": "fixed", "alpha": 0.1},
    )


def main():
    tests = (
        test_namespace_contract_and_domain_isolation,
        test_namespace_specific_context_changes_only_that_stream,
        test_cross_process_stability_and_golden_value,
        test_audit_and_traffic_reject_every_posthoc_analysis_knob,
        test_audit_and_traffic_are_invariant_across_posthoc_grid,
        test_seed_bundle_json_round_trip_and_replay_validation,
        test_bundle_rejects_posthoc_knobs_even_when_common_context_is_contaminated,
    )
    for test in tests:
        test()
        print(f"  [PASS] {test.__name__}")
    print(f"semantic seed checks: PASS ({len(tests)} tests)")


if __name__ == "__main__":
    main()
