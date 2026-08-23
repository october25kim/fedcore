"""Common-random-number and nested audit-budget regression test."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcore.accounting.draws import bootstrap_draw, derive_audit_seed  # noqa: E402
from fedcore.accounting.ids import recover_fold_ids  # noqa: E402
from fedcore.accounting.provenance import discover_runs  # noqa: E402


def test_nested_bootstrap_prefixes():
    candidate = next((item for item in discover_runs(".") if item[1] is not None), None)
    if candidate is None:
        print("nested accounting test: SKIP (no resolvable frozen run)")
        return
    path, spec = candidate
    seed = derive_audit_seed("bootstrap", spec.run_id, "replicate0")
    with np.load(path, allow_pickle=True) as z:
        ids = recover_fold_ids(spec, npz=z)
        small = bootstrap_draw(spec, z, ids, alpha=0.2, ratio=0.5, audit_seed=seed)
        large = bootstrap_draw(spec, z, ids, alpha=0.2, ratio=2.0, audit_seed=seed)
    assert small.audit_seed == large.audit_seed
    for a, b in zip(small.strata, large.strata):
        assert np.array_equal(a.sampled_ids, b.sampled_ids[: len(a.sampled_ids)])


def main():
    test_nested_bootstrap_prefixes()
    print("nested accounting tests: PASS")


if __name__ == "__main__":
    main()
