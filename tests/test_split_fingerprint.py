"""Split-fingerprint provenance checks (Item 1 of the ws4090 sync).

Standalone, torch-free, no dataset download. Exercises the REAL split pipeline
(``open_set_split`` -> ``dirichlet_partition`` -> ``build_calibration``) plus the
new ``fold_fingerprint`` / ``add_split_fingerprint`` helpers, and verifies the
invariants that make the fingerprint a valid drift detector:

  1. determinism  -- same (seed, split) => same fingerprint (within one env);
  2. round-trip   -- ``add_split_fingerprint`` stores exactly the fingerprint a
                     consumer recomputes from the stored ``{fold}_client/_y_open``;
  3. sensitivity  -- perturbing one fold element changes the fingerprint;
  4. stored-npz   -- recompute from any stored npz is stable & fold-distinct.

NOTE: we deliberately do NOT pin a fingerprint VALUE here. The value is what
detects cross-environment drift (numpy RNG-stream changes at a pinned seed); a
pinned hash would make an env-portable gate falsely fail. This test checks the
mechanism's invariants, which hold in any environment.

Run: ``python3 tests/test_split_fingerprint.py``  (exit 0 = PASS).
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcore.data.fedosr_split import (  # noqa: E402
    add_split_fingerprint,
    build_calibration,
    build_identity_only_traffic,
    dirichlet_partition,
    fold_fingerprint,
    fold_identity_fingerprint,
    open_set_split,
)


def _gather_fold(calib, fold):
    """Local copy of run_cifar._gather_fold (torch-free) for the payload build."""
    idx, y_open, client = [], [], []
    for j, cf in enumerate(calib):
        f = cf[fold]
        idx.append(np.asarray(f["idx"]))
        y_open.append(np.asarray(f["y_open"]))
        client.append(np.full(len(f["idx"]), j))
    return np.concatenate(idx), np.concatenate(y_open), np.concatenate(client)


def _build_payload(
    seed, n_known=6, n_clients=5, dirichlet_alpha=0.5, n_classes=10, n_per_class=500
):
    """Run the actual split pipeline on synthetic labels -> npz-style payload."""
    rng = np.random.default_rng(
        1234
    )  # data itself is fixed; only the split seed varies
    labels = np.repeat(np.arange(n_classes), n_per_class)
    rng.shuffle(labels)

    known, unknown, remap = open_set_split(labels, n_known, seed)
    known_idx = np.where(np.isin(labels, known))[0]
    known_remap = np.array([remap[int(c)] for c in labels[known_idx]])
    _ = dirichlet_partition(known_idx, known_remap, n_clients, dirichlet_alpha, seed)

    # calibration folds from a held-out "test" pool (mirrors run_cifar)
    unknown_idx = np.where(np.isin(labels, unknown))[0]
    calib = build_calibration(
        known_idx,
        known_remap,
        unknown_idx,
        n_clients,
        (0.4, 0.3, 0.3),
        0.30,
        seed,
    )
    raw = {}
    for fold in ("prop", "cert", "test"):
        _idx, y_open, client = _gather_fold(calib, fold)
        raw[f"{fold}_sample_idx"] = _idx
        raw[f"{fold}_y_open"] = y_open
        raw[f"{fold}_client"] = client
    return raw


def check_determinism():
    a = _build_payload(seed=0)
    b = _build_payload(seed=0)
    add_split_fingerprint(a, 0)
    add_split_fingerprint(b, 0)
    for fold in ("prop", "cert", "test"):
        assert str(a[f"{fold}_fp"]) == str(
            b[f"{fold}_fp"]
        ), f"determinism FAIL on {fold}: {a[f'{fold}_fp']} != {b[f'{fold}_fp']}"
    assert str(a["numpy_version"]) == np.__version__
    assert int(a["split_seed"]) == 0
    print(
        f"  [determinism] OK   prop={a['prop_fp']} cert={a['cert_fp']} test={a['test_fp']}"
    )


def check_round_trip():
    raw = _build_payload(seed=3)
    add_split_fingerprint(raw, 3)
    for fold in ("prop", "cert", "test"):
        recomputed = fold_fingerprint(raw[f"{fold}_client"], raw[f"{fold}_y_open"])
        assert recomputed == str(
            raw[f"{fold}_fp"]
        ), f"round-trip FAIL on {fold}: stored {raw[f'{fold}_fp']} != recomputed {recomputed}"
    print("  [round-trip]  OK   stored fp == fp recomputed from stored folds")


def check_sensitivity():
    raw = _build_payload(seed=1)
    base = fold_fingerprint(raw["cert_client"], raw["cert_y_open"])
    # flip one client assignment
    pert_client = raw["cert_client"].copy()
    pert_client[0] = (pert_client[0] + 1) % (pert_client.max() + 1)
    assert (
        fold_fingerprint(pert_client, raw["cert_y_open"]) != base
    ), "client perturbation not detected"
    # flip one label
    pert_y = raw["cert_y_open"].copy()
    pert_y[0] = pert_y[0] + 1
    assert (
        fold_fingerprint(raw["cert_client"], pert_y) != base
    ), "label perturbation not detected"
    # different seed => different split => different fp (overwhelmingly likely)
    other = _build_payload(seed=2)
    assert (
        fold_fingerprint(other["cert_client"], other["cert_y_open"]) != base
    ), "different-seed split collided"
    print("  [sensitivity] OK   single-element / seed perturbations all change the fp")


def check_identity_sensitivity():
    raw = _build_payload(seed=1)
    base = fold_identity_fingerprint(
        raw["cert_sample_idx"], raw["cert_client"], raw["cert_y_open"]
    )
    # Find two same-(client,label) examples. Swapping them is invisible to the
    # legacy fingerprint but must change the identity-committing digest.
    pair = None
    for i in range(len(raw["cert_y_open"])):
        matches = np.flatnonzero(
            (raw["cert_client"] == raw["cert_client"][i])
            & (raw["cert_y_open"] == raw["cert_y_open"][i])
        )
        if len(matches) > 1:
            pair = (int(matches[0]), int(matches[1]))
            break
    assert pair is not None
    swapped = raw["cert_sample_idx"].copy()
    swapped[list(pair)] = swapped[list(pair[::-1])]
    assert fold_fingerprint(raw["cert_client"], raw["cert_y_open"]) == fold_fingerprint(
        raw["cert_client"], raw["cert_y_open"]
    )
    assert (
        fold_identity_fingerprint(swapped, raw["cert_client"], raw["cert_y_open"])
        != base
    ), "same-label sample-ID swap was not detected"
    add_split_fingerprint(raw, 1)
    assert str(raw["cert_identity_sha256"]) == base
    print("  [identity]     OK   same-label sample-ID swap changes identity digest")


def check_identity_only_traffic():
    candidates = np.arange(200, dtype=np.int64)
    first = build_identity_only_traffic(
        candidates,
        n_traffic=60,
        n_clients=3,
        client_probabilities=[0.2, 0.3, 0.5],
        seed=91,
    )
    second = build_identity_only_traffic(
        candidates,
        n_traffic=60,
        n_clients=3,
        client_probabilities=[0.2, 0.3, 0.5],
        seed=91,
    )
    np.testing.assert_array_equal(first["idx"], second["idx"])
    np.testing.assert_array_equal(first["client"], second["client"])
    assert len(np.unique(first["idx"])) == 60
    assert set(first) == {"idx", "client"}
    print("  [traffic]      OK   identity-only draw is unique and replayable")


def check_stored_npz():
    files = sorted(glob.glob("runs/r2_cifar100_*_logits.npz"))[:4]
    if not files:
        print("  [stored-npz]  SKIP (no runs/r2_cifar100_*_logits.npz present)")
        return
    seen = {}
    for f in files:
        d = np.load(f)
        if "test_client" not in d:
            continue
        fp1 = fold_fingerprint(d["test_client"], d["test_y_open"])
        fp2 = fold_fingerprint(d["test_client"], d["test_y_open"])
        assert fp1 == fp2, f"recompute unstable on {f}"
        seen[os.path.basename(f)] = fp1
    assert len(set(seen.values())) == len(
        seen
    ), "distinct runs shared a test-fold fingerprint"
    print(f"  [stored-npz]  OK   {len(seen)} runs, distinct & stable recompute:")
    for name, fp in seen.items():
        print(f"                 {fp}  {name}")


def main():
    print("split-fingerprint verification")
    check_determinism()
    check_round_trip()
    check_sensitivity()
    check_identity_sensitivity()
    check_identity_only_traffic()
    check_stored_npz()
    print("ALL PASS")


if __name__ == "__main__":
    main()
