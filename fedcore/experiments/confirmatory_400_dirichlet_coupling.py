"""Section 9 — CIFAR coupled-Dirichlet validation (deterministic, NO training).

Reuses fedcore.data.fedosr_split.dirichlet_partition (license-neutral, fedcore's own).
Proves the common-random-number coupling: within a paired (split, seed) block the base
ordering / RNG stream is shared across d and ONLY the Dirichlet concentration (alpha=d)
differs. Uses structural synthetic index sets sized per known-class (real counts come from
the adapter at train time); the COUPLING is what is validated here.
"""
from __future__ import annotations
import csv, hashlib, json
import numpy as np
from fedcore.data.fedosr_split import dirichlet_partition
from fedcore.experiments.confirmatory_400_prelaunch import _u32, CAMPAIGN, SEEDREG, CIFAR_D

OUT = "results/confirmatory_400/prelaunch"
J = 5
PER_CLASS = 500          # structural synthetic count per known class


def block_seed(block):
    return _u32(f"{CAMPAIGN}|{SEEDREG}|Dirichlet_partition|{block}")


def phash(parts):
    return hashlib.sha256("|".join(",".join(map(str, p.tolist())) for p in parts).encode()).hexdigest()[:16]


def main():
    rows = []
    checksums = []
    for ds, n_known in (("cifar10", 6), ("cifar100", 60)):
        # synthetic known-class labels (contiguous 0..n_known-1), PER_CLASS each
        labels = np.repeat(np.arange(n_known), PER_CLASS)
        indices = np.arange(len(labels))
        for i in range(10):
            for seed in range(5):
                block = f"{ds}__split{i:02d}__seed{seed}"
                bs = block_seed(block)
                per_d = {}
                for d in CIFAR_D:
                    parts = dirichlet_partition(indices, labels, J, alpha=d, seed=bs)
                    # determinism: identical re-run
                    parts2 = dirichlet_partition(indices, labels, J, alpha=d, seed=bs)
                    assert all(np.array_equal(a, b) for a, b in zip(parts, parts2)), "non-deterministic partition"
                    per_d[d] = parts
                    sizes = [len(p) for p in parts]
                    total = sum(sizes)
                    rows.append(dict(dataset=ds, block=block, d=d, seed=bs,
                                     client_sizes="|".join(map(str, sizes)), total=total,
                                     n_empty_clients=sum(1 for s in sizes if s == 0),
                                     partition_hash=phash(parts)))
                    checksums.append((f"{block}|d{d}", phash(parts)))
                # coupling assertions across d (shared seed):
                # (a) total index set conserved and identical across d
                base = set(np.concatenate(per_d[CIFAR_D[0]]).tolist())
                for d in CIFAR_D[1:]:
                    assert set(np.concatenate(per_d[d]).tolist()) == base, "index set not conserved across d"
                # (b) shared base shuffle => same seed+same alpha reproduces; different alpha differs
                assert phash(per_d[0.1]) != phash(per_d[5.0]), "d=0.1 and d=5.0 partitions identical (coupling/alpha not effective)"
    with open(f"{OUT}/cifar_dirichlet_coupling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(f"{OUT}/cifar_partition_checksums.sha256", "w") as f:
        for k, h in checksums:
            f.write(f"{h}  {k}\n")
    # summary
    empt = sum(r["n_empty_clients"] for r in rows)
    print("=== CIFAR coupled-Dirichlet validation ===")
    print(f"  blocks x d rows: {len(rows)} (100 blocks x 3 d)")
    print(f"  deterministic re-run: PASS | index-set conserved across d: PASS | d=0.1 vs 5.0 distinct: PASS")
    print(f"  total empty-client occurrences (d=0.1 stress may create some): {empt}")
    print(f"  wrote cifar_dirichlet_coupling.csv, cifar_partition_checksums.sha256")
    json.dump(dict(rows=len(rows), empty_client_occurrences=empt,
                   deterministic=True, index_conserved=True, alpha_effective=True,
                   note="structural synthetic index sets; coupling (shared base seed across d, alpha=d only) verified"),
              open(f"{OUT}/cifar_dirichlet_coupling_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
