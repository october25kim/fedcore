"""Confirmatory v3 campaign — deterministic frozen 400-cell matrix + class splits.

Prospectively frozen, OUTCOME-INDEPENDENT. New namespace (results/confirmatory_v2),
distinct from the original sealed preregistration. All seeds derive from a versioned
SHA-256 semantic key over a 'confirmatory_v2' campaign tag — never from outcomes.
"""
from __future__ import annotations
import csv, hashlib, json, os

CAMPAIGN = "confirmatory_v2"
SEEDREG = "v1"
OUT = "results/confirmatory_v2"
DATASETS = {
    "cifar10":   dict(n_classes=10,  n_known=6,  n_unknown=4,  dsver="cifar10-v1"),
    "cifar100":  dict(n_classes=100, n_known=60, n_unknown=40, dsver="cifar100-v1"),
    "officehome":dict(n_classes=65,  n_known=45, n_unknown=20, dsver="officehome-dedup-v1"),
}
N_SPLITS = 10
N_REPS = 5
CIFAR_D = [0.1, 0.5, 5.0]
SEED_PURPOSES = ["split", "fold", "partition", "train", "loader", "audit", "traffic", "noise", "solver"]


def h32(*parts: str) -> int:
    """Deterministic uint32 from a versioned semantic key (sha256, not outcome-dependent)."""
    key = "|".join([CAMPAIGN, SEEDREG] + list(parts))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def class_split(dataset: str, split_i: int):
    """Known/unknown class ids from a versioned SHA-256 key only (no difficulty/outcome)."""
    cfg = DATASETS[dataset]
    key = f"{cfg['dsver']}|split_{split_i:02d}|seedreg_{SEEDREG}"
    seed = int.from_bytes(hashlib.sha256(("confirmatory_v2|classsplit|" + key).encode()).digest()[:8], "big")
    import random
    rng = random.Random(seed)
    perm = list(range(cfg["n_classes"]))
    rng.shuffle(perm)
    known = sorted(perm[: cfg["n_known"]])
    unknown = sorted(perm[cfg["n_known"]:])
    split_hash = hashlib.sha256(json.dumps({"known": known, "unknown": unknown}, sort_keys=True).encode()).hexdigest()
    return known, unknown, split_hash


def build():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(f"{OUT}/prelaunch", exist_ok=True)
    # ---- frozen class splits ----
    splits = []
    for ds in DATASETS:
        for i in range(N_SPLITS):
            known, unknown, sh = class_split(ds, i)
            splits.append(dict(dataset=ds, split_id=f"{ds}_split_{i:02d}",
                               n_known=len(known), n_unknown=len(unknown),
                               known_classes=" ".join(map(str, known)),
                               unknown_classes=" ".join(map(str, unknown)),
                               split_sha256=sh))
    with open(f"{OUT}/class_splits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(splits[0].keys())); w.writeheader(); w.writerows(splits)

    # ---- 400-cell matrix ----
    rows = []
    def seeds_for(cell_id: str):
        return {f"seed_{p}": h32(p, cell_id) for p in SEED_PURPOSES}

    # CIFAR-10 / CIFAR-100: 10 splits x 5 reps x 3 d = 150 each (FedPD-PROSER only)
    for ds in ("cifar10", "cifar100"):
        pipe = f"{ds}_fedpd_proser"
        for i in range(N_SPLITS):
            for rep in range(N_REPS):
                for d in CIFAR_D:
                    cell = f"{pipe}__{ds}_split_{i:02d}__rep{rep}__d{d}"
                    rows.append(dict(semantic_id=cell, dataset=ds, pipeline_id=pipe, arm="fedpd_proser",
                                     backbone="wide_resnet_28_10", input_res="32x32",
                                     split_id=f"{ds}_split_{i:02d}", train_rep=rep, d=d,
                                     paired_block=f"{ds}_split_{i:02d}__rep{rep}",
                                     aggregation="fedpd", lpd=True, gdca_ot=True,
                                     reuse_class="fresh_training", **seeds_for(cell)))
    # Office-Home: 10 splits x 5 reps x 2 arms = 100 (matched FedAvg + FedPD)
    for arm, pipe, agg, lpd, gdca in [
        ("fedavg_proser", "officehome_proser_fedavg", "fedavg", False, False),
        ("fedpd_proser",  "officehome_proser_fedpd",  "fedpd",  True,  True),
    ]:
        for i in range(N_SPLITS):
            for rep in range(N_REPS):
                cell = f"{pipe}__officehome_split_{i:02d}__rep{rep}"
                rows.append(dict(semantic_id=cell, dataset="officehome", pipeline_id=pipe, arm=arm,
                                 backbone="wide_resnet_28_10", input_res="32x32",
                                 split_id=f"officehome_split_{i:02d}", train_rep=rep, d="",
                                 paired_block=f"officehome_split_{i:02d}__rep{rep}",
                                 aggregation=agg, lpd=lpd, gdca_ot=gdca,
                                 reuse_class="fresh_training", **seeds_for(cell)))

    fields = list(rows[0].keys())
    with open(f"{OUT}/final_training_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    # ---- cardinality / pairing / dup checks ----
    ids = [r["semantic_id"] for r in rows]
    n_dup = len(ids) - len(set(ids))
    oh = [r for r in rows if r["dataset"] == "officehome"]
    oh_fedavg = [r for r in oh if r["aggregation"] == "fedavg"]
    oh_fedpd = [r for r in oh if r["aggregation"] == "fedpd"]
    oh_blocks = {}
    for r in oh:
        oh_blocks.setdefault(r["paired_block"], set()).add(r["aggregation"])
    complete_pairs = sum(1 for v in oh_blocks.values() if v == {"fedavg", "fedpd"})
    unpaired = sum(1 for v in oh_blocks.values() if v != {"fedavg", "fedpd"})
    checks = dict(
        total_rows=len(rows), unique_ids=len(set(ids)), duplicate_ids=n_dup,
        cifar10_rows=sum(1 for r in rows if r["dataset"] == "cifar10"),
        cifar100_rows=sum(1 for r in rows if r["dataset"] == "cifar100"),
        officehome_rows=len(oh),
        officehome_proser_fedavg_rows=len(oh_fedavg),
        officehome_proser_fedpd_rows=len(oh_fedpd),
        officehome_complete_paired_blocks=complete_pairs,
        officehome_unpaired_blocks=unpaired,
        all_fresh_training=all(r["reuse_class"] == "fresh_training" for r in rows),
        exact_reuse_count=sum(1 for r in rows if r["reuse_class"] == "exact_reuse"),
    )
    gate_expect = dict(total_rows=400, duplicate_ids=0, cifar10_rows=150, cifar100_rows=150,
                       officehome_rows=100, officehome_proser_fedavg_rows=50,
                       officehome_proser_fedpd_rows=50, officehome_complete_paired_blocks=50,
                       officehome_unpaired_blocks=0)
    checks["matches_gate_expectation"] = all(checks[k] == v for k, v in gate_expect.items())
    json.dump(checks, open(f"{OUT}/prelaunch/matrix_cardinality.json", "w"), indent=2)

    # matrix sha256
    mh = hashlib.sha256(open(f"{OUT}/final_training_matrix.csv", "rb").read()).hexdigest()
    print("=== confirmatory_v2 frozen matrix ===")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"  matrix sha256: {mh[:24]}")
    print(f"  class splits: {len(splits)} ({N_SPLITS}/dataset x {len(DATASETS)})")


if __name__ == "__main__":
    build()
