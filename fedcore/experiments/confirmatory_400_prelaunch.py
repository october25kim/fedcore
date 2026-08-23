"""Confirmatory-400 campaign — deterministic frozen prelaunch objects (NO training here).

Produces, outcome-independently, from versioned SHA-256 semantic keys only:
  * balanced open-set class splits (fail-closed on the balance constraints);
  * the versioned semantic seed registry;
  * the frozen 400-cell training matrix (+ cardinality/pairing gate);
  * the exact-reuse audit (all fresh: no Wide-ResNet-28-10 legacy exists).

Namespace: results/confirmatory_400/. Distinct from the sealed prereg.
"""
from __future__ import annotations
import csv, hashlib, json, os

CAMPAIGN = "confirmatory_400"
SEEDREG = "v1"
OUT = "results/confirmatory_400"
PRE = f"{OUT}/prelaunch"
N_SPLITS = 10
N_REPS = 5
CIFAR_D = [0.1, 0.5, 5.0]

DATASETS = {
    "cifar10":    dict(n_classes=10,  n_known=6,  n_unknown=4,  dsver="cifar10-v1"),
    "cifar100":   dict(n_classes=100, n_known=60, n_unknown=40, dsver="cifar100-v1"),
    "officehome": dict(n_classes=65,  n_known=45, n_unknown=20, dsver="officehome-dedup-v1"),
}
SEED_FAMILIES = [
    "class_split", "training_initialization", "base_data_permutation", "augmentation",
    "minibatch_order", "client_schedule", "Dirichlet_partition", "proposal_fold",
    "certification_reservoir", "primary_audit_draw", "secondary_audit_redraw",
    "traffic_draw", "evaluation",
]


def _u32(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def sha_rank(*parts) -> int:
    return int.from_bytes(hashlib.sha256(("|".join([CAMPAIGN, SEEDREG, *map(str, parts)])).encode()).digest()[:8], "big")


def unknown_targets(dataset: str):
    """Per-class target unknown-count: CIFAR uniform 4; Office-Home 5 classes x4 + 60 x3."""
    cfg = DATASETS[dataset]
    C = cfg["n_classes"]
    total = cfg["n_unknown"] * N_SPLITS
    base = total // C
    extra = total - base * C                      # classes that get one more
    # choose which classes get the +1 by deterministic SHA-256 rank
    order = sorted(range(C), key=lambda c: sha_rank(dataset, "extra_unknown", c))
    tgt = {c: base for c in range(C)}
    for c in order[:extra]:
        tgt[c] += 1
    return tgt


def balanced_splits(dataset: str):
    """Deterministic greedy: draw down highest-remaining-budget classes, minimize pairwise
    co-occurrence with classes already chosen in the split, SHA-256 tie-break."""
    cfg = DATASETS[dataset]
    C, k = cfg["n_classes"], cfg["n_unknown"]
    remaining = unknown_targets(dataset)
    cooc = {c: {} for c in range(C)}              # co-occurrence counts
    splits = []
    for i in range(N_SPLITS):
        chosen: list[int] = []
        for _ in range(k):
            cands = [c for c in range(C) if remaining[c] > 0 and c not in chosen]
            def key(c):
                overlap = sum(cooc[c].get(o, 0) for o in chosen)
                return (-remaining[c], overlap, sha_rank(dataset, "pick", i, c))
            pick = min(cands, key=key)
            chosen.append(pick); remaining[pick] -= 1
        for a in chosen:
            for b in chosen:
                if a != b:
                    cooc[a][b] = cooc[a].get(b, 0) + 1
        unk = sorted(chosen)
        kn = sorted(c for c in range(C) if c not in unk)
        splits.append((kn, unk))
    return splits


def verify_balance(dataset: str, splits):
    cfg = DATASETS[dataset]
    from collections import Counter
    freq = Counter()
    for _, unk in splits:
        assert len(unk) == cfg["n_unknown"], f"{dataset} split has wrong unknown count"
        assert len(set(unk)) == cfg["n_unknown"], "duplicate unknown in a split"
        freq.update(unk)
    tgt = unknown_targets(dataset)
    for c in range(cfg["n_classes"]):
        assert freq[c] == tgt[c], f"{dataset} class {c}: {freq[c]} != target {tgt[c]}"
    counts = sorted(freq.values())
    return dict(dataset=dataset, min_unknown_freq=counts[0], max_unknown_freq=counts[-1],
                freq_spread=counts[-1] - counts[0],
                n_classes_4x=sum(1 for v in freq.values() if v == 4),
                n_classes_3x=sum(1 for v in freq.values() if v == 3),
                balance_ok=True)


def build():
    os.makedirs(PRE, exist_ok=True)
    balance = []
    for ds in DATASETS:
        splits = balanced_splits(ds)
        balance.append(verify_balance(ds, splits))       # raises AssertionError -> fail closed
        rows = []
        for i, (kn, unk) in enumerate(splits):
            sh = hashlib.sha256(json.dumps({"known": kn, "unknown": unk}, sort_keys=True).encode()).hexdigest()
            rows.append(dict(split_id=f"{ds}_split_{i:02d}", n_known=len(kn), n_unknown=len(unk),
                             known_classes=" ".join(map(str, kn)), unknown_classes=" ".join(map(str, unk)),
                             split_sha256=sh))
        with open(f"{PRE}/{ds}_class_splits.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump({"constraints": {
        "cifar10": "every class unknown exactly 4x",
        "cifar100": "every class unknown exactly 4x",
        "officehome": "5 classes 4x + 60 classes 3x, spread<=1"},
        "verified": balance, "all_pass": True},
        open(f"{PRE}/class_split_balance_report.json", "w"), indent=2)

    # ---- seed registry (versioned; audit/traffic seeds independent of outcome params) ----
    registry = {"campaign": CAMPAIGN, "version": SEEDREG, "rule": "sha256(campaign|version|family|cell_id)[:4] -> uint32",
                "families": SEED_FAMILIES,
                "audit_traffic_independence": "class_split/audit/traffic/eval seeds are functions of (family, cell_id) only; NEVER of alpha/delta/rho/score/gamma/threshold/selector/certificate/solver/allocation",
                "paired_sharing": "within a paired block all non-intervention family seeds are shared exactly across arms/d"}
    json.dump(registry, open(f"{PRE}/semantic_seed_registry.json", "w"), indent=2)
    open(f"{PRE}/semantic_seed_registry.sha256", "w").write(
        hashlib.sha256(open(f"{PRE}/semantic_seed_registry.json", "rb").read()).hexdigest() + "  semantic_seed_registry.json\n")

    def seeds(cell):  # non-intervention seed families shared within a paired block
        return {f"seed_{fam}": _u32(f"{CAMPAIGN}|{SEEDREG}|{fam}|{cell}") for fam in SEED_FAMILIES}

    # ---- 400-cell matrix ----
    matrix = []
    for ds in ("cifar10", "cifar100"):
        pipe = f"{ds}_fedpd_proser"
        for i in range(N_SPLITS):
            for rep in range(N_REPS):
                block = f"{ds}__split{i:02d}__seed{rep}"
                for d in CIFAR_D:
                    cell = f"{pipe}__split{i:02d}__seed{rep}__d{d}"
                    # COMMON RANDOM NUMBERS: every family seed (incl. Dirichlet_partition) is
                    # keyed on the BLOCK and SHARED across d. d enters ONLY as the Dirichlet
                    # concentration (alpha) at partition time, so the base ordering and RNG
                    # stream are identical across d and only the concentration differs.
                    s = {f"seed_{fam}": _u32(f"{CAMPAIGN}|{SEEDREG}|{fam}|{block}") for fam in SEED_FAMILIES}
                    matrix.append(dict(semantic_id=cell, dataset=ds, pipeline_id=pipe, arm="fedpd_proser",
                                       backbone="wide_resnet_28_10", input_res="32x32",
                                       split_id=f"{ds}_split_{i:02d}", train_seed=rep, d=d,
                                       paired_block=block, aggregation="fedpd", lpd=True, gdca_ot=True,
                                       reuse_class="fresh_training", **s))
    for arm, pipe, agg, lpd, gdca in [
        ("fedavg_proser", "officehome_fedavg_proser", "fedavg", False, False),
        ("fedpd_proser",  "officehome_fedpd_proser",  "fedpd",  True,  True),
    ]:
        for i in range(N_SPLITS):
            for rep in range(N_REPS):
                block = f"officehome__split{i:02d}__seed{rep}"
                cell = f"{pipe}__split{i:02d}__seed{rep}"
                matrix.append(dict(semantic_id=cell, dataset="officehome", pipeline_id=pipe, arm=arm,
                                   backbone="wide_resnet_28_10", input_res="32x32",
                                   split_id=f"officehome_split_{i:02d}", train_seed=rep, d="",
                                   paired_block=block, aggregation=agg, lpd=lpd, gdca_ot=gdca,
                                   reuse_class="fresh_training", **seeds(block)))
    fields = list(matrix[0].keys())
    with open(f"{OUT}/final_training_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(matrix)
    mh = hashlib.sha256(open(f"{OUT}/final_training_matrix.csv", "rb").read()).hexdigest()
    open(f"{OUT}/final_training_matrix.sha256", "w").write(f"{mh}  final_training_matrix.csv\n")

    # ---- exact-reuse audit: legacy backbones are ResNet-GN (CIFAR) / ConvNeXt-Tiny (Office-Home),
    #      never Wide-ResNet-28-10 -> every row is fresh_training. ----
    with open(f"{PRE}/exact_reuse_audit.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["semantic_id", "reuse_class", "reason"])
        for r in matrix:
            w.writerow([r["semantic_id"], "fresh_training",
                        "no legacy artifact shares (backbone=wide_resnet_28_10, campaign=confirmatory_400, seedreg=v1); prior CIFAR=ResNet-GN, prior Office-Home=ConvNeXt-Tiny@224"])

    # ---- cardinality / pairing gate ----
    ids = [r["semantic_id"] for r in matrix]
    def blocks(ds, agg=None):
        b = {}
        for r in matrix:
            if r["dataset"] != ds:
                continue
            if agg and r["aggregation"] != agg:
                pass
            b.setdefault(r["paired_block"], []).append(r)
        return b
    oh = [r for r in matrix if r["dataset"] == "officehome"]
    oh_blocks = {}
    for r in oh:
        oh_blocks.setdefault(r["paired_block"], set()).add(r["aggregation"])
    c10_blocks = {}
    for r in matrix:
        if r["dataset"] == "cifar10":
            c10_blocks.setdefault(r["paired_block"], set()).add(r["d"])
    c100_blocks = {}
    for r in matrix:
        if r["dataset"] == "cifar100":
            c100_blocks.setdefault(r["paired_block"], set()).add(r["d"])
    card = dict(
        total_rows=len(matrix), duplicate_semantic_ids=len(ids) - len(set(ids)),
        cifar10_rows=sum(1 for r in matrix if r["dataset"] == "cifar10"),
        cifar100_rows=sum(1 for r in matrix if r["dataset"] == "cifar100"),
        officehome_rows=len(oh),
        officehome_fedavg_rows=sum(1 for r in oh if r["aggregation"] == "fedavg"),
        officehome_fedpd_rows=sum(1 for r in oh if r["aggregation"] == "fedpd"),
        officehome_complete_paired_blocks=sum(1 for v in oh_blocks.values() if v == {"fedavg", "fedpd"}),
        officehome_unpaired_blocks=sum(1 for v in oh_blocks.values() if v != {"fedavg", "fedpd"}),
        cifar10_complete_3d_blocks=sum(1 for v in c10_blocks.values() if len(v) == 3),
        cifar100_complete_3d_blocks=sum(1 for v in c100_blocks.values() if len(v) == 3),
        all_fresh_training=all(r["reuse_class"] == "fresh_training" for r in matrix),
        matrix_sha256=mh,
    )
    expect = dict(total_rows=400, duplicate_semantic_ids=0, cifar10_rows=150, cifar100_rows=150,
                  officehome_rows=100, officehome_fedavg_rows=50, officehome_fedpd_rows=50,
                  officehome_complete_paired_blocks=50, officehome_unpaired_blocks=0,
                  cifar10_complete_3d_blocks=50, cifar100_complete_3d_blocks=50)
    card["matches_gate_expectation"] = all(card[k] == v for k, v in expect.items())
    json.dump(card, open(f"{PRE}/matrix_cardinality.json", "w"), indent=2)

    print("=== confirmatory_400 deterministic prelaunch objects ===")
    for b in balance:
        print(f"  balance {b['dataset']}: 4x={b['n_classes_4x']} 3x={b['n_classes_3x']} spread={b['freq_spread']} ok={b['balance_ok']}")
    for k, v in card.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    build()
