"""MedMNIST (PathMNIST) backbone-sweep matrix: 100 cells (WRN-28-10 + ResNeXt-29).

Owner spec: two strong CIFAR-native backbones x 50 cells each = 100. Design:
5 class-splits (distinct held-out unknown-class triples over 9 PathMNIST classes)
x 5 training reps x d in {0.5, 5.0} = 50 cells per backbone. Native 28x28, plain
size-weighted FedAvg, n_known=6 / n_unknown=3, 5 Dirichlet clients. Certified
post-hoc via the common per-obs schema (MSP/energy/margin family, client full
simplex + grouped + pooled), the SAME machinery as the CIFAR sweep.

The confirmation showed PathMNIST clears the distribution-free full-simplex at
alpha=0.20 (which Fed-ISIC never did): large per-client reservoirs (~287) satisfy
the Theorem-2 feasibility floor, and a well-trained model reaches AUROC ~0.85.
d=5.0 (near-IID) certified U=0.129; d=0.5 tests moderate non-IID.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

CAMPAIGN = "medmnist_sweep"
OUT = "results/medmnist_sweep"
DATASET = "pathmnist"
N_KNOWN, N_UNKNOWN = 6, 3
N_CLIENTS = 5
D_VALUES = [0.5, 5.0]
N_REPS = 5

# 5 class-splits: distinct held-out unknown triples over the 9 PathMNIST classes,
# each class held out 1-2 times across splits (balanced coverage).
CLASS_SPLITS = {
    "pathmnist_split_00": [6, 7, 8],
    "pathmnist_split_01": [0, 1, 2],
    "pathmnist_split_02": [3, 4, 5],
    "pathmnist_split_03": [1, 4, 7],
    "pathmnist_split_04": [2, 5, 8],
}
BACKBONES = ["wrn28_10", "resnext29_8x64d"]


def _u32(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def build():
    os.makedirs(f"{OUT}/prelaunch", exist_ok=True)
    rows = []
    for bb in BACKBONES:
        for split_id, unknown in CLASS_SPLITS.items():
            i = int(split_id.split("_")[-1])
            for rep in range(N_REPS):
                for d in D_VALUES:
                    cell = f"{bb}__{split_id}__seed{rep}__d{d}"
                    block = f"{split_id}__seed{rep}"
                    rows.append(dict(
                        semantic_id=cell, dataset=DATASET, backbone=bb,
                        pipeline_id=f"MEDMNIST_{bb.upper()}_PLAIN_FEDAVG",
                        input_res="28x28", aggregation="fedavg", proser=False,
                        n_known=N_KNOWN, n_unknown=N_UNKNOWN, n_clients=N_CLIENTS,
                        split_id=split_id, unknown_classes=",".join(map(str, unknown)),
                        train_rep=rep, d=d, paired_block=block, reuse_class="fresh_training",
                        seed_class_split=_u32(f"{CAMPAIGN}|class_split|{block}"),
                        seed_partition=_u32(f"{CAMPAIGN}|partition|{block}"),
                        seed_audit=_u32(f"{CAMPAIGN}|audit|{block}")))
    fields = list(rows[0].keys())
    with open(f"{OUT}/final_training_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    mh = hashlib.sha256(open(f"{OUT}/final_training_matrix.csv", "rb").read()).hexdigest()
    open(f"{OUT}/final_training_matrix.sha256", "w").write(f"{mh}  final_training_matrix.csv\n")

    ids = [r["semantic_id"] for r in rows]
    card = dict(total_rows=len(rows), duplicate_semantic_ids=len(ids) - len(set(ids)),
                wrn28_10=sum(1 for r in rows if r["backbone"] == "wrn28_10"),
                resnext29_8x64d=sum(1 for r in rows if r["backbone"] == "resnext29_8x64d"),
                class_splits=len(CLASS_SPLITS), d_values=D_VALUES, n_reps=N_REPS,
                matrix_sha256=mh)
    expect = dict(total_rows=100, duplicate_semantic_ids=0, wrn28_10=50, resnext29_8x64d=50,
                  class_splits=5, n_reps=5)
    card["matches_gate_expectation"] = all(card[k] == v for k, v in expect.items())
    json.dump(card, open(f"{OUT}/prelaunch/matrix_cardinality.json", "w"), indent=2)
    for k, v in card.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    build()
