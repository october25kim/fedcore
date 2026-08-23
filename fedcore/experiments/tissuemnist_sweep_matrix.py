"""TissueMNIST backbone-sweep matrix: 100 cells (WRN-28-10 + ResNeXt-29).

Second MedMNIST arm (owner: run TissueMNIST too if PathMNIST is good -- it was:
20/24 full-simplex certified). TissueMNIST is the LARGEST MedMNIST (train 165k /
test 47k, 8 classes, 28x28 GRAYSCALE, kidney cortex tissue) -> even larger per-
client reservoirs. Design mirrors the PathMNIST sweep: 5 class-splits x 5 reps x
d{0.5, 5.0} x {WRN-28-10, ResNeXt-29} plain FedAvg. n_known=5 / n_unknown=3.
Grayscale is repeated to 3 channels by the runner loader.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

CAMPAIGN = "tissuemnist_sweep"
OUT = "results/tissuemnist_sweep"
DATASET = "tissuemnist"
N_KNOWN, N_UNKNOWN = 5, 3
N_CLIENTS = 5
D_VALUES = [0.5, 5.0]
N_REPS = 5
CLASS_SPLITS = {
    "tissuemnist_split_00": [5, 6, 7],
    "tissuemnist_split_01": [0, 1, 2],
    "tissuemnist_split_02": [2, 3, 4],
    "tissuemnist_split_03": [1, 4, 7],
    "tissuemnist_split_04": [0, 3, 6],
}
BACKBONES = ["resnext29_8x64d"]   # medical = ResNeXt-plain ONLY (owner: WRN/FedProx are CIFAR-only)


def _u32(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def build():
    os.makedirs(f"{OUT}/prelaunch", exist_ok=True)
    rows = []
    for bb in BACKBONES:
        for split_id, unknown in CLASS_SPLITS.items():
            for rep in range(N_REPS):
                for d in D_VALUES:
                    cell = f"{bb}__{split_id}__seed{rep}__d{d}"
                    block = f"{split_id}__seed{rep}"
                    rows.append(dict(
                        semantic_id=cell, dataset=DATASET, backbone=bb,
                        pipeline_id=f"TISSUEMNIST_{bb.upper()}_PLAIN_FEDAVG",
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
                matrix_sha256=mh)
    card["matches_gate_expectation"] = (card["total_rows"] == 50 and card["duplicate_semantic_ids"] == 0
                                        and card["wrn28_10"] == 0 and card["resnext29_8x64d"] == 50)
    json.dump(card, open(f"{OUT}/prelaunch/matrix_cardinality.json", "w"), indent=2)
    for k, v in card.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    build()
