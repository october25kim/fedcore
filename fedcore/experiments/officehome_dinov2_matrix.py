"""Office-Home DINOv2 ViT-L/14 frozen-head arm matrix (50 cells; owner spec).

One arm, 10 class splits x 5 training reps = 50 cells, reusing the SAME frozen
Confirmatory-400R Office-Home splits + block seeds (CRN) as the existing ConvNeXt
full/frozen arms, so the DINOv2 arm is paired with the existing ConvNeXt results.
Recipe: run_officehome --pipeline C (dinov2_vitl14_frozen_linear) in the
fedcore-c400r-dino image; 224-native ImageNet transform; only the linear head is
federated. NOT launched here -- prelaunch matrix only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

from fedcore.experiments.confirmatory_400_prelaunch import (
    _u32, CAMPAIGN as C400, SEEDREG, SEED_FAMILIES, N_SPLITS, N_REPS, DATASETS,
)

CAMPAIGN = "officehome_dinov2"
OUT = "results/officehome_dinov2"
N_KNOWN = DATASETS["officehome"]["n_known"]      # 45
N_UNKNOWN = DATASETS["officehome"]["n_unknown"]  # 20


def seeds(block):
    return {f"seed_{fam}": _u32(f"{C400}|{SEEDREG}|{fam}|{block}") for fam in SEED_FAMILIES}


def build():
    os.makedirs(f"{OUT}/prelaunch", exist_ok=True)
    rows = []
    for i in range(N_SPLITS):
        for rep in range(N_REPS):
            block = f"officehome__split{i:02d}__seed{rep}"
            cell = f"officehome_dinov2_frozen__split{i:02d}__seed{rep}"
            rows.append(dict(
                semantic_id=cell, dataset="officehome",
                pipeline_id="OFFICEHOME_DINOV2_VITL14_FROZEN_LINEAR",
                pipeline="C", backbone="dinov2_vitl14", input_res="224x224",
                arm="dinov2_frozen_linear", aggregation="fedavg",
                frozen_encoder=True, n_known=N_KNOWN, n_unknown=N_UNKNOWN,
                split_id=f"officehome_split_{i:02d}", train_rep=rep,
                paired_block=block, reuse_class="fresh_training", **seeds(block)))

    fields = list(rows[0].keys())
    with open(f"{OUT}/final_training_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    mh = hashlib.sha256(open(f"{OUT}/final_training_matrix.csv", "rb").read()).hexdigest()
    open(f"{OUT}/final_training_matrix.sha256", "w").write(f"{mh}  final_training_matrix.csv\n")

    ids = [r["semantic_id"] for r in rows]
    blocks = {r["paired_block"] for r in rows}
    card = dict(total_rows=len(rows), duplicate_semantic_ids=len(ids) - len(set(ids)),
                unique_blocks=len(blocks), n_known=N_KNOWN, n_unknown=N_UNKNOWN,
                matrix_sha256=mh)
    expect = dict(total_rows=50, duplicate_semantic_ids=0, unique_blocks=50)
    card["matches_gate_expectation"] = all(card[k] == v for k, v in expect.items())
    json.dump(card, open(f"{OUT}/prelaunch/matrix_cardinality.json", "w"), indent=2)
    for k, v in card.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    build()
