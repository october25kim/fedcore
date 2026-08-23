"""Medical ResNeXt-FedProx matrix: PathMNIST + TissueMNIST, ResNeXt-29 8x64d FedProx mu=0.1.

100 cells (pathmnist 50 + tissuemnist 50) forming the medical half of the ResNeXt-FedProx
methodology headline (ResNeXt-FedProx - ResNeXt-plain). Each cell REUSES the exact seed
formula of its source plain campaign (medmnist_sweep for pathmnist, tissuemnist_sweep for
tissuemnist) so it is CRN-paired to that dataset's ResNeXt-plain cell (same split/rep/d).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

from fedcore.experiments.medmnist_sweep_matrix import (
    CLASS_SPLITS as PATH_SPLITS, N_KNOWN as PATH_NK, N_UNKNOWN as PATH_NU,
    D_VALUES, N_REPS, N_CLIENTS, _u32,
)
from fedcore.experiments.tissuemnist_sweep_matrix import (
    CLASS_SPLITS as TISS_SPLITS, N_KNOWN as TISS_NK, N_UNKNOWN as TISS_NU,
)

OUT = "results/medmnist_resnext_fedprox_sweep"
BACKBONE = "resnext29_8x64d"
MU = 0.1
ARM = f"resnext29_8x64d_fedprox_mu{MU:g}"

# (dataset, source_campaign_for_seed_pairing, class_splits, n_known, n_unknown)
DATASETS = [
    ("pathmnist", "medmnist_sweep", PATH_SPLITS, PATH_NK, PATH_NU),
    ("tissuemnist", "tissuemnist_sweep", TISS_SPLITS, TISS_NK, TISS_NU),
]


def build():
    os.makedirs(f"{OUT}/prelaunch", exist_ok=True)
    rows = []
    for dataset, src_camp, splits, nk, nu in DATASETS:
        for split_id, unknown in splits.items():
            for rep in range(N_REPS):
                block = f"{split_id}__seed{rep}"           # SAME block as source plain campaign
                for d in D_VALUES:
                    cell = f"{ARM}__{dataset}__{split_id}__seed{rep}__d{d}"
                    rows.append(dict(
                        semantic_id=cell, dataset=dataset, backbone=BACKBONE,
                        pipeline_id=f"MEDMNIST_{dataset.upper()}_RESNEXT29_FEDPROX",
                        input_res="28x28", aggregation="fedavg_fedprox", fedprox_mu=MU,
                        proser=False, n_known=nk, n_unknown=nu, n_clients=N_CLIENTS,
                        split_id=split_id, unknown_classes=",".join(map(str, unknown)),
                        train_rep=rep, d=d, paired_block=block, reuse_class="fresh_training",
                        paired_plain=f"resnext29_8x64d__{split_id}__seed{rep}__d{d}",
                        seed_class_split=_u32(f"{src_camp}|class_split|{block}"),
                        seed_partition=_u32(f"{src_camp}|partition|{block}"),
                        seed_audit=_u32(f"{src_camp}|audit|{block}")))
    fields = list(rows[0].keys())
    with open(f"{OUT}/final_training_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    mh = hashlib.sha256(open(f"{OUT}/final_training_matrix.csv", "rb").read()).hexdigest()
    open(f"{OUT}/final_training_matrix.sha256", "w").write(f"{mh}  final_training_matrix.csv\n")
    ids = [r["semantic_id"] for r in rows]
    card = dict(total_rows=len(rows), duplicate_semantic_ids=len(ids) - len(set(ids)),
                backbone=BACKBONE, fedprox_mu=MU,
                pathmnist=sum(1 for r in rows if r["dataset"] == "pathmnist"),
                tissuemnist=sum(1 for r in rows if r["dataset"] == "tissuemnist"),
                matrix_sha256=mh)
    card["matches_gate_expectation"] = (card["total_rows"] == 100 and card["duplicate_semantic_ids"] == 0
                                        and card["pathmnist"] == 50 and card["tissuemnist"] == 50)
    json.dump(card, open(f"{OUT}/prelaunch/matrix_cardinality.json", "w"), indent=2)
    for k, v in card.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    build()
