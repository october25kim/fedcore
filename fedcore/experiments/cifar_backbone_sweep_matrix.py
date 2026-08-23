"""Revised CIFAR backbone-sweep matrix (native 32x32; owner spec 2026-08-06).

Three arms over the SAME frozen confirmatory-400R CIFAR splits + block seeds
(common random numbers => paired), for cifar10 (n_known=6) and cifar100
(n_known=60), full design 10 splits x 5 reps x d{0.1,0.5,5.0}:

  A. wrn28_10_proser_fedavg      -- EXACT-REUSE reference to the already-trained
                                     Confirmatory-400R WRN-28-10 PROSER-FedAvg cells
                                     (reuse_class=exact_reuse; ref_semantic_id points
                                     at the existing cell; NOT retrained).
  B. wrn28_10_plain_fedavg       -- 300 FRESH cells (plain size-weighted FedAvg, CE).
  C. resnext29_8x64d_plain_fedavg-- 300 FRESH cells (plain size-weighted FedAvg, CE).

All native 32x32; no 224-resize / ConvNeXt / DINOv2 (that pretrained-CIFAR design
is frozen SUPERSEDED_BEFORE_LAUNCH). Recertification (separate step) runs all three
arms on ONE fresh audit stream with the common selector family MSP/energy/margin x
gamma{0.3,0.5,0.7,1.0}, client full simplex, alpha=0.20, Holm/IUT primary; the
PROSER dummy-vs-known score is an arm-A secondary diagnostic only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

from fedcore.experiments.confirmatory_400_prelaunch import (
    _u32, CAMPAIGN as C400, SEEDREG, SEED_FAMILIES, N_SPLITS, N_REPS, CIFAR_D,
)

CAMPAIGN = "cifar_backbone_sweep"
OUT = "results/cifar_backbone_sweep"

DATASETS = {"cifar10": 6, "cifar100": 60}

ARMS = [
    {"arm": "wrn28_10_proser_fedavg", "backbone": "wide_resnet_28_10",
     "pipeline_id": "CIFAR_WRN28_10_PROSER_FEDAVG",
     "reuse_class": "exact_reuse", "proser": True,
     "ref_campaign": "confirmatory_400r", "ref_pipe_suffix": "proser_fedavg"},
    {"arm": "wrn28_10_plain_fedavg", "backbone": "wrn28_10",
     "pipeline_id": "CIFAR_WRN28_10_PLAIN_FEDAVG",
     "reuse_class": "fresh_training", "proser": False},
    {"arm": "resnext29_8x64d_plain_fedavg", "backbone": "resnext29_8x64d",
     "pipeline_id": "CIFAR_RESNEXT29_8X64D_PLAIN_FEDAVG",
     "reuse_class": "fresh_training", "proser": False},
]


def seeds(block):
    # Reuse the SAME confirmatory-400R seed definitions (CRN): B/C share A's splits.
    return {f"seed_{fam}": _u32(f"{C400}|{SEEDREG}|{fam}|{block}") for fam in SEED_FAMILIES}


def build():
    os.makedirs(f"{OUT}/prelaunch", exist_ok=True)
    rows = []
    for spec in ARMS:
        arm = spec["arm"]
        for ds, n_known in DATASETS.items():
            for i in range(N_SPLITS):
                for rep in range(N_REPS):
                    block = f"{ds}__split{i:02d}__seed{rep}"
                    for d in CIFAR_D:
                        cell = f"{arm}__{ds}__split{i:02d}__seed{rep}__d{d}"
                        ref = ""
                        if spec["reuse_class"] == "exact_reuse":
                            ref = f"{ds}_{spec['ref_pipe_suffix']}__split{i:02d}__seed{rep}__d{d}"
                        rows.append(dict(
                            semantic_id=cell, arm=arm, dataset=ds,
                            pipeline_id=spec["pipeline_id"], backbone=spec["backbone"],
                            input_res="32x32", aggregation="fedavg",
                            proser=spec["proser"], lpd=False, gdca_ot=False,
                            n_known=n_known, n_clients=5,
                            split_id=f"{ds}_split_{i:02d}", train_rep=rep, d=d,
                            paired_block=block, reuse_class=spec["reuse_class"],
                            ref_campaign=spec.get("ref_campaign", ""),
                            ref_semantic_id=ref, **seeds(block)))

    fields = list(rows[0].keys())
    with open(f"{OUT}/final_training_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    mh = hashlib.sha256(open(f"{OUT}/final_training_matrix.csv", "rb").read()).hexdigest()
    open(f"{OUT}/final_training_matrix.sha256", "w").write(f"{mh}  final_training_matrix.csv\n")

    # cardinality gate
    def n(**kw):
        return sum(1 for r in rows if all(str(r[k]) == str(v) for k, v in kw.items()))
    ids = [r["semantic_id"] for r in rows]
    fresh_blocks = {}
    for r in rows:
        if r["reuse_class"] == "fresh_training":
            fresh_blocks.setdefault((r["arm"], r["dataset"], r["paired_block"]), set()).add(r["d"])
    card = dict(
        total_rows=len(rows), duplicate_semantic_ids=len(ids) - len(set(ids)),
        arm_A_exact_reuse=n(reuse_class="exact_reuse"),
        arm_B_wrn_plain=n(arm="wrn28_10_plain_fedavg"),
        arm_C_resnext_plain=n(arm="resnext29_8x64d_plain_fedavg"),
        cifar10=n(dataset="cifar10"), cifar100=n(dataset="cifar100"),
        fresh_cells=n(reuse_class="fresh_training"),
        complete_3d_fresh_blocks=sum(1 for v in fresh_blocks.values() if len(v) == 3),
        matrix_sha256=mh,
    )
    expect = dict(total_rows=900, duplicate_semantic_ids=0, arm_A_exact_reuse=300,
                  arm_B_wrn_plain=300, arm_C_resnext_plain=300, cifar10=450, cifar100=450,
                  fresh_cells=600, complete_3d_fresh_blocks=200)
    card["matches_gate_expectation"] = all(card[k] == v for k, v in expect.items())
    json.dump(card, open(f"{OUT}/prelaunch/matrix_cardinality.json", "w"), indent=2)
    for k, v in card.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    build()
