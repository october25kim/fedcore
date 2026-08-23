"""CIFAR FedProx methodology-arm matrix (owner-approved 2026-08-11).

The on-thesis METHODOLOGY axis for Fed-CORE: a faithful heterogeneity-robust FL
optimizer (FedProx, Li et al. MLSys 2020 -- proximal (mu/2)||w - w_t||^2) paired
against plain FedAvg at MATCHED backbone/split/seed/d, to test whether a method
that curbs client drift under non-IID raises CertifiedCoverage@alpha.

Not "FedPD++": the ICCV'23 FedPD parameter-disentanglement code is unreleased in
every pinned upstream (only a PROSER FedAvg baseline + an OT neuron-alignment
fusion + dead visualization masks; WRN open-set head is broken as released), so a
faithful FedPD++ arm is impossible without core-method re-authoring (the exact
provenance risk the owner closed for Office-Home, DECISION_O4). FedProx is a clean,
architecture-agnostic, GPL-free, well-specified alternative that realizes the SAME
scientific question.

Design = an EXACT mirror of backbone-sweep Arm B (resnext29_8x64d_plain_fedavg): same
frozen confirmatory-400R block seeds (common random numbers => the FedProx cell and
its Arm-B twin share splits/partitions/audit folds and differ ONLY in the proximal
term), WRN-28-10 native 32x32, cifar10 (n_known=6) + cifar100 (n_known=60),
10 splits x 5 reps x d{0.1,0.5,5.0} = 300 cells. Single mu=0.1 (moderate
heterogeneity); d=5.0 is the near-IID null control (FedProx ~ FedAvg there).
Certified through the identical common-schema post-hoc path as every other arm.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

from fedcore.experiments.confirmatory_400_prelaunch import (
    _u32, CAMPAIGN as C400, SEEDREG, SEED_FAMILIES, N_SPLITS, N_REPS, CIFAR_D,
)

CAMPAIGN = "cifar_resnext_fedprox_sweep"
OUT = "results/cifar_resnext_fedprox_sweep"
DATASETS = {"cifar10": 6, "cifar100": 60}
BACKBONE = "resnext29_8x64d"
FEDPROX_MU = 0.1
ARM = f"resnext29_8x64d_fedprox_mu{FEDPROX_MU:g}"
PIPELINE = "CIFAR_RESNEXT29_8X64D_FEDPROX"


def seeds(block):
    # SAME confirmatory-400R seed definitions as Arm B (CRN) => exact pairing.
    return {f"seed_{fam}": _u32(f"{C400}|{SEEDREG}|{fam}|{block}") for fam in SEED_FAMILIES}


def build():
    os.makedirs(f"{OUT}/prelaunch", exist_ok=True)
    rows = []
    for ds, n_known in DATASETS.items():
        for i in range(N_SPLITS):
            for rep in range(N_REPS):
                block = f"{ds}__split{i:02d}__seed{rep}"
                for d in CIFAR_D:
                    cell = f"{ARM}__{ds}__split{i:02d}__seed{rep}__d{d}"
                    rows.append(dict(
                        semantic_id=cell, arm=ARM, dataset=ds,
                        pipeline_id=PIPELINE, backbone=BACKBONE,
                        input_res="32x32", aggregation="fedavg_fedprox",
                        fedprox_mu=FEDPROX_MU, proser=False, lpd=False, gdca_ot=False,
                        n_known=n_known, n_clients=5,
                        split_id=f"{ds}_split_{i:02d}", train_rep=rep, d=d,
                        paired_block=block, reuse_class="fresh_training",
                        paired_arm_c=f"resnext29_8x64d_plain_fedavg__{ds}__split{i:02d}__seed{rep}__d{d}",
                        **seeds(block)))
    fields = list(rows[0].keys())
    with open(f"{OUT}/final_training_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    mh = hashlib.sha256(open(f"{OUT}/final_training_matrix.csv", "rb").read()).hexdigest()
    open(f"{OUT}/final_training_matrix.sha256", "w").write(f"{mh}  final_training_matrix.csv\n")

    ids = [r["semantic_id"] for r in rows]
    card = dict(
        total_rows=len(rows), duplicate_semantic_ids=len(ids) - len(set(ids)),
        backbone=BACKBONE, fedprox_mu=FEDPROX_MU,
        cifar10=sum(1 for r in rows if r["dataset"] == "cifar10"),
        cifar100=sum(1 for r in rows if r["dataset"] == "cifar100"),
        d_values=CIFAR_D, n_splits=N_SPLITS, n_reps=N_REPS, matrix_sha256=mh)
    expect = dict(total_rows=300, duplicate_semantic_ids=0, cifar10=150, cifar100=150)
    card["matches_gate_expectation"] = all(card[k] == v for k, v in expect.items())
    json.dump(card, open(f"{OUT}/prelaunch/matrix_cardinality.json", "w"), indent=2)
    for k, v in card.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    build()
