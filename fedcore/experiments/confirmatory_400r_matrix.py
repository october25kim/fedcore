"""Confirmatory-400R frozen matrix — corrected pipelines (NO FedPD/GDCA/OT anywhere).

Reuses the frozen balanced class splits + semantic seed hierarchy from confirmatory_400
(scientific definitions unchanged) with CORRECTED pipelines:
  * CIFAR-10/100: WRN28_10_PROSER_FEDAVG (PROSER local + plain weighted FedAvg) x d.
  * Office-Home:  ConvNeXt-Tiny @224 FedAvg, Arm A full fine-tune vs Arm B frozen-linear.
No LPD, no GDCA/OT, no FedPD aggregation, no DigitModel -> the core-mechanism blocker is gone.
"""
from __future__ import annotations
import csv, hashlib, json, os
from fedcore.experiments.confirmatory_400_prelaunch import _u32, CAMPAIGN as C400, SEEDREG, SEED_FAMILIES, N_SPLITS, N_REPS, CIFAR_D

CAMPAIGN = "confirmatory_400r"
OUT = "results/confirmatory_400r"


def seeds(block):
    # reuse the SAME scientific seed definitions as confirmatory_400 (unchanged), keyed on block
    return {f"seed_{fam}": _u32(f"{C400}|{SEEDREG}|{fam}|{block}") for fam in SEED_FAMILIES}


def build():
    os.makedirs(f"{OUT}/prelaunch", exist_ok=True)
    m = []
    # CIFAR-10 / CIFAR-100 PROSER-FedAvg (d couples ONLY Dirichlet concentration; seeds block-keyed = CRN)
    for ds in ("cifar10", "cifar100"):
        pipe = f"{ds}_proser_fedavg"
        for i in range(N_SPLITS):
            for rep in range(N_REPS):
                block = f"{ds}__split{i:02d}__seed{rep}"
                for d in CIFAR_D:
                    cell = f"{pipe}__split{i:02d}__seed{rep}__d{d}"
                    m.append(dict(semantic_id=cell, dataset=ds, pipeline_id=pipe,
                                  implementation_label="WRN28_10_PROSER_FEDAVG",
                                  backbone="wide_resnet_28_10", input_res="32x32",
                                  arm="proser_fedavg", aggregation="fedavg", lpd=False, gdca_ot=False,
                                  split_id=f"{ds}_split_{i:02d}", train_rep=rep, d=d,
                                  paired_block=block, reuse_class="fresh_training", **seeds(block)))
    # Office-Home ConvNeXt @224 FedAvg: Arm A full-FT, Arm B frozen-linear
    for arm, pipe, mode in [("convnext_full_ft", "officehome_convnext_full", "full_finetune"),
                            ("convnext_frozen_linear", "officehome_convnext_frozen", "frozen_encoder_linear_head")]:
        for i in range(N_SPLITS):
            for rep in range(N_REPS):
                block = f"officehome__split{i:02d}__seed{rep}"
                cell = f"{pipe}__split{i:02d}__seed{rep}"
                m.append(dict(semantic_id=cell, dataset="officehome", pipeline_id=pipe,
                              implementation_label=f"CONVNEXT_TINY_224_{mode.upper()}_FEDAVG",
                              backbone="convnext_tiny_in1k", input_res="224x224",
                              arm=arm, aggregation="fedavg", lpd=False, gdca_ot=False,
                              split_id=f"officehome_split_{i:02d}", train_rep=rep, d="",
                              paired_block=block, reuse_class="fresh_training", **seeds(block)))

    fields = list(m[0].keys())
    with open(f"{OUT}/final_training_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(m)
    mh = hashlib.sha256(open(f"{OUT}/final_training_matrix.csv", "rb").read()).hexdigest()
    open(f"{OUT}/final_training_matrix.sha256", "w").write(f"{mh}  final_training_matrix.csv\n")

    # exact-reuse audit
    with open(f"{OUT}/prelaunch/exact_reuse_audit.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["semantic_id", "reuse_class", "reason"])
        for r in m:
            w.writerow([r["semantic_id"], "fresh_training",
                        "new balanced 400R splits + corrected pipeline labels; no historical cell has exact semantic identity (prior CIFAR used run_cifar splits; prior Office-Home was the ConvNeXt selector-family study, not this full-vs-frozen paired design)"])

    # cardinality gate
    ids = [r["semantic_id"] for r in m]
    oh = [r for r in m if r["dataset"] == "officehome"]
    ohb = {}
    for r in oh:
        ohb.setdefault(r["paired_block"], set()).add(r["arm"])
    c10b, c100b = {}, {}
    for r in m:
        if r["dataset"] == "cifar10": c10b.setdefault(r["paired_block"], set()).add(r["d"])
        if r["dataset"] == "cifar100": c100b.setdefault(r["paired_block"], set()).add(r["d"])
    card = dict(total_rows=len(m), duplicate_semantic_ids=len(ids) - len(set(ids)),
                cifar10_proser_fedavg=sum(1 for r in m if r["pipeline_id"] == "cifar10_proser_fedavg"),
                cifar100_proser_fedavg=sum(1 for r in m if r["pipeline_id"] == "cifar100_proser_fedavg"),
                officehome_convnext_full=sum(1 for r in oh if r["arm"] == "convnext_full_ft"),
                officehome_convnext_frozen=sum(1 for r in oh if r["arm"] == "convnext_frozen_linear"),
                officehome_complete_paired_blocks=sum(1 for v in ohb.values() if v == {"convnext_full_ft", "convnext_frozen_linear"}),
                officehome_unpaired_blocks=sum(1 for v in ohb.values() if v != {"convnext_full_ft", "convnext_frozen_linear"}),
                cifar10_complete_3d_blocks=sum(1 for v in c10b.values() if len(v) == 3),
                cifar100_complete_3d_blocks=sum(1 for v in c100b.values() if len(v) == 3),
                no_fedpd_no_gdca_ot=all((not r["lpd"]) and (not r["gdca_ot"]) for r in m),
                matrix_sha256=mh)
    expect = dict(total_rows=400, duplicate_semantic_ids=0, cifar10_proser_fedavg=150,
                  cifar100_proser_fedavg=150, officehome_convnext_full=50, officehome_convnext_frozen=50,
                  officehome_complete_paired_blocks=50, officehome_unpaired_blocks=0,
                  cifar10_complete_3d_blocks=50, cifar100_complete_3d_blocks=50)
    card["matches_gate_expectation"] = all(card[k] == v for k, v in expect.items())
    json.dump(card, open(f"{OUT}/prelaunch/matrix_cardinality.json", "w"), indent=2)
    for k, v in card.items(): print(f"  {k}: {v}")


if __name__ == "__main__":
    build()
