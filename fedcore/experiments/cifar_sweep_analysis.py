"""Final paired analysis for the CIFAR backbone sweep (owner-specified comparisons).

Reads the recertification per-cell certificates (recertify_cifar_sweep.py) and
reports the three PAIRED, zero-imputed CertifiedCoverage@alpha=0.20 comparisons,
paired by (dataset, split, rep, d) block:

  PROSER-objective effect          : arm A (WRN PROSER)   - arm B (WRN plain)
  CIFAR-native architecture effect : arm C (ResNeXt plain)- arm B (WRN plain)
  secondary whole-pipeline         : arm C (ResNeXt plain)- arm A (WRN PROSER)

Metric per cell = cert_coverage_lcb (the Holm-certified coverage LCB; 0 when the
cell does not certify -- zero-imputation). Paired mean difference with a paired
percentile bootstrap 95% CI, reported per-d and pooled, per dataset and overall.

Owner caveat honoured: ResNeXt is NOT asserted to improve certification (its
published advantage is closed-set classification). This script only reports the
paired differences; it makes no superiority claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

ARM_A = "wrn28_10_proser_fedavg"
ARM_B = "wrn28_10_plain_fedavg"
ARM_C = "resnext29_8x64d_plain_fedavg"
COMPARISONS = [
    ("PROSER_objective_effect", ARM_A, ARM_B),
    ("CIFAR_native_architecture_effect", ARM_C, ARM_B),
    ("secondary_whole_pipeline", ARM_C, ARM_A),
]
N_BOOT = 10000


def _block(r):
    return (r["dataset"], r["split_id"], r["train_rep"], str(r["d"]))


def _boot_ci(diffs, seed=0):
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _paired(cov, blocks, hi, lo):
    diffs, hv, lv = [], [], []
    for b in blocks:
        if hi in cov[b] and lo in cov[b]:
            diffs.append(cov[b][hi] - cov[b][lo])
            hv.append(cov[b][hi]); lv.append(cov[b][lo])
    diffs = np.asarray(diffs)
    lo_ci, hi_ci = _boot_ci(diffs)
    return {
        "n_paired_blocks": len(diffs),
        "mean_coverage_hi_arm": float(np.mean(hv)) if hv else float("nan"),
        "mean_coverage_lo_arm": float(np.mean(lv)) if lv else float("nan"),
        "mean_paired_diff": float(np.mean(diffs)) if len(diffs) else float("nan"),
        "boot_ci95": [lo_ci, hi_ci],
        "frac_hi_certifies": float(np.mean([1.0 if v > 0 else 0.0 for v in hv])) if hv else float("nan"),
        "frac_lo_certifies": float(np.mean([1.0 if v > 0 else 0.0 for v in lv])) if lv else float("nan"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-cell", default="results/cifar_backbone_sweep/recert/per_cell_certificates.csv")
    ap.add_argument("--out", default="results/cifar_backbone_sweep/recert/paired_analysis.json")
    args = ap.parse_args(argv)

    rows = list(csv.DictReader(open(args.per_cell, newline="")))
    # cov[block][arm] = cert_coverage_lcb (zero-imputed already in the recert output)
    cov = {}
    for r in rows:
        cov.setdefault(_block(r), {})[r["arm"]] = float(r["cert_coverage_lcb"])

    all_blocks = list(cov)
    ds_values = sorted({b[0] for b in all_blocks})
    d_values = sorted({b[3] for b in all_blocks}, key=float)

    report = {"alpha": 0.20, "metric": "zero_imputed_holm_certified_coverage_lcb",
              "comparisons": {}, "owner_caveat":
              "ResNeXt-29 is not asserted to improve certification; its published "
              "advantage concerns closed-set classification. Paired differences only."}
    for name, hi, lo in COMPARISONS:
        entry = {"hi_arm": hi, "lo_arm": lo, "pooled": _paired(cov, all_blocks, hi, lo),
                 "by_dataset": {}, "by_d": {}, "by_dataset_d": {}}
        for ds in ds_values:
            entry["by_dataset"][ds] = _paired(cov, [b for b in all_blocks if b[0] == ds], hi, lo)
        for d in d_values:
            entry["by_d"][d] = _paired(cov, [b for b in all_blocks if b[3] == d], hi, lo)
        for ds in ds_values:
            for d in d_values:
                key = f"{ds}__d{d}"
                entry["by_dataset_d"][key] = _paired(
                    cov, [b for b in all_blocks if b[0] == ds and b[3] == d], hi, lo)
        report["comparisons"][name] = entry

    # per-arm marginal certified-coverage summary
    report["per_arm_marginal"] = {}
    for arm in (ARM_A, ARM_B, ARM_C):
        vals = [cov[b][arm] for b in all_blocks if arm in cov[b]]
        report["per_arm_marginal"][arm] = {
            "n_cells": len(vals),
            "mean_certified_coverage": float(np.mean(vals)) if vals else float("nan"),
            "frac_cells_certified": float(np.mean([1.0 if v > 0 else 0.0 for v in vals])) if vals else float("nan"),
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
