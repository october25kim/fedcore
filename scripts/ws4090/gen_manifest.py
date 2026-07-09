#!/usr/bin/env python3
"""Emit dispatch manifests for the GPU training grids R1 / R2 / R5.

Each line is  LABEL <TAB> EXPECTED_NPZ <TAB> COMMAND  (see dispatch.py). Commands
are device-agnostic (the dispatcher pins CUDA_VISIBLE_DEVICES). Every run_cifar
call sets an explicit --out so the logits npz name encodes (task, J, backbone, d,
noise, seed) and never collides -- the default tag omits J and backbone.

Seed floor: 10 seeds {0..9} for R1/R2/R5 (per the queue; these are NOT detector
retraining cells). Job ORDER front-loads the highest-value cells so an interrupted
sweep still yields the most-complete grid:
  R1: seeds 0-4 for J=20 then J=10, then seeds 5-9 J=20 then J=10.
  R2: backbone-major resnet18gn -> resnet18bn -> simplecnn (complete earlier ones first).
  R5: rate-major, d-major.

Run: python scripts/ws4090/gen_manifest.py [--outdir scripts/ws4090]
"""
from __future__ import annotations

import argparse
import os

RUN = "python -m fedcore.experiments.run_cifar"
COMMON = "--rounds 50 --local_epochs 2 --alpha 0.10 --delta 0.10 --data_root data"
SEEDS = tuple(range(10))
# backbone label -> (--backbone, --norm)
BB = {"resnet18gn": ("resnet18", "gn"), "resnet18bn": ("resnet18", "bn"),
      "simplecnn": ("simplecnn", "bn")}


def _line(label, npz, cmd):
    return f"{label}\t{npz}\t{cmd}"


def r1_jobs():
    """CIFAR-10 client scaling: J in {10,20}, resnet18gn, d=0.5, clean, seeds 0..9."""
    bb, nm = BB["resnet18gn"]
    lines = []
    order = ([(20, s) for s in range(5)] + [(10, s) for s in range(5)] +
             [(20, s) for s in range(5, 10)] + [(10, s) for s in range(5, 10)])
    for J, s in order:
        out = f"runs/r1_cifar10_J{J}_d0.5_resnet18gn_seed{s}.csv"
        npz = out[:-4] + "_logits.npz"
        cmd = (f"{RUN} --dataset cifar10 --n_known 6 --n_clients {J} "
               f"--dirichlet_alpha 0.5 --seed {s} --backbone {bb} --norm {nm} "
               f"{COMMON} --out {out}")
        lines.append(_line(f"r1_J{J}_s{s}", npz, cmd))
    return lines


def r2_jobs():
    """CIFAR-100 multi-model: {resnet18gn,resnet18bn,simplecnn} x d{0.5,5}, n_known 60, seeds 0..9."""
    lines = []
    for lbl in ("resnet18gn", "resnet18bn", "simplecnn"):
        bb, nm = BB[lbl]
        for d in ("0.5", "5"):
            for s in SEEDS:
                out = f"runs/r2_cifar100_{lbl}_d{d}_seed{s}.csv"
                npz = out[:-4] + "_logits.npz"
                cmd = (f"{RUN} --dataset cifar100 --n_known 60 --n_clients 5 "
                       f"--dirichlet_alpha {d} --seed {s} --backbone {bb} --norm {nm} "
                       f"{COMMON} --out {out}")
                lines.append(_line(f"r2_{lbl}_d{d}_s{s}", npz, cmd))
    return lines


def r5_jobs():
    """Corruption curve: symmetric rates {0.1,0.2,0.35} x d{0.5,5}, resnet18gn, seeds 0..9.
    run_cifar corrupts TRAIN labels only; calibration folds stay clean (split hygiene built-in)."""
    bb, nm = BB["resnet18gn"]
    lines = []
    for rate in ("0.1", "0.2", "0.35"):
        for d in ("0.5", "5"):
            for s in SEEDS:
                out = f"runs/r5_cifar10_d{d}_sym{rate}_seed{s}.csv"
                npz = out[:-4] + "_logits.npz"
                cmd = (f"{RUN} --dataset cifar10 --n_known 6 --n_clients 5 "
                       f"--dirichlet_alpha {d} --seed {s} --backbone {bb} --norm {nm} "
                       f"--noise_type symmetric --noise_rate {rate} "
                       f"{COMMON} --out {out}")
                lines.append(_line(f"r5_d{d}_sym{rate}_s{s}", npz, cmd))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="scripts/ws4090")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for name, jobs in (("R1", r1_jobs()), ("R2", r2_jobs()), ("R5", r5_jobs())):
        path = os.path.join(args.outdir, f"manifest_{name}.txt")
        with open(path, "w") as f:
            f.write(f"# manifest_{name}: {len(jobs)} run_cifar jobs (device-agnostic; dispatch pins the GPU)\n")
            f.write("\n".join(jobs) + "\n")
        print(f"wrote {path}  ({len(jobs)} jobs)")


if __name__ == "__main__":
    main()
