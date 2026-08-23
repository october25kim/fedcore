"""CIFAR plain-FedAvg bitwise-resume equivalence + WRN-plain smoke (prelaunch gate).

Proves the run_cifar_common training is restart-safe at the bit level: an
uninterrupted N-round run and an (interrupt-after-round-0 + --resume) run produce
BYTE-IDENTICAL final model weights. Also doubles as the WRN-28-10 plain-FedAvg
smoke (trains + exports the common per-obs schema). Emits a JSON report.

Usage: python -m fedcore.experiments.cifar_resume_check \
    --backbone wrn28_10 --dataset cifar10 --n-known 6 --rounds 2 \
    --data-root /repo/data --unknown-classes 4,5,7,9 --workdir <rw dir> \
    --report <path.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from fedcore.experiments.run_cifar_common import (
    _seed_deterministic, build_job, export_common, msp_score,
)
from fedcore.models.fed_train import fedavg
from fedcore.models.models import make_model
import fedcore.experiments.confirmatory_400r_common_schema as CS


def _state_hash(model) -> str:
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        v = sd[k].detach().cpu().contiguous().numpy()
        h.update(k.encode())
        h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


class _Interrupt(RuntimeError):
    pass


def _run_fedavg(args, device, checkpoint, rounds, resume, round_end_callback=None):
    _seed_deterministic(args.seed)
    job = build_job(args.dataset, args.data_root, args.n_known, args.n_clients,
                    args.dirichlet_alpha, args.seed, args.unknown_classes)
    make = lambda: make_model(args.n_known, backbone=args.backbone, norm=args.norm)
    meta = {"training_config_sha256": "cifar-resume-check"}
    model = fedavg(make, job["client_datasets"], rounds, args.local_epochs, args.lr,
                   args.batch_size, device, loader_seed=args.seed,
                   checkpoint_path=checkpoint, resume=resume, checkpoint_every=1,
                   checkpoint_metadata=meta, round_end_callback=round_end_callback)
    return model, job


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--norm", default="bn")
    p.add_argument("--n-known", dest="n_known", type=int, required=True)
    p.add_argument("--n-clients", dest="n_clients", type=int, default=5)
    p.add_argument("--dirichlet-alpha", dest="dirichlet_alpha", type=float, default=0.1)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--local-epochs", dest="local_epochs", type=int, default=1)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", dest="data_root", required=True)
    p.add_argument("--unknown-classes", dest="unknown_classes", default=None,
                   type=lambda s: [int(x) for x in s.split(",")] if s else None)
    p.add_argument("--workdir", required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    os.makedirs(args.workdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # (1) uninterrupted full run -> also the WRN-plain smoke (export common schema)
    full_ckpt = os.path.join(args.workdir, "full.pt")
    if os.path.exists(full_ckpt):
        os.remove(full_ckpt)
    full, job = _run_fedavg(args, device, full_ckpt, args.rounds, resume=False)
    full_hash = _state_hash(full)
    smoke_npz = os.path.join(args.workdir, "smoke_common.npz")
    export_common(full, job, device, args.dataset, "resume_check", smoke_npz,
                  args.backbone, args.batch_size)
    desc = CS.describe_common_npz(smoke_npz)
    z = np.load(smoke_npz, allow_pickle=False)
    kl = z["test__known_logits"]; y = z["test__true_known_class_index_or_neg1"]
    k = y >= 0
    known_acc = float((kl.argmax(1)[k] == y[k]).mean()) if k.sum() else float("nan")
    smoke_finite = bool(np.isfinite(kl).all())

    # (2) interrupt AFTER round 0 (same total rounds), then --resume to the end.
    # fedavg writes the round-r checkpoint BEFORE round_end_callback(r), so raising
    # at r==0 leaves a valid round_completed=0 / rounds_requested=args.rounds ckpt.
    part_ckpt = os.path.join(args.workdir, "part.pt")
    if os.path.exists(part_ckpt):
        os.remove(part_ckpt)

    def interrupt(r):
        if r == 0:
            raise _Interrupt

    try:
        _run_fedavg(args, device, part_ckpt, args.rounds, resume=False,
                    round_end_callback=interrupt)
    except _Interrupt:
        pass
    else:
        raise AssertionError("fixture interruption did not occur (rounds must be >= 2)")
    resumed, _ = _run_fedavg(args, device, part_ckpt, args.rounds, resume=True)
    resumed_hash = _state_hash(resumed)

    bitwise_identical = (full_hash == resumed_hash)
    report = {
        "campaign": "cifar_backbone_sweep",
        "check": "bitwise_resume_equivalence + plain_fedavg_smoke",
        "backbone": args.backbone, "dataset": args.dataset, "rounds": args.rounds,
        "device": device,
        "bitwise_resume_identical": bitwise_identical,
        "uninterrupted_state_sha256": full_hash,
        "resumed_state_sha256": resumed_hash,
        "smoke_common_schema": desc["meta"],
        "smoke_known_acc": round(known_acc, 4),
        "smoke_finite": smoke_finite,
        "verdict": "PASS" if (bitwise_identical and smoke_finite) else "FAIL",
    }
    tmp = f"{args.report}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    os.replace(tmp, args.report)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
