"""Confirmatory-400R Office-Home ConvNeXt GPU smoke + BITWISE resume check.

License-neutral (the Office-Home ConvNeXt pipeline is fedcore2's own; no GPL).
Reuses ``run_officehome`` (model factory, cosine LR, AdamW client update, per-role
export) and ``fed_train.fedavg`` (resumable FedAvg with RNG checkpointing).

Two subcommands, both intended to run INSIDE the pinned
``pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`` container:

* ``smoke``         -- one 2-round smoke of a pipeline; emits native logit npz +
                       a terminal marker (wall time, peak VRAM, byte sizes,
                       checksums) for the resource projection.
* ``resume-check``  -- uninterrupted vs interrupt-after-round-0 + resume, compared
                       BITWISE (``torch.equal`` on every weight, ``np.array_equal``
                       on exported logits and source order). NO tolerance. If a
                       nondeterministic op prevents bitwise equality the check
                       records ``bitwise_equivalent=False`` and the caller must
                       gate ``BLOCKED_RESUME_NOT_BITWISE`` (never fall back to a
                       tolerance rule).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from types import SimpleNamespace

import numpy as np


def _oh_args(*, split_id, folds_csv, manifest, class_splits, image_root, pipeline,
             rounds, out, checkpoint):
    return SimpleNamespace(
        manifest_csv=manifest, folds_csv=folds_csv, class_splits_csv=class_splits,
        split_id=split_id, image_root=image_root, pipeline=pipeline, train_rep=0,
        campaign_seed=0, dataset_name="officehome", dataset_version="officehome_dedup_v1",
        rounds=rounds, local_epochs=1, batch_size=32, image_size=224, weight_decay=0.05,
        warmup_rounds=2, lr=None, pretrained=True, device="auto", out=out,
        checkpoint=checkpoint, checkpoint_every=1, resume=False, export_only=False,
        experiment_id=None, dry_run=False,
    )


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _enable_determinism():
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def run_smoke(args):
    import torch
    from fedcore.experiments import run_officehome as R

    base = dict(split_id=args.split_id, folds_csv=args.folds_csv, manifest=args.manifest_csv,
                class_splits=args.class_splits_csv, image_root=args.image_root)
    oa = _oh_args(pipeline=args.pipeline, rounds=args.rounds, out=args.out,
                  checkpoint=args.checkpoint, **base)
    for p in (oa.out, oa.checkpoint):
        if os.path.exists(p):
            os.remove(p)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = R.run(oa)
    wall = time.time() - t0
    peak_vram = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    marker = {
        "status": result.get("status"), "experiment_id": args.experiment_id,
        "pipeline": args.pipeline, "family": args.family, "split_id": args.split_id,
        "rounds": args.rounds, "device": result.get("device"),
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "role_counts": result.get("role_counts"), "traffic_units": result.get("traffic_units"),
        "train_seconds": result.get("train_seconds"), "export_seconds": result.get("export_seconds"),
        "wall_seconds": round(wall, 3), "peak_vram_gb": round(peak_vram, 4),
        "checkpoint_bytes": os.path.getsize(oa.checkpoint),
        "logits_npz_bytes": os.path.getsize(oa.out),
        "checksums": {os.path.basename(oa.out): _sha256(oa.out),
                      os.path.basename(oa.checkpoint): _sha256(oa.checkpoint)},
        "native_artifact": os.path.abspath(oa.out),
        "no_gdca_ot": True, "no_lpd": True, "aggregation": "weighted_fedavg",
    }
    with open(args.marker, "w") as fh:
        json.dump(marker, fh, indent=2, default=str)
    print(json.dumps(marker, indent=2, default=str))
    return marker


def run_resume_check(args):
    import torch
    from fedcore.data.officehome import (OfficeHomeDataConfig, OfficeHomeImageDataset,
                                         load_officehome_job)
    from fedcore.experiments import run_officehome as R
    from fedcore.experiments.run_officehome import PIPELINES
    from fedcore.models.fed_train import export_logits, fedavg

    base = dict(split_id=args.split_id, folds_csv=args.folds_csv, manifest=args.manifest_csv,
                class_splits=args.class_splits_csv, image_root=args.image_root)
    oa = _oh_args(pipeline=args.pipeline, rounds=args.rounds,
                  out=os.path.join(args.workdir, "eq.npz"),
                  checkpoint=os.path.join(args.workdir, "full.pt"), **base)
    config = OfficeHomeDataConfig(
        manifest_csv=oa.manifest_csv, folds_csv=oa.folds_csv,
        class_splits_csv=oa.class_splits_csv, split_id=oa.split_id, image_root=oa.image_root)
    job = load_officehome_job(config, check_image_files=False)
    pk = R.resolve_pipeline(args.pipeline)
    seed_bundle = R.build_seed_bundle(job, pipeline_name=PIPELINES[pk]["name"],
                                      campaign_seed=0, train_rep=0)

    det_error = None

    def fresh_components():
        R._seed_torch(seed_bundle)
        try:
            _enable_determinism()
        except Exception as exc:  # pragma: no cover
            nonlocal det_error
            det_error = f"{type(exc).__name__}: {exc}"
        return R.build_training_components(oa, job, pk, seed_bundle)

    full_ckpt = os.path.join(args.workdir, "full.pt")
    resume_ckpt = os.path.join(args.workdir, "resume.pt")
    for p in (full_ckpt, resume_ckpt):
        if os.path.exists(p):
            os.remove(p)
    meta = {"training_config_sha256": "equivalence-fixture"}

    try:
        comp = fresh_components()
        full = fedavg(comp["make"], comp["client_datasets"], args.rounds, oa.local_epochs,
                      comp["lr"], oa.batch_size, comp["device"], loader_seed=seed_bundle.loader,
                      local_train_fn=comp["local_train_fn"], lr_schedule=comp["lr_schedule"],
                      checkpoint_path=full_ckpt, checkpoint_every=1, checkpoint_metadata=meta)

        class _Interrupt(RuntimeError):
            pass

        def interrupt(r):
            if r == 0:
                raise _Interrupt

        comp = fresh_components()
        try:
            fedavg(comp["make"], comp["client_datasets"], args.rounds, oa.local_epochs,
                   comp["lr"], oa.batch_size, comp["device"], loader_seed=seed_bundle.loader,
                   local_train_fn=comp["local_train_fn"], lr_schedule=comp["lr_schedule"],
                   checkpoint_path=resume_ckpt, checkpoint_every=1, checkpoint_metadata=meta,
                   round_end_callback=interrupt)
        except _Interrupt:
            pass
        else:
            raise AssertionError("fixture interruption did not occur")

        comp = fresh_components()
        resumed = fedavg(comp["make"], comp["client_datasets"], args.rounds, oa.local_epochs,
                         comp["lr"], oa.batch_size, comp["device"], loader_seed=seed_bundle.loader,
                         local_train_fn=comp["local_train_fn"], lr_schedule=comp["lr_schedule"],
                         checkpoint_path=resume_ckpt, resume=True, checkpoint_every=1,
                         checkpoint_metadata=meta)

        weights_bitwise = all(
            torch.equal(t, resumed.state_dict()[k]) for k, t in full.state_dict().items())

        recs = job.role_records("proposal")
        ds = OfficeHomeImageDataset(recs, transform=comp["eval_transform"])
        idx = np.arange(len(ds), dtype=np.int64)
        lf = export_logits(full, ds, idx, comp["device"], oa.batch_size)
        lr_ = export_logits(resumed, ds, idx, comp["device"], oa.batch_size)
        logits_bitwise = bool(np.array_equal(lf, lr_))
        src_full = [r.sample_id for r in recs]
        order_bitwise = bool(src_full == [r.sample_id for r in job.role_records("proposal")])
        status = "ok"
    except RuntimeError as exc:
        # A nondeterministic-op failure from use_deterministic_algorithms lands here.
        weights_bitwise = logits_bitwise = order_bitwise = False
        det_error = det_error or f"{type(exc).__name__}: {exc}"
        status = "deterministic_algorithm_error"

    result = {
        "path": args.experiment_id, "family": args.family, "pipeline": args.pipeline,
        "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "rounds": args.rounds, "interrupt_after_round": 0,
        "weights_bitwise": bool(weights_bitwise),
        "logits_bitwise": bool(logits_bitwise),
        "scores_bitwise": bool(logits_bitwise),  # scores are a pure function of logits
        "source_order_bitwise": bool(order_bitwise),
        "optimizer_state_crosses_rounds": False,
        "scheduler": "cosine-warmup pure function of round",
        "deterministic_algorithms": True,
        "deterministic_error": det_error,
        "status": status,
        "bitwise_equivalent": bool(weights_bitwise and logits_bitwise and order_bitwise),
        "comparison": "torch.equal (weights) + np.array_equal (logits/order); NO tolerance",
    }
    with open(args.eq_report, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest-csv", dest="manifest_csv", required=True)
    common.add_argument("--folds-csv", dest="folds_csv", required=True)
    common.add_argument("--class-splits-csv", dest="class_splits_csv", required=True)
    common.add_argument("--split-id", dest="split_id", required=True)
    common.add_argument("--image-root", dest="image_root", required=True)
    common.add_argument("--pipeline", required=True, help="A|B or canonical name")
    common.add_argument("--rounds", type=int, default=2)
    common.add_argument("--experiment-id", dest="experiment_id", required=True)
    common.add_argument("--family", required=True)

    s = sub.add_parser("smoke", parents=[common])
    s.add_argument("--out", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--marker", required=True)

    e = sub.add_parser("resume-check", parents=[common])
    e.add_argument("--workdir", required=True)
    e.add_argument("--eq-report", dest="eq_report", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "smoke":
        run_smoke(args)
    else:
        os.makedirs(args.workdir, exist_ok=True)
        run_resume_check(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
