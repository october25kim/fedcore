"""In-container GPU prelaunch checks for the Office-Home arm.

Runs, on the allowed physical GPUs, the torch-dependent prelaunch checks:

* a 2-round smoke of each pipeline on ``split_0 x train_rep_0`` (measures wall
  time for the resource projection);
* interrupted-vs-resumed bit-exact equivalence for both pipelines (crash after
  round 0's checkpoint, then ``--resume``, compare final weights and exported
  logits);
* native sample-ID round trip (exported artifact IDs == frozen fold IDs).

Writes a machine-readable JSON report. This module is intended to be executed
INSIDE the pinned ``pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`` container.
It never launches the 50-cell matrix -- only the two smoke cells at 2 rounds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from types import SimpleNamespace

import numpy as np


def _args(split_id, folds_csv, manifest, class_splits, image_root, pipeline, rounds, out, checkpoint):
    return SimpleNamespace(
        manifest_csv=manifest,
        folds_csv=folds_csv,
        class_splits_csv=class_splits,
        split_id=split_id,
        image_root=image_root,
        pipeline=pipeline,
        train_rep=0,
        campaign_seed=0,
        dataset_name="officehome",
        dataset_version="officehome_dedup_v1",
        rounds=rounds,
        local_epochs=1,
        batch_size=32,
        image_size=224,
        weight_decay=0.05,
        warmup_rounds=2,
        lr=None,
        pretrained=True,
        device="auto",
        out=out,
        checkpoint=checkpoint,
        checkpoint_every=1,
        resume=False,
        export_only=False,
        experiment_id=None,
        dry_run=False,
    )


def _fold_sample_ids(folds_csv, split_manifest_role):
    ids = {}
    with open(folds_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            ids.setdefault(row["role"], set()).add(row["sample_id"])
    return ids


def _smoke(pipeline, base, out_dir, rounds):
    from fedcore.experiments import run_officehome as R

    out = os.path.join(out_dir, f"{pipeline}_smoke_logits.npz")
    ckpt = os.path.join(out_dir, f"{pipeline}_smoke.pt")
    for p in (out, ckpt):
        if os.path.exists(p):
            os.remove(p)
    args = _args(pipeline=pipeline, rounds=rounds, out=out, checkpoint=ckpt, **base)
    t0 = time.time()
    result = R.run(args)
    elapsed = time.time() - t0
    return result, elapsed, out


def _resume_equivalence(pipeline, base, out_dir, rounds):
    """fedavg-level interrupted-vs-resumed exact replay for one pipeline."""

    import torch

    from fedcore.data.officehome import OfficeHomeDataConfig, load_officehome_job
    from fedcore.experiments import run_officehome as R
    from fedcore.models.fed_train import export_logits

    args = _args(pipeline=pipeline, rounds=rounds,
                 out=os.path.join(out_dir, f"{pipeline}_eq.npz"),
                 checkpoint=os.path.join(out_dir, f"{pipeline}_full.pt"), **base)
    config = OfficeHomeDataConfig(
        manifest_csv=args.manifest_csv, folds_csv=args.folds_csv,
        class_splits_csv=args.class_splits_csv, split_id=args.split_id,
        image_root=args.image_root,
    )
    job = load_officehome_job(config, check_image_files=False)
    from fedcore.experiments.run_officehome import PIPELINES
    seed_bundle = R.build_seed_bundle(
        job, pipeline_name=PIPELINES[R.resolve_pipeline(pipeline)]["name"],
        campaign_seed=0, train_rep=0,
    )
    pk = R.resolve_pipeline(pipeline)

    def fresh_components():
        R._seed_torch(seed_bundle)
        return R.build_training_components(args, job, pk, seed_bundle)

    from fedcore.models.fed_train import fedavg

    full_ckpt = os.path.join(out_dir, f"{pipeline}_full.pt")
    resume_ckpt = os.path.join(out_dir, f"{pipeline}_resume.pt")
    for p in (full_ckpt, resume_ckpt):
        if os.path.exists(p):
            os.remove(p)

    meta = {"training_config_sha256": "equivalence-fixture"}

    # (1) Full uninterrupted run.
    comp = fresh_components()
    full = fedavg(
        comp["make"], comp["client_datasets"], rounds, args.local_epochs, comp["lr"],
        args.batch_size, comp["device"], loader_seed=seed_bundle.loader,
        local_train_fn=comp["local_train_fn"], lr_schedule=comp["lr_schedule"],
        checkpoint_path=full_ckpt, checkpoint_every=1, checkpoint_metadata=meta,
    )

    # (2) Interrupted after round 0 (checkpoint written), then resumed.
    class _Interrupt(RuntimeError):
        pass

    def interrupt(r):
        if r == 0:
            raise _Interrupt

    comp = fresh_components()
    try:
        fedavg(
            comp["make"], comp["client_datasets"], rounds, args.local_epochs, comp["lr"],
            args.batch_size, comp["device"], loader_seed=seed_bundle.loader,
            local_train_fn=comp["local_train_fn"], lr_schedule=comp["lr_schedule"],
            checkpoint_path=resume_ckpt, checkpoint_every=1, checkpoint_metadata=meta,
            round_end_callback=interrupt,
        )
    except _Interrupt:
        pass
    else:
        raise AssertionError("fixture interruption did not occur")

    comp = fresh_components()
    resumed = fedavg(
        comp["make"], comp["client_datasets"], rounds, args.local_epochs, comp["lr"],
        args.batch_size, comp["device"], loader_seed=seed_bundle.loader,
        local_train_fn=comp["local_train_fn"], lr_schedule=comp["lr_schedule"],
        checkpoint_path=resume_ckpt, resume=True, checkpoint_every=1, checkpoint_metadata=meta,
    )

    weights_equal = all(
        torch.equal(t, resumed.state_dict()[k]) for k, t in full.state_dict().items()
    )

    # Compare exported proposal logits too (weights-equal implies logit-equal).
    from fedcore.data.officehome import OfficeHomeImageDataset
    recs = job.role_records("proposal")
    ds = OfficeHomeImageDataset(recs, transform=comp["eval_transform"])
    idx = np.arange(len(ds), dtype=np.int64)
    lf = export_logits(full, ds, idx, comp["device"], args.batch_size)
    lr_ = export_logits(resumed, ds, idx, comp["device"], args.batch_size)
    logits_equal = bool(np.allclose(lf, lr_, atol=1e-6))

    return {"weights_equal": bool(weights_equal), "logits_equal": logits_equal}


def _id_roundtrip(artifact_path, folds_csv):
    fold_ids = _fold_sample_ids(folds_csv, None)
    data = np.load(artifact_path, allow_pickle=False)
    role_map = {"prop": "proposal", "cert": "certification", "eval": "evaluation", "traffic": "traffic"}
    checks = {}
    all_ok = True
    for out_role, fold_role in role_map.items():
        art_ids = set(np.asarray(data[f"{out_role}_sample_id"], dtype=str).tolist())
        expected = fold_ids.get(fold_role, set())
        ok = art_ids == expected
        checks[out_role] = {"artifact": len(art_ids), "fold": len(expected), "equal": ok}
        all_ok = all_ok and ok
    return {"per_role": checks, "all_equal": all_ok}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--folds-csv", required=True)
    parser.add_argument("--class-splits-csv", required=True)
    parser.add_argument("--split-id", default="officehome_split_0")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    base = dict(
        split_id=args.split_id,
        folds_csv=args.folds_csv,
        manifest=args.manifest_csv,
        class_splits=args.class_splits_csv,
        image_root=args.image_root,
    )

    import torch

    report = {
        "device_info": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        },
        "rounds": args.rounds,
        "pipelines": {},
    }

    for pipeline in ("A", "B"):
        result, elapsed, out = _smoke(pipeline, base, args.out_dir, args.rounds)
        idrt = _id_roundtrip(out, args.folds_csv)
        eq = _resume_equivalence(pipeline, base, args.out_dir, args.rounds)
        report["pipelines"][pipeline] = {
            "smoke_wall_s": round(elapsed, 3),
            "smoke_status": result.get("status"),
            "experiment_id": result.get("experiment_id"),
            "role_counts": result.get("role_counts"),
            "id_roundtrip": idrt,
            "resume_equivalence": eq,
            "pass": bool(idrt["all_equal"] and eq["weights_equal"] and eq["logits_equal"]),
        }

    report["all_pass"] = all(v["pass"] for v in report["pipelines"].values())
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
