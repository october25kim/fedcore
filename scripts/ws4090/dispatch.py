#!/usr/bin/env python3
"""Generic N-GPU job dispatcher for the Fed-CORE training grids (R1/R2/R5/...).

Packs up to JOBS_PER_GPU jobs per GPU in flight, pinning each to a single device
via CUDA_VISIBLE_DEVICES, and refills a slot as soon as its job exits. This turns a
long serial grid of independent (config, seed) runs into an N-way parallel sweep
-- the whole point of moving to the multi-GPU box. With JOBS_PER_GPU=1 (default)
this is exactly the historical one-job-per-GPU behavior.

Manifest format (one job per line; blank / #-comment lines ignored):

    LABEL <TAB> EXPECTED_OUTPUT <TAB> SHELL_COMMAND [<TAB> MODEL]

  * LABEL           : short unique id, used for the per-job log filename.
  * EXPECTED_OUTPUT : path that exists iff the job already succeeded; with
                      --skip-existing / RESUME=1 the job is skipped when it is
                      present (idempotent, resumable re-runs). Use "-" to never
                      skip. Resume matches on the per-run OUTPUT (npz), exactly
                      how run_cifar writes results -- NOT on a shared CSV row.
  * SHELL_COMMAND   : a device-AGNOSTIC command (do NOT set CUDA_VISIBLE_DEVICES
                      yourself -- the dispatcher sets it per slot).
  * MODEL           : OPTIONAL packing class (e.g. resnet18, wrn, foogd_full).
                      Selects the per-model cap when JOBS_PER_GPU is a map.
                      Absent -> class "default".

Packing (JOBS_PER_GPU, --jobs-per-gpu, or env JOBS_PER_GPU):
  * a scalar N            -> N jobs per GPU for every class.
  * a per-model map, e.g. "resnet18=3,wrn=2,foogd_full=1,default=3" -> each GPU
    holds up to cap[class] jobs of that class; mixed classes share a GPU by
    weight (a class-c job costs 1/cap[c] of the GPU, capacity 1.0), so a GPU is
    never packed past its tightest model's budget. Conservative for 24 GB.

Guardrails honored: NO silent retries (a failed job is logged, recorded as
failed, and the sweep continues); every job's stdout+stderr is captured; a status
CSV is written so nothing is hidden. FIFO order is preserved so a front-loaded
manifest still yields the most-complete grid if interrupted.

Env knobs (CLI flags take precedence when explicitly passed):
    NUM_GPUS=4            -> --gpus 0,1,..,NUM_GPUS-1   (GPUS="0,2" overrides ids)
    JOBS_PER_GPU=3        -> --jobs-per-gpu (scalar or "cifar=3,wrn=2,..." map)
    RESUME=1              -> --skip-existing
    DRY_RUN=1             -> --dry-run

Usage:
    python scripts/ws4090/dispatch.py manifest_R2.txt --gpus 0,1,2,3 --skip-existing
    JOBS_PER_GPU=3 RESUME=1 python scripts/ws4090/dispatch.py manifest_R2.txt
    DRY_RUN=1 python scripts/ws4090/dispatch.py manifest_R2.txt
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time

_CAP_EPS = 1e-9


def parse_manifest(path):
    jobs = []
    with open(path) as f:
        for ln, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                sys.exit(f"[dispatch] manifest line {ln}: expected >=3 tab fields, got {len(parts)}: {line!r}")
            label, expected = parts[0].strip(), parts[1].strip()
            # MODEL is an OPTIONAL trailing field; anything between CMD and it is
            # part of the (tab-free) command. We take the LAST field as MODEL only
            # when there are >=4 fields, else the whole tail is the command.
            if len(parts) >= 4:
                model = parts[-1].strip() or "default"
                cmd = "\t".join(parts[2:-1]).strip()
            else:
                model = "default"
                cmd = parts[2].strip()
            jobs.append({"label": label, "expected": expected, "cmd": cmd, "model": model})
    labels = [j["label"] for j in jobs]
    if len(labels) != len(set(labels)):
        sys.exit("[dispatch] duplicate LABELs in manifest (log files would collide)")
    return jobs


def parse_jobs_per_gpu(spec):
    """Return a dict class -> cap(int). A bare int applies to 'default' (and any
    class via fallback). A "a=3,b=2" map sets per-class caps; a "default" key (or
    the implicit fallback of 1) covers unlisted classes."""
    spec = str(spec).strip()
    if not spec:
        return {"default": 1}
    if "=" not in spec:
        n = int(spec)
        if n < 1:
            sys.exit(f"[dispatch] JOBS_PER_GPU must be >=1, got {n}")
        return {"default": n}
    caps = {}
    for kv in spec.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, _, v = kv.partition("=")
        n = int(v)
        if n < 1:
            sys.exit(f"[dispatch] JOBS_PER_GPU cap for {k!r} must be >=1, got {n}")
        caps[k.strip()] = n
    caps.setdefault("default", 1)
    return caps


def _cap_for(model, caps):
    return caps.get(model, caps.get("default", 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids (default: env GPUS, else 0..NUM_GPUS-1, else 0,1,2,3)")
    ap.add_argument("--jobs-per-gpu", default=None,
                    help="int, or per-model map 'resnet18=3,wrn=2,foogd_full=1' (default: env JOBS_PER_GPU or 1)")
    ap.add_argument("--logdir", default="runs/ws4090_logs")
    ap.add_argument("--status", default=None, help="status CSV (default runs/ws4090_logs/<manifest>.status.csv)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a job whose EXPECTED_OUTPUT already exists (or env RESUME=1)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan + GPU assignment, run nothing (or env DRY_RUN=1)")
    args = ap.parse_args()

    # --- resolve gpus (CLI > GPUS env > NUM_GPUS env > default) ---
    gpu_spec = args.gpus if args.gpus is not None else os.environ.get("GPUS")
    if gpu_spec is None:
        n_gpus = int(os.environ.get("NUM_GPUS", "4"))
        gpu_spec = ",".join(str(i) for i in range(n_gpus))
    gpus = [g.strip() for g in gpu_spec.split(",") if g.strip() != ""]
    if not gpus:
        sys.exit("[dispatch] no GPUs resolved")

    # --- resolve packing / knobs (CLI > env) ---
    jpg_spec = args.jobs_per_gpu if args.jobs_per_gpu is not None else os.environ.get("JOBS_PER_GPU", "1")
    caps = parse_jobs_per_gpu(jpg_spec)
    skip_existing = args.skip_existing or os.environ.get("RESUME", "") in ("1", "true", "True")
    dry_run = args.dry_run or os.environ.get("DRY_RUN", "") in ("1", "true", "True")

    jobs = parse_manifest(args.manifest)
    os.makedirs(args.logdir, exist_ok=True)
    status_path = args.status or os.path.join(
        args.logdir, os.path.basename(args.manifest) + ".status.csv")

    # skip already-done jobs (idempotent/resumable on the per-run output)
    todo, skipped = [], []
    for j in jobs:
        if skip_existing and j["expected"] != "-" and os.path.exists(j["expected"]):
            skipped.append(j)
        else:
            todo.append(j)

    def weight(job):
        return 1.0 / _cap_for(job["model"], caps)

    print(f"[dispatch] manifest={args.manifest}  jobs={len(jobs)}  todo={len(todo)}  "
          f"skipped(existing)={len(skipped)}  gpus={gpus}  jobs_per_gpu={caps}")
    if dry_run:
        # Simulate the exact FIFO packing to show a valid GPU assignment per cell.
        load = {g: 0.0 for g in gpus}
        cap = 1.0 + _CAP_EPS
        wave, assigned = 0, 0
        pending = list(todo)
        while pending:
            placed_any = False
            wave += 1
            # greedily fill this wave without exceeding per-GPU capacity
            still = []
            for j in pending:
                w = weight(j)
                g = min((g for g in gpus if load[g] + w <= cap), key=lambda g: load[g], default=None)
                if g is None:
                    still.append(j)
                    continue
                load[g] += w
                assigned += 1
                placed_any = True
                print(f"  wave{wave} gpu{g} [{j['label']}] model={j['model']} w={w:.3f} -> {j['expected']}\n        {j['cmd']}")
            pending = still
            if not placed_any:
                sys.exit(f"[dispatch] DRY_RUN: {len(pending)} job(s) cannot be placed on any GPU with caps={caps}")
            load = {g: 0.0 for g in gpus}  # next wave starts fresh (jobs from this wave 'complete')
        for j in skipped:
            print(f"  SKIP [{j['label']}] (exists: {j['expected']})")
        print(f"[dispatch] DRY_RUN plan: {assigned} job(s) across {len(gpus)} GPU(s) in {wave} wave(s); "
              f"packing={caps}")
        print(f"[dispatch] DRY_RUN totals: todo={len(todo)} skipped={len(skipped)} gpus={gpus} jobs_per_gpu={caps}")
        return

    cap = 1.0 + _CAP_EPS
    load = {g: 0.0 for g in gpus}     # used capacity per GPU (float; 1.0 == full)
    running = []                      # list of {job,p,logf,t0,gpu,weight}
    results = []                      # status rows
    queue = list(todo)
    t_start = time.time()

    def launch(gpu, job):
        w = weight(job)
        logf = open(os.path.join(args.logdir, job["label"] + ".log"), "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        logf.write(f"# label={job['label']}\n# gpu={gpu}\n# model={job['model']}\n# cmd={job['cmd']}\n\n")
        logf.flush()
        p = subprocess.Popen(job["cmd"], shell=True, stdout=logf,
                             stderr=subprocess.STDOUT, env=env)
        load[gpu] += w
        running.append({"job": job, "p": p, "logf": logf, "t0": time.time(),
                        "gpu": gpu, "weight": w})
        print(f"[dispatch] START gpu{gpu} [{job['label']}] (load={load[gpu]:.2f})  "
              f"({len(queue)} queued, {len(results)} done, {len(running)} running)")

    while queue or running:
        # fill any free capacity, FIFO (front-loaded manifest order preserved)
        progressed = True
        while queue and progressed:
            job = queue[0]
            w = weight(job)
            g = min((g for g in gpus if load[g] + w <= cap), key=lambda g: load[g], default=None)
            if g is None:
                progressed = False
            else:
                queue.pop(0)
                launch(g, job)
        time.sleep(2)
        # reap finished jobs
        for r in list(running):
            rc = r["p"].poll()
            if rc is None:
                continue
            r["logf"].close()
            dt = round(time.time() - r["t0"], 1)
            job = r["job"]
            ok = (rc == 0)
            if ok and job["expected"] != "-" and not os.path.exists(job["expected"]):
                ok = False  # exited 0 but no output -> treat as failure, do not hide
                note = "rc0_but_no_output"
            else:
                note = "" if ok else f"rc={rc}"
            print(f"[dispatch] {'OK  ' if ok else 'FAIL'} gpu{r['gpu']} [{job['label']}] {dt}s {note}")
            results.append({"label": job["label"], "gpu": r["gpu"], "returncode": rc,
                            "seconds": dt, "ok": int(ok), "expected": job["expected"],
                            "note": note})
            load[r["gpu"]] -= r["weight"]
            running.remove(r)

    # status CSV (append the skipped, too, for a complete record)
    for j in skipped:
        results.append({"label": j["label"], "gpu": "-", "returncode": 0, "seconds": 0,
                        "ok": 1, "expected": j["expected"], "note": "skipped_existing"})
    with open(status_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "gpu", "returncode", "seconds", "ok", "expected", "note"])
        w.writeheader(); w.writerows(results)

    nfail = sum(1 for r in results if not r["ok"])
    total_dt = round(time.time() - t_start, 1)
    print(f"\n[dispatch] DONE in {total_dt}s  ok={len(results)-nfail}/{len(results)}  failed={nfail}")
    print(f"[dispatch] status -> {status_path}  logs -> {args.logdir}/")
    if nfail:
        print("[dispatch] FAILED jobs:")
        for r in results:
            if not r["ok"]:
                print(f"    [{r['label']}] {r['note']}  (see {args.logdir}/{r['label']}.log)")
        sys.exit(1)


if __name__ == "__main__":
    main()
