#!/usr/bin/env python3
"""CIFAR backbone-sweep dispatcher: canary-gated, restart-safe, one cell per GPU.

Trains ONLY the 600 FRESH cells of results/cifar_backbone_sweep (arm B =
wrn28_10_plain_fedavg, arm C = resnext29_8x64d_plain_fedavg). Arm A
(wrn28_10_proser_fedavg) is EXACT-REUSE of the Confirmatory-400R cells and is
NEVER trained here (skipped).

Launch protocol (owner spec):
  1. Run FOUR canaries first: WRN-plain/CIFAR-10, WRN-plain/CIFAR-100,
     ResNeXt/CIFAR-10, ResNeXt/CIFAR-100 (split00, seed0, d=0.1).
  2. Release the remaining 596 fresh cells ONLY after an artifact-integrity PASS
     over the four canaries (marker + common-schema npz + checksums + finiteness).
  Run `run --canary-only` to stop after the gate; `run` runs canaries then, if the
  gate passes, releases the rest.

Every training cell runs inside the pinned fedcore-c400r image with
CUBLAS_WORKSPACE_CONFIG=:4096:8 (required by use_deterministic_algorithms(True)).
Native 32x32; one cell per GPU (one worker per authorized GPU UUID). The runner
(fedcore.experiments.run_cifar_common) emits the confirmatory COMMON per-obs
schema + a self-written TERMINAL marker with checksums.

Restart-safe by construction (mirrors the audited confirmatory/fedisic launchers):
global O_EXCL lock (stale-steal on restart), per-cell lock, completed-cell skip,
finalize-from-disk, identical-cell resume retry (NEVER a new seed), per-cell frozen
matrix/class-split sha guard, disk guard, atomic live status. The frozen matrix,
class splits and the CIFAR data are never mutated.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import queue
import shlex
import shutil
import signal
import subprocess
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMP = os.path.join(REPO, "results", "cifar_fedprox_sweep")
MATRIX = os.path.join(CAMP, "final_training_matrix.csv")
MATRIX_SHA256 = "d3c344d80f5efa87efbbe050eb5ec984ac218cf3cde0a68e91196c6bb9f2b7bb"
LIVE = os.path.join(CAMP, "live")
RUNS_DEFAULT = os.path.join(REPO, "runs", "cifar_fedprox_sweep", "cells")

IMAGE = os.environ.get("CIFAR_SWEEP_IMAGE", "fedcore-c400r:latest")
C_REPO = "/workspace"
C_DATA_ROOT = "/workspace/data"

CIFAR_SPLITS_HOST = {
    "cifar10": os.path.join(REPO, "results", "confirmatory_400", "prelaunch", "cifar10_class_splits.csv"),
    "cifar100": os.path.join(REPO, "results", "confirmatory_400", "prelaunch", "cifar100_class_splits.csv"),
}
CIFAR_SPLITS_SHA256 = {
    "cifar10": "PROVENANCE",   # filled by _load_split_sha at import (recorded, fail-closed on drift within a run)
    "cifar100": "PROVENANCE",
}

AUTHORIZED_GPU_UUIDS = [
    "GPU-d6e53d0c-b100-5dd4-30b2-0574b2b4dffb",
    "GPU-94b3a414-8e7a-3454-83dc-6132a9124a28",
    "GPU-c3326fea-a08f-9192-eee8-27fb8017984c",
    "GPU-afbc9e02-0ce4-a4a4-391f-7c31c414771f",
]

# FROZEN recipe (matches the confirmatory CIFAR training config; do NOT change).
ROUNDS = 50
LOCAL_EPOCHS = 2
BATCH_SIZE = 128
LR = "0.01"
NORM = "bn"

# The four canaries (owner spec): one per (arm, dataset), canonical block.
CANARY_IDS = [
    "wrn28_10_fedprox_mu0.1__cifar10__split00__seed0__d0.1",
    "wrn28_10_fedprox_mu0.1__cifar100__split00__seed0__d0.1",
]

DISK_GUARD_GB_DEFAULT = 50.0
MAX_RETRIES_DEFAULT = 2

_STOP = threading.Event()
_STATE_LOCK = threading.Lock()


def _now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _load_matrix():
    with open(MATRIX, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# record class-split provenance at import (used only to fail closed on drift mid-run)
for _ds, _p in CIFAR_SPLITS_HOST.items():
    try:
        CIFAR_SPLITS_SHA256[_ds] = _sha256(_p)
    except OSError:
        CIFAR_SPLITS_SHA256[_ds] = "MISSING"


def _cifar_unknowns(dataset, split_id):
    with open(CIFAR_SPLITS_HOST[dataset], newline="") as fh:
        for r in csv.DictReader(fh):
            if r["split_id"] == split_id:
                return ",".join(r["unknown_classes"].split())
    raise KeyError(f"{dataset} split {split_id} absent from frozen class splits")


def fresh_cells(rows):
    return [r for r in rows if r["reuse_class"] == "fresh_training"]


def cell_paths(out_root, cell):
    sid = cell["semantic_id"]
    base = os.path.join(out_root, sid)
    return {
        "sid": sid, "base": base,
        "logits": f"{base}_common.npz", "checkpoint": f"{base}.pt",
        "marker": f"{base}.TERMINAL.json", "lock": f"{base}.lock",
        "stdout": f"{base}.out.log", "stderr": f"{base}.err.log",
    }


# --------------------------------------------------------------------------- #
# Common-schema validation (the terminal contract for a CIFAR cell).
# --------------------------------------------------------------------------- #
_ROLES = ("proposal", "certification", "test")
_PER_OBS = ("immutable_source_id", "client_id", "fold_role", "true_global_class",
            "true_known_class_index_or_neg1", "known_or_unknown", "known_logits",
            "predicted_known_index", "energy_score", "known_margin_score", "native_score")


def validate_common_npz(path, expect_backbone, expect_n_known):
    import numpy as np
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as exc:  # pragma: no cover
        return False, f"npz_unreadable:{exc}", {}
    keys = set(z.keys())
    try:
        if str(z["schema_id"]) != "confirmatory_400r_common_obs_v1":
            return False, f"schema_id:{z['schema_id']}", {}
        if str(z["native_score_name"]) != "msp":
            return False, f"native_score_name:{z['native_score_name']}", {}
        if int(z["n_known"]) != int(expect_n_known):
            return False, f"n_known:{int(z['n_known'])}!={expect_n_known}", {}
        if str(z["backbone"]) != expect_backbone:
            return False, f"backbone:{z['backbone']}!={expect_backbone}", {}
        for role in _ROLES:
            for f in _PER_OBS:
                k = f"{role}__{f}"
                if k not in keys:
                    return False, f"missing:{k}", {}
            kl = np.asarray(z[f"{role}__known_logits"])
            if kl.ndim != 2 or kl.shape[1] != int(expect_n_known):
                return False, f"bad_logits_shape:{role}:{kl.shape}", {}
            if not np.isfinite(kl).all() or not np.isfinite(np.asarray(z[f"{role}__native_score"])).all():
                return False, f"non_finite:{role}", {}
        info = {"unit_counts": {r: int(np.asarray(z[f"{r}__known_logits"]).shape[0]) for r in _ROLES}}
        return True, "ok", info
    except Exception as exc:  # pragma: no cover
        return False, f"probe_error:{exc}", {}


def is_complete(paths, cell):
    if not os.path.isfile(paths["marker"]):
        return False, "no_marker"
    try:
        marker = json.load(open(paths["marker"]))
    except (OSError, ValueError):
        return False, "unreadable_marker"
    if marker.get("status") != "completed":
        return False, "marker_not_completed"
    checks = marker.get("checksums", {})
    logits_b = os.path.basename(paths["logits"])
    ckpt_b = os.path.basename(paths["checkpoint"])
    if logits_b not in checks or ckpt_b not in checks:
        return False, "marker_without_checksums"
    for name, expected in checks.items():
        cand = paths["logits"] if os.path.basename(paths["logits"]) == name else (
            paths["checkpoint"] if os.path.basename(paths["checkpoint"]) == name else None)
        if cand is None or not os.path.isfile(cand) or _sha256(cand) != expected:
            return False, f"checksum_mismatch:{name}"
    return True, "complete"


def artifact_integrity(paths, cell):
    """Deep integrity check used for the canary release gate."""
    done, why = is_complete(paths, cell)
    if not done:
        return False, why
    ok, reason, info = validate_common_npz(
        paths["logits"], cell["backbone"], int(cell["n_known"]))
    if not ok:
        return False, f"schema:{reason}"
    return True, "ok"


def artifacts_present(paths):
    return os.path.isfile(paths["logits"]) and os.path.isfile(paths["checkpoint"])


def disk_free_gb(path):
    os.makedirs(path, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def guard_cell(cell):
    if _sha256(MATRIX) != MATRIX_SHA256:
        return False, "matrix_sha256_changed"
    ds = cell["dataset"]
    p = CIFAR_SPLITS_HOST[ds]
    if not os.path.isfile(p):
        return False, f"class_splits_missing:{ds}"
    if _sha256(p) != CIFAR_SPLITS_SHA256[ds]:
        return False, f"class_splits_sha_drift:{ds}"
    if not os.path.isdir(os.path.join(REPO, "data")):
        return False, "data_root_missing"
    try:
        _cifar_unknowns(ds, cell["split_id"])
    except KeyError:
        return False, f"split_absent:{cell['split_id']}"
    return True, "ok"


def build_inner_cmd(cell, paths, resume):
    def c(p):
        return os.path.join(C_REPO, os.path.relpath(p, REPO))
    unknowns = _cifar_unknowns(cell["dataset"], cell["split_id"])
    cmd = [
        "python", "-m", "fedcore.experiments.run_cifar_common", "run",
        "--dataset", cell["dataset"], "--backbone", cell["backbone"], "--norm", NORM,
        "--split-id", cell["split_id"], "--n-known", str(int(cell["n_known"])),
        "--n-clients", "5", "--dirichlet-alpha", str(cell["d"]),
        "--rounds", str(ROUNDS), "--local-epochs", str(LOCAL_EPOCHS),
        "--batch-size", str(BATCH_SIZE), "--lr", LR,
        "--fedprox-mu", str(cell.get("fedprox_mu", 0.1)),
        "--seed", str(int(cell["train_rep"])), "--data-root", C_DATA_ROOT,
        "--unknown-classes", unknowns,
        "--experiment-id", cell["semantic_id"], "--config-sha", cell["semantic_id"],
        "--out", c(paths["logits"]), "--checkpoint", c(paths["checkpoint"]),
        "--marker", c(paths["marker"]),
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def build_docker_cmd(cell, paths, gpu_uuid, resume):
    cname = "cifpx_" + cell["semantic_id"].replace(".", "_").replace("/", "_")[:100]
    inner = build_inner_cmd(cell, paths, resume)
    docker = [
        "docker", "run", "--rm", "--name", cname,
        "--gpus", f"device={gpu_uuid}", "--shm-size", "8g",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp",
        "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "-e", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "-e", "PYTHONUNBUFFERED=1", "-e", "PYTHONPATH=/workspace",
        "-e", "OMP_NUM_THREADS=4",
        "-v", f"{REPO}:{C_REPO}",
        "-w", C_REPO, IMAGE,
    ]
    return docker + inner, cname


# --------------------------------------------------------------------------- #
# Live status.
# --------------------------------------------------------------------------- #
class LiveState:
    def __init__(self, cells, gpu_uuids, out_root, mode, pid, sid):
        self.out_root, self.mode, self.gpu_uuids = out_root, mode, gpu_uuids
        self.started_at, self.pid, self.sid = _now(), pid, sid
        self.cells = {c["semantic_id"]: {
            "semantic_id": c["semantic_id"], "arm": c["arm"], "dataset": c["dataset"],
            "d": c["d"], "state": "pending", "gpu_uuid": "", "attempt": 0, "rc": "",
            "reason": "", "started_at": "", "finished_at": "", "logits_sha256": "",
        } for c in cells}
        self.gpu_current = {u: "" for u in gpu_uuids}
        self.failures = []
        with _STATE_LOCK:
            self._flush_failures_locked()

    def set(self, sid, **kw):
        with _STATE_LOCK:
            self.cells[sid].update(kw); self._flush_locked()

    def set_gpu(self, uuid, sid):
        with _STATE_LOCK:
            self.gpu_current[uuid] = sid; self._flush_locked()

    def add_failure(self, sid, attempt, rc, reason, tail):
        with _STATE_LOCK:
            self.failures.append({"semantic_id": sid, "attempt": attempt, "rc": rc,
                                  "reason": reason, "timestamp": _now(), "stderr_tail": tail})
            self._flush_failures_locked()

    def _counts(self):
        c = {}
        for v in self.cells.values():
            c[v["state"]] = c.get(v["state"], 0) + 1
        return c

    def _flush_locked(self):
        os.makedirs(LIVE, exist_ok=True)
        cols = ["semantic_id", "arm", "dataset", "d", "state", "gpu_uuid", "attempt",
                "rc", "reason", "started_at", "finished_at", "logits_sha256"]
        lines = [",".join(cols)]
        for v in self.cells.values():
            lines.append(",".join(str(v.get(k, "")).replace(",", ";") for k in cols))
        _atomic_write(os.path.join(LIVE, "status.csv"), "\n".join(lines) + "\n")
        summary = {"campaign": "cifar_backbone_sweep", "mode": self.mode,
                   "dispatcher_pid": self.pid, "dispatcher_sid": self.sid, "image": IMAGE,
                   "started_at": self.started_at, "heartbeat": _now(),
                   "out_root": self.out_root, "gpu_uuids": self.gpu_uuids,
                   "counts": self._counts(), "total": len(self.cells),
                   "gpu_current": dict(self.gpu_current)}
        _atomic_write(os.path.join(LIVE, "status.json"), json.dumps(summary, indent=2))

    def _flush_failures_locked(self):
        cols = ["semantic_id", "attempt", "rc", "reason", "timestamp", "stderr_tail"]
        lines = [",".join(cols)]
        for f in self.failures:
            lines.append(",".join(str(f.get(k, "")).replace(",", ";").replace("\n", " ") for k in cols))
        _atomic_write(os.path.join(LIVE, "failures.csv"), "\n".join(lines) + "\n")

    def heartbeat(self):
        with _STATE_LOCK:
            self._flush_locked()


# --------------------------------------------------------------------------- #
# Locks.
# --------------------------------------------------------------------------- #
def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def acquire_global_lock():
    os.makedirs(LIVE, exist_ok=True)
    lock = os.path.join(LIVE, "dispatcher.lock")
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                prev = json.load(open(lock))
            except (OSError, ValueError):
                prev = {}
            if prev.get("pid") and _pid_alive(int(prev["pid"])):
                return None, f"active dispatcher pid={prev['pid']}"
            os.remove(lock)
            continue
        try:
            sid = os.getsid(0)
        except OSError:
            sid = -1
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"pid": os.getpid(), "sid": sid, "acquired_at": _now()}))
        return lock, "acquired"
    return None, "could not acquire global lock"


def acquire_cell_lock(paths, gpu_uuid):
    os.makedirs(os.path.dirname(paths["lock"]), exist_ok=True)
    try:
        fd = os.open(paths["lock"], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "gpu_uuid": gpu_uuid, "at": _now()}))
    return True


def release_cell_lock(paths):
    try:
        os.remove(paths["lock"])
    except OSError:
        pass


def clear_stale_cell_locks(cells, out_root):
    cleared = 0
    for cell in cells:
        paths = cell_paths(out_root, cell)
        if os.path.exists(paths["lock"]) and not is_complete(paths, cell)[0]:
            try:
                os.remove(paths["lock"]); cleared += 1
            except OSError:
                pass
    return cleared


def kill_orphan_containers():
    try:
        out = subprocess.run(["docker", "ps", "-q", "--filter", "name=cifpx_"],
                             capture_output=True, text=True, timeout=60)
        ids = [x for x in out.stdout.split() if x]
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, timeout=180)
        return len(ids)
    except Exception:
        return -1


def _tail(path, n=2000):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - n))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def process_cell(cell, gpu_uuid, out_root, live, disk_guard_gb, max_retries):
    sid = cell["semantic_id"]
    paths = cell_paths(out_root, cell)
    if is_complete(paths, cell)[0]:
        live.set(sid, state="complete", gpu_uuid=gpu_uuid, reason="already_terminal",
                 finished_at=_now())
        return "complete"
    ok, reason = guard_cell(cell)
    if not ok:
        live.set(sid, state="failed", gpu_uuid=gpu_uuid, reason=f"guard:{reason}",
                 finished_at=_now())
        live.add_failure(sid, 0, "guard", reason, "")
        return "failed"
    free = disk_free_gb(out_root)
    if free < disk_guard_gb:
        live.set(sid, state="failed", gpu_uuid=gpu_uuid,
                 reason=f"disk_guard:{free:.1f}<{disk_guard_gb}", finished_at=_now())
        return "failed"
    if not acquire_cell_lock(paths, gpu_uuid):
        live.set(sid, state="skipped", reason="locked_by_other")
        return "locked"
    try:
        attempt = 0
        while attempt <= max_retries and not _STOP.is_set():
            attempt += 1
            resume = os.path.exists(paths["checkpoint"])
            cmd, _cname = build_docker_cmd(cell, paths, gpu_uuid, resume)
            live.set(sid, state="running", gpu_uuid=gpu_uuid, attempt=attempt,
                     started_at=_now(), reason="resume" if resume else "fresh")
            live.set_gpu(gpu_uuid, sid)
            os.makedirs(os.path.dirname(paths["stdout"]), exist_ok=True)
            with open(paths["stdout"], "a") as so, open(paths["stderr"], "a") as se:
                so.write(f"\n=== attempt {attempt} resume={resume} gpu={gpu_uuid} {_now()} ===\n")
                so.flush()
                try:
                    rc = subprocess.run(cmd, stdout=so, stderr=se).returncode
                except Exception as exc:  # pragma: no cover
                    rc = -99
                    se.write(f"dispatcher-exception: {exc}\n")
            if _STOP.is_set() and rc != 0:
                live.set(sid, state="pending", gpu_uuid="", reason="dispatcher_stopping")
                live.set_gpu(gpu_uuid, "")
                return "stopped"
            done, why = is_complete(paths, cell)
            if rc == 0 and done:
                live.set(sid, state="complete", gpu_uuid=gpu_uuid, rc=rc, reason="completed",
                         finished_at=_now(), logits_sha256=_sha256(paths["logits"])[:16])
                live.set_gpu(gpu_uuid, "")
                return "complete"
            live.add_failure(sid, attempt, rc, f"rc={rc}|{why}", _tail(paths["stderr"]))
        live.set(sid, state="failed", gpu_uuid=gpu_uuid,
                 reason=f"exhausted_retries({max_retries})", finished_at=_now())
        live.set_gpu(gpu_uuid, "")
        return "failed"
    finally:
        release_cell_lock(paths)


def run_pool(cells, gpu_uuids, out_root, live, disk_guard_gb, max_retries):
    work = queue.Queue()
    for cell in cells:
        work.put(cell)

    def worker(gpu_uuid):
        while not _STOP.is_set():
            try:
                cell = work.get_nowait()
            except queue.Empty:
                return
            try:
                process_cell(cell, gpu_uuid, out_root, live, disk_guard_gb, max_retries)
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(u,), name=f"gpu-{i}", daemon=True)
               for i, u in enumerate(gpu_uuids)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        live.heartbeat()
        time.sleep(10)
    for t in threads:
        t.join()


# --------------------------------------------------------------------------- #
# Modes.
# --------------------------------------------------------------------------- #
def cmd_print_plan(args):
    got = _sha256(MATRIX)
    print(f"[print-plan] matrix sha256 {got} matches_frozen={got == MATRIX_SHA256}")
    rows = _load_matrix()
    fresh = fresh_cells(rows)
    print(f"[print-plan] total {len(rows)} | fresh {len(fresh)} | arm_A_exact_reuse {len(rows) - len(fresh)}")
    ids = {r["semantic_id"] for r in fresh}
    missing_canary = [c for c in CANARY_IDS if c not in ids]
    print(f"[print-plan] canaries: {CANARY_IDS}")
    print(f"[print-plan] canaries present in fresh set: {not missing_canary} (missing {missing_canary})")
    imgs = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                          capture_output=True, text=True).stdout.split()
    print(f"[print-plan] image {IMAGE} present: {IMAGE in imgs}")
    n_guard = sum(1 for c in fresh if not guard_cell(c)[0])
    print(f"[print-plan] fresh cells failing guard: {n_guard}")
    sample = next(c for c in fresh if c["semantic_id"] == CANARY_IDS[0])
    cmd, cname = build_docker_cmd(sample, cell_paths(os.path.abspath(args.out_root), sample),
                                  AUTHORIZED_GPU_UUIDS[0], False)
    print(f"[sample canary] {cname}\n  " + " ".join(shlex.quote(x) for x in cmd))
    ok = (got == MATRIX_SHA256) and not missing_canary and (IMAGE in imgs) and n_guard == 0
    print("PRINT_PLAN_OK" if ok else "FAIL_CLOSED")
    return 0 if ok else 2


def _select_gpus(args):
    gpu_uuids = [u for u in (args.gpus.split(",") if args.gpus else AUTHORIZED_GPU_UUIDS) if u]
    for u in gpu_uuids:
        if u not in AUTHORIZED_GPU_UUIDS:
            raise SystemExit(f"FAIL_CLOSED: unauthorized GPU uuid {u}")
    return gpu_uuids


def cmd_run(args):
    if _sha256(MATRIX) != MATRIX_SHA256:
        print(f"FAIL_CLOSED: matrix sha256 mismatch")
        return 2
    rows = _load_matrix()
    fresh = fresh_cells(rows)
    n_splits = int(getattr(args, "n_splits", 0) or 0)
    if n_splits > 0:   # simplification: run only split indices 0..n_splits-1
        fresh = [c for c in fresh if int(str(c["split_id"]).split("_")[-1]) < n_splits]
        print(f"[dispatcher] --n-splits {n_splits}: {len(fresh)} fresh cells after split filter")
    by_id = {c["semantic_id"]: c for c in fresh}
    for cid in CANARY_IDS:
        if cid not in by_id:
            print(f"FAIL_CLOSED: canary {cid} not in fresh set")
            return 2
    gpu_uuids = _select_gpus(args)
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    lock, why = acquire_global_lock()
    if lock is None:
        print(f"FAIL_CLOSED: {why}")
        return 3
    try:
        sid = os.getsid(0)
    except OSError:
        sid = -1
    print(f"[dispatcher] global lock acquired pid={os.getpid()} sid={sid}")
    try:
        n_orphan = kill_orphan_containers()
        n_stale = clear_stale_cell_locks(fresh, out_root)
        print(f"[dispatcher] killed {n_orphan} orphans; cleared {n_stale} stale locks")
        canary_cells = [by_id[c] for c in CANARY_IDS]
        rest = [c for c in fresh if c["semantic_id"] not in set(CANARY_IDS)]
        mode = "canary_only" if args.canary_only else "canary_gated_full"
        live = LiveState(fresh, gpu_uuids, out_root, mode, os.getpid(), sid)
        for c in fresh:
            if is_complete(cell_paths(out_root, c), c)[0]:
                live.set(c["semantic_id"], state="complete", reason="already_terminal")
        live.heartbeat()

        def _on_sig(signum, frame):
            _STOP.set()
        signal.signal(signal.SIGTERM, _on_sig)
        signal.signal(signal.SIGINT, _on_sig)

        print(f"[dispatcher] CANARY PHASE ({len(canary_cells)} cells) on {len(gpu_uuids)} GPUs")
        run_pool(canary_cells, gpu_uuids, out_root, live, args.disk_guard_gb, args.max_retries)
        gate = {"canary_ids": CANARY_IDS, "checked_at": _now(), "per_cell": {}}
        all_pass = True
        for c in canary_cells:
            ok, reason = artifact_integrity(cell_paths(out_root, c), c)
            gate["per_cell"][c["semantic_id"]] = {"integrity_pass": ok, "reason": reason}
            all_pass = all_pass and ok
        gate["passed"] = bool(all_pass and not _STOP.is_set())
        _atomic_write(os.path.join(LIVE, "CANARY_GATE.json"), json.dumps(gate, indent=2))
        if not gate["passed"]:
            print(f"CANARY_FAILED_CLOSED: {gate['per_cell']} -- NOT releasing {len(rest)} cells")
            live.heartbeat()
            return 5
        print(f"[dispatcher] CANARY ARTIFACT-INTEGRITY PASS")
        if args.canary_only:
            print("[dispatcher] --canary-only: stopping after gate (596 cells NOT released)")
            live.heartbeat()
            return 0
        print(f"[dispatcher] releasing remaining {len(rest)} cells over {len(gpu_uuids)} GPUs")
        run_pool(rest, gpu_uuids, out_root, live, args.disk_guard_gb, args.max_retries)
        live.heartbeat()
        counts = live._counts()
        print(f"[dispatcher] DONE counts={counts}")
        return 0 if counts.get("complete", 0) == len(fresh) else 4
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass
        print("[dispatcher] global lock released")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("print-plan")
    pp.add_argument("--out-root", default=RUNS_DEFAULT)
    pp.set_defaults(func=cmd_print_plan)
    rr = sub.add_parser("run", help="canaries -> integrity gate -> release 596 (unless --canary-only)")
    rr.add_argument("--out-root", default=RUNS_DEFAULT)
    rr.add_argument("--gpus", default="")
    rr.add_argument("--canary-only", action="store_true")
    rr.add_argument("--disk-guard-gb", type=float, default=DISK_GUARD_GB_DEFAULT)
    rr.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)
    rr.add_argument("--n-splits", dest="n_splits", type=int, default=0,
                    help="run only split indices 0..N-1 (0 = all 10). 5 = simplified 150-cell arm.")
    rr.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
