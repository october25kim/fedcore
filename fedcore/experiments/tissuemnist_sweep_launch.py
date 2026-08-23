#!/usr/bin/env python3
"""MedMNIST backbone-sweep dispatcher: restart-safe, one cell per GPU.

Trains all 100 fresh cells of results/medmnist_sweep (WRN-28-10 + ResNeXt-29 plain
FedAvg on PathMNIST, 28x28). Every cell runs inside the pinned fedcore-c400r image
with CUBLAS_WORKSPACE_CONFIG=:4096:8; the runner
(fedcore.experiments.run_medmnist_common) emits the confirmatory COMMON per-obs
schema + a self-written TERMINAL marker with checksums. Unknown classes come from
the matrix cell (no external class-split file). Restart-safe: global O_EXCL lock
(stale-steal), per-cell lock, completed-cell skip, identical-cell resume retry,
per-cell frozen matrix sha guard, disk guard, atomic live status.
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
CAMP = os.path.join(REPO, "results", "tissuemnist_sweep")
MATRIX = os.path.join(CAMP, "final_training_matrix.csv")
MATRIX_SHA256 = "947b919e6bbe9484f85eee3b80095ff0549bfc46db7499e45040838c5c6c8c31"
LIVE = os.path.join(CAMP, "live")
RUNS_DEFAULT = os.path.join(REPO, "runs", "tissuemnist_sweep", "cells")
DATA_NPZ = os.path.join(REPO, "data", "medmnist", "tissuemnist.npz")

IMAGE = os.environ.get("TISSUEMNIST_IMAGE", "fedcore-c400r:latest")
C_REPO = "/workspace"

AUTHORIZED_GPU_UUIDS = [
    "GPU-d6e53d0c-b100-5dd4-30b2-0574b2b4dffb",
    "GPU-94b3a414-8e7a-3454-83dc-6132a9124a28",
    "GPU-c3326fea-a08f-9192-eee8-27fb8017984c",
    "GPU-afbc9e02-0ce4-a4a4-391f-7c31c414771f",
]

# FROZEN recipe (confirmation-validated: acc 0.896 / AUROC 0.850 at these settings).
ROUNDS = 25
LOCAL_EPOCHS = 2
BATCH_SIZE = 128
LR = "0.01"
NORM = "bn"
DISK_GUARD_GB_DEFAULT = 30.0
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
        fh.write(text); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def _load_matrix():
    with open(MATRIX, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def cell_paths(out_root, cell):
    sid = cell["semantic_id"]
    base = os.path.join(out_root, sid)
    return {"sid": sid, "base": base, "logits": f"{base}_common.npz",
            "checkpoint": f"{base}.pt", "marker": f"{base}.TERMINAL.json",
            "lock": f"{base}.lock", "stdout": f"{base}.out.log", "stderr": f"{base}.err.log"}


_ROLES = ("proposal", "certification", "test")
_PER_OBS = ("immutable_source_id", "client_id", "fold_role", "true_global_class",
            "true_known_class_index_or_neg1", "known_or_unknown", "known_logits",
            "predicted_known_index", "energy_score", "known_margin_score", "native_score")


def validate_common_npz(path, backbone, n_known):
    import numpy as np
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as exc:
        return False, f"npz_unreadable:{exc}"
    keys = set(z.keys())
    try:
        if str(z["schema_id"]) != "confirmatory_400r_common_obs_v1":
            return False, "schema_id"
        if str(z["native_score_name"]) != "msp" or int(z["n_known"]) != int(n_known):
            return False, "meta"
        if str(z["backbone"]) != backbone:
            return False, f"backbone:{z['backbone']}"
        for role in _ROLES:
            for f in _PER_OBS:
                if f"{role}__{f}" not in keys:
                    return False, f"missing:{role}__{f}"
            kl = np.asarray(z[f"{role}__known_logits"])
            if kl.ndim != 2 or kl.shape[1] != int(n_known) or not np.isfinite(kl).all():
                return False, f"logits:{role}"
        return True, "ok"
    except Exception as exc:
        return False, f"probe:{exc}"


def is_complete(paths, cell):
    if not os.path.isfile(paths["marker"]):
        return False, "no_marker"
    try:
        m = json.load(open(paths["marker"]))
    except (OSError, ValueError):
        return False, "unreadable_marker"
    if m.get("status") != "completed":
        return False, "not_completed"
    checks = m.get("checksums", {})
    lb, cb = os.path.basename(paths["logits"]), os.path.basename(paths["checkpoint"])
    if lb not in checks or cb not in checks:
        return False, "no_checksums"
    for name, exp in checks.items():
        cand = paths["logits"] if os.path.basename(paths["logits"]) == name else (
            paths["checkpoint"] if os.path.basename(paths["checkpoint"]) == name else None)
        if cand is None or not os.path.isfile(cand) or _sha256(cand) != exp:
            return False, f"mismatch:{name}"
    ok, why = validate_common_npz(paths["logits"], cell["backbone"], int(cell["n_known"]))
    return (ok, "complete" if ok else f"schema:{why}")


def disk_free_gb(path):
    os.makedirs(path, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def guard_cell(cell):
    if _sha256(MATRIX) != MATRIX_SHA256:
        return False, "matrix_sha256_changed"
    if not os.path.isfile(DATA_NPZ):
        return False, "tissuemnist_npz_missing"
    return True, "ok"


def build_inner_cmd(cell, paths, resume):
    def c(p):
        return os.path.join(C_REPO, os.path.relpath(p, REPO))
    cmd = [
        "python", "-m", "fedcore.experiments.run_medmnist_common", "run",
        "--dataset", cell["dataset"], "--backbone", cell["backbone"], "--norm", NORM,
        "--split-id", cell["split_id"], "--n-known", str(int(cell["n_known"])),
        "--n-clients", str(int(cell["n_clients"])), "--dirichlet-alpha", str(cell["d"]),
        "--rounds", str(ROUNDS), "--local-epochs", str(LOCAL_EPOCHS),
        "--batch-size", str(BATCH_SIZE), "--lr", LR,
        "--seed", str(int(cell["train_rep"])), "--data-root", "data/medmnist",
        "--unknown-classes", cell["unknown_classes"],
        "--experiment-id", cell["semantic_id"], "--config-sha", cell["semantic_id"],
        "--out", c(paths["logits"]), "--checkpoint", c(paths["checkpoint"]),
        "--marker", c(paths["marker"]),
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def build_docker_cmd(cell, paths, gpu_uuid, resume):
    cname = "tiss_" + cell["semantic_id"].replace(".", "_").replace("/", "_")[:100]
    inner = build_inner_cmd(cell, paths, resume)
    docker = [
        "docker", "run", "--rm", "--name", cname, "--gpus", f"device={gpu_uuid}",
        "--shm-size", "8g", "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp", "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "-e", "CUBLAS_WORKSPACE_CONFIG=:4096:8", "-e", "PYTHONUNBUFFERED=1",
        "-e", "PYTHONPATH=/workspace", "-e", "OMP_NUM_THREADS=4",
        "-v", f"{REPO}:{C_REPO}", "-w", C_REPO, IMAGE,
    ]
    return docker + inner, cname


class LiveState:
    def __init__(self, cells, gpu_uuids, out_root, pid, sid):
        self.out_root, self.gpu_uuids, self.pid, self.sid = out_root, gpu_uuids, pid, sid
        self.started_at = _now()
        self.cells = {c["semantic_id"]: {"semantic_id": c["semantic_id"], "backbone": c["backbone"],
                      "split_id": c["split_id"], "d": c["d"], "state": "pending", "gpu_uuid": "",
                      "attempt": 0, "rc": "", "reason": "", "started_at": "", "finished_at": "",
                      "logits_sha256": ""} for c in cells}
        self.gpu_current = {u: "" for u in gpu_uuids}
        self.failures = []
        with _STATE_LOCK:
            self._flush_failures_locked()

    def set(self, sid, **kw):
        with _STATE_LOCK:
            self.cells[sid].update(kw); self._flush_locked()

    def set_gpu(self, u, sid):
        with _STATE_LOCK:
            self.gpu_current[u] = sid; self._flush_locked()

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
        cols = ["semantic_id", "backbone", "split_id", "d", "state", "gpu_uuid", "attempt",
                "rc", "reason", "started_at", "finished_at", "logits_sha256"]
        lines = [",".join(cols)]
        for v in self.cells.values():
            lines.append(",".join(str(v.get(k, "")).replace(",", ";") for k in cols))
        _atomic_write(os.path.join(LIVE, "status.csv"), "\n".join(lines) + "\n")
        _atomic_write(os.path.join(LIVE, "status.json"), json.dumps(
            {"campaign": "tissuemnist_sweep", "dispatcher_pid": self.pid, "dispatcher_sid": self.sid,
             "image": IMAGE, "started_at": self.started_at, "heartbeat": _now(),
             "out_root": self.out_root, "gpu_uuids": self.gpu_uuids, "counts": self._counts(),
             "total": len(self.cells), "gpu_current": dict(self.gpu_current)}, indent=2))

    def _flush_failures_locked(self):
        cols = ["semantic_id", "attempt", "rc", "reason", "timestamp", "stderr_tail"]
        lines = [",".join(cols)]
        for f in self.failures:
            lines.append(",".join(str(f.get(k, "")).replace(",", ";").replace("\n", " ") for k in cols))
        _atomic_write(os.path.join(LIVE, "failures.csv"), "\n".join(lines) + "\n")

    def heartbeat(self):
        with _STATE_LOCK:
            self._flush_locked()


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
            os.remove(lock); continue
        try:
            sid = os.getsid(0)
        except OSError:
            sid = -1
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"pid": os.getpid(), "sid": sid, "acquired_at": _now()}))
        return lock, "acquired"
    return None, "could not acquire lock"


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
    n = 0
    for cell in cells:
        paths = cell_paths(out_root, cell)
        if os.path.exists(paths["lock"]) and not is_complete(paths, cell)[0]:
            try:
                os.remove(paths["lock"]); n += 1
            except OSError:
                pass
    return n


def kill_orphan_containers():
    try:
        out = subprocess.run(["docker", "ps", "-q", "--filter", "name=tiss_"],
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
        live.set(sid, state="complete", gpu_uuid=gpu_uuid, reason="already_terminal", finished_at=_now())
        return "complete"
    ok, reason = guard_cell(cell)
    if not ok:
        live.set(sid, state="failed", gpu_uuid=gpu_uuid, reason=f"guard:{reason}", finished_at=_now())
        live.add_failure(sid, 0, "guard", reason, "")
        return "failed"
    if disk_free_gb(out_root) < disk_guard_gb:
        live.set(sid, state="failed", gpu_uuid=gpu_uuid, reason="disk_guard", finished_at=_now())
        return "failed"
    if not acquire_cell_lock(paths, gpu_uuid):
        live.set(sid, state="skipped", reason="locked_by_other")
        return "locked"
    try:
        attempt = 0
        while attempt <= max_retries and not _STOP.is_set():
            attempt += 1
            resume = os.path.exists(paths["checkpoint"])
            cmd, _c = build_docker_cmd(cell, paths, gpu_uuid, resume)
            live.set(sid, state="running", gpu_uuid=gpu_uuid, attempt=attempt,
                     started_at=_now(), reason="resume" if resume else "fresh")
            live.set_gpu(gpu_uuid, sid)
            os.makedirs(os.path.dirname(paths["stdout"]), exist_ok=True)
            with open(paths["stdout"], "a") as so, open(paths["stderr"], "a") as se:
                so.write(f"\n=== attempt {attempt} resume={resume} gpu={gpu_uuid} {_now()} ===\n"); so.flush()
                try:
                    rc = subprocess.run(cmd, stdout=so, stderr=se).returncode
                except Exception as exc:
                    rc = -99; se.write(f"dispatcher-exception: {exc}\n")
            if _STOP.is_set() and rc != 0:
                live.set(sid, state="pending", gpu_uuid="", reason="dispatcher_stopping")
                live.set_gpu(gpu_uuid, ""); return "stopped"
            done, why = is_complete(paths, cell)
            if rc == 0 and done:
                live.set(sid, state="complete", gpu_uuid=gpu_uuid, rc=rc, reason="completed",
                         finished_at=_now(), logits_sha256=_sha256(paths["logits"])[:16])
                live.set_gpu(gpu_uuid, ""); return "complete"
            live.add_failure(sid, attempt, rc, f"rc={rc}|{why}", _tail(paths["stderr"]))
        live.set(sid, state="failed", gpu_uuid=gpu_uuid, reason=f"exhausted({max_retries})", finished_at=_now())
        live.set_gpu(gpu_uuid, ""); return "failed"
    finally:
        release_cell_lock(paths)


def run_pool(cells, gpu_uuids, out_root, live, disk_guard_gb, max_retries):
    work = queue.Queue()
    for c in cells:
        work.put(c)

    def worker(u):
        while not _STOP.is_set():
            try:
                cell = work.get_nowait()
            except queue.Empty:
                return
            try:
                process_cell(cell, u, out_root, live, disk_guard_gb, max_retries)
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(u,), name=f"gpu-{i}", daemon=True)
               for i, u in enumerate(gpu_uuids)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        live.heartbeat(); time.sleep(10)
    for t in threads:
        t.join()


def cmd_print_plan(args):
    got = _sha256(MATRIX)
    print(f"[print-plan] matrix sha256 {got} matches_frozen={got == MATRIX_SHA256}")
    rows = _load_matrix()
    print(f"[print-plan] cells {len(rows)} | wrn {sum(1 for r in rows if r['backbone']=='wrn28_10')} "
          f"| resnext {sum(1 for r in rows if r['backbone']=='resnext29_8x64d')}")
    imgs = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                          capture_output=True, text=True).stdout.split()
    print(f"[print-plan] image {IMAGE} present: {IMAGE in imgs} | tissuemnist.npz: {os.path.isfile(DATA_NPZ)}")
    nfail = sum(1 for c in rows if not guard_cell(c)[0])
    print(f"[print-plan] cells failing guard: {nfail}")
    s = rows[0]
    cmd, cname = build_docker_cmd(s, cell_paths(os.path.abspath(args.out_root), s), AUTHORIZED_GPU_UUIDS[0], False)
    print(f"[sample] {cname}\n  " + " ".join(shlex.quote(x) for x in cmd))
    ok = got == MATRIX_SHA256 and IMAGE in imgs and os.path.isfile(DATA_NPZ) and nfail == 0
    print("PRINT_PLAN_OK" if ok else "FAIL_CLOSED")
    return 0 if ok else 2


def cmd_run(args):
    if _sha256(MATRIX) != MATRIX_SHA256:
        print("FAIL_CLOSED: matrix sha mismatch"); return 2
    cells = _load_matrix()
    gpu_uuids = [u for u in (args.gpus.split(",") if args.gpus else AUTHORIZED_GPU_UUIDS) if u]
    for u in gpu_uuids:
        if u not in AUTHORIZED_GPU_UUIDS:
            print(f"FAIL_CLOSED: unauthorized GPU {u}"); return 2
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    lock, why = acquire_global_lock()
    if lock is None:
        print(f"FAIL_CLOSED: {why}"); return 3
    try:
        sid = os.getsid(0)
    except OSError:
        sid = -1
    print(f"[dispatcher] global lock acquired pid={os.getpid()} sid={sid}")
    try:
        n_o = kill_orphan_containers()
        n_s = clear_stale_cell_locks(cells, out_root)
        print(f"[dispatcher] killed {n_o} orphans; cleared {n_s} stale locks")
        live = LiveState(cells, gpu_uuids, out_root, os.getpid(), sid)
        for c in cells:
            if is_complete(cell_paths(out_root, c), c)[0]:
                live.set(c["semantic_id"], state="complete", reason="already_terminal")
        live.heartbeat()

        def _sig(s, f):
            _STOP.set()
        signal.signal(signal.SIGTERM, _sig); signal.signal(signal.SIGINT, _sig)
        print(f"[dispatcher] tissuemnist_sweep cells={len(cells)} gpus={len(gpu_uuids)}")
        run_pool(cells, gpu_uuids, out_root, live, args.disk_guard_gb, args.max_retries)
        live.heartbeat()
        counts = live._counts()
        print(f"[dispatcher] DONE counts={counts}")
        return 0 if counts.get("complete", 0) == len(cells) else 4
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass
        print("[dispatcher] global lock released")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("print-plan"); pp.add_argument("--out-root", default=RUNS_DEFAULT)
    pp.set_defaults(func=cmd_print_plan)
    rr = sub.add_parser("run")
    rr.add_argument("--out-root", default=RUNS_DEFAULT)
    rr.add_argument("--gpus", default="")
    rr.add_argument("--disk-guard-gb", type=float, default=DISK_GUARD_GB_DEFAULT)
    rr.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)
    rr.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
