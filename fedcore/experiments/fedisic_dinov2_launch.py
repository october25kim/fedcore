#!/usr/bin/env python3
"""Fed-ISIC ConvNeXt-Tiny 50-cell REAL-SUBMIT dispatcher (host-side, GPL-isolated via docker).

This is the actual-launch dispatcher for the owner-authorized
``FEDISIC_CONVNEXT_TINY_FULL_FEDAVG`` intervention (design_declaration
``fedisic_convnext_tiny_full_fedavg_intervention_v1``).  It runs on the HOST with
the standard library + numpy alone (no torch, no ``fedcore`` import), and executes
every training cell inside the pinned ``fedcore-c400r`` container.  The training
code (``fedcore.experiments.run_fed_isic``) is touched only as a subprocess inside
the container.

Restart-safe by construction (mirrors the audited confirmatory_400r launcher):

* ONE global dispatcher lock (``O_CREAT|O_EXCL``); a second overlapping dispatcher
  refuses to start while the first is alive, and steals a stale lock on restart.
* per-cell atomic lock; on startup every stale per-cell lock of a non-terminal
  cell is cleared (safe, because the global lock guarantees single ownership).
* completion = terminal marker + rc0 + checkpoint + logits + embeddings +
  source-IDs + the 3 support statuses + the marker's recorded sha256 checksums
  matching the on-disk artifacts.  A bare checkpoint is NOT complete.
* completed-cell skip: a terminal cell is never rerun.
* finalize: a cell whose logits+checkpoint are on disk but whose marker is
  missing/invalid is finalized (schema re-validated, marker (re)written from
  on-disk checksums) WITHOUT retraining -- covers a crash between artifact write
  and marker write.
* identical-cell technical retry (SAME frozen seeds/fold/heldout, resume from
  checkpoint); NEVER a new seed, NEVER a reserve-seed substitution.
* per-cell frozen-input sha256 guard (matrix + metadata + fold) before every launch.
* disk-space guard before every launch.
* per-cell stdout/stderr/status; live/{status.csv,status.json,gpu_assignments.csv,
  failures.csv} maintained atomically.

The frozen matrix, folds, metadata and the read-only repo mount are never mutated
by this file.  Outputs land only under ``runs/fedisic_convnext_corrected`` (rw
mount) and ``results/fedisic_convnext_corrected/live`` (host).

CORRECTED sibling of ``fedisic_convnext_launch.py``: same 50-cell frozen matrix,
same folds/held-out/reps, same certification-facing eval transform, BUT the
training recipe removes the two audit confounds -- ``--loss focal --focal-gamma 2``
(FLamby imbalance-aware weighted focal) and ``--transform
convnext-imagenet-mildcrop`` (lesion-preserving Resize(256)+RandomCrop(224)).
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

import numpy as np

# --------------------------------------------------------------------------- #
# Frozen locations and invariants (host paths).
# --------------------------------------------------------------------------- #
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMP = os.path.join(REPO, "results", "fedisic_convnext_corrected")
MATRIX = os.path.join(CAMP, "final_training_matrix.csv")
MATRIX_SHA256 = "3550089b288840ed68a1f9457c22de97125dd6462e4ce0b251ac2d473adaa9f2"
LAUNCH_SNAPSHOT = os.path.join(CAMP, "launch", "LAUNCH_SNAPSHOT.json")
FOLD_CHECKSUMS = os.path.join(CAMP, "prelaunch", "fold_checksums.sha256")

# Output roots (all under the repo mount so container writes persist past --rm).
INTERVENTION_RUNS = os.path.join(REPO, "runs", "fedisic_convnext_corrected")
RUNS_DEFAULT = os.path.join(INTERVENTION_RUNS, "cells")
LIVE = os.path.join(CAMP, "live")

# Container image (built once from pytorch/pytorch:2.3.0-cuda12.1-cudnn8 + deps).
IMAGE = os.environ.get("FIC_IMAGE", "fedcore-c400r:latest")

# Frozen data inputs (host paths; also the in-container repo-relative paths since
# the repo is bind-mounted at /repo).
METADATA_REL = "results/source_data/fed_isic2019_metadata.csv"
METADATA_SHA256 = "2460578b6f4c46fa475f02114c19894519ffd251677ecca6f63117a669525e7c"
IMAGE_ROOT_REL = "data/isic2019/ISIC_2019_Training_Input_preprocessed"
TORCH_CACHE_HOST = os.path.join(REPO, "data", "officehome", "torch_cache")

C_REPO = "/repo"          # repo mounted read-only here
C_TORCH_HOME = "/tc"      # torch cache mounted read-only here

AUTHORIZED_GPU_UUIDS = [
    "GPU-d6e53d0c-b100-5dd4-30b2-0574b2b4dffb",
    "GPU-94b3a414-8e7a-3454-83dc-6132a9124a28",
    "GPU-c3326fea-a08f-9192-eee8-27fb8017984c",
    "GPU-afbc9e02-0ce4-a4a4-391f-7c31c414771f",
]

# FROZEN recipe (corrected design_declaration recipe block; do NOT change).
# Identical to the original intervention EXCEPT the two audit-confound fixes:
#   loss:      ce   -> focal (FLamby weighted focal, gamma=2)
#   transform: convnext-imagenet -> convnext-imagenet-mildcrop (lesion-preserving)
ROUNDS = 30
WARMUP_ROUNDS = 2
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LR = "1e-4"
WEIGHT_DECAY = "0.05"
IMAGE_SIZE = 224
LOSS = "focal"
FOCAL_GAMMA = "2"
TRANSFORM = "convnext-imagenet-mildcrop"

DISK_GUARD_GB_DEFAULT = 50.0
MAX_RETRIES_DEFAULT = 2

_STOP = threading.Event()
_STATE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Small helpers (pure stdlib).
# --------------------------------------------------------------------------- #
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


def _load_fold_checksums():
    table = {}
    with open(FOLD_CHECKSUMS) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split(maxsplit=1)
            table[rel.strip()] = digest
    return table


_FOLD_CHECKSUMS = _load_fold_checksums()


# --------------------------------------------------------------------------- #
# Cell derivation from the frozen matrix (NEVER from a mutable side file).
#   semantic_id = fedisic_convnext__{task_id}__seed{train_rep}
#   fold        = fed_isic2019_folds_split{task_id:02d}_seed{train_rep}.csv
#   heldout     = "A+B" -> "A,B" ; model-replicate = train_rep
# --------------------------------------------------------------------------- #
def cell_fold_rel(cell):
    t = int(cell["task_id"])
    r = int(cell["train_rep"])
    return f"results/source_data/fed_isic2019_folds_split{t:02d}_seed{r}.csv"


def cell_heldout_csv(cell):
    return cell["heldout_diagnosis"].replace("+", ",")


def cell_paths(out_root, cell):
    sid = cell["semantic_id"]
    base = os.path.join(out_root, sid)
    return {
        "sid": sid,
        "base": base,
        "logits": f"{base}_logits.npz",
        "checkpoint": f"{base}.pt",
        "marker": f"{base}.TERMINAL.json",
        "lock": f"{base}.lock",
        "status": f"{base}.status.json",
        "stdout": f"{base}.out.log",
        "stderr": f"{base}.err.log",
    }


# --------------------------------------------------------------------------- #
# Export-schema validation (the terminal contract).  Reads the npz written by
# run_fed_isic and confirms it carries, per fold, logits + 768-d ConvNeXt
# embeddings + opaque source-IDs, plus the 3 independent support statuses.
# Structural-failure cells are still terminally complete: a support status may be
# False (recorded), but the full schema must be present.
# --------------------------------------------------------------------------- #
_FOLDS = ("prop", "cert", "test")
_SUPPORT_KEYS = (
    "proposal_support_valid",
    "certification_a3_valid",
    "evaluation_unknown_metrics_defined",
)
_REASON_KEYS = (
    "proposal_support_reason",
    "certification_a3_reason",
    "evaluation_unknown_reason",
)


def validate_npz(logits_path):
    """Return (ok, reason, info). Fail-closed: any missing schema element -> not ok."""
    try:
        z = np.load(logits_path, allow_pickle=True)
    except Exception as exc:  # pragma: no cover - corrupt/partial file
        return False, f"npz_unreadable:{exc}", {}
    keys = set(z.keys())
    try:
        if str(z["backbone"]) != "convnext_tiny":
            return False, f"backbone_not_convnext:{z['backbone']}", {}
        if not bool(np.asarray(z["penultimate_embedding_included"]).item()):
            return False, "embedding_not_included", {}
        for f in _FOLDS:
            for suffix in ("logits", "embedding", "sample_id"):
                k = f"{f}_{suffix}"
                if k not in keys:
                    return False, f"missing_key:{k}", {}
            emb = np.asarray(z[f"{f}_embedding"])
            lg = np.asarray(z[f"{f}_logits"])
            sid = np.asarray(z[f"{f}_sample_id"])
            if emb.ndim != 2 or emb.shape[1] != 768:
                return False, f"bad_embedding_shape:{f}:{emb.shape}", {}
            if lg.shape[0] != emb.shape[0] or sid.shape[0] != emb.shape[0]:
                return False, f"row_misalignment:{f}", {}
            if not np.isfinite(emb).all() or not np.isfinite(lg).all():
                return False, f"non_finite:{f}", {}
        for k in _SUPPORT_KEYS:
            if k not in keys:
                return False, f"missing_support_status:{k}", {}
        statuses = {k: bool(np.asarray(z[k]).item()) for k in _SUPPORT_KEYS}
        reasons = {k: str(z[k]) for k in _REASON_KEYS if k in keys}
        info = {
            "support_statuses": statuses,
            "support_reasons": reasons,
            "experiment_id": str(z["experiment_id"]),
            "heldout_diagnoses": [str(x) for x in np.asarray(z["heldout_diagnoses"])],
            "unit_counts": {f: int(np.asarray(z[f"{f}_sample_id"]).shape[0]) for f in _FOLDS},
            "n_keys": len(keys),
        }
        return True, "ok", info
    except Exception as exc:  # pragma: no cover
        return False, f"schema_probe_error:{exc}", {}


# --------------------------------------------------------------------------- #
# Completion / marker.
# --------------------------------------------------------------------------- #
def is_complete(paths):
    """DONE iff terminal marker exists, records checksums + 3 statuses, and the
    recorded checksums match the on-disk logits + checkpoint."""
    if not os.path.isfile(paths["marker"]):
        return False, "no_marker"
    try:
        with open(paths["marker"]) as fh:
            marker = json.load(fh)
    except (OSError, ValueError):
        return False, "unreadable_marker"
    if not marker.get("schema_validated", False):
        return False, "marker_schema_not_validated"
    if "support_statuses" not in marker:
        return False, "marker_without_support_statuses"
    checksums = marker.get("checksums", {})
    if not checksums or os.path.basename(paths["logits"]) not in checksums \
            or os.path.basename(paths["checkpoint"]) not in checksums:
        return False, "marker_without_full_checksums"
    for name, expected in checksums.items():
        candidate = None
        for key in ("logits", "checkpoint"):
            if os.path.basename(paths[key]) == name:
                candidate = paths[key]
        if candidate is None or not os.path.isfile(candidate):
            return False, f"artifact_missing:{name}"
        if _sha256(candidate) != expected:
            return False, f"checksum_mismatch:{name}"
    return True, "complete"


def artifacts_present(paths):
    return os.path.isfile(paths["logits"]) and os.path.isfile(paths["checkpoint"])


def write_marker(cell, paths, gpu_uuid, rc, attempt, source):
    """Validate the npz schema, then (re)write the terminal marker atomically.
    Returns (ok, reason)."""
    if not artifacts_present(paths):
        return False, "artifacts_absent"
    ok, reason, info = validate_npz(paths["logits"])
    if not ok:
        return False, f"schema_invalid:{reason}"
    checksums = {
        os.path.basename(paths["logits"]): _sha256(paths["logits"]),
        os.path.basename(paths["checkpoint"]): _sha256(paths["checkpoint"]),
    }
    marker = {
        "status": "completed",
        "schema_validated": True,
        "campaign": "fedisic_convnext_corrected",
        "pipeline_id": cell["pipeline_id"],
        "semantic_id": cell["semantic_id"],
        "task_id": cell["task_id"],
        "train_rep": cell["train_rep"],
        "heldout_diagnosis": cell["heldout_diagnosis"],
        "canonical_structural_status": cell["canonical_structural_status"],
        "fold_rel": cell_fold_rel(cell),
        "backbone": "convnext_tiny",
        "rounds": ROUNDS,
        "rc": rc,
        "attempt": attempt,
        "gpu_uuid": gpu_uuid,
        "checksums": checksums,
        "logits_npz_bytes": os.path.getsize(paths["logits"]),
        "checkpoint_bytes": os.path.getsize(paths["checkpoint"]),
        "support_statuses": info["support_statuses"],
        "support_reasons": info.get("support_reasons", {}),
        "unit_counts": info.get("unit_counts", {}),
        "npz_experiment_id": info.get("experiment_id", ""),
        "n_npz_keys": info.get("n_keys", 0),
        "marker_written_by": f"fedisic_convnext_corrected_launch.{source}",
        "marker_written_at": _now(),
    }
    _atomic_write(paths["marker"], json.dumps(marker, indent=2, default=str))
    return True, "ok"


def disk_free_gb(path):
    os.makedirs(path, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


# --------------------------------------------------------------------------- #
# Per-cell frozen-input guard, fail-closed.
# --------------------------------------------------------------------------- #
def guard_cell(cell):
    if _sha256(MATRIX) != MATRIX_SHA256:
        return False, "matrix_sha256_changed"
    meta_abs = os.path.join(REPO, METADATA_REL)
    if not os.path.isfile(meta_abs) or _sha256(meta_abs) != METADATA_SHA256:
        return False, "metadata_sha256_mismatch"
    fold_rel = cell_fold_rel(cell)
    fold_abs = os.path.join(REPO, fold_rel)
    if not os.path.isfile(fold_abs):
        return False, f"fold_missing:{fold_rel}"
    want = _FOLD_CHECKSUMS.get(fold_rel)
    if want is None:
        return False, f"fold_not_in_checksums:{fold_rel}"
    if _sha256(fold_abs) != want:
        return False, f"fold_sha256_mismatch:{fold_rel}"
    if not os.path.isdir(os.path.join(REPO, IMAGE_ROOT_REL)):
        return False, "image_root_missing"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Command construction (in-container repo-relative paths).
# --------------------------------------------------------------------------- #
def build_inner_cmd(cell, paths, resume):
    logits_rel = os.path.relpath(paths["logits"], REPO)
    ckpt_rel = os.path.relpath(paths["checkpoint"], REPO)
    cmd = [
        "python", "-m", "fedcore.experiments.run_fed_isic",
        METADATA_REL, cell_fold_rel(cell),
        "--center-col", "center", "--diagnosis-col", "diagnosis",
        "--patient-col", "patient_id", "--lesion-col", "lesion_id",
        "--image-col", "image_id", "--unit-col", "lesion_id",
        "--image-root", IMAGE_ROOT_REL, "--image-extension", ".jpg",
        "--held-out-diagnoses", cell_heldout_csv(cell),
        "--model-replicate", str(int(cell["train_rep"])),
        "--backbone", "convnext_tiny", "--transform", TRANSFORM,
        "--optimizer", "adamw", "--loss", LOSS, "--focal-gamma", FOCAL_GAMMA,
        "--pretrained",
        "--image-size", str(IMAGE_SIZE), "--rounds", str(ROUNDS),
        "--warmup-rounds", str(WARMUP_ROUNDS), "--local-epochs", str(LOCAL_EPOCHS),
        "--batch-size", str(BATCH_SIZE), "--lr", LR, "--weight-decay", WEIGHT_DECAY,
        "--out", logits_rel, "--checkpoint", ckpt_rel,
        "--allow-unbound-legacy",
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def build_docker_cmd(cell, paths, gpu_uuid, resume):
    cname = "ficc_" + cell["semantic_id"].replace(".", "_").replace("/", "_")[:100]
    inner = build_inner_cmd(cell, paths, resume)
    bash_line = "umask 022 && " + shlex.join(inner)
    docker = [
        "docker", "run", "--rm", "--name", cname,
        "--gpus", f"device={gpu_uuid}",
        "--shm-size", "8g",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-e", f"TORCH_HOME={C_TORCH_HOME}",
        "-e", f"PYTHONPATH={C_REPO}",
        "-e", "MPLCONFIGDIR=/tmp",
        "-e", "OMP_NUM_THREADS=4",
        "-e", "PYTHONUNBUFFERED=1",
        "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "-v", f"{REPO}:{C_REPO}:ro",
        "-v", f"{INTERVENTION_RUNS}:{C_REPO}/runs/fedisic_convnext_corrected:rw",
        "-v", f"{TORCH_CACHE_HOST}:{C_TORCH_HOME}:ro",
        "-w", C_REPO, IMAGE,
        "bash", "-lc", bash_line,
    ]
    return docker, cname


# --------------------------------------------------------------------------- #
# Live status.
# --------------------------------------------------------------------------- #
class LiveState:
    def __init__(self, cells, gpu_uuids, out_root, mode, pid, sid):
        self.out_root = out_root
        self.mode = mode
        self.gpu_uuids = gpu_uuids
        self.started_at = _now()
        self.pid = pid
        self.sid = sid
        self.cells = {c["semantic_id"]: {
            "semantic_id": c["semantic_id"],
            "task_id": c["task_id"], "train_rep": c["train_rep"],
            "structural_status": c["canonical_structural_status"],
            "state": "pending", "gpu_uuid": "", "attempt": 0, "rc": "", "reason": "",
            "started_at": "", "finished_at": "",
            "logits_sha256": "", "checkpoint_sha256": "",
        } for c in cells}
        self.gpu_current = {u: "" for u in gpu_uuids}
        self.failures = []
        with _STATE_LOCK:
            self._flush_failures_locked()

    def set(self, sid, **kw):
        with _STATE_LOCK:
            self.cells[sid].update(kw)
            self._flush_locked()

    def set_gpu(self, uuid, sid):
        with _STATE_LOCK:
            self.gpu_current[uuid] = sid
            self._flush_locked()

    def add_failure(self, sid, attempt, rc, reason, stderr_tail):
        with _STATE_LOCK:
            self.failures.append({
                "semantic_id": sid, "attempt": attempt, "rc": rc,
                "reason": reason, "timestamp": _now(), "stderr_tail": stderr_tail,
            })
            self._flush_failures_locked()

    def _counts(self):
        c = {}
        for v in self.cells.values():
            c[v["state"]] = c.get(v["state"], 0) + 1
        return c

    def _flush_locked(self):
        os.makedirs(LIVE, exist_ok=True)
        cols = ["semantic_id", "task_id", "train_rep", "structural_status", "state",
                "gpu_uuid", "attempt", "rc", "reason", "started_at", "finished_at",
                "logits_sha256", "checkpoint_sha256"]
        lines = [",".join(cols)]
        for v in self.cells.values():
            lines.append(",".join(str(v.get(k, "")).replace(",", ";") for k in cols))
        _atomic_write(os.path.join(LIVE, "status.csv"), "\n".join(lines) + "\n")
        summary = {
            "campaign": "fedisic_convnext_corrected", "mode": self.mode,
            "dispatcher_pid": self.pid, "dispatcher_sid": self.sid,
            "image": IMAGE, "started_at": self.started_at, "heartbeat": _now(),
            "out_root": self.out_root, "gpu_uuids": self.gpu_uuids,
            "counts": self._counts(), "total": len(self.cells),
            "gpu_current": dict(self.gpu_current),
        }
        _atomic_write(os.path.join(LIVE, "status.json"), json.dumps(summary, indent=2))
        gcols = ["gpu_index", "gpu_uuid", "current_semantic_id"]
        glines = [",".join(gcols)]
        for i, u in enumerate(self.gpu_uuids):
            glines.append(",".join([str(i), u, self.gpu_current.get(u, "")]))
        _atomic_write(os.path.join(LIVE, "gpu_assignments.csv"), "\n".join(glines) + "\n")

    def _flush_failures_locked(self):
        fcols = ["semantic_id", "attempt", "rc", "reason", "timestamp", "stderr_tail"]
        lines = [",".join(fcols)]
        for f in self.failures:
            lines.append(",".join(str(f.get(k, "")).replace(",", ";").replace("\n", " ")
                                   for k in fcols))
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
    for _attempt in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                with open(lock) as fh:
                    prev = json.load(fh)
            except (OSError, ValueError):
                prev = {}
            prev_pid = prev.get("pid")
            if prev_pid and _pid_alive(int(prev_pid)):
                return None, f"active dispatcher pid={prev_pid} holds {lock}"
            os.remove(lock)  # stale -> steal
            continue
        try:
            sid = os.getsid(0)
        except OSError:
            sid = -1
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"pid": os.getpid(), "ppid": os.getppid(), "sid": sid,
                                 "acquired_at": _now(), "host": os.uname().nodename,
                                 "image": IMAGE}))
        return lock, "acquired"
    return None, "could not acquire global lock"


def acquire_cell_lock(paths, gpu_uuid):
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
        if os.path.exists(paths["lock"]):
            done, _ = is_complete(paths)
            if not done:
                try:
                    os.remove(paths["lock"])
                    cleared += 1
                except OSError:
                    pass
    return cleared


def kill_orphan_containers():
    try:
        out = subprocess.run(["docker", "ps", "-q", "--filter", "name=ficc_"],
                             capture_output=True, text=True, timeout=60)
        ids = [x for x in out.stdout.split() if x]
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, timeout=180)
        return len(ids)
    except Exception:
        return -1


# --------------------------------------------------------------------------- #
# Per-cell processing.
# --------------------------------------------------------------------------- #
def _tail(path, n=2000):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - n))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def process_cell(cell, gpu_uuid, out_root, live, disk_guard_gb, max_retries):
    sid = cell["semantic_id"]
    paths = cell_paths(out_root, cell)

    done, _ = is_complete(paths)
    if done:
        live.set(sid, state="complete", gpu_uuid=gpu_uuid, reason="already_terminal",
                 finished_at=_now())
        return "complete"

    # Crash between artifact write and marker write: finalize without retraining.
    if artifacts_present(paths):
        ok, reason = write_marker(cell, paths, gpu_uuid, 0, 0, "finalize")
        if ok:
            done, _ = is_complete(paths)
            if done:
                live.set(sid, state="complete", gpu_uuid=gpu_uuid,
                         reason="finalized_from_disk", finished_at=_now(),
                         logits_sha256=_sha256(paths["logits"])[:16],
                         checkpoint_sha256=_sha256(paths["checkpoint"])[:16])
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
                 reason=f"disk_guard:{free:.1f}GB<{disk_guard_gb}GB", finished_at=_now())
        live.add_failure(sid, 0, "disk", f"free={free:.1f}GB", "")
        return "failed"

    if not acquire_cell_lock(paths, gpu_uuid):
        live.set(sid, state="skipped", reason="locked_by_other")
        return "locked"

    try:
        attempt = 0
        while attempt <= max_retries and not _STOP.is_set():
            attempt += 1
            resume = os.path.exists(paths["checkpoint"])
            cmd, cname = build_docker_cmd(cell, paths, gpu_uuid, resume)
            live.set(sid, state="running", gpu_uuid=gpu_uuid, attempt=attempt,
                     started_at=_now(), reason="resume" if resume else "fresh")
            live.set_gpu(gpu_uuid, sid)
            os.makedirs(os.path.dirname(paths["stdout"]), exist_ok=True)
            with open(paths["stdout"], "a") as so, open(paths["stderr"], "a") as se:
                so.write(f"\n=== attempt {attempt} resume={resume} gpu={gpu_uuid} "
                         f"{_now()} ===\n")
                so.flush()
                try:
                    proc = subprocess.run(cmd, stdout=so, stderr=se)
                    rc = proc.returncode
                except Exception as exc:  # pragma: no cover
                    rc = -99
                    se.write(f"dispatcher-exception: {exc}\n")

            if _STOP.is_set() and rc != 0:
                live.set(sid, state="pending", gpu_uuid="", reason="dispatcher_stopping")
                live.set_gpu(gpu_uuid, "")
                return "stopped"

            if rc == 0:
                ok, mreason = write_marker(cell, paths, gpu_uuid, rc, attempt, "run")
                done, why = is_complete(paths)
                if ok and done:
                    live.set(sid, state="complete", gpu_uuid=gpu_uuid, rc=rc,
                             reason="completed", finished_at=_now(),
                             logits_sha256=_sha256(paths["logits"])[:16],
                             checkpoint_sha256=_sha256(paths["checkpoint"])[:16])
                    live.set_gpu(gpu_uuid, "")
                    return "complete"
                stderr_tail = _tail(paths["stderr"])
                live.add_failure(sid, attempt, rc,
                                 f"rc0_but_incomplete:{mreason}|{why}", stderr_tail)
            else:
                stderr_tail = _tail(paths["stderr"])
                live.add_failure(sid, attempt, rc, "nonzero_rc", stderr_tail)

        live.set(sid, state="failed", gpu_uuid=gpu_uuid,
                 reason=f"exhausted_retries({max_retries})", finished_at=_now())
        live.set_gpu(gpu_uuid, "")
        return "failed"
    finally:
        release_cell_lock(paths)


# --------------------------------------------------------------------------- #
# Worker pool (one worker per GPU UUID).
# --------------------------------------------------------------------------- #
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
    print(f"[print-plan] matrix rows: {len(_load_matrix())}")
    print(f"[print-plan] matrix sha256: {got}")
    print(f"[print-plan] matches frozen: {got == MATRIX_SHA256}")
    if got != MATRIX_SHA256:
        print("FAIL_CLOSED: matrix hash mismatch")
        return 2
    rows = _load_matrix()
    missing = []
    for p in (MATRIX, os.path.join(REPO, METADATA_REL),
              os.path.join(REPO, IMAGE_ROOT_REL), TORCH_CACHE_HOST,
              os.path.join(TORCH_CACHE_HOST, "hub", "checkpoints",
                           "convnext_tiny-983f1562.pth")):
        if not os.path.exists(p):
            missing.append(p)
    print(f"[print-plan] missing host inputs: {missing}")
    n_guard_fail = 0
    for cell in rows:
        ok, reason = guard_cell(cell)
        if not ok:
            n_guard_fail += 1
            if n_guard_fail <= 5:
                print(f"  guard FAIL {cell['semantic_id']}: {reason}")
    print(f"[print-plan] cells failing guard: {n_guard_fail}")
    # structural distribution
    dist = {}
    for cell in rows:
        dist[cell["canonical_structural_status"]] = \
            dist.get(cell["canonical_structural_status"], 0) + 1
    print(f"[print-plan] structural status distribution: {dist}")
    sample = rows[0]
    paths = cell_paths(os.path.abspath(args.out_root), sample)
    cmd, cname = build_docker_cmd(sample, paths, AUTHORIZED_GPU_UUIDS[0], resume=False)
    print(f"\n[sample] {sample['semantic_id']} container={cname}")
    print("  " + " ".join(shlex.quote(x) for x in cmd))
    if missing or n_guard_fail:
        print("FAIL_CLOSED: missing inputs or guard failures")
        return 2
    print("PRINT_PLAN_OK")
    return 0


def _select_cells(rows, only):
    if not only:
        return rows
    want = set(only)
    sel = [r for r in rows if r["semantic_id"] in want]
    found = {r["semantic_id"] for r in sel}
    missing = want - found
    if missing:
        raise SystemExit(f"FAIL_CLOSED: --only ids not in matrix: {sorted(missing)}")
    return sel


def cmd_run(args):
    got = _sha256(MATRIX)
    if got != MATRIX_SHA256:
        print(f"FAIL_CLOSED: matrix sha256 {got} != frozen {MATRIX_SHA256}")
        return 2
    rows = _load_matrix()
    cells = _select_cells(rows, args.only)
    gpu_uuids = [u for u in (args.gpus.split(",") if args.gpus else AUTHORIZED_GPU_UUIDS) if u]
    for u in gpu_uuids:
        if u not in AUTHORIZED_GPU_UUIDS:
            print(f"FAIL_CLOSED: unauthorized GPU uuid {u}")
            return 2
    out_root = os.path.abspath(args.out_root)
    if os.path.commonpath([out_root, REPO]) != REPO:
        print(f"FAIL_CLOSED: out_root {out_root} not under repo mount {REPO}")
        return 2
    if os.path.commonpath([out_root, INTERVENTION_RUNS]) != INTERVENTION_RUNS:
        print(f"FAIL_CLOSED: out_root {out_root} not under writable mount {INTERVENTION_RUNS}")
        return 2
    os.makedirs(out_root, exist_ok=True)

    lock, why = acquire_global_lock()
    if lock is None:
        print(f"FAIL_CLOSED: {why}")
        return 3
    try:
        sid = os.getsid(0)
    except OSError:
        sid = -1
    print(f"[dispatcher] global lock acquired pid={os.getpid()} sid={sid} ppid={os.getppid()} ({why})")

    try:
        n_orphan = kill_orphan_containers()
        n_stale = clear_stale_cell_locks(cells, out_root)
        print(f"[dispatcher] killed {n_orphan} orphan containers; cleared {n_stale} stale cell locks")

        canary_ids = list(args.canary or [])
        sel_ids = {c["semantic_id"] for c in cells}
        missing_canary = [c for c in canary_ids if c not in sel_ids]
        if missing_canary:
            print(f"FAIL_CLOSED: --canary ids not in selected cells: {missing_canary}")
            return 2
        canary_cells = [c for c in cells if c["semantic_id"] in set(canary_ids)]
        rest_cells = [c for c in cells if c["semantic_id"] not in set(canary_ids)]

        mode = ("canary_gated_full" if canary_ids else
                ("only_subset" if args.only else "full_matrix"))
        live = LiveState(cells, gpu_uuids, out_root, mode, os.getpid(), sid)
        for cell in cells:
            done, _ = is_complete(cell_paths(out_root, cell))
            if done:
                live.set(cell["semantic_id"], state="complete", reason="already_terminal")
        live.heartbeat()

        def _on_sig(signum, frame):
            _STOP.set()
        signal.signal(signal.SIGTERM, _on_sig)
        signal.signal(signal.SIGINT, _on_sig)

        print(f"[dispatcher] mode={mode} cells={len(cells)} canary={len(canary_cells)} "
              f"gpus={len(gpu_uuids)} out_root={out_root}")

        if canary_ids:
            print(f"[dispatcher] CANARY PHASE: {canary_ids}")
            run_pool(canary_cells, gpu_uuids, out_root, live, args.disk_guard_gb,
                     args.max_retries)
            gate = {"canary_ids": canary_ids, "checked_at": _now(), "per_cell": {}}
            all_pass = True
            for c in canary_cells:
                done, why = is_complete(cell_paths(out_root, c))
                gate["per_cell"][c["semantic_id"]] = {"complete": done, "why": why}
                all_pass = all_pass and done
            gate["passed"] = bool(all_pass and not _STOP.is_set())
            _atomic_write(os.path.join(LIVE, "CANARY_GATE.json"),
                          json.dumps(gate, indent=2))
            if not gate["passed"]:
                print(f"CANARY_FAILED_CLOSED: {gate['per_cell']} -- NOT releasing the "
                      f"remaining {len(rest_cells)} cells")
                live.heartbeat()
                return 5
            print(f"[dispatcher] CANARY GATE PASSED -> releasing {len(rest_cells)} cells "
                  f"over {len(gpu_uuids)} GPUs")

        release_cells = rest_cells if canary_ids else cells
        run_pool(release_cells, gpu_uuids, out_root, live, args.disk_guard_gb,
                 args.max_retries)
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

    pp = sub.add_parser("print-plan", help="verify hashes/inputs + print a sample docker cmd")
    pp.add_argument("--out-root", default=RUNS_DEFAULT)
    pp.set_defaults(func=cmd_print_plan)

    rr = sub.add_parser("run", help="real submit over the 50-cell matrix (or --only subset)")
    rr.add_argument("--out-root", default=RUNS_DEFAULT)
    rr.add_argument("--gpus", default="", help="comma-separated GPU UUIDs (default: all 4 authorized)")
    rr.add_argument("--only", action="append", default=[], help="restrict to these semantic_ids")
    rr.add_argument("--canary", action="append", default=[],
                    help="semantic_ids to run FIRST as a gate; the rest launch only if ALL pass")
    rr.add_argument("--disk-guard-gb", type=float, default=DISK_GUARD_GB_DEFAULT)
    rr.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)
    rr.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
