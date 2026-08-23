#!/usr/bin/env python3
"""Fed-ISIC DUAL-campaign UNIFIED dispatcher: no idle GPUs, robust scheduling.

Runs BOTH Fed-ISIC model-intervention arms from ONE work queue over ALL authorized
GPUs, so a freed GPU immediately pulls the next pending cell from EITHER campaign
(the frozen DINOv2 linear-probe finishes much faster than the ConvNeXt full-FT, so
a static per-campaign GPU split would leave GPUs idle at the tail -- this does not):

* ``convnext_corrected`` -- ConvNeXt-Tiny full fine-tune, FLamby weighted focal +
  lesion-preserving crop (audit-confound-corrected), image ``fedcore-c400r``.
* ``dinov2`` -- DINOv2 ViT-L/14 FROZEN linear-probe (only the linear head trains),
  FLamby weighted focal, image ``fedcore-c400r-dino`` (timm + baked SSL weights,
  loaded offline).

Each cell carries its campaign spec (image, backbone, transform, loss, lr,
penultimate-embedding dim, container prefix, matrix hash, output roots), so the
generic worker builds the right container per cell. Everything else mirrors the
audited single-campaign launcher (fedisic_convnext_corrected_launch):

* ONE global dispatcher lock (O_CREAT|O_EXCL) under results/fedisic_dual/live;
  a stale lock is stolen on restart, a live one refuses a second dispatcher.
* per-cell atomic lock; stale locks of non-terminal cells cleared at startup.
* completion = terminal marker + rc0 + checkpoint + logits + embeddings +
  source-IDs + the 3 support statuses + recorded sha256 checksums matching disk.
* completed-cell skip; finalize-from-disk; identical-cell resume retry (SAME frozen
  seeds/fold/heldout; NEVER a new seed).
* per-cell frozen-input sha256 guard (per-campaign matrix + metadata + fold).
* disk guard; per-cell logs; atomic live/{status,gpu_assignments,failures}.

Frozen matrices/folds/metadata and the read-only repo mount are never mutated.
Outputs land only under each campaign's runs/<campaign> (rw) and the shared
results/fedisic_dual/live (host).
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

# --------------------------------------------------------------------------- #
# Frozen locations and invariants (host paths).
# --------------------------------------------------------------------------- #
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DUAL = os.path.join(REPO, "results", "fedisic_dual")
LIVE = os.path.join(DUAL, "live")

METADATA_REL = "results/source_data/fed_isic2019_metadata.csv"
METADATA_SHA256 = "2460578b6f4c46fa475f02114c19894519ffd251677ecca6f63117a669525e7c"
IMAGE_ROOT_REL = "data/isic2019/ISIC_2019_Training_Input_preprocessed"
TORCH_CACHE_HOST = os.path.join(REPO, "data", "officehome", "torch_cache")

C_REPO = "/repo"
C_TORCH_HOME = "/tc"

AUTHORIZED_GPU_UUIDS = [
    "GPU-d6e53d0c-b100-5dd4-30b2-0574b2b4dffb",
    "GPU-94b3a414-8e7a-3454-83dc-6132a9124a28",
    "GPU-c3326fea-a08f-9192-eee8-27fb8017984c",
    "GPU-afbc9e02-0ce4-a4a4-391f-7c31c414771f",
]

# Shared FROZEN recipe knobs (both arms; do NOT change).
ROUNDS = 30
WARMUP_ROUNDS = 2
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
WEIGHT_DECAY = "0.05"
IMAGE_SIZE = 224
FOCAL_GAMMA = "2"

# --------------------------------------------------------------------------- #
# Per-campaign specs. Each cell in the combined queue resolves its spec by
# campaign key. lr, transform, image, embedding dim and container prefix differ.
# --------------------------------------------------------------------------- #
SPECS = {
    "convnext_corrected": {
        "camp": "fedisic_convnext_corrected",
        "matrix": os.path.join(REPO, "results", "fedisic_convnext_corrected", "final_training_matrix.csv"),
        "matrix_sha256": "3550089b288840ed68a1f9457c22de97125dd6462e4ce0b251ac2d473adaa9f2",
        "fold_checksums": os.path.join(REPO, "results", "fedisic_convnext_corrected", "prelaunch", "fold_checksums.sha256"),
        "runs": os.path.join(REPO, "runs", "fedisic_convnext_corrected"),
        "image": "fedcore-c400r:latest",
        "backbone": "convnext_tiny",
        "transform": "convnext-imagenet-mildcrop",
        "loss": "focal",
        "lr": "1e-4",
        "embedding_dim": 768,
        "cprefix": "ficc_",
        "mounts": [("torch_cache_ro",)],
        "env": [],
    },
    "dinov2": {
        "camp": "fedisic_dinov2",
        "matrix": os.path.join(REPO, "results", "fedisic_dinov2", "final_training_matrix.csv"),
        "matrix_sha256": "8664c44cd67d63d5da197dc240e76bc57e7b623ceb789a1750832ac223bf1b1e",
        "fold_checksums": os.path.join(REPO, "results", "fedisic_dinov2", "prelaunch", "fold_checksums.sha256"),
        "runs": os.path.join(REPO, "runs", "fedisic_dinov2"),
        "image": "fedcore-c400r-dino:latest",
        "backbone": "dinov2_vitl14",
        "transform": "dinov2-imagenet",
        "loss": "focal",
        "lr": "1e-3",
        "embedding_dim": 1024,
        "cprefix": "ficd_",
        "mounts": [],
        "env": ["-e", "HF_HOME=/opt/hf", "-e", "HF_HUB_OFFLINE=1"],
    },
}
CPREFIXES = tuple(sorted({s["cprefix"] for s in SPECS.values()}))

DISK_GUARD_GB_DEFAULT = 50.0
MAX_RETRIES_DEFAULT = 2

_STOP = threading.Event()
_STATE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Helpers.
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


def _load_fold_checksums(path):
    table = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split(maxsplit=1)
            table[rel.strip()] = digest
    return table


def load_all_cells():
    """Read both frozen matrices; tag each row with its campaign key + spec."""
    cells = []
    for key, spec in SPECS.items():
        with open(spec["matrix"], newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                row = dict(row)
                row["__campaign"] = key
                cells.append(row)
    return cells


_FOLD_CHECKSUMS = {k: _load_fold_checksums(s["fold_checksums"]) for k, s in SPECS.items()}


def spec_of(cell):
    return SPECS[cell["__campaign"]]


# --------------------------------------------------------------------------- #
# Cell derivation (fold + heldout are campaign-independent: same 50 folds).
# --------------------------------------------------------------------------- #
def cell_fold_rel(cell):
    t = int(cell["task_id"])
    r = int(cell["train_rep"])
    return f"results/source_data/fed_isic2019_folds_split{t:02d}_seed{r}.csv"


def cell_heldout_csv(cell):
    return cell["heldout_diagnosis"].replace("+", ",")


def cell_paths(cell):
    spec = spec_of(cell)
    sid = cell["semantic_id"]
    base = os.path.join(spec["runs"], "cells", sid)
    return {
        "sid": sid,
        "base": base,
        "logits": f"{base}_logits.npz",
        "checkpoint": f"{base}.pt",
        "marker": f"{base}.TERMINAL.json",
        "lock": f"{base}.lock",
        "stdout": f"{base}.out.log",
        "stderr": f"{base}.err.log",
    }


# --------------------------------------------------------------------------- #
# Export-schema validation (campaign-aware backbone + embedding dim).
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


def validate_npz(logits_path, spec):
    """Fail-closed schema check against the cell's campaign spec."""
    import numpy as np

    try:
        z = np.load(logits_path, allow_pickle=True)
    except Exception as exc:  # pragma: no cover
        return False, f"npz_unreadable:{exc}", {}
    keys = set(z.keys())
    try:
        if str(z["backbone"]) != spec["backbone"]:
            return False, f"backbone_mismatch:{z['backbone']}!={spec['backbone']}", {}
        if not bool(np.asarray(z["penultimate_embedding_included"]).item()):
            return False, "embedding_not_included", {}
        want_dim = int(spec["embedding_dim"])
        for f in _FOLDS:
            for suffix in ("logits", "embedding", "sample_id"):
                k = f"{f}_{suffix}"
                if k not in keys:
                    return False, f"missing_key:{k}", {}
            emb = np.asarray(z[f"{f}_embedding"])
            lg = np.asarray(z[f"{f}_logits"])
            sid = np.asarray(z[f"{f}_sample_id"])
            if emb.ndim != 2 or emb.shape[1] != want_dim:
                return False, f"bad_embedding_shape:{f}:{emb.shape}!=*x{want_dim}", {}
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
    spec = spec_of(cell)
    if not artifacts_present(paths):
        return False, "artifacts_absent"
    ok, reason, info = validate_npz(paths["logits"], spec)
    if not ok:
        return False, f"schema_invalid:{reason}"
    checksums = {
        os.path.basename(paths["logits"]): _sha256(paths["logits"]),
        os.path.basename(paths["checkpoint"]): _sha256(paths["checkpoint"]),
    }
    marker = {
        "status": "completed",
        "schema_validated": True,
        "campaign": spec["camp"],
        "pipeline_id": cell.get("pipeline_id", ""),
        "semantic_id": cell["semantic_id"],
        "task_id": cell["task_id"],
        "train_rep": cell["train_rep"],
        "heldout_diagnosis": cell["heldout_diagnosis"],
        "canonical_structural_status": cell["canonical_structural_status"],
        "fold_rel": cell_fold_rel(cell),
        "backbone": spec["backbone"],
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
        "marker_written_by": f"fedisic_dual_launch.{source}",
        "marker_written_at": _now(),
    }
    _atomic_write(paths["marker"], json.dumps(marker, indent=2, default=str))
    return True, "ok"


def disk_free_gb(path):
    os.makedirs(path, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


# --------------------------------------------------------------------------- #
# Per-cell frozen-input guard (per-campaign matrix + shared metadata/fold).
# --------------------------------------------------------------------------- #
def guard_cell(cell):
    spec = spec_of(cell)
    if _sha256(spec["matrix"]) != spec["matrix_sha256"]:
        return False, "matrix_sha256_changed"
    meta_abs = os.path.join(REPO, METADATA_REL)
    if not os.path.isfile(meta_abs) or _sha256(meta_abs) != METADATA_SHA256:
        return False, "metadata_sha256_mismatch"
    fold_rel = cell_fold_rel(cell)
    fold_abs = os.path.join(REPO, fold_rel)
    if not os.path.isfile(fold_abs):
        return False, f"fold_missing:{fold_rel}"
    want = _FOLD_CHECKSUMS[cell["__campaign"]].get(fold_rel)
    if want is None:
        return False, f"fold_not_in_checksums:{fold_rel}"
    if _sha256(fold_abs) != want:
        return False, f"fold_sha256_mismatch:{fold_rel}"
    if not os.path.isdir(os.path.join(REPO, IMAGE_ROOT_REL)):
        return False, "image_root_missing"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Command construction (per-campaign recipe/image/mounts).
# --------------------------------------------------------------------------- #
def build_inner_cmd(cell, paths, resume):
    spec = spec_of(cell)
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
        "--backbone", spec["backbone"], "--transform", spec["transform"],
        "--optimizer", "adamw", "--loss", spec["loss"], "--focal-gamma", FOCAL_GAMMA,
        "--pretrained",
        "--image-size", str(IMAGE_SIZE), "--rounds", str(ROUNDS),
        "--warmup-rounds", str(WARMUP_ROUNDS), "--local-epochs", str(LOCAL_EPOCHS),
        "--batch-size", str(BATCH_SIZE), "--lr", spec["lr"], "--weight-decay", WEIGHT_DECAY,
        "--out", logits_rel, "--checkpoint", ckpt_rel,
        "--allow-unbound-legacy",
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def build_docker_cmd(cell, paths, gpu_uuid, resume):
    spec = spec_of(cell)
    cname = spec["cprefix"] + cell["semantic_id"].replace(".", "_").replace("/", "_")[:100]
    inner = build_inner_cmd(cell, paths, resume)
    bash_line = "umask 022 && " + shlex.join(inner)
    runs_container = f"{C_REPO}/runs/{spec['camp']}"
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
        *spec["env"],
        "-v", f"{REPO}:{C_REPO}:ro",
        "-v", f"{spec['runs']}:{runs_container}:rw",
    ]
    for m in spec["mounts"]:
        if m == ("torch_cache_ro",):
            docker += ["-v", f"{TORCH_CACHE_HOST}:{C_TORCH_HOME}:ro"]
    docker += ["-w", C_REPO, spec["image"], "bash", "-lc", bash_line]
    return docker, cname


# --------------------------------------------------------------------------- #
# Live status (combined over both campaigns).
# --------------------------------------------------------------------------- #
class LiveState:
    def __init__(self, cells, gpu_uuids, pid, sid):
        self.gpu_uuids = gpu_uuids
        self.started_at = _now()
        self.pid = pid
        self.sid = sid
        self.cells = {c["semantic_id"]: {
            "semantic_id": c["semantic_id"], "campaign": c["__campaign"],
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

    def _counts_by_campaign(self):
        out = {}
        for v in self.cells.values():
            d = out.setdefault(v["campaign"], {})
            d[v["state"]] = d.get(v["state"], 0) + 1
        return out

    def _flush_locked(self):
        os.makedirs(LIVE, exist_ok=True)
        cols = ["semantic_id", "campaign", "task_id", "train_rep", "structural_status",
                "state", "gpu_uuid", "attempt", "rc", "reason", "started_at",
                "finished_at", "logits_sha256", "checkpoint_sha256"]
        lines = [",".join(cols)]
        for v in self.cells.values():
            lines.append(",".join(str(v.get(k, "")).replace(",", ";") for k in cols))
        _atomic_write(os.path.join(LIVE, "status.csv"), "\n".join(lines) + "\n")
        summary = {
            "campaign": "fedisic_dual", "dispatcher_pid": self.pid,
            "dispatcher_sid": self.sid, "started_at": self.started_at,
            "heartbeat": _now(), "gpu_uuids": self.gpu_uuids,
            "counts": self._counts(), "counts_by_campaign": self._counts_by_campaign(),
            "total": len(self.cells), "gpu_current": dict(self.gpu_current),
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
            os.remove(lock)
            continue
        try:
            sid = os.getsid(0)
        except OSError:
            sid = -1
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"pid": os.getpid(), "ppid": os.getppid(), "sid": sid,
                                 "acquired_at": _now(), "host": os.uname().nodename}))
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


def clear_stale_cell_locks(cells):
    cleared = 0
    for cell in cells:
        paths = cell_paths(cell)
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
    killed = 0
    for pref in CPREFIXES:
        try:
            out = subprocess.run(["docker", "ps", "-q", "--filter", f"name={pref}"],
                                 capture_output=True, text=True, timeout=60)
            ids = [x for x in out.stdout.split() if x]
            if ids:
                subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, timeout=180)
                killed += len(ids)
        except Exception:
            pass
    return killed


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


def process_cell(cell, gpu_uuid, live, disk_guard_gb, max_retries):
    sid = cell["semantic_id"]
    paths = cell_paths(cell)

    done, _ = is_complete(paths)
    if done:
        live.set(sid, state="complete", gpu_uuid=gpu_uuid, reason="already_terminal",
                 finished_at=_now())
        return "complete"

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

    free = disk_free_gb(os.path.dirname(paths["base"]))
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
# Worker pool (one worker per GPU UUID; combined queue).
# --------------------------------------------------------------------------- #
def run_pool(cells, gpu_uuids, live, disk_guard_gb, max_retries):
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
                process_cell(cell, gpu_uuid, live, disk_guard_gb, max_retries)
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
# Ordering: interleave campaigns so both make progress from the start; put the
# lighter DINOv2 cells first-ish so GPUs never starve while ConvNeXt cells are long.
# --------------------------------------------------------------------------- #
def order_cells(cells):
    by_camp = {}
    for c in cells:
        by_camp.setdefault(c["__campaign"], []).append(c)
    for v in by_camp.values():
        v.sort(key=lambda c: (int(c["task_id"]), int(c["train_rep"])))
    order = []
    keys = list(by_camp)
    i = 0
    while any(by_camp[k] for k in keys):
        k = keys[i % len(keys)]
        if by_camp[k]:
            order.append(by_camp[k].pop(0))
        i += 1
    return order


# --------------------------------------------------------------------------- #
# Modes.
# --------------------------------------------------------------------------- #
def cmd_print_plan(args):
    ok_all = True
    for key, spec in SPECS.items():
        got = _sha256(spec["matrix"]) if os.path.exists(spec["matrix"]) else "MISSING"
        match = got == spec["matrix_sha256"]
        ok_all = ok_all and match
        print(f"[{key}] matrix sha256 {got} matches_frozen={match} image={spec['image']}")
    cells = load_all_cells()
    print(f"[print-plan] total cells: {len(cells)} "
          f"({', '.join(k+':'+str(sum(1 for c in cells if c['__campaign']==k)) for k in SPECS)})")
    # docker images present?
    imgs = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                          capture_output=True, text=True).stdout.split()
    for key, spec in SPECS.items():
        present = spec["image"] in imgs
        print(f"[{key}] image present: {present}")
        ok_all = ok_all and present
    missing = [p for p in (os.path.join(REPO, METADATA_REL),
                           os.path.join(REPO, IMAGE_ROOT_REL)) if not os.path.exists(p)]
    print(f"[print-plan] missing shared inputs: {missing}")
    ok_all = ok_all and not missing
    n_guard_fail = 0
    for cell in cells:
        gok, reason = guard_cell(cell)
        if not gok:
            n_guard_fail += 1
            if n_guard_fail <= 5:
                print(f"  guard FAIL {cell['semantic_id']}: {reason}")
    print(f"[print-plan] cells failing guard: {n_guard_fail}")
    ok_all = ok_all and n_guard_fail == 0
    for key in SPECS:
        sample = next(c for c in cells if c["__campaign"] == key)
        cmd, cname = build_docker_cmd(sample, cell_paths(sample), AUTHORIZED_GPU_UUIDS[0], False)
        print(f"\n[sample {key}] {sample['semantic_id']} container={cname}")
        print("  " + " ".join(shlex.quote(x) for x in cmd))
    print("\nPRINT_PLAN_OK" if ok_all else "\nFAIL_CLOSED")
    return 0 if ok_all else 2


def cmd_run(args):
    for key, spec in SPECS.items():
        got = _sha256(spec["matrix"])
        if got != spec["matrix_sha256"]:
            print(f"FAIL_CLOSED: {key} matrix sha256 {got} != frozen {spec['matrix_sha256']}")
            return 2
    cells = order_cells(load_all_cells())
    gpu_uuids = [u for u in (args.gpus.split(",") if args.gpus else AUTHORIZED_GPU_UUIDS) if u]
    for u in gpu_uuids:
        if u not in AUTHORIZED_GPU_UUIDS:
            print(f"FAIL_CLOSED: unauthorized GPU uuid {u}")
            return 2

    lock, why = acquire_global_lock()
    if lock is None:
        print(f"FAIL_CLOSED: {why}")
        return 3
    try:
        sid = os.getsid(0)
    except OSError:
        sid = -1
    print(f"[dispatcher] global lock acquired pid={os.getpid()} sid={sid} ({why})")

    try:
        n_orphan = kill_orphan_containers()
        n_stale = clear_stale_cell_locks(cells)
        print(f"[dispatcher] killed {n_orphan} orphan containers; cleared {n_stale} stale cell locks")

        live = LiveState(cells, gpu_uuids, os.getpid(), sid)
        for cell in cells:
            done, _ = is_complete(cell_paths(cell))
            if done:
                live.set(cell["semantic_id"], state="complete", reason="already_terminal")
        live.heartbeat()

        def _on_sig(signum, frame):
            _STOP.set()
        signal.signal(signal.SIGTERM, _on_sig)
        signal.signal(signal.SIGINT, _on_sig)

        print(f"[dispatcher] mode=dual cells={len(cells)} gpus={len(gpu_uuids)}")
        run_pool(cells, gpu_uuids, live, args.disk_guard_gb, args.max_retries)
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
    pp = sub.add_parser("print-plan")
    pp.set_defaults(func=cmd_print_plan)
    rr = sub.add_parser("run")
    rr.add_argument("--gpus", default="", help="comma-separated GPU UUIDs (default all 4)")
    rr.add_argument("--disk-guard-gb", type=float, default=DISK_GUARD_GB_DEFAULT)
    rr.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)
    rr.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
