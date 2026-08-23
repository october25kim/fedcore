"""Regression tests for the CUDA RNG resume bug in ``fedcore.models.fed_train``.

Bug: ``--resume`` restored ``cuda_rng_state_all`` with ``torch.load(..., map_location=
<cuda>)``, which moved the stored CUDA RNG states onto the GPU.  ``torch.cuda.
set_rng_state_all`` accepts ONLY CPU uint8 (ByteTensor) states, so restore raised
``TypeError: RNG state must be a torch.ByteTensor``.

The fix (a) loads the checkpoint device-independently (map_location="cpu"),
(b) serializes CUDA RNG states as CPU uint8 on save, (c) coerces + validates them
on restore, (d) restores Python + NumPy + Torch-CPU + Torch-CUDA RNG, and
(e) fails closed on a malformed state or a CUDA device-count mismatch, while
skipping (not failing) CUDA RNG on a CPU-only host.

Covers owner-§8 cases:
  1. GPU save -> same GPU restore                  [test_case1_same_gpu_resume_equiv]
  2. GPU save -> docker restart -> restore         [test_case2_docker_restart_resume + test_case2_map_location_cuda_byte_coercion]
  3. CUDA_VISIBLE_DEVICES remap                    [test_case3_cuda_visible_devices_remap]
  4. single-GPU                                    [test_case4_single_gpu]
  5. multi-GPU                                     [test_case5_multi_gpu]
  6. CPU-only load of a GPU checkpoint             [test_case6_cpu_only_load]
  7. malformed RNG fail-closed                     [test_case7_*]
  8. uninterrupted-vs-interrupted equivalence      [test_case8_cpu_resume_equiv]

Torch-requiring tests are skipped when torch is unavailable (host); run them in the
pinned Docker image ``pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime``.  GPU cases
additionally skip when too few CUDA devices are visible.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    import pytest
except ImportError:  # pragma: no cover - pytest always present in CI/Docker
    pytest = None

try:
    import torch

    HAVE_TORCH = True
    HAVE_CUDA = torch.cuda.is_available()
    N_GPU = torch.cuda.device_count() if HAVE_CUDA else 0
except Exception:  # pragma: no cover
    HAVE_TORCH = False
    HAVE_CUDA = False
    N_GPU = 0


_SKIP_EXC = (pytest.skip.Exception,) if pytest is not None else (SystemExit,)


def _need(cond, reason):
    if cond:
        return
    if pytest is not None:
        pytest.skip(reason)
    raise SystemExit(f"SKIP: {reason}")


# --------------------------------------------------------------------------- #
# Subprocess worker: a tiny deterministic FedAvg run used for FRESH-PROCESS
# (docker-restart) resume-equivalence under a chosen CUDA_VISIBLE_DEVICES.
# --------------------------------------------------------------------------- #
_WORKER = r"""
import argparse, os, random, sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True)     # full | interrupt | resume
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()

from fedcore.models.fed_train import fedavg

def make_model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))

g = torch.Generator().manual_seed(0)
x = torch.randn(60, 4, generator=g)
y = (torch.arange(60) % 3).long()
clients = [TensorDataset(x[:30], y[:30]), TensorDataset(x[30:], y[30:])]

kw = dict(
    make_model_fn=make_model, client_datasets=clients, rounds=3, local_epochs=1,
    lr=0.05, batch_size=8, device=a.device, loader_seed=4242, checkpoint_every=1,
    checkpoint_metadata={"training_config_sha256": "rngfix-fixture"},
    checkpoint_path=a.ckpt,
)

def seed_all():
    random.seed(2024); np.random.seed(2024); torch.manual_seed(2024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2024)

if a.mode == "full":
    seed_all()
    m = fedavg(**kw)
elif a.mode == "interrupt":
    seed_all()
    class Stop(RuntimeError): pass
    def cb(r):
        if r == 0:
            raise Stop
    try:
        fedavg(**kw, round_end_callback=cb)
    except Stop:
        pass
    open(a.out, "w").write("interrupted")
    sys.exit(0)
elif a.mode == "resume":
    m = fedavg(**kw, resume=True)   # NO reseed: RNG is restored from the checkpoint
else:
    raise SystemExit("bad mode")

torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, a.out)
print("WORKER_OK", a.mode)
"""


def _run_worker(mode, ckpt, out, device, cvd):
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = cvd
    return subprocess.run(
        [sys.executable, "-c", _WORKER, "--mode", mode, "--ckpt", ckpt,
         "--out", out, "--device", device],
        env=env, capture_output=True, text=True,
    )


def _fresh_process_resume_equiv(cvd_full, cvd_resume, device):
    """full run  ==  (fresh-process interrupt) + (fresh-process resume)."""
    with tempfile.TemporaryDirectory() as td:
        full_ck = os.path.join(td, "full.pt")
        full_out = os.path.join(td, "full_out.pt")
        res_ck = os.path.join(td, "res.pt")
        marker = os.path.join(td, "marker.txt")
        res_out = os.path.join(td, "res_out.pt")

        r = _run_worker("full", full_ck, full_out, device, cvd_full)
        assert r.returncode == 0, f"full failed:\n{r.stdout}\n{r.stderr}"
        r = _run_worker("interrupt", res_ck, marker, device, cvd_full)
        assert r.returncode == 0, f"interrupt failed:\n{r.stdout}\n{r.stderr}"
        assert os.path.exists(res_ck), "interrupt did not write a checkpoint"
        r = _run_worker("resume", res_ck, res_out, device, cvd_resume)
        assert r.returncode == 0, f"resume failed:\n{r.stdout}\n{r.stderr}"

        a = torch.load(full_out, map_location="cpu")
        b = torch.load(res_out, map_location="cpu")
        assert set(a) == set(b)
        for k in a:
            assert torch.equal(a[k], b[k]), f"resume model diverged at {k}"


# --------------------------------------------------------------------------- #
# Case 1 — GPU save -> same-GPU restore (in-process), plus byte-exact CUDA RNG
# --------------------------------------------------------------------------- #
def test_case1_same_gpu_resume_equiv():
    _need(HAVE_TORCH, "torch unavailable")
    _need(HAVE_CUDA, "no CUDA device")
    from torch import nn
    from torch.utils.data import TensorDataset
    from fedcore.models.fed_train import fedavg

    def make_model():
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))

    g = torch.Generator().manual_seed(1)
    x = torch.randn(60, 4, generator=g)
    y = (torch.arange(60) % 3).long()
    clients = [TensorDataset(x[:30], y[:30]), TensorDataset(x[30:], y[30:])]
    kw = dict(make_model_fn=make_model, client_datasets=clients, rounds=3,
              local_epochs=1, lr=0.05, batch_size=8, device="cuda:0",
              loader_seed=99, checkpoint_every=1,
              checkpoint_metadata={"training_config_sha256": "fx"})
    with tempfile.TemporaryDirectory() as td:
        torch.manual_seed(7); np.random.seed(7); torch.cuda.manual_seed_all(7)
        full = fedavg(**kw, checkpoint_path=os.path.join(td, "f.pt"))

        class Stop(RuntimeError):
            pass

        def cb(r):
            if r == 0:
                raise Stop

        rp = os.path.join(td, "r.pt")
        torch.manual_seed(7); np.random.seed(7); torch.cuda.manual_seed_all(7)
        try:
            fedavg(**kw, checkpoint_path=rp, round_end_callback=cb)
        except Stop:
            pass
        # This resume would raise TypeError under the old (buggy) restore path.
        resumed = fedavg(**kw, checkpoint_path=rp, resume=True)
        for name, t in full.state_dict().items():
            assert torch.equal(t, resumed.state_dict()[name]), name


def test_case1b_cuda_rng_bytes_roundtrip():
    """capture -> save -> load(cpu) -> restore reproduces the exact CUDA RNG bytes."""
    _need(HAVE_TORCH, "torch unavailable")
    _need(HAVE_CUDA, "no CUDA device")
    from fedcore.models.fed_train import capture_rng_states, restore_rng_states

    torch.cuda.manual_seed_all(1234)
    before = [s.clone() for s in torch.cuda.get_rng_state_all()]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "rng.pt")
        torch.save(capture_rng_states(), p)
        # perturb the live CUDA RNG so a no-op restore would be detectable
        torch.cuda.manual_seed_all(999)
        payload = torch.load(p, map_location="cpu")
        rep = restore_rng_states(payload)
    after = torch.cuda.get_rng_state_all()
    assert rep["cuda"].startswith("restored_")
    for i, (a, b) in enumerate(zip(before, after)):
        assert torch.equal(a.cpu(), b.cpu()), f"cuda rng device {i} not restored"


# --------------------------------------------------------------------------- #
# Case 2 — docker restart (fresh interpreter) + the exact map_location=cuda trigger
# --------------------------------------------------------------------------- #
def test_case2_map_location_cuda_byte_coercion():
    """The precise bug trigger: states loaded onto CUDA must still restore."""
    _need(HAVE_TORCH, "torch unavailable")
    _need(HAVE_CUDA, "no CUDA device")
    from fedcore.models.fed_train import capture_rng_states, restore_rng_states

    torch.cuda.manual_seed_all(555)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "rng.pt")
        torch.save(capture_rng_states(), p)
        # Reproduce the OLD buggy reload: map_location onto the GPU.
        payload = torch.load(p, map_location="cuda:0")
        assert payload["cuda_rng_state_all"][0].is_cuda  # states are on GPU here
        # New restore coerces them back to CPU uint8 instead of raising TypeError.
        rep = restore_rng_states(payload)
        assert rep["cuda"].startswith("restored_")


def test_case2_docker_restart_resume():
    _need(HAVE_TORCH, "torch unavailable")
    _need(N_GPU >= 1, "needs >=1 CUDA device")
    _fresh_process_resume_equiv("0", "0", "cuda:0")


# --------------------------------------------------------------------------- #
# Case 3 — CUDA_VISIBLE_DEVICES remap (save on 0,1 ; resume on 2,3)
# --------------------------------------------------------------------------- #
def test_case3_cuda_visible_devices_remap():
    _need(HAVE_TORCH, "torch unavailable")
    _need(N_GPU >= 4, "needs >=4 CUDA devices for a 2->2 remap")
    _fresh_process_resume_equiv("0,1", "2,3", "cuda:0")


# --------------------------------------------------------------------------- #
# Case 4 — single-GPU
# --------------------------------------------------------------------------- #
def test_case4_single_gpu():
    _need(HAVE_TORCH, "torch unavailable")
    _need(N_GPU >= 1, "needs >=1 CUDA device")
    _fresh_process_resume_equiv("0", "0", "cuda:0")


# --------------------------------------------------------------------------- #
# Case 5 — multi-GPU
# --------------------------------------------------------------------------- #
def test_case5_multi_gpu():
    _need(HAVE_TORCH, "torch unavailable")
    _need(N_GPU >= 2, "needs >=2 CUDA devices")
    _fresh_process_resume_equiv("0,1", "0,1", "cuda:0")


# --------------------------------------------------------------------------- #
# Case 6 — CPU-only load of a GPU checkpoint (skip CUDA RNG, restore the rest)
# --------------------------------------------------------------------------- #
def test_case6_cpu_only_load():
    _need(HAVE_TORCH, "torch unavailable")
    import random as _random
    from fedcore.models import fed_train
    from fedcore.models.fed_train import restore_rng_states

    # Fabricate a "GPU checkpoint": a list of CPU uint8 CUDA RNG states, as written
    # by capture_rng_states on a GPU host, plus python/numpy/torch-cpu states.
    _random.seed(3); np.random.seed(3); torch.manual_seed(3)
    payload = {
        "python_random_state": _random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state().to(torch.uint8),
        "cuda_rng_state_all": [torch.zeros(16, dtype=torch.uint8)],
    }
    # Force the CPU-only branch regardless of the host having a GPU.
    orig = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        rep = restore_rng_states(payload)
    finally:
        torch.cuda.is_available = orig
    assert rep["cuda"] == "skipped_cpu_only_host"
    assert rep["python"] and rep["numpy"] and rep["torch_cpu"]


# --------------------------------------------------------------------------- #
# Case 7 — malformed RNG fails closed
# --------------------------------------------------------------------------- #
def test_case7_malformed_state_helpers():
    _need(HAVE_TORCH, "torch unavailable")
    from fedcore.models.fed_train import _to_cpu_byte_rng

    def _raises(exc, fn):
        try:
            fn()
        except exc:
            return True
        return False

    assert _raises(TypeError, lambda: _to_cpu_byte_rng("not-a-tensor", "x"))
    assert _raises(TypeError, lambda: _to_cpu_byte_rng(torch.zeros(8, dtype=torch.float32), "x"))
    assert _raises(ValueError, lambda: _to_cpu_byte_rng(torch.zeros(2, 2, dtype=torch.uint8), "x"))
    assert _raises(ValueError, lambda: _to_cpu_byte_rng(torch.zeros(0, dtype=torch.uint8), "x"))
    # a well-formed state passes and is CPU uint8 1-D
    ok = _to_cpu_byte_rng(torch.arange(5, dtype=torch.uint8), "x")
    assert ok.device.type == "cpu" and ok.dtype == torch.uint8 and ok.dim() == 1


def test_case7_malformed_torch_state_in_restore():
    _need(HAVE_TORCH, "torch unavailable")
    from fedcore.models.fed_train import restore_rng_states
    try:
        restore_rng_states({"torch_rng_state": torch.zeros(4, dtype=torch.float32)})
    except TypeError:
        return
    raise AssertionError("malformed torch_rng_state did not fail closed")


def test_case7_device_count_mismatch_fail_closed():
    _need(HAVE_TORCH, "torch unavailable")
    _need(HAVE_CUDA, "device-count check only meaningful with CUDA")
    from fedcore.models.fed_train import restore_rng_states
    # N+1 states while N devices visible -> fail closed.
    bad = [torch.zeros(16, dtype=torch.uint8) for _ in range(N_GPU + 1)]
    try:
        restore_rng_states({"cuda_rng_state_all": bad})
    except RuntimeError:
        return
    raise AssertionError("device-count mismatch did not fail closed")


# --------------------------------------------------------------------------- #
# Case 8 — uninterrupted vs interrupted equivalence (device-independent, CPU)
# --------------------------------------------------------------------------- #
def test_case8_cpu_resume_equiv():
    _need(HAVE_TORCH, "torch unavailable")
    from torch import nn
    from torch.utils.data import TensorDataset
    from fedcore.models.fed_train import fedavg

    x = torch.arange(80, dtype=torch.float32).reshape(40, 2) / 80.0
    y = (torch.arange(40) % 2).long()
    clients = [TensorDataset(x[:20], y[:20]), TensorDataset(x[20:], y[20:])]

    def make_model():
        return nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

    kw = dict(make_model_fn=make_model, client_datasets=clients, rounds=3,
              local_epochs=1, lr=0.02, batch_size=5, device="cpu",
              loader_seed=777, checkpoint_every=1,
              checkpoint_metadata={"training_config_sha256": "fixture"})
    with tempfile.TemporaryDirectory() as td:
        import random as _random
        _random.seed(123); torch.manual_seed(123); np.random.seed(123)
        full = fedavg(**kw, checkpoint_path=os.path.join(td, "full.pt"))

        class Stop(RuntimeError):
            pass

        def cb(r):
            if r == 0:
                raise Stop

        rp = os.path.join(td, "resume.pt")
        _random.seed(123); torch.manual_seed(123); np.random.seed(123)
        try:
            fedavg(**kw, checkpoint_path=rp, round_end_callback=cb)
        except Stop:
            pass
        resumed = fedavg(**kw, checkpoint_path=rp, resume=True)
        for name, t in full.state_dict().items():
            assert torch.equal(t, resumed.state_dict()[name]), name


def main():
    fns = [
        ("case8_cpu_resume_equiv", test_case8_cpu_resume_equiv),
        ("case7_malformed_state_helpers", test_case7_malformed_state_helpers),
        ("case7_malformed_torch_state", test_case7_malformed_torch_state_in_restore),
        ("case7_device_count_mismatch", test_case7_device_count_mismatch_fail_closed),
        ("case6_cpu_only_load", test_case6_cpu_only_load),
        ("case1_same_gpu_resume_equiv", test_case1_same_gpu_resume_equiv),
        ("case1b_cuda_rng_bytes_roundtrip", test_case1b_cuda_rng_bytes_roundtrip),
        ("case2_map_location_cuda", test_case2_map_location_cuda_byte_coercion),
        ("case2_docker_restart", test_case2_docker_restart_resume),
        ("case4_single_gpu", test_case4_single_gpu),
        ("case5_multi_gpu", test_case5_multi_gpu),
        ("case3_cvd_remap", test_case3_cuda_visible_devices_remap),
    ]
    for name, fn in fns:
        try:
            fn()
            print(f"[PASS] {name}")
        except _SKIP_EXC as e:
            print(f"[SKIP] {name}: {e}")
        except Exception as e:  # pragma: no cover
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            raise
    print("rng resume tests: DONE")


if __name__ == "__main__":
    main()
