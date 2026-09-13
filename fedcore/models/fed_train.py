"""FedAvg training and logit export (torch).

``fedavg`` performs size-weighted averaging of client model parameters over a
number of communication rounds; non-float buffers (e.g. BatchNorm
``num_batches_tracked``) are copied from the first client rather than averaged.
``export_logits`` produces logits over the known classes for a set of dataset
indices, which then feed the (torch-free) certification core.
"""

from __future__ import annotations

import copy
import os
import random
from typing import Callable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


# --------------------------------------------------------------------------- #
# RNG checkpoint helpers
#
# torch.Generator.set_state and torch.cuda.set_rng_state_all accept ONLY a CPU
# uint8 (ByteTensor).  A checkpoint reloaded with ``torch.load(..., map_location=
# <cuda>)`` puts the stored CUDA RNG states on the GPU, so restoring them raises
# ``TypeError: RNG state must be a torch.ByteTensor``.  That is the exact resume
# bug this module guards against.  The two helpers below make the format
# canonical (CPU uint8) on save and coerce+validate on restore, so a checkpoint
# written on GPU restores cleanly in a fresh process (e.g. after a Docker
# restart), on a remapped ``CUDA_VISIBLE_DEVICES`` of the same device count, or
# on a CPU-only host (where CUDA RNG is skipped, not an error).
# --------------------------------------------------------------------------- #

def _to_cpu_byte_rng(state: object, what: str) -> torch.Tensor:
    """Return an RNG state as a contiguous CPU uint8 (ByteTensor), or fail closed.

    ``map_location`` only ever changes a tensor's *device*, never its dtype, so a
    non-uint8 dtype (or a non-tensor / non-1-D value) means a corrupt or foreign
    state and is rejected rather than silently coerced.
    """
    if not torch.is_tensor(state):
        raise TypeError(
            f"{what} RNG state must be a torch.Tensor, got {type(state).__name__}"
        )
    tensor = state.detach().to(device="cpu")
    if tensor.dtype != torch.uint8:
        raise TypeError(
            f"{what} RNG state must be uint8 (ByteTensor), got dtype {tensor.dtype}"
        )
    tensor = tensor.contiguous()
    if tensor.dim() != 1 or tensor.numel() == 0:
        raise ValueError(
            f"{what} RNG state must be a non-empty 1-D ByteTensor, "
            f"got shape {tuple(tensor.shape)}"
        )
    return tensor


def capture_rng_states() -> dict:
    """Snapshot Python / NumPy / Torch-CPU / Torch-CUDA RNG for a checkpoint.

    CUDA states are serialized as CPU uint8 tensors so the checkpoint is
    device-independent and restores regardless of ``torch.load`` map_location.
    """
    states: dict = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state().detach().to(device="cpu").to(torch.uint8),
    }
    if torch.cuda.is_available():
        states["cuda_rng_state_all"] = [
            s.detach().to(device="cpu").to(torch.uint8).contiguous()
            for s in torch.cuda.get_rng_state_all()
        ]
    else:
        states["cuda_rng_state_all"] = None
    return states


def restore_rng_states(payload: dict) -> dict:
    """Restore Python / NumPy / Torch-CPU / Torch-CUDA RNG from a checkpoint.

    Fails closed on a malformed CUDA state or on a CUDA device-count mismatch.
    On a CPU-only host a GPU checkpoint's CUDA RNG states are skipped (not an
    error) so the load stays device-independent where possible; the model,
    Python, NumPy and Torch-CPU RNG are still restored.  Returns a small report
    of what was restored (used by the regression tests).
    """
    report = {"python": False, "numpy": False, "torch_cpu": False, "cuda": "none_in_checkpoint"}

    py_state = payload.get("python_random_state")
    if py_state is not None:
        # random.setstate needs (version, tuple-of-ints, gauss_next); pickle round-trips it.
        random.setstate(tuple(py_state) if isinstance(py_state, list) else py_state)
        report["python"] = True

    np_state = payload.get("numpy_random_state")
    if np_state is not None:
        np.random.set_state(np_state)
        report["numpy"] = True

    torch_state = payload.get("torch_rng_state")
    if torch_state is not None:
        torch.set_rng_state(_to_cpu_byte_rng(torch_state, "torch(cpu)"))
        report["torch_cpu"] = True

    saved = payload.get("cuda_rng_state_all")
    if saved is None:
        return report
    if not torch.cuda.is_available():
        report["cuda"] = "skipped_cpu_only_host"
        return report
    if not isinstance(saved, (list, tuple)):
        raise TypeError(
            "cuda_rng_state_all must be a list of ByteTensors, "
            f"got {type(saved).__name__}"
        )
    visible = torch.cuda.device_count()
    if len(saved) != visible:
        raise RuntimeError(
            f"checkpoint carries {len(saved)} CUDA RNG state(s) but {visible} CUDA "
            "device(s) are visible; refusing to restore a device-count-incompatible "
            "checkpoint (align CUDA_VISIBLE_DEVICES to the saved device count)"
        )
    coerced = [_to_cpu_byte_rng(s, f"cuda[{i}]") for i, s in enumerate(saved)]
    torch.cuda.set_rng_state_all(coerced)
    report["cuda"] = f"restored_{visible}_device(s)"
    return report


def local_train(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
) -> nn.Module:
    """Train ``model`` in place with SGD+momentum on cross-entropy."""
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def make_fedprox_local_train_fn(mu: float) -> Callable[..., nn.Module]:
    """Faithful FedProx (Li et al., MLSys 2020) client update as a ``local_train_fn``.

    Identical to :func:`local_train` (SGD momentum=0.9, weight_decay=5e-4,
    cross-entropy) EXCEPT it adds the FedProx proximal term ``(mu/2)||w - w_t||^2``
    to the local objective, where ``w_t`` is the global weight snapshot at the start
    of this round (the model arrives already loaded with the global state, so we
    snapshot it before the first local step). The proximal gradient ``mu*(w - w_t)``
    is added to each parameter's gradient after ``backward()``. With ``mu=0`` this
    reduces exactly to :func:`local_train`, so ``FedProx - FedAvg`` isolates the
    proximal term alone (matched everything else). Architecture-agnostic: it operates
    on ``model.parameters()`` and needs no per-backbone surgery.
    """
    def fedprox_local_train(model, loader, epochs, lr, device):
        model.to(device).train()
        global_w = [p.detach().clone() for p in model.parameters()]
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        ce = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = ce(model(xb), yb)
                loss.backward()
                for p, w0 in zip(model.parameters(), global_w):
                    if p.grad is not None:
                        p.grad.add_(p.data - w0, alpha=mu)
                opt.step()
        return model
    return fedprox_local_train


def _average_state_dicts(state_dicts: List[dict], weights: Sequence[float]) -> dict:
    """Size-weighted average of float tensors; non-float buffers from client 0."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    avg = copy.deepcopy(state_dicts[0])
    for key, ref in avg.items():
        if torch.is_floating_point(ref):
            stacked = torch.stack(
                [sd[key].float() * w for sd, w in zip(state_dicts, weights)], dim=0
            )
            avg[key] = stacked.sum(dim=0).to(ref.dtype)
        else:
            avg[key] = state_dicts[0][key]  # e.g. num_batches_tracked
    return avg


def fedavg(
    make_model_fn: Callable[[], nn.Module],
    client_datasets: List[Dataset],
    rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    *,
    loader_seed: int = 0,
    checkpoint_path: str | None = None,
    resume: bool = False,
    checkpoint_every: int = 1,
    checkpoint_metadata: dict | None = None,
    round_end_callback: Callable[[int], None] | None = None,
    local_train_fn: Callable[..., nn.Module] | None = None,
    lr_schedule: Callable[[int], float] | None = None,
) -> nn.Module:
    """Run resumable FedAvg and return the final global model.

    Minibatch permutations are derived from ``(loader_seed, round, client)`` so
    restarting at a checkpoint cannot change later client orders.  Checkpoints are
    atomically replaced after complete aggregation rounds and include RNG states.
    The default arguments preserve the legacy in-memory behavior.

    ``local_train_fn`` overrides the client update rule; it must accept
    ``(model, loader, epochs, lr, device)`` exactly as :func:`local_train` does.
    ``None`` (the default) uses :func:`local_train`, i.e. the CIFAR/FedPD arms'
    SGD+momentum/cross-entropy, so existing callers are bit-identical.  The
    Fed-ISIC arm passes FLamby's Adam + weighted focal loss
    (:func:`fedcore.medical.flamby.make_local_train_fn`).

    ``lr_schedule`` optionally maps a 0-indexed round to that round's learning
    rate (used by the Office-Home arm's cosine schedule with warmup). Because the
    rate is a pure function of the round index, it restores exactly on ``--resume``
    -- the resumed run recomputes ``lr_schedule(r)`` for the same ``r``. ``None``
    (the default) keeps the fixed ``lr`` for every round, so existing callers are
    bit-identical.
    """
    client_update = local_train if local_train_fn is None else local_train_fn
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    global_model = make_model_fn().to(device)
    sizes = [len(ds) for ds in client_datasets]
    start_round = 0

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        # Device-independent load: pull the whole payload to CPU so the stored
        # CUDA RNG states stay CPU uint8 (ByteTensor) rather than being moved to
        # the GPU by map_location, which is what broke torch.cuda.set_rng_state_all
        # on --resume.  The model weights are copied onto ``device`` below via
        # load_state_dict into the already-on-device global_model.
        payload = torch.load(checkpoint_path, map_location="cpu")
        expected_meta = dict(checkpoint_metadata or {})
        stored_meta = dict(payload.get("metadata", {}))
        expected_hash = expected_meta.get("training_config_sha256")
        if (
            expected_hash is not None
            and stored_meta.get("training_config_sha256") != expected_hash
        ):
            raise ValueError(
                "checkpoint training_config_sha256 does not match the requested job"
            )
        if int(payload.get("loader_seed", loader_seed)) != int(loader_seed):
            raise ValueError("checkpoint loader_seed does not match the requested job")
        if int(payload.get("rounds_requested", rounds)) != int(rounds):
            raise ValueError(
                "checkpoint rounds_requested does not match the requested job"
            )
        global_model.load_state_dict(payload["model_state_dict"])
        start_round = int(payload["round_completed"]) + 1
        if start_round > rounds:
            raise ValueError(
                f"checkpoint completed round {start_round - 1}, beyond requested rounds={rounds}"
            )
        # Restore Python + NumPy + Torch-CPU + Torch-CUDA RNG, failing closed on a
        # malformed state or a CUDA device-count mismatch (see restore_rng_states).
        restore_rng_states(payload)

    for r in range(start_round, rounds):
        round_lr = float(lr) if lr_schedule is None else float(lr_schedule(r))
        local_states: List[dict] = []
        used_sizes: List[int] = []
        for client_id, (ds, size) in enumerate(zip(client_datasets, sizes)):
            if size == 0:
                continue
            local = make_model_fn().to(device)
            local.load_state_dict(copy.deepcopy(global_model.state_dict()))
            generator = torch.Generator()
            generator.manual_seed(
                (int(loader_seed) + 1_000_003 * r + 97_409 * client_id) % (2**63 - 1)
            )
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                generator=generator,
            )
            client_update(local, loader, local_epochs, round_lr, device)
            local_states.append(local.state_dict())
            used_sizes.append(size)
        if local_states:
            global_model.load_state_dict(_average_state_dicts(local_states, used_sizes))
        if checkpoint_path and ((r + 1) % checkpoint_every == 0 or r + 1 == rounds):
            os.makedirs(
                os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True
            )
            payload = {
                # schema 2: adds python_random_state and stores CUDA RNG as CPU
                # uint8; schema-1 checkpoints (no python state) still resume via
                # the .get() guards in restore_rng_states.
                "schema_version": 2,
                "round_completed": r,
                "rounds_requested": rounds,
                "model_state_dict": global_model.state_dict(),
                "loader_seed": int(loader_seed),
                "metadata": dict(checkpoint_metadata or {}),
                **capture_rng_states(),
            }
            tmp = f"{checkpoint_path}.tmp.{os.getpid()}"
            torch.save(payload, tmp)
            os.replace(tmp, checkpoint_path)
        if round_end_callback is not None:
            round_end_callback(r)
    return global_model


@torch.no_grad()
def export_logits(
    model: nn.Module,
    base_dataset: Dataset,
    indices: Sequence[int],
    device: str,
    bs: int = 256,
) -> np.ndarray:
    """Return logits of shape ``(len(indices), n_known)`` for ``indices``."""
    model.to(device).eval()
    subset = Subset(base_dataset, list(indices))
    loader = DataLoader(subset, batch_size=bs, shuffle=False)
    out: List[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0,))


@torch.no_grad()
def export_logits_and_penultimate(
    model: nn.Module,
    base_dataset: Dataset,
    indices: Sequence[int],
    device: str,
    penultimate_module: nn.Module,
    bs: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(logits, penultimate_embeddings)`` in one forward pass.

    The penultimate embedding is captured as the INPUT to ``penultimate_module``
    (for torchvision ConvNeXt this is ``model.classifier[2]``, the linear head, so
    the captured tensor is the 768-d post-LayerNorm/flatten feature). Uses a
    forward-pre-hook that is always removed, so the model is left untouched. The
    two arrays are row-aligned with ``indices`` exactly as :func:`export_logits`.
    """
    model.to(device).eval()
    captured: dict[str, "torch.Tensor"] = {}

    def _hook(_module, inputs):
        captured["emb"] = inputs[0].detach()

    handle = penultimate_module.register_forward_pre_hook(_hook)
    try:
        subset = Subset(base_dataset, list(indices))
        loader = DataLoader(subset, batch_size=bs, shuffle=False)
        logits_out: List[np.ndarray] = []
        emb_out: List[np.ndarray] = []
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            if "emb" not in captured:
                raise RuntimeError("penultimate hook did not fire")
            logits_out.append(logits.cpu().numpy())
            emb_out.append(captured["emb"].reshape(captured["emb"].shape[0], -1).cpu().numpy())
    finally:
        handle.remove()
    if not logits_out:
        return np.zeros((0,)), np.zeros((0,))
    return np.concatenate(logits_out, axis=0), np.concatenate(emb_out, axis=0)
