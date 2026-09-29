"""Registered deterministic training primitives for the FedCORE PACS v3 study.

Nothing launches automatically. The scientific training path is deliberately
disabled until artifact-level readiness and run authorization are implemented.
``--validate-only`` checks the immutable in-container path contract without
opening any mounted input or importing torch. ``--validate-weights-only`` may
hash the five inputs and strict-load the two pretrained weights on CPU, but it
does not decode PACS records or start training.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

from fedcore.experiments.v3_contract import (
    CLIENTS,
    PROTOCOL_ID,
    ContractError,
    ModelCell,
    load_model_cells,
    sha256_file,
)
from fedcore.experiments.v3_input_binding import REGISTERED_SCIENTIFIC_INPUTS


SEED_NAMESPACE_ID = "FEDCORE-IJAR-V34-PACS-PROSPECTIVE-HSB-v2"
ARCHITECTURES = ("resnet18", "convnext_tiny")
FEDERATED_ROUNDS = 30
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
BASE_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
WEIGHT_FILENAMES = {
    "resnet18": "resnet18-f37072fd.pth",
    "convnext_tiny": "convnext_tiny-983f1562.pth",
}
INPUT_SHA256 = {spec.name: spec.sha256 for spec in REGISTERED_SCIENTIFIC_INPUTS}
WEIGHT_SHA256 = {
    architecture: INPUT_SHA256[filename]
    for architecture, filename in WEIGHT_FILENAMES.items()
}
EXPECTED_CONTAINER_PATHS = {
    "pacs_archive": Path("/inputs/PACS_mirror.zip"),
    "image_manifest": Path("/inputs/IMAGE_MANIFEST.csv"),
    "fold_manifest": Path("/inputs/FOLD_MANIFEST.csv"),
    "resnet18_weights": Path("/inputs/resnet18-f37072fd.pth"),
    "convnext_tiny_weights": Path("/inputs/convnext_tiny-983f1562.pth"),
    "output_dir": Path("/output"),
}


@dataclass(frozen=True)
class TrainingResult:
    model: Any
    completed_rounds: int
    stopped: bool
    checkpoint_path: Path


def seed_for(label: str, parts: Sequence[object]) -> int:
    compact = json.dumps(list(parts), separators=(",", ":"), ensure_ascii=True)
    payload = f"{SEED_NAMESPACE_ID}|{label}|{compact}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def torch_seed(value: int) -> int:
    return int(value) % (2**63)


def seed_registered_rngs(value: int):
    """Seed Python and torch and return the registered full-width PCG64 RNG."""

    import numpy as np
    import torch

    value = int(value)
    random.seed(value)
    torch.manual_seed(torch_seed(value))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed(value))
    return np.random.Generator(np.random.PCG64(value))


def configure_deterministic_torch() -> None:
    """Apply the registered FP32 deterministic runtime settings."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def registered_round_lr(round_index: int) -> float:
    if not 0 <= int(round_index) < FEDERATED_ROUNDS:
        raise ContractError(f"round index outside [0,{FEDERATED_ROUNDS - 1}]: {round_index}")
    round_index = int(round_index)
    if round_index < 2:
        return BASE_LEARNING_RATE * (round_index + 1) / 2.0
    return BASE_LEARNING_RATE * 0.5 * (
        1.0 + math.cos(math.pi * (round_index - 2) / 28.0)
    )


def make_registered_optimizer(model, round_index: int):
    import torch

    return torch.optim.AdamW(
        model.parameters(),
        lr=registered_round_lr(round_index),
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
    )


def _load_offline_state_dict(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or not payload:
        raise ContractError(f"pretrained weight file is not a state_dict: {path}")
    if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        payload = payload["state_dict"]
    if not all(isinstance(key, str) for key in payload):
        raise ContractError(f"pretrained state_dict contains a non-string key: {path}")
    return payload


def build_registered_model(
    architecture: str,
    weights_path: Path,
    *,
    model_init_seed_uint64: int,
    n_known: int = 4,
):
    """Build an offline ImageNet-initialized model with a fresh four-class head."""

    if architecture not in ARCHITECTURES:
        raise ContractError(f"unsupported registered architecture: {architecture!r}")
    if n_known != 4:
        raise ContractError(f"registered known-class head width is four, observed {n_known}")
    weights_path = Path(weights_path)
    if weights_path.name != WEIGHT_FILENAMES[architecture]:
        raise ContractError(f"weight filename drift for {architecture}: {weights_path.name!r}")
    observed_hash = sha256_file(weights_path)
    if observed_hash != WEIGHT_SHA256[architecture]:
        raise ContractError(f"weight SHA-256 mismatch for {architecture}: observed {observed_hash}")
    configure_deterministic_torch()
    seed_registered_rngs(model_init_seed_uint64)
    import torch
    import torchvision

    state = _load_offline_state_dict(weights_path)
    if architecture == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        model.load_state_dict(state, strict=True)
        model.fc = torch.nn.Linear(model.fc.in_features, n_known)
    else:
        model = torchvision.models.convnext_tiny(weights=None, stochastic_depth_prob=0.1)
        model.load_state_dict(state, strict=True)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, n_known)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def _cpu_state_dict(model) -> dict[str, Any]:
    return {
        key: value.detach().to(device="cpu").contiguous().clone()
        for key, value in model.state_dict().items()
    }


def state_dict_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().to(device="cpu").contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def aggregate_state_dicts_float64(
    states: Sequence[Mapping[str, Any]],
    counts: Sequence[int],
    domains: Sequence[str],
) -> dict[str, Any]:
    """Registered size-weighted FedAvg with deterministic non-float donation."""

    import torch

    if len(states) != 4 or len(counts) != 4 or tuple(domains) != CLIENTS:
        raise ContractError("FedAvg requires the four clients in registered lexical order")
    if any(int(count) <= 0 for count in counts):
        raise ContractError("all four registered clients must contain training samples")
    keys = tuple(states[0].keys())
    if any(tuple(state.keys()) != keys for state in states[1:]):
        raise ContractError("client state_dict key/order mismatch")
    total = int(sum(int(count) for count in counts))
    donor = min(range(4), key=lambda index: (-int(counts[index]), domains[index]))
    result: dict[str, Any] = {}
    for key in keys:
        reference = states[0][key]
        if any(
            state[key].shape != reference.shape or state[key].dtype != reference.dtype
            for state in states[1:]
        ):
            raise ContractError(f"client tensor schema mismatch for {key}")
        if torch.is_floating_point(reference):
            accumulator = torch.zeros_like(reference, dtype=torch.float64, device="cpu")
            for state, count in zip(states, counts):
                accumulator.add_(
                    state[key].detach().to(device="cpu", dtype=torch.float64),
                    alpha=float(int(count)),
                )
            result[key] = accumulator.div_(float(total)).to(dtype=reference.dtype)
        else:
            result[key] = states[donor][key].detach().to(device="cpu").clone()
    return result


def train_registered_client(
    global_model,
    dataset,
    *,
    round_index: int,
    local_seed_uint64: int,
    device: str,
) -> tuple[dict[str, Any], int]:
    """Perform one registered one-epoch client update with a fresh optimizer."""

    import torch
    from torch.utils.data import DataLoader

    if len(dataset) <= 0:
        raise ContractError("registered client dataset is empty")
    seed_registered_rngs(local_seed_uint64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(torch_seed(local_seed_uint64))
    local = copy.deepcopy(global_model).to(device)
    local.train()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )
    optimizer = make_registered_optimizer(local, round_index)
    steps = 0
    for _ in range(LOCAL_EPOCHS):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = local(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite registered training loss")
            loss.backward()
            optimizer.step()
            steps += 1
    state = _cpu_state_dict(local)
    if any(
        torch.is_floating_point(value) and not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise FloatingPointError("nonfinite tensor after registered client update")
    return state, steps


def run_registered_round(
    global_model,
    client_datasets: Mapping[str, Any],
    *,
    architecture: str,
    split: int,
    nominal_training_seed: int,
    round_index: int,
    device: str,
) -> tuple[Any, dict[str, int]]:
    """Run one complete registered round and return the aggregated global model."""

    if tuple(client_datasets.keys()) != CLIENTS:
        raise ContractError("client dataset mapping must follow the registered client order")
    states: list[dict[str, Any]] = []
    counts: list[int] = []
    steps: dict[str, int] = {}
    for domain in CLIENTS:
        dataset = client_datasets[domain]
        local_seed = seed_for(
            "local_training",
            [architecture, int(split), int(nominal_training_seed), int(round_index), domain],
        )
        state, n_steps = train_registered_client(
            global_model,
            dataset,
            round_index=round_index,
            local_seed_uint64=local_seed,
            device=device,
        )
        states.append(state)
        counts.append(len(dataset))
        steps[domain] = n_steps
    global_model.load_state_dict(
        aggregate_state_dicts_float64(states, counts, CLIENTS), strict=True
    )
    return global_model, steps


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    import torch

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)


def save_round_checkpoint(
    destination: Path,
    model,
    *,
    cell: ModelCell,
    completed_rounds: int,
    weight_sha256: str,
    authorization_sha256: Mapping[str, str],
    client_steps: Mapping[str, int],
) -> dict[str, Any]:
    if not 1 <= int(completed_rounds) <= FEDERATED_ROUNDS:
        raise ContractError("completed_rounds outside the registered range")
    state = _cpu_state_dict(model)
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": cell.model_id,
        "architecture": cell.architecture,
        "split": cell.split,
        "nominal_training_seed": cell.nominal_training_seed,
        "model_init_seed_uint64": cell.model_init_seed_uint64,
        "completed_rounds": int(completed_rounds),
        "rounds_requested": FEDERATED_ROUNDS,
        "weight_sha256": weight_sha256,
        "readiness_sha256": authorization_sha256["readiness_sha256"],
        "input_binding_sha256": authorization_sha256["input_binding_sha256"],
        "run_authorization_sha256": authorization_sha256["run_authorization_sha256"],
        "client_steps": {domain: int(client_steps[domain]) for domain in CLIENTS},
        "model_state_sha256": state_dict_sha256(state),
        "model_state_dict": state,
    }
    _atomic_torch_save(payload, Path(destination))
    return payload


def load_round_checkpoint(
    path: Path,
    *,
    cell: ModelCell,
    weight_sha256: str,
    authorization_sha256: Mapping[str, str],
) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "model_id": cell.model_id,
        "architecture": cell.architecture,
        "split": cell.split,
        "nominal_training_seed": cell.nominal_training_seed,
        "model_init_seed_uint64": cell.model_init_seed_uint64,
        "rounds_requested": FEDERATED_ROUNDS,
        "weight_sha256": weight_sha256,
        "readiness_sha256": authorization_sha256["readiness_sha256"],
        "input_binding_sha256": authorization_sha256["input_binding_sha256"],
        "run_authorization_sha256": authorization_sha256["run_authorization_sha256"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ContractError(f"checkpoint {key} mismatch")
    completed = int(payload.get("completed_rounds", -1))
    if not 1 <= completed <= FEDERATED_ROUNDS:
        raise ContractError("checkpoint completed_rounds is invalid")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ContractError("checkpoint model_state_dict is absent")
    if state_dict_sha256(state) != payload.get("model_state_sha256"):
        raise ContractError("checkpoint state digest mismatch")
    return payload


def train_registered_model(
    cell: ModelCell,
    client_datasets: Mapping[str, Any],
    *,
    weights_path: Path,
    output_dir: Path,
    authorization_sha256: Mapping[str, str],
    device: str,
    resume: bool,
    should_stop: Callable[[], bool] | None = None,
) -> TrainingResult:
    """Train one frozen model cell, checkpointing at complete round boundaries."""

    if cell.architecture not in ARCHITECTURES:
        raise ContractError(f"unregistered model architecture: {cell.architecture}")
    configure_deterministic_torch()
    model = build_registered_model(
        cell.architecture,
        weights_path,
        model_init_seed_uint64=cell.model_init_seed_uint64,
    ).to(device)
    checkpoint = Path(output_dir) / "checkpoints" / cell.model_id / "latest.pt"
    start_round = 0
    if resume:
        if not checkpoint.is_file():
            raise ContractError("resume requested but registered checkpoint is absent")
        payload = load_round_checkpoint(
            checkpoint,
            cell=cell,
            weight_sha256=WEIGHT_SHA256[cell.architecture],
            authorization_sha256=authorization_sha256,
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        start_round = int(payload["completed_rounds"])
    elif checkpoint.exists():
        raise ContractError("checkpoint already exists; use explicit resume")

    completed = start_round
    for round_index in range(start_round, FEDERATED_ROUNDS):
        if should_stop is not None and should_stop():
            return TrainingResult(model, completed, True, checkpoint)
        model, client_steps = run_registered_round(
            model,
            client_datasets,
            architecture=cell.architecture,
            split=cell.split,
            nominal_training_seed=cell.nominal_training_seed,
            round_index=round_index,
            device=device,
        )
        completed = round_index + 1
        save_round_checkpoint(
            checkpoint,
            model,
            cell=cell,
            completed_rounds=completed,
            weight_sha256=WEIGHT_SHA256[cell.architecture],
            authorization_sha256=authorization_sha256,
            client_steps=client_steps,
        )
    return TrainingResult(model, completed, False, checkpoint)


def verify_bound_input_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    """Verify the five mounted files before any model or image is opened."""

    observed: dict[str, str] = {}
    for name, expected in INPUT_SHA256.items():
        path = Path(paths[name])
        if path.name != name or not path.is_file() or path.is_symlink():
            raise ContractError(f"invalid mounted scientific input: {path}")
        digest = sha256_file(path)
        if digest != expected:
            raise ContractError(f"mounted scientific input SHA-256 mismatch: {name}")
        observed[name] = digest
    return observed


def _validate_cli_binding_paths(args: argparse.Namespace) -> None:
    for field, expected in EXPECTED_CONTAINER_PATHS.items():
        observed = Path(getattr(args, field))
        if observed != expected:
            raise ContractError(
                f"in-container path drift for {field}: expected {expected}, observed {observed}"
            )


def _require_scalar_authorization(args: argparse.Namespace) -> dict[str, str]:
    """Refuse scientific training until artifact-level authorization is implemented.

    Scalar digests are identifiers, not authorization evidence.  Accepting three
    caller-supplied hexadecimal strings would let a direct container invocation
    bypass the sealed host gate.  The static input/runtime binding therefore
    keeps the training entrypoint closed.  A later release must mount and verify
    the launch plan, execution-readiness gate, input-binding receipt, and signed
    run-authorization artifact before this function may return.
    """

    del args
    raise ContractError(
        "scientific training refused: artifact-bound authorization is not implemented"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacs-archive", required=True)
    parser.add_argument("--image-manifest", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--resnet18-weights", required=True)
    parser.add_argument("--convnext-tiny-weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--readiness-sha256")
    parser.add_argument("--input-binding-sha256")
    parser.add_argument("--run-authorization-sha256")
    parser.add_argument("--training-authorized", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-weights-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_cli_binding_paths(args)
    if args.validate_only and args.validate_weights_only:
        raise ContractError("choose only one validation profile")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "status": "PASS_STATIC_TRAINING_CLI_BINDING",
                    "PACS_opened": False,
                    "torch_imported_by_cli": False,
                    "GPU_used": False,
                    "training_started": False,
                },
                sort_keys=True,
            )
        )
        return 0


    input_paths = {
        "PACS_mirror.zip": Path(args.pacs_archive),
        "IMAGE_MANIFEST.csv": Path(args.image_manifest),
        "FOLD_MANIFEST.csv": Path(args.fold_manifest),
        "resnet18-f37072fd.pth": Path(args.resnet18_weights),
        "convnext_tiny-983f1562.pth": Path(args.convnext_tiny_weights),
    }
    if args.validate_weights_only:
        observed = verify_bound_input_hashes(input_paths)
        models = {}
        for architecture in ARCHITECTURES:
            model = build_registered_model(
                architecture,
                input_paths[WEIGHT_FILENAMES[architecture]],
                model_init_seed_uint64=0,
            )
            head = model.fc if architecture == "resnet18" else model.classifier[2]
            models[architecture] = {
                "head_out_features": int(head.out_features),
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                "weight_sha256": observed[WEIGHT_FILENAMES[architecture]],
            }
        print(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "status": "PASS_CPU_STRICT_WEIGHT_BINDING",
                    "models": models,
                    "all_five_input_hashes_verified": True,
                    "PACS_archive_hashed": True,
                    "PACS_records_decoded": False,
                    "PACS_labels_inspected": False,
                    "GPU_used": False,
                    "training_started": False,
                },
                sort_keys=True,
            )
        )
        return 0

    authorization = _require_scalar_authorization(args)
    if not args.model_id:
        raise ContractError("scientific training refused: --model-id is required")
    cells = {cell.model_id: cell for cell in load_model_cells()}
    try:
        cell = cells[args.model_id]
    except KeyError as exc:
        raise ContractError(f"scientific training refused: unknown model_id {args.model_id!r}") from exc

    verify_bound_input_hashes(input_paths)

    from fedcore.experiments.v3_pacs_data import (
        PACSZipDataset,
        build_registered_transforms,
        load_pacs_records,
        records_by_use_and_domain,
    )

    records = load_pacs_records(
        Path(args.image_manifest),
        Path(args.fold_manifest),
        split=cell.split,
        known_classes=cell.known_classes,
        expected_image_count=9991,
    )
    train_records = records_by_use_and_domain(records, "train")
    train_transform, _ = build_registered_transforms()
    datasets = {
        domain: PACSZipDataset(
            Path(args.pacs_archive),
            train_records[domain],
            transform=train_transform,
            allow_unknown=False,
        )
        for domain in CLIENTS
    }
    weights_path = input_paths[WEIGHT_FILENAMES[cell.architecture]]
    result = train_registered_model(
        cell,
        datasets,
        weights_path=weights_path,
        output_dir=Path(args.output_dir),
        authorization_sha256=authorization,
        device=args.device,
        resume=bool(args.resume),
    )
    print(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "model_id": cell.model_id,
                "completed_rounds": result.completed_rounds,
                "stopped": result.stopped,
                "checkpoint": str(result.checkpoint_path),
                "checkpoint_sha256": sha256_file(result.checkpoint_path),
            },
            sort_keys=True,
        )
    )
    return 0 if result.completed_rounds == FEDERATED_ROUNDS and not result.stopped else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURES",
    "BASE_LEARNING_RATE",
    "BATCH_SIZE",
    "EXPECTED_CONTAINER_PATHS",
    "FEDERATED_ROUNDS",
    "INPUT_SHA256",
    "LOCAL_EPOCHS",
    "SEED_NAMESPACE_ID",
    "TrainingResult",
    "WEIGHT_FILENAMES",
    "WEIGHT_SHA256",
    "aggregate_state_dicts_float64",
    "build_registered_model",
    "configure_deterministic_torch",
    "load_round_checkpoint",
    "main",
    "make_registered_optimizer",
    "registered_round_lr",
    "run_registered_round",
    "save_round_checkpoint",
    "seed_for",
    "seed_registered_rngs",
    "state_dict_sha256",
    "torch_seed",
    "train_registered_client",
    "train_registered_model",
    "verify_bound_input_hashes",
]
