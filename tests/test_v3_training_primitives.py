"""PACS-free tests for the sealed v3 data and training primitives."""

from __future__ import annotations

import builtins
import csv
import hashlib
import io
from pathlib import Path
import zipfile

import pytest

from fedcore.experiments.v3_contract import CLIENTS, ContractError, ModelCell
from fedcore.experiments.v3_pacs_data import (
    FOLD_MANIFEST_COLUMNS,
    IMAGE_MANIFEST_COLUMNS,
    PACSDataError,
    PACSZipDataset,
    archive_member_from_manifest_path,
    build_registered_transforms,
    load_pacs_train_records,
    records_by_use_and_domain,
)
from fedcore.experiments import v3_train as training


def _write_csv(path: Path, columns, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _png_bytes(size=(8, 8), color=(20, 40, 60)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _synthetic_manifests(tmp_path: Path):
    images = []
    folds = []
    members = {}
    ordinal = 0
    # Every domain has nonempty train/proposal/audit support and an explicitly
    # unused unknown whose global role remains train.
    definitions = (
        ("dog", "train", "train", 0),
        ("horse", "proposal", "proposal", -1),
        ("elephant", "audit", "audit", 1),
        ("house", "train", "unused", -1),
    )
    for domain in CLIENTS:
        for class_name, role, use, label in definitions:
            identity = f"{ordinal:064x}"
            member = f"PACS/{domain}/{class_name}/synthetic_{ordinal:03d}.png"
            path = f"data/v1_pacs/extracted/{member}"
            images.append(
                {
                    "path": path,
                    "domain": domain,
                    "class": class_name,
                    "identity": identity,
                    "width": "8",
                    "height": "8",
                    "role": role,
                }
            )
            folds.append(
                {
                    "split": "0",
                    "identity": identity,
                    "domain": domain,
                    "class": class_name,
                    "role": role,
                    "use": use,
                    "label": str(label),
                    "path": path,
                }
            )
            members[member] = _png_bytes(color=(20 + ordinal, 40, 60))
            ordinal += 1
    image_path = tmp_path / "IMAGE_MANIFEST.csv"
    fold_path = tmp_path / "FOLD_MANIFEST.csv"
    archive_path = tmp_path / "PACS_mirror.zip"
    _write_csv(image_path, IMAGE_MANIFEST_COLUMNS, images)
    _write_csv(fold_path, FOLD_MANIFEST_COLUMNS, folds)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return image_path, fold_path, archive_path, folds


def test_train_manifest_join_excludes_registered_unknown_unused_rows(tmp_path):
    image_path, fold_path, _, _ = _synthetic_manifests(tmp_path)
    records = load_pacs_train_records(
        image_path,
        fold_path,
        split=0,
        known_classes=("dog", "elephant", "giraffe", "guitar"),
        expected_image_count=16,
    )
    assert len(records) == 4
    assert all(record.use == "train" for record in records)
    grouped = records_by_use_and_domain(records, "train")
    assert tuple(grouped) == CLIENTS
    assert all(len(grouped[domain]) == 1 for domain in CLIENTS)
    assert all(grouped[domain][0].class_name == "dog" for domain in CLIENTS)


def test_manifest_rejects_unknown_reallocation(tmp_path):
    image_path, fold_path, _, folds = _synthetic_manifests(tmp_path)
    for row in folds:
        if row["class"] == "house":
            row["use"] = "train"
            break
    _write_csv(fold_path, FOLD_MANIFEST_COLUMNS, folds)
    with pytest.raises(PACSDataError, match="unknown label entered train stage"):
        load_pacs_train_records(
            image_path,
            fold_path,
            split=0,
            known_classes=("dog", "elephant", "giraffe", "guitar"),
        )


def test_zip_dataset_and_registered_transforms_are_pacs_free_fixture(tmp_path):
    image_path, fold_path, archive_path, _ = _synthetic_manifests(tmp_path)
    records = load_pacs_train_records(
        image_path,
        fold_path,
        split=0,
        known_classes=("dog", "elephant", "giraffe", "guitar"),
    )
    train = records_by_use_and_domain(records, "train")["art_painting"]
    dataset = PACSZipDataset(archive_path, train, transform=None, allow_unknown=False)
    image, label = dataset[0]
    assert image.mode == "RGB" and image.size == (8, 8) and label == 0
    archive_handle = dataset._archive
    image_again, label_again = dataset[0]
    assert dataset._archive is archive_handle
    assert image_again.tobytes() == image.tobytes() and label_again == label
    dataset.close()
    assert dataset._archive is None
    train_transform, inference_transform = build_registered_transforms()
    assert train_transform.transforms[1].size == (224, 224)
    assert train_transform.transforms[1].scale == (0.08, 1.0)
    assert inference_transform.transforms[1].size == 256
    assert inference_transform.transforms[2].size == (224, 224)


def test_archive_path_mapping_is_fail_closed():
    assert (
        archive_member_from_manifest_path(
            "data/v1_pacs/extracted/PACS/sketch/dog/pic.png"
        )
        == "PACS/sketch/dog/pic.png"
    )
    with pytest.raises(PACSDataError):
        archive_member_from_manifest_path("outside/sketch/dog/pic.png")


def test_seed_and_learning_rate_match_registered_manifest():
    assert training.seed_for("model_init", ["resnet18", 0, 0]) == 11902935454557521285
    assert training.registered_round_lr(0) == 0.00005
    assert training.registered_round_lr(1) == 0.0001
    assert training.registered_round_lr(2) == 0.0001
    assert training.registered_round_lr(29) == pytest.approx(
        0.0001 * 0.5 * (1 + __import__("math").cos(__import__("math").pi * 27 / 28))
    )
    with pytest.raises(ContractError):
        training.registered_round_lr(30)


def test_float64_fedavg_and_largest_lexical_nonfloat_donor():
    torch = pytest.importorskip("torch")
    states = []
    for index in range(4):
        states.append(
            {
                "weight": torch.tensor([float(index + 1)], dtype=torch.float32),
                "counter": torch.tensor(index + 10, dtype=torch.int64),
            }
        )
    result = training.aggregate_state_dicts_float64(
        states, [1, 5, 5, 2], CLIENTS
    )
    expected = (1 * 1 + 2 * 5 + 3 * 5 + 4 * 2) / 13
    assert result["weight"].dtype == torch.float32
    assert result["weight"].item() == pytest.approx(expected)
    # cartoon and photo tie at five records; lexical cartoon donates the buffer.
    assert result["counter"].item() == 11


class _FakeResNet:
    pass


@pytest.mark.parametrize("architecture", ["resnet18", "convnext_tiny"])
def test_offline_weight_load_and_fresh_four_class_head(tmp_path, monkeypatch, architecture):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")

    class FakeResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 3)
            self.fc = torch.nn.Linear(3, 1000)

        def forward(self, value):
            return self.fc(self.backbone(value))

    class FakeConvNeXt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 3)
            self.classifier = torch.nn.Sequential(
                torch.nn.Identity(), torch.nn.Identity(), torch.nn.Linear(3, 1000)
            )

        def forward(self, value):
            return self.classifier(self.backbone(value))

    if architecture == "resnet18":
        source = FakeResNet()

        def factory(*, weights):
            assert weights is None
            return FakeResNet()

        monkeypatch.setattr(torchvision.models, "resnet18", factory)
    else:
        source = FakeConvNeXt()

        def factory(*, weights, stochastic_depth_prob):
            assert weights is None and stochastic_depth_prob == 0.1
            return FakeConvNeXt()

        monkeypatch.setattr(torchvision.models, "convnext_tiny", factory)
    path = tmp_path / training.WEIGHT_FILENAMES[architecture]
    torch.save(source.state_dict(), path)
    monkeypatch.setitem(training.WEIGHT_SHA256, architecture, hashlib.sha256(path.read_bytes()).hexdigest())
    first = training.build_registered_model(
        architecture, path, model_init_seed_uint64=123456789
    )
    second = training.build_registered_model(
        architecture, path, model_init_seed_uint64=123456789
    )
    head1 = first.fc if architecture == "resnet18" else first.classifier[2]
    head2 = second.fc if architecture == "resnet18" else second.classifier[2]
    assert head1.out_features == 4
    assert torch.equal(head1.weight, head2.weight)
    assert torch.equal(first.backbone.weight, source.backbone.weight)
    assert all(parameter.requires_grad for parameter in first.parameters())


def _tiny_clients(torch):
    from torch.utils.data import TensorDataset

    clients = {}
    for index, domain in enumerate(CLIENTS):
        x = torch.tensor(
            [[index / 10.0, 0.0], [index / 10.0, 1.0]], dtype=torch.float32
        )
        y = torch.tensor([index % 4, (index + 1) % 4], dtype=torch.int64)
        clients[domain] = TensorDataset(x, y)
    return clients


def test_round_checkpoint_resume_is_exact_on_synthetic_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    cell = ModelCell(
        model_id="fixture",
        architecture="resnet18",
        split=0,
        nominal_training_seed=0,
        model_init_seed_uint64=123,
        known_classes=("dog", "elephant", "giraffe", "guitar"),
        unknown_classes=("horse", "house", "person"),
    )
    authorization = {
        "readiness_sha256": "1" * 64,
        "input_binding_sha256": "2" * 64,
        "run_authorization_sha256": "3" * 64,
    }
    torch.manual_seed(77)
    initial = torch.nn.Linear(2, 4)
    initial_state = {key: value.clone() for key, value in initial.state_dict().items()}
    clients = _tiny_clients(torch)

    uninterrupted = torch.nn.Linear(2, 4)
    uninterrupted.load_state_dict(initial_state)
    uninterrupted, _ = training.run_registered_round(
        uninterrupted,
        clients,
        architecture="resnet18",
        split=0,
        nominal_training_seed=0,
        round_index=0,
        device="cpu",
    )
    state_after_round_zero = {
        key: value.clone() for key, value in uninterrupted.state_dict().items()
    }
    uninterrupted, _ = training.run_registered_round(
        uninterrupted,
        clients,
        architecture="resnet18",
        split=0,
        nominal_training_seed=0,
        round_index=1,
        device="cpu",
    )

    checkpoint = tmp_path / "latest.pt"
    round_zero = torch.nn.Linear(2, 4)
    round_zero.load_state_dict(state_after_round_zero)
    training.save_round_checkpoint(
        checkpoint,
        round_zero,
        cell=cell,
        completed_rounds=1,
        weight_sha256="a" * 64,
        authorization_sha256=authorization,
        client_steps={domain: 1 for domain in CLIENTS},
    )
    payload = training.load_round_checkpoint(
        checkpoint,
        cell=cell,
        weight_sha256="a" * 64,
        authorization_sha256=authorization,
    )
    resumed = torch.nn.Linear(2, 4)
    resumed.load_state_dict(payload["model_state_dict"])
    resumed, _ = training.run_registered_round(
        resumed,
        clients,
        architecture="resnet18",
        split=0,
        nominal_training_seed=0,
        round_index=1,
        device="cpu",
    )
    for key in uninterrupted.state_dict():
        assert torch.equal(uninterrupted.state_dict()[key], resumed.state_dict()[key])


def _static_cli_args():
    return [
        "--pacs-archive",
        "/inputs/PACS_mirror.zip",
        "--image-manifest",
        "/inputs/IMAGE_MANIFEST.csv",
        "--fold-manifest",
        "/inputs/FOLD_MANIFEST.csv",
        "--resnet18-weights",
        "/inputs/resnet18-f37072fd.pth",
        "--convnext-tiny-weights",
        "/inputs/convnext_tiny-983f1562.pth",
        "--output-dir",
        "/output",
    ]


def test_validate_only_does_not_import_torch_or_open_pacs(monkeypatch, capsys):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "torchvision", "PIL"}:
            raise AssertionError(f"validate-only imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert training.main([*_static_cli_args(), "--validate-only"]) == 0
    output = capsys.readouterr().out
    assert "PASS_STATIC_TRAINING_CLI_BINDING" in output
    assert '"PACS_opened": false' in output


def test_normal_cli_fails_before_input_access_without_authorization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("input access occurred before authorization")

    monkeypatch.setattr(training, "verify_bound_input_hashes", forbidden)
    with pytest.raises(ContractError, match="artifact-bound authorization"):
        training.main([*_static_cli_args(), "--model-id", "fixture"])


def test_weight_binding_profile_never_requires_or_starts_training(monkeypatch, capsys):
    torch = pytest.importorskip("torch")

    class BoundModel(torch.nn.Module):
        def __init__(self, architecture):
            super().__init__()
            if architecture == "resnet18":
                self.fc = torch.nn.Linear(2, 4)
            else:
                self.classifier = torch.nn.Sequential(
                    torch.nn.Identity(), torch.nn.Identity(), torch.nn.Linear(2, 4)
                )

    monkeypatch.setattr(
        training,
        "verify_bound_input_hashes",
        lambda paths: dict(training.INPUT_SHA256),
    )
    monkeypatch.setattr(
        training,
        "build_registered_model",
        lambda architecture, weights_path, model_init_seed_uint64: BoundModel(architecture),
    )
    assert training.main([*_static_cli_args(), "--validate-weights-only"]) == 0
    report = __import__("json").loads(capsys.readouterr().out)
    assert report["status"] == "PASS_CPU_STRICT_WEIGHT_BINDING"
    assert report["all_five_input_hashes_verified"] is True
    assert report["PACS_records_decoded"] is False
    assert report["GPU_used"] is False
    assert report["training_started"] is False
