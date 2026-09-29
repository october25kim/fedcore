"""Focused fail-closed tests for the v3 five-input and mount contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pytest

from fedcore.experiments.v3_contract import ContractError
from fedcore.experiments.v3_input_binding import ScientificInputSpec, bind_scientific_inputs
from fedcore.experiments.v3_validate import validate_mount_scope


NAMES = (
    "PACS_mirror.zip",
    "IMAGE_MANIFEST.csv",
    "FOLD_MANIFEST.csv",
    "resnet18-f37072fd.pth",
    "convnext_tiny-983f1562.pth",
)


def _specs(source_dir: Path) -> tuple[ScientificInputSpec, ...]:
    result = []
    for index, name in enumerate(NAMES):
        payload = (f"fixture-{index}-" * (index + 1)).encode("ascii")
        path = source_dir / name
        path.write_bytes(payload)
        result.append(
            ScientificInputSpec(
                name=name,
                source_path=str(path),
                container_path=f"/inputs/{name}",
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
        )
    return tuple(result)


def test_exact_five_input_binding_is_read_only_canonical_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    staged = tmp_path / "isolated" / "inputs"
    receipt_path = tmp_path / "seals" / "INPUT_BINDING.json"
    source.mkdir()
    specs = _specs(source)
    first = bind_scientific_inputs(staged, specs=specs, receipt_path=receipt_path)
    second = bind_scientific_inputs(staged, specs=specs, receipt_path=receipt_path)
    assert first == second
    assert first["status"] == "PASS_FIVE_INPUT_BINDING"
    assert first["isolated_input_file_count"] == 5
    assert sorted(item.name for item in staged.iterdir()) == sorted(NAMES)
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == first
    assert receipt_path.read_bytes().endswith(b"\n")
    for name in NAMES:
        path = staged / name
        assert path.is_file() and not path.is_symlink()
        assert stat.S_IMODE(path.stat().st_mode) == 0o444


@pytest.mark.parametrize("failure", ("missing", "bad_size", "bad_hash", "source_symlink"))
def test_source_contract_failures_do_not_emit_receipt(tmp_path: Path, failure: str) -> None:
    source = tmp_path / "source"
    staged = tmp_path / "inputs"
    receipt = tmp_path / "receipt.json"
    source.mkdir()
    specs = list(_specs(source))
    target = Path(specs[0].source_path)
    if failure == "missing":
        target.unlink()
    elif failure == "bad_size":
        target.write_bytes(target.read_bytes() + b"x")
    elif failure == "bad_hash":
        payload = target.read_bytes()
        target.write_bytes(b"x" * len(payload))
    else:
        original = target.read_bytes()
        target.unlink()
        real = source / "real.bin"
        real.write_bytes(original)
        target.symlink_to(real)
    with pytest.raises(ContractError):
        bind_scientific_inputs(staged, specs=specs, receipt_path=receipt)
    assert not receipt.exists()


def test_staged_extra_wrong_file_and_symlink_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    specs = _specs(source)

    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()
    (extra_dir / "unexpected").write_bytes(b"x")
    with pytest.raises(ContractError, match="unexpected isolated input"):
        bind_scientific_inputs(extra_dir, specs=specs)

    wrong_dir = tmp_path / "wrong"
    wrong_dir.mkdir()
    (wrong_dir / NAMES[0]).write_bytes(b"wrong")
    with pytest.raises(ContractError, match="staged input (size|SHA-256) mismatch"):
        bind_scientific_inputs(wrong_dir, specs=specs)

    link_dir = tmp_path / "link"
    link_dir.mkdir()
    (link_dir / NAMES[0]).symlink_to(Path(specs[0].source_path))
    with pytest.raises(ContractError, match="must not be a symlink"):
        bind_scientific_inputs(link_dir, specs=specs)


def _mount(source: Path, destination: str, read_only: object) -> dict[str, object]:
    return {
        "type": "bind",
        "source": str(source),
        "destination": destination,
        "read_only": read_only,
    }


def test_mount_boolean_is_strict_and_false_string_is_not_truthy(tmp_path: Path) -> None:
    row = _mount(tmp_path / "output", "/output", False)
    observed = [{**row, "read_only": "false"}]
    validate_mount_scope(observed, [row], forbidden_roots=[])
    with pytest.raises(ContractError, match="not a boolean"):
        validate_mount_scope([{**row, "read_only": 1}], [row], forbidden_roots=[])


def test_duplicate_mounts_and_destinations_fail_closed(tmp_path: Path) -> None:
    first = _mount(tmp_path / "one", "/inputs/one", True)
    second = _mount(tmp_path / "two", "/inputs/one", True)
    with pytest.raises(ContractError, match="duplicate observed mount"):
        validate_mount_scope([first, first], [first], forbidden_roots=[])
    with pytest.raises(ContractError, match="duplicate observed destination"):
        validate_mount_scope([first, second], [first, second], forbidden_roots=[])
    with pytest.raises(ContractError, match="duplicate expected mount"):
        validate_mount_scope([first], [first, first], forbidden_roots=[])


def test_forbidden_source_is_rejected_even_when_expected(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    source = legacy / "PACS_mirror.zip"
    row = _mount(source, "/inputs/PACS_mirror.zip", True)
    with pytest.raises(ContractError, match="forbidden source"):
        validate_mount_scope([row], [row], forbidden_roots=[legacy])


def test_receipt_cannot_live_inside_exact_input_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    specs = _specs(source)
    staged = tmp_path / "inputs"
    with pytest.raises(ContractError, match="receipt must be stored outside"):
        bind_scientific_inputs(
            staged,
            specs=specs,
            receipt_path=staged / "nested" / "INPUT_BINDING.json",
        )
