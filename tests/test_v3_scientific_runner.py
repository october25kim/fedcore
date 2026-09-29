"""PACS-free tests for host-side v3 scientific launch staging."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from fedcore.experiments.v3_contract import PROTOCOL_ID, ContractError, canonical_json_bytes
from fedcore.experiments.v3_input_binding import bind_scientific_inputs
from fedcore.experiments.v3_train import main as training_main
from fedcore.experiments.v3_scientific_runner import (
    EXPECTED_GPU_UUID,
    PREBINDING_STATUS,
    ScientificInputSpec,
    bind_authorized_launch,
    build_launch_plan,
    fixed_model_cells,
    initial_cell_state,
    validate_cell_state,
    validate_cell_state_transition,
    validate_empty_output_dir,
    validate_launch_plan_integrity,
    validate_prebinding_gate,
    validate_readiness_gate,
    validate_run_authorization,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _prebinding(path: Path) -> Path:
    _write_json(
        path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": PREBINDING_STATUS,
            "scientific_runner_bound": False,
            "scientific_inputs_bound": False,
            "runs_scientific_empty": True,
            "training_authorized": False,
            "pacs_training_started": False,
            "pacs_inference_started": False,
        },
    )
    return path


def _readiness(path: Path, plan: dict[str, object]) -> Path:
    _write_json(
        path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "PASS_EXECUTION_READY",
            "end_to_end_scientific_runner_bound": True,
            "scientific_inputs_bound": True,
            "scientific_mount_receipt_bound": True,
            "runs_scientific_empty": True,
            "training_authorized": False,
            "pacs_training_started": False,
            "pacs_inference_started": False,
            "launch_plan_sha256": plan["launch_plan_sha256"],
            "input_binding_sha256": plan["input_binding_sha256"],
            "container_image_id": plan["image"]["id"],
        },
    )
    return path


def _inputs(
    tmp_path: Path,
) -> tuple[Path, tuple[ScientificInputSpec, ...], Path]:
    names = (
        "PACS_mirror.zip",
        "IMAGE_MANIFEST.csv",
        "FOLD_MANIFEST.csv",
        "resnet18-f37072fd.pth",
        "convnext_tiny-983f1562.pth",
    )
    specs: list[ScientificInputSpec] = []
    source_dir = tmp_path / "original-sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        source = source_dir / name
        source.write_bytes(f"synthetic-{index}-{name}".encode())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = Path("/inputs") / name
        spec = ScientificInputSpec(
            name=name,
            source_path=str(source),
            container_path=str(destination),
            sha256=digest,
            size_bytes=source.stat().st_size,
        )
        specs.append(spec)
    binding = tmp_path / "INPUT_BINDING.json"
    staged = tmp_path / "isolated-inputs"
    bind_scientific_inputs(
        staged,
        specs=tuple(specs),
        receipt_path=binding,
    )
    return binding, tuple(specs), staged


def _plan(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    gate = _prebinding(tmp_path / "PREBINDING_GATE.json")
    binding, specs, staged = _inputs(tmp_path)
    output = tmp_path / "scientific-output"
    output.mkdir()
    plan = build_launch_plan(
        prebinding_gate=gate,
        input_binding=binding,
        output_dir=output,
        specs=specs,
        expected_input_dir=staged,
        expected_output_dir=output,
        image_tag="synthetic-image:sealed",
        image_id="sha256:" + "a" * 64,
    )
    return plan, gate, binding


def test_fixed_cell_order_and_initial_state_are_exact() -> None:
    cells = fixed_model_cells()
    assert len(cells) == 30
    assert [
        (cell.split, cell.nominal_training_seed, cell.architecture) for cell in cells[:6]
    ] == [
        (0, 0, "resnet18"),
        (0, 0, "convnext_tiny"),
        (0, 1, "resnet18"),
        (0, 1, "convnext_tiny"),
        (0, 2, "resnet18"),
        (0, 2, "convnext_tiny"),
    ]
    state = initial_cell_state(cells)
    validate_cell_state(state, cells)
    assert [row["ordinal"] for row in state["cells"]] == list(range(30))
    assert {row["stage"] for row in state["cells"]} == {"NOT_STARTED"}


def test_cell_state_rejects_reordering_skips_and_concurrency() -> None:
    cells = fixed_model_cells()
    state = initial_cell_state(cells)
    reordered = copy.deepcopy(state)
    reordered["cells"][0], reordered["cells"][1] = (
        reordered["cells"][1],
        reordered["cells"][0],
    )
    with pytest.raises(ContractError, match="order drift"):
        validate_cell_state(reordered, cells)
    skipped = copy.deepcopy(state)
    skipped["cells"][1]["stage"] = "COMPLETE"
    with pytest.raises(ContractError, match="terminal cell"):
        validate_cell_state(skipped, cells)
    concurrent = copy.deepcopy(state)
    concurrent["cells"][0]["stage"] = "TRAINING"
    concurrent["cells"][1]["stage"] = "TRAINING"
    with pytest.raises(ContractError, match="more than one active"):
        validate_cell_state(concurrent, cells)


def test_cell_state_transition_allows_one_step_and_rejects_stage_skip() -> None:
    cells = fixed_model_cells()
    initial = initial_cell_state(cells)
    training = copy.deepcopy(initial)
    training["cells"][0]["stage"] = "TRAINING"
    validate_cell_state_transition(initial, training, cells)
    checkpoint = copy.deepcopy(training)
    checkpoint["cells"][0]["stage"] = "CHECKPOINT_SEALED"
    validate_cell_state_transition(training, checkpoint, cells)
    skipped = copy.deepcopy(training)
    skipped["cells"][0]["stage"] = "PROPOSAL_SEALED"
    with pytest.raises(ContractError, match="invalid cell-stage transition"):
        validate_cell_state_transition(training, skipped, cells)
    two_changes = copy.deepcopy(training)
    two_changes["cells"][0]["stage"] = "CHECKPOINT_SEALED"
    two_changes["cells"][1]["stage"] = "MODEL_FAILURE"
    with pytest.raises(ContractError):
        validate_cell_state_transition(training, two_changes, cells)


def test_prebinding_then_readiness_do_not_authorize_training(tmp_path: Path) -> None:
    plan, prebinding, _ = _plan(tmp_path / "plan")
    observed_prebinding = validate_prebinding_gate(prebinding)
    assert observed_prebinding["status"] == PREBINDING_STATUS
    gate = _readiness(tmp_path / "EXECUTION_READINESS.json", plan)
    with pytest.raises(ContractError, match="end-to-end artifact-bound runner"):
        validate_readiness_gate(gate, launch_plan=plan)
    assert plan["execution_allowed"] is False
    assert all(cell["docker_argv"] is None for cell in plan["cell_launches"])
    missing = tmp_path / "RUN_AUTHORIZATION.json"
    with pytest.raises(ContractError, match="end-to-end artifact-bound runner"):
        validate_run_authorization(missing, plan, gate)


def test_plan_has_exact_mount_scope_gpu_and_no_execution(tmp_path: Path) -> None:
    plan, _, _ = _plan(tmp_path)
    mounts = plan["mounts"]
    assert len(mounts) == 7
    assert sum(row["type"] == "bind" and row["read_only"] for row in mounts) == 5
    assert sum(row["type"] == "bind" and not row["read_only"] for row in mounts) == 1
    assert sum(row["type"] == "tmpfs" and row["destination"] == "/tmp" for row in mounts) == 1
    assert plan["runtime"]["gpu_uuid"] == EXPECTED_GPU_UUID
    assert plan["runtime"]["max_concurrent_scientific_processes"] == 1
    assert plan["runtime"]["single_process"] is True
    assert plan["output_empty_at_plan"] is True
    assert plan["PACS_archive_hashed_for_binding"] is True
    assert plan["PACS_records_decoded"] is False
    assert plan["PACS_labels_inspected"] is False
    assert plan["PACS_scientific_content_opened"] is False
    assert plan["torch_imported"] is False
    assert plan["CUDA_touched"] is False
    mounted_sources = {
        row["source"] for row in mounts if row["type"] == "bind" and row["read_only"]
    }
    assert mounted_sources == {row["source"] for row in plan["input_inventory"]}
    assert not any(row["provenance_source"] in mounted_sources for row in plan["input_inventory"])
    launches = plan["cell_launches"]
    assert len(launches) == 30
    for ordinal, launch in enumerate(launches):
        argv = launch["docker_argv_template"]
        assert launch["ordinal"] == ordinal
        assert "--training-authorized" not in argv
        assert "--run-authorization-sha256" not in argv
        assert argv.count("--mount") == 6
        assert argv[argv.index("--model-id") + 1] == launch["model_id"]


def test_planned_container_arguments_match_static_training_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan, _, _ = _plan(tmp_path)
    argv = plan["cell_launches"][0]["docker_argv_template"]
    image_index = argv.index("sha256:" + "a" * 64)
    container_argv = argv[image_index + 1 :]
    assert container_argv[:2] == ["-m", "fedcore.experiments.v3_train"]
    assert training_main([*container_argv[2:], "--validate-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PASS_STATIC_TRAINING_CLI_BINDING"
    assert report["PACS_opened"] is False
    assert report["torch_imported_by_cli"] is False
    assert report["GPU_used"] is False


def test_input_tamper_and_nonempty_output_fail_closed(tmp_path: Path) -> None:
    gate = _prebinding(tmp_path / "PREBINDING_GATE.json")
    binding, specs, staged = _inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    staged_file = staged / specs[0].name
    staged_file.chmod(0o644)
    staged_file.write_bytes(b"tampered")
    staged_file.chmod(0o444)
    with pytest.raises(ContractError, match="byte-size drift|SHA-256 drift"):
        build_launch_plan(
            prebinding_gate=gate,
            input_binding=binding,
            output_dir=output,
            specs=specs,
            expected_input_dir=staged,
            expected_output_dir=output,
            image_id="sha256:" + "b" * 64,
        )
    staged_file.chmod(0o644)
    staged_file.write_bytes(b"synthetic-0-PACS_mirror.zip")
    staged_file.chmod(0o444)
    (output / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ContractError, match="must be empty"):
        build_launch_plan(
            prebinding_gate=gate,
            input_binding=binding,
            output_dir=output,
            specs=specs,
            expected_input_dir=staged,
            expected_output_dir=output,
            image_id="sha256:" + "b" * 64,
        )


def test_output_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "output"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ContractError, match="real directory"):
        validate_empty_output_dir(link, expected_output_dir=link)


def test_separate_authorization_binds_exact_plan_and_only_emits_argv(
    tmp_path: Path,
) -> None:
    plan, _, _ = _plan(tmp_path)
    readiness = _readiness(tmp_path / "EXECUTION_READINESS.json", plan)
    authorization = tmp_path / "RUN_AUTHORIZATION.json"
    value = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "RUN_AUTHORIZED",
        "training_authorized": True,
        "launch_plan_sha256": plan["launch_plan_sha256"],
        "readiness_gate_sha256": hashlib.sha256(readiness.read_bytes()).hexdigest(),
        "input_binding_sha256": plan["input_binding_sha256"],
        "gpu_uuid": EXPECTED_GPU_UUID,
        "max_concurrent_scientific_processes": 1,
        "authorized_by": "Sanghoon Kim",
    }
    _write_json(authorization, value)
    with pytest.raises(ContractError, match="end-to-end artifact-bound runner"):
        bind_authorized_launch(plan, readiness, authorization)
    tampered = copy.deepcopy(plan)
    tampered["launch_plan_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="self-hash mismatch"):
        validate_launch_plan_integrity(tampered)


def test_arbitrary_scalar_digests_cannot_authorize_training() -> None:
    argv = [
        "--pacs-archive", "/inputs/PACS_mirror.zip",
        "--image-manifest", "/inputs/IMAGE_MANIFEST.csv",
        "--fold-manifest", "/inputs/FOLD_MANIFEST.csv",
        "--resnet18-weights", "/inputs/resnet18-f37072fd.pth",
        "--convnext-tiny-weights", "/inputs/convnext_tiny-983f1562.pth",
        "--output-dir", "/output",
        "--model-id", "pacs__resnet18__split00__seed0",
        "--readiness-sha256", "1" * 64,
        "--input-binding-sha256", "2" * 64,
        "--run-authorization-sha256", "3" * 64,
        "--training-authorized",
    ]
    with pytest.raises(ContractError, match="artifact-bound authorization"):
        training_main(argv)
