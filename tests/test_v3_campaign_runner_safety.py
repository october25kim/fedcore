"""PACS-free safety tests for the v3 scientific campaign runner."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fedcore.experiments.v3_contract import ContractError, ModelCell, sha256_file
from fedcore.experiments import v3_campaign_runner as runner


AUTHORIZATION = "a" * 64
STATE_HASH = "b" * 64


def _cell() -> ModelCell:
    return ModelCell(
        model_id="fixture",
        architecture="resnet18",
        split=0,
        nominal_training_seed=17,
        model_init_seed_uint64=123,
        known_classes=("dog", "elephant", "giraffe", "guitar"),
        unknown_classes=("horse", "house", "person"),
    )


def _synthetic_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[str]
) -> tuple[argparse.Namespace, SimpleNamespace]:
    inputs = {
        "PACS_mirror.zip": tmp_path / "inputs" / "PACS_mirror.zip",
        "IMAGE_MANIFEST.csv": tmp_path / "inputs" / "IMAGE_MANIFEST.csv",
        "FOLD_MANIFEST.csv": tmp_path / "inputs" / "FOLD_MANIFEST.csv",
        "resnet18-f37072fd.pth": tmp_path / "inputs" / "resnet18-f37072fd.pth",
        "convnext_tiny-983f1562.pth": (
            tmp_path / "inputs" / "convnext_tiny-983f1562.pth"
        ),
    }
    output = tmp_path / "output"
    authorization_dir = tmp_path / "authorization"
    output.mkdir()
    authorization_dir.mkdir()
    monkeypatch.setattr(runner, "EXPECTED_INPUT_PATHS", inputs)
    monkeypatch.setattr(runner, "EXPECTED_OUTPUT_DIR", output)
    monkeypatch.setattr(runner, "EXPECTED_AUTHORIZATION_DIR", authorization_dir)
    monkeypatch.setenv("FEDCORE_SOURCE_COMMIT", "f" * 40)
    authority = SimpleNamespace(
        source_commit="f" * 40,
        image_id="sha256:" + "1" * 64,
        authorization_id="synthetic-authorization",
        readiness_sha256="2" * 64,
        run_authorization_sha256=AUTHORIZATION,
        authorized_plan_sha256="3" * 64,
        runtime_receipt_sha256="4" * 64,
        input_binding_sha256="5" * 64,
    )

    def authorize(*args, **kwargs):
        events.append("authorization")
        return authority

    def verify_inputs(observed):
        events.append("input_hashes")
        assert observed == inputs
        return {name: str(index) * 64 for index, name in enumerate(inputs, start=1)}

    def verify_contract():
        events.append("scientific_contract")

    def lease(path, payload):
        events.append("lease")
        runner._write_exclusive_json(path, payload)

    original_write_state = runner.write_state_history

    def write_state(output_dir, state):
        events.append("state_init")
        return original_write_state(output_dir, state)

    monkeypatch.setattr(runner, "validate_mounted_authorization_bundle", authorize)
    monkeypatch.setattr(runner, "verify_bound_input_hashes", verify_inputs)
    monkeypatch.setattr(runner, "verify_scientific_contract", verify_contract)
    monkeypatch.setattr(runner, "acquire_exclusive_lease", lease)
    monkeypatch.setattr(runner, "write_state_history", write_state)
    args = argparse.Namespace(
        authorization_dir=str(authorization_dir),
        pacs_archive=str(inputs["PACS_mirror.zip"]),
        image_manifest=str(inputs["IMAGE_MANIFEST.csv"]),
        fold_manifest=str(inputs["FOLD_MANIFEST.csv"]),
        resnet18_weights=str(inputs["resnet18-f37072fd.pth"]),
        convnext_tiny_weights=str(inputs["convnext_tiny-983f1562.pth"]),
        output_dir=str(output),
        device="cuda:0",
    )
    return args, authority


def _run_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_train):
    monkeypatch.setattr(runner, "train_registered_model", fake_train)
    return runner._train_with_preproposal_restart(
        _cell(),
        {},
        ordinal=0,
        output=tmp_path,
        weights_path=tmp_path / "resnet18-f37072fd.pth",
        authorization_sha256={
            "readiness_sha256": "1" * 64,
            "input_binding_sha256": "2" * 64,
            "run_authorization_sha256": AUTHORIZATION,
        },
        run_authorization_sha256=AUTHORIZATION,
        state_sha256=STATE_HASH,
        device="cpu",
    )


def test_preproposal_io_restart_resumes_exact_validated_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "checkpoints" / "fixture" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"sealed-round-seven")
    monkeypatch.setattr(
        runner,
        "load_round_checkpoint",
        lambda *args, **kwargs: {"completed_rounds": 7},
    )
    calls: list[bool] = []

    def fake_train(*args, resume, **kwargs):
        calls.append(resume)
        if len(calls) == 1:
            raise OSError("temporary storage read failure")
        return SimpleNamespace(
            completed_rounds=30,
            stopped=False,
            checkpoint_path=checkpoint,
            model=object(),
        )

    result = _run_training(tmp_path, monkeypatch, fake_train)
    assert result.completed_rounds == 30
    assert calls == [False, True]
    receipt_path = (
        tmp_path / "restart_receipts" / "00_fixture" / "RESTART_001.json"
    )
    outcome_path = receipt_path.with_name("RESTART_OUTCOME_001.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert receipt["same_training_seed"] is True
    assert receipt["proposal_accessed"] is False
    assert receipt["audit_accessed"] is False
    assert receipt["resume_from_checkpoint"] is True
    assert receipt["completed_rounds"] == 7
    assert receipt["checkpoint_sha256"] == sha256_file(checkpoint)
    assert outcome["status"] == "PREPROPOSAL_SAME_CELL_RESTART_COMPLETED"
    assert outcome["further_restart_permitted"] is False
    with pytest.raises(ContractError, match="already exists"):
        runner._write_exclusive_json(receipt_path, {"forged": True})


def test_preproposal_io_restart_uses_same_seed_round_zero_without_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[bool] = []

    def fake_train(*args, resume, **kwargs):
        calls.append(resume)
        if len(calls) == 1:
            raise OSError("transient filesystem failure")
        return SimpleNamespace(
            completed_rounds=30,
            stopped=False,
            checkpoint_path=tmp_path / "checkpoints" / "fixture" / "latest.pt",
            model=object(),
        )

    _run_training(tmp_path, monkeypatch, fake_train)
    assert calls == [False, False]
    receipt = json.loads(
        (
            tmp_path / "restart_receipts" / "00_fixture" / "RESTART_001.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["resume_from_checkpoint"] is False
    assert receipt["checkpoint_sha256"] is None
    assert receipt["completed_rounds"] == 0


def test_retry_is_single_shot_and_second_failure_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = 0

    def fake_train(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise OSError(f"technical failure {calls}")

    with pytest.raises(OSError, match="technical failure 2"):
        _run_training(tmp_path, monkeypatch, fake_train)
    assert calls == 2
    outcome = json.loads(
        (
            tmp_path
            / "restart_receipts"
            / "00_fixture"
            / "RESTART_OUTCOME_001.json"
        ).read_text(encoding="utf-8")
    )
    assert outcome["status"] == "PREPROPOSAL_SAME_CELL_RESTART_FAILED"
    assert outcome["further_restart_permitted"] is False
    assert outcome["failure_type"] == "OSError"


def test_restart_that_returns_early_is_sealed_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = 0

    def fake_train(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("restartable I/O interruption")
        return SimpleNamespace(
            completed_rounds=9,
            stopped=True,
            checkpoint_path=tmp_path / "checkpoints" / "fixture" / "latest.pt",
            model=object(),
        )

    with pytest.raises(ContractError, match="before registered round 30"):
        _run_training(tmp_path, monkeypatch, fake_train)
    outcome = json.loads(
        (
            tmp_path
            / "restart_receipts"
            / "00_fixture"
            / "RESTART_OUTCOME_001.json"
        ).read_text(encoding="utf-8")
    )
    assert outcome["status"] == "PREPROPOSAL_SAME_CELL_RESTART_FAILED"
    assert outcome["failure_type"] == "ContractError"


def test_contract_breach_and_nonfinite_training_are_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for failure in (ContractError("drift"), FloatingPointError("nonfinite loss")):
        calls = 0

        def fake_train(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise failure

        with pytest.raises(type(failure)):
            _run_training(tmp_path, monkeypatch, fake_train)
        assert calls == 1
        assert not (tmp_path / "restart_receipts").exists()


def test_postproposal_restart_is_explicitly_prohibited(tmp_path: Path):
    cell = _cell()
    with pytest.raises(ContractError, match="after proposal"):
        runner._assert_preproposal_restart_allowed(
            tmp_path, cell, 0, primary_stage="PROPOSAL_SEALED"
        )
    bundle = tmp_path / "primary_fragments" / "00_fixture"
    bundle.mkdir(parents=True)
    (bundle / "PROPOSAL_SEAL.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ContractError, match="after proposal"):
        runner._assert_preproposal_restart_allowed(
            tmp_path, cell, 0, primary_stage="TRAINING"
        )


def test_primary_inference_uses_reloaded_sealed_checkpoint_not_training_object():
    source = inspect.getsource(runner._run_primary_phase)
    reload_at = source.index("inference_model = _load_sealed_model")
    proposal_at = source.index("proposal_frames = _infer_by_client")
    audit_at = source.index("primary_frames = _infer_registered_draws_by_client")
    assert reload_at < proposal_at < audit_at
    assert "result.model" not in source
    assert "_infer_by_client(\n                inference_model" in source
    assert "_infer_registered_draws_by_client(\n            inference_model" in source


def test_post_primary_full_truth_is_global_primary_seal_gated():
    source = inspect.getsource(runner._run_post_primary_phase)
    verify_at = source.index("primary_authority = verify_primary_seal(output)")
    truth_at = source.index("audit_stage_records = load_pacs_full_audit_records")
    assert verify_at < truth_at
    assert 'primary_seal_path=output / "PRIMARY_SEAL.json"' in source
    assert "expected_primary_seal_sha256=(" in source


def test_checkpoint_inventory_binds_successful_and_partial_files(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints" / "fixture"
    checkpoint_dir.mkdir(parents=True)
    latest = checkpoint_dir / "latest.pt"
    seal = checkpoint_dir / "CHECKPOINT_SEAL.json"
    latest.write_bytes(b"partial-or-complete-checkpoint")
    seal.write_text("{}\n", encoding="utf-8")
    inventory_path, inventory = runner._write_checkpoint_inventory(tmp_path)
    assert [row["path"] for row in inventory["files"]] == [
        "fixture/CHECKPOINT_SEAL.json",
        "fixture/latest.pt",
    ]
    assert inventory["files"][1]["sha256"] == sha256_file(latest)
    assert len(inventory["checkpoint_tree_sha256"]) == 64
    assert sha256_file(inventory_path)
    with pytest.raises(ContractError, match="already exists"):
        runner._write_checkpoint_inventory(tmp_path)


def test_terminal_failure_receipt_is_append_only_and_binds_partial_checkpoint(
    tmp_path: Path,
):
    checkpoint = tmp_path / "checkpoints" / "fixture" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"round-boundary-state")
    state = {
        "campaign_phase": "HOLD_INCOMPLETE",
        "state_sha256": STATE_HASH,
        "sequence_number": 3,
        "active_model_id": "fixture",
        "cells": [{"primary_stage": "TRAINING"}],
    }
    authority = SimpleNamespace(run_authorization_sha256=AUTHORIZATION)
    receipt_path = runner._write_campaign_failure_receipt(
        tmp_path,
        exc=OSError("terminal storage failure"),
        authority=authority,
        state=state,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "HOLD_INCOMPLETE"
    assert receipt["failure_classification"] == "INCOMPLETE_TECHNICAL_FAILURE"
    assert receipt["same_protocol_retry_permitted"] is False
    assert receipt["checkpoint_files"] == [
        {"path": "fixture/latest.pt", "sha256": sha256_file(checkpoint)}
    ]
    with pytest.raises(ContractError, match="already exists"):
        runner._write_campaign_failure_receipt(
            tmp_path,
            exc=OSError("forged replacement"),
            authority=authority,
            state=state,
        )


def test_only_deterministic_contract_breach_is_invalid():
    assert runner._failure_campaign_phase(ContractError("drift")) == "INVALID"
    assert runner._failure_campaign_phase(OSError("I/O")) == "HOLD_INCOMPLETE"
    assert runner._failure_campaign_phase(RuntimeError("CUDA driver lost")) == "HOLD_INCOMPLETE"
    assert runner._failure_campaign_phase(RuntimeError("unexpected")) == "INVALID"
    assert runner._failure_campaign_phase(ValueError("implementation mismatch")) == "INVALID"


def test_unverified_state_failure_preserves_both_causes(tmp_path: Path):
    authority = SimpleNamespace(run_authorization_sha256=AUTHORIZATION)
    receipt_path = runner._write_unverified_state_failure_receipt(
        tmp_path,
        exc=OSError("original storage failure"),
        state_exception=ContractError("broken state chain"),
        authority=authority,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "INVALID"
    assert receipt["failure_classification"] == "UNVERIFIED_CAMPAIGN_STATE"
    assert receipt["original_exception_type"] == "OSError"
    assert receipt["state_exception_type"] == "ContractError"
    assert receipt["same_protocol_retry_permitted"] is False


def test_run_campaign_pacs_free_calls_full_fresh_orchestration_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events: list[str] = []
    args, _ = _synthetic_campaign(tmp_path, monkeypatch, events)

    def primary_phase(**kwargs):
        events.append("primary")
        assert len(kwargs["cells"]) == 30
        assert len(kwargs["state"]["cells"]) == 30
        assert (Path(args.output_dir) / "RUN_PROVENANCE.json").is_file()
        return kwargs["state"], [{"synthetic": True}]

    def post_phase(**kwargs):
        events.append("post_primary")
        assert kwargs["primary_fragments"] == [{"synthetic": True}]
        return kwargs["state"]

    def completion(output, state, authority, observed):
        events.append("completion")
        runner.write_json(output / "FINAL_VALIDATION.json", {"status": "VALID_NEGATIVE"})
        runner.write_json(output / "EXECUTION_MANIFEST.json", {"status": "PASS"})
        return {"completion_sha256": "c" * 64}

    monkeypatch.setattr(runner, "_run_primary_phase", primary_phase)
    monkeypatch.setattr(runner, "_run_post_primary_phase", post_phase)
    monkeypatch.setattr(runner, "_write_execution_completion", completion)
    result = runner.run_campaign(args)
    assert events == [
        "authorization",
        "input_hashes",
        "scientific_contract",
        "lease",
        "state_init",
        "primary",
        "post_primary",
        "completion",
    ]
    assert result["status"] == "CAMPAIGN_FINALIZED"
    assert result["completion_sha256"] == "c" * 64
    assert len(result["execution_manifest_sha256"]) == 64


def test_run_campaign_preserves_original_and_state_verification_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events: list[str] = []
    args, _ = _synthetic_campaign(tmp_path, monkeypatch, events)

    def fail_primary(**kwargs):
        raise OSError("synthetic primary I/O failure")

    monkeypatch.setattr(runner, "_run_primary_phase", fail_primary)
    monkeypatch.setattr(
        runner,
        "verify_state_history",
        lambda *args, **kwargs: (_ for _ in ()).throw(ContractError("state chain drift")),
    )
    with pytest.raises(ContractError, match="unverifiable state history"):
        runner.run_campaign(args)
    pointer = json.loads(
        (Path(args.output_dir) / "RUN_FAILURE.json").read_text(encoding="utf-8")
    )
    assert pointer["status"] == "INVALID"
    receipt = json.loads(
        (
            Path(args.output_dir)
            / pointer["failure_receipt_path"]
        ).read_text(encoding="utf-8")
    )
    assert receipt["original_exception_type"] == "OSError"
    assert receipt["state_exception_type"] == "ContractError"


@pytest.mark.parametrize(
    ("failure", "expected_phase"),
    [
        (OSError("synthetic technical failure"), "HOLD_INCOMPLETE"),
        (ContractError("synthetic contract drift"), "INVALID"),
    ],
)
def test_run_campaign_fail_closed_phase_and_receipt_match_failure_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected_phase: str,
):
    events: list[str] = []
    args, _ = _synthetic_campaign(tmp_path, monkeypatch, events)

    def fail_primary(**kwargs):
        checkpoint = Path(args.output_dir) / "checkpoints" / "partial" / "latest.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"preserved partial checkpoint")
        raise failure

    monkeypatch.setattr(runner, "_run_primary_phase", fail_primary)
    with pytest.raises(type(failure), match=str(failure)):
        runner.run_campaign(args)
    pointer = json.loads(
        (Path(args.output_dir) / "RUN_FAILURE.json").read_text(encoding="utf-8")
    )
    assert pointer["status"] == expected_phase
    receipt = json.loads(
        (Path(args.output_dir) / pointer["failure_receipt_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == expected_phase
    assert receipt["same_protocol_retry_permitted"] is False
    assert receipt["checkpoint_files"][0]["path"] == "partial/latest.pt"


def test_run_campaign_preserves_original_failure_when_terminal_state_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events: list[str] = []
    args, _ = _synthetic_campaign(tmp_path, monkeypatch, events)
    original_write = runner.write_state_history
    calls = 0

    def fail_primary(**kwargs):
        raise OSError("original campaign failure")

    def flaky_state_write(output, state):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(output, state)
        raise OSError("terminal state write failure")

    monkeypatch.setattr(runner, "_run_primary_phase", fail_primary)
    monkeypatch.setattr(runner, "write_state_history", flaky_state_write)
    with pytest.raises(ContractError, match="terminal failure state could not be sealed"):
        runner.run_campaign(args)
    pointer = json.loads(
        (Path(args.output_dir) / "RUN_FAILURE.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (Path(args.output_dir) / pointer["failure_receipt_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert receipt["original_message"] == "original campaign failure"
    assert receipt["state_exception_message"] == "terminal state write failure"
