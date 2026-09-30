"""PACS-free stage-order tests for the authorized scientific runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fedcore.experiments import v3_campaign_runner as runner
from fedcore.experiments.v3_contract import CLIENTS, load_model_cells


def test_primary_runner_seals_proposal_before_any_audit_operation(
    tmp_path: Path, monkeypatch
) -> None:
    model = load_model_cells()[0]
    checkpoint = tmp_path / "checkpoints" / model.model_id / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"sealed checkpoint")
    authority = SimpleNamespace(
        run_authorization_sha256="a" * 64,
        checkpoint_fields=lambda: {
            "readiness_sha256": "b" * 64,
            "input_binding_sha256": "c" * 64,
            "run_authorization_sha256": "a" * 64,
        },
    )
    events: list[str] = []
    proposal_marker = tmp_path / "proposal-sealed"

    monkeypatch.setattr(runner, "build_registered_transforms", lambda: (object(), object()))

    def load_seeds():
        events.append("audit_seeds")
        assert proposal_marker.is_file()
        return {}

    monkeypatch.setattr(runner, "load_audit_seeds", load_seeds)

    def load_train(*args, **kwargs):
        events.append("load:train")
        return (SimpleNamespace(use="train"),)

    def load_proposal(*args, **kwargs):
        events.append("load:proposal")
        return (SimpleNamespace(use="proposal"),)

    monkeypatch.setattr(runner, "load_pacs_train_records", load_train)
    monkeypatch.setattr(runner, "load_pacs_proposal_records", load_proposal)

    def load_audit_frame(*args, **kwargs):
        events.append("load:audit_frame")
        assert proposal_marker.is_file()
        assert kwargs["proposal_seal_path"].name == "PROPOSAL_SEAL.json"
        assert len(kwargs["expected_proposal_seal_sha256"]) == 64
        return object()

    monkeypatch.setattr(runner, "load_pacs_audit_frame", load_audit_frame)

    def frame_by_domain(_records):
        events.append("audit_frame_grouped")
        return {
            client: tuple(
                SimpleNamespace(identity=f"{client}-{index}")
                for index in range(256)
            )
            for client in CLIENTS
        }

    monkeypatch.setattr(runner, "audit_frame_by_domain", frame_by_domain)

    def load_selected(*args, **kwargs):
        events.append("load:selected_audit_truth")
        assert events.index("audit_indices") < events.index("load:selected_audit_truth")
        return (SimpleNamespace(use="audit"),)

    monkeypatch.setattr(runner, "load_pacs_selected_audit_records", load_selected)

    def records_by_use(_records, use):
        events.append(f"records:{use}")
        if use == "audit":
            assert proposal_marker.is_file()
        return {client: (object(),) for client in CLIENTS}

    monkeypatch.setattr(runner, "records_by_use_and_domain", records_by_use)
    monkeypatch.setattr(runner, "_training_datasets", lambda *args: {})
    monkeypatch.setattr(runner, "_close_datasets", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "_train_with_preproposal_restart",
        lambda *args, **kwargs: SimpleNamespace(
            stopped=False,
            completed_rounds=30,
            checkpoint_path=checkpoint,
            model=object(),
        ),
    )

    checkpoint_seal = checkpoint.parent / "CHECKPOINT_SEAL.json"
    checkpoint_seal.write_bytes(b"checkpoint seal")
    def seal_checkpoint(*args):
        events.append("checkpoint_sealed")
        return checkpoint_seal

    monkeypatch.setattr(runner, "_checkpoint_seal", seal_checkpoint)
    monkeypatch.setattr(runner, "_load_sealed_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner,
        "_infer_by_client",
        lambda *args, **kwargs: {client: object() for client in CLIENTS},
    )
    members = runner._proposal_placeholder_members()
    monkeypatch.setattr(runner, "proposal_members", lambda frames: members)

    def write_proposal(cell_dir, fragment):
        events.append("proposal_sealed")
        proposal_marker.write_text("sealed", encoding="utf-8")
        Path(cell_dir).mkdir(parents=True)
        path = Path(cell_dir) / "PROPOSAL_SEAL.json"
        path.write_text("sealed", encoding="utf-8")
        return path

    monkeypatch.setattr(runner, "write_proposal_fragment_bundle", write_proposal)

    def audit_plan(*args, **kwargs):
        events.append("audit_indices")
        assert proposal_marker.is_file()
        return {client: list(range(256)) for client in CLIENTS}

    monkeypatch.setattr(runner, "audit_index_plan", audit_plan)

    def infer_audit(*args, **kwargs):
        events.append("audit_inference")
        assert proposal_marker.is_file()
        return {client: object() for client in CLIENTS}

    monkeypatch.setattr(runner, "_infer_registered_draws_by_client", infer_audit)
    monkeypatch.setattr(runner, "audit_sequence_sha256", lambda *args, **kwargs: "d" * 64)
    monkeypatch.setattr(
        runner,
        "_primary_success_fragment",
        lambda *args, **kwargs: {"training_status": "terminal_success"},
    )

    def append_primary(cell_dir, fragment, audit):
        events.append("primary_appended")
        assert proposal_marker.is_file()
        path = Path(cell_dir) / "PRIMARY_FRAGMENT_SEAL.json"
        path.write_text("primary", encoding="utf-8")
        return path

    monkeypatch.setattr(runner, "append_primary_fragment_bundle", append_primary)
    monkeypatch.setattr(runner, "load_primary_fragment_bundle", lambda *args: {})
    monkeypatch.setattr(
        runner,
        "seal_primary_campaign",
        lambda *args: SimpleNamespace(primary_seal_sha256="e" * 64),
    )
    monkeypatch.setattr(
        runner,
        "_write_and_advance",
        lambda output, state, cells, authorization_hash, **transition: state,
    )

    runner._run_primary_phase(
        output=tmp_path,
        cells=(model,),
        state={"state_sha256": "f" * 64},
        authority=authority,
        inputs={
            "PACS_mirror.zip": tmp_path / "PACS.zip",
            "IMAGE_MANIFEST.csv": tmp_path / "images.csv",
            "FOLD_MANIFEST.csv": tmp_path / "folds.csv",
            runner.WEIGHT_FILENAMES[model.architecture]: tmp_path / "weights.pth",
        },
        device="cuda:0",
    )

    proposal = events.index("proposal_sealed")
    assert events.index("load:train") < events.index("checkpoint_sealed")
    assert events.index("checkpoint_sealed") < events.index("load:proposal")
    assert events.index("load:proposal") < proposal
    assert proposal < events.index("load:audit_frame")
    assert events.index("load:audit_frame") < events.index("audit_indices")
    assert events.index("audit_indices") < events.index("load:selected_audit_truth")
    assert proposal < events.index("records:audit")
    assert proposal < events.index("audit_seeds")
    assert proposal < events.index("audit_indices")
    assert proposal < events.index("audit_inference")
    assert events.index("audit_inference") < events.index("primary_appended")
