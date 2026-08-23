"""Torch-free Fed-ISIC metadata, unit aggregation, and dry-run tests."""

from __future__ import annotations

import builtins
import csv
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcore.experiments import run_fed_isic
from fedcore.campaign.artifacts import canonical_json, semantic_hash
from fedcore.campaign.plan import training_cell_experiment_id
from fedcore.medical.data import (
    MedicalDataConfig,
    aggregate_unit_logits,
    audit_artifact_arrays,
    flatten_unit_images,
    load_fed_isic_job,
    traffic_identity_arrays,
)


METADATA_FIELDS = ["center", "diagnosis", "patient", "lesion", "image"]
FOLD_FIELDS = [
    "audit_unit_id",
    "fold",
    "center",
    "diagnosis",
    "patient_id",
    "lesion_ids_json",
    "image_ids_json",
]


def _write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(directory):
    metadata_path = os.path.join(directory, "metadata.csv")
    folds_path = os.path.join(directory, "folds.csv")
    metadata = []
    folds = []
    for center in ("site-a", "site-b"):
        for diagnosis in ("d0", "d1", "d2"):
            for fold in ("train", "proposal", "certification", "test", "traffic"):
                unit = f"{center}-{diagnosis}-{fold}"
                images = [f"{unit}-image-{index}" for index in range(2)]
                for image in images:
                    metadata.append(
                        {
                            "center": center,
                            "diagnosis": diagnosis,
                            "patient": f"{unit}-patient",
                            "lesion": unit,
                            "image": image,
                        }
                    )
                folds.append(
                    {
                        "audit_unit_id": unit,
                        "fold": fold,
                        "center": center,
                        "diagnosis": diagnosis,
                        "patient_id": f"{unit}-patient",
                        "lesion_ids_json": json.dumps([unit]),
                        "image_ids_json": json.dumps(images),
                    }
                )
    _write_csv(metadata_path, METADATA_FIELDS, metadata)
    _write_csv(folds_path, FOLD_FIELDS, folds)
    return metadata_path, folds_path, metadata, folds


def _config(metadata, folds):
    return MedicalDataConfig(
        metadata_csv=metadata,
        folds_csv=folds,
        center_col="center",
        diagnosis_col="diagnosis",
        patient_col="patient",
        lesion_col="lesion",
        image_col="image",
        unit_col="lesion",
    )


def _plan_cell(**overrides):
    cell = {
        "family": "fed_isic2019",
        "heldout_diagnosis": "d2",
        "model_seed": 0,
        "campaign_seed": 0,
        "dataset": "fed-isic2019",
        "rounds": 50,
        "local_epochs": 2,
        "lr": 0.01,
        "batch_size": 64,
        "backbone": "resnet18",
        "norm": "gn",
        "pretrained": False,
        "image_size": 224,
    }
    cell.update(overrides)
    return cell


def _write_plan_cell(path, config):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(canonical_json(config) + "\n")


def _parse_runner_args(metadata, folds, *extra):
    parser = run_fed_isic.build_parser()
    args = parser.parse_args(
        [
            metadata,
            folds,
            "--held-out-diagnosis",
            "d2",
            "--center-col",
            "center",
            "--diagnosis-col",
            "diagnosis",
            "--patient-col",
            "patient",
            "--lesion-col",
            "lesion",
            "--image-col",
            "image",
            "--unit-col",
            "lesion",
            "--dry-run",
            *extra,
        ]
    )
    run_fed_isic._normalize_paths(args, parser)
    return args


def test_leave_one_diagnosis_job_uses_known_train_images_and_distinct_units():
    with tempfile.TemporaryDirectory() as directory:
        metadata, folds, _, _ = _fixture(directory)
        job = load_fed_isic_job(_config(metadata, folds), "d2")
        assert job.centers == ("site-a", "site-b")
        assert job.known_diagnoses == ("d0", "d1")
        assert job.diagnosis_to_label == {"d0": 0, "d1": 1}
        for mapping, key in (
            (job.center_to_client, "new-site"),
            (job.diagnosis_to_label, "new-diagnosis"),
            (job.units_by_fold, "proposal"),
        ):
            try:
                mapping[key] = 999
            except TypeError:
                pass
            else:
                raise AssertionError("validated medical job mapping remained mutable")
        # Each site has two known train diagnoses x two images. Held-out train
        # rows are present in metadata but never become a training example.
        assert [len(records) for records in job.training_by_client] == [4, 4]
        assert all(
            record.diagnosis != "d2" and record.fold == "train"
            for records in job.training_by_client
            for record in records
        )
        for fold in ("proposal", "certification", "test"):
            units = job.fold_units(fold)
            assert len(units) == 6
            assert len({unit.sample_id for unit in units}) == 6
            assert all(unit.image_multiplicity == 2 for unit in units)
            assert all(
                (unit.y_open == -1) == (unit.diagnosis == "d2") for unit in units
            )

        fold_ids = [
            {unit.sample_id for unit in job.fold_units(fold)}
            for fold in ("train", "proposal", "certification", "test", "traffic")
        ]
        assert all(
            not (left & right)
            for index, left in enumerate(fold_ids)
            for right in fold_ids[index + 1 :]
        )


def test_heldout_pair_maps_both_classes_to_unknown_and_relabels_knowns():
    """The pre-registered roster holds out a PAIR (2 of 8), not one diagnosis."""
    with tempfile.TemporaryDirectory() as directory:
        metadata, folds, _, _ = _fixture(directory)
        job = load_fed_isic_job(_config(metadata, folds), ["d1", "d2"])
        assert job.heldout_diagnoses == ("d1", "d2")
        assert job.heldout_label == "d1+d2"
        # Knowns are relabelled contiguously from 0 over the SURVIVING classes.
        assert job.known_diagnoses == ("d0",)
        assert job.diagnosis_to_label == {"d0": 0}
        # Neither pair member may ever become a training example.
        assert all(
            record.diagnosis == "d0"
            for records in job.training_by_client
            for record in records
        )
        for fold in ("proposal", "certification", "test"):
            units = job.fold_units(fold)
            assert all(
                (unit.y_open == -1) == (unit.diagnosis in {"d1", "d2"})
                for unit in units
            )
            # BOTH members must actually appear as labelled unknowns.
            assert {unit.diagnosis for unit in units if unit.y_open == -1} == {
                "d1",
                "d2",
            }
        # Order must not create a second identity for the same split.
        assert load_fed_isic_job(_config(metadata, folds), "d2,d1").heldout_label == (
            "d1+d2"
        )
        # The singular accessor must refuse to silently pick one of the two.
        try:
            job.heldout_diagnosis
        except ValueError:
            pass
        else:
            raise AssertionError("heldout_diagnosis returned one member of a pair")


def test_audit_arrays_enforce_the_declared_heldout_pair():
    with tempfile.TemporaryDirectory() as directory:
        metadata, folds, _, _ = _fixture(directory)
        job = load_fed_isic_job(_config(metadata, folds), ["d1", "d2"])
        units = job.fold_units("certification")
        logits = np.zeros((sum(u.image_multiplicity for u in units), 1))
        arrays = audit_artifact_arrays(units, logits, ["d1", "d2"])
        assert set(arrays["y_open"].tolist()) == {-1, 0}
        # Declaring a pair the fold does not carry must fail closed.
        try:
            audit_artifact_arrays(units, logits, ["d0", "d1"])
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched held-out declaration was accepted")


def test_traffic_fold_is_optional_because_the_prereg_declares_none():
    """data.folds spends the whole audit pool on prop/cert/test; traffic is not
    pre-registered for Fed-ISIC and must not be a hard requirement."""
    with tempfile.TemporaryDirectory() as directory:
        metadata_path = os.path.join(directory, "metadata.csv")
        folds_path = os.path.join(directory, "folds.csv")
        metadata = []
        folds = []
        for center in ("site-a", "site-b"):
            for diagnosis in ("d0", "d1", "d2"):
                for fold in ("train", "proposal", "certification", "test"):
                    unit = f"{center}-{diagnosis}-{fold}"
                    images = [f"{unit}-image-{index}" for index in range(2)]
                    for image in images:
                        metadata.append(
                            {
                                "center": center,
                                "diagnosis": diagnosis,
                                "patient": f"{unit}-patient",
                                "lesion": unit,
                                "image": image,
                            }
                        )
                    folds.append(
                        {
                            "audit_unit_id": unit,
                            "fold": fold,
                            "center": center,
                            "diagnosis": diagnosis,
                            "patient_id": f"{unit}-patient",
                            "lesion_ids_json": json.dumps([unit]),
                            "image_ids_json": json.dumps(images),
                        }
                    )
        _write_csv(metadata_path, METADATA_FIELDS, metadata)
        _write_csv(folds_path, FOLD_FIELDS, folds)
        job = load_fed_isic_job(_config(metadata_path, folds_path), ["d1", "d2"])
        assert job.fold_units("traffic") == ()


def test_repeated_image_logits_are_averaged_once_per_unit_and_multiplicity_persists():
    with tempfile.TemporaryDirectory() as directory:
        metadata, folds, _, _ = _fixture(directory)
        job = load_fed_isic_job(_config(metadata, folds), "d2")
        units = job.fold_units("proposal")
        images, parents = flatten_unit_images(units)
        assert len(images) == 2 * len(units)
        image_logits = np.arange(len(images) * 2, dtype=float).reshape(len(images), 2)
        mean, multiplicity = aggregate_unit_logits(
            image_logits, parents, [unit.sample_id for unit in units]
        )
        np.testing.assert_allclose(mean[0], image_logits[:2].mean(axis=0))
        np.testing.assert_array_equal(multiplicity, np.full(len(units), 2))

        artifact = audit_artifact_arrays(units, image_logits)
        assert artifact["logits"].shape == (len(units), 2)
        np.testing.assert_array_equal(artifact["image_multiplicity"], multiplicity)
        assert len(set(artifact["sample_id"].tolist())) == len(units)


def test_traffic_export_is_site_identity_only():
    with tempfile.TemporaryDirectory() as directory:
        metadata, folds, _, _ = _fixture(directory)
        job = load_fed_isic_job(_config(metadata, folds), "d2")
        traffic = traffic_identity_arrays(job.fold_units("traffic"))
        assert set(traffic) == {"sample_id", "client", "site_id"}
        assert (
            "y_open" not in traffic
            and "diagnosis" not in traffic
            and "logits" not in traffic
        )
        assert set(traffic["site_id"].tolist()) == {"site-a", "site-b"}


def test_loader_fails_closed_on_multi_diagnosis_unit_and_fold_overlap():
    with tempfile.TemporaryDirectory() as directory:
        metadata_path, folds_path, metadata, folds = _fixture(directory)
        metadata[1]["diagnosis"] = "d1"  # same lesion/unit now spans diagnoses
        _write_csv(metadata_path, METADATA_FIELDS, metadata)
        try:
            load_fed_isic_job(_config(metadata_path, folds_path), "d2")
        except ValueError as exc:
            assert "multiple diagnoses" in str(exc)
        else:
            raise AssertionError("multi-diagnosis unit was accepted")

        metadata_path, folds_path, metadata, folds = _fixture(directory)
        folds.append(dict(folds[0], fold="test"))
        _write_csv(folds_path, FOLD_FIELDS, folds)
        try:
            load_fed_isic_job(_config(metadata_path, folds_path), "d2")
        except ValueError as exc:
            assert "duplicate/overlapping" in str(exc)
        else:
            raise AssertionError("overlapping frozen unit was accepted")


def test_cli_dry_run_does_not_import_torch_or_open_images(capsys):
    with tempfile.TemporaryDirectory() as directory:
        metadata, folds, _, _ = _fixture(directory)
        original_import = builtins.__import__

        def reject_heavy(name, *args, **kwargs):
            if (
                name == "torch"
                or name.startswith("torch.")
                or name == "torchvision"
                or name.startswith("torchvision.")
                or name == "PIL"
                or name.startswith("PIL.")
            ):
                raise AssertionError(f"dry-run imported heavy image stack: {name}")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = reject_heavy
        try:
            rc = run_fed_isic.main(
                [
                    metadata,
                    folds,
                    "--held-out-diagnosis",
                    "d2",
                    "--model-replicate",
                    "1",
                    "--experiment-id",
                    "planned-medical-cell",
                    "--center-col",
                    "center",
                    "--diagnosis-col",
                    "diagnosis",
                    "--patient-col",
                    "patient",
                    "--lesion-col",
                    "lesion",
                    "--image-col",
                    "image",
                    "--unit-col",
                    "lesion",
                    "--backbone",
                    "simplecnn",
                    "--dry-run",
                ]
            )
        finally:
            builtins.__import__ = original_import
        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "metadata_validated"
        assert report["image_files_checked"] is False
        assert report["heldout_diagnosis"] == "d2"
        assert report["model_replicate"] == 1
        assert report["experiment_id"] == "planned-medical-cell"
        assert report["training_config"]["norm"] == "bn"
        forbidden = {"alpha", "delta", "rho", "policy", "score", "certificate_variant"}
        assert forbidden.isdisjoint(report["training_config"])
        assert report["dataset_sha256"]
        assert report["fold_sha256"]
        assert report["training_config_sha256"]


def test_authoritative_plan_cell_match_uses_plan_id_and_separate_hashes(capsys):
    with tempfile.TemporaryDirectory() as directory:
        metadata, folds, _, _ = _fixture(directory)
        plan_path = os.path.join(directory, "plan-cell.json")
        cell = _plan_cell()
        _write_plan_cell(plan_path, cell)
        rc = run_fed_isic.main(
            [
                metadata,
                folds,
                "--held-out-diagnosis",
                "d2",
                "--center-col",
                "center",
                "--diagnosis-col",
                "diagnosis",
                "--patient-col",
                "patient",
                "--lesion-col",
                "lesion",
                "--image-col",
                "image",
                "--unit-col",
                "lesion",
                "--plan-cell-config",
                plan_path,
                "--dry-run",
            ]
        )
        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert report["plan_cell_bound"] is True
        assert report["experiment_id"] == training_cell_experiment_id(cell)
        assert report["plan_cell_config_json"] == canonical_json(cell)
        assert report["plan_cell_config_sha256"] == semantic_hash(cell)
        assert report["training_config_sha256"] == semantic_hash(
            report["training_config"]
        )
        assert report["training_config_json"] == canonical_json(
            report["training_config"]
        )
        assert report["plan_cell_config_sha256"] != report["training_config_sha256"]


def test_authoritative_plan_cell_rejects_selector_and_training_arg_mismatch():
    with tempfile.TemporaryDirectory() as directory:
        metadata, folds, _, _ = _fixture(directory)
        unbound = _parse_runner_args(metadata, folds)
        unbound.dry_run = False
        try:
            run_fed_isic.prepare_job(unbound)
        except ValueError as exc:
            assert "requires --plan-cell-config" in str(exc)
        else:
            raise AssertionError("non-dry unbound training cell was accepted")

        mismatches = (
            _plan_cell(family="cifar", split_id="s0", dirichlet_alpha=0.1),
            _plan_cell(heldout_diagnosis="d1"),
            _plan_cell(model_seed=2),
            _plan_cell(campaign_seed=9),
            _plan_cell(rounds=99),
            {"config": _plan_cell(), "campaign_seed": 0},
        )
        for index, cell in enumerate(mismatches):
            plan_path = os.path.join(directory, f"bad-plan-{index}.json")
            _write_plan_cell(plan_path, cell)
            args = _parse_runner_args(metadata, folds, "--plan-cell-config", plan_path)
            try:
                run_fed_isic.prepare_job(args)
            except ValueError:
                pass
            else:
                raise AssertionError(f"plan-cell mismatch {index} was accepted")

        valid_path = os.path.join(directory, "valid-plan.json")
        valid = _plan_cell()
        _write_plan_cell(valid_path, valid)
        args = _parse_runner_args(
            metadata,
            folds,
            "--plan-cell-config",
            valid_path,
            "--experiment-id",
            "wrong-plan-cell",
        )
        try:
            run_fed_isic.prepare_job(args)
        except ValueError as exc:
            assert "authoritative plan-cell ID" in str(exc)
        else:
            raise AssertionError("mismatched explicit experiment ID was accepted")


def test_semantic_training_identity_is_portable_across_image_mounts():
    with tempfile.TemporaryDirectory() as directory:
        metadata, folds, _, _ = _fixture(directory)
        plan_path = os.path.join(directory, "plan-cell.json")
        _write_plan_cell(plan_path, _plan_cell())

        def prepare(root):
            args = _parse_runner_args(
                metadata,
                folds,
                "--image-root",
                root,
                "--plan-cell-config",
                plan_path,
            )
            return run_fed_isic.prepare_job(args)

        left = prepare(os.path.join(directory, "mount-a"))
        right = prepare(os.path.join(directory, "mount-b"))
        assert left[3] == right[3]  # training_config_sha256
        assert left[4] == right[4] == training_cell_experiment_id(_plan_cell())
        assert left[1] == right[1]  # complete semantic seed bundle
        assert (
            left[7].config_sha256
            == right[7].config_sha256
            == semantic_hash(_plan_cell())
        )


def main():
    # Standalone runner mirrors the repository's torch-free test style.
    tests = (
        test_leave_one_diagnosis_job_uses_known_train_images_and_distinct_units,
        test_heldout_pair_maps_both_classes_to_unknown_and_relabels_knowns,
        test_audit_arrays_enforce_the_declared_heldout_pair,
        test_traffic_fold_is_optional_because_the_prereg_declares_none,
        test_repeated_image_logits_are_averaged_once_per_unit_and_multiplicity_persists,
        test_traffic_export_is_site_identity_only,
        test_loader_fails_closed_on_multi_diagnosis_unit_and_fold_overlap,
        test_authoritative_plan_cell_rejects_selector_and_training_arg_mismatch,
        test_semantic_training_identity_is_portable_across_image_mounts,
    )
    for test in tests:
        test()
        print(f"  [PASS] {test.__name__}")
    print(
        f"medical data checks: PASS ({len(tests)} standalone tests; pytest adds CLI checks)"
    )


if __name__ == "__main__":
    main()
