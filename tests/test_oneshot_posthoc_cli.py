"""Focused immutable-artifact tests for the one-shot post-hoc runner."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcore.campaign.artifacts import file_sha256, semantic_hash  # noqa: E402
from fedcore.experiments.run_oneshot_posthoc import (  # noqa: E402
    JOINT_CONDITIONAL_CERTIFICATE_VARIANT,
    PosthocRequest,
    run_posthoc,
)
from fedcore.seeds import SEED_NAMESPACES, SeedBundle  # noqa: E402


_DATASET_SHA256 = "a" * 64


def _fold(seed: int, prefix: str, n: int = 360):
    rng = np.random.default_rng(seed)
    client = np.tile(np.arange(3, dtype=np.int64), n // 3)
    y = rng.integers(0, 2, size=n, dtype=np.int64)
    unknown = rng.random(n) < 0.25
    y[unknown] = -1
    logits = rng.normal(0.0, 0.25, size=(n, 2))
    known = y >= 0
    logits[np.flatnonzero(known), y[known]] += 3.0
    # Unknown observations are deliberately ambiguous and therefore low MSP.
    logits[unknown] *= 0.2
    ids = np.asarray([f"{prefix}:{index}" for index in range(n)], dtype="U32")
    return logits, y, client, ids


def _write_artifact(
    path: str,
    *,
    traffic: bool = False,
    overlap: bool = False,
    traffic_outcome: bool = False,
    omit_ids: bool = False,
    bad_config_hash: bool = False,
    omit_seed_ledger: bool = False,
    raw_seed_fallback: bool = False,
    omit_dataset_hash: bool = False,
    bad_dataset_hash: bool = False,
    mismatched_dataset_hash: bool = False,
    changed_labels: bool = False,
    changed_logits: bool = False,
) -> None:
    config = {
        "schema_version": 1,
        "dataset": "fixture",
        "n_clients": 3,
        "dirichlet_alpha": 0.5,
        "model_seed": 2,
        "dataset_sha256": _DATASET_SHA256,
    }
    if raw_seed_fallback:
        config["seeds"] = {"audit_draw": 123, "traffic_draw": 456}
    bundle = SeedBundle.derive(71, common_context={"campaign": "posthoc-fixture"})
    arrays: dict[str, np.ndarray] = {
        "dataset": np.asarray("fixture"),
        "experiment_id": np.asarray("fixture-training-cell"),
        "training_config_json": np.asarray(
            json.dumps(config, sort_keys=True, separators=(",", ":"))
        ),
        "training_config_sha256": np.asarray(
            "0" * 64 if bad_config_hash else semantic_hash(config)
        ),
    }
    if not omit_seed_ledger:
        arrays["seed_ledger_json"] = np.asarray(bundle.to_json())
    if not omit_dataset_hash:
        arrays["dataset_sha256"] = np.asarray(
            (
                "invalid"
                if bad_dataset_hash
                else ("c" * 64 if mismatched_dataset_hash else _DATASET_SHA256)
            )
        )
    for index, fold in enumerate(("prop", "cert", "test"), start=1):
        logits, labels, clients, ids = _fold(index, fold)
        if changed_labels:
            labels = labels.copy()
            known = labels >= 0
            labels[known] = 1 - labels[known]
        if changed_logits:
            logits = logits.copy()
            logits[:, 0] += 0.125
        if overlap and fold == "test":
            ids[0] = "cert:0"
        arrays[f"{fold}_logits"] = logits
        arrays[f"{fold}_y_open"] = labels
        arrays[f"{fold}_client"] = clients
        if not (omit_ids and fold == "cert"):
            arrays[f"{fold}_sample_id"] = ids
    if traffic:
        arrays["traffic_sample_id"] = np.asarray(
            [f"traffic:{index}" for index in range(180)], dtype="U32"
        )
        arrays["traffic_client"] = np.tile(np.arange(3), 60).astype(np.int64)
        if traffic_outcome:
            arrays["traffic_y_open"] = np.zeros(180, dtype=np.int64)
    np.savez_compressed(path, **arrays)


def _request(root: str, artifact: str, **changes) -> PosthocRequest:
    values = {
        "input_path": artifact,
        "output_path": os.path.join(root, "result.json"),
        "manifest_path": os.path.join(root, "result.manifest.json"),
        "scores": ("msp",),
        "gammas": (0.7,),
        "alpha": 0.1,
        "total_delta": 0.1,
        "delta_conditional_risk": 0.05,
        "delta_acceptance_lower": 0.05,
        "delta_acceptance_upper": 0.0,
        "delta_mixture": 0.0,
        "mixture_mode": "simplex",
        "certificate_variant": JOINT_CONDITIONAL_CERTIFICATE_VARIANT,
        "rho": None,
        "mixture_center": None,
    }
    values.update(changes)
    return PosthocRequest(**values)


def _expect_failure(fn, exception=ValueError):
    try:
        fn()
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}")


def test_simplex_replays_four_policies_and_writes_bound_manifest():
    with tempfile.TemporaryDirectory() as root:
        artifact = os.path.join(root, "logits.npz")
        _write_artifact(artifact)
        request = _request(root, artifact)
        result = run_posthoc(request)
        assert result["row_count"] == 4
        assert len({row["cert_sample_ids_sha256"] for row in result["rows"]}) == 1
        for policy in ("global", "client_specific"):
            rows = [row for row in result["rows"] if row["threshold_policy"] == policy]
            assert len({(row["cert_n"], row["cert_k"]) for row in rows}) == 1

        with open(request.output_path, encoding="utf-8") as handle:
            persisted = json.load(handle)
        with open(request.manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        assert persisted["posthoc_config"]["test_usage"] == "evaluation_only"
        assert manifest["artifacts"][0]["sha256"] == file_sha256(artifact)
        assert manifest["artifacts"][1]["sha256"] == file_sha256(request.output_path)
        assert set(manifest["seeds"]) == set(SEED_NAMESPACES)
        assert len(manifest["seeds"]) == len(SEED_NAMESPACES)
        assert manifest["dataset_hash"] == _DATASET_SHA256
        _expect_failure(lambda: run_posthoc(request), FileExistsError)


def test_declared_rho_box_runs_without_a_mixture_confidence_spend():
    with tempfile.TemporaryDirectory() as root:
        artifact = os.path.join(root, "logits.npz")
        _write_artifact(artifact)
        result = run_posthoc(
            _request(
                root,
                artifact,
                mixture_mode="rho",
                rho=0.2,
                mixture_center=(0.4, 0.35, 0.25),
                delta_conditional_risk=0.04,
                delta_acceptance_lower=0.03,
                delta_acceptance_upper=0.03,
            )
        )
        assert result["row_count"] == 4
        config = result["posthoc_config"]
        assert config["lambda_lower"] is not None
        assert config["failure_budget"]["delta_mixture"] == 0.0


def test_traffic_box_consumes_identity_only_and_records_common_traffic_count():
    with tempfile.TemporaryDirectory() as root:
        artifact = os.path.join(root, "logits.npz")
        _write_artifact(artifact, traffic=True)
        result = run_posthoc(
            _request(
                root,
                artifact,
                mixture_mode="traffic",
                delta_conditional_risk=0.03,
                delta_acceptance_lower=0.025,
                delta_acceptance_upper=0.025,
                delta_mixture=0.02,
            )
        )
        assert result["posthoc_config"]["traffic_usage"] == "identity_only"
        assert {row["traffic_observation_count"] for row in result["rows"]} == {180}


def test_nested_audit_and_traffic_subsamples_share_analysis_invariant_seeds():
    with tempfile.TemporaryDirectory() as root:
        artifact = os.path.join(root, "logits.npz")
        _write_artifact(artifact, traffic=True)
        common = {
            "mixture_mode": "traffic",
            "delta_conditional_risk": 0.03,
            "delta_acceptance_lower": 0.025,
            "delta_acceptance_upper": 0.025,
            "delta_mixture": 0.02,
        }
        small_root = os.path.join(root, "small")
        large_root = os.path.join(root, "large")
        small = run_posthoc(
            _request(
                small_root,
                artifact,
                audit_fraction=0.25,
                traffic_sample_size=30,
                **common,
            )
        )
        large = run_posthoc(
            _request(
                large_root,
                artifact,
                audit_fraction=0.5,
                traffic_sample_size=60,
                alpha=0.2,
                **common,
            )
        )
        assert (
            small["posthoc_config"]["audit_sampling_seed"]
            == large["posthoc_config"]["audit_sampling_seed"]
        )
        assert (
            small["posthoc_config"]["traffic_sampling_seed"]
            == large["posthoc_config"]["traffic_sampling_seed"]
        )
        assert {row["cert_observation_count"] for row in small["rows"]} == {90}
        assert {row["cert_observation_count"] for row in large["rows"]} == {180}
        assert {row["traffic_observation_count"] for row in small["rows"]} == {30}
        assert {row["traffic_observation_count"] for row in large["rows"]} == {60}


def test_sampling_is_invariant_to_labels_and_logits():
    with tempfile.TemporaryDirectory() as root:
        results = []
        for name, options in (
            ("original", {}),
            ("labels", {"changed_labels": True}),
            ("logits", {"changed_logits": True}),
        ):
            artifact = os.path.join(root, f"{name}.npz")
            _write_artifact(artifact, traffic=True, **options)
            results.append(
                run_posthoc(
                    _request(
                        os.path.join(root, name),
                        artifact,
                        mixture_mode="traffic",
                        delta_conditional_risk=0.03,
                        delta_acceptance_lower=0.025,
                        delta_acceptance_upper=0.025,
                        delta_mixture=0.02,
                        audit_fraction=0.5,
                        traffic_sample_size=60,
                    )
                )
            )
        configs = [result["posthoc_config"] for result in results]
        for field in (
            "audit_sampling_seed",
            "audit_sample_ids_sha256",
            "traffic_sampling_seed",
            "traffic_sample_ids_sha256",
            "cert_sampling_identity_sha256",
            "traffic_sampling_identity_sha256",
        ):
            assert len({config[field] for config in configs}) == 1
        assert results[0]["fold_sha256"] != results[1]["fold_sha256"]


def test_unknown_certificate_variant_and_unverified_provenance_fail_closed():
    fixtures = (
        ({}, {"certificate_variant": "pooled_binomial"}),
        ({"omit_seed_ledger": True, "raw_seed_fallback": True}, {}),
        ({"omit_dataset_hash": True}, {}),
        ({"bad_dataset_hash": True}, {}),
        ({"mismatched_dataset_hash": True}, {}),
    )
    for artifact_options, request_options in fixtures:
        with tempfile.TemporaryDirectory() as root:
            artifact = os.path.join(root, "logits.npz")
            _write_artifact(artifact, **artifact_options)
            _expect_failure(
                lambda: run_posthoc(_request(root, artifact, **request_options))
            )
            assert not os.path.exists(os.path.join(root, "result.json"))


def test_overlap_missing_ids_bad_hash_and_traffic_outcomes_fail_closed():
    fixtures = (
        ({"overlap": True}, {}),
        ({"omit_ids": True}, {}),
        ({"bad_config_hash": True}, {}),
        (
            {"traffic": True, "traffic_outcome": True},
            {
                "mixture_mode": "traffic",
                "delta_conditional_risk": 0.03,
                "delta_acceptance_lower": 0.025,
                "delta_acceptance_upper": 0.025,
                "delta_mixture": 0.02,
            },
        ),
    )
    for artifact_options, request_options in fixtures:
        with tempfile.TemporaryDirectory() as root:
            artifact = os.path.join(root, "logits.npz")
            _write_artifact(artifact, **artifact_options)
            _expect_failure(
                lambda: run_posthoc(_request(root, artifact, **request_options))
            )
            assert not os.path.exists(os.path.join(root, "result.json"))


def main():
    test_simplex_replays_four_policies_and_writes_bound_manifest()
    test_declared_rho_box_runs_without_a_mixture_confidence_spend()
    test_traffic_box_consumes_identity_only_and_records_common_traffic_count()
    test_nested_audit_and_traffic_subsamples_share_analysis_invariant_seeds()
    test_sampling_is_invariant_to_labels_and_logits()
    test_unknown_certificate_variant_and_unverified_provenance_fail_closed()
    test_overlap_missing_ids_bad_hash_and_traffic_outcomes_fail_closed()
    print("one-shot posthoc CLI tests: PASS")


if __name__ == "__main__":
    main()
