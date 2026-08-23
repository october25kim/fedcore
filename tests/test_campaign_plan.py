"""One-shot matrix separation and completeness tests."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcore.campaign.plan import (  # noqa: E402
    expand_training_cells,
    training_cell_experiment_id,
    validate_plan,
    validate_terminal_coverage,
    validate_training_cell_binding,
)
from fedcore.certificate.variants import (  # noqa: E402
    JOINT_CONDITIONAL_CERTIFICATE_VARIANT,
)


def _plan():
    return {
        "schema_version": 1,
        "campaign_seed": 123,
        "cifar": {
            "class_splits": [
                {"id": f"s{i}", "unknown_classes": [i, (i + 5) % 10]} for i in range(5)
            ],
            "model_seeds": [0, 1, 2],
            "dirichlet_alpha": [0.1, 0.5, 5.0],
            "training": {"dataset": "cifar10", "rounds": 1},
        },
        "medical": {
            "heldout_diagnoses": [f"d{i}" for i in range(8)],
            "model_seeds": [0, 1, 2],
            "training": {"rounds": 1},
        },
        "posthoc": {
            "alpha": [0.1],
            "total_delta": [0.05, 0.1],
            "rho": [0.0, 0.1],
            "threshold_policies": ["global", "client_specific"],
            "allocation_policies": ["uniform", "proposal_informed"],
            "audit_budget_fractions": [0.5, 1.0],
            "traffic_sample_sizes": [100],
            "scores": ["msp"],
            "certificate_variants": [JOINT_CONDITIONAL_CERTIFICATE_VARIANT],
            "gammas": [0.5, 0.7, 1.0],
        },
    }


def test_matrix_is_45_plus_24_and_posthoc_does_not_multiply_training():
    plan = _plan()
    validate_plan(plan)
    cells = expand_training_cells(plan)
    assert len(cells) == 69
    assert len({cell.experiment_id for cell in cells}) == 69
    assert all(cell.config["campaign_seed"] == 123 for cell in cells)
    mutated = _plan()
    mutated["posthoc"]["alpha"] = [0.05, 0.1, 0.2]
    mutated["posthoc"]["scores"] = ["msp", "energy"]
    assert [cell.experiment_id for cell in expand_training_cells(mutated)] == [
        cell.experiment_id for cell in cells
    ]


def test_posthoc_key_in_training_is_rejected():
    plan = _plan()
    plan["cifar"]["training"]["alpha"] = 0.1
    try:
        expand_training_cells(plan)
    except ValueError:
        pass
    else:
        raise AssertionError("post-hoc alpha entered training config")


def test_completeness_fails_loudly():
    cells = expand_training_cells(_plan())
    statuses = {cell.experiment_id: "succeeded" for cell in cells}
    validate_terminal_coverage(cells, statuses)
    statuses.pop(cells[0].experiment_id)
    try:
        validate_terminal_coverage(cells, statuses)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing matrix cell was silently accepted")


def test_runner_binding_uses_the_same_cell_id_and_fails_on_drift():
    cell = expand_training_cells(_plan())[0]
    assert training_cell_experiment_id(cell.config) == cell.experiment_id
    validate_training_cell_binding(
        cell.config,
        {
            "family": "cifar",
            "split_id": cell.config["split_id"],
            "model_seed": cell.config["model_seed"],
            "campaign_seed": 123,
        },
    )
    try:
        validate_training_cell_binding(cell.config, {"model_seed": 999})
    except ValueError:
        pass
    else:
        raise AssertionError("runner/plan model-seed drift was accepted")


def test_unregistered_certificate_variant_fails_at_plan_gate():
    plan = _plan()
    plan["posthoc"]["certificate_variants"] = ["pooled_binomial"]
    try:
        validate_plan(plan)
    except ValueError:
        pass
    else:
        raise AssertionError("unimplemented certificate variant passed plan validation")


def main():
    test_matrix_is_45_plus_24_and_posthoc_does_not_multiply_training()
    test_posthoc_key_in_training_is_rejected()
    test_completeness_fails_loudly()
    test_runner_binding_uses_the_same_cell_id_and_fails_on_drift()
    test_unregistered_certificate_variant_fails_at_plan_gate()
    print("campaign plan tests: PASS")


if __name__ == "__main__":
    main()
