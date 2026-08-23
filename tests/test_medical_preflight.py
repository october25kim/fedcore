"""Metadata-only medical preflight fixture tests."""

from __future__ import annotations

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcore.medical.preflight import PreflightConfig, run_preflight  # noqa: E402


def test_units_never_cross_folds_and_replay_is_deterministic():
    with tempfile.TemporaryDirectory() as td:
        metadata = os.path.join(td, "metadata.csv")
        fields = ["center", "diagnosis", "patient", "lesion", "image"]
        rows = []
        for center in ("A", "B"):
            for diagnosis in ("d0", "d1"):
                for unit in range(40):
                    patient = f"{center}-{diagnosis}-p{unit}"
                    lesion = f"{center}-{diagnosis}-l{unit}"
                    for image in range(2):
                        rows.append(
                            {
                                "center": center,
                                "diagnosis": diagnosis,
                                "patient": patient,
                                "lesion": lesion,
                                "image": f"{lesion}-i{image}",
                            }
                        )
        with open(metadata, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        config = PreflightConfig(
            metadata_csv=metadata,
            center_col="center",
            diagnosis_col="diagnosis",
            patient_col="patient",
            lesion_col="lesion",
            image_col="image",
            unit_col="lesion",
            planning_acceptance_rate=0.8,
            alpha=0.2,
            delta=0.1,
            seed=11,
            out_dir=os.path.join(td, "out"),
        )
        first = run_preflight(config)
        first_hash = first["fold_sha256"]
        second = run_preflight(config)
        assert second["fold_sha256"] == first_hash
        with open(first["folds_path"], newline="", encoding="utf-8") as handle:
            fold_rows = list(csv.DictReader(handle))
        assert len(fold_rows) == 160
        assert len({row["audit_unit_id"] for row in fold_rows}) == 160
        # Repeated images are represented inside one unit row, never Bernoulli rows.
        assert all(row["image_ids_json"].count("-i") == 2 for row in fold_rows)


def main():
    test_units_never_cross_folds_and_replay_is_deterministic()
    print("medical preflight tests: PASS")


if __name__ == "__main__":
    main()
