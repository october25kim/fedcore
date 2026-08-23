"""Reservoir-accounting figures generated only from source rows."""

from __future__ import annotations

from collections import defaultdict
import os
from typing import Sequence

import numpy as np


def _save(fig, stem: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=180, bbox_inches="tight")


def make_unique_fraction(rows: Sequence[dict], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    points = defaultdict(list)
    for row in rows:
        if row["draw_mode"] != "audit_bootstrap" or row["reservoir_size_M"] <= 0:
            continue
        M, n = row["reservoir_size_M"], row["requested_draw_count_n"]
        if n <= 0:
            continue
        points[float(n) / M].append(float(row["unique_sampled_count"]) / n)
    x = np.array(sorted(points))
    y = np.array([np.mean(points[value]) for value in x])
    curve_x = np.linspace(0.01, max(4.2, float(x.max()) if len(x) else 1.0), 400)
    # E[unique]/n = (1-exp(-n/M))/(n/M), shown with exact finite-M points in CSV.
    curve_y = -np.expm1(-curve_x) / curve_x
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    if len(x):
        ax.plot(x, y, "o", label="Archived redraws")
    ax.plot(curve_x, curve_y, "-", label="Occupancy approximation")
    ax.set(
        xlabel="Nominal draws / reservoir size",
        ylabel="Distinct / nominal draws",
        ylim=(0.0, 1.02),
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "unique_fraction_vs_ratio"))
    plt.close(fig)


def make_label_budget(rows: Sequence[dict], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    modes = defaultdict(
        lambda: {"operational": {}, "test": {}, "nominal": 0, "unique": 0}
    )
    for row in rows:
        mode = row["draw_mode"]
        run_id = row["run_id"]
        modes[mode]["operational"][run_id] = row["operational_unique_trusted_labels"]
        modes[mode]["test"][run_id] = row["research_only_test_labels"]
        modes[mode]["nominal"] += row["nominal_certification_draws"]
        modes[mode]["unique"] += row["unique_labels_used_in_certification_draw"]
    labels = sorted(modes)
    operational = [sum(modes[m]["operational"].values()) for m in labels]
    research = [sum(modes[m]["test"].values()) for m in labels]
    nominal = [modes[m]["nominal"] for m in labels]
    unique = [modes[m]["unique"] for m in labels]
    fig, ax = plt.subplots(figsize=(max(5.2, 1.3 * len(labels)), 3.8))
    x = np.arange(len(labels))
    width = 0.2
    ax.bar(x - 1.5 * width, operational, width, label="Operational unique labels")
    ax.bar(x - 0.5 * width, research, width, label="Research-only test labels")
    ax.bar(x + 0.5 * width, nominal, width, label="Nominal certification draws")
    ax.bar(x + 1.5 * width, unique, width, label="Unique certification labels")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Count across accounted rows")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "label_budget_separation"))
    plt.close(fig)


def make_all_figures(rows: Sequence[dict], out_dir: str) -> None:
    make_unique_fraction(rows, out_dir)
    make_label_budget(rows, out_dir)
