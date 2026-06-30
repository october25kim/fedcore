"""Fake-logit end-to-end smoke test (no torch).

Synthesizes ``J`` heterogeneous clients with a built-in *confidence deformation*
(a fraction of unknown points get a spurious high-confidence known boost, so
they look acceptable but are errors) and one hardest / most-contaminated client.
It then runs the IDENTICAL certification path used for real CIFAR
(``scored_views`` -> ``certify_grid``) for ``Lambda in {simplex, box}``.

Purpose: validate wiring, not science. Expected behavior (acceptance gate):
the full metric schema is emitted; the box-Lambda certificate certifies a few
``(score, gamma=0.5)`` combos with ``cert_risk_ucb`` just under ``alpha`` while
the (worst-case) simplex certificate certifies none.

Run: ``python experiments/fedcore/run_smoke.py``
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from fedcore.certify import METRIC_KEYS, certify_best_gamma, certify_grid
from fedcore.scores import scored_views


# --------------------------------------------------------------------------- #
# Synthetic generator
# --------------------------------------------------------------------------- #
@dataclass
class SmokeSpec:
    """Knobs for the synthetic FedOSR smoke (tuned to the acceptance gate)."""

    n_known: int = 6
    n_clients: int = 5
    n_known_per_client: int = 700          # known points per client (pre-split)
    unknown_contamination: float = 0.30
    folds: Tuple[float, float, float] = (0.4, 0.3, 0.3)
    # per-client known-class confidence (higher => more correct & confident).
    # Clean knowns are well-separated (mu) so a threshold can exclude the
    # spurious unknowns, whose boost is deliberately smaller (def_boost < mu).
    mu_good: float = 6.0
    mu_bad: float = 4.0
    # spurious-confidence rate among unknown points (the deformation); the
    # high-risk last client is far more contaminated.
    def_rate_good: float = 0.08
    def_rate_bad: float = 0.50
    def_boost: float = 2.0
    seed: int = 0


def _gen_known(rng, n, n_known, mu) -> Tuple[np.ndarray, np.ndarray]:
    """Generate known-point logits and their true remapped labels."""
    y = rng.integers(0, n_known, size=n)
    logits = rng.normal(0.0, 1.0, size=(n, n_known))
    logits[np.arange(n), y] += mu
    return logits, y


def _gen_unknown(rng, n, n_known, def_rate, boost) -> np.ndarray:
    """Generate unknown-point logits; a ``def_rate`` fraction get a spurious boost."""
    logits = rng.normal(0.0, 1.0, size=(n, n_known))
    deformed = rng.random(n) < def_rate
    boost_cls = rng.integers(0, n_known, size=n)
    logits[np.arange(n)[deformed], boost_cls[deformed]] += boost
    return logits


def generate_smoke(spec: SmokeSpec):
    """Return per-fold (logits, y_open, client) arrays for the smoke."""
    rng = np.random.default_rng(spec.seed)
    pf, cf, tf = spec.folds
    fold_names = ("prop", "cert", "test")

    fold_data: Dict[str, Dict[str, list]] = {
        fn: {"logits": [], "y_open": [], "client": []} for fn in fold_names
    }

    for j in range(spec.n_clients):
        is_bad = j == spec.n_clients - 1
        mu = spec.mu_bad if is_bad else spec.mu_good
        def_rate = spec.def_rate_bad if is_bad else spec.def_rate_good

        n_k = spec.n_known_per_client
        n_unk = int(round(spec.unknown_contamination / (1 - spec.unknown_contamination) * n_k))

        klog, ky = _gen_known(rng, n_k, spec.n_known, mu)
        ulog = _gen_unknown(rng, n_unk, spec.n_known, def_rate, spec.def_boost)

        # shuffle and split each pool into prop/cert/test
        for logits, labels in ((klog, ky), (ulog, np.full(len(ulog), -1))):
            n = len(logits)
            perm = rng.permutation(n)
            logits, labels = logits[perm], labels[perm]
            n_prop = int(round(n * pf))
            n_cert = int(round(n * cf))
            bounds = [(0, n_prop), (n_prop, n_prop + n_cert), (n_prop + n_cert, n)]
            for (s, e), fn in zip(bounds, fold_names):
                fold_data[fn]["logits"].append(logits[s:e])
                fold_data[fn]["y_open"].append(labels[s:e])
                fold_data[fn]["client"].append(np.full(e - s, j))

    out = {}
    for fn in fold_names:
        out[fn] = {
            "logits": np.concatenate(fold_data[fn]["logits"], axis=0),
            "y_open": np.concatenate(fold_data[fn]["y_open"], axis=0),
            "client": np.concatenate(fold_data[fn]["client"], axis=0),
        }
    return out


# --------------------------------------------------------------------------- #
# reporting helpers (shared shape with run_cifar)
# --------------------------------------------------------------------------- #
def print_metric_table(rows: List[Dict[str, object]]) -> None:
    """Print the canonical metric schema as a fixed-width table."""
    cols = METRIC_KEYS
    widths = {c: max(len(c), 8) for c in cols}
    header = "  ".join(f"{c:>{widths[c]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:>{widths[c]}.4f}")
            else:
                cells.append(f"{str(v):>{widths[c]}}")
        print("  ".join(cells))


def save_csv(rows: List[Dict[str, object]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in METRIC_KEYS})


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def best_gamma_sanity(alpha: float = 0.10, delta: float = 0.10,
                      gammas=(0.2, 0.3, 0.5, 0.7, 1.0)) -> None:
    """CPU sanity for certify_best_gamma: (i) runs, (ii) certified => test_risk
    <= alpha, (iii) gamma-selection-on-proposal does not break validity (the
    unsafe rate among certified deploys stays <= delta over many trials, using a
    large test fold as ground truth)."""
    scores = ("msp", "neg_entropy", "margin", "energy")
    print("\n=== certify_best_gamma CPU sanity ===")
    # (i)+(ii) single shot on the default smoke
    spec = SmokeSpec()
    data = generate_smoke(spec)
    views = {fn: scored_views(data[fn]["logits"], data[fn]["y_open"],
                              data[fn]["client"], list(scores)) for fn in ("prop", "cert", "test")}
    for s in scores:
        r = certify_best_gamma(views["prop"][s], views["cert"][s], views["test"][s],
                               score_name=s, gammas=gammas, alpha=alpha, delta=delta,
                               n_clients=spec.n_clients, dirichlet_alpha=float("nan"),
                               Lambda="box", box=0.10, seed=spec.seed)
        flag = "OK" if (not r["certified"] or r["test_risk"] <= alpha) else "VIOLATION"
        print(f"  {s:>11}: gamma*={r['gamma_star']} certified={r['certified']} "
              f"u_proxy={r['u_proxy']:.3f} cert_ucb={r['cert_risk_ucb']:.3f} "
              f"cov_lcb={r['cert_coverage_lcb']:.3f} test_risk={r['test_risk']:.3f} [{flag}]")
        assert (not r["certified"]) or (r["test_risk"] <= alpha + 1e-9), \
            f"certified but test_risk>alpha for {s}"

    # (iii) validity Monte-Carlo: certified deploys must be unsafe <= delta
    n_trials, unsafe, n_cert = 120, 0, 0
    for t in range(n_trials):
        # cleaner population so a meaningful fraction certifies (real validity test)
        sp = SmokeSpec(seed=1000 + t, n_known_per_client=600, mu_good=7.0,
                       mu_bad=6.0, def_boost=1.5, def_rate_good=0.03, def_rate_bad=0.20)
        d = generate_smoke(sp)
        v = {fn: scored_views(d[fn]["logits"], d[fn]["y_open"], d[fn]["client"],
                              ["energy"]) for fn in ("prop", "cert", "test")}
        r = certify_best_gamma(v["prop"]["energy"], v["cert"]["energy"], v["test"]["energy"],
                               score_name="energy", gammas=gammas, alpha=alpha, delta=delta,
                               n_clients=sp.n_clients, dirichlet_alpha=float("nan"),
                               Lambda="simplex", seed=sp.seed)
        if r["certified"]:
            n_cert += 1
            if r["test_risk"] > alpha:   # large test fold ~ ground truth
                unsafe += 1
    rate = unsafe / n_cert if n_cert else 0.0
    print(f"  validity MC: {n_cert}/{n_trials} certified; unsafe-among-certified "
          f"= {rate:.3f} (<= delta={delta}: {rate <= delta})")


def main() -> None:
    spec = SmokeSpec()
    alpha, delta = 0.10, 0.10
    gammas = (0.5, 0.7, 1.0)
    scores = ("msp", "neg_entropy", "margin", "energy")
    box_radius = 0.10

    data = generate_smoke(spec)
    views = {
        fn: scored_views(
            data[fn]["logits"], data[fn]["y_open"], data[fn]["client"], list(scores)
        )
        for fn in ("prop", "cert", "test")
    }

    rows = certify_grid(
        views["prop"], views["cert"], views["test"],
        scores=scores, gammas=gammas, alpha=alpha, delta=delta,
        Lambdas=("simplex", "box"), n_clients=spec.n_clients,
        dirichlet_alpha=float("nan"), box=box_radius, seed=spec.seed,
    )

    print_metric_table(rows)

    n_cert_box = sum(1 for r in rows if r["Lambda"] == "box" and r["certified"])
    n_cert_sx = sum(1 for r in rows if r["Lambda"] == "simplex" and r["certified"])
    print(f"\ncertified: simplex {n_cert_sx}/12, box {n_cert_box}/12")

    # smoke_results.csv stays in the flat experiments/fedcore dir (gitignored), NOT inside the
    # package. After the project-root hoist this module is fedcore/fedcore/experiments/run_smoke.py,
    # so anchor two levels up to the project root, then into experiments/fedcore -- preserving the
    # exact pre-hoist output path so the structure-only move does not relocate the artifact.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(project_root, "experiments", "fedcore")
    out_path = os.path.join(out_dir, "smoke_results.csv")
    save_csv(rows, out_path)
    print(f"saved {out_path}")

    best_gamma_sanity(alpha, delta, gammas)


if __name__ == "__main__":
    main()
