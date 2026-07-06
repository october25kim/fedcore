"""Fake-logit FedOSR smoke test -- runs end-to-end in the sandbox (no torch).

Synthesizes logits + open-set labels across heterogeneous clients with a built-in
confidence DEFORMATION (unknown points can produce high-confidence known
predictions, and one client is both harder and more unknown-contaminated). This
exercises the full score -> risk-buffered selector -> stratified certificate path
and emits the project's metric schema, mirroring the project's fake-logit Docker
smoke before committing GPU time to real CIFAR.
"""
from __future__ import annotations

import csv

import numpy as np

from certify import certify_grid
from config import FedOSRConfig
from scores import compute_score


def _synth_fold(
    rng: np.random.Generator,
    n_per_client: int,
    n_clients: int,
    C: int,
    unk_frac_by_client: np.ndarray,
    acc_by_client: np.ndarray,
    deform: float,
) -> dict:
    """Generate logits, open-set labels, and client ids for one fold.

    known point: true class logit boosted so argmax==true with prob ~acc_j.
    unknown point: flat-ish logits, but with probability ``deform`` it gets a
    spurious boost on a random known class (high-confidence wrong accept).
    """
    logits_all, y_all, client_all = [], [], []
    for j in range(n_clients):
        nj = n_per_client
        n_unk = int(round(unk_frac_by_client[j] * nj))
        n_known = nj - n_unk

        # known points
        z = rng.normal(0, 1.0, size=(n_known, C))
        y = rng.integers(0, C, size=n_known)
        boost = rng.normal(2.5, 1.0, size=n_known) * acc_by_client[j]
        z[np.arange(n_known), y] += np.clip(boost, 0, None)

        # unknown points (open-set label = -1)
        zu = rng.normal(0, 1.0, size=(n_unk, C))
        deformed = rng.random(n_unk) < deform
        if deformed.any():
            cls = rng.integers(0, C, size=int(deformed.sum()))
            zu[np.where(deformed)[0], cls] += rng.normal(2.2, 0.8, size=int(deformed.sum()))
        yu = np.full(n_unk, -1)

        logits_all.append(np.vstack([z, zu]))
        y_all.append(np.concatenate([y, yu]))
        client_all.append(np.full(nj, j))

    logits = np.vstack(logits_all)
    y_open = np.concatenate(y_all)
    client = np.concatenate(client_all)
    pred = logits.argmax(axis=1)
    return {"logits": logits, "y_open": y_open, "client": client, "pred": pred}


def _scored(fold: dict, score_names) -> dict:
    """Attach each named score to per-score views of a fold."""
    out = {}
    for s in score_names:
        out[s] = {
            "score": compute_score(s, fold["logits"]),
            "pred": fold["pred"],
            "y_open": fold["y_open"],
            "client": fold["client"],
        }
    return out


def main() -> None:
    cfg = FedOSRConfig(n_known=6, n_clients=5, dirichlet_alpha=0.1, alpha=0.10, delta=0.10)
    rng = np.random.default_rng(cfg.seed)
    C = cfg.n_known
    J = cfg.n_clients

    # heterogeneity: client 4 is hardest + most unknown-contaminated (deformation)
    acc_by_client = np.array([1.0, 1.0, 0.97, 0.95, 0.9])
    unk_frac = np.array([0.12, 0.12, 0.15, 0.15, 0.20])

    prop_fold = _synth_fold(rng, 1000, J, C, unk_frac, acc_by_client, deform=0.06)
    cert_fold = _synth_fold(rng, 1000, J, C, unk_frac, acc_by_client, deform=0.06)
    test_fold = _synth_fold(rng, 1000, J, C, unk_frac, acc_by_client, deform=0.06)

    prop = _scored(prop_fold, cfg.scores)
    cert = _scored(cert_fold, cfg.scores)
    test = _scored(test_fold, cfg.scores)

    # box Lambda around the (known, here uniform) client data fractions +/- radius
    base = np.full(J, 1.0 / J)
    box = (np.clip(base - 0.10, 0, 1), np.clip(base + 0.10, 0, 1))

    cols = ["score_name", "gamma", "certified", "cert_risk_ucb", "cert_coverage_lcb",
            "cert_n", "cert_k", "prop_coverage", "prop_risk", "test_coverage", "test_risk"]
    print(f"Fed-CORE FAKE-LOGIT SMOKE | alpha={cfg.alpha} delta={cfg.delta} "
          f"J={J} C={C}")
    print("(certified=True iff cert_risk_ucb <= alpha)")

    all_rows: list[dict] = []
    for Lambda in ("simplex", "box"):
        rows = certify_grid(
            prop=prop, cert=cert, test=test,
            score_names=cfg.scores, gammas=cfg.gammas,
            alpha=cfg.alpha, delta=cfg.delta,
            n_clients=J, dirichlet_alpha=cfg.dirichlet_alpha,
            Lambda=Lambda, box=box if Lambda == "box" else None,
        )
        for r in rows:
            r["Lambda"] = Lambda
        all_rows.extend(rows)
        n_cert = sum(r["certified"] for r in rows)
        print(f"\n--- Lambda = {Lambda}  ({n_cert}/{len(rows)} certified) ---")
        w = {c: max(len(c), 8) for c in cols}
        print("  ".join(c.rjust(w[c]) for c in cols))
        print("-" * (sum(w.values()) + 2 * (len(cols) - 1)))
        for r in rows:
            print("  ".join(str(r[c]).rjust(w[c]) for c in cols))

    with open("smoke_results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wr.writeheader(); wr.writerows(all_rows)
    print("\nsaved -> smoke_results.csv")
    print("\nSMOKE OK: pipeline ran end-to-end and emitted the full metric schema.")
    print("Expected pattern: full-simplex is robust-but-conservative (hardest client")
    print("caps it); box-Lambda recovers tightness and certifies for small gamma.")


if __name__ == "__main__":
    main()
