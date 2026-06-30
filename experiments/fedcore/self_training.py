"""Proposition 4: certified federated self-training (core loop logic).

The idea: turn the Fed-CORE certificate into an *admission control* for
pseudo-labels. The trusted calibration set is partitioned into ``T`` disjoint
audit folds. At round ``t`` the model proposes a selector, certifies the accepted
pseudo-label error on audit fold ``C^(t)`` at the per-round level ``delta / T``
(union bound over rounds), and folds the newly accepted pseudo-labels back into
training ONLY IF the certificate is ``<= alpha``. This guarantees that the
contamination injected per round is certified ``<= alpha`` with overall
confidence ``1 - delta``.

This module is torch-free: it exposes the decision logic (:func:`round_decision`)
plus a synthetic accuracy-dynamics model so the qualitative behavior (certified
stays safe and improves; naive diverges under non-IID corruption) can be
demonstrated on CPU. The same :func:`round_decision` gate can wrap a real torch
training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from certificates import conditional_risk_certificate, cp_upper


@dataclass
class RoundDecision:
    """Outcome of one self-training round's admission check."""

    cert_risk_ucb: float
    admit: bool
    A: np.ndarray
    K: np.ndarray
    n: np.ndarray


def round_decision(
    A,
    K,
    n,
    *,
    alpha: float,
    delta: float,
    n_rounds: int,
    Lambda: str = "simplex",
    lam=None,
    box: float = 0.15,
    seed: int = 0,
    gated: bool = True,
) -> RoundDecision:
    """Certify this round's accepted pseudo-labels and decide whether to admit.

    The certificate uses the per-round budget ``delta / n_rounds`` (union bound
    over ``n_rounds`` audit folds). With ``gated=False`` (the naive baseline) the
    pseudo-labels are admitted regardless of the certificate.
    """
    per_round_delta = delta / n_rounds
    cert = conditional_risk_certificate(
        A, K, n, per_round_delta, Lambda=Lambda, lam=lam, box=box, seed=seed
    )
    admit = True if not gated else bool(cert.feasible and cert.U <= alpha)
    return RoundDecision(
        cert_risk_ucb=float(cert.U), admit=admit,
        A=np.asarray(A), K=np.asarray(K), n=np.asarray(n),
    )


# --------------------------------------------------------------------------- #
# Synthetic accuracy-dynamics model (CPU demonstration)
# --------------------------------------------------------------------------- #
def _client_contamination(acc: float, corruption: float) -> float:
    """Contamination of the high-confidence pseudo-labels a client proposes.

    Clean clients: as accuracy rises the accepted (confident) pseudo-labels get
    cleaner, so contamination -> a small floor. A corrupted client's confidence
    is miscalibrated, so its confident pseudo-labels stay wrong at roughly the
    corruption rate (and worsen if the model degrades).
    """
    base = 0.02 + 0.25 * (1.0 - acc)
    return float(np.clip(base + corruption, 0.0, 1.0))


def _accuracy_update(
    acc: float, n_clean: float, n_wrong: float, n_ref: float, eta: float, beta: float
) -> float:
    """Confirmation-bias accuracy update.

    Admitted CLEAN pseudo-labels pull accuracy up toward 1; admitted WRONG
    pseudo-labels push it down, amplified by ``beta`` (the model learns the wrong
    label and compounds it next round -> divergence when wrong labels dominate).
    """
    up = (n_clean / n_ref) * (1.0 - acc)
    down = beta * (n_wrong / n_ref) * acc
    return float(np.clip(acc + eta * (up - down), 0.0, 1.0))


def simulate_self_training(
    *,
    mode: str,                 # 'certified' | 'naive' | 'none'
    n_clients: int = 5,
    n_rounds: int = 10,
    audit_n_per_client: int = 200,
    corruption_bad: float = 0.45,
    corruption_good: float = 0.0,
    alpha: float = 0.10,
    delta: float = 0.10,
    gamma: float = 0.7,
    init_acc: float = 0.75,
    eta: float = 0.6,
    beta: float = 2.5,
    Lambda: str = "simplex",
    seed: int = 0,
):
    """Run a synthetic self-training trajectory and return per-round records.

    Pseudo-labels are admitted PER CLIENT: a client's confident pseudo-labels are
    folded back into training only if that client passes the per-client
    certificate (level ``delta / (T * J)``). The naive baseline admits every
    client; ``none`` never self-trains.
    """
    rng = np.random.default_rng(seed)
    acc = init_acc
    corr = np.array([corruption_good] * (n_clients - 1) + [corruption_bad])
    n = np.full(n_clients, audit_n_per_client)
    J = n_clients
    n_ref = float(audit_n_per_client * 0.6 * n_clients)
    per_client_eps = delta / (n_rounds * J)

    records = []
    for t in range(n_rounds):
        if mode == "none":
            records.append({"round": t, "acc": acc, "cert_risk_ucb": np.nan,
                            "n_admit_clients": 0, "true_contam": np.nan})
            continue

        contam = np.array([_client_contamination(acc, c) for c in corr])
        A = (n * 0.6).astype(int)
        K = rng.binomial(A, contam)

        # the audit certificate is computed on the same counts (the trusted fold
        # mirrors the per-client accepted error); record the simplex UCB
        dec = round_decision(
            A, K, n, alpha=alpha, delta=delta, n_rounds=n_rounds,
            Lambda=Lambda, seed=seed + t, gated=False,
        )

        # per-client admission decision
        n_clean = n_wrong = 0.0
        n_admit_clients = 0
        for j in range(J):
            if mode == "certified":
                rbar_j = cp_upper(int(K[j]), int(A[j]), per_client_eps)
                admit_j = rbar_j <= alpha
            else:  # naive: admit all clients
                admit_j = True
            if admit_j:
                n_admit_clients += 1
                n_wrong += K[j]
                n_clean += A[j] - K[j]

        acc = _accuracy_update(acc, n_clean, n_wrong, n_ref, eta, beta)
        admit_contam = n_wrong / max(1.0, n_clean + n_wrong)

        records.append({
            "round": t, "acc": acc, "cert_risk_ucb": dec.cert_risk_ucb,
            "n_admit_clients": n_admit_clients, "true_contam": admit_contam,
        })
    return records
