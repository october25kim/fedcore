"""STEP 4(f): necessity on real-data scores -- naive deploy is unsafe > delta.

Using the model's ACTUAL per-client open-set scores (exported by ``run_cifar.py``
to ``*_logits.npz``), we bootstrap the certification fold and ask, for each
deploy rule, how often it deploys a configuration whose held-out TEST risk
exceeds ``alpha`` (an UNSAFE deployment):

* ``naive``    -- deploy iff pooled empirical risk on the cert fold <= alpha;
* ``Fed-CORE`` -- deploy iff the conditional certificate <= alpha.

The certificate guarantees the Fed-CORE unsafe-deploy rate <= delta; the naive
rule should exceed delta. If no npz is supplied, a synthetic scored population
(the smoke generator) is used so the script runs on CPU.

Run: ``python experiments/fedcore/exp_necessity_real.py [--npz runs/<tag>_logits.npz]``
"""

from __future__ import annotations

import argparse

import numpy as np

from fedcore.certificate import conditional_risk_certificate
from fedcore.experiments.run_smoke import SmokeSpec, generate_smoke
from fedcore.scores import compute_score, scored_views
from fedcore.selector import choose_threshold, counts_per_client, empirical_risk_coverage, open_set_error

ALPHA = 0.10
DELTA = 0.10
GAMMA = 1.0
SCORE = "energy"
N_BOOT = 400
SEED = 0


def _load(npz_path):
    if npz_path:
        d = np.load(npz_path)
        return ({k: d[f"cert_{k}"] for k in ("logits", "y_open", "client")},
                {k: d[f"test_{k}"] for k in ("logits", "y_open", "client")},
                int(d["cert_client"].max()) + 1)
    spec = SmokeSpec()
    data = generate_smoke(spec)
    cert = {k: data["cert"][k] for k in ("logits", "y_open", "client")}
    test = {k: data["test"][k] for k in ("logits", "y_open", "client")}
    return cert, test, spec.n_clients


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="path to *_logits.npz from run_cifar")
    ap.add_argument("--score", default=SCORE)
    args = ap.parse_args()

    cert, test, n_clients = _load(args.npz)
    src = args.npz or "synthetic smoke"
    print(f"STEP 4(f) necessity on real-data scores (source={src}, "
          f"score={args.score}, alpha={ALPHA}, delta={DELTA})")

    cert_score = compute_score(args.score, cert["logits"])
    cert_pred = cert["logits"].argmax(1)
    test_score = compute_score(args.score, test["logits"])
    test_pred = test["logits"].argmax(1)
    test_err = open_set_error(test_pred, test["y_open"])

    rng = np.random.default_rng(SEED)
    n = len(cert_score)
    naive_unsafe = fed_unsafe = naive_deploy = fed_deploy = 0

    for _ in range(N_BOOT):
        bi = rng.integers(0, n, size=n)  # bootstrap the cert fold
        s, p, yo, cl = cert_score[bi], cert_pred[bi], cert["y_open"][bi], cert["client"][bi]

        # choose a threshold targeting gamma*alpha on this resample (proposal proxy)
        sel = choose_threshold(s, p, yo, GAMMA, ALPHA)

        # naive deploy: pooled empirical risk on the cert fold <= alpha
        err = open_set_error(p, yo)
        _, emp_risk = empirical_risk_coverage(s, err, sel.threshold)
        naive = sel.feasible and emp_risk <= ALPHA

        # Fed-CORE deploy: conditional certificate <= alpha
        A, K, nn = counts_per_client(s, p, yo, cl, sel, n_clients)
        U = conditional_risk_certificate(A, K, nn, DELTA, Lambda="simplex").U
        fed = sel.feasible and U <= ALPHA

        # ground-truth safety: held-out TEST risk at this threshold
        _, true_risk = empirical_risk_coverage(test_score, test_err, sel.threshold)
        unsafe = true_risk > ALPHA

        naive_deploy += naive
        fed_deploy += fed
        naive_unsafe += naive and unsafe
        fed_unsafe += fed and unsafe

    print(f"\n{'rule':>10} {'deploy%':>9} {'unsafe-deploy%':>15} {'<=delta':>9}")
    print("-" * 46)
    nu = naive_unsafe / N_BOOT
    fu = fed_unsafe / N_BOOT
    print(f"{'naive':>10} {100*naive_deploy/N_BOOT:>8.1f}% {100*nu:>14.1f}% "
          f"{str(nu <= DELTA):>9}")
    print(f"{'Fed-CORE':>10} {100*fed_deploy/N_BOOT:>8.1f}% {100*fu:>14.1f}% "
          f"{str(fu <= DELTA):>9}")
    print("-" * 46)
    print(f"Expectation: naive unsafe-deploy rate > delta={DELTA}; Fed-CORE <= delta.")


if __name__ == "__main__":
    main()
