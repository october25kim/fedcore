# CLAUDE.md — Fed-CORE (Federated Certified Open-Set Recognition)

> Read this file **and** `AGENTS.md` before doing anything. This file is the
> single source of truth for the project's current direction. It **supersedes**
> the older project configuration that described "Selective Risk Control after
> Corrupted Training (SRCC / Paper 1)" as the flagship — **SRCC has been
> abandoned** (for novelty reasons), and its machinery is now absorbed into
> Fed-CORE as a special case. Likewise, any instruction mentioning **RC-OWPL,
> pseudo-labeling, accept/defer/reject triage, or open-world pseudo-label
> contamination as the main object is OBSOLETE.**

## 1. What this project is

Fed-CORE certifies, with a **finite-sample distribution-free upper confidence
bound**, the **accepted selective risk** of a federated-trained open-set
classifier — the probability that an *accepted* prediction is wrong — under
client heterogeneity and unknown deployment mixture, computing the certificate
from **secure-aggregatable counts only**.

This is **not** a FedOSR-accuracy paper and **not** a noisy-label-robust-training
paper. The object is *certification of which open-set predictions can be safely
accepted*, not improving the model.

**One-line thesis.** Heterogeneity (and client-side label corruption) deforms the
confidence–correctness ranking; a small trusted clean calibration set held across
clients should be used to **certify** which predictions are safe to accept, not
to repair the model.

**Pipeline.**
```
heterogeneous/corrupted FL-trained classifier
  -> score-agnostic accept/reject proposal (risk-buffered, gamma*alpha)
  -> federated independent certification under partial exchangeability
  -> certified accepted coverage (secure-aggregation-only leakage)
```

## 2. Core theory (must be preserved)

- **Controlled object:** `R_sel(lambda) = sum_j lam_j m_j / sum_j lam_j a_j`,
  where `a_j = P_j(accept)`, `m_j = P_j(accept & error)`, `r_j = m_j/a_j`. Goal:
  certify `R_sel(lambda) <= alpha` and maximize accepted coverage.
- **Theorem 1/1' (core, non-reducible) — CONDITIONAL selective-risk certificate.**
  Use the conditional law `K_j | A_j ~ Bin(A_j, r_j)` to bound `r_j` directly:
  `rbar_j = U+(K_j, A_j; delta/J)`. Full simplex: `Ubar = max_j rbar_j` (deploy
  iff `<= alpha`). Bounded Lambda (Thm 1', recommended): also bound
  `a_j in [alow_j, ahigh_j]` and solve the robust linear-fractional
  `sup_{lam in Lambda, a in box} (sum lam a rbar)/(sum lam a)`. This is uniformly
  TIGHTER than the old mass-ratio `max_j mbar_j/alow_j` (now an App-C baseline).
  Edge cases: zero accepted coverage => non-deployable; vanishing denominator
  bound => infeasible (do NOT drop the client).
- **Non-reducibility crux.** Naive pooling of the federated accepted set into one
  binomial is **invalid** under heterogeneity (per-client `r_j` differ → the
  pooled accepted-error count is Poisson-binomial, not binomial). Not a corollary
  of centralized CP, nor of federated conformal (Lu et al., which bounds a
  *quantile/coverage*, not a post-selection *risk ratio*).
- **Theorem 2 (feasibility).** Per-client OBSERVED accepted count
  `A_j >= ln(J/delta)/(-ln(1-alpha)) = Omega(ln(J/delta)/alpha)`. Expected-count
  form is the corollary.
- **Proposition 3 (pooled, subordinate).** Tighter pooled bound `U+(sum K, sum A; delta)`
  ONLY under matched-mixture i.i.d. calibration. Two open gaps: (L) Lemma L
  (binomial CP conservative for Poisson-binomial mean — numerically SUPPORTED,
  worst-case coverage 0.919>=0.90, formal proof TODO); (C) roster-composition
  coupling between the pooled mean and `R_sel(lambda)`. Keep below Thm 1/1'.
- **Privacy taxonomy (corrected).** Only the POOLED certificate is sum-only
  secure-aggregatable. The STRATIFIED certificate needs per-client `(A_j,K_j)`;
  a GROUPED-stratified variant (G public strata, >=k clients each) secure-
  aggregates within groups as the tunable compromise. Do NOT claim "two counts
  only" for the stratified certificate.
- **Calibration assumption (state openly).** Certifying unknown rejection needs
  the certification fold to contain LABELED unknown-class points. "Distribution-
  free" is w.r.t. the calibration distribution, not the whole unknown universe.

## 3. Canonical metric schema (do not rename)

`certified, cert_risk_ucb, cert_coverage_lcb, cert_n, cert_k, prop_coverage,
prop_risk, test_coverage, test_risk, score_name, gamma, alpha, delta, Lambda,
dirichlet_alpha, n_clients`. Headline metric: **CertifiedCoverage@alpha**.

## 4. Repo layout

```
Fed-CORE_draft.md                       paper draft (Intro/RW/Method + theorems + proof sketches)
FedOSR_meta_analysis_and_novelty_brief.md   meta-analysis + novelty verification
experiments/fedcore/
  certificates.py   CP primitives + stratified (Thm1) + pooled (Thm3) certificates
  clients.py        synthetic heterogeneous client populations
  exp_lemma_L.py    Lemma L numerical verification              (runs on CPU)
  exp_pooling_fail.py  pooling-fail ablation (non-reducibility) (runs on CPU)
  config.py scores.py selector.py certify.py   numpy certification core
  fedosr_split.py   open-set split + Dirichlet non-IID + calibration folds
  models.py fed_train.py   FedAvg + logit export (torch)
  run_smoke.py      fake-logit end-to-end smoke (no torch)
  run_cifar.py      real CIFAR-10/100 FedOSR run (torch+torchvision, GPU)
```

## 5. How to run (Docker-first)

```bash
# CPU sanity (no torch needed)
python experiments/fedcore/exp_lemma_L.py
python experiments/fedcore/exp_pooling_fail.py
python experiments/fedcore/run_smoke.py

# GPU, real data (4070) — prefer the Docker wrapper
bash scripts/docker_cifar.sh                 # uses env vars; see the script
# or directly inside a torch container:
python experiments/fedcore/run_cifar.py --dataset cifar10 --n_known 6 \
    --n_clients 5 --dirichlet_alpha 0.1 --rounds 50 --local_epochs 2 \
    --alpha 0.10 --delta 0.10
```

## 6. Workflow rules

- **Docker-first, smoke-first.** Validate `run_smoke.py` (fake logits) before
  committing GPU time; mirror the project's smoke discipline.
- **Split hygiene (critical).** proposal / certification / test folds must stay
  disjoint. **Never** use test labels in proposal or certification. The selector
  must be chosen on the proposal fold only (independent of certification labels).
- **Risk buffer.** Select with `prop_risk <= gamma*alpha`, `gamma in {0.5,0.7,1.0}`.
  Do not conclude `gamma=1.0` suffices without checking the buffer matters.
- **Report results in the fixed format** (see §8 of the project response style:
  진단 요약 / 확인한 명령 / 핵심 결과 / 판정 / 다음 행동).
- **Central question this project exists to answer:** *Is certified accepted
  coverage non-trivial at CIFAR scale under non-IID corruption?*

## 7. Do NOT

- Overclaim from the synthetic smoke; it validates wiring, not science.
- Judge success by accuracy or AUROC — the object is `cert_*` risk/coverage.
- Mix or leak the proposal/certification/test splits.
- Promote the pooled certificate (Thm 3) over the stratified one (Thm 1) — Thm 1
  is the unconditional main result; Thm 3 is the matched-λ tightening.
- Reintroduce SRCC, RC-OWPL, or pseudo-labeling as the main object.
- Hide failed commands. Report them.

## 8. Language

한국어로 토론·진단·계획. 코드·파일명·변수명·config key·commit message는 영어.
최종 deliverable(paper text, abstracts, methodology)도 영어.
