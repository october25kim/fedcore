# AGENTS.md — Fed-CORE (for Codex / Claude Code / any coding agent)

> Self-contained agent brief. Mirrors `CLAUDE.md`; if they ever disagree,
> `CLAUDE.md` wins. Read both before acting.

## Project in one paragraph

**Fed-CORE = Federated Certified Open-Set Recognition.** Goal: certify the
**accepted selective risk** of a federated-trained open-set classifier (the
probability an *accepted* prediction is wrong) with a finite-sample,
distribution-free upper confidence bound, under client heterogeneity and unknown
deployment mixture, using only secure-aggregatable counts. It is NOT a
FedOSR-accuracy paper and NOT a noisy-label-training paper. **SRCC/Paper 1 is
abandoned; RC-OWPL / pseudo-labeling are obsolete** — do not reintroduce them as
the main object.

## What must stay true

- Object: `R_sel(lambda) = sum_j lam_j m_j / sum_j lam_j a_j`, `r_j=m_j/a_j`; certify `<= alpha`.
- **Theorem 1/1' (main): CONDITIONAL selective-risk certificate.** `K_j|A_j ~ Bin(A_j,r_j)`
  => `rbar_j = U+(K_j,A_j; delta/J)`. Simplex: `Ubar = max_j rbar_j`. Bounded
  Lambda (recommended): also `a_j in [alow_j,ahigh_j]`, robust linear-fractional
  `sup_{lam in Lambda, a in box} (sum lam a rbar)/(sum lam a)`. TIGHTER than the
  old mass-ratio `max_j mbar_j/alow_j` (App-C baseline only). Handle edge cases:
  zero coverage => non-deployable; vanishing denom bound => infeasible.
- **Why non-trivial:** pooling accepted points across heterogeneous clients is
  invalid (Poisson-binomial, not binomial). Thm 1 ≠ centralized CP and ≠
  federated conformal coverage (Lu et al.).
- **Theorem 2:** per-client feasibility on OBSERVED count `A_j >= ln(J/delta)/(-ln(1-alpha))`.
- **Proposition 3 (subordinate):** pooled, matched-mixture i.i.d. calibration only;
  two open gaps (Lemma L; roster-composition coupling). Do not promote above Thm 1/1'.
- **Privacy:** only POOLED is sum-only secure-agg; STRATIFIED needs per-client
  counts; GROUPED-stratified is the compromise. Calibration must contain LABELED
  unknowns; "distribution-free" is w.r.t. the calibration distribution.
- Metric schema keys (fixed): `certified, cert_risk_ucb, cert_coverage_lcb,
  cert_n, cert_k, prop_coverage, prop_risk, test_coverage, test_risk,
  score_name, gamma, alpha, delta, Lambda, dirichlet_alpha, n_clients`.

## Commands

```bash
# One-time: editable install so `import fedcore` resolves (core hoisted to project-root fedcore/;
# experiments/fedcore/*.py remain backward-compat shims so these commands are unchanged).
pip install -e .
# CPU, no torch
python experiments/fedcore/exp_lemma_L.py
python experiments/fedcore/exp_pooling_fail.py
python experiments/fedcore/run_smoke.py
# GPU (4070), real CIFAR
bash scripts/docker_cifar.sh
python experiments/fedcore/run_cifar.py --dataset cifar10 --n_known 6 \
    --dirichlet_alpha 0.1 --rounds 50 --local_epochs 2 --alpha 0.10 --delta 0.10
```

## Hard rules

1. proposal / certification / test folds disjoint; never use test labels in
   proposal or certification; selector chosen on proposal fold only.
2. Risk buffer `prop_risk <= gamma*alpha`, `gamma in {0.5,0.7,1.0}`.
3. Judge by `cert_*` risk/coverage, never by accuracy/AUROC.
4. Theorem 1 is the unconditional result; do not elevate Theorem 3 above it.
5. Docker-first, smoke-first. Don't hide failed commands.
6. Korean for discussion; English for code, configs, and final text.

## Definition of done for the next milestone

Run `run_cifar.py` for cifar10 (clean, then symmetric-35%, then asymmetric-20%
client-side corruption), seeds 0/1/2, α=0.10, δ=0.10, dirichlet_alpha in
{0.1,0.5,5}. Report `CertifiedCoverage@alpha` per (score, gamma, Lambda).
Central question: is certified accepted coverage non-trivial at CIFAR scale
under non-IID corruption?
