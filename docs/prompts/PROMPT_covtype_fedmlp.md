# Claude Code prompt — covtype FedAvg-MLP (CPU): try for an HONEST 2nd-domain positive

Context. The covtype "0.43" headline was a **seed-0 + selection-optimistic artifact**;
under the fixed score (MSP) with worst-group `G=2` it certifies **0/5** over five seeds,
and only under a best-of-scores rule reaches `0.10±0.17` at `alpha=0.20` (2/5). The draft
now reports covtype honestly as a feasibility-edge second domain, **not** a stable positive.

The limiting factor is the base model's accepted-risk `r_hat ≈ 0.14`, which sits close to
`alpha=0.20`, so the `(alpha - r_hat)^-2` sample requirement explodes and certification is
fragile. A **stronger federated model lowers `r_hat`** and could earn a *genuine*
multi-seed positive. This is a CPU job (no GPU budget). If it still fails at the fixed
score, report that — do **not** switch to selection-optimistic scoring to manufacture a
positive.

Paste the fenced block into Claude Code in the FedCORE repo.

```text
READ CLAUDE.md AND AGENTS.md FIRST. This is Fed-CORE (federated certified open-set
recognition). Object = certified accepted selective risk; metrics = cert_risk_ucb,
cert_coverage_lcb, test_risk, test_coverage. NEVER use test labels in proposal/cert.
Judge by cert_* and matched-risk, not accuracy. Report failures. CPU only.

GOAL. Earn an HONEST worst-group covtype certified-coverage positive at alpha=0.20 over
5 seeds with the FIXED score (MSP), by replacing the current covtype base model with a
stronger FedAvg-trained MLP that lowers the accepted-risk r_hat below alpha.

STEP 1 — base model. In experiments/fedcore/, build (or extend run_tabular.py with) a
federated MLP for covtype:
  - data: covtype (7 classes). Open-set split: hold out 2 classes as `unknown`
    (same protocol as the existing covtype run / fedosr_split.py); standardize features.
  - clients: J as in the current covtype config; Dirichlet non-IID over known classes
    (reuse the existing dirichlet_alpha used for covtype, e.g. 0.5 and 0.1).
  - model: MLP, 2–3 hidden layers (e.g. 256-128-64), ReLU, LayerNorm or GroupNorm
    (NOT BatchNorm — FL backbone rule), dropout optional.
  - training: FedAvg, local epochs 1–2, enough rounds to converge (CPU-cheap on covtype);
    export per-point logits for proposal/certification/test folds via the existing
    fold machinery (fedosr_split.py). Do not leak fold labels across splits.

STEP 2 — certify (reuse the existing path). Run the certificate (certify.py /
certificates.py) on the exported logits with:
  - score = MSP (FIXED; this is the honest protocol). Optionally also record energy/margin
    for reference, but the headline number is fixed-MSP.
  - worst-group G=2, cert_frac=0.5, delta=0.10, gamma in {0.5,0.7,1.0}.
  - seeds {0,1,2,3,4}; alpha in {0.20,0.25,0.30}.

STEP 3 — report (fixed format). For fixed-MSP, worst-group G=2, per alpha:
  - CertifiedCoverage@alpha mean±std and n_pass/5
  - median cert_risk_ucb among certified seeds; test_risk; r_hat (accepted empirical risk)
  - the same for d (dirichlet) variants you ran
Save runs/covtype_fedmlp_main.csv (per seed×alpha×gamma rows with the canonical schema:
certified, cert_risk_ucb, cert_coverage_lcb, cert_n, cert_k, prop_coverage, prop_risk,
test_coverage, test_risk, score_name, gamma, alpha, delta, dirichlet_alpha, n_clients)
and runs/covtype_fedmlp_agg.csv (aggregated mean±std, n_pass/5).

DECISION RULE (honest).
  - strong go: fixed-MSP CertCov@0.20 ≥ ~0.15 with n_pass ≥ 4/5 → covtype becomes a real
    2nd-domain positive; sync the draft headline/Table 1.
  - moderate go: n_pass 2–3/5 with positive mean → still seed-variable; keep current
    honest "feasibility-edge" framing but cite the lower fixed-MSP r_hat as improvement.
  - fail: still 0/5 at fixed MSP → report it plainly; the paper stands on CIFAR + the
    feasibility law, covtype stays a feasibility-edge corroboration. Do NOT switch to
    selection-optimistic scoring to produce a positive.

Stop-and-ask before any large compute. Output exact numbers so the Mac-side
Fed-CORE_draft.md (Abstract / Table 1 / §5 headline / Conclusion) can be synced.
```

---

### Notes for Sanghoon
- 이 작업의 **유일한 목적은 정직한 r̂ 인하**입니다. fixed-MSP/G=2에서 통과해야 진짜
  positive입니다. selection-optimistic로 다시 0.43을 만들면 안 됩니다.
- 결과가 어떻든 draft는 이미 정직하게 정리돼 있으므로, **성공 시에만** headline을
  "두 도메인 positive"로 복구합니다(이때도 fixed-MSP 기준 숫자로).
- CPU 작업이라 4070 GPU 큐와 독립적으로 돌릴 수 있습니다.
