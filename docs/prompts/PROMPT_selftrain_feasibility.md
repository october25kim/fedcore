# Claude Code prompt — certified self-training under feasibility interventions

Goal. Move the self-training use case from "contamination gate only" toward
**contamination-certified performance gain when feasibility allows**. The current round-wise
result halts almost immediately (audit fold split into T, base model's $\hat r$ near $\alpha$,
certified set = easiest samples), so accuracy is only comparable. This package tests whether,
with (i) a stronger base detector, (ii) a one-shot (non-split) certification, (iii) a larger
audit budget, and (iv) a tuned pseudo-label loss weight, certified self-training yields a real
accuracy gain while keeping per-batch contamination certified $\le\alpha$.

**Honesty first.** Never fabricate a row. Selection of the selector/threshold must use the
PROPOSAL fold only (never adapt to the certification fold). Report negatives plainly — every
outcome is publishable (see "interpretation" at the end). Docker-first; the fixed report format.

Paste the fenced block into Claude Code in the FedCORE repo.

```text
READ CLAUDE.md AND AGENTS.md FIRST. Object = certified accepted selective risk; the self-training
guarantee is per-round pseudo-label contamination <= alpha (Prop. 4), NOT accuracy. Split hygiene:
proposal/certification/test disjoint; selector chosen on the proposal fold; pseudo-label admission
certified on a fold independent of the model that produced the labels. Judge by cert_* and the
metrics below, report halts and failures.

PRIORITY PACKAGE (run in this order; stop-and-ask before large GPU spend).

P1 — One-shot certified self-training (remove the T-way fold split).
  Pipeline: (1) train base model; (2) pick selector on proposal fold; (3) certify the accepted set
  on the FULL certification fold at level delta (one shot, no delta/T); (4) extract certified
  accepted pseudo-labels from the unlabeled pool; (5) fine-tune ONCE on supervised + admitted
  pseudo-labels; (6) evaluate. Compare modes {none, naive, certified one-shot} plus an
  oracle-clean-pseudo-label upper bound. Hypothesis: without the fold split, more pseudo-labels are
  admissible, so a gain becomes possible.

P2 — FedPD-PROSER as the self-training base model (strongest detector).
  Repeat P1 with base in {FedAvg+MSP, FedPD-PROSER}. FedPD-PROSER (native AUROC ~0.80, CertCov@0.20
  ~0.48 multi-seed, certifies even at alpha=0.10) should yield a larger, lower-risk accepted set,
  hence more admitted pseudo-labels. This is the most likely route to a real gain.

P3 — Audit-budget sweep for self-training.
  Multiply the certification-fold size by {1x, 2x, 4x} (subsample the available trusted pool;
  hold the model fixed per setting). At each budget log: admitted pseudo-label count, contamination,
  halt frequency, final accuracy, CertifiedCoverage@alpha. Question: does a larger audit budget turn
  certified self-training into an actual accuracy gain (the feasibility law applied to self-training)?

P4 — Pseudo-label loss-weight sweep.
  Loss = L_sup + beta * L_pseudo, beta in {0.1, 0.25, 0.5, 1.0} (optionally confidence-weighted
  w(x)*L_pseudo with w = score margin). Find the beta band where admitted pseudo-labels actually
  raise accuracy; confirm over-large beta hurts. Keep contamination <= alpha throughout.

GRID (keep it bounded). dataset = CIFAR-10; alpha in {0.10, 0.20}; base in {FedAvg+MSP,
FedPD-PROSER}; mode in {none, naive, certified one-shot, certified round-wise}; audit budget in
{1x, 2x, 4x}; beta in {0.1, 0.25, 0.5, 1.0}. Start with the smallest informative slice
(FedPD-PROSER + certified one-shot + alpha=0.20 + beta=0.25), then expand only where it pays.

METRICS (log per run to runs/selftrain_pkg.csv).
  pseudo-label contamination (realized error rate of admitted batch),
  admitted pseudo-label count, halt frequency / round,
  final known accuracy, final balanced accuracy (class-balanced),
  final accepted selective risk (test), final CertifiedCoverage@alpha,
  plus base_model, alpha, mode, audit_mult, beta, seed.

OPTIONAL STRETCH (2nd priority; only if P1-P4 leave headroom).
  (a) class-balanced certified pseudo-label selection — apply a class quota; the selection RULE must
      be fixed on the proposal fold or pre-defined, NOT adapted to the certification fold (else
      validity breaks). Report balanced/tail accuracy.
  (b) alpha_train curriculum — admit at alpha_train in {0.20 -> 0.15 -> 0.10} across rounds, but
      RE-CERTIFY final deployment separately at the target alpha_eval. State explicitly:
      "training batches are certified at their own training risk budgets; final deployment is
      re-certified at the target alpha." Do NOT claim alpha=0.10 contamination for alpha_train=0.20
      batches.
  (c) utility-aware selector — on the proposal fold, maximize a utility-weighted coverage (class /
      client balance, informativeness) s.t. proxy risk <= gamma*alpha; compare gain vs coverage-max.

OUTPUT. runs/selftrain_pkg.csv (+ an aggregate), and an updated F8-style figure only if a positive
gain appears (else keep the current admission/halt Figure 6). A short REPORT_selftrain_pkg.md.

REPORT (fixed format) per priority: 진단 요약 / 확인한 명령 / 핵심 결과 (base, alpha, mode,
audit_mult, beta, contamination, admitted_count, halt_freq, known_acc, balanced_acc, test_risk,
CertCov@alpha) / 판정 (strong/moderate/warning/fail) / 다음 행동. Output exact numbers so the
Mac-side Section 5.6 can be updated.

INTERPRETATION (every outcome is publishable — frame honestly):
  - gain only with larger audit budget  -> "safe pseudo-labels become useful once feasibility is met."
  - gain only with FedPD-PROSER          -> "certified self-training needs a sufficiently strong base detector."
  - gain only at alpha=0.20              -> "smaller risk budgets stay feasibility-limited (Theorem 2)."
  - gain only with class-balanced/utility -> "coverage-max admission is safe but not utility-optimal."
  - no gain anywhere                     -> "certified self-training is a contamination gate; accuracy gain needs more feasibility budget."
```

---

### Notes for Sanghoon
- 1순위(P1–P4)만으로도 핵심 질문("충분한 budget·강한 detector면 certified gain이 가능한가?")에 답할 수 있습니다.
- 가장 가능성 높은 조합: **FedPD-PROSER + one-shot + alpha=0.20 + beta=0.25**.
- 정직성 가드: selector/selection rule은 proposal fold에서만, certification fold에 adaptive 금지;
  alpha_train curriculum은 final deployment를 target alpha로 별도 re-certify(과대주장 금지).
- 결과가 오면 제가 §5.6을 업데이트하겠습니다 — 양의 gain이면 "contamination-controlled performance
  improvement under sufficient feasibility"로 격상, 음수면 현재 "admission gate" 메시지를 그 조건
  (budget/detector/alpha)으로 더 날카롭게 다듬습니다. KO는 요청 시 동기화.
