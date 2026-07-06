# Claude Code prompt — certified self-training in the label-scarce regime

Goal. The current self-training gain (FedPD-PROSER, labeled_frac=0.5) is **safe but
statistically weak**: $+0.030$ over three seeds, $2/3$ positive, $95\%$ $t$-CI $[-0.04,+0.10]$,
because (i) the oracle clean upper bound is only $+0.045$ — at half labels the base is already
strong, so pseudo-labels add little headroom — and (ii) one seed is feasibility-limited to zero
admissions, inflating the variance. This package tests the scientifically correct setting for
self-training: **a label-scarce regime, where the oracle headroom is large, combined with a larger
audit budget that removes the zero-admission seeds.** Hypothesis: certified self-training then
yields a **larger, statistically clearer, still-safe** accuracy gain — precisely where labels are
scarce, which is the regime self-training is for.

**Honesty first.** Never fabricate a row. Always report the oracle (clean-pseudo-label) upper bound
alongside the certified gain — the claim is "certified captures a large fraction of the headroom
*safely*", never an accuracy-SOTA claim. Selector/threshold chosen on the PROPOSAL fold only (never
adapted to the certification fold). Keep per-batch contamination certified $\le\alpha$. Report halts
and negatives plainly. Docker-first; the fixed report format.

**Positioning (do not drift).** This stays a **supporting / secondary** result. The primary
self-training result remains the contamination *gate* (Figure 6). Even a large gain here is framed
as "a safe accuracy gain in the label-scarce regime", not as repositioning Fed-CORE into an SSL /
self-training accuracy paper. Do NOT add SSL-SOTA baselines or accuracy-benchmark comparisons.

Paste the fenced block into Claude Code in the FedCORE repo.

```text
READ CLAUDE.md AND AGENTS.md FIRST. Object = certified accepted selective risk; the self-training
guarantee is per-round/per-batch pseudo-label contamination <= alpha (one-shot at level delta),
NOT accuracy. The accuracy gain is a SUPPORTING empirical result, always reported with its oracle
clean upper bound. Split hygiene: proposal/certification/test disjoint; selector on the proposal
fold; pseudo-labels certified on a fold independent of the model that produced them. A5 condition:
the unlabeled pool's unknown rate MUST be matched to the deployment/certification unknown rate
(0.30) — without this the certificate is anti-conservative. Judge by the metrics below; report
halts and failures. Docker-first.

CENTRAL QUESTION. In a label-scarce regime, does (a) the oracle clean-pseudo-label headroom grow
large, and (b) certified self-training capture a large fraction of it SAFELY (contam <= alpha) with
a gain that is statistically separable from zero at n=3?

BASE MODEL. FedPD-PROSER (strongest detector; the one that gains). Standard recipe: closed-set CE
pretrain -> PROSER fine-tune (from scratch does NOT converge). One-shot certification (single delta,
no delta/T split), the P1 setup. beta = 1.0 (best from the prior P4 sweep). dataset = CIFAR-10,
6 known classes, n_clients = 5, dirichlet d = 5 (match the existing self-training cell).

GRID (run in this order; STOP-AND-ASK before each expansion of GPU spend).

STEP 0 — cheapest informative cell (oracle-ceiling probe).
  labeled_frac = 0.10, audit_mult = 4x, alpha = 0.20, beta = 1.0, seed = 0, modes {none, certified,
  oracle}. PURPOSE: confirm the oracle (clean) gain rises clearly above the half-label +0.045 before
  spending more GPU. If the oracle gain at labeled_frac=0.10 is NOT meaningfully larger than +0.045,
  STOP and report — the headroom hypothesis fails and we keep the current supporting result.

STEP 1 — label-scarcity axis (if Step 0 shows a larger oracle ceiling).
  labeled_frac in {0.10, 0.25}; audit_mult = 4x; alpha = 0.20; beta = 1.0; seed = 0;
  modes {none, certified, oracle}. Map how oracle headroom and certified gain scale as labels thin.
  Watch the feasibility tension: fewer labels -> weaker base -> higher r_hat -> certified may admit
  less or halt. A "sweet spot" (intermediate labeled_frac maximizing the certified gain) is itself a
  reportable finding.

STEP 2 — audit-budget rescue (kill the zero-admission seeds).
  At the best labeled_frac from Step 1, sweep audit_mult in {4x, 8x} (subsample the trusted pool;
  hold the model fixed per setting). PURPOSE: a larger audit budget should let every seed admit
  (Theorem 2), raising the mean and SHRINKING the variance so the gain separates from zero.

STEP 3 — multi-seed confirmation (only if Steps 1-2 give a clear single-seed gain).
  Best (labeled_frac, audit_mult) cell, seeds {0, 1, 2} (and {3,4} only if still borderline),
  modes {none, certified, oracle}. Report paired per-seed delta: compute Δ_s = acc_certified(s) −
  acc_none(s) per seed, then mean ± SAMPLE SD (ddof=1) and a 95% t-CI; also Δ_oracle(s) and the
  capture ratio Δ_certified(s)/Δ_oracle(s) per seed. Only claim a gain if the t-CI excludes zero.

METRICS (log per run to runs/selftrain_lowlabel.csv; aggregate with aggregate_selftrain.py so the
convergence guard known_acc<0.40 and seed-aware n_seeds apply).
  labeled_frac, audit_mult, alpha, beta, seed, mode,
  known_acc, balanced_acc, none/certified/oracle gain (paired),
  realized_contam (admitted batch), admitted_count, halted/infeasible, halt_freq,
  test_risk, CertifiedCoverage@alpha, cert_risk_ucb.

OUTPUT.
  runs/selftrain_lowlabel.csv (+ _agg.csv via aggregate_selftrain.py).
  REPORT_selftrain_lowlabel.md (fixed format).
  A figure ONLY if a clear, separable-from-zero positive gain appears: a labeled_frac-axis panel
  (oracle ceiling and certified gain vs labeled_frac) + the safety panel (contam <= alpha). Reuse
  the make_selftrain_gain.py style; SAMPLE SD (ddof=1) error bars to match the manuscript. If no
  clear gain, produce NO new figure — keep the current supporting Figure 9.

REPORT (fixed format) per step:
  진단 요약 / 확인한 명령 / 핵심 결과 (labeled_frac, audit_mult, alpha, beta, mode, oracle_gain,
  certified_gain, paired t-CI, contam, admitted_count, halt_freq, CertCov@alpha) /
  판정 (strong/moderate/warning/fail) / 다음 행동. Output exact numbers so the Mac-side Section 5.6 /
  Figure 9 can be updated.

INTERPRETATION (every outcome is publishable — frame honestly, keep it SUPPORTING):
  - large oracle ceiling + clear safe certified gain (t-CI > 0)
        -> "certified self-training delivers a large, safe accuracy gain precisely where labels are
            scarce — the regime self-training is for — capturing X% of the clean headroom."
  - oracle ceiling grows but certified stays flat / halts (feasibility-limited)
        -> "in the label-scarce regime the headroom is real but the trusted audit budget, not the
            detector, becomes the binding constraint (Theorem 2)." (then Step 2 audit-budget is the story)
  - gain clear only at an intermediate labeled_frac
        -> "the certified self-training gain is maximized in an intermediate label regime: enough
            labels to certify a useful accepted set, few enough that pseudo-labels add real signal."
  - no larger ceiling even at labeled_frac=0.10
        -> "self-training headroom is small on this split regardless of label budget; the certified
            gate, not the gain, remains the contribution." (keep current Figure 9, no change)
```

---

### Notes for Sanghoon
- Step 0(저비용 1셀)가 게이트입니다 — **oracle 천장이 +0.045 위로 분명히 커지는지**부터 확인하고,
  안 커지면 즉시 멈추고 현재 supporting 결과를 유지합니다. GPU 낭비 방지.
- 가장 가능성 높은 성공 경로: labeled_frac≈0.10~0.25에서 천장이 커지고, audit 8x로 zero-admission
  seed를 없애 분산을 줄여 t-CI가 0을 배제하는 그림.
- 정직 가드: oracle UB 항상 동반 보고, contam≤α per batch, selector는 proposal fold에서만,
  paired per-seed Δ + sample SD(ddof=1) + t-CI, 수렴 가드(known_acc<0.4 제외), seed-aware n.
- A5 매칭(U의 unknown rate ↔ deployment/cert 0.30)을 저-라벨에서도 반드시 유지 — 안 하면
  anti-conservative.
- 포지셔닝: 결과가 강해도 **헤드라인은 gate(Figure 6)**, 이건 "label-scarce safe gain"
  supporting입니다. SSL-SOTA 정확도 비교는 추가하지 않습니다(논문 정체성 보호).
- 결과가 오면 제가 §5.6 supporting 문단 + Figure 9를 그 조건(labeled_frac/audit budget)으로
  업데이트하겠습니다. 양성·분명하면 "safe gain where labels are scarce"로 격상, 아니면 현 supporting
  유지. KO는 요청 시 동기화.
