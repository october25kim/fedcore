# Two Claude Code prompts (Ubuntu 4070)

Paste each fenced block separately. Both assume `CLAUDE.md` + `AGENTS.md` are at the
repo root and `experiments/fedcore/` already exists and passes its acceptance gate
(see `PROMPT_for_claude_code_4070.md`). Run the smoke + 3 CPU experiments first.

---

## Prompt (i) — Certified federated self-training (Proposition 4)

```text
GOAL: implement and validate CERTIFIED federated self-training (Fed-CORE
Proposition 4) plus the baselines it must beat. First read CLAUDE.md + AGENTS.md
and confirm experiments/fedcore/ passes its acceptance gate.

WHAT PROPOSITION 4 SAYS (the contract you must honor):
Self-training makes the round-t model f_t depend on what was accepted at round
t-1, so a certification fold reused across rounds is NOT independent of (f_t, A_t)
and the certificate would be invalid. Fix it by DATA-SPLITTING IN TIME: split the
trusted set into T disjoint audit folds C^(1..T); certify round t ONLY on the
fresh fold C^(t) at level delta/T. Then
   P( for all t<=T : R_sel(A_t) <= Ubar^(t) ) >= 1 - delta,
so every injected pseudo-label batch has certified contamination <= alpha
simultaneously across all rounds. This is the whole point — do not violate it.

PROTOCOL (implement exactly):
- Data per client: (1) LABELED known-class train data (non-IID, Dirichlet);
  (2) an UNLABELED pool U (known + unknown classes, no labels) — the pseudo-label
  source; (3) a TRUSTED clean calibration set split into a FIXED proposal pool P
  and T DISJOINT certification folds C^(1..T) (plus a held-out test fold).
- Round t = 1..T:
    1. Train FedAvg -> f_t on (labeled data + currently-accepted pseudo-labels).
    2. Choose the selector A_t (risk-buffered threshold, gamma*alpha) on P using
       f_t.  [P may be reused across rounds; it is part of the algorithm.]
    3. Certify A_t on the FRESH fold C^(t) at level delta/T via the conditional
       certificate (Thm 1/1'); record cert_risk_ucb^(t), cert_coverage_lcb^(t).
    4. Pseudo-label U with f_t; ACCEPT the subset with A_t=1 (predicted known
       class). Recompute acceptance each round (labels may change as f improves).
    5. Next round's training set = labeled data + accepted pseudo-labels.
- INVARIANT TO ASSERT IN CODE: C^(t) is never used for training or for selecting
  A_t. f_t and A_t are formed only from labeled data, P, U, and folds C^(<t).
  Add an assertion that the index sets of {C^(t)} are pairwise disjoint and
  disjoint from P and from any pseudo-label indices.
- ASSUMPTION TO STATE: U and the calibration folds share the deployment mixture
  Q_lambda, so the certified accepted-risk bound transfers to the contamination
  of accepted pseudo-labels drawn from U.

BASELINES TO COMPARE:
- naive self-training: pick the threshold by a heuristic (fixed confidence 0.95,
  AND/OR empirical accepted risk <= alpha on P with NO buffer and NO certificate)
  and self-train identically. No per-round certificate.
- no self-training: train once on labeled data only.

FILES TO CREATE under experiments/fedcore/:
  selftrain.py            audit-fold partition + the certified/naive loops
                          (torch for training; reuse certify.py/selector.py)
  run_selftrain_smoke.py  CPU, NO torch: validate Proposition 4 itself on fake
                          per-round logits (no real model). Over many trials with
                          a FIXED sequence of selectors whose TRUE per-round
                          R_sel = alpha, estimate the SIMULTANEOUS unsafe rate
                          P(exists t: R_sel(A_t) > Ubar^(t)). Show it is <= delta
                          WITH the delta/T split, and that using delta each round
                          (no /T) inflates it ABOVE delta. This justifies the split.
  run_selftrain_cifar.py  torch, 4070: the real CIFAR-10/100 self-training run
                          (certified vs naive vs none), reusing fedosr_split.py,
                          models.py, fed_train.py, noise.py.

METRICS per round: downstream test accuracy (known-class acc + open-set
acc/coverage), REALIZED pseudo-label contamination on U (ground truth is
available in the experiment), cert_risk_ucb^(t), cert_coverage_lcb^(t),
#pseudo-labels accepted, T_used. Headline plots: (a) realized contamination vs
round — certified stays <= alpha, naive grows; (b) downstream accuracy vs round —
certified improves and beats naive (which degrades/diverges) and no-self-training,
especially at low Dirichlet alpha with symmetric/asymmetric label corruption.

ACCEPTANCE GATE:
- run_selftrain_smoke.py: simultaneous unsafe rate <= delta with delta/T; > delta
  without the split. Print both.
- run_selftrain_cifar.py (smoke-size first: few rounds, few epochs): runs
  end-to-end; certified per-round realized contamination <= cert_risk_ucb^(t)
  in (close to) all rounds; naive contamination exceeds alpha under corruption.

THEORY/FEASIBILITY NOTE TO RESPECT: T is bounded by the trusted-set size via the
Theorem 2 per-fold threshold A_j^(t) >= ln(J/delta')/(-ln(1-alpha)) with
delta'=delta/T. Report when a fold is too small to certify (infeasible round) and
stop adding rounds rather than faking a certificate.

RULES: Docker-first, smoke-first. Never use C^(t) or test labels for training or
selection. Judge by cert_* / contamination / downstream accuracy, not AUROC.
Report each run as 진단 요약 / 확인한 명령 / 핵심 결과 / 판정 / 다음 행동.
After it works, update Fed-CORE_draft.md §4.7 and §5(B) with the real numbers.
```

---

## Prompt (ii) — Formal proof of Lemma L

```text
GOAL: settle Lemma L (used by Fed-CORE Proposition 3) — either a complete formal
proof with a precise citation, a proof-under-a-stated-condition plus a PROVABLE
conservative correction, or (worst case) a clearly scoped conjecture with an
adversarial numerical certificate. Read CLAUDE.md + AGENTS.md first. Be rigorous
and intellectually honest; do NOT claim a clean proof you cannot defend.

PRECISE STATEMENT TO PROVE.
Let Z_1,...,Z_A be independent, Z_i ~ Bernoulli(r_i), S = sum_i Z_i, and
rbar = (1/A) sum_i r_i. Let U+(k, A; delta) be the one-sided binomial
Clopper-Pearson upper limit (the largest p with P(Bin(A,p) <= k) >= delta).
LEMMA L: P( rbar <= U+(S, A; delta) ) >= 1 - delta.
(That is: the binomial CP upper limit, applied to a Poisson-binomial count, is a
valid 1-delta upper confidence bound for its mean.)

REQUIRED REDUCTION (do this explicitly).
Show that Lemma L is equivalent to a POINTWISE LOWER-TAIL DOMINATION:
   for every integer b <= mu := A*rbar,  P_PB(S <= b) <= P_{Bin(A, rbar)}(S <= b).
Derive the threshold k_delta = max{ k : P_{Bin(A,rbar)}(X <= k) < delta } and show
U+(S,A;delta) < rbar  <=>  S <= k_delta, with k_delta <= mu, so validity follows
from the domination at b = k_delta. State this reduction cleanly.

PROOF ROUTES — try in this order, report what actually works.
1. LITERATURE: find the exact theorem establishing the pointwise lower-tail
   domination of a Poisson-binomial by the equal-mean binomial. Candidates to
   check (verify the precise statement/conditions, do not cite from memory):
   Hoeffding (1956, Ann. Math. Statist.); Gleser (1975); Anderson & Samuels
   (1967); Boland & Proschan and Schur-convexity results for PB tail
   probabilities; Darroch (1964) on the mode. If a theorem gives exactly the
   pointwise domination for b <= mu, cite it precisely and you are done.
2. SCHUR-CONCAVITY PROOF: prove P_PB(S <= b) is Schur-concave in (r_1,...,r_A)
   for b <= mu, so the equal-coordinate vector (binomial) maximizes it. Use the
   standard PB convolution recursion and the two-coordinate transfer
   (r_i, r_j) -> (r_i+eps, r_j-eps): show the sign of the derivative of
   P(S <= b) along this transfer is <= 0 for b <= mu. This is the cleanest
   self-contained route; carry it out carefully (the key is a telescoping of
   P(S=k) terms when perturbing two coordinates).
3. CONVEX-ORDER PARTIAL RESULT (state its limit): PB <=_cx Bin(A, rbar) gives,
   via E[(b-S)_+] = sum_{j<b} F_S(j), the INTEGRATED domination
   sum_{j<b} F_PB(j) <= sum_{j<b} F_Bin(j) for all b. Note this is second-order
   (integrated) and does NOT by itself give the pointwise bound; explain why, so
   the reader sees the gap that routes 1-2 must close.
4. IF POINTWISE FAILS UNCONDITIONALLY: exhibit a counterexample, and instead
   prove a CONSERVATIVE CORRECTION that holds unconditionally — e.g. use a
   Bernstein/Bennett bound with the PB variance sum_i r_i(1-r_i) <= A*rbar(1-rbar)
   (so the binomial variance dominates), giving a slightly inflated but PROVABLE
   upper bound; quantify the inflation vs the exact CP limit.

NUMERICAL CERTIFICATE (adversarial). Extend experiments/fedcore/exp_lemma_L.py to
SEARCH for a violation: fine grid / random search over r-vectors (especially
two-point and three-point configs, varying A and rbar, including small A and
rbar near common alpha values), reporting the minimum empirical coverage and any
config with coverage < 1 - delta beyond Monte-Carlo error. Current evidence:
worst-case coverage ~0.919 >= 0.90 at delta=0.1; try to break it harder.

DELIVERABLE. Write the result into a self-contained proof note
(experiments/fedcore/LEMMA_L_proof.md or a new appendix) containing: the precise
statement, the reduction, the chosen proof (with exact citation or full
Schur-concavity argument), explicit conditions if any, the conservative variant
if needed, and the adversarial numerical evidence. Then update Fed-CORE_draft.md
§4.5 / Proposition 3: if Lemma L is proved, promote the contamination/pooled
result accordingly (still keep it subordinate to Thm 1/1' until BOTH gaps —
Lemma L AND the roster-composition coupling, Gap 2 — are closed); if only a
conditional/conservative result holds, state exactly that. Be explicit about what
is proved vs assumed vs numerically supported.
```

---

### Notes for Sanghoon (not part of the prompts)

- Prompt (i) is the only genuinely NEW code (Proposition 4 self-training loop);
  the rest reuses the validated package. Its CPU smoke (`run_selftrain_smoke.py`)
  is the key correctness check — it validates Proposition 4 itself, independent of
  any real model.
- Prompt (ii) closes "Gap 1" only. **Gap 2 (roster-composition coupling) remains
  separate**; Proposition 3 should stay subordinate to Theorems 1/1' until both
  are closed. The prompt says this explicitly so Claude Code won't over-promote.
