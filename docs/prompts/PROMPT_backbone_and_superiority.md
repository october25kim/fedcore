# Claude Code prompt — backbone push (worst-group α=0.1) + C3 superiority (matched-risk vs SOTA)

Paste the fenced block into Claude Code on the 4070. Two experiments: (1) push the
*worst-group* certificate below α=0.1 by lowering realized risk r̂ via a stronger
backbone (Theorem-2 lever, since the sample requirement scales as (α−r̂)⁻²);
(2) build the C3 superiority table T4 — Fed-CORE's *certified* coverage vs
oracle-tuned (test-peeking) FedOSR baselines and an unsafe naive threshold.

```text
CONTEXT (read CLAUDE.md + AGENTS.md first). Status: the α=0.1 null is a
Theorem-2 feasibility collapse; the γ lever was refuted (starves per-group counts);
the staircase shows worst-group G=2 sits at cert_ucb≈0.153 (~0.05 above α=0.1) with
test_risk≈0.055. The two real levers are (i) lower realized risk r̂ via a stronger
backbone — high leverage because required accepted count ∝ z²r̂(1−r̂)/(α−r̂)² — and
(ii) more per-group calibration. This prompt does (1) backbone + (2) the C3
superiority comparison. Docker-first; reuse the existing experiments/fedcore package.

============================================================
EXPERIMENT 1 — backbone push for worst-group CertCov@0.1 (d=5)
============================================================
GOAL: get the WORST-GROUP (grouped-stratified, G∈{2,3}) certificate below α=0.10
with non-trivial coverage, by lowering r̂.

DO:
1. models.py: add ResNet-18 (CIFAR variant: 3x3 stem, no maxpool) as an
   alternative backbone; keep SimpleCNN. Add --backbone {simplecnn,resnet18}.
   Optionally add --pretrained (torchvision ResNet-18 features) as a second arm.
2. run_cifar.py: support --backbone, longer --rounds (e.g. 100-200), and stronger
   local training; keep everything else (open-set split, calibration, certify)
   identical. Export logits npz as before.
3. Re-run cifar10 d=5 clean with: {SimpleCNN (baseline), ResNet-18, ResNet-18
   +more rounds, (optional) pretrained}. For each, report realized r̂ (test_risk
   at the chosen operating point) and then run exp_feasibility_lever post-hoc:
   the grouped staircase G∈{5,3,2,1} at α=0.10, box-Λ, certify_best_gamma with the
   proxy safety margin.
MEASURE / PLOT: for each backbone, (a) r̂ (test_risk), (b) per-group cert_n,
   (c) cert_risk_ucb and CertCov@0.1 for G=2,3,5,1; (d) overlay the required count
   z²r̂(1−r̂)/(α−r̂)² to show how lowering r̂ shrinks the requirement.
EXPECTED / REPORT honestly:
   - Hypothesis: ResNet-18 lowers r̂ (e.g. 0.055 → ~0.02-0.03), shrinking the
     required accepted count several-fold, so worst-group G=2/3 crosses α=0.10 with
     CertCov>0. If it crosses → that is the headline worst-group positive.
   - If it still does not cross at G=2, report the exact (available vs required)
     per-group accepted count and state the remaining gap; combine with larger
     calibration (cert_frac, fewer groups) and report the combination that works.
   - Do NOT fake certification; G=1 pooled stays a near-IID-only bonus.

============================================================
EXPERIMENT 2 — C3 superiority table T4 (matched-risk vs SOTA)
============================================================
GOAL: show Fed-CORE's CERTIFIED coverage is close to what an oracle-tuned
(test-peeking) FedOSR baseline achieves at the same realized risk — while being the
only method with a finite-sample guarantee — and that a no-peek naive threshold is
unsafe.

BASE MODEL FAMILIES (scores to plug into the SAME certify path; these are base
models, NOT competing certifiers):
   - FedAvg + standard scores: MSP, entropy, margin, energy (already implemented).
   - FOOGD score-norm: implement the feature-space score-norm detector as a score
     function on the trained model (reuse the exported features/logits).
   - (Stretch, optional) FedPD / FedOSS training recipes; if time-boxed out, state
     clearly that they are deferred and use the score families above as the base
     models. Do not claim to have run methods you did not run.

PROTOCOL (per base model + d level, cifar10 d∈{5,0.5}):
   For a grid of operating thresholds t on the held-out test fold:
   (A) ORACLE-TUNED baseline (peeks at TEST labels — a cheat used only to define an
       upper bound): pick t so realized test risk = α; record its coverage. Mark
       "guarantee = NO (used test labels)".
   (B) NAIVE no-peek: pick t by empirical risk ≤ α on the PROPOSAL fold (no
       certificate); record realized TEST risk (may exceed α → unsafe) and coverage.
   (C) Fed-CORE: certify_best_gamma (box-Λ, proxy margin) on proposal→cert folds;
       record CertifiedCoverage@α, realized test risk, certified flag.
MEASURE -> build T4 (one row per base-model×method):
   columns = [base model, method, coverage, realized test_risk, risk≤α?,
              finite-sample guarantee?]. 
EXPECTED / REPORT:
   - Fed-CORE (C) coverage ≈ oracle (A) coverage, both with realized risk ≤ α, but
     only Fed-CORE has the guarantee; naive (B) either exceeds α (unsafe) or
     under-covers. Report the coverage gap Fed-CORE→oracle (the "price of honesty").
   - Also report price-of-federation: Fed-CORE vs a centralized oracle (pool all
     calibration as one i.i.d. set) as d→large.

VALIDITY RULES (both experiments): proposal/cert/test folds disjoint; the oracle
arm's test-peeking is used ONLY to compute the upper-bound row and is clearly
labeled; Fed-CORE never sees test labels; grouping is public/fixed; judge by
cert_*/coverage-at-matched-risk, never raw AUROC. Docker-first, smoke-first; report
failed commands; use the fixed format 진단/명령/핵심결과/판정/다음행동.

DELIVERABLES: --backbone in models.py/run_cifar.py; FOOGD score-norm in scores.py;
T4 builder (exp_superiority.py extension) writing T4.csv; README blocks for both;
figures: backbone staircase (CertCov@0.1 vs per-group n, per backbone) and T4 bar
(coverage at matched risk, guarantee annotated). After draft sync, update
Fed-CORE_draft.md §5 (S-a/S-b/S-c) and the headline if worst-group α=0.1 crosses.
```

---

### Notes for Sanghoon
- **Experiment 1 is the higher-value one**: a worst-group ($G\ge2$) CertCov@0.1 > 0
  would be the paper's headline positive and removes the "only pooled/near-IID works"
  caveat. The backbone lever is principled (r̂ enters as $(\alpha-\hat r)^{-2}$).
- **Experiment 2 (T4)** is what turns "we're valid" into "we're valid *and* cost
  almost nothing in coverage vs a cheating oracle" — the most persuasive superiority
  evidence for reviewers.
- FedPD/FedOSS full training is marked optional/stretch so the run is not blocked;
  the score-plug-in framing keeps it honest and feasible.
- Figs F0 (problem diagram) and F1 (pooling collapse) are already generated on the
  Mac side under `experiments/fedcore/figs/`; regenerate or copy as needed.
