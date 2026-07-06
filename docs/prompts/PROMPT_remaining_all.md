# Claude Code prompt — remaining work to submission (single chained run)

Paste the fenced block into Claude Code on the 4070. It runs the rest of the
campaign autonomously, with a seed GATE that guards the expensive stages. Report
after every STAGE in the fixed format; stop and surface blockers.

```text
CONTEXT (read CLAUDE.md + AGENTS.md first). Done: theory (Thm 1/1', Thm 2, Lemma L
resolved, Prop 4), CPU experiments, CIFAR ladder, T4 superiority, P1 aggregate,
P2 core figs (F5/F6/F7), and a seed0 headline positive — ResNet-18 (CIFAR stem +
GroupNorm) at d=5 lets the worst-group (G=2) certificate cross alpha=0.10, but only
seed0 and only with cert_frac>=0.33 (a TWO-lever Theorem-2 effect: lower r_hat AND
larger per-group calibration). Goal: finish a submission-ready result section.
Docker-first, smoke-first. Never use test labels in proposal/cert. Judge by
cert_*/coverage-at-matched-risk, not AUROC. Report failed commands. Do NOT fake any
certification or crossing. Headline metric: CertifiedCoverage@alpha.

============================================================
STAGE 0 — SEED GATE (do first; decides the headline and what follows)
============================================================
Re-run cifar10 d=5 clean, ResNet-18 (CIFAR stem, GroupNorm), seeds {1,2} with the
EXACT config that gave the seed0 crossover (record rounds, cert_frac, box-Lambda,
proxy margin, gamma grid). Ensure cert/test raw-logit npz is exported for all of
{0,1,2} (seed0 clean npz was missing — fix). Run exp_feasibility_lever at alpha=0.10
(grouped G in {5,3,2,1}, box-Lambda, certify_best_gamma + proxy margin) at the
crossover cert_frac. Aggregate across {0,1,2}: mean+/-std of cert_risk_ucb and
CertCov@0.1 for G=2 and G=3.
  GATE:
   - PASS  = G=2 (or at least G=3) has cert_ucb<=0.10 and CertCov@0.1>0 in >=2/3
     seeds. Headline = "worst-group alpha=0.1 certified at CIFAR scale (ResNet +
     cert_frac>=0.33): CertCov = X% +/- Y%". Proceed to STAGE 3 with ResNet primary.
   - FAIL  = does not replicate. Do NOT chase alpha=0.1. Reframe headline to the
     feasibility LAW + the alpha=0.20 worst-group positive (already solid); run
     STAGE 3 with SimpleCNN as primary and report alpha=0.1 as feasibility-edge.
  Always report the crossover cert_frac as part of the claim (not a hidden knob).

============================================================
STAGE 1 — P2 residual CPU assets (parallel with STAGE 0; cheap, always)
============================================================
matplotlib mathtext-safe (\leq, \mathrm; NOT \le, \rm), colorblind-safe, PDF+PNG in
figs/, CSV in runs/:
  F2 necessity-real (naive vs Fed-CORE unsafe-deploy rate; highlight cells naive>delta).
  T1 main results (from runs/agg_main.csv, mean+/-std; SimpleCNN + ResNet rows).
  T2 efficiency (conditional vs mass-ratio vs box vs pooled; median U, valid?).
  T3 necessity ({naive-emp, fed-CP recast, pooled-CP, Fed-CORE} x unsafe-rate, valid?).
  T4 superiority (extend runs/T4.csv with seeds, mean+/-std).
  T5 score-agnostic (4 scores x test_risk, CertCov; all valid).
  T6 privacy taxonomy (pooled/stratified/grouped x released-stats, leakage, scope).
  T7 self-train delta/T validity (simultaneous unsafe rate with vs without delta/T).

============================================================
STAGE 2 — P3 breadth dataset (always; prevents "CIFAR-only" desk-reject)
============================================================
Add ONE non-CIFAR benchmark through the SAME pipeline (open_set_split +
dirichlet_partition + build_calibration + certify): TinyImageNet OSR split, OR a
tabular/medical FL benchmark (cheaper on CPU, fits the safety-sensitive narrative).
Report the full metric schema; add its rows to T1.

============================================================
STAGE 3 — main ladder (primary backbone per STAGE 0 gate)
============================================================
Run clean / symmetric-0.35 / asymmetric-0.20  x  d in {0.1,0.5,5}  x seeds {0,1,2},
with the primary backbone (ResNet if GATE=PASS, else SimpleCNN). Use
certify_best_gamma + box-Lambda; export npz; aggregate to runs/agg_main.csv. This
fills T1 (the headline table) and regenerates the staircase F6 and T4 with the
primary backbone. Time-box: if GPU is tight, prioritize d in {5,0.5} clean+sym0.35
first, asym and d=0.1 second.

============================================================
STAGE 4 — strengthen (P4 ablations + P5 self-training real)
============================================================
P4 (mostly post-hoc/CPU):
  H2 split-leakage : reuse test labels in proposal/cert -> show guarantee breaks
                     (empirical risk > alpha beyond delta). Proves split hygiene.
  H5 client-subsample : vary #participating clients -> CertCov + feasibility.
  H6 unknown-prop  : sweep fraction of unknown-labeled audit points.
  corruption curve : sym/asym rate in {0,0.1,0.2,0.35,0.5} x d in {0.5,5} ->
                     CertCov@alpha vs noise rate.
P5 self-training real (feasible regime, primary backbone, d=5):
  run_selftrain_cifar, T rounds with round-wise disjoint audit folds (Prop 4).
  Plot F8 = per-round contamination (certified<=alpha vs naive unbounded) +
  downstream accuracy (certified vs naive vs none). HONEST headline: "prevents
  catastrophic contamination while retaining useful pseudo-labels" — NOT a
  guaranteed accuracy gain.

============================================================
STAGE 5 — assemble
============================================================
Regenerate ALL figs (F0,F1 already exist; F2,F5,F6,F7,F8) and tables (T1-T7) with
seeds, consistent style. Update README; after draft sync, update Fed-CORE_draft.md
Section 5 with the seed-aggregated tables/figures and the headline per the GATE.

REPORTING: after each STAGE -> 진단 요약 / 확인한 명령 / 핵심 결과 (mean+/-std) /
판정 (strong/moderate/warning/fail) / 다음 행동. After STAGE 0, state GATE=PASS/FAIL
explicitly. Stop and ask if GPU budget will be exceeded before STAGE 3/4.

RULES: proposal/cert/test disjoint; grouping public/fixed; never use test labels in
proposal/cert (except the clearly-labeled oracle/leakage rows); smoke-first;
report all failures; no faked crossings; G=1 pooled stays a near-IID-only bonus.
```

---

### Notes for Sanghoon
- One paste runs the whole remainder. **STAGE 0 is the gate**: it decides whether
  the headline is "worst-group α=0.1 certified" (PASS) or "feasibility law + α=0.20"
  (FAIL) before any expensive ladder GPU is spent.
- STAGES 1–2 (CPU paper assets + one breadth dataset) run regardless and are the
  highest value-per-cost — they make the paper exist and dodge desk-reject.
- If you want to cap GPU, the prompt tells Claude Code to time-box STAGE 3 and to
  stop-and-ask before exceeding budget.
