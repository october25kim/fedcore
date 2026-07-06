# Claude Code prompt — consolidation campaign toward submission

Paste the fenced block into Claude Code on the 4070. This is a prioritized campaign
that turns the existing scattered results into a paper-ready result section (tables +
figures with seeds), then fills the gaps a journal will require. Work top-down by
priority; report after each STEP in the fixed format and stop to surface blockers.

```text
CONTEXT (read CLAUDE.md + AGENTS.md first). The method, theory (Thm 1/1', Thm 2,
Lemma L resolved, Prop 4), and most experiments are done; T4 superiority and the
feasibility staircase are in. EXP1 (ResNet backbone) may be finishing. The goal now
is a SUBMISSION-READY result section: seeds + aggregated tables + publication
figures + the breadth/ablation gaps. Docker-first, smoke-first. Never use test
labels in proposal/certification. Judge by cert_*/coverage-at-matched-risk, not
AUROC. Report failed commands. Headline metric: CertifiedCoverage@alpha.

PRIORITY ORDER: P1 (seeds+aggregation) -> P2 (paper figures/tables) -> P3 (breadth
dataset) -> P4 (ablations + corruption curve) -> P5 (self-training real) ->
P6 (conditional ResNet ladder). Do P1+P2 first; they make the paper exist.

============================================================
STEP P1 — seeds + aggregation (MUST; cheap)
============================================================
Run the main CIFAR ladder for seeds {0,1,2} (reuse cached models where possible;
otherwise retrain). For every (dataset, d, corruption, alpha) cell, aggregate
across seeds: report mean +/- std of cert_risk_ucb, CertifiedCoverage@alpha,
test_risk, test_coverage, and the chosen gamma*/G/Lambda. Add an aggregation
utility (experiments/fedcore/aggregate.py) that reads runs/*.csv and emits
runs/agg_main.csv with mean+/-std. NO single-seed numbers in the final tables.

============================================================
STEP P2 — publication figures + tables from REAL/CPU data (MUST)
============================================================
Produce paper-ready PDF figures (figs/) and CSV-backed tables (runs/) with seeds:
  F2 necessity        : unsafe-deploy rate vs true risk, naive vs Fed-CORE (real-logit version)
  F5 alpha-frontier   : CertCov@alpha vs alpha (d=5), proxy-margin (robust), mean+/-std
  F6 feasibility law  : CertCov@0.1 (and cert_ucb) vs per-group accepted count, Theorem-2 floor overlay (THE signature figure)
  F7 hetero collapse  : certified% / min cert_ucb vs Dirichlet d, with the Thm-2 boundary
  F8 self-training    : per-round contamination + downstream acc, certified vs naive vs none (see P5 for real run)
  (F0 problem diagram, F1 pooling collapse already exist under figs/ — reuse.)
Tables: T1 main results (P1 aggregate), T2 certificate efficiency (conditional vs
mass-ratio vs box vs pooled), T3 necessity, T4 superiority (already runs/T4.csv —
add seeds), T5 score-agnostic, T6 privacy taxonomy (pooled/stratified/grouped:
released stats, leakage, validity scope), T7 self-training delta/T validity.
All figures: matplotlib, mathtext-safe (use \leq, \mathrm; NOT \le, \rm), saved as
PDF (vector) + PNG. Consistent style; colorblind-safe palette.

============================================================
STEP P3 — one breadth dataset (MUST for PR/TNNLS)
============================================================
Add at least ONE non-CIFAR benchmark to answer "CIFAR-only" reviews. Cheapest
options, pick by available compute:
  (a) TinyImageNet OSR split (held-out classes as unknown) — vision breadth; or
  (b) a tabular / medical FL benchmark (e.g. a public clinical tabular set) — fast
      on CPU, and strengthens the "safety-sensitive FL" narrative.
Wire it through the SAME pipeline (open_set_split + dirichlet_partition +
build_calibration + certify). Report the same metric schema; include in T1.

============================================================
STEP P4 — remaining ablations + corruption curve (STRENGTHEN)
============================================================
  H2 split-leakage   : deliberately reuse test labels in proposal/cert -> show the
                        guarantee breaks (empirical risk > alpha beyond delta).
                        This proves the split hygiene is load-bearing.
  H5 client subsample: vary #participating clients in calibration -> CertCov + feasibility.
  H6 unknown-prop    : sweep the fraction of unknown-labeled audit points -> effect
                        on certificate (scarcity of labeled unknowns).
  Corruption curve   : sym/asym noise rate in {0,0.1,0.2,0.35,0.5} x d in {0.5,5}
                        -> CertCov@alpha vs noise rate (connects to corrupted-training origin).

============================================================
STEP P5 — certified self-training on real data, feasible regime (STRENGTHEN)
============================================================
Run run_selftrain_cifar at a regime where certification is feasible (d=5, and the
better backbone if P6 lands), T rounds with round-wise disjoint audit folds
(Prop 4). Report per-round contamination (certified <= alpha vs naive unbounded)
and downstream accuracy (certified vs naive vs none). HONEST headline: "prevents
catastrophic contamination while retaining useful pseudo-labels" — do NOT claim a
guaranteed accuracy gain. Produce F8 from this run.

============================================================
STEP P6 — CONDITIONAL: ResNet primary ladder (if EXP1 wins)
============================================================
If EXP1 shows ResNet-18 lowers realized risk r_hat enough that worst-group (G>=2)
CertCov@0.1 > 0 at d=5: re-run the MAIN ladder (clean/sym0.35/asym0.20 x d in
{0.1,0.5,5} x seeds) with ResNet-18 as the primary backbone, and regenerate T1, the
staircase (F6), and T4 with it. If EXP1 does NOT cross worst-group alpha=0.1, keep
SimpleCNN ladder as main, report ResNet as "lowers r_hat and raises coverage but
worst-group alpha=0.1 needs more per-group calibration", and quantify the residual
gap. Either way, do not fake a crossing.

REPORTING: after each STEP, give 진단 요약 / 확인한 명령 / 핵심 결과 (with mean+/-std)
/ 판정 (strong/moderate/warning/fail) / 다음 행동. Update README per STEP; after
draft sync, update Fed-CORE_draft.md §5 with the aggregated tables/figures.
DELIVERABLES: aggregate.py, figs/*.pdf (F2,F5,F6,F7,F8), runs/agg_*.csv,
runs/T{1..7}.csv, breadth-dataset loader, ablation scripts.
```

---

### Notes for Sanghoon
- **P1+P2 are the real unlock**: they convert "many runs" into "a paper's Section 5"
  (seeded tables + the signature figures F5/F6/F8). Do these before anything new.
- **P3 (one extra dataset)** is the single most common journal-desk-reject fix for
  "CIFAR-only"; a tabular/medical FL set is the cheapest way to satisfy it and fits
  the safety-sensitive narrative.
- **P6 is conditional** on EXP1 — the prompt tells Claude Code to branch honestly
  rather than assume ResNet wins.
- This campaign + the existing theory/figures gets you to a submittable draft for
  Pattern Recognition / Information Fusion.
