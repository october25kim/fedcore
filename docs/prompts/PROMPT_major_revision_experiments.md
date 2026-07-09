# PROMPT — Major-revision experiments (post-R-queue; 2026-07-09)

Read `CLAUDE.md` and `AGENTS.md` first and follow them. Docker-first for
anything touching torch; CPU tasks run on the host. Data artifacts only (CSV
under `runs/`, scripts committed); no manuscript edits. Report per task in the
fixed format (진단 요약 / 확인한 명령 / 핵심 결과 / 판정 / 다음 행동); report
failed commands as failed; no silent retries.

Context: the R-queue (client scaling, CIFAR-100 multi-model, FedPD/FOOGD seed
extensions, simplex positive, covtype, corruption, rho/gamma) has been
integrated into the manuscript. This queue answers the external
major-revision review. Manuscript-side changes already made (for orientation):
new Proposition 3 states the grouped certificate is EXACT only under i.i.d.
group-mixture audit sampling, and our per-client quota sampling is declared as
a caveat backed by the resampling study; the pooling claim was reframed as
wrong-target-under-mixture-shift; Table 6 = CIFAR-100 multi-model; Figure 4d =
client scaling.

## Seed policy
Minimum 10 seeds per new cell ({0..9}); detector retraining pipelines (M3, M6)
keep the floor-5 exception, flagged per cell. Per-seed rows always.

## M0 (P0, GPU) — Complete the self-training 5-seed cell
`runs/selftrain_gain_5seed.csv` contains only seeds {0,1,2} despite its name.
Run seeds {3,4} (FedPD-PROSER base, 4x audit budget, one-shot delta
certification, same recipe) and append
(seed,certified_gain,oracle_gain,admitted,max_contamination). This is the last
sub-floor cell in the paper.

## M1 (P0, CPU) — Grouped-certificate validity stress
Directly supports the new Proposition 3 and answers the review's main theory
attack ("is grouping hidden pooling?"). Two studies, one CSV
`runs/grouped_validity_stress.csv`
(study,sampling,G,within_group_risk_spread,n_per_client,coverage,mean_ucb,trials):
1. SYNTHETIC: population of J=6 clients in G=2 groups; sweep the within-group
   risk spread r_j in {(0.02,0.02,0.02), (0.01,0.02,0.06), (0.01,0.02,0.20)}
   per group at fixed group means; compare QUOTA sampling (fixed n_j per
   client) versus IID-GROUP-MIXTURE sampling (each audit point drawn from
   P_g); T >= 5,000 trials per cell; report empirical coverage of the grouped
   certificate against the group-mixture target at delta=0.10.
   Acceptance: iid-group sampling covers >= 1-delta everywhere (Proposition 3
   exactness); quota sampling's worst cell quantifies the caveat.
2. REAL LOGITS: on stored GN d in {0.5, 5} npz, re-run the resampling harness
   with BOTH sampling modes for G=2 and report the coverage gap.
Either outcome strengthens the paper: small gap = caveat is immaterial; large
gap = the A6 sampling condition is load-bearing and honestly quantified.

## M2 (P0, GPU) — Headline seed equalization to 10
Table 5(a) rests on 5 seeds while supporting results use 10. Train seeds
{5..9} for the three baseline configs (GN d=5, GN d=0.5, BN d=5; FedAvg+MSP,
same protocol as existing T9 rows: cert_frac 0.5, box 0.15, gamma grid,
delta=0.10) = 15 training runs. Extend `runs/T9_diagnostics.csv` AND
`runs/T9_diagnostics_simul.csv` (simultaneous delta/2 budget, which is now the
headline convention). Report new per-cell mean±std at both alpha in
{0.10, 0.20} so Table 5(a) can be updated to 10 seeds.

## M3 (P1, GPU) — Full-simplex expansion on FedPD-PROSER [detector: floor 5]
The simplex positive currently covers d=5, alpha=0.20 only. Extend
`runs/simplex_positive.csv` with: (a) d=0.5, alpha=0.20, seeds {0..4};
(b) d=5 and d=0.5 at alpha=0.10, seeds {0..4} (enlarged audit budget,
cert_frac 0.5). A clean alpha=0.10 simplex failure is also reportable as the
measured price of client-simplex robustness at the hard target.

## M4 (P1, GPU) — Unknown-class split robustness
All CIFAR-10 results use one known/unknown split. Run TWO alternative splits
(pre-declare them, e.g. rotate which 4 classes are unknown: splits B and C),
GN baseline, d=0.5, clean, seeds {0..9} per split, grouped G=2, alpha in
{0.10, 0.20} -> `runs/unknown_split_robustness.csv` (T9 schema + split
column). Acceptance: CertCov@0.20 within seed-noise of the primary split, or
an honest report of split sensitivity.

## M5 (P1, CPU) — A4 composition stress beyond the unknown fraction
Section 5.3 stresses the unknown FRACTION; the review asks for COMPOSITION.
On stored GN logits, hold the audit unknown fraction fixed at 0.30 but reweight
WHICH held-out classes appear in the audit fold versus deployment (e.g. audit
unknowns drawn from 2 of the 4 held-out classes; deployment uses all 4, and the
reverse). Report coverage of the true deployment risk ->
`runs/a4_composition_stress.csv`
(audit_classes,deploy_classes,coverage,mean_ucb,true_risk,trials).
Expected: composition mismatch can break validity even at matched fraction;
this sharpens the A4 wording with data.

## M6 (P1, GPU) — FedPD-PROSER on CIFAR-100 [detector: floor 5, 3 acceptable]
One cell: d=5, alpha in {0.10, 0.20}, seeds {0,1,2}, grouped G=2, pretrain-
then-finetune recipe, n_known 60. Purpose: test whether the strong-detector
effect transfers, upgrading the thin CIFAR-100 baseline coverage (0.05-0.09)
-> extend `runs/cifar100_multimodel.csv` (backbone=fedpd_proser).

## M7 (P2, CPU) — DP count-release ablation
Laplace noise on the released pairs (A_g, K_g) at epsilon in {1, 3, 10},
delta_DP handling documented; widen the CP level to absorb the noise bound;
grouped G=2 on stored GN d=5 logits, 10 seeds ->
`runs/dp_count_release.csv` (epsilon,seed,alpha,cert_risk_ucb,
cert_coverage_lcb,certified). Question: how much certified coverage does a
formal DP count release cost? No DP theorem claimed; empirical cost curve only.

## M8 (P2, CPU) — Client scaling to J=50 (synthetic)
Extend the Theorem-3 scaling study synthetically to J in {10, 20, 50} at fixed
total audit budget, G in {J, 10, 5, 2}, T >= 2,000 trials ->
`runs/client_scaling_synth.csv`. Purpose: show the J-scaling law beyond the
GPU-feasible range; complements Figure 4d.

## M9 (P2, CPU) — Oracle comparison consolidation
One CSV for a possible comparison figure: Fed-CORE (grouped G=2), matched-
mixture pooled diagnostic, and test-peeking oracle coverage on stored GN d=5
logits at alpha=0.20, 5 seeds -> `runs/oracle_comparison.csv`
(method,seed,coverage_or_lcb,valid,uses_test_labels). Values already exist
piecemeal; consolidate under one protocol.

## Priorities
M0 > M1 > M2 > M3 > M4 > M6 > M5 > M7 > M8 > M9.
CPU tasks (M1, M5, M7, M8, M9) can run in parallel with the GPU queue.
If GPU time is short: M0 + M2 + M3 remove the biggest remaining attack
surfaces (last 3-seed cell, headline seed asymmetry, simplex scope).

## Guardrails
- proposal/certification/test split hygiene is inviolable.
- corruption on TRAIN labels only; calibration folds stay clean.
- contiguous pre-declared grouping; state the rule in every CSV header.
- never commit runs/, data/, weights; DO commit new scripts.
- pre-declare the M4 splits and M1 populations in the script before running;
  no post-hoc selection of favorable configurations.
