# PROMPT — Figure/Table upgrade + diagnostics regeneration (Ubuntu 4070)

Read `CLAUDE.md` and `AGENTS.md` at the repo root FIRST and follow them. This
prompt continues the manuscript revision in `docs/Fed-CORE_draft.md` (source of
truth; `bash build_docx.sh` regenerates the docx). Docker-first for anything
that touches torch; everything below except Task E is CPU-only and may run on
the host or in the container.

Context of what was already done (do NOT redo):

- `experiments/fedcore/exp_resampling_validity.py` ran on the 55 stored
  `runs/*_logits.npz`: 526,000 certificate evaluations, 70,025 deployments,
  61 violations (rate 8.7e-4, CP95 UCB 1.1e-3, delta=0.10); 56/61 violations
  at gamma=1.0. Wired into the draft as Figure 3 (Section 5.2).
- `experiments/fedcore/figs/make_e1_figs.py` produced FE1/FE2/FE3
  (Figures 3, 12, 5 in the draft).
- Draft table numbering: Table 1 prior-work map, Table 2 variants, Table 3
  config, Table 4 necessity, Table 5 main results, Table 6 edge cases,
  Table 7 superiority, Table 8 base models, Table 9 self-training.
- Figure numbering: 1 problem, 2 pooling, 3 resampling validity, 4 A4 stress,
  5 anatomy, 6 feasibility law, 7 frontier, 8 stress axes, 9 self-training,
  10 client scaling (App C), 11 self-training gain (App C), 12 tightness (App C).

## Task A — Identify the exact headline aggregation protocol (correctness gate; do this first)

`exp_resampling_validity.py` and `make_e1_figs.py` group clients into G=2
CONTIGUOUS blocks via `np.array_split(np.arange(J), G)` and use the STORED
prop/cert/test folds in the npz. The original headline (Table 5: 0.392/0.353
at alpha=0.20, `agg_main.csv` CertCovG2 columns, "cf=0.5" in the T1 header)
was produced by an aggregation script that lives on this machine (not in the
repo — locate it; it produced `runs/agg_main.csv` and `runs/T1_main.csv`).

IMPORTANT FINDING from the laptop session (2026-07-06): reconstructing with
contiguous grouping + stored folds does NOT reproduce agg_main exactly —
GN d=5 matches n_pass 2/5 but median UCB differs (0.092 vs 0.0878), and GN
d=0.5 gives n_pass 2/5 vs agg's 3/5. The "cf=0.5" tag suggests the original
script RE-SPLIT the trusted pool with cert fraction 0.5 rather than using the
stored 0.34/0.33/0.33 folds, and its gamma-selection rule may differ.

1. Find the script; identify (a) fold re-splitting protocol, (b) G=2 grouping
   rule, (c) gamma selection rule, (d) WHICH coverage quantity CertCov reports
   (test-fold accepted coverage? cert-fold coverage? cert_coverage_lcb?).
2. Commit the script into `experiments/fedcore/` (reproducibility gap).
3. State the identified coverage quantity explicitly in every Table 5/6/7/8
   caption and in the CertifiedCoverage@alpha definition in Section 5.1
   (the draft currently defines the certified quantity as cert_coverage_lcb
   per Corollary 1 and calls test_coverage the deployment estimate — align
   the captions with what the numbers actually are).
4. If the protocol differs materially from the resampling script's, note in
   Section 5.2 that the resampling study uses its own fixed protocol (stored
   folds, contiguous groups) — the draft already says "contiguous public
   groups, fixed a priori" — and do NOT silently renumber it.

## Task B — Table 5 diagnostics at alpha=0.20

Table 5 currently reports median worst-group cert UCB and mean test_risk from
the alpha=0.10 aggregates only (that is what `agg_main.csv` contains). Using
the stored logits and the SAME protocol that produced the alpha=0.20 headline
(worst-group G=2, fixed MSP, cert_frac=0.5), compute per cell (ResNet-GN d=5,
ResNet-GN d=0.5, ResNet-BN d=5; 5 seeds each):

- median worst-group `cert_risk_ucb` at alpha=0.20
- median per-group accepted count `cert_n` (per group, not summed)
- `cert_coverage_lcb` mean over certified seeds

Write `runs/T9_diagnostics.csv` with schema
`backbone,d,alpha,seed,cert_risk_ucb_G2,cert_n_min_group,cert_k_worst_group,cert_coverage_lcb,test_risk,test_coverage,certified`
and extend Table 5 in the draft so the main table itself shows, per cell:
`certified seeds`, `cert_n` (min per-group), `cert_k` (worst group),
`cert_risk_ucb` (median), `cert_coverage_lcb` (mean over certified),
`test_risk`, `test_coverage` — at BOTH alpha=0.10 and alpha=0.20 (replace the
"Diagnostics ... are from the alpha=0.10 runs" caveat). The reviewer
requirement is that the paper itself, not the code release, carries the
certification diagnostics. Do not alter the CertifiedCoverage numbers
themselves; if Task A reveals they are not cert_coverage_lcb, label the
column accordingly rather than recomputing.

## Task C — Regenerate four figures to match the rewritten captions

The draft's captions were shortened and de-editorialized; the figure files
still carry the old embedded titles/annotations. Regenerate (PDF+PNG, same
filenames, scripts committed under `experiments/fedcore/figs/`):

1. `F6_feasibility_law` (draft Figure 6): 3 panels per the new caption —
   (a) worst-group UCB vs per-group accepted count with a VERTICAL Theorem-2
   floor marker at ln(J/delta)/(-ln(1-alpha)) approx 37/client; (b)
   CertifiedCoverage@0.10 vs grouping; (c) budget sweep. Remove any in-figure
   editorial text ("we report honestly" etc.); keep annotations minimal.
2. `F5_alpha_frontier` (draft Figure 7): relabel the alpha=0.20 vertical
   marker from "practical operating point" to "feasibility demonstration
   point"; add a marker at alpha=0.10 labeled "feasibility edge".
3. `FA5_unknown_proportion` (draft Figure 4): retitle to "Audit
   representativeness (A4) stress test"; x-axis label "calibration unknown
   fraction"; keep the 1-delta reference line.
4. `FJ_client_scaling` (draft Figure 10): reframe as "the price of
   federation": extend the J sweep to include J=1 (centralized anchor; same
   total audit budget) so the curve reads "certified coverage cost of
   splitting a fixed audit budget across J clients". CPU-only synthetic —
   extend `exp_ablation_extra.py` if that is where FJ came from.

Publication-ready embedded titles (replace debug-style titles):

| current embedded title | replacement |
|---|---|
| "Non-reducibility: pooling collapses, stratified holds" | "Pooling certifies the wrong mixture under heterogeneity" |
| "F5 Certified-coverage frontier (cifar10 d=5)" | "Certified coverage frontier on CIFAR-10" |
| "A5 Audit must represent unknowns" | "Audit representativeness (A4) is required for validity" |
| "J Per-client starvation / log(J) penalty" | "Fixed audit budgets induce a federation penalty" |
| "SAFE accuracy gain" | "Certified self-training: accuracy gain under admitted pseudo-labels" |

Also: increase font sizes in the 3-panel feasibility figure (journal column
width), and unify all in-figure assumption labels to A4 (several PNGs still
say A5).

After regenerating: `bash build_docx.sh` and visually check the four figures
in the docx.

## Task D — covtype procedurally-valid multi-score cell (CPU, uses stored covtype logits if present; otherwise skip and report)

The draft demoted the covtype best-of-scores number to a non-guaranteed
diagnostic (Section 5.5, Table 6). Produce the procedurally valid version:
select the score on the PROPOSAL fold (or Bonferroni delta/4 across the four
scores — implement both, report both), then certify once. Update the covtype
row of Table 6 and the paragraph below it with whichever valid protocol you
ran. If covtype logits are not stored, rerun the covtype pipeline first
(CPU/MLP-scale) with the same seeds.

## Task E (optional, GPU, only if time permits) — seed extensions

1. FedPD-PROSER: extend 3 -> 5 seeds at d in {0.5, 5}, alpha in {0.10, 0.20}
   (Table 8; keep the pretrain-then-finetune recipe; Docker).
2. Self-training gain (Figure 11): extend to n=5 seeds at the 4x budget cell.

Priorities: A > B > C > D > E. A/B/C/D are CPU-only.

## Guardrails

- proposal/certification/test split hygiene is inviolable; never use test
  labels in proposal or certification.
- Do not change any theorem statement, the metric schema, or already-verified
  numbers without reporting the discrepancy first.
- Never commit `runs/`, `data/`, weights, or generated docx/pdf/png (see
  `.gitignore`); DO commit the scripts.
- Report in the fixed format (진단 요약 / 확인한 명령 / 핵심 결과 / 판정 /
  다음 행동), one report per task. Report failed commands as failed.
- Commit messages in English, one commit per task, prefix `figs:`, `tables:`,
  `exp:` as appropriate.
