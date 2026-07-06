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

## Task A — Verify the G=2 grouping rule (correctness gate; do this first)

`exp_resampling_validity.py` and `make_e1_figs.py` group clients into G=2
CONTIGUOUS blocks via `np.array_split(np.arange(J), G)`. The original worst-
group headline (Table 5: 0.392/0.353 at alpha=0.20, and `agg_main.csv`
CertCovG2 columns) was produced by an aggregation script that lives on this
machine (it is not in the repo — locate it; it produced `runs/agg_main.csv`
and `runs/T1_main.csv`).

1. Find that script and identify its G=2 grouping rule.
2. If it matches contiguous `array_split`: commit the script into
   `experiments/fedcore/` (it is a reproducibility gap) and report "match".
3. If it differs (e.g., sorted-by-size pairing): re-run
   `python experiments/fedcore/exp_resampling_validity.py` and
   `python experiments/fedcore/figs/make_e1_figs.py` with the original rule
   (add a `--grouping` flag rather than editing constants), then update the
   numbers in `docs/Fed-CORE_draft.md` Sections 5.2 and 5.4 and the Figure 3/5
   captions. Report old vs new numbers explicitly.

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
`backbone,d,alpha,seed,cert_risk_ucb_G2,cert_n_min_group,cert_coverage_lcb,test_risk,certified`
and extend Table 5 in the draft with the alpha=0.20 medians (replace the
"Diagnostics ... are from the alpha=0.10 runs" caveat in the caption and the
Section 5.5 headline paragraph accordingly). Do not alter the CertifiedCoverage
numbers themselves.

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
