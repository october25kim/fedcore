# Fed-CORE — Submission Checklist (target: Information Sciences, Elsevier)

Status legend: ✅ done · 🟡 needs a pass · ⬜ blocking, not done.
Updated 2026-07-09 after the ws4090 expansion queue (R1–R8) was integrated.

## A. Blocking placeholders (must fill before submission)
- ⬜ **Affiliation** — title block (draft line 5): `[Department / University, City, Country]` → fill with the actual department / university / city / country.
- ⬜ **Funding** — Acknowledgements (draft line 459): `[funding source to be completed]` → fill with the grant/number, or replace with "This research received no specific grant from any funding agency" if none.

## B. Content — integrated this session (verified against CSVs)
- ✅ §5.4 client scaling (R1): per-client (G=J) collapses 0/10 at J=10,20 (α=0.20); grouping G=2 restores 6/10 (0.16–0.26). Closes "not federated scale".
- ✅ CIFAR-100 second positive (R2): Abstract / §5.1 / §5.5 / §7 upgraded; 3 backbones × 2 d, 7–8/10, cov 0.05–0.09 at α=0.20; α=0.10 at the edge (3/60). Old single-seed CIFAR-100 negatives removed from Table 5(c).
- ✅ Table 5(b) detectors → 5 seeds (T8/R8a): FedPD-PROSER d=5 α=0.20 **0.532±0.099 (5/5)**; FOOGD 0.089±0.088 (4/5); AUROC 0.81>0.74>0.68>0.47. R8a native-score reconciliation applied.
- ✅ R6 full-simplex positive: FedPD J=5 Theorem-1 per-client 5/5 @α=0.20, mean 0.390 — removes the "grouping = hidden pooling" attack (§5.5 + §7).
- ✅ **δ/2 headline transition**: CertifiedCoverage@α now reported at the simultaneous budget (Corollary 1, δ_r=δ_c=δ/2); headline GN d=5 α=0.20 **0.341 (4/5)**, d=0.5 0.350 (5/5), BN 0.428 (5/5); §5.1 metric definition, Table 5(a), superiority, §7 all updated.
- ✅ R4 (knob) / R5 (corruption, ten-seed seeded) / R7 (covtype 10-seed A2, honest 4/10 edge) prose updated.
- ✅ **Figure 5b** regenerated from `corruption_curve_seeded.csv` (ten-seed symmetric sweep); caption precised.
- ✅ **Theorem 3 converse** added (§4.5): minimax two-point argument showing the feasibility floor is information-theoretic, not a Clopper–Pearson artifact — closes the "is the floor a bound artifact?" attack. *(New theoretical paragraph — give it one review read.)*

## C. Manuscript completeness (current structure — appendix-free)
- ✅ Abstract (math-free), §1 Intro, §2 Related Work (Table 1), §3 Problem Setup (Fig 1, Prop 1, Table 2 + A1–A6), §4 Method (Prop 1 / Lemma 1 / Thm 1 / Thm 2 / Cor 1 / Prop 2 / Thm 3 / Remark 1 / Prop 3, inline proofs), §5 Experiments, §6 Limitations, §7 Conclusion, References [1]–[42].
- ✅ 5 figures, 6 tables; captions 1–2 sentences; all results explained in body prose.
- ✅ Theorem numbering LOCKED — do not renumber.
- ✅ CRediT / Declaration of competing interest / Data availability — complete.

## D. References
- ✅ 42 references, sentence-per-citation (Table 1 cells excepted), appearance-order numbering; metadata web-verified (4 earlier errors fixed).
- ✅ **[37]** (X. Yu, J. Liu, arXiv:2606.08517, 2026) — **VERIFIED real** (title/arXiv/authors/content match; Xiaoli Yu + Jiamiao Liu). Not fabricated.

## E. Honesty gate (final read)
- ✅ Validity: no held-out violation in any certified cell; resampling study 61/526,000 (8.7×10⁻⁴).
- ✅ α=0.20 framed as finite-sample feasibility demonstration, not a safety target; α=0.10 seed-variable.
- ✅ covtype honest edge (4/10 @α=0.30, 8/10 goal not met); corruption collapse honest.
- 🟡 Final overclaim sweep — scan for residual "robust / tight / improves / novel / very" (supervisor guideline).

## F. Packaging (Information Sciences)
- 🟡 **Highlights** (3–5 bullets, ≤85 chars each) — confirm present/updated for the δ/2 + two-positive-dataset story.
- ⬜ Cover letter — contribution = object + finite-sample certificate + pooled-invalidity + feasibility law (with its converse); not an accuracy method.
- ⬜ Remove internal artifacts (`PROMPT_*.md`, `HANDOFF_*.md`, this checklist, `REF_rationale.md`) from the submission bundle.
- ⬜ Final `bash build_docx.sh` after A is filled → submit `docs/Fed-CORE_draft.docx`.

## G. Deferred / optional (non-blocking)
- Canonical git repo (october25kim/fedcore) commit of the ws4090 scripts + theorem1 O(J log J) fix (code is backed up on the laptop; repo/token issue pending).
- Korean draft sync — only on explicit request.

## Highest-value remaining, in order
1. Fill A (affiliation + funding).
2. Review the new Theorem-3 converse paragraph (§4.5) + the final overclaim sweep (E).
3. Highlights + cover letter (F).
4. Strip internal artifacts, final build, submit.
