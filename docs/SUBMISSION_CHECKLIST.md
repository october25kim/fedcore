# Fed-CORE — Submission Checklist (target: Pattern Recognition)

Status legend: ✅ done · 🟡 partial / needs a pass · ⬜ not started.
Primary venue **Pattern Recognition**; alternates Information Fusion → IEEE TNNLS.

## 1. Manuscript completeness
- ✅ Abstract — reflects final results (valid everywhere; feasibility law; α=0.20 robust 2-domain; α=0.10 seed-variable).
- ✅ Introduction — problem, three-literature gap, "why not pooling / not union bound", contributions, findings preview.
- ✅ Related Work — FedOSR (FedPD/FedPD++/FedOSS/FOOGD/FedOV/FedNovel), federated CP (Lu/Plassier/Rob-FCP), conformal risk-control (CRC/SCRC/SCoRE/joint), decentralized novelty; positioning sentence.
- ✅ Problem Setup — clients/mixture, accepted selective risk R_sel(λ), split, risk buffer, **audited-deployment caveat early**.
- ✅ Method — Thm 1/1′ (conditional certificate + edge cases), Thm 2 (feasibility), §4.4 privacy taxonomy, Prop 3 (subordinate, Lemma L resolved + Gap 2 open), §4.6 score-agnostic, §4.7 utilizations + Prop 4.
- ✅ Experiments §5 + §5.1 results (T1 law table, F2/F5/F6/F7/F8/F9, T2–T7, covtype breadth, H2 leakage).
- ✅ Limitations — calibration assumption, privacy, self-training, conservativeness.
- ✅ Conclusion — empirical headline + contribution framing.
- 🟡 **Merge §5 "proposed plots/experiments" prose into past-tense Results** (currently mixes plan + §5.1 results) — a writing pass for the camera-ready.
- ⬜ Appendices: A notation, B venue (internal — remove before submission), C mass-ratio baseline, **+ proofs** (Thm 1/1′/2 full proofs, Lemma L from `LEMMA_L_proof.md`, Prop 4), **+ BN/GN comparison table**, **+ best-of-N-scores (optimistic) table**.

## 2. Figures (embed PDFs from figs/)
- ✅ F0 problem diagram (svg/png) · F1 pooling collapse (pdf/png).
- ✅ F2 necessity · F5 α-frontier · F6 feasibility law (5-seed band) · F7 hetero collapse.
- ✅ F8 self-training (per-round contamination + acc) · F9 corruption curve.
- ⬜ Ensure all are vector PDF, consistent fonts/palette, mathtext-safe, captioned; embed in manuscript.

## 3. Tables (from runs/*.csv)
- ✅ T1 main (GN-primary, mean±std, fixed-MSP) · T2 efficiency · T3 necessity · T4 superiority · T5 score-agnostic · T6 privacy taxonomy · T7 self-train δ/T.
- ⬜ Appendix: BN/GN comparison; best-of-N-scores (optimistic).

## 4. Theory / claims calibration (honesty gate)
- ✅ Thm 1/1′ proof sketches; edge cases in statements.
- ✅ Lemma L resolved (threshold domination + Bernstein fallback); Prop 3 subordinate (Gap 2 open) — not in abstract.
- ✅ Privacy: only pooled is sum-only (corrected); grouped-stratified compromise.
- ✅ "distribution-free" scoped to the audited distribution (stated early).
- ✅ Self-training: contamination-control guaranteed, **no claimed accuracy gain**.
- ✅ α=0.10 reported as seed-variable (2–3/5), cert_frac=0.5 in the claim; no faked crossing.
- ⬜ Final read for any residual overclaim (esp. "robust", "tight", "improves").

## 5. Reproducibility
- ✅ Code: certificates/certify/clients/selector/scores/config/fedosr_split/noise/models/fed_train; run_smoke/run_cifar/run_selftrain/run_tabular; exp_* (lemma_L, pooling_fail, necessity, leakage, feasibility_lever); aggregate/tables/make_figures.
- ✅ CPU acceptance gate (lemma_L 0.918 / pooling collapse / necessity 0.48·0.07·0.00 / smoke).
- ⬜ Seeds fixed and listed; exact configs (rounds, cert_frac=0.5, norm=gn, box-Λ, proxy margin) in an appendix/README table.
- ⬜ Anonymized code release + `runs/` schema doc (gitignore heavy artifacts).
- ⬜ Dockerfile / env pin (note: scipy installed at container start — bake into image).

## 6. Breadth / reviewer-objection coverage
- ✅ Two domains (CIFAR-10/100 vision + covtype tabular FL) — "CIFAR-only" defended.
- ✅ GroupNorm primary — "BatchNorm-in-FL" objection removed (BN/GN in appendix).
- ✅ "just per-client CP + union bound?" rebuttal (§4.2c).
- ✅ Necessity (H2 leakage, pooling-fail, naive-threshold) — "why the machinery is needed".
- 🟡 TinyImageNet (vision breadth) — optional if a reviewer pushes; covtype already covers tabular.

## 7. Submission packaging
- ⬜ Format to Pattern Recognition (Elsevier) template; abstract ≤ word limit; highlights (3–5 bullets).
- ⬜ Cover letter: state contribution = object + certificate + pooled-invalidity + feasibility law (not accuracy); suggest reviewers.
- ⬜ Remove internal artifacts (venue-rationale appendix, PROMPT_*.md, HANDOFF.md) from the submission bundle.
- ⬜ Title finalize (e.g., "Fed-CORE: Certifying Accepted Selective Risk for Federated Open-Set Recognition").

## Highest-value remaining (in order)
1. Writing pass: merge §5 plan→Results past tense; add the proof appendix (Lemma L, Thm 1/1′/2, Prop 4).
2. Embed figures/tables into the manuscript; finalize captions.
3. Reproducibility appendix (exact configs/seeds) + code/data release prep.
4. Format to PR template + cover letter + highlights.
5. (Optional) TinyImageNet for extra vision breadth; longer-trained GN for the α=0.10 edge.
