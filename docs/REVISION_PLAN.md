# Fed-CORE — Critical review & revision plan (2026-07-10)

Purpose of this document (request item 6). You asked for six things: (1) cut to ~40 pages, (2) a
section-by-section critical theory review, (3) more persuasive experimental results backed by `runs/`,
(4) explicit purpose + interpretation per experiment, (5) a final reference renumber in body order, and
(6) a written rationale for every change. Items (1) and (3–4) pull in opposite directions (compress vs.
expand), so the plan below resolves them: **cut redundancy and hedging, and reinvest the space in sharper
per-experiment purpose/interpretation.** This document is the critique + plan + rationale; the edits are
executed only after you approve the cut list (Section E), because a 40-page target means deleting content.

Current length: **15,103 words** (Abstract 330 / §1 1320 / §2 1010 / §3 1373 / §4 3477 / §5 5448 / §6 517 /
§7 491 / refs 991). Double-spaced this is ~55–60 pp of body. Target ~40 pp ⇒ **cut ≈ 4,500 words of body**
(mostly redundancy and §5 hedging), then reinvest ~1,000 in experiment interpretation.

---

## A. Section-by-section critical review (request item 2)

### Abstract
- **Overclaim risk.** "second real-data positive at 0.20" reads as strong, but the headline CIFAR-10 cell
  is now 0.341 (4/5) under the joint δ/2 budget. State the primary number honestly and mark α=0.20 as the
  feasibility-demonstration target once.
- **Too long / redundant with §1.** The Poisson-binomial pooling argument appears here *and* in §1 *and*
  §2 *and* §4.4 (four times). Abstract should assert it in one clause and not re-derive.

### §1 Introduction
- **Strength:** the hospital-analogy opening and the four-literature gap are clear and honest.
- **Redundancy (cut target).** The "why pooling is invalid / Poisson-binomial" is stated at length here
  (¶33) and repeated in §2, §4.4, and the abstract. Keep the intuition here (2 sentences), move the
  derivation to §4.4 only.
- **Novelty is under-sold on the hard part.** The intro leans on "pooling is invalid" as the hook, which
  is the *elementary* half. The genuinely new object is the **mixture-robust linear-fractional certificate
  over an unknown λ** (Thm 2). Foreground that; "pooling is invalid" is the motivation, not the contribution.
- **Contribution list overlaps the abstract.** Tighten from four dense bullets to three.

### §2 Related Work
- **Citation padding (cut target).** ¶56 lists nine adjacent works one sentence each (clustered
  co-meta-learning, fairness, scheduling, graph autoencoders, concept drift…). Several are only loosely
  related; a reviewer reads this as breadth-signalling. Keep the 3–4 genuinely adjacent, fold the rest.
- **Repeated positioning.** The "in one sentence: Fed-CORE is the first federated…" appears in §1 and §2.
  Keep it once (end of §2), with Table 1.
- **Reference-number drift (request item 5).** The joint-certificate neighbor (verified real: X. Yu &
  J. Liu, arXiv:2606.08517) is cited as **[38]** in §2 but the number floats vs. the list; RCPS is [37]
  in one place. Needs a full body-order renumber + citation-vs-list audit.

### §3 Problem Setup
- **A4/A4′ (audit representativeness) is the load-bearing assumption and the biggest attack surface.**
  The guarantee requires the audit fold to carry unknowns at ≥ the *deployment* unknown rate — but the
  deployment rate is unknown at calibration time. A reviewer will press this hard. Currently stated but
  under-defended: add one sentence on how a deployer conservatively over-samples unknowns, and concede
  plainly that mis-specifying it voids validity (Fig 3/§6 already show the anti-conservative collapse).
- **"Distribution-free" is scoped, correctly, but the caveat is significant.** It is distribution-free
  w.r.t. the *audit* distribution, not the true unknown universe. Fine — but keep the scoping sentence
  adjacent to every "distribution-free" claim (currently only stated once).
- **A6 (grouped) weakens the target and the headline uses it.** The main positives are G=2 group-mixture
  certificates; the client-simplex guarantee is demonstrated by exactly one cell (R6). Own this: the
  headline is a *group-mixture* guarantee; R6 is the evidence it is not merely pooling.

### §4 Method and Theory
- **Thm 2 (bounded-Λ) robustness is only inside a user-declared box.** ρ=0.15 is arbitrary and the
  guarantee is void outside the box. This is the price of escaping worst-client domination; state the
  trade-off crisply (simplex = assumption-free but worst-client-dominated; box = tighter but declared).
- **The δ/J union penalty limits federated scale — this is the method's core structural limit.** The
  per-client certificate pays a lnJ penalty and worst-client domination, so it collapses as J grows at
  fixed audit budget (R1 confirms 0/10 at J=10,20). Grouping is the only escape, and it coarsens the
  target. Own this directly in §4.5, not just empirically in §5.4.
- **Two acknowledged open theoretical gaps.** Remark 1 (Lemma L: numerically supported, formal proof TODO)
  and Prop 3 (Gap 2: roster-composition coupling not closed). Both are honest but a reviewer counts them.
  Recommendation: keep Lemma L's numerical support, and either (a) close Lemma L with the short domination
  argument if it exists, or (b) demote Prop 3 to a one-paragraph remark so the two gaps do not read as an
  unfinished theory section. The **new Theorem-3 converse** (just added) is a genuine strengthening —
  it turns the feasibility floor from a possible "CP artifact" into an information-theoretic law.
- **Privacy selling-point is weaker than it first sounds.** Only the *pooled* (invalid) or *grouped*
  variants are sum-only secure-aggregatable; the exact stratified certificate needs per-client pairs.
  The paper corrects this honestly — but then "privacy-preserving" should not be a headline verb; frame
  it as an information-flow characterization with a grouped compromise.

### §5 Experiments — the biggest lever for both cutting and persuasion
- **Hedging is repeated ~5×.** "finite-sample feasibility demonstration, not a safety target" recurs.
  State it once, prominently, then stop repeating.
- **Absolute coverage looks small and α is loose — address head-on, don't bury.** The headline is 0.341
  accepted-and-certified at α=0.20; α=0.10 collapses. A reviewer's first reaction is "34% coverage at a
  20% error budget — is this useful?" The honest, persuasive answer (to add in interpretation): the
  alternative is *zero* guaranteed coverage; Fed-CORE certifies the largest safe fraction and *correctly
  declines* the rest, and the feasibility law (now with a converse) proves the collapse at α=0.10 is
  fundamental, not a looseness. Make this the framing, not a caveat.
- **No head-to-head against a real method.** "Superiority" is vs. a test-peeking oracle and a naive
  threshold. A reviewer will want centralized SCRC/joint-cert applied naively to federated data, shown to
  be anti-conservative (Prop 2 is the theory; add the empirical head-to-head if `runs/` supports it, else
  state why it is positioned as an oracle).
- **Per-experiment purpose+interpretation is implicit (request items 3–4).** Each result should open with
  one purpose sentence (which reviewer attack it closes) and close with one interpretation sentence (what
  the number means for a deployer). Mapping: §5.2 validity → "does it ever lie?"; §5.3 A4 → "when is it
  valid?"; §5.4 feasibility+R1 → "when is it non-vacuous, and does it scale?"; §5.5 detectors+R2+R6 →
  "does coverage track detector quality, on a second dataset, without hidden pooling?"; §5.6 self-training.

### §6 Limitations & §7 Conclusion
- Honest and appropriately scoped. Trim §7 ~15% (it restates the abstract). §6 keep.

---

## B. Page-reduction plan (request item 1) — target ≈ 40 pp

| lever | where | approx words cut |
|---|---|---|
| De-duplicate Poisson-binomial/pooling (state once in §4.4) | Abstract, §1, §2, §4.4 | ~350 |
| Trim §1 contribution list + repeated positioning | §1 | ~300 |
| Fold §2 adjacent-lines citation padding | §2 ¶56 | ~150 |
| Remove repeated α=0.20 hedging; state once | §5 | ~600 |
| Compress verbose diagnostics prose (keep numbers) | §5.3–5.5 | ~1,200 |
| Tighten proof prose (keep every step, cut narration) | §4 | ~500 |
| Demote Prop 3 to a remark (if you approve) | §4.6 | ~300 |
| Trim §7 conclusion | §7 | ~150 |
| **Total cut** | | **~3,550** |
| **Reinvest in experiment purpose/interpretation** | §5 | **+~1,000** |
| **Net** | | **~−2,550 words ⇒ ~46–48 pp; a second pass reaches ~40** |

Reaching a hard 40 pp likely needs one structural cut too — the most defensible is **moving the synthetic
controlled study and one stress figure to a "supplementary" that INS allows as an online appendix**,
keeping the real-data story in the 40 pp. Flag for your decision (Section E).

## C. Experiment strengthening (items 3–4) — runs-backed, per experiment
For each, I will add a one-line **Purpose** and one-line **Interpretation**, and pull the exact numbers
from `runs/` (already verified this session). Concretely:
- **R1 (client_scaling)** — Purpose: does the certificate scale with the federation? Interpretation: the
  monotone G=2→J degradation (0.16→0.02, 6/10→0/10) is Theorem 3 on real data; grouping is the declared,
  non-pooling rescue.
- **R2 (cifar100_multimodel)** — Purpose: is CIFAR-10 a fluke of one dataset/backbone? Interpretation:
  6/6 cells non-vacuous across 3 backbones × 2 d (7–8/10), a properly powered second positive.
- **T8 detectors (5-seed)** — Purpose: does a real FedOSR detector help? Interpretation: coverage tracks
  native AUROC (0.81>0.74>0.68>0.47) at fixed validity — the score-agnostic claim made concrete.
- **R6 (simplex_positive)** — Purpose: is the grouped headline hidden pooling? Interpretation: a full
  client-simplex 5/5 positive (0.390) rules it out.
- **R4/R5/R7** — buffer necessity, corruption collapse, covtype honest edge.

## D. Reference renumber (item 5)
Full pass: (1) list every in-text `[n]`; (2) reorder the reference list to first-appearance order;
(3) verify each entry is cited and each citation resolves; (4) fix the current drift (RCPS vs joint-cert
around [37]/[38]); (5) confirm [38] = arXiv:2606.08517 (verified real). Done last, after prose edits settle.

## E. What needs your approval before I execute
1. **Prop 3 → remark?** (removes one open-gap read, saves ~300 words). Yes / keep as Proposition.
2. **Move the synthetic study + one stress figure to an online supplement** to hit a hard 40 pp? Yes / keep
   all in the main body (then ~44–46 pp).
3. **Head-to-head baseline:** do you have `runs/` for centralized-SCRC-applied-naively (to show empirical
   anti-conservativeness), or should Prop 2 remain the theory-only argument with the oracle framing?

Approve E1–E3 (or say "your call") and I will execute in this order: §5 purpose/interpretation + hedging
cut (items 3,4,1) → §4/§1/§2 de-duplication + tightening (item 1) → reference renumber (item 5) → final
build, with a per-change rationale log appended here (item 6).
