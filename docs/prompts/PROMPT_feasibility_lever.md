# Claude Code prompt — feasibility lever (grouped-stratified, post-hoc, no retraining)

Paste the fenced block into Claude Code on the 4070. This is the decisive, cheap
test of the α=0.1 diagnosis: certified coverage is limited by per-client accepted
counts (Theorem 2), not by the operating point. It reuses EXPORTED logits — no
GPU retraining.

```text
CONTEXT (read CLAUDE.md + AGENTS.md first). The real CIFAR ladder gives
CertifiedCoverage@alpha=0.1 = 0; the gamma patch refuted operating-point tuning
and showed the binding constraint is per-client accepted count vs the Theorem-2
floor (cert_n 500->151 ~30/client < 37; UCB rose 0.185->0.222). Now test the
ACTUAL lever directly, post-hoc, on the SAME exported cert/test logits (no retrain).

GOAL: show that raising per-GROUP accepted counts via the GROUPED-STRATIFIED
certificate (paper sec 4.4) flips alpha=0.1 from vacuous to non-vacuous, producing
a clean Theorem-2 staircase. Grouped-stratified is a legitimate paper variant
(worst-GROUP guarantee over G public strata of >=k clients), and doubles as the
privacy compromise — so this is not cheating.

INPUTS: the exported raw-logit npz from cifar10 d=5 clean (cert fold + test fold,
with per-point client id, y_open, logits). If not present, re-export via run_cifar
with the npz flag (no retrain needed beyond what already ran).

IMPLEMENT (post-hoc, CPU is fine):
1. certificates.py / certify.py: add a grouped-stratified path. Given a client->
   group map (G groups), aggregate per-GROUP counts (A_g, K_g) and apply the
   conditional certificate over the G groups (eps = delta/G for simplex-over-
   groups, or box over the group simplex). G=1 reduces to the pooled certificate
   (valid here only because d=5 is near-IID/near-matched — flag this caveat).
2. exp_feasibility_lever.py (new): from the SAME exported logits, sweep:
   (a) grouping G in {5 (per-client), 3, 2, 1 (pooled)};
   (b) calibration size: cert-fold fraction in {0.33, 0.5, 0.7} of the trusted pool.
   For each (G, frac): run certify_best_gamma at alpha=0.10 (box-Lambda over
   groups), record per-group cert_n, cert_risk_ucb, CertifiedCoverage@0.1,
   test_risk (must stay <= alpha when certified).
   Plot/table: cert_risk_ucb and CertCov@0.1 vs per-group cert_n, overlaying the
   Theorem-2 floor ln(G/delta)/(-ln(1-alpha)) AND the (alpha - rhat)^2 sample
   requirement z^2 rhat(1-rhat)/(alpha-rhat)^2 with rhat from the data.

3. PATCH (frontier monotonicity): add a proxy SAFETY MARGIN to certify_best_gamma
   — choose gamma* using proposal-proxy U_proxy <= alpha - epsilon (epsilon ~
   0.25*(alpha - rhat) or a small constant); re-run the d=5 alpha-frontier
   {0.10,0.15,0.20,0.25} and confirm it is monotone and that alpha=0.20 stays ~0.16.

VALIDITY (do not break): grouping must be a PUBLIC, data-independent client->group
map (e.g., by client index), fixed before seeing cert labels. The worst-group
certificate at eps=delta/G is valid. G=1 (pooled) is valid only under matched
mixture — report it but label it clearly as the near-IID-only bonus, kept
subordinate to the grouped (G>=2) result.

EXPECTED / REPORT (진단/명령/핵심결과/판정/다음행동):
- Hypothesis: as G decreases (counts per group rise), cert_risk_ucb decreases
  monotonically through alpha=0.1; the smallest G (or largest cert frac) with
  per-group cert_n above the (alpha-rhat)^2 requirement (~few hundred for
  rhat~0.08) gives CertCov@0.1 > 0. Report the crossover.
- If even G=2 with the largest cert fraction stays vacuous at alpha=0.1, report
  the exact gap (available vs required per-group cert_n) and conclude a stronger
  backbone (lower rhat -> larger (alpha-rhat) -> far smaller requirement) is the
  remaining lever. Do NOT fake certification.
- Headline to capture for the paper: a single figure "CertifiedCoverage@0.1 vs
  per-group accepted count, with the Theorem-2 floor" — this turns the honest
  null into a quantitative law.

RULES: never certify on the test fold; grouping is public/fixed; report failures;
judge by cert_* not AUROC. Update README + (after draft sync) Fed-CORE_draft.md
sec 5.1 with the staircase result.
```

---

### Note for Sanghoon
- This is the experiment that converts "α=0.1 is null" into "α=0.1 is governed by a
  feasibility law we can move with calibration budget" — the difference between a
  weak null and a characterized phenomenon (much stronger for review).
- It is post-hoc on exported logits, so it is fast and needs no GPU.
- Grouped-stratified here is the SAME object as the §4.4 privacy compromise, so one
  experiment serves both the feasibility story and the privacy story.
- If G=2–3 flips α=0.1 positive, that is the paper's headline; if only G=1 (pooled,
  near-IID) does, report it honestly as the matched-mixture bonus and push the
  backbone lever for the worst-group result.
