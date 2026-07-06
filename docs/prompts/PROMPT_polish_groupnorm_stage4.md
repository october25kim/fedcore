# Claude Code prompt — polish + GroupNorm re-run (key) + STAGE 4

Paste the fenced block into Claude Code on the 4070. Section 5 is complete; this
removes the last real weakness (BatchNorm-in-FL + α=0.1 seed variance via GroupNorm)
and adds the strengthening ablations. Priority order; time-box GPU; stop-and-ask
before exceeding budget; do not fake any result.

```text
CONTEXT (read CLAUDE.md + AGENTS.md first). Section 5 is assembled (T1 the-law
table, F2/F5/F6/F7, T2-T7, covtype breadth). Method valid everywhere (zero false
certificates). Robust headline = feasibility law + α=0.20 worst-group on two domains
(CIFAR d=5 ResNet 0.444±0.056 5/5; covtype 0.433). The α=0.10 worst-group result is
seed-variable (3/5) and the ResNet currently uses BatchNorm — both are addressed
here. Fixed score = MSP for main numbers; cert_ucb summarized by median among
certified seeds. Docker-first; proposal/cert/test disjoint; judge by cert_*, not
AUROC; report failures.

PRIORITY 1 — CPU polish (cheap, do first).
- F6 5-seed band: replace the single-seed staircase with a d=5 shaded band (mean
  +/- std over the 5 seeds) for cert_ucb and CertCov@0.1 vs per-group accepted
  count, Theorem-2 floor overlay. Regenerate figs/F6 (pdf+png), mathtext-safe.
- Aggregation correctness in aggregate.py / tables:
   * report CertifiedCoverage@alpha mean+/-std (well-defined; 0 when uncertified)
     as the PRIMARY metric;
   * summarize cert_risk_ucb by the MEDIAN among certified seeds (uncertified ->
     U=+inf by construction; never report a mean that is inf);
   * MAIN T1 numbers use the FIXED score MSP; move any best-of-N-scores selection
     to an appendix table clearly labeled "selection over N scores (optimistic)".
- Regenerate T1 with these conventions.

PRIORITY 2 — GroupNorm re-run (the key experiment; GPU).
Rationale: BatchNorm under non-IID FedAvg diverges in running stats and likely
drives the alpha=0.10 seed variance; GroupNorm is the standard FL-appropriate choice
and may both remove a reviewer objection AND stabilize the result.
- Ensure models.py ResNet-18 supports --norm {bn,gn} (GroupNorm ~32 groups).
- Re-run cifar10 ResNet-18 + GroupNorm, d in {5, 0.5}, clean, seeds {0,1,2,3,4},
  cert_frac=0.5, export npz. Recompute worst-group G=2 (and G=3) CertCov@{0.10,0.20}
  and the 5-seed gate (n_pass/5), median cert_ucb, r_hat per seed.
- COMPARE BatchNorm vs GroupNorm: r_hat, seed std of r_hat, n_pass@alpha=0.10.
  JUDGE honestly:
    * if GroupNorm raises n_pass to >=4/5 at alpha=0.10 -> upgrade the headline to a
      more robust worst-group alpha=0.10 result (state norm=GN, cert_frac=0.5);
    * if not -> report GroupNorm as the principled FL normalization with comparable
      coverage, keep alpha=0.10 as seed-variable, and put the BN/GN comparison in an
      appendix. Either way GroupNorm becomes the primary backbone for the paper
      (BN is not defensible as the FL default).
- Update T1, F6, and the headline accordingly.

PRIORITY 3 — STAGE 4 (strengthen; mostly CPU/medium GPU).
- H2 split-leakage (cheap, high value): deliberately reuse test labels in the
  proposal/cert fold and show the guarantee BREAKS (empirical accepted risk > alpha
  beyond delta). This proves split hygiene is load-bearing. Figure/row.
- Corruption curve: sym/asym noise rate in {0,0.1,0.2,0.35,0.5} x d in {0.5,5}
  (GroupNorm) -> CertCov@alpha vs noise rate; shows the law's corruption axis.
- P5 self-training real (F8): run_selftrain_cifar in the feasible regime (d=5,
  GroupNorm), T rounds with round-wise disjoint audit folds (Prop 4). Plot per-round
  contamination (certified<=alpha vs naive unbounded) + downstream accuracy
  (certified vs naive vs none). HONEST headline: "prevents catastrophic
  contamination while retaining useful pseudo-labels" — NOT a guaranteed accuracy
  gain. Generate figs/F8.
- (H5 client-subsample, H6 unknown-proportion: optional, only if time remains.)

REPORT after each PRIORITY in the fixed format (진단/명령/핵심결과 mean+/-std/
판정/다음행동). After PRIORITY 2 state whether GroupNorm strengthens alpha=0.10.
Stop-and-ask before exceeding GPU budget. Update README; the draft (Fed-CORE_draft.md)
is on the Mac side — output the exact T1/headline deltas so they can be synced.

RULES: never use test labels in proposal/cert (except the labeled H2 leakage row);
grouping public/fixed; fixed-MSP main, best-of-N appendix; median cert_ucb; no faked
crossings; G=1 pooled stays a near-IID-only bonus.
```

---

### Notes for Sanghoon
- **PRIORITY 2 (GroupNorm) is the highest-value remaining run**: it attacks the two
  coupled weaknesses (BN-in-FL objection + α=0.1 seed variance) at once, and could
  upgrade the α=0.10 headline. Do it before STAGE 4.
- PRIORITY 1 is pure correctness/clarity (median ucb, fixed-MSP main) and is cheap —
  it should land regardless.
- The draft on the Mac is already updated with the BN T1; when GroupNorm numbers come
  back, paste the new T1 rows + the GN/BN verdict and I'll finalize §5.1.
