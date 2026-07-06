# Claude Code prompt — re-run the ResNet backbone staircase (EXP1), robustly

The ResNet staircase did not run. Paste the fenced block into Claude Code on the
4070. It first DIAGNOSES the failure, then fixes the FL+ResNet pitfalls, then goes
smoke → full → staircase. Do not blind-rerun before Step 0 finds the root cause.

```text
CONTEXT (read CLAUDE.md + AGENTS.md first). EXP1 (ResNet-18 backbone, cifar10 d=5,
~80 rounds, alpha-frontier + feasibility staircase) FAILED TO RUN. Goal: find why,
fix robustly, and produce the per-backbone staircase so we can see whether a lower
realized risk r_hat pushes the worst-group (G>=2) CertCov@0.1 above zero. Docker-
first, smoke-first. Do not fake any certification.

STEP 0 — DIAGNOSE (do this first; report before re-running).
- Reproduce the failure with the exact command that was used; capture the full
  traceback / exit status / OOM message / docker error.
- Classify the root cause into one of: (i) ResNet not implemented / --backbone not
  wired in models.py & run_cifar.py; (ii) CUDA OOM; (iii) BatchNorm failure under
  non-IID FedAvg (tiny/size-1 client batches make BN stats blow up or error);
  (iv) docker image missing torchvision/scipy; (v) npz export path / schema
  mismatch that exp_feasibility_lever expects; (vi) other (state it).
- Report the cause in one line before fixing.

STEP 1 — FIX the FL+ResNet pitfalls (these are the usual culprits).
- **CIFAR-stem ResNet-18** (critical): use a CIFAR variant — first conv = 3x3,
  stride 1, padding 1, and NO initial maxpool. The stock torchvision ResNet-18
  (7x7 stride-2 + maxpool) destroys 32x32 inputs and gives near-chance accuracy
  (which would also keep r_hat high and defeat the whole point). Verify by a
  forward pass on a (2,3,32,32) tensor and by sane warmup accuracy.
- **Normalization for FL** (critical): BatchNorm is fragile under non-IID FedAvg
  (per-client running stats diverge; size-1 batches error). Add a --norm
  {bn,gn} flag and default ResNet to **GroupNorm** (e.g. 32 groups) for the FL
  runs; this is standard practice for FedAvg. Keep BN selectable for comparison.
- **OOM guards**: expose --batch_size (default 64, allow 32), optional AMP
  (torch.cuda.amp), and drop_last=False with a guard so no batch has size 1 under
  BN; if BN is used, set track_running_stats appropriately or skip size-1 batches.
- **Wire --backbone {simplecnn,resnet18}** end-to-end in models.py + run_cifar.py;
  keep SimpleCNN as the default so nothing else breaks.
- Ensure the npz export (cert/test raw logits + per-point client id, y_open) is
  written in the SAME schema exp_feasibility_lever already consumes.

STEP 2 — SMOKE before GPU time.
- CPU: instantiate resnet18(n_known) with GroupNorm, forward (2,3,32,32), assert
  output shape (2,n_known); run 1 FedAvg round on a tiny subset (CPU) to confirm
  the loop, logit export, and certify path all execute.
- GPU: 3-round run on cifar10 d=5 to confirm training decreases loss and accuracy
  rises above chance (sanity that the CIFAR stem works). Only then go full.

STEP 3 — FULL RUN.
- cifar10 d=5 clean, ResNet-18 + GroupNorm, ~80-100 rounds, export logits npz.
- Record the operating-point realized risk r_hat (= test_risk among accepted at the
  certified-coverage-maximizing selector) and the closed-set test accuracy.

STEP 4 — STAIRCASE + the key comparison.
- Run exp_feasibility_lever on the ResNet npz: grouped-stratified G in {5,3,2,1},
  box-Lambda, certify_best_gamma with the proxy safety margin, at alpha=0.10
  (also record the alpha-frontier {0.10,0.15,0.20,0.25}).
- Build the per-backbone comparison: SimpleCNN vs ResNet-18 ->
  table of (r_hat, per-group cert_n, cert_risk_ucb, CertCov@0.1) for G=5,3,2,1,
  and overlay the required accepted count z^2 r_hat(1-r_hat)/(alpha-r_hat)^2 to show
  how lower r_hat shrinks the requirement. Save figs/fig_backbone_staircase.pdf.

STEP 5 — REPORT (fixed format) and JUDGE honestly.
- Confirm r_hat actually dropped vs SimpleCNN (check the ACCEPTED-set test_risk,
  not just accuracy — unknown leakage, not closed-set acc, drives r_hat).
- Does worst-group G=2 cert_risk_ucb cross alpha=0.10 with CertCov>0?
    * If YES -> this is the headline worst-group positive; note CertCov and r_hat.
    * If NO  -> report the exact available vs required per-group cert_n, then try
      the combination (ResNet + larger cert_frac / G=2 vs G=3); if still short,
      state "ResNet lowers r_hat and raises coverage but worst-group alpha=0.1
      needs more per-group calibration", and quantify the residual gap. Do NOT
      fake a crossing; G=1 pooled stays a near-IID-only bonus.
- Watch for overfitting (80-100 rounds ResNet under non-IID): if proposal-proxy
  becomes optimistic vs the cert fold, keep/raise the proxy margin; judge only on
  cert/test folds.

RULES: proposal/cert/test disjoint; grouping public/fixed; never use test labels in
proposal/cert; report the original failure and any further failures; smoke-first;
update README + (after draft sync) Fed-CORE_draft.md §5.1 with the backbone row.
```

---

### Notes for Sanghoon
- The two most likely root causes are **(a) stock torchvision ResNet stem on 32×32**
  (kills accuracy → r̂ stays high) and **(b) BatchNorm under non-IID FedAvg**. The
  prompt fixes both up front (CIFAR stem + GroupNorm), so even if Step 0 finds a
  different error, the run will be on a correct FL-ResNet.
- The decisive readout is **r̂ (accepted-set test_risk), not accuracy** — that is
  what moves the (α−r̂)⁻² requirement and decides the G=2 crossover.
- If ResNet still doesn't cross worst-group α=0.1, that is a fine honest outcome:
  the paper already stands on the feasibility law + T4; ResNet is upside.
