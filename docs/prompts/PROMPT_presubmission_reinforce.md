# Claude Code prompt — pre-submission reinforcement (GPU must-dos + real-data ablations)

Paste the fenced block into Claude Code on the 4070. An external review (Strong GO,
8.6/10) flagged these as the must-do reinforcements before submission. The
synthetic versions of the calibration-budget and unknown-proportion ablations are
already done on the Mac side (`exp_ablation_extra.py`, Figs 9–11); this prompt
adds the *real-data* counterparts and the multi-seed strengthening. Docker-first;
no faked numbers; report in the fixed format.

```text
CONTEXT (read CLAUDE.md + AGENTS.md first). The paper's §5 is real-data-anchored;
headline = feasibility law + worst-group alpha=0.20 positive (CIFAR d5 G2 ~0.29 T4;
covtype 0.43). Two acknowledged weaknesses to close: (1) the alpha=0.20 positive is
single-config; (2) experiments use FedAvg/CNN/ResNet scores, not a real FedOSR base
model. Plus two real-data ablations to mirror the synthetic Figs 10/A5 and 9/A4.
Primary backbone = ResNet-GroupNorm. Never use test labels in proposal/cert. Judge
by cert_* / matched-risk. Report failures.

PRIORITY 1 — alpha=0.20 multi-seed (close the single-config gap).
Run cifar10 d=5, ResNet-GroupNorm, G=2 worst-group, cert_frac=0.5, seeds {0,1,2,3,4}
at alpha=0.20 (the existing logits npz can be reused if the certify path supports
alpha=0.20; otherwise re-export). Report CertCov@0.20 mean+/-std and n_pass/5, fixed
MSP. Goal: replace "single-config ~0.29" with "0.xx +/- 0.yy over 5 seeds." Do the
same for d=0.5. This is the single most important strengthening.

PRIORITY 2 — covtype multi-seed (stabilize the 2nd domain).
Re-run the covtype tabular FL pipeline for >=3 seeds; report CertCov@{0.20,0.25,0.30}
mean+/-std. Stabilizes the second-domain positive (currently 1 seed).

PRIORITY 3 — one real FedOSR base model (close the "not real FedOSR" gap).
Put Fed-CORE's certification on top of at least ONE real FedOSR-style base model:
  - easiest: implement the FOOGD score-norm detector on the trained features and
    report it as a genuine FedOSR score (not just MSP/energy); OR
  - if a FedPD / FedOSS implementation or checkpoint is available, train it and
    certify its open-set score.
Report: same base model, Fed-CORE certifies CertCov@alpha on its score. If full
FedPD/FedOSS training is infeasible in budget, state that explicitly and label the
score heads "representative FedOSR-style scores" — do not overclaim.

PRIORITY 4 — real-data ablations (post-hoc on exported logits; mirror Figs 9 & 10).
  (A4-real) calibration-budget sweep: vary the trusted/cert-fold SIZE (e.g. fraction
    of the test pool used for calibration, or subsample), holding the model fixed;
    plot CertCov@0.10 and cert_ucb vs calibration size. Question: does alpha=0.10
    become non-vacuous with a larger audit budget (as the synthetic A4 predicts)?
  (A5-real) unknown-proportion sweep: subsample the unknown-labeled points in the
    certification fold to fractions {0.25,0.5,0.75,1.0} of the deployment rate; check
    empirical coverage of the true accepted risk. Confirm the synthetic A5 finding:
    under-representing unknowns is anti-conservative.
  Save runs/ablation_calib_budget.csv, runs/ablation_unknown_prop.csv and figures.

WRITING FIXES (apply in the manuscript / report deltas for the Mac draft):
  - state "0/N false certificates across <N> runs" with the explicit denominator,
    not just "no false certificate";
  - keep "valid" as theorem-validity + "valid in all tested settings (0/N false
    certificates)" empirically — do not write "valid everywhere";
  - keep Proposition 3 subordinate (Gap 2 open); do not promote it;
  - keep self-training as contamination-control only (no accuracy-gain claim);
  - keep "distribution-free w.r.t. the audited deployment distribution."

REPORT (fixed format) after each PRIORITY; output the exact numbers (mean+/-std,
n_pass/N) so the Mac-side Fed-CORE_draft.md §5/T1/headline can be synced. Stop-and-
ask before exceeding GPU budget; PRIORITY 1 and 4 are the highest value.
```

---

### Notes for Sanghoon
- **P1 (α=0.20 5-seed) is the top item** — it converts the single-config headline
  into a seeded result, the review's #1 weakness.
- **P4 is cheap (post-hoc, no retrain)** and directly corroborates the synthetic
  Figs 9–10 on real data — high value per GPU-hour.
- **P3 (one real FedOSR base)** mainly needs *one* model to defend the "certification
  layer for FedOSR" framing; FOOGD score-norm is the cheapest honest route.
- The synthetic ablations (A4/A5/J → Figs 9/10/11) are already in the draft; P4 just
  adds the real-data mirror.
