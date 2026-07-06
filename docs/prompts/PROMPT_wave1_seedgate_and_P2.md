# Claude Code prompt — Wave 1: de-risk the headline (seed gate) + finish P2 CPU assets

Paste the fenced block into Claude Code on the 4070. This MUST run before committing
GPU to the full P6 ladder: it checks whether the worst-group α=0.1 crossover (seed0
only) replicates across seeds, and completes the cheap CPU paper assets.

```text
CONTEXT (read CLAUDE.md + AGENTS.md first). EXP1 produced a headline positive:
with ResNet-18 (CIFAR stem + GroupNorm) at d=5, lower realized risk r_hat let the
worst-group (G=2) grouped-stratified certificate cross alpha=0.10 — BUT only for
seed0, and only with cert_frac>=0.33 (i.e. the crossover is the COMBINATION of two
Theorem-2 levers: lower r_hat AND larger per-group calibration, not backbone alone).
Before investing GPU in the full ResNet ladder, (A) verify the crossover replicates
across seeds, and (B) finish the cheap CPU paper assets. Docker-first. Never use
test labels in proposal/cert. Do not fake any crossing.

============================================================
STEP A — SEED GATE (decides the headline; cheap, do first)
============================================================
Re-run cifar10 d=5 clean with ResNet-18 (CIFAR stem, GroupNorm) for seeds {1,2}
using the SAME config that produced the seed0 crossover (record that config exactly:
rounds, cert_frac, box-Lambda, proxy margin, gamma grid). Ensure the cert/test
raw-logit npz is EXPORTED for every seed (seed0's clean npz was missing before —
fix that so all of {0,1,2} have npz).

For each seed, run exp_feasibility_lever at alpha=0.10 (box-Lambda, grouped G in
{5,3,2,1}, certify_best_gamma with proxy margin) at the crossover cert_frac. Then
aggregate across seeds {0,1,2}:
  report mean +/- std of cert_risk_ucb and CertifiedCoverage@0.1 for G=2 and G=3,
  and the per-group accepted count.

GATE / JUDGEMENT (be explicit):
  - PASS  (headline holds): G=2 (or at least G=3) has cert_risk_ucb <= 0.10 and
    CertCov@0.1 > 0 in >=2 of 3 seeds, with mean CertCov reported +/- std.
    => the paper's headline is "worst-group alpha=0.1 certified at CIFAR scale
       (ResNet + cert_frac>=0.33), CertCov = X% +/- Y%". Recommend proceeding to
       P3 + P6.
  - FAIL  (single-seed fluke): if it does not replicate, DO NOT chase alpha=0.1.
    Reframe the headline to the feasibility LAW + the alpha=0.20 worst-group
    positive (already solid) and report alpha=0.1 as "at the feasibility edge,
    achievable only at the most favorable seed/cert_frac". Either way report the
    exact numbers; never fake the crossing.
Always state the crossover cert_frac alongside the result (it is part of the claim,
not a hidden knob).

============================================================
STEP B — finish P2 CPU paper assets (parallel; cheap, no GPU)
============================================================
Produce these from existing runs (matplotlib mathtext-safe: \leq, \mathrm; NOT
\le, \rm; colorblind-safe; PDF + PNG in figs/, CSVs in runs/):
  F2 necessity-real : unsafe-deploy rate (naive empirical vs Fed-CORE) on real
                      logits, across cells; highlight the cells where naive > delta.
  T1 main results   : from runs/agg_main.csv, mean +/- std of cert_risk_ucb,
                      CertCov@alpha, test_risk, test_coverage per (dataset,d,
                      corruption,alpha); SimpleCNN and ResNet rows.
  T2 efficiency     : conditional vs mass-ratio vs box vs pooled (median U, valid?).
  T3 necessity      : {naive-empirical, federated-CP recast, pooled-CP, Fed-CORE} x
                      (unsafe-deploy rate, controls-right-object?, valid?).
  T4 superiority    : extend runs/T4.csv with seeds (mean +/- std).
  T5 score-agnostic : 4 scores x (test_risk, CertCov@alpha), all valid.
  T6 privacy taxonomy: pooled/stratified/grouped x (released stats, server learns,
                      validity scope) — table only (conceptual).
  T7 self-train dT  : simultaneous unsafe rate with vs without delta/T split.
  (F8 self-training is deferred to P5; note it as pending.)

============================================================
STEP C — REPORT
============================================================
Fixed format (진단 요약 / 확인한 명령 / 핵심 결과 with mean+/-std / 판정 / 다음 행동).
State the GATE outcome (PASS/FAIL) clearly, since it decides whether P6/P3 are worth
the GPU. Update README; after draft sync, update Fed-CORE_draft.md §5.1 with the
seed-aggregated crossover (and its cert_frac) and the P2 tables/figures.

RULES: proposal/cert/test disjoint; grouping public/fixed; report the missing-npz
fix and any failures; smoke-first; judge by cert_*/matched-risk, not AUROC.
```

---

### Notes for Sanghoon
- **STEP A is a gate, not busywork**: a single-seed α=0.1 crossover is not yet a
  result. ~2 short ResNet runs settle whether the headline is real before you spend
  hours on the full P6 ladder.
- The crossover being a **two-lever combination (ResNet + cert_frac)** is fine and
  on-message (it *is* the Theorem-2 feasibility law in action) — but must be stated
  as such, with the cert_frac in the claim.
- After the gate: if PASS → I'll generate the P3 (breadth) + P6 (full ResNet ladder)
  prompt; if FAIL → we reframe to the feasibility-law + α=0.20 headline (already
  strong) and skip the α=0.1 chase. If you'd instead prefer to run **everything
  autonomously now** (gate + P3 + P6 + P4 + P5 chained, with the gate guarding the
  rest), tell me and I'll produce that single chained prompt.
