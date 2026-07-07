# PROMPT — Publication-sufficiency expansion experiments (Ubuntu 4070)

Read `CLAUDE.md` and `AGENTS.md` first and follow them. Docker-first for
anything touching torch; CPU tasks may run on the host. This prompt fills the
experimental gaps identified in the pre-submission sufficiency audit
(2026-07-07). The manuscript lives on the laptop; produce DATA ARTIFACTS ONLY
(CSV under `runs/`, scripts committed), plus the fixed-format report per task.

Current paper numbering for orientation: Theorem 3 = feasibility law;
Table 5 = master real-data diagnostics (a: FedAvg baseline / b: detectors /
c: edge); Figures 4-5 = feasibility/stress.

## Why these experiments (the gaps)

G1. Every real experiment uses J=5 clients. A federated-certification paper
    with no real-data evidence beyond five clients is exposed to a one-line
    rejection ("not federated scale"). Theorem 3 predicts exactly how the
    certificate must degrade and how grouping must rescue it as J grows —
    this is a testable prediction we have only verified synthetically.
G2. The only stable positive dataset is CIFAR-10. CIFAR-100 appears in the
    paper as a stress domain on ONE seed with the weak SimpleCNN backbone;
    citing it in the Abstract on that basis is fragile.
G3. Detector cells (FedPD, FOOGD) rest on 3 seeds; FOOGD-SAG on 1; the
    self-training gain on 3. Means±std over 3 seeds invite attack.
G4. The deployment knobs the deployer must choose — box radius rho and
    proposal buffer gamma — have a stated protocol but no real-data
    sensitivity evidence.
G5. The corruption curve (Fig. 5b) uses single fixed configurations.

## Task E1 (P0) — Client scaling on real CIFAR-10  [GPU]

For J in {10, 20} (J=5 exists), ResNet-18-GN, d=0.5, clean, seeds {0,1,2}:

```bash
python experiments/fedcore/run_cifar.py --dataset cifar10 --n_known 6 \
  --n_clients J --dirichlet_alpha 0.5 --rounds 50 --local_epochs 2 \
  --alpha 0.10 --delta 0.10 --seed S --out runs/cifar10_J{J}_gn_seed{S}.csv
```

(--backbone flag: use whatever selects resnet18gn in this repo; export logits
npz as usual.) Then compute, per seed and per alpha in {0.10, 0.20}, the
grouped certificate at G in {J, J/2, 5, 2} (contiguous pre-declared groups)
and write `runs/client_scaling.csv` with schema
`J,seed,alpha,G,cert_risk_ucb,cert_coverage_lcb,cert_n_min_group,certified,test_risk,test_coverage`.

Acceptance: the Theorem-3 pattern on real data — per-client (G=J) bounds
degrade with J at fixed total data, and grouping restores certification;
report where CertCov@0.20 survives. Also run ONE partial-participation
variant if the trainer supports client sampling (participation 0.5 at J=20,
seed 0); if it does not, say so and skip — do not hack the trainer.

## Task E2 (P0) — CIFAR-100 with the headline backbone  [GPU]

ResNet-18-GN, n_known 60, d in {0.5, 5}, clean, seeds {0,1,2}:

```bash
python experiments/fedcore/run_cifar.py --dataset cifar100 --n_known 60 \
  --n_clients 5 --dirichlet_alpha D --rounds 50 --local_epochs 2 \
  --alpha 0.10 --delta 0.10 --seed S
```

Evaluate grouped G=2 at alpha in {0.10, 0.20}; append rows to
`runs/T9_diagnostics.csv` (same schema) and write a per-cell summary
`runs/cifar100_gn.csv`. Either outcome is publishable: a positive at
alpha=0.20 upgrades CIFAR-100 from stress domain to second positive; a clean
3-seed negative replaces the current single-seed SimpleCNN evidence.

## Task E3 (P1) — Seed extensions  [GPU; supersedes previous Task E]

1. FedPD-PROSER seeds {3,4} at d in {0.5, 5} (pretrain-then-finetune recipe),
   alpha in {0.10, 0.20} -> extend `runs/T8_fedosr_bases.csv` and
   `runs/T9_diagnostics.csv`.
2. FOOGD-SM3D seeds {3,4} at d=5, alpha=0.20 -> same files.
3. Self-training 4x-budget cell (FedPD base, one-shot delta) seeds {3,4} ->
   `runs/selftrain_gain_5seed.csv`
   (seed,certified_gain,oracle_gain,admitted,max_contamination).

## Task E4 (P1) — Deployment-knob sensitivity  [CPU, stored logits]

On the stored GN d=5 and d=0.5 logits (5 seeds), alpha in {0.10, 0.20}:

1. Box-radius sweep: rho in {0.05, 0.10, 0.15, 0.25, simplex} ->
   `runs/rho_sensitivity.csv`
   (rho,seed,d,alpha,cert_risk_ucb,cert_coverage_lcb,certified).
   Question answered: how much coverage does a wider (safer) box cost, and
   is validity ever endangered inside the box (it must not be).
2. Gamma ablation: fix gamma in {0.5, 0.7, 1.0} (no selection) ->
   `runs/gamma_ablation.csv` (gamma,seed,d,alpha,cert_n,cert_risk_ucb,
   cert_coverage_lcb,certified,test_risk). Question answered: the real-data
   trade-off behind the buffer (gamma=1.0 richer accepted set but boundary
   risk; gamma=0.5 starves counts) — complements the resampling finding that
   56/61 violations sat at gamma=1.0.

## Task E5 (P2) — Corruption curve, seeded  [GPU]

Extend Fig. 5b cells to 3 seeds: symmetric rates {0.1, 0.2, 0.35} at
d in {0.5, 5}, ResNet-18-GN -> `runs/corruption_curve_3seed.csv`
(noise_type,rate,d,seed,CertCov@0.10,CertCov@0.20,test_risk).

## Task E6 (P1) — One client-simplex (or small-J) deep positive  [GPU]

The main real-data positives are grouped certificates under A6. One
full-simplex deep positive removes the "grouping = hidden pooling" attack.
Attempt, in order, stopping at the first non-vacuous cell:
1. CIFAR-10 / FedPD-PROSER / J=5 / full simplex (Theorem 1) / alpha=0.20,
   with an enlarged audit budget (cert_frac 0.5 or the 4x protocol).
2. Same with J=3 (retrain FedPD at J=3, seeds {0,1,2}).
Write `runs/simplex_positive.csv`
(J,seed,alpha,cert_risk_ucb,cert_coverage_lcb,cert_n_min_client,certified).
A 2/3-or-better cell is reportable; a clean failure is also informative —
report it as the measured price of client-simplex robustness.

## Task E7 (P1) — covtype as a stable second positive  [GPU/CPU]

Goal: 4/5 or 5/5 non-vacuous under a fully A2-compliant protocol.
Proposal-fold score selection (not fixed MSP), enlarged audit budget, and if
helpful a slightly stronger tabular backbone (wider MLP). alpha in
{0.20, 0.25, 0.30}, delta=0.10, grouped G=2, seeds {0..4} ->
`runs/covtype_stable.csv` (same schema as T9). If 5-seed stability is not
reachable, report the best honest cell; the paper keeps covtype as a
feasibility-edge domain in that case.

## Task E8 (P0) — Detector diagnostics reconciliation  [CPU]

T9_diagnostics.csv (54 rows) disagrees with the published detector cells:
FedPD d=5 alpha=0.20 mean cert_coverage_lcb 0.4912 vs T8's 0.483, and FOOGD
d=5 0.3498 vs T8's 0.071 (5x). Identify the protocol difference (gamma grid?
fold definition? representative-head vs native score? coverage quantity?).
Deliver ONE consistent protocol for panel (b): regenerate detector rows of
T9 under exactly the protocol that produced T8's published numbers, or
document why T8's FOOGD number is a different quantity and state which is
correct for the paper. Until resolved, the manuscript keeps panel (b)
without T9 diagnostics. Also regenerate the full T9 at the simultaneous
budget delta_r=delta_c=delta/2 so a future headline switch is drop-in.

## Priorities and budget

E8 > E1 > E2 > E3 > E6 > E7 > E4 > E5 (E8 and E4 are CPU). E4 is CPU and can run anytime. If GPU time is short,
E1 (J=20 only) + E2 already remove the two biggest attack surfaces.

## Guardrails

- proposal/certification/test split hygiene is inviolable.
- corruption goes on TRAIN labels only; calibration folds stay clean.
- contiguous pre-declared grouping; state the rule in every CSV header
  comment.
- never commit runs/, data/, weights; DO commit new scripts.
- report per task in the fixed format (진단 요약 / 확인한 명령 / 핵심 결과 /
  판정 / 다음 행동); report failed commands as failed; no silent retries.
- do NOT edit any manuscript text; data artifacts only.
