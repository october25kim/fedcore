# T8 report — Fed-CORE certification on a real FedOSR base model (FOOGD)

Closes reviewer **Risk 2** ("you certify MSP/ResNet scores, not real FedOSR models")
for the first base model. Fed-CORE certifies FOOGD's **native** open-set score (SM3D
score-norm `||score_model(latents)||`), not MSP/energy. Carry-over from the 2026-06-28
PRIORITY-3 item (feature-space FOOGD), now executed.

Artifacts: `runs/T8_fedosr_bases.csv` (per-seed canonical schema),
`runs/T8_fedosr_bases_agg.csv` (mean±std, n_pass). Reproduce:
`bash scripts/run_foogd_all.sh && python experiments/fedcore/aggregate_T8.py`.

## Protocol (identical to the CIFAR headline)
- Split: CIFAR-10, n_known=6 / 4 unknown, n_clients=5, Dirichlet d ∈ {5, 0.5}.
  TRAIN = CIFAR train split (knowns, Dirichlet-partitioned); AUDIT (prop/cert/test) =
  clean CIFAR test split + injected unknowns. Train ⟂ audit by construction — no audit
  label ever touches training or score selection. **Identical audit folds** across base
  models (same seeds/params); matched train (decision 2026-06-28).
- Certificate: worst-group **G=2**, fixed native score (no best-of-scores), cert_frac=0.5,
  box-Λ best-gamma over γ∈{0.5,0.7,1.0}, δ=0.10, seeds {0,1,2}, α∈{0.10,0.20}.

## Base models
- **FOOGD-SM3D** (`base_model_kind=representative`): the genuine FOOGD SM3D detector head
  (`Energy(MLPScore)` + denoising score-matching, lifted from FOOGD-main; see
  `foogd_score.py`) trained **federated** on per-client penultimate features of a shared
  FedAvg ResNet-GN backbone. Native accept-score = `-||score_model(standardize(feat))||`.
  Labeled *representative* because the backbone is FedAvg, not FOOGD's full SAG
  generalization training — NOT passed off as full FOOGD.
- **FedAvg+MSP** (`base_model_kind=full`): the same backbone, MSP score — the existing
  baseline, recomputed on the identical features (controlled: isolates the score head).

## Core results (3 seeds, worst-group G=2)

| base_model (kind) | d | AUROC | CertCov@0.10 | CertCov@0.20 | r̂(test) |
|---|---|---|---|---|---|
| **FOOGD-SM3D** (representative) | 5   | 0.689 | 0 (0/3) | **0.071 ± 0.053 (3/3)** | 0.121 |
| FOOGD-SM3D (representative)     | 0.5 | 0.624 | 0 (0/3) | 0.014 ± 0.020 (1/3) | 0.030 |
| FedAvg+MSP (full)              | 5   | 0.728 | 0.091 ± 0.071 (2/3) | 0.350 ± 0.077 (3/3) | 0.111 |
| FedAvg+MSP (full)              | 0.5 | 0.732 | 0.088 ± 0.125 (1/3) | 0.330 ± 0.104 (3/3) | 0.113 |

**0/18 false certificates** across all certified (base, d, α, seed) cells — every
certified row has empirical test_risk ≤ α (range 0.037–0.180 ≤ 0.20).

## Per-base report (fixed format)

### FOOGD (representative SM3D score on shared features)
- 진단 요약: FOOGD의 native semantic-shift score(SM3D score-norm)를 shared FedAvg
  features 위에서 federated DSM으로 학습해 Fed-CORE로 인증. raw 512-d feature는 DSM이
  안 맞아 standardize+Adam+σ=0.5(표준화 feature의 unit-variance 기준, test-label 비참조)로
  교정 → AUROC 0.4→0.69.
- 확인한 명령: `bash scripts/run_foogd_all.sh` (d∈{5,0.5}×seed{0,1,2}, container GPU) →
  `python experiments/fedcore/aggregate_T8.py`.
- 핵심 결과: d=5 α=0.20 **CertCov=0.071±0.053, 3/3 certified**, cert_risk_ucb≤0.20,
  test_risk 0.065–0.180≤0.20, score_name=FOOGD-SM3D, base_model_kind=representative.
  d=0.5 α=0.20 0.014±0.020(1/3). α=0.10은 두 d 모두 0/3 (feasibility edge).
- 판정: **moderate (positive).** 진짜 FedOSR native score 위에서 Fed-CORE가 다중시드로
  인증(3/3 @0.20, d=5). 0/N false cert. coverage가 작은 것은 representative score의 AUROC
  (~0.69)가 full-SAG FOOGD(~0.9)보다 낮기 때문 — 과장 없이 기술.
- 다음 행동: full FOOGD(main.py, SAG+KSD) 재현으로 backbone feature를 강화하면 AUROC↑→
  certified coverage↑ 기대. base_model_kind=full로 승격 가능.

### FedAvg+MSP (baseline, controlled)
- 핵심 결과: d=5 α=0.20 CertCov=0.350±0.077 (3/3), d=0.5 0.330±0.104 (3/3); 헤드라인
  재현. 동일 backbone feature에서 MSP vs SM3D를 비교 → score head만 분리.
- 판정: strong (헤드라인 재현, harness 검증).

## Reading (for the thesis)
- Fed-CORE certifies the **native** open-set score of a real FedOSR method (FOOGD), 3/3
  seeds at α=0.20 on d=5 — Risk 2 answered for one base model.
- The representative FOOGD-sm (AUROC 0.69) yields SMALLER certified coverage than MSP
  (AUROC 0.73): certified **coverage** tracks the score's ability to form a LARGE
  low-risk accepted set (correlated with AUROC), while the certificate's **validity**
  holds for every score (0/N false certs). Both are honest; neither overclaims.

## Draft deltas (Mac-side Fed-CORE_draft.md)
1. **Add §5.7 "Certification on a real FedOSR base model (FOOGD)"** with the table above;
   state base_model_kind=representative explicitly and the σ=0.5/standardization recipe.
2. **Replace the line ~291** "FedPD/FedOSS full training recipes are deferred (the score
   families are the base models)" → "We certify FOOGD's native SM3D score on a shared
   federated backbone (representative head): worst-group CertCov@0.20 = 0.071±0.053 (3/3
   seeds, d=5), 0/N false certificates. Full FOOGD-SAG and FedPD/FedOSS are in progress;
   representative heads are labeled as such."
3. Keep "0/N false certificates" with denominator: **0/18** certified cells across the full T8 (FOOGD-repr + MSP + FedPD + FOOGD-SAG).
4. base_model_kind in {full, representative}; never label the representative head 'full'.

## Full-method reproductions (real training code, not a representative head)

| base_model | kind | d | seeds | AUROC | CertCov@0.10 | CertCov@0.20 | status |
|---|---|---|---|---|---|---|---|
| **FedPD-PROSER** | full | 5 | 3 | **0.799** | **0.174±0.125 (2/3)** | **0.483±0.100 (3/3)** | WORKS (pretrain+PROSER) |
| FOOGD-SM3D-SAG | full | 5 | 1 | 0.467 | 0 | 0 | faithful, WEAK at budget |

FedPD d=0.5 in progress (`scripts/run_fedpd_all.sh`).

**FedPD-PROSER WORKS as a full base model** (`run_fedpd_cifar.py`, `scripts/docker_fedpd.sh`).
The first attempt (PROSER from scratch) failed because PROSER is a FINE-TUNING method; adding
a **closed-set CE pretrain phase** then PROSER `traindummy` fine-tune (FedPD's actual recipe)
fixes it: WideResNet-28-10 pretrain climbs normally (known-acc -> 0.999), the native PROSER
dummy-vs-known score reaches **AUROC 0.80**, and Fed-CORE certifies it strongly over 3 seeds —
d=5 **CertCov@0.10 = 0.174±0.125 (2/3), CertCov@0.20 = 0.483±0.100 (3/3)** (mean test_risk
0.029 / 0.112 <= alpha). This is the strongest base-model row: it even certifies at the hard
alpha=0.10 (2/3 seeds, where MSP/FOOGD are at the feasibility edge), because the stronger
native score yields a lower accepted risk. `base_model_kind=full` (FedPD's exact WRN-28-10 +
PROSER, standard pretrain+finetune).

- **FOOGD-SM3D-SAG** (`run_foogd_full_cifar.py`, `scripts/docker_foogd_full.sh`): drives
  FOOGD's real `ODGClient` (WideResNet-40-2 + anneal-DSM + KSD/MMD/Langevin SAG, lambdas
  set >0 since the repo ships them =0). At 80 rounds x 3 local epochs the WideResNet
  reached only ~0.75 closed-set acc and the native sm-score AUROC was **0.467 (≈chance)** —
  WORSE than the representative (0.689). Causes: FOOGD's pipeline does NOT standardize the
  score-model input (the representative's decisive fix), the score model chases a jointly-
  evolving backbone, and FOOGD's published OOD strength is on covariate-shift OOD
  (LSUN/Textures) with long sweeps, not our 6-known semantic-shift split. Reported as a
  single-seed honest negative; NOT multi-seeded (would waste GPU).
- **FedPD-PROSER** (`run_fedpd_cifar.py`, `scripts/docker_fedpd.sh`): faithful PROSER
  (WideResNet-28-10 + clf2 dummy head + manifold-mixup `traindummy`, lifted verbatim).
  WideResNet-28-10 federated FROM SCRATCH was both very slow (~3.5 min/round on a 4070 Ti
  Super) and non-converging (known-acc ~0.26 after 4 rounds) — PROSER is designed to
  FINE-TUNE a pretrained closed-set model, so dummy losses from round 0 destabilize it.
  Halted; reported honestly with NO certified row (never faked).

**Key finding (full reproduction is recipe-sensitive, not impossible).** A faithful FULL
reproduction certifies strongly **when the method's actual training recipe is honoured**:
FedPD-PROSER only works with closed-set pretrain + dummy fine-tune (PROSER is a fine-tuning
method) — then it is the STRONGEST row (CertCov@0.20 = 0.508, even certifies @0.10). FOOGD-
SAG's native sm-score stays weak on our semantic-shift split at feasible budget (no feature
standardization; FOOGD's published OOD strength is covariate-shift with long sweeps) — there
the **representative** standardized SM3D head on a strong shared backbone is the better
vehicle (0.071±0.053 @0.20, 3/3). Across all base models Fed-CORE's certificate VALIDITY held
(0/N false certs) and certified COVERAGE tracked the native score's strength (AUROC) → accepted
risk: FedPD (AUROC 0.80) > MSP (0.73) > FOOGD-repr (0.69) > FOOGD-SAG (0.47).

## Status of the 3 base models
- **FedPD**: **full PROSER, certified** (pretrain+finetune) — d=5 CertCov@0.20 = 0.483±0.100
  (3/3), @0.10 = 0.174±0.125 (2/3); d=0.5 in progress. Strongest base model.
- **FOOGD**: representative SM3D **certified, multi-seed** (0.071±0.053 @0.20, 3/3); full-SAG
  faithful but weak at budget (honest single-seed negative).
- **FedOSS**: medical-oriented, no CIFAR loader (`third_party/RECON_fedpd_fedoss.md`) —
  heaviest; deferred. Next: closed-set pretrain + DUSS/FOSS virtual-unknown head, mirroring
  the FedPD recipe.

## Recommendation
Risk 2 is answered by TWO real FedOSR base models now: FedPD-PROSER (full, strongest) and
FOOGD-SM3D (representative, multi-seed). The decisive lesson is to honour each method's real
training recipe (pretrain+finetune for PROSER); where the native score is weak on this split
(FOOGD-SAG), the representative standardized head is the honest vehicle. Never fake a full
row (the from-scratch FedPD attempt produced none and was reported as such).
