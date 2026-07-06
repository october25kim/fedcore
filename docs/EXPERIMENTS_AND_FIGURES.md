# Fed-CORE — 필수 실험 · 핵심 Figure · 핵심 Table 총정리

이 문서는 "무엇을 왜 실험하는가"를 한눈에 보기 위한 지도다. 각 실험은
**증명하려는 주장(claim) → 셋업 → 측정값 → 기대 결과 → 상태 → 파일** 순으로 적었다.
상태: ✅ CPU 검증 / 🟩 실데이터 완료 / 🟡 일부/진행 / ⬜ 미실행.

핵심 metric은 항상 **CertifiedCoverage@α** = "오답률을 α 이하로 *보증*하면서 받아들인 비율".

> **캠페인 완료 (FINAL).** 모든 핵심 실험(A1·B1·B2·C1–C4·D1·D2·E1–E3·F1·G1·G2·H2 + corruption curve)이 CPU 검증 또는 실데이터로 완료. Primary backbone = **ResNet-GroupNorm**. 헤드라인 = **feasibility law(F6) + α=0.20 worst-group robust(5/5, BN·GN, CIFAR+covtype)**; α=0.10 = feasibility-edge(2–3/5). 거짓 인증 0. 그림 **F0–F9** + 표 **T1–T7** 생성 완료. 잔여는 manuscript finalization(체크리스트는 `SUBMISSION_CHECKLIST.md`).

---

## 0. 큰 그림 — 실험은 4가지 질문에 답한다

1. **맞는가? (Validity)** — 인증서가 약속한 $1-\delta$로 risk를 진짜 통제하나?
2. **필요한가? (Necessity)** — 기존/순진한 방법으론 왜 안 되나?
3. **더 나은가? (Superiority/Efficiency)** — 되는 것들 중 제일 많이·유용하게 받아들이나?
4. **쓸모 있나? (Utilization)** — 받아들인 걸로 실제 무엇을 얻나(자동화·자기학습)?

여기에 이 논문 고유의 다섯째 축이 더 붙는다:

5. **언제 가능한가? (Feasibility law)** — certified coverage가 (위험목표 α, per-client 보정량, 이질성)에 어떻게 좌우되나? (Theorem 2)

---

## 1. 필수 실험 목록 (그룹별)

### 그룹 A — 인증서가 "맞는가" (Validity / 이론 토대)

| ID | 실험 | 증명하는 것 | 셋업 | 측정 | 기대 | 상태 | 파일 |
|---|---|---|---|---|---|---|---|
| A1 | **Validity** | 경험적 coverage $\ge 1-\delta$ | 여러 이질성·여러 score | $\Pr(R_{\rm sel}\le \bar U)$ | $\ge 0.90$ 전부 | ✅(0.982–0.999) | `exp_validity.py` |
| A2 | **Lemma L** | pooled 인증서의 근거(이항 CP가 Poisson-binomial mean에 보수적) | 990 cfg 적대 탐색 + 증명 | 최소 coverage | $\ge 1-\delta$ (0.902–0.99) | ✅ + 증명노트 | `exp_lemma_L.py`, `LEMMA_L_proof.md` |

### 그룹 B — "필요한가" (Necessity) — 이 논문 존재 이유

| ID | 실험 | 증명하는 것 | 측정 | 기대 | 상태 | 파일 |
|---|---|---|---|---|---|---|
| B1 | **인증서 필요성** | 순진한 empirical thresholding은 위험 | unsafe-deploy rate $\Pr(\text{deploy}\mid R>\alpha)$ | naive $\gg\delta$ vs Fed-CORE $\le\delta$ | ✅(naive 0.48–0.52, 실데이터 0.12) vs 0.00 | `exp_necessity.py`,`exp_necessity_real.py` |
| B2 | **비환원성 (pooling-fail)** | 그냥 합치면(pool) 무효 | mixture shift에서 R_sel coverage | pooled→0 붕괴, stratified→1.0 | ✅ | `exp_pooling_fail.py` |
| B3 | **risk≠coverage** | federated-CP는 엉뚱한 양 통제 | coverage-rule의 실제 accepted risk | 통제 실패 | 🟡(개념+harness) | `exp_superiority.py` |

### 그룹 C — "더 나은가" (Superiority / Efficiency)

| ID | 실험 | 증명하는 것 | 측정 | 기대 | 상태 | 파일 |
|---|---|---|---|---|---|---|
| C1 | **Tightness** | conditional가 mass-ratio보다 tight, box가 더 tight | median $\bar U$ | box < simplex < mass < (pooled-invalid) | ✅(0.13/0.37/0.45) | `exp_tightness.py` |
| C2 | **CertCov@α frontier** | 위험목표 대비 인증 coverage | CertCov vs α | 단조, 비공허 구간 존재 | ✅CPU + 🟩실(α=0.2→16%) | `exp_frontier.py`, `run_cifar --alpha_frontier` |
| C3 | **Matched-risk vs SOTA** | 보증 없는 SOTA를 정답지로 oracle-튜닝해도 Fed-CORE가 근접+유일 보증 | coverage @ realized risk=α | Fed-CORE ≈ oracle, 단 보증有 | ⬜(base model 필요) | `exp_superiority.py` + FedPD/FedOSS/FOOGD |
| C4 | **Price of federation** | 중앙집중 oracle과의 격차가 이질성↓에서 소멸 | gap vs heterogeneity | 0.31→0.07 | ✅ | `exp_superiority.py` |

### 그룹 D — "언제 가능한가" (Feasibility law) — 이 논문의 정직한 핵심 발견

| ID | 실험 | 증명하는 것 | 측정 | 기대 | 상태 | 파일 |
|---|---|---|---|---|---|---|
| D1 | **Heterogeneity collapse** | 이질성↑ → 인증 붕괴, Theorem-2 floor에서 교차 | certified% vs Dirichlet d | floor(≈37/client)에서 교차 | ✅(66→31→9→0) | `exp_hetero_collapse.py` |
| D2 | **Feasibility lever (★결정적)** | α=0.1 null의 근인=per-group 카운트 부족, grouped-stratified로 비공허화 | CertCov@0.1 vs per-group cert_n (+floor) | 카운트↑ → 계단식 전환 | ⬜(post-hoc, 재학습0) | `exp_feasibility_lever.py` (신규) |

### 그룹 E — 실 CIFAR 본 결과 (Main results)

| ID | 실험 | 증명하는 것 | 측정 | 상태 | 파일 |
|---|---|---|---|---|---|
| E1 | **CIFAR ladder** | 실 스케일에서의 인증 동작 + 이질성/부패 의존 | cert_risk_ucb, CertCov@α, test_risk | 🟩(12런; clean d:0.1→5 ucb 0.53→0.185, α=0.1=0) | `run_cifar.py`,`scripts/run_ladder.sh` |
| E2 | **Corruption effect** | sym35/asym20가 인증을 악화 | clean 대비 cert_ucb | 🟩(악화 확인) | 동일 |
| E3 | **CIFAR-100 확장** | 다클래스에서의 한계(undersampling) | cert_n, CertCov | 🟩(near-vacuous, 보정 필요) | 동일 |

### 그룹 F — Score-agnostic (주장 뒷받침)

| ID | 실험 | 증명하는 것 | 측정 | 기대 | 상태 | 파일 |
|---|---|---|---|---|---|---|
| F1 | **Score-agnostic** | 어떤 score든 validity 유지, coverage만 차이 | 4 score × (test_risk, CertCov) | 전부 valid | ✅(test_risk≈0.044) | `exp_score_agnostic.py` |

### 그룹 G — Utilization ("so what", 핵심 payoff)

| ID | 실험 | 증명하는 것 | 측정 | 기대 | 상태 | 파일 |
|---|---|---|---|---|---|---|
| G1 | **Safe automation** | 보증된 risk 하 자동화율 = CertCov@α; 무보증은 α 위반 or 적게 자동화 | automation_rate, realized risk | Fed-CORE 안전+더 많음 | ✅(0.66) | `exp_utilization.py` |
| G2 | **Certified self-training (Prop 4)** | accept을 FedAvg에 되먹여 안전하게 성능↑; naive는 발산 | per-round contamination, downstream acc | certified↑·오염≤α, naive 발산 | ✅CPU + 🟩실(오염 naive 0.59–0.98 vs cert 0) | `exp_self_training.py`,`self_training.py`,`run_selftrain_*.sh` |
| G3 | **Prop 4 δ/T 계약** | δ/T split이 동시 유효성에 필수 | simultaneous unsafe rate | δ/T→≤δ, 미적용→>δ | ✅(0.086 vs 0.386) | `run_selftrain_smoke.py` |

### 그룹 H — Ablations (보강)

| ID | 항목 | 증명하는 것 | 상태 |
|---|---|---|---|
| H1 | risk-buffer γ on/off (+ best-γ) | buffer 필요성, best-γ 유효성 | ✅(best-γ unsafe 0.000) |
| H2 | split vs leakage | test label 누설 시 보증 붕괴 | ⬜(권장) |
| H3 | Λ: simplex vs box vs grouped | robustness↔tightness↔privacy | 🟡(simplex/box ✅, grouped=D2) |
| H4 | small risky client stress | Theorem-2 worst-client domination | ✅(collapse 곡선) |
| H5 | client subsampling | 일부 client 참여 시 변화 | ⬜ |
| H6 | calibration unknown-proportion | unknown-labeled audit 희소성 영향 | ⬜ |

### 그룹 I — Privacy (taxonomy 주장)

| ID | 항목 | 증명하는 것 | 상태 |
|---|---|---|---|
| I1 | 누설 분류표 | pooled만 sum-only; stratified는 per-client; grouped은 절충 | 🟩(개념+D2가 grouped 실증) |

---

## 2. 핵심 Figure (리뷰어가 기억할 그림)

> 표시: ★ = 반드시 본문 / ☆ = 본문 권장 / (부록) = appendix 가능

- **F0 ★ Problem diagram (개념도, 본문 Fig.1).** 한 그림으로 **accepted selective risk** $R_{\rm sel}(\lambda)$를 (i) AUROC/FPR95(순위), (ii) federated-CP의 prediction-set coverage, (iii) batch-FDR과 **시각적으로 구분**. accept/reject 결정 + open-set 라벨 + deployment mixture를 도식화. → "우리가 통제하는 양이 무엇이고 무엇이 *아닌지*." (포지셔닝)
- **F1 ★ Non-reducibility (pooling-fail).** x=deployment mixture shift, y=R_sel coverage. naive pooled가 shift에서 **0으로 붕괴**, stratified는 $\ge 1-\delta$ 평탄. → "왜 그냥 합치면 안 되는가." (B2)
- **F2 ★ Necessity.** x=true risk(경계 근처), y=unsafe-deploy rate. naive-empirical 곡선이 δ를 크게 초과, Fed-CORE는 δ 이하. → "왜 인증서가 필요한가." (B1)
- **F3 ★ Validity.** x=이질성(Dirichlet d), y=경험적 coverage. 모든 인증서가 $1-\delta$ 위. → "맞다." (A1)
- **F4 ☆ Tightness.** 4 인증서(conditional/mass-ratio/box/pooled)의 $\bar U$ 분포(box/violin). box < simplex < mass; pooled tightest지만 invalid 표시. → "효율적이다." (C1)
- **F5 ★ Real CIFAR α-frontier.** x=α∈{.1,.15,.2,.25}, y=CertCov@α (d=5). robust 비공허: α=0.20에서 ResNet-GN **0.39±0.10(5/5)**(BN 0.43). → **양성 핵심 결과.** (C2/E1)
- **F6 ★ Feasibility law (signature).** x=per-group accepted count, y=CertCov@0.1 (또는 cert_ucb), **Theorem-2 floor 수직선** overlay. grouped-stratified로 카운트↑ → α=0.1 계단식 전환. → "honest null을 *법칙*으로." (D2)
- **F7 ★ Heterogeneity collapse.** x=Dirichlet d, y=certified% (또는 min cert_ucb), Theorem-2 경계 표시. → "언제 무너지는가." (D1/E1)
- **F8 ★ Certified self-training (G2).** x=self-training round, 좌y=per-round 오염(certified ≤α vs naive 0.19–0.67 발산), 우y=downstream accuracy. → **utilization payoff = catastrophic 오염 방지**(정확도 향상은 *주장 안 함*, honest).
- **F9 ★ Corruption curve (the-law의 corruption 축).** x=client TRAIN noise rate {0,.1,.2,.35,.5}, y=worst-group CertCov@0.20 (sym·asym, d∈{0.5,5}). noise≥0.1에서 (clean calibration이어도) **0으로 붕괴**. → "corruption이 모델을 열화시켜 인증할 안전 영역이 사라짐." (E2)
- **F10 (부록) Lemma L.** 적대 탐색 최소 coverage vs cfg; global domination 반례 + $k_\delta$ domination 그림. (A2)
- *(부록) Price of federation* — 별도 그림 대신 T4/superiority에 흡수(중앙 oracle 대비 gap). (C4)

권장 본문 6컷: **F0(problem diagram), F1(pooling), F5(α-frontier), F6(feasibility law, signature), F8(self-train)** + **F9(corruption)** 또는 F2(necessity). F7+F9가 법칙의 두 축(이질성·corruption)을 완성. (리뷰어 권고: F0·F1·F5·F6·F8이 없으면 theorem note처럼 보일 수 있음.)

---

## 3. 핵심 Table

> Venue 우선순위 (리뷰 반영): **Pattern Recognition → Information Fusion → TNNLS**, TIFS 보류.

- **T1 ★ Main CIFAR results.** 행=run(dataset×d×{clean,sym35,asym20}×seed), 열=`cert_risk_ucb, CertCov@α, cert_n, test_risk, test_coverage` (선택 score/γ*/Λ). → 본 결과 표. (E1–E3)
- **T2 ☆ Certificate efficiency.** 행=인증서(conditional/mass-ratio/box/pooled), 열=median $\bar U$, valid?(off-matched), 가정. → tightness+validity 한 표. (C1)
- **T3 ★ Necessity / comparison.** 행={naive-empirical, federated-CP recast, pooled-CP, Fed-CORE}, 열=unsafe-deploy rate, controls-right-object?, valid?. → 비교 핵심. (B1–B3)
- **T4 ★ Matched-risk vs oracle-tuned SOTA.** 행={FedPD,FedOSS,FOOGD,Fed-CORE}, 열=coverage @ realized risk=α(SOTA는 정답지 oracle 튜닝), has finite-sample guarantee?. → "보증을 거의 무비용으로." (C3)
- **T5 ☆ Score-agnostic.** 행=4 score, 열=test_risk, CertCov@α. 전부 valid. (F1)
- **T6 ☆ Privacy taxonomy.** 행={pooled, stratified, grouped-stratified}, 열=released statistics, server learns, validity scope. (I1; draft §4.4에 이미 있음)
- **T7 ☆ Self-training δ/T validity.** 행={δ/T split, no split}, 열=simultaneous unsafe rate. (G3)
- **T8 (부록) Ablation summary.** γ, split, Λ, stress 등. (H)

권장 본문 4표: **T1, T3, T4** + (T2 또는 T5).

---

## 4. 실험 → 주장 → 논문 절 매핑

| Claim | 실험 | Figure/Table | 절 |
|---|---|---|---|
| 인증서가 valid | A1 | F3 | §4.2, §5.1 |
| pooling 무효(비환원) | B2 | F1, T3 | §4.2/§4.5, §5 |
| 인증서 필요(순진법 위험) | B1 | F2, T3 | §5(necessity) |
| conditional이 main·tighter | C1 | F4, T2 | §4.2, App C |
| 비공허 인증 가능 | C2 | F5 | §5.1 |
| SOTA 대비 거의 무비용+보증 | C3 | T4 | §5(superiority) |
| feasibility law(Thm 2) | D1,D2 | F6, F7 | §4.3, §5.1 |
| score-agnostic | F1 | T5 | §4.6 |
| safe automation | G1 | (T) | §4.7A, §5 |
| certified self-training | G2,G3 | F8, T7 | §4.7B, §5 |
| privacy taxonomy | I1 | T6 | §4.4 |
| Lemma L | A2 | F10 | §4.5, App |

---

## 5. 우선순위 (저널 투고 기준)

**없으면 안 됨 (must-have):**
- A1 validity, B1 necessity, B2 pooling-fail, C2 frontier(실), D1+D2 feasibility law, E1 main CIFAR, F1 score-agnostic, G2 self-training, A2 Lemma L, G3 δ/T.
- Figures: F1, F3, F5, F6, F8. Tables: T1, T3.

**강력 권장 (reviewer 설득력↑):**
- C1 tightness, C3 matched-risk vs SOTA(T4), C4 price-of-fed, F7 collapse.
- 이 중 **C3(T4)와 D2(F6)** 가 합·불을 가르는 두 카드: 전자는 "SOTA 대비 가치", 후자는 "null을 법칙으로".

**남은 실데이터 작업(현재 기준):**
1. **D2 feasibility lever** (grouped-stratified, post-hoc — 재학습0) → F6, α=0.1 비공허화 시도.
2. **C3 matched-risk vs SOTA** (FedPD/FedOSS/FOOGD base model 학습 후 logit만 우리 인증에 투입) → T4.
3. **backbone 강화**(ResNet/more rounds) → 분리력↑ → C2/G2 강화.
4. seeds·CIFAR-100 보강, H2(leakage)·H5·H6 ablation.

---

## 6. 한 줄 요약

> 본문은 **F1(왜 합치면 안 되나)·F3(맞다)·F5(비공허 양성)·F6(feasibility 법칙)·F8(self-training 효용)** 다섯 그림과 **T1(본 결과)·T3(necessity)·T4(SOTA 대비)** 세 표로 끝낸다. 이 8개가 "valid·necessary·superior·useful, 그리고 언제 가능한지까지" 모두 말한다.
