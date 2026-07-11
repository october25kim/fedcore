# Fed-CORE Draft — Critical Review (2026-07-10)

대상: `docs/Fed-CORE_draft.md` (55pp 렌더링 기준, 14,876 words, 그림 5, 표 7, 참고문헌 43).
목적: (1) 섹션별 논리 비판, (2) 실험 목적·해석 정리, (3) 45pp 감축 계획, (4) reference 재번호 진단.
판정 기준: 저널(Information Sciences 계열) major revision을 견딜 수 있는가.

**총평: moderate go.** 이론의 뼈대(conditional CP 인증서 + 비환원성 + feasibility law)는 견고하고
정직성(honest null 보고)이 강점이다. 그러나 (a) 약속-실행 불일치 2건(§2 oracle, §5.1 re-cast
baseline), (b) 그림-본문 불일치 2건(Fig 3, Fig 4c), (c) 심한 서술 중복(quota-gap 3회, "in one
sentence" 2회, R_sel 정의 2회)이 리뷰어 신뢰를 깎는다. 아래 순서로 고치면 45pp와 설득력을 동시에
얻는다.

---

## 1. 섹션별 비판

### Abstract

* **과밀.** 단일 문단 ~340 words. "Pooling accepted calibration points … stratified calibration"
  문장은 60+ words로 abstract에서 Poisson-binomial 기제까지 설명하려 한다. 기제는 본문 몫이다.
* **결과 나열 과다.** FedPD-PROSER의 α 값 2개, 데이터셋 3개, full-simplex 확인까지 전부 나열.
  abstract는 (문제 → 왜 기존 방법 불가 → 방법 → 핵심 결과 1–2개 → feasibility law)로 충분.
* 마지막 문장 "All results are finite-sample feasibility demonstrations, not safety
  recommendations."는 정직하지만 abstract 끝을 방어로 닫는다. Limitations로 이동 권장.

### §1 Introduction

* **약속의 조건 누락.** "This paper provides exactly that promise."는 A1–A6 조건부다. 병원 서사
  직후 한 절로 "under an audited monitoring set and stated assumptions"를 붙이지 않으면
  overclaim으로 읽힌다.
* **중복.** R_sel(λ) 식이 §1과 §3에 동일하게 두 번 등장. "Why this is hard" 문단은 §4.4
  Proposition 2와 실질 중복. §1은 직관 1–2문장으로 줄이고 formal 정의·기제는 §3/§4.4에 위임.
* **기여 bullet 과대.** 두 번째 bullet은 ~120 words에 Thm 1–3, Cor 1, Prop 2–3까지 다 담았다.
  bullet 하나당 주장 하나가 원칙. 4개 bullet 각 2문장 이내로 재구성 필요.
* 문헌 나열(three literatures)은 논리 전개가 좋다. 유지.

### §2 Related Work

* **인용 뭉치기(citation stuffing).** "Adjacent lines addressed individual facets …
  [27,29,30], [28,33], [31,32], [34,35]" — 8편을 한 문장에 나열하고 개별 engagement가 없다.
  Inf. Sci. 커스텀 인용으로 보이며 리뷰어가 가장 먼저 지적하는 패턴. 각 묶음에 반 문장씩의
  관련성 서술을 남기고 나머지는 삭제하거나, 아예 한 문장으로 압축.
* **약속-실행 불일치 #1.** "we use these methods as **centralized oracles** (upper bounds)" —
  §5에서 실제로 [16,17,38]을 oracle로 실행하지 않는다. §5.5의 oracle은 *test-peeking oracle*이지
  centralized 방법 실행이 아니다. → "positioned conceptually"로 통일하고 "used as oracles" 표현
  삭제 (§5.1과 함께 수정).
* **"In one sentence" 2회.** §1 끝과 §2 Positioning에 같은 수사가 반복. 하나만 유지.
* Table 1은 좋다. 유지.

### §3 Problem Setup

* **Proposition 1 증명의 엄밀성.** two-world 구성은 표준이고 방향은 맞다. 다만:
  (i) "the procedure deploys with probability p₀" — 관측 분포가 같으므로 deploy *확률*이 같다는
  주장인데, "observable input is identical"은 실현값이 아니라 **분포의 동일성**으로 서술해야
  정확하다. (ii) randomized procedure까지 커버함을 명시. (iii) population statement에서 조건부
  라벨분포 flip의 측도론적 정당화가 한 줄 필요. → 본문에서는 "Proof sketch"로 명명하거나 두 문장
  보강. 주장 자체(신뢰 라벨 필요성)는 옳다.
* **Dangling reference.** risk-buffered proposal이 "inherited from the centralized framework"
  — 폐기된 SRCC를 가리키는 유령 참조다. 인용도 없다. → "a standard device in selective risk
  control"로 바꾸거나 자기completeness 있게 서술.
* A4/A4′ 구분과 Table 2는 이 논문의 정직성 자산. 유지.

### §4 Method and Theory

* **Theorem 1 (conditional).** K_j|A_j ~ Bin(A_j, r_j) + tower property 논증은 옳다.
  acceptance-slack 없는 conditional 형태가 mass-ratio(m̄/a̲)형을 지배한다는 판단도 맞다
  (프로젝트 지침의 Theorem 1은 mass-ratio형이었으나 draft의 conditional형이 사양상 우월 —
  의도된 진화로 확인). 비판점: max_j r̄_j의 worst-client 보수성은 §4.2, §4.3, §6에서 세 번
  언급된다. 한 번이면 된다.
* **Theorem 2.** 분자·분모에 같은 a를 쓰는 sup 정식화와 vertex 논증(pseudolinearity) 유효.
  ε=δ/3J의 3-event 분해도 명확. 다만 "worst-client domination is avoided"는 Λ가 좁을 때만
  참이므로 조건부 서술로 완화 필요 (Λ=simplex면 Thm 1과 동일해짐).
* **Theorem 3 converse.** minimax 하한 논증은 스케치 수준이다: deploy 결정이 (A,K)의 함수라는
  가정, r=α+ε 대립가설에서의 LR 논증이 축약돼 있다. 지금처럼 본문 유지하되 "sketch of a minimax
  argument"로 라벨링하는 것이 정직하다. (결론 자체는 옳다: floor는 CP의 느슨함이 아니라
  정보이론적.)
* **Proposition 3 scope + quota-gap 서술 3중복.** 같은 내용(quota sampling ⇒ 그룹 카운트가
  conditionally Poisson-binomial ⇒ exactness 미주장 ⇒ 3중 실증 evidence)이 §4.6(Prop 3 scope),
  §5.5(headline 문단), §6(privacy 문단)에 반복. **§4.6에 한 번만** 완전히 서술하고 나머지는
  1문장 cross-reference로. 최대 감축 포인트 중 하나.
* **DP ablation 수치가 prose에 매몰.** §4.6의 Laplace-noise 수치 나열(0.90/0.90/0.77, 0.295→
  0.234, 0.40→0.001)은 문장으로 읽기 어렵다. 형식 이론이 없는 부가 결과이므로 2문장으로 압축
  (구체 수치는 절반만 유지).
* **Remark 1 (pooled diagnostic).** Lemma-L 의존성을 이론으로 격상하지 않은 것: 프로젝트 지침
  ("Theorem 3 pooled를 격상하지 말 것")과 일치. 올바른 처리.

### §5 Experiments

* **약속-실행 불일치 #2 (가장 중요).** §5.1: "we compare against re-cast nearest methods
  (federated CP [9], decentralized novelty FDR [18])" — **§5 어디에도 이 두 방법의 실행 결과가
  없다.** Table 4는 naive/leaked/pooled/Fed-CORE 뿐이다. 리뷰어가 "어디에 있나"라고 물으면
  답이 없다. → 문장을 "nearest methods certify different functionals and are contrasted
  conceptually (Table 1); the operative invalid alternative under our object — pooled CP — is
  evaluated directly"로 정정. (또는 FCP-recast 실험을 실제로 추가해야 하는데, 이는 major
  revision 대응 카드로 남겨두는 것이 낫다.)
* **§5.1 한 문단 과밀.** setup 전체가 한 문단(~430 words). 데이터셋/모델/프로토콜/인증서
  파라미터로 4문단 분할하면 오히려 줄이면서 읽힌다.
* **§5.3 그림-본문 불일치.** 본문·캡션은 synthetic 스터디(deployment fraction 0.06, 3,000
  draws, coverage 0.522/0.913/0.992)를 서술하는데 실제 그림(ablation_unknown_prop.png)은
  real-logit ρ-sweep(deployment 0.30, coverage 0.005→1.0)이다. **캡션이 데이터를 잘못
  설명하는 상태.** → 2-panel로 재작도: (a) synthetic sweep, (b) real-logit ρ-sweep.
* **§5.4 Figure 4(c) 불일치.** 본문: "CertifiedCoverage@0.10 rising from 0 to ≈0.21". 그림
  (c): 0.03→0.12 부근에서 평탄, 0.21 없음. `ablation_calib_budget.csv`는 n_cert=5000에서
  CertCov 0.061(2/5 pass)이고 상단 예산은 별도 파일로 보인다. → 어느 쪽이든 **그림과 본문
  수치의 출처를 단일 CSV로 통일**하고 재작도. (수치 불일치는 리뷰어에게 치명적.)
* **§5.5 서술 과잉.** "How to read the coverage numbers" 문단은 좋은 방어지만 ~180 words이고,
  Table 5 앞 diagnostics 문단(~250 words)과 δ/2 vs full-δ 예산 비교가 얽혀 있다. 예산 비교는
  각주/캡션 1문장으로. BN/GN 헤드라인 서술 반복 압축.
* **Asymmetric corruption 미보고.** §5.1이 asym 0.20을 약속하고 asym 데이터가 runs에 존재하는데
  (corruption_curve.csv asym 행, d0.1 asym0.2 seed0-2 npz) §5.5와 Fig 5는 symmetric만 보여준다.
  → Fig 5(b)에 asym 곡선 추가 (단일 시드임을 명시) 또는 setup에서 asym 약속 삭제. 전자 권장.
* **Table 4의 α=0.05.** setup에서 "synthetic boundary study only"로 해명돼 있으나, 표 캡션에도
  한 번 더 명시하면 오해 소지 제거.
* **§5.2, 5.6은 견고.** resampling 526k evaluations + γ=1.0 집중(56/61)은 이 논문에서 가장
  설득력 있는 validity evidence다. buffer 필요성의 실증으로 §5.5 knob 문단과 연결한 것도 좋다.

### §6 Limitations

* 전반적으로 성실. 추가 권장 2개: (i) 벤치마크가 vision(CIFAR)+tabular(covtype) 소수 도메인,
  J≤20 cross-silo 규모라는 외적 타당도 한계, (ii) unknown-split 의존성(§5.5 robustness 결과)을
  한 줄로 명시적 limitation으로 승격.

### §7 Conclusion

* 결과 수치 재나열(5/5, 0.39, J=20 …)이 많다. Conclusion은 object/method/law 3문장 + future
  work로 충분. ~40% 감축 가능.

### References

* **본문 등장 순서와 번호 불일치 존재.** 확인된 예: [28]은 [29,30]보다 늦게 등장, [20]
  (Hoeffding)은 §4.6 Remark 1에서 처음 인용되는데 [39]–[43](§2 Table 1, §3)보다 앞 번호.
  → 전면 재번호 필요 (스크립트로 처리; 아래 change log 참조).
* [13,16,17,18,38] arXiv 2025–2026 항목들은 투고 시점에 출판 상태 재확인 필요 (needs
  verification — 특히 [17] arXiv:2603.24704, [18] arXiv:2605.08263, [38] arXiv:2606.08517).

---

## 2. 실험별 목적·해석 (그림/표 단위)

| 항목 | 목적 (무엇을 증명) | 읽는 법 / 해석 | 상태 |
|---|---|---|---|
| **Fig 1** (problem diagram) | 인증 대상 R_sel(λ)이 AUROC·coverage·FDR과 다른 functional임을 시각 구분 | 4개 질문 중 4번째만 Fed-CORE가 인증; fold 3개의 진입 지점 | 유지 |
| **Fig 2** (pooling collapse) | **비환원성(Prop 2)**: pooled CP는 matched mixture에서만 valid, shift에서 coverage 0 붕괴 | x=고위험 클라이언트로의 shift, y=Pr(R_sel≤Ū). stratified 평탄 ≥1−δ = Thm 1; box는 선언된 Λ 안에서만 보장(음영) | 재작도(스타일) |
| **Fig 3** (A4 stress) | **validity의 경계**: audit fold가 unknown을 deployment보다 적게 담으면 반보수적 | (a) synthetic: 0.06 기준 미달 시 coverage 0.522→0.057 붕괴, (b) real ρ-sweep: ρ<1에서 붕괴. "audit은 unknown 비율을 *추적*해야 한다" | **본문-그림 불일치 수정** |
| **Table 4** (validity/necessity) | naive threshold·leaked split·pooled 전부 unsafe rate>δ, Fed-CORE만 ≤δ | boundary regime에서 naive 0.49 = 동전던지기 수준. resampling 행: 526k 평가에서 violation 8.7e-4 ≪ δ | 유지 |
| **Fig 4** (feasibility law) | **Theorem 3**: 인증 가능성은 per-stratum accepted count가 지배 | (a) 카운트↑→Ū↓, floor 수직선. (b) grouping G↓=카운트↑→CertCov 계단식 상승. (c) audit budget 동일 lever. (d) J=10,20 스케일링: G=J vacuous → G=2 회복 | 재작도(데이터 소스 통일) |
| **Table 5** (real CIFAR-10) | 실데이터 비공허 인증 + 실패 모드 분류 | α=0.20 GN 0.269–0.273(8–9/10)이 헤드라인; med A_g≫floor 확인. (c) 블록: 각 vacuous 셀은 risk-driven/count-driven으로 태깅 — "왜 안 되나"가 결과의 일부 | 유지 |
| **Table 5(b)** (detectors) | score-agnosticism의 실전형: 좋은 detector = 더 많은 certified coverage, validity는 불변 | CertCov가 AUROC 순서(0.81>0.74>0.68>0.47)를 정확히 따름; FedPD-PROSER만 α=0.10 통과 | 유지 |
| **Table 6** (CIFAR-100) | 두 번째 실데이터 양성; 60-class에서 절대 coverage 작음의 원인 규명 | 6/6 셀 non-vacuous @0.20이지만 0.05–0.09: 모델의 accepted-error가 α에 근접해 safe margin이 얇음 (feasibility law의 (α−r̂)⁻² 항) | 유지 |
| **Fig 5** (stress axes) | feasibility law의 나머지 두 축: 이질성(d↓)과 corruption(noise↑)이 인증을 붕괴시킴 | (a) d↓→per-group count↓→bound↑. (b) **calibration이 clean이어도** train noise 0.1에서 CertCov ≈0.03, ≥0.2에서 0 — corruption은 모델 r̂ 자체를 α 위로 밀어 "인증할 안전영역"을 소멸시킴 | **asym 추가 재작도** |
| **Table 7** (self-training) | Prop 4의 δ/T 계약이 필수임을 실증 | fresh-fold 0.086≤δ vs reused-fold 0.386>δ; naive는 오염 0.19–0.67로 발산하는데 인증 게이트는 0건 admit — "정확도 부스터가 아니라 admission gate" | 유지 |
| **unknown-split robustness** | 인증 coverage의 분산 원천 분해 | split 간 SD(0.115) > seed 간 SD(0.095): 무엇을 unknown으로 두느냐가 학습 랜덤성보다 큼 — 정직하게 보고됨 | 유지 |

---

## 3. 45pp 감축 계획 (본문 압축만; 증명은 본문 유지)

| 위치 | 조치 | 예상 감축 |
|---|---|---|
| Abstract | 340→170 words | 0.3pp |
| §1 | R_sel 식 제거(§3로 위임), "why hard" 압축, bullet 재구성 | 1.0pp |
| §2 | adjacent-lines 압축, positioning 중복 제거 | 0.8pp |
| §4.2/4.3 | worst-client 보수성 언급 1회화, Thm 2 서두 압축 | 0.5pp |
| §4.6 | quota-gap 단일화, DP prose 압축, staircase 서술 압축 | 1.2pp |
| §5.1 | 문단 분할+압축, 불일치 문장 정정 | 0.5pp |
| §5.4–5.5 | 반복 서술(quota-gap, 예산 비교, BN/GN) 압축 | 2.5pp |
| §7 | 수치 재나열 제거 | 0.7pp |
| Table 5 주변 | diagnostics 문단→캡션화 | 0.7pp |
| 그림 캡션 | 본문 중복 서술 제거 | 0.3pp |
| 합계 | | **~8.5–10pp** |

## 4. 우선순위 액션 (이번 편집에서 수행)

1. §5.1 re-cast baseline 문장 정정, §2 "oracle" 표현 정정 (허위 약속 제거) — **필수**
2. Fig 3 재작도(2-panel) + 캡션 정정 — **필수**
3. Fig 4 데이터 소스 통일 재작도 + (c) 본문 수치 정합 — **필수**
4. Fig 5 재작도(asym 포함) — 권장
5. Reference 전면 재번호 — **필수**
6. 감축 편집(§3 표 참조) — **필수**
7. Prop 1 "Proof sketch" 라벨 + 2문장 보강, Thm 3 converse "sketch" 라벨 — 권장
8. SRCC 유령 참조("inherited from the centralized framework") 제거 — **필수**
