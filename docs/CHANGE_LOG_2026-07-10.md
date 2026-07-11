# Fed-CORE Draft — Change Log (2026-07-10)

대상: `docs/Fed-CORE_draft.md` (source of truth) → `build_docx.sh` 재빌드.
비교 기준: `docs/Fed-CORE_draft_base_2026-07-10.md` (오늘자 스냅샷).
동반 문서: `docs/CRITICAL_REVIEW_2026-07-10.md` (변경의 근거가 되는 섹션별 비판).

**결과 요약: 55pp → 47pp** (LibreOffice 렌더링 기준; Word에서 ±1pp 가능),
**15,984 → 약 12,900 words (−19%)**. 정리(theorem/proposition/lemma/corollary),
증명, 실험 결과 수치는 전부 보존. 삭제된 것은 중복 서술과 군더더기뿐이다.

---

## 1. 정합성 수정 (가장 중요 — 리뷰어 신뢰 문제)

| # | 위치 | 문제 | 조치 | 이유 |
|---|---|---|---|---|
| C1 | §5.1 | "we compare against re-cast nearest methods (federated CP, decentralized novelty FDR)" — **실제로 §5에 해당 실험 없음** | "nearest methods certify different functionals … contrasted by object in Table 1; the operative invalid alternative for our functional — pooled CP — is evaluated directly"로 정정 | 약속-실행 불일치. 리뷰어가 "그 비교 어디 있나" 물으면 답이 없는 상태였음 |
| C2 | §2 | "we use these methods as **centralized oracles** (upper bounds)" — §5.5의 oracle은 test-peeking oracle이지 [16,17,38] 실행이 아님 | "positioned conceptually and are not drop-in valid baselines"로 정정 | 동일한 약속-실행 불일치 |
| C3 | §5.3 / Fig 3 | 본문·캡션은 synthetic 스터디(0.06 기준, 3,000 draws)를 서술하는데 **그림은 real-logit ρ-sweep(0.30 기준)** | 그림을 2-panel로 재생성: (a) synthetic, (b) real. 캡션·본문 재작성 | 그림-본문 불일치. synthetic 수치(0.057/0.522/0.913/0.992)는 `paper_figs_v2.py`가 seed=1로 **정확히 재현**함을 확인 |
| C4 | §5.4 / Fig 4(c) | 본문 "CertifiedCoverage@0.10 rising from 0 to ≈0.21" — `runs/ablation_calib_budget.csv`와 불일치 (실제 0 → 0.061±0.100, 2/5 seeds) | 본문 수치를 CSV 기준으로 정정, 그림 (c)도 같은 CSV에서 재생성 | 수치 불일치 제거. 그림·본문·데이터 소스 단일화 |
| C5 | §3 | risk buffer가 "inherited from the centralized framework" — 폐기된 SRCC를 가리키는 무인용 유령 참조 | "a standard safety-margin device in selective risk control"로 교체 | SRCC 흔적 제거 (프로젝트 방침) |
| C6 | §5.5/Fig 5 | §5.1이 asymmetric 20% corruption을 약속하는데 결과 미보고 | Fig 5(b)에 asym d=5 곡선(단일 seed, dashed) 추가 + 본문 1문장 | 약속-실행 정합; asym 데이터는 `runs/corruption_curve.csv`에 존재했음 |

## 2. 그림 변경 (`experiments/fedcore/figs/paper_figs_v2.py` 신규)

기존 `paper_figs.py`는 리팩토링 이전 모듈(`certificates`, `clients` 등)을 import해 실행
불가(stale)였고, figs/의 PNG 일부는 구버전 산출물이었다. v2는 현행 `fedcore` 패키지
API로 포팅했고, 모든 수치는 `runs/*.csv`, `runs/*_logits.npz`에서 재계산한다.

- **Fig 1** (fig0_problem_diagram): 유지 (v1 디자인 그대로 재생성).
- **Fig 2** (fig1_pooling_collapse): 스타일 정리 (xlabel 2행 처리, 주석 위치).
  데이터는 기존 `fig2_coverage.json` MC 캐시 그대로.
- **Fig 3** (ablation_unknown_prop): **2-panel로 재구성** (C3). (a) synthetic
  (3,000 draws/point, deployment 0.06), (b) real CIFAR-10 (5 seeds × 40 redraws,
  deployment 0.30, `runs/ablation_unknown_prop.csv`).
- **Fig 4** (F6_feasibility_law): 4-panel 재생성. (a)(b) npz에서 staircase 재계산,
  (c) `ablation_calib_budget.csv`로 교체 (C4), (d) `client_scaling.csv`에서
  J∈{10,20} 스케일링 + seed 통과율 주석.
- **Fig 5** (F7_hetero_collapse): (a) real-logit per-seed grouped bound vs d
  (캡션과 일치하도록 재생성 — 기존 PNG는 SimpleCNN 구버전), (b) corruption 축:
  `corruption_curve_seeded.csv` 10-seed 평균±sd + asym 단일-seed 곡선 (C6);
  rate-0 anchor는 Table 5 clean headline임을 캡션에 명시.

## 3. 본문 압축 (섹션별; 논리·수치 보존, 표현만 압축)

- **Abstract**: 340 → ~210 words. 결과 나열 축소, 기제 설명은 본문에 위임.
- **§1**: R_sel display 식 제거(§3와 중복; 정의는 §3 단일화), "why this is hard"
  압축, 기여 bullet 4개를 주장당 1-bullet로 재구성, FCP/centralized 문단 압축.
- **§2**: 인용 뭉치기 문단([26]–[34])을 한 문장으로 정리(인용은 전부 유지),
  Positioning의 "In one sentence" 중복 제거, FedOSR/FCP/selective-risk 문단 압축.
- **§3**: Figure-1 걸이 문단, target-risk 문단, calibration 문단 압축.
  **Prop 1 증명 → "Proof sketch"로 명명** + 분포 동일성/randomized procedure
  명시 (리뷰 문서 §1의 엄밀성 지적 반영).
- **§4**: Thm 2 서두의 "worst-client domination is avoided"를 조건부로 완화
  (Λ=simplex면 Thm 1과 일치함을 명시). Thm 3 converse를 "minimax sketch"로
  명명 + 압축. **quota-gap 서술 3중복(§4.6/§5.5/§6)을 §4.6 단일 서술 +
  cross-reference로 통합.** DP ablation prose 압축. 증명(Thm 1, Thm 2, Cor 1,
  Prop 2, Prop 3, Thm 3, Prop 4)은 **모든 논리 단계 유지, 문장만 단축.**
- **§5**: §5.1을 4문단(Data/Metric/Baselines/Parameters)으로 재구성 + C1 반영.
  §5.2–5.6 전반의 반복 서술 압축 (Backbone 문단은 Headline 문단에 병합,
  Table 5 주석 문단은 캡션 *Notes*로 이동).
- **§6**: 압축 + **External validity 항목 신설** (J≤20 cross-silo, 도메인 범위,
  unknown-split 의존성 — 리뷰 문서 지적 반영).
- **§7**: 결과 수치 재나열 제거, 3문단 유지하되 각 40% 내외 압축.

## 4. 조판 변경 (`ins_format.py`)

- 캡션 문단(Figure N. / Table N.)을 단일간격 10.5pt로 (표는 기존에 이미 10pt
  단일간격). 본문 더블스페이스·12pt·라인넘버 등 Inf. Sci. 요구 스타일은 불변.
- md 그림에 width 속성 부여 (fig0 88%, fig1 78%, Fig3/Fig5 92%).

## 5. Reference 전면 재번호

본문 첫 등장 순서 기준으로 43개 전부 재번호 (스크립트 검증: 등장 순서 단조 ✓,
목록 1–43 순서 ✓, 미인용 0 ✓). 주요 이동:

- 구 [21] FedPD++ → **[20]** (§2 첫 등장)
- 구 [20] Hoeffding → **[43]** (Remark 1에서만 인용 — 실제 첫 등장이 가장 늦음)
- 구 [28] fair FL → **[29]**, 구 [33] reject-option discrimination → **[30]**
- 구 [39] FedOSS → **[38]**, 구 [40]–[43] (MSP/energy/SelectiveNet/El-Yaniv) → [39]–[42]
- 다중 인용 그룹은 재번호 후 오름차순 정렬.

**Needs verification (투고 전 확인):** [16] arXiv:2512.12844, [17] arXiv:2603.24704,
[18] arXiv:2605.08263, [37(구38)] arXiv:2606.08517 — 출판 상태 갱신 여부; [13]
arXiv:2510.13037.

## 6. 하지 않은 것 (의도적)

- **증명의 Appendix 이동 없음** — 사용자 선택("본문 압축만") 준수.
- **KO draft (`Fed-CORE_draft_KO.md/.docx`) 미변경** — 명시 요청 시에만 동기화.
- 정리 번호 체계, 실험 수치, 헤드라인 주장 일절 변경 없음.
- git commit 하지 않음 (작업 전부터 unrelated 변경이 working tree에 있어 커밋은
  사용자 판단에 위임).

## 7. 2차 작업분 (같은 날, 실험·그림 보강)

- **신규 실험 — FCP coverage-rule recast (B3 완료).**
  `fedcore/experiments/exp_fcp_recast.py` 신규: federated CP를 "singleton
  prediction set이면 accept" selector로 recast (저자들 스스로 guarantee-free로
  표기한 heuristic). 18개 clean ResNet run에서 **acceptance 84%,
  realized accepted risk 0.225–0.382 — α=0.10/0.20을 18/18 run에서 초과**
  (`runs/fcp_recast.csv`). Table 4에 행 추가 + §5.2에 해석 문단 추가.
  → §5.1의 "recast 평가" 약속이 이제 실제 실행으로 뒷받침됨 (1차 작업에서
  약속을 완화했던 C1을 실행 추가로 재해결).
- **신규 Figure 6** (`F8_frontier_detectors.png`, §5.5):
  (a) real-data certified-coverage frontier — CertCov@α vs α∈[0.05,0.30]
  (GN/BN 5 seeds, FedPD native 3 seeds; npz에서 재계산). baseline은
  α∈(0.05,0.10]에서 비공허화, FedPD는 α=0.05부터 인증 — feasibility law의
  target 축을 실데이터로 시각화. **주의: default fold 프로토콜이라 Table 5(a)
  (cert_frac 0.5)와 절대값 비교 불가 — 캡션에 명시.**
  (b) AUROC vs CertCov@0.20 scatter (T8 aggregates) — score-agnostic thesis의
  시각화 (단조 관계, validity 불변). FedPD 'sm' score는 낮을수록 known이라
  부호 반전(-sm) 필요했음 (스크립트에 주석).
- **표 정돈**: Table 7 헤더 축약(줄바꿈 방지), Table 3 "pooled (Rem. 1)" 축약,
  Table 2 A4′ 행 축약, Table 4 캡션에 FCP-recast 행 설명 추가. 전 표 렌더링
  검수(페이지 캡처) — 폭 초과 없음.
- 페이지: 47 → **48** (Figure 6 + FCP 문단 순증 ~1pp; 45 내외 유지).

## 8. 3차 작업분 (2026-07-11, ws4090 캠페인 반영)

- **FCP-recast resampling 격상**: 18/18 점추정 → 18,000-redraw 위반율
  (α=0.10: 1.0000, α=0.20: 0.9999; CP95 하한 0.9998/0.9997). Table 4 행·§5.2 갱신.
- **δ-sensitivity**: §5.5 knobs 문단에 1문장 반영 (GN d5 α=0.20:
  δ∈{0.05,0.10,0.20} → 0.319/0.341/0.392; graceful degradation).
- **Fig 5(b) asym 승격**: 단일-seed dashed → **10-seed mean±sd** (ws4090 60-run
  캠페인, 40.1 GPU-h). sym과 동형 붕괴; asym 0.2/d0.5의 단일 certified seed는
  본문에 정직 기재. 캡션 갱신.
- **Fig 6(a) FedPD 5-seed**: seed 3–4 npz 회수로 3→5 seeds. frontier 전 구간
  지배 유지 (α=0.05부터 비공허).
- 페이지: 48 → **49** (resampling·δ 문장 순증).
- 커밋: runs/·npz는 gitignore 정책상 제외, 코드·reports·docs만 커밋 (서버 제안
  커밋 목록에서 runs/* 항목은 정책 사유로 미적용).

## 9. 잔여 옵션 (49pp → 45pp가 꼭 필요하면)

1. 증명 7개를 Appendix로 이동: 약 −3pp (가장 효과 큼, 본문 가독성도 향상).
2. Table 2(가정 목록)를 반으로 압축: −0.5pp.
3. §5.2 synthetic 요약과 Table 4 통합: −0.5pp.

현재 48pp는 "45 내외" 상단으로 판단하며, 위 1번만 수행해도 45pp 안팎이 된다.
