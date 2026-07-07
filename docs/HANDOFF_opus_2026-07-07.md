# Fed-CORE 인계 문서 (2026-07-07, HEAD `78f75e9`)

새 채팅에서 이 문서만 읽으면 프로젝트의 현재 상태를 완전히 파악할 수 있도록
작성한 인계 문서다. repo root의 `CLAUDE.md`, `AGENTS.md`가 규칙의 원본이며,
이 문서는 그 위에서 "지금까지 무엇을 했고 무엇이 남았는가"를 기록한다.

---

## 1. 프로젝트 정체

**Fed-CORE: Federated Certified Open-Set Recognition** — Information Sciences
(Elsevier) 투고 목표. 이질적(non-IID)·corrupted 환경에서 federated로 학습된
open-set classifier의 **accepted selective risk** `R_sel(λ) = Σλ_j m_j / Σλ_j a_j`
를, 재학습 없이, 클라이언트에 분산된 작은 trusted clean calibration set으로,
finite-sample distribution-free하게 certify하는 논문. FedOSR-accuracy 논문도
noisy-label 논문도 아니다. Headline metric은 **CertifiedCoverage@α**
(= seed별 `cert_coverage_lcb`의 평균, 수치 일치 검증 완료).

작업 환경 두 대: **laptop**(원고·수치 통합·이 repo)과 **Ubuntu 4070**(GPU 실험,
Claude Code 실행). 산출물은 사용자가 수동 복사로 laptop에 반입("복사완료" 패턴).

## 2. 원고 현재 상태

- 소스: `docs/Fed-CORE_draft.md` (545줄) → `build_docx.sh` (pandoc +
  `ins_format.py` 후처리) → `docs/Fed-CORE_draft.docx` (~49쪽, TNR 12pt
  double-spaced, booktabs 표, 줄번호).
- 구조 (**appendix 없음**, 본문 완결): Abstract / §1 Introduction /
  §2 Related Work (Table 1) / §3 Problem Setup (Fig 1, Prop 1 + proof,
  Table 2 notation + A1–A6) / §4 Method and Theory (inline proofs) /
  §5 Experiments / §6 Limitations / §7 Conclusion / References [1]–[42].
- Figure 5개, Table 6개. 모든 캡션은 1–2문장, 해설은 전부 본문에서.
- 인용 42편, 문장당 1건 규칙(Table 1 셀만 예외), 등장 순서 번호. 전 메타데이터
  web 검증 완료(오류 4건 수정: FedOV 저자, FOOGD 저자 이니셜, Xie 이니셜,
  Hoeffding 연도). 인용 사유는 `docs/REF_rationale.md`.

### 정리(theorem) 최종 번호 — 절대 흔들지 말 것

| 번호 | 내용 |
|---|---|
| Prop 1 | trusted clean label 필요성 (two-world impossibility) |
| Lemma 1 | acceptance-reweighted decomposition |
| Thm 1 | full-simplex stratified certificate: `Ū = max_j r̄_j`, `r̄_j = U⁺(K_j, A_j; δ/J)` (conditional binomial `K_j\|A_j ~ Bin(A_j, r_j)`) |
| Thm 2 | bounded-Λ robust linear-fractional (구 1′; box-vertex 도달, pseudolinearity) |
| Cor 1 | coverage LCB, budget split δ_r + δ_c ≤ δ |
| Prop 2 | naive pooling anti-conservative (constructive; Poisson-binomial ≠ binomial) |
| Thm 3 | feasibility law: certify 가능 iff `A_j ≥ ln(J/δ)/(−ln(1−α))` |
| Remark 1 | pooled matched-mixture bound (Proposition에서 강등; Lemma L은 수치 지지 0.919≥0.90, 형식 증명 TODO) |
| Prop 3 | round-wise self-training safety (one-shot delta 인증) |

가정 A1–A6. 핵심 주의: **A4′**(audit over-representation은 stochastic
dominance 하에서만 보수적)는 empirical protocol이지 theorem assumption이
아니다. **A6**(grouped certificate)은 group-mixture guarantee일 뿐 client-level
simplex 보증이 아님을 본문이 명시한다. Privacy: pooled만 sum-only secure-agg,
stratified는 per-client `(A_j, K_j)` 필요, grouped가 절충 — "two counts only"
주장 금지.

## 3. 확정된 핵심 수치 (검증 완료, 원고 반영됨)

- **Table 5(a) full diagnostics** (54-row `runs/T9_diagnostics.csv`, G=2 grouped,
  FedAvg baseline, CIFAR-10, 5 seeds): GN d=5 α=0.20 **CertCov 0.392±0.097
  (5/5 certified)**; GN d=0.5 α=0.20 **0.353±0.130 (5/5)**; BN d=5 α=0.20
  0.431±0.048 (5/5); α=0.10 행들은 2/5–3/5, 0.077–0.106 (정직하게 보고 —
  collapse 자체가 결과).
- **Resampling validity (E1, 로컬 실행)**: 526,000 인증 평가, 70,025 deploy 판정,
  위반 61건 (rate 8.7×10⁻⁴, CP95 UCB 1.1×10⁻³). 61건 중 56건이 γ=1.0 →
  risk-buffer 정당화의 실증 근거.
- **Pooling-fail ablation**: mixture shift에서 naive pooled CP coverage 0% 붕괴,
  stratified는 전 mixture 유지 (Prop 2의 실험 대응).
- **Detectors (T8, 3 seeds)**: FedPD-PROSER가 최강 real-data positive
  (α∈{0.10, 0.20} 인증, d=5 α=0.20 LCB 0.483). FOOGD-SM3D published 0.071.
- **covtype (valid 프로토콜)**: 0.068±0.135 (1/5)@α=0.2 → 0.213±0.247 (3/5)@0.3
  — feasibility-edge stress domain으로 서술.
- **γ grid 실제값**: {0.2, 0.3, 0.5, 0.7, 1.0} (문서화 오류 {0.5,0.7,1.0}에서
  수정 완료). α=0.05는 결과가 없어 파라미터 목록에서 삭제.
- **δ/2 simultaneous recompute** (`runs/delta_split_recompute.csv`): GN d=5
  α=0.20 headline 0.392→0.341 (4/5), 나머지 셀 −0.003 수준.

## 4. 적용 완료된 리뷰 이력 (외부 리뷰 5회 + 자체 감사)

1. 1차(구조·표현) + 2차(runs 기반 figure/table 강화) 반영.
2. 3차 "conditional strong go": A4/A4′ 분리, A6 headline scoping, Corollary 1
   신설, Thm 1′/2 증명 수정.
3. 4차: **appendix-free 전면 재구조화** (inline proofs, 표 병합, figure 5개).
4. 5차 submission-shaping P0/P1/P2: CertCov 정의 명문화, grouped 표현 일괄,
   privacy 분류 수정 등.
5. 자체 sufficiency 감사(G1–G5) + 외부 sufficiency 리뷰: Table 5(a) 11열 진단
   전면화, claim 협소화 문장들("not a rediscovery of pooling", "do not claim
   broad empirical coverage", "prevents unsafe pseudo-label ingestion",
   "feasibility-edge stress domains"), 확장실험 프롬프트 작성.
6. 형식: 지도교수 Academic Paper Guidelines ver 2.2 준수 (약어 최초 전체표기,
   무축약형, 0–10 숫자 단어, 시제, Table/Figure/Section 대문자, "novel/very"
   회피) + 사용자 추가 규칙 (**모든 figure/table 결과를 본문에서 설명, em-dash
   삽입구 금지, 캡션 1–2문장, 문장당 인용 1건, 총 인용 40–50**).

## 5. 미해결 이슈 (정직성 플래그 — 새 채팅이 반드시 알아야 함)

1. **T8 vs T9 detector 불일치 (P0)**: T9 재계산에서 FOOGD d=5 LCB 0.350 vs
   발표치 0.071 (5배), FedPD 0.491 vs 0.483. 프로토콜 차이(γ grid? fold 정의?
   score?) 미규명. **Table 5(b) 진단열은 이 때문에 의도적으로 비워둠.**
   4070 Task R8/E8이 조화 담당.
2. **Simultaneous δ/2 headline 전환**: 사용자 승인됨. 그러나 per-seed
   simultaneous T9 재생성(std 필요)이 선행조건 → E8에 위임. 현재 원고는
   auxiliary δ/2 convention을 명시하고 §5.5에 delta-split robustness 문장 유지.
3. **로컬 headline 재구성 실패 이력**: contiguous grouping + 저장된 fold로
   d=0.5에서 2/5 vs 집계 3/5 불일치 → 수치 미삽입, 4070에 위임, 4070이 자기
   re-split 프로토콜로 MATCH 확인. fold 재분할 프로토콜("cf=0.5")이 저장된
   npz fold와 다름을 기억할 것.
4. **Theorem 3 converse 부재**: feasibility floor가 근본적인지 CP artifact인지
   공격 가능. 반 페이지 converse lemma 추가가 남은 이론 업그레이드 중 비용 대비
   효과 최대 (권고 상태, 미착수).
5. Pre-submission 수동 항목: [Department/University]·[funding] placeholder,
   [37] (X. Yu, J. Liu) arXiv 직접 확인.

## 6. 진행 중 / 대기 중 실험 (4070)

- **실행 중 (Task E)**: FedPD-PROSER seed 3→5 (d∈{0.5,5}, α∈{0.10,0.20}) +
  self-training 4×-budget 5-seed → `runs/T9_diagnostics.csv` 추가행,
  `runs/selftrain_gain_5seed.csv`.
- **대기 큐 (프롬프트 전달 완료)**: `docs/prompts/PROMPT_expansion_experiments_4070.md`
  기준, 우선순위 **R8(E8) > R1(E1) > R2(E2) > R3 > R6 > R7 > R4 > R5**:
  R8 detector 조화+simultaneous T9 (CPU, 즉시), R1 client scaling J∈{10,20},
  R2 CIFAR-100 다중모델 {resnet18gn, resnet18bn, simplecnn} × d∈{0.5,5} ×
  **10 seeds** (60 runs), R3 FOOGD seed 확장, R4 ρ/γ sensitivity (CPU),
  R5 corruption seeded, R6 client-simplex/small-J positive, R7 covtype stable.
  Seed 정책: 목표 10, 하한 5.
- 조율 규칙: R8의 T9 재생성은 Task E 완료 후 1회; GPU 큐는 Task E 종료 후.

## 7. 판정 이력

moderate go → conditional strong go(3차 리뷰) → **현재: conditional strong go
충족 상태, E-결과(특히 E8+E1+E2) 도착 시 strong go**. 투고 타이밍 권고:
전체 완료를 기다리지 말고 P0(E8·E1·E2) 도착 시 투고, E3–E7은 리비전 탄약.
Novelty 시간 창 주의: SCRC/SCoRE/joint-certificate 라인이 ~3개월 간격으로 이동 중.

## 8. 작업 규칙 (요약; 원본은 CLAUDE.md)

- **검증 안 된 수치는 원고에 절대 삽입 금지.** 실패한 명령·불일치는 보고.
- proposal/certification/test split hygiene 불가침. corruption은 train label만.
- Thm 3(pooled 계열)·Remark 1을 Thm 1/2보다 격상 금지. accuracy/AUROC로 성공
  판단 금지. certified-coverage collapse 숨기기 금지.
- metric schema 이름 변경 금지: `certified, cert_risk_ucb, cert_coverage_lcb,
  cert_n, cert_k, prop_coverage, prop_risk, test_coverage, test_risk,
  score_name, gamma, alpha, delta, Lambda, dirichlet_alpha, n_clients`.
- 한국어로 토론, 영어로 code/commit/paper text. 한국어 번역본
  (`Fed-CORE_draft_KO.md`)은 **명시 요청 시에만** 동기화.
- 실험 해석 보고 형식: 진단 요약 / 확인한 명령 / 핵심 결과 / 판정 / 다음 행동.
- `runs/`, `data/`, weights 미커밋; 새 스크립트는 커밋.

## 9. 파일 지도

| 경로 | 역할 |
|---|---|
| `docs/Fed-CORE_draft.md` | 원고 단일 소스 (docx는 파생물) |
| `build_docx.sh`, `ins_format.py` | pandoc 빌드 + Inf. Sci. 스타일 후처리 |
| `experiments/fedcore/figs/paper_figs.py` | figure 5개 통합 생성기 (Okabe-Ito) |
| `experiments/fedcore/exp_resampling_validity.py` | E1 validity 실험 |
| `runs/T9_diagnostics.csv` | 54행 진단 마스터 (Task E가 확장 중) |
| `docs/REF_rationale.md` | 인용 42편 사유 기록 |
| `docs/prompts/PROMPT_expansion_experiments_4070.md` | 잔여 실험 명세 (R/E 태스크) |
| `docs/SUBMISSION_CHECKLIST.md` | 투고 체크리스트 |

## 10. 다음 행동 (도착 순서대로)

1. Task E 산출물 반입 → 신규 seed 행 검증 후 Table 5(a)·§5.6 갱신 (mean±std
   재계산, 3-seed→5-seed 표기 갱신).
2. R8 결과 반입 → detector 프로토콜 확정 → Table 5(b) 진단열 완성 →
   simultaneous T9로 headline 전환 (0.392→0.341 예상, 본문 문장 조정).
3. R1/R2 반입 → §5.4 client-scaling 결과 + CIFAR-100 다중모델 결과 통합 →
   Abstract·Conclusion의 stress-domain 문장 재검토.
4. (실험 무관, 언제든) Theorem 3 converse lemma 초안 작성.
5. placeholder 채우기 → 최종 빌드 → 투고.
