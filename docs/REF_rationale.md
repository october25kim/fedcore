# Reference Rationale — Fed-CORE (최종 번호 기준, 2026-07-06 2차 재정렬 이후)

목적: 각 인용의 (a) 본문 내 위치, (b) 인용 이유를 기록한다. 리뷰 대응과
camera-ready 시 인용 정리의 근거 문서. 규칙: 문장당 인용 1건(Table 1의
method-family 셀만 그룹 허용), 총 42편, 본문 등장 순서로 번호 부여.

## Foundations (§1 도입 개념)

| # | Reference | 위치 | 인용 이유 |
|---|---|---|---|
| 1 | McMahan et al., FedAvg (AISTATS 2017) | §1 FL 정의, §5.1 | FL 표준 학습 알고리즘의 원 논문. 본문 전체의 base 학습이 FedAvg |
| 2 | Kairouz et al., FL survey (FnTML 14, 2021) | §1 | FL 배포 현실(프라이버시·이질성)을 포괄하는 표준 서베이. 배포 전제 문장의 근거 |
| 3 | Chow (IEEE TIT 16(1), 1970) | §1 abstention, §3 selector | 거절 옵션의 시조. selector 개념의 고전적 뿌리 |
| 4 | Scheirer et al. (TPAMI 35(7), 2013) | §1 OSR 정의 | OSR 용어·문제 정식화의 원전 |

## FedOSR / OOD (§1, §2, Table 1, §5)

| # | Reference | 위치 | 인용 이유 |
|---|---|---|---|
| 5 | FedPD (ICCV 2023) | §1, §2, Table 1, §5.1, §5.5 | 대표 FedOSR이자 본 논문의 최강 base detector(PROSER score 인증 대상) |
| 6 | FOOGD (NeurIPS 2024) | §1, §2, Table 1, §5.1, §5.5 | FL OOD gen.+det. 동시 목표의 대표. SM3D score 인증 대상. 형식 보증이 MMD 추정오차에 그침을 §2에서 대비 |
| 7 | FedOV (ICLR 2023) | §1, §2 | one-shot FL+label skew에서 open-set voting. FedOSR 스펙트럼의 폭 |
| 8 | FedNovel (arXiv 2023) | §1, §2 | 연합 novel-class 발견 라인 대표. "발견" vs "인증" 목적 대비 |
| 21 | FedPD++ (IJCV 2026) | §2 | FedPD의 저널 확장. 최신 상태 반영 |
| 22 | FedoSSL (ICML 2023) | §2 | 연합 open-world 준지도 대표. novel-class 학습 라인과의 경계 |
| 23 | FOSTER (ICLR 2023) | §2 | 이질성을 OOD 탐지 신호로 쓰는 역발상 라인 커버 |
| 24 | Noise-Resistant FedOSR (KSEM 2026) | §2 | corruption 하 FedOSR 최근접. "모델 수리" vs "인증" 대비점 |
| 25 | Compact wrapping OSR (Inf. Sci. 680, 2024) | §2 | 중앙 OSR 최신 경향 + venue 적합성 |
| 26 | Structural-entropy FGL (Inf. Sci. 718, 2025) | §2 | 그래프 연합 이질성. 이질성 문제의 일반성 |
| 38 | FedOSS (IEEE TMI 43(1), 2024) | §2, Table 1, §5.1 | 의료영상 FedOSR 대표. 재현 최고비용으로 defer 판단 대상 |

## Federated conformal (§1, §2, §4.4, Table 1)

| # | Reference | 위치 | 인용 이유 |
|---|---|---|---|
| 9 | Lu et al., FCP (ICML 2023) | §1, §2, §4.4, Table 1 | 연합 conformal의 기준점. partial exchangeability 차용, functional 차이를 §4.4에서 정면 대비 |
| 10 | Plassier et al. (ICML 2023) | §1, §2 | FCP의 label-shift 확장. 라인의 활발함+closed-set 한계 동시 표시 |
| 11 | Rob-FCP (ICML 2024) | §1, §2 | FCP의 Byzantine 확장. 동일 |

## Conformal novelty / open-set / selective risk (§1, §2, Table 1)

| # | Reference | 위치 | 인용 이유 |
|---|---|---|---|
| 12 | Bates et al. (Ann. Stat. 51(1), 2023) | §1, §2 | 중앙 conformal novelty-FDR의 정초. §1 제3연구군의 anchor |
| 13 | Xie et al. (arXiv 2025) | §1, §2 | conformal open-set 분류(Good–Turing)의 최신 대표 |
| 14 | Hallberg Szabadváry et al. (MLwA 20, 2025) | §1, §2 | reject-option conformal의 최신 대표. accepted singleton 오류 보증 |
| 15 | CRC (ICLR 2024) | §1, §2, Table 1 | conformal risk control 기준점 |
| 16 | SCRC (arXiv 2025) | §1, §2, Table 1 | 선택 후 risk control의 최근접 중앙 prior |
| 17 | SCoRE (arXiv 2026) | §1, §2, Table 1 | e-value selective risk의 최근접 중앙 prior |
| 18 | Loh & Xiang (arXiv 2026) | §1, §2, Table 1 | 유일한 분산 conformal novelty. batch-FDR object로 차별화 |
| 36 | RCPS (J. ACM 68, 2021) | §2 | distribution-free risk control의 정초. CRC의 전신 |
| 37 | Joint certificate (arXiv 2026) | §2, §5.1, Table 1 | selected risk+acceptance 동시 bound의 가장 최근접 중앙 prior. §5.1 oracle 비교 배제 사유의 대표 인용 |

## Method primitives (§1, §3, §4)

| # | Reference | 위치 | 인용 이유 |
|---|---|---|---|
| 19 | Clopper–Pearson (Biometrika 26, 1934) | §1, §4.1 | 이항 정확 신뢰구간의 원전 — 인증서의 수학적 원자 |
| 20 | Hoeffding (Ann. Math. Statist. 27, 1956) | §1, §4.6 | Poisson-binomial vs binomial 꼬리 비교의 고전. pooling 실패 논증·Remark 1의 배경 |
| 39 | MSP (ICLR 2017) | §3, §5.1 | 기본 confidence score의 원전 |
| 40 | Energy score (NeurIPS 2020) | §3, §5.1 | 대표 대안 score의 원전 |
| 41 | SelectiveNet (ICML 2019) | §3 | 학습된 reject option의 현대 대표. selector의 근거 |
| 42 | El-Yaniv & Wiener (JMLR 11, 2010) | §3 target risk | selective risk 개념의 정초. R_sel이 그 연합 open-set 확장임을 명시 |

## Adjacent reliability (§2, 문장당 1건으로 분리 서술)

| # | Reference | 인용 이유 |
|---|---|---|
| 27 | Ren et al. (Inf. Sci. 647, 2023) | 클러스터드 co-meta-learning 개인화 축 |
| 28 | Li et al. (Inf. Sci. 619, 2023) | 공정성-이질성 축 |
| 29 | Pan et al. (Inf. Sci. 712, 2025) | global vs personalized trade-off 축 |
| 30 | Yu et al. (Inf. Sci. 678, 2024) | 비볼록 pairwise fusion 클러스터링 축 |
| 31 | FedRich (Inf. Sci. 645, 2023) | 휴리스틱 스케줄링 축 |
| 32 | Zhou & Yang (Inf. Sci. 661, 2024) | 통신효율·프라이버시 집계 축 |
| 33 | Kamiran et al. (Inf. Sci. 425, 2018) | reject option의 응용 확장(차별 통제) |
| 34 | Du et al. (Inf. Sci. 608, 2022) | 비지도 outlier 탐지 축 |
| 35 | Coelho et al. (Inf. Sci. 625, 2023) | concept drift 탐지 축 |

공통 이유: 배포 신뢰성의 개별 축(개인화·공정성·클러스터링·스케줄링·통신·
거절·이상치·드리프트)을 다루지만 accepted-risk의 유한표본 인증은 부재함을
보여 novelty 경계를 긋는다. venue(Inf. Sci.) 적합성 겸용.

## §1에 새로 추가된 5편과 그 이유 (2026-07-06)

1. **Kairouz et al. 2021** [2] — FL 배포 전제 문장의 근거가 FedAvg 하나뿐이었음.
2. **Scheirer et al. 2013** [4] — OSR 용어의 원전이 무인용이었음.
3. **Chow 1970** [3] — abstention 문장이 무인용. 거절 옵션의 시조 표기.
4. **Bates et al. 2023** [12] — 제3연구군 문단이 인용 0건이었음. 중앙 novelty-FDR 원류 anchor.
5. **El-Yaniv & Wiener 2010** [42] — 중심 개념 selective risk의 정초 문헌 부재 해소.

## 재정렬 기록

- 2026-07-06 (1차): 등장 순서 재번호(42편, 미사용 0).
- 2026-07-06 (2차): 단일 인용 문장 규칙 적용(문장당 1건, Table 1 셀 제외) 후
  재번호. 42편 전부 사용 재확인. 본 문서의 번호는 2차 기준.
