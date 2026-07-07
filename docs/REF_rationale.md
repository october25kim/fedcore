# Reference Rationale — Fed-CORE (numbering as of appearance-order renumbering)

목적: 각 인용의 (a) 본문 내 위치, (b) 인용 이유를 기록한다. 리뷰 대응과
camera-ready 시 인용 정리의 근거 문서. 번호는 등장 순서 재정렬 이후 기준.

## Foundations (§1 도입 개념)

| # | Reference | 위치 | 인용 이유 |
|---|---|---|---|
| 1 | McMahan et al., FedAvg (AISTATS 2017) | §1 FL 정의, §5.1 학습 프로토콜 | FL 표준 학습 알고리즘의 원 논문. 본문 전체의 base 학습이 FedAvg |
| 2 | Kairouz et al., Advances and Open Problems in FL (FnTML 2021) | §1 FL 정의 | FL의 배포 현실(프라이버시·이질성·시스템 제약)을 포괄하는 표준 서베이. "raw data를 공개하지 않는 협력 학습"이라는 전제의 근거 |
| 3 | Chow, reject tradeoff (IEEE TIT 1970) | §1 abstention, §3 selector | 거절 옵션(reject option)의 시조 논문. Fed-CORE의 selector가 계승하는 고전 개념의 출처 |
| 4 | Scheirer et al., Toward Open Set Recognition (TPAMI 2013) | §1 OSR 정의 | OSR 문제 정식화의 시조. "open-set recognition (OSR)"이라는 용어와 open-space risk 개념의 원전 |

## FedOSR / OOD base models (§1, §2, §5.5)

| # | Reference | 위치 | 인용 이유 |
|---|---|---|---|
| 5 | FedPD (ICCV 2023) | §1, §2, Table 1, §5.5 | 대표 FedOSR 방법이자 본 논문의 **최강 base detector**(PROSER score 인증 대상). ranking-metric 평가 관행의 대표 사례 |
| 6 | FOOGD (NeurIPS 2024) | §1, §2, Table 1, §5.5 | FL에서 OOD generalization+detection 동시 목표의 대표. SM3D score가 인증 대상 base. 형식 보증이 score-model 추정오차(MMD)에 그침을 §2에서 대비 |
| 7 | FedOSS (IEEE TMI 2024) | §1, §2, Table 1 | 의료영상 FedOSR 대표 사례. 재현 최고비용 옵션으로 §5.5에서 defer 판단의 대상 |
| 8 | FedPD++ (IJCV 2026) | §2 | FedPD의 저널 확장판. 최신 상태 반영 |
| 9 | FedOV (ICLR 2023) | §2 | one-shot FL + label skew에서 unknown-class 학습이라는 인접 접근. FedOSR 스펙트럼의 폭을 보여줌 |
| 10 | FedNovel (arXiv 2023) | §2 | 연합 novel-class 발견/학습 라인 대표. "발견"과 "인증"의 목적 차이 대비 |
| 21 | FedoSSL (ICML 2023) | §2 | 연합 open-world 준지도의 대표. novel class를 학습하는 라인과의 경계 설정 |
| 22 | FOSTER (ICLR 2023) | §2 | 이질성 자체를 OOD 탐지 신호로 쓰는 역발상 라인. 이질성-탐지 상호작용 문헌의 커버리지 |
| 23 | Noise-Resistant FedOSR (KSEM 2026) | §2 | corruption 하 FedOSR 중 최근접(베이지안 불확실성+라벨 교정). "모델 수리" 접근과 "인증" 접근의 대비점 |
| 24 | Adversarial compact wrapping (Inf. Sci. 2024) | §2 | 중앙집중 OSR의 최신 경향(수용영역 압축) 대표. venue 적합성 겸 중앙 OSR 커버리지 |
| 25 | Structural-entropy FGL (Inf. Sci. 2025) | §2 | 이미지 밖(그래프) 연합 이질성 연구의 대표. 이질성 문제의 일반성 근거 |

## Federated conformal prediction (§1, §2, Table 1)

| # | Reference | 위치 | 인용 이유 |
|---|---|---|---|
| 11 | Lu et al., FCP (ICML 2023) | §1, §2, §4.4, Table 1 | 연합 conformal의 기준점. partial exchangeability 관점을 차용하되 대상 functional(quantile/coverage vs accepted risk ratio)이 다름을 §4.4 non-reducibility (b)에서 정면 대비 |
| 12 | Plassier et al. (ICML 2023) | §1, §2 | FCP의 label-shift 확장. FCP 라인이 활발함과 여전히 closed-set임을 동시에 보여줌 |
| 13 | Rob-FCP (ICML 2024) | §1, §2 | FCP의 Byzantine 확장. 동일 |

## Conformal novelty / open-set / selective risk (§1, §2, Table 1)

| # | Reference | 위치 | 인용 이유 |
|---|---|---|---|
| 14 | Bates et al., conformal p-values (Ann. Stat. 2023) | §1, §2 | 중앙집중 conformal novelty-FDR 라인의 정초 논문. §1 "제3의 연구군" 문단의 anchor |
| 15 | Xie et al., open-set conformal (arXiv 2025) | §1, §2 | conformal open-set 분류(Good–Turing p-values)의 최신 대표 |
| 16 | Hallberg Szabadváry et al. (MLwA 2025) | §1, §2 | reject-option conformal 분류의 최신 대표. 중앙집중·이항 설정임을 대비 |
| 17 | CRC (ICLR 2024) | §1, §2, Table 1 | conformal risk control의 기준점. monotone risk 일반화의 원전 |
| 18 | SCRC (arXiv 2025) | §1, §2, §5.1, Table 1 | 선택 후 risk control의 최근접 중앙 prior. novelty 방어의 핵심 대비 대상 |
| 19 | SCoRE (arXiv 2026) | §1, §2, §5.1, Table 1 | e-value 기반 selective risk의 최근접 중앙 prior |
| 37* | Joint certificate (arXiv 2026) | §2, §5.1, Table 1 | selected risk ratio+acceptance를 동시 bound하는 가장 최근접 중앙 prior. Fed-CORE가 "federated counterpart"임을 규정하는 비교축 (*새 번호는 재정렬 후 목록 참조) |
| 20 | Loh & Xiang, decentralized novelty (arXiv 2026) | §1, §2, §5.1, Table 1 | 유일한 분산 conformal novelty. 최근접 분산 prior이나 batch-FDR object로 차별화 |

## Method primitives (§3–§4)

| # | Reference | 위치 | 인용 이유 |
|---|---|---|---|
| — | Clopper–Pearson (Biometrika 1934) | §1, §4.1 | 인증서의 이항 정확 구간 원전 |
| — | Hoeffding (AMS 1956) | §1, §4.6 Remark 1 | Poisson-binomial vs binomial 꼬리 비교의 고전. pooling 실패 논증과 Remark 1의 수학적 배경 |
| — | El-Yaniv & Wiener (JMLR 2010) | §3 target risk, selector | selective risk / risk-coverage 개념의 정초. R_sel이 이 개념의 연합 open-set 확장임을 명시 |
| — | SelectiveNet (ICML 2019) | §3 | 학습된 reject option의 현대적 대표. selector 개념의 근거 |
| — | MSP (ICLR 2017) | §3, §5.1 | 기본 score의 원전 |
| — | Energy score (NeurIPS 2020) | §3, §5.1 | 대표 대안 score의 원전 |
| — | RCPS (J. ACM 2021) | §2 | distribution-free risk control의 정초. CRC의 전신 |

## Adjacent reliability (§2 한 문단)

| # | Reference | 인용 이유 (공통: 배포 신뢰성의 개별 축을 다루나 accepted-risk 인증은 부재함을 한 문단으로 커버) |
|---|---|---|
| 26–31 | personalization/fairness, clustering, scheduling, comm.-efficient FL (Inf. Sci. 2023–2025) | venue 적합성 + 연합 이질성 대응의 스펙트럼. 각각 개인화·공정성·클러스터링·스케줄링·통신효율 축 |
| 32 | Kamiran et al., reject option (Inf. Sci. 2018) | reject option의 응용 확장(차별 통제). 거절 개념의 저널 내 선행 |
| 33 | Du et al., GAE outlier (Inf. Sci. 2022) | 비지도 outlier 탐지 축 |
| 34 | Coelho et al., quadtree drift (Inf. Sci. 2023) | concept drift 탐지 축(분포 이동 인접 문제) |

## §1에 새로 추가된 5편과 그 이유 (2026-07-06)

1. **Kairouz et al. 2021** — §1이 FL의 배포 전제를 서술하면서 근거 문헌이
   FedAvg 하나뿐이었음. 시스템·프라이버시·이질성을 포괄하는 표준 서베이로
   전제 문장의 하중을 분산.
2. **Scheirer et al. 2013** — "This is open-set recognition (OSR)"이 무인용
   상태였음. OSR 용어의 원전 없이는 도입부가 2차 문헌 위에 서게 됨.
3. **Chow 1970** — abstention 문장이 무인용. 거절 옵션의 시조를 인용해
   "accept/reject"라는 논문의 기본 동작이 고전에 뿌리내림을 표시.
4. **Bates et al. 2023** — §1 "제3의 연구군" 문단이 세 라인을 나열하면서
   인용이 전무했음. conformal novelty-FDR 라인의 정초 논문을 anchor로 삽입
   (기존에는 분산 버전 [20]만 인용되어 중앙 원류가 비어 있었음).
5. **El-Yaniv & Wiener 2010** — 논문의 중심 개념인 selective risk의 정초
   문헌이 부재했음. §3 target risk 정의와 selector 문장에 삽입해 R_sel이
   고전적 selective risk의 연합 open-set 확장임을 명시.

## 재정렬 기록

- 2026-07-06: 전 인용을 본문 등장 순서로 재번호(42편, 미사용 0).
  새 순서: [1] FedAvg → [2] Kairouz → [3] Chow → [4] Scheirer →
  [5–10] FedOSR → [11–13] FCP 라인 → [14–16] conformal novelty/open-set →
  [17–19] centralized selective risk → [20] decentralized novelty → 이후
  §2–§5 등장 순.
