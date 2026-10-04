# FedCORE S1 v2 (DomainNet-6 × PACS 6클래스 전이) — 실행 보고서

- 작성: 2026-09-25 21:55 KST
- 프로토콜: `FEDCORE-S1-DOMAINNET6-PACSCLS-20260925-v2`
  - PROTOCOL.json sha256 `06dd7e1a22f3d504dc76dd115b3f1803754ffe022a9d436e7dcef8a20c7c142b`
  - 봉인 20:47 KST. 이미지 다운로드 전에 봉인했다.
  - prereg tarball sha256 `fff59002…ac37`
- **판정: KILL → NOT_TRANSFER.** V1 규칙과 임계값을 그대로 적용했다.
  - INVALID/INCOMPLETE/HOLD_VALIDITY 사유는 없다.
  - 서술적 결과일 뿐이며, V1 결과(HOLD)를 다시 쓰지 않는다.

## 1. 한 줄 요약
V1 PACS 체크포인트 30개를 고정한 채 DomainNet 6개 도메인(6클래스, person 제외)에 적용했다. B=512에서는 거의 모든 절차가 인증을 거부했다.
- ResNet-18: 모든 절차에서 인증 0/1500건이었다. 따라서 Δ≡0이고 CI는 [0,0]이다.
- ConvNeXt-Tiny: 인증률이 1.3% 이하였고, g=−0.00035였다.
- 결과적으로 V1 KILL 조항 두 개가 모두 충족됐다.
  - 조항 1: 어떤 fixed-control CI 상한 ≤ 0.
  - 조항 2: 두 아키텍처 모두 g < .002.
- **선택 규칙(V1_SCALE_CP)의 우위는 이 외부 도메인으로 전이되지 않았다.** 다만 이 KILL은 "대조군이 이긴" 결과가 아니다. 인증이 거의 일어나지 않는 **바닥(floor) 영역**에서 차이가 사라진 결과다. 해석은 5절에 있다.

## 2. 봉인된 핵심 결정 (PROTOCOL.md/JSON)
- **별개 실험.** S1 v1(INVALID, person 부재)이 실패한 뒤 새로 등록했다.
  - 새 protocol_id와 새 bootstrap seed `2026092501`을 쓴다.
  - 봉인 시점에 DomainNet 픽셀과 모델 출력은 0건이었다.
- **데이터.**
  - 공식 **cleaned** 버전이다(공식 페이지 권장, 중복 제거, S1 v1 증거와 같은 목록).
  - 정확한 URL 6개와 HEAD 크기를 봉인했다.
  - train∪test 목록을 모두 쓰고, 6클래스 항목은 서브샘플링하지 않고 전부 쓴다.
- **역할.** proposal/audit를 50/50로 나눴다. `H=SHA256(pid|role|identity)`, 2H<2^256이면 proposal이다.
  - V1의 training 60%는 예측기가 고정이라 쓸모가 없어서 제거했다. proposal:audit 비율 1:1은 유지했다.
- **분할.** V1의 5개 분할을 그대로 썼다(고정된 4-way 헤드가 분할을 정한다).
- **person 비대칭.**
  - split 1·2 (person이 known): 헤드와 logit 4개를 그대로 쓴다. 제거나 재정규화는 없다. person 헤드는 입력을 받지 않으며, argmax=person은 V1 규칙대로 오류다. V1의 `err=(label<0)|(argmax!=label)` 자체가 이 처리이므로 구조를 바꿀 필요가 없었다.
  - split 0·3·4 (person이 unknown): unknown 클래스가 3개에서 2개로 줄어든다.
  - PRIMARY는 5개 분할 전체, SECONDARY(민감도)는 {0,3,4}, {1,2}는 서술용이다.
- **예산.**
  - J=6이며, V1 tail 공식을 J=6에서 계산한다(예: .05/72).
  - 총 B는 V1과 같은 {256, 512(주), 1024}이다.
  - 도메인당 floor(B/6)=42/85/170개를 뽑는다. 나머지 4/2/4개는 어느 도메인에서도 뽑지 않는다.
  - 셀 크기가 예산보다 작아도(clipart dog 70, infograph giraffe 168, sketch horse 101) 문제가 없다. 도메인 audit frame에서 복원추출하기 때문이다.
- **지지 게이트.**
  - 36개 (class, domain) 셀마다 proposal ≥10, audit ≥10이어야 한다.
  - 하나라도 미달이면 전체 INVALID이며, 셀이나 도메인을 버리지 않는다.
- **그 외.** 누락 허용치(도메인당 ≤0.5%), 라벨 충돌 시 INVALID, PACS 정확 동일 이미지가 있으면 HOLD_VALIDITY.
- **방법·지표·임계값.** 모두 V1과 같다.
  - CONFIRM(=TRANSFER): 아키텍처마다 min Δ ≥ .01이고 CI 하한 14개가 모두 > 0.
  - KILL(=NOT_TRANSFER): 어떤 CI 상한 ≤ 0, 또는 두 아키텍처 모두 min Δ < .002.
  - 나머지는 HOLD.
- S1 v1과의 모든 차이는 PROTOCOL.md 6절 DIFF 표에 있다.

## 3. 데이터 (G1–G4, G6)

| domain | zip sha256 | bytes (=HEAD) |
|---|---|---:|
| clipart (groundtruth/clipart.zip) | `f8f248a3e8bbc9868c523ae24e6601b4c2ad102afc60c0efeb4f3255df22354d` | 1,270,909,081 |
| infograph | `891e14f476317e97b6f06d6498c11d0a5b8d0c8b57210765ed9beb072da611fa` | 4,338,659,804 |
| painting (groundtruth/painting.zip) | `fa47e6d405503ea0286cabd767f176bd30e988b63ddd7b9db16cf030c9770015` | 3,679,366,174 |
| quickdraw | `9f37df2a8c3733703dbb35ad9964202a65f675ca30137756457537536b485d84` | 459,920,701 |
| real | `d7fe010b430c5b510f85d657d38c4a1924822695fabea4fef57bd00dba251d63` | 6,062,039,589 |
| sketch | `23a7dbae7016b343e934399beaa8a68728fb5e570ebe2c71df9b75cf91c499fc` | 2,633,992,926 |

- **G1 PASS.** ubuntu에서 20:48–21:06 KST에 받았고, 크기가 6/6 일치했다.
- **G2 PASS.** 목록 항목 12,027개 중 12,027개가 zip에 있고 디코드됐다(누락·손상 0). 셀별 개수는 공식 목록과 정확히 같다. 클래스 정수 라벨도 도메인 간에 일관됐다.
- **G3 PASS.** 도메인 내 중복 0, 라벨 충돌 0, 도메인 간 공유 identity 0.
- **G4 PASS.** 최소 셀은 proposal 36(clipart dog), audit 34(clipart dog)였다.
  - 도메인 audit/proposal frame: clipart 357/368, infograph 619/672, painting 1042/1033, quickdraw 1499/1501, real 1865/1883, sketch 606/582.
- **G6 PASS.** V1 PACS identity 9,991개와 정확히 겹치는 것이 0개였다.
- **G5 PASS.** 체크포인트 30/30 sha256 일치, state digest 30/30 일치, V1 코드 해시 8/8 일치.
- **G7 dry-check PASS.** 캐시 변환과 V1 변환의 max|Δ|=0(48장), 라벨맵이 프로토콜과 같았고, logit은 (N,4)이며 유한했다. ECA나 점수 요약은 계산하지 않았다.

## 4. 결과 (주 분석, B=512, 분모: 아키텍처당 15 models × 2 t × 100 replicates)
- rows 162,000 (primary 54,000), candidate flags 1,512,000, validity groups 1,260.
- HOLD_VALIDITY 플래그 0개(임계 p=3.97e-5). 최대 joint-failure 비율은 0.0023이었다.

| arch | g = min Δ̂ (7 fixed) | argmin | CI 하한>0 | CI 상한≤0 |
|---|---:|---|---:|---:|
| ResNet-18 | **0.00000** | (7개 모두 Δ≡0) | 0/7 | **7/7** (모두 [0,0]) |
| ConvNeXt-Tiny | **−0.00035** | DIRECT_BETTING_H [−0.00103, +0.00018] | 4/7 | 0/7 |

ConvNeXt 대비 (V1_SCALE_CP − control):

| control | Δ̂ | CI |
|---|---|---|
| RATIO / MASS_CP / HOEFFDING_H | +0.00104 | [+0.00014, +0.00255] |
| CP_IUT_H | +0.00073 | [+0.00014, +0.00155] |
| COMPOSED | +0.00032 | [0, +0.00079] |
| SCALE_H | +0.00013 | [0, +0.00045] |
| DIRECT_BETTING_H | −0.00035 | [−0.00103, +0.00018] |
| 진단 GEOMETRY_DIRECT_CP | −0.00048 | [−0.00112, −0.00002] |

**인증률 (B=512, 분모 1500).**
- ResNet-18: 모든 절차, 두 t 모두 0건.
- ConvNeXt, V1_SCALE_CP: t=.5에서 12건(0.8%), t=1에서 2건(0.13%).
  - DIRECT_BETTING_H t=.5: 19건. RATIO/MASS/HOEFFDING: 0건.

B=1024(보조, 서술용):
- ConvNeXt V1_SCALE_CP: t=.5 인증률 18.8%(ECA 0.047), t=1 3.1%.
- DIRECT t=.5: 18.1%. ResNet-18은 최대 0.47%(DIRECT t=.5).

**SECONDARY (person이 unknown인 split {0,3,4}, 아키텍처당 9 models, 서술용).**
- V1 규칙을 적용해도 KILL이다.
- ResNet-18은 Δ≡0, ConvNeXt g=−0.00059(DIRECT).
- {1,2}에서는 두 아키텍처 모두 인증 0이라 Δ≡0이다.

**win/tie/loss (per-model Δ, B=512).**
- ResNet-18: 모든 대조군에서 0/15/0.
- ConvNeXt: RATIO류 3/12/0, DIRECT 0/12/3.

**person 진단 (split 1·2, audit frame에서 argmax=person인 비율, 결정 이후에 계산).**
- ConvNeXt: 0.077(real)–0.319(infograph).
- ResNet-18: 0.045(quickdraw)–0.175(infograph).
- split 1·2의 인증이 0인 것과 방향이 맞는다(인과 주장은 아니다).

**V1 대비 (서술적 비교, 검정 아님, V1 잠금값 그대로).**

| arch | V1 PACS | S1 v2 DomainNet-6 |
|---|---|---|
| ConvNeXt | g=0.01563, CI 하한 7/7 > 0 | g=−0.00035 |
| ResNet-18 | g=0.00034, CI 하한 6/7 > 0 | g=0 |

- V1에서 ConvNeXt의 RATIO 대비 Δ는 +0.063이었고, S1 v2에서는 +0.00104다.
- 전체 표는 TABLES_ko.md와 COMPARISON_V1_descriptive.csv에 있다.

## 5. 해석 (주장 범위)
- 봉인된 판정은 **NOT_TRANSFER(KILL)**이다. V1 HOLD 서사는 유지하고, V1을 CONFIRM으로 재해석하지 않는다.
- 효과가 사라진 이유는 **인증 바닥 효과**다. 모든 절차의 ECA가 0 근처였다. 결정 이후 진실값으로 보면, 각 모델에서 위험이 가장 낮은 적격 후보의 t=.5 최악-혼합 위험은 0.058–0.175로 α=.2보다 낮았다(30개 모델 모두 적격 후보 12/12). 그런데도 도메인당 85 draws와 J=6 union-bound tail로는 위험을 인증할 수 없었다.
  - 따라서 이 결과는 "V1_SCALE_CP가 열등하다"는 증거가 아니다. "이 정보·예산 계약에서 외부 도메인으로 옮기면 모든 절차가 거의 인증하지 못하고, 선택 규칙의 우위도 나타나지 않는다"는 뜻이다.
  - B=1024에서 우위가 나타나는지는 보조(서술) 수치로만 참고할 수 있다. 예산 확대는 사후 변경이므로 판정 근거가 될 수 없다.
- ResNet-18의 "CI 상한 ≤ 0" 조항은 Δ≡0이라는 퇴화 사례로 충족됐다. 그러나 두 번째 조항(두 아키텍처 모두 g < .002)만으로도 KILL이므로 판정은 이 해석에 좌우되지 않는다.
- RATIO가 일반적으로 우월하다는 주장이나 기하가 단독 원인이라는 주장은 하지 않는다. 진단용 GEOMETRY_DIRECT_CP는 ConvNeXt에서 V1_SCALE_CP보다 약간 높았다(CI 상한 −0.00002). 이것은 fixed comparator가 아니고 서술용이다.

## 6. 자원
- **ubuntu-4070 GPU0:** dry-check 2.3 s + 추론 443 s(모델별 합, 단계 wall 445 s) = **0.124 GPU-h**.
  - 21:41:12 KST 실행 시점에 GPU0에는 compute 프로세스가 없었다(37MiB, 0%).
  - 거의 같은 시각(~21:40:41)에 다른 작업(LoRA e1b D1 학습)이 GPU0을 쓰기 시작해 추론 동안 GPU를 공유했다. 추론은 결정적이므로 공유는 wall time에만 영향을 준다.
  - 그 전(21:12–21:41)에는 같은 사용자의 다른 작업(LoRA e1b)이 GPU를 쓰고 있어서 idle 감시 스크립트로 대기했다.
- **workstation4 GPU0:** **0 GPU-h.** 20:44, 21:02, 21:16 확인 때마다 타인 작업이 100%(~11GB)를 쓰고 있어 사용하지 않았다. GPU1–3은 건드리지 않았다.
- 합계 0.124 GPU-h / 상한 8 h.
- CPU: 디코드 12 s, audit+analysis 14 s. 다운로드는 약 18 분(18.4 GB).

## 7. 산출물
- **prereg:** `outputs/fedcore_s1v2_prereg_20260925/` (Mac, box, ubuntu). HASHES 11/11이 세 곳에서 일치하고, tarball `.tar.gz`와 `.sha256`이 있다.
- **run:** ubuntu `/home/sanghoon/Desktop/Workspace/Fedcore/outputs/fedcore_s1v2_run_20260925/`. Mac과 box에 동기화했다.
  - DECISION.json, CONTRASTS.csv, G_BY_ARCH.json, OUTCOMES.csv(162k rows), VALIDITY_GROUPS.csv, PER_MODEL_DELTAS.csv, PER_MODEL_ECA.csv, ECA_BY_ARCH_PROCEDURE_T_BUDGET.csv
  - CONTRASTS_SECONDARY_person_unknown_splits_0_3_4.csv, CONTRASTS_DESCRIPTIVE_person_known_splits_1_2.csv, CONTRASTS_BY_T_descriptive.csv, COMPARISON_V1_descriptive.csv
  - SECONDARY.json, PERSON_DIAGNOSTIC.json, BOOTSTRAP.json, DENOMINATORS.json
  - DATA_BINDING.md/json, data/DATA_CHECK.json(+MANIFEST/EXCLUSIONS/DUPLICATES)
  - RUN_BINDING.json, DRYCHECK.json, GPU_TIME.json, GPU_LEDGER, LAUNCH_LOG.txt
  - logits/(30), audit/(30), code/, FIDELITY.md, HASHES.sha256
- **데이터:** ubuntu `data/domainnet_s1v2/` (zips 18.4 GB, lists, prepared crop cache 1.8 GB).
