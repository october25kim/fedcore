# Fed-CORE: Federated Certified Open-Set Recognition via Selective Risk Control

Sanghoon Kim

[소속 — 학과 / 대학, 도시, 국가]

교신저자. E-mail address: october25kim@gmail.com

> 검토용 한국어 번역본입니다. 최종 제출 deliverable은 영문판(`Fed-CORE_draft.docx`)이며, 본 번역본은 내용 검토 편의를 위한 것입니다. 수식·그림·표·참고문헌(영문)·지표명(예: `cert_risk_ucb`)은 원문 그대로 유지했습니다.

---

## Highlights

- 재학습 없이 federated open-set 모델의 accepted selective risk를 인증한다.
- conditional-binomial 인증서는 naive pooling이 anti-conservative해지는 상황에서도 유효하다.
- 인증 가능 coverage는 risk와 audit 크기에 대한 finite-sample feasibility law를 따른다.
- worst-group 인증 coverage가 CIFAR-10에서 risk target 0.20일 때 양(positive)이다.
- 인증형 self-training은 주입되는 pseudo-label 오염을 라운드 전체에 걸쳐 bound한다.

---

## Abstract

Federated learning은 학습 중 한 번도 보지 못한 클래스의 입력이 테스트 시점에 등장할 수 있는 안전 민감 응용에 점점 더 많이 배치되고 있다. 기존 federated open-set recognition 방법은 unknown을 reject하지만 그 reject 품질을 경험적으로만(AUROC, FPR95) 평가할 뿐, accept한 예측의 오류율에 대한 보장을 제공하지 못한다. 반대로 federated conformal prediction은 finite-sample 보장을 제공하나 이는 closed-set prediction-set coverage에 한정된다. 본 연구의 목적은 client heterogeneity와 미지의 deployment mixture 하에서, federated open-set 분류기의 accepted selective risk(즉 accept된 예측이 틀릴 확률)를 finite-sample distribution-free 보장으로 인증하는 것이다. 우리는 임의의 federated open-set 모델 위에 얹는 사후(post-hoc) 인증 레이어 Fed-CORE를 제안한다. 핵심 난점은, 이질적인 client들의 accept된 calibration 점을 단순히 pooling하면 pooled accepted-error count가 binomial이 아니라 Poisson-binomial이 되어 anti-conservative해진다는 점이다. 대신 우리는 각 client의 conditional selective risk를 직접 — accept 수가 주어졌을 때 accepted-error의 binomial 법칙으로부터 — bound하고, 전역 risk를 mixture에 강건한 worst case로 인증한다. 또한 client별 feasibility threshold를 유도하고, 인증서를 safe automation 및 certified federated self-training으로 연결한다. 합성 및 실제 federated 벤치마크(CIFAR-10/100과 tabular task) 전반에서 인증서는 명시된 audit-대표성 조건(audit fold가 deployment unknown 발생률을 대표하거나 그 이상으로 담아야 함) 하에서 시험된 모든 설정에서 유효했고, naive pooling은 예측대로 붕괴했으며, 인증 coverage는 feasibility law를 따른다. CIFAR-10에서 본 방법은 risk target 0.20에서 false certificate 없이 5-seed worst-group 인증 coverage 0.39를 달성하며 — 이는 safety 권고가 아니라 finite-sample feasibility demonstration으로 보고한다 — 더 작은 risk target과 두 번째 도메인은 feasibility 경계에 위치한다.

**Keywords:** Federated learning; Open-set recognition; Selective risk control; Conformal prediction; Distribution-free certification; Uncertainty quantification

---

## 1. Introduction

Federated learning은 원시 데이터를 공개하지 않는 client들에 걸쳐 공유 모델을 학습한다. 실제로 이러한 모델은 *open-world* 조건에서 배치된다. 사기 탐지기는 새로운 사기 패턴을 만나고, 임상 모델은 어떤 병원의 학습 데이터에도 없던 질병 아형을 만난다. 따라서 배치된 모델은 known 클래스를 분류할 뿐 아니라 안전하게 분류할 수 없는 입력에 대해서는 **기권(abstain)**해야 한다. 이것이 open-set recognition(OSR)이며, 그 federated 형태(FedOSR)는 현재 활발한 연구 영역이다.

주요 FedOSR 방법들 — parameter-disentanglement aggregation(FedPD, FedPD++) [1,2], score-model OOD detection(FOOGD) [3], open-set voting(FedOV) [4], novel-class discovery(FedNovel) [5] — 은 unknown reject의 *품질*을 개선하고 이를 AUROC, FPR95 같은 ranking 지표로 보고한다. 그러나 이들 중 어느 것도 배치자(deployer)가 실제로 답을 필요로 하는 질문에 답하지 못한다. **이 모델의 confident한 예측을 accept하여 행동에 옮길 때, 그중 최악의 오류율은 얼마이며, 그것을 허용오차 α 이하로 보장할 수 있는가?** ranking 지표는 이에 답하지 못한다. AUROC가 높은 모델이라도 임의의 고정 운용 threshold에서 accept된 예측의 오류율이 허용 불가능할 수 있으며, 배치는 바로 그 threshold를 고정한다.

두 번째, 별개의 문헌은 FL에서 finite-sample 보장을 제공한다. federated conformal prediction(FCP) [6]은 i.i.d. 가정을 partial-exchangeability로 완화한 하에서 marginal coverage를 갖는 prediction set을 구성하며, 최근 변형들은 label shift [7]와 Byzantine client [8]에 대한 강건성을 더한다. 그러나 FCP는 closed-set이다. 즉 참 레이블이 known 클래스 안에 있다고 가정하고, 그 레이블이 반환된 집합에 포함됨을 보장한다. unknown을 reject하지 않으며, 저자들 스스로 밝히듯 그 selective-classification 시연은 *보장 없는* 휴리스틱이다. prediction set의 coverage와 *accept된 점 예측의 risk*는 서로 다른 범함수이며, 하나가 다른 하나를 함의하지 않는다.

세 번째 문헌은 unknown reject를 인증한다 — false-discovery-rate(FDR) 제어를 갖는 conformal novelty detection, conformal open-set classification, selective conformal risk control 등 — 그러나 거의 전적으로 **중앙집중(centralized)** 설정이다. 최근의 유일한 분산형 진입은 순수 novelty-detection(inlier/outlier) 틀에서 테스트 점 집합에 대한 **batch FDR**를 제어할 뿐, known-class 분류기도 accepted selective risk 개념도 없다. 테스트 batch에 대한 FDR와 분류기의 accept된 예측의 selective risk는 다시금 서로 다른 대상이며, 서로 다른 finite-sample 기법을 요구한다.

이 교집합 — **federated이며 heterogeneity를 인지하는, open-set 분류기의 accepted selective risk에 대한 finite-sample 인증서** — 은 비어 있다. 이를 채우는 것이 본 논문의 기여이다.

**왜 어렵고, 단순 조합이 아닌가.** 이 문제를 "(중앙집중 i.i.d. 사례처럼) Clopper–Pearson selective-risk 인증서를 pooling된 federated calibration 데이터에 적용"하는 것으로 보고 싶은 유혹이 있다. 이는 *무효*하다. heterogeneity 하에서 서로 다른 client의 accept된 calibration 점은 서로 다른 conditional 오류 확률 $r_j$를 가진다. 따라서 pooled accept 수에 조건부인 pooled accepted-error count는 성공 확률이 서로 다른 binomial들의 합(Poisson-binomial)이지 단일 binomial이 아니며, 표준 Clopper–Pearson 구성은 적용되지 않는다. 게다가 deployment mixture가 고오류 client에 가중치를 더 주면 그 사용은 anti-conservative해질 수 있다. 올바른 대상은 deployment-mixture 가중 비율이다.
$$ R_{\mathrm{sel}}(\lambda)=\frac{\sum_j \lambda_j\, m_j}{\sum_j \lambda_j\, a_j},\qquad m_j=\Pr_{P_j}(\text{accept}\wedge\text{error}),\ a_j=\Pr_{P_j}(\text{accept}), $$
여기서 deployment 가중치 $\lambda$는 calibration 시점에 미지이다. 이 비율을 finite-sample, distribution-free, 그리고 미지의 $\lambda$에 대해 강건하게 인증하는 것은, 중앙집중 사례의 단일-binomial 인증서로도, federated conformal prediction의 quantile-coverage 인증서로도 환원되지 *않는* 진정으로 새로운 통계 문제이다.

본 연구의 목적은, 모델을 재학습하지 않고, client heterogeneity와 미지의 deployment mixture 하에서 federated open-set 분류기의 accepted selective risk를 finite-sample distribution-free 보장으로 인증하는 것이다. 우리의 가설은, 작은 신뢰(trusted) clean audit set이 비록 이질적인 global 모델을 *고치기*에는 너무 작더라도, 그 모델의 어떤 예측을 안전하게 accept할 수 있는지를 *인증*하기에는 충분하며, 달성 가능한 인증 coverage가 모델 ranking 품질이 아니라 finite-sample feasibility law에 의해 지배된다는 것이다. 우리는 이 가설을 conditional-binomial 인증서, 인증서가 비자명해지는 시점에 대한 통제된 연구, 그리고 합성·실제 federated 벤치마크에서 평가한 두 가지 downstream use를 통해 검증한다.

본 연구의 주요 기여는 다음과 같이 요약된다:

- 우리는 federated accepted selective risk $R_{\mathrm{sel}}(\lambda)$를 federated open-set recognition의 인증 대상으로 형식화하고, 이것이 prediction-set coverage, ranking 지표, batch false discovery rate와는 다른 범함수임을 보인다(Section 3).
- 우리는 conditional 법칙 $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$로부터 $R_{\mathrm{sel}}(\lambda)$에 대한 finite-sample distribution-free 인증서를 유도하며, 이는 client heterogeneity와 미지의 deployment mixture 하에서 유효하다(Theorem 1/1′). 아울러 client별 feasibility threshold(Theorem 2)와 정직한 privacy taxonomy를 제시하고, federated calibration count의 naive pooling이 anti-conservative하여 중앙집중 conformal prediction으로 환원되지 않음을 증명한다(Section 4).
- 우리는 risk target, group별 audit count, heterogeneity, corruption에 대한 feasibility law를 통해 인증형 open-set 배치가 통계적으로 가능해지는 시점을 특성화하며, 작은 risk target에서 나타나는 외견상 null을 정량적 현상으로 전환한다(Section 4.3, 5).
- 우리는 인증서의 두 가지 downstream use, 즉 보장된 risk에서의 safe automation과 오염이 bound된 certified federated self-training(Proposition 4)을 제시하고, 합성·실제 federated 벤치마크에서 평가하며, 시험된 모든 설정에서 인증서가 유효함을 확인한다(Section 4.7, 5).

위치 짓기(positioning) 주석: Fed-CORE는 새로운 FedOSR 알고리즘이 아니라, **임의의 FedOSR / open-set FL 모델을 위한 인증 레이어**로 이해하는 것이 가장 적절하며, 그 출력은 safe automation과 certified self-training에 *사용*된다.

**Findings (미리보기).** 합성·실제 federated 벤치마크(CIFAR-10/100과 tabular FL task) 전반에서 인증서는 명시된 audit-대표성 조건 하에서 시험된 모든 설정에서 유효했다 — 어떤 seed, data set, normalization에서도 false certificate가 없었다 — 그리고 naive pooling은 예측대로 mixture shift 하에서 붕괴했다. 인증 coverage는 **feasibility law**($(\alpha-\hat r)^{-2}$ group별 표본 threshold)를 따른다. 우리는 CIFAR-10 ResNet-GroupNorm에서 5-seed worst-group $\alpha{=}0.20$ 인증 coverage가 양임을 보고하며($d{=}5$에서 $0.392\pm0.097$, $d{=}0.5$에서 $0.353\pm0.130$, 둘 다 $5/5$, false certificate $0/10$), 두 번째 federated 도메인(tabular FL)은 동일한 feasibility law를 그 경계에서 보인다(seed에 따라 가변적이고 selection-optimistic scoring 하에서만 비자명). $\alpha{=}0.10$ 영역은 seed-가변 feasibility 경계이다. 정직한 메시지는 *특성화된, 유효한 인증서*이지 표제용 정확도 수치가 아니다.

본 논문의 나머지 구성은 다음과 같다. Section 2는 federated open-set recognition, federated conformal prediction, conformal risk control의 관련 연구를 검토한다. Section 3은 문제와 accepted selective risk를 형식화한다. Section 4는 인증서, feasibility threshold, privacy taxonomy, 두 가지 downstream use를 전개한다. Section 5는 실험을 보고하고, Section 6은 한계를 논의한다. Section 7은 결론을 맺는다.

---

## 2. Related Work

**Federated open-set / novel-class / OOD recognition.** FedPD [1]와 그 확장 FedPD++ [2]는 FedOSR를 두 병폐 — closed-set과 open-set 목표 간의 client 간 *inter-set 간섭*, 그리고 heterogeneity에서 비롯되는 client 간 *intra-set 불일치* — 로 틀짓고, parameter disentanglement와 divide-and-conquer aggregation으로 이를 다룬다. FOOGD [3]는 OOD generalization과 detection을 함께 겨냥한다. 그 SM³D 모듈은 feature-space score model을 학습하며(검출 규칙: score norm에 대한 threshold), SAG 모듈은 Stein 항등식을 통해 feature 불변성을 정규화한다. 결정적으로 FOOGD의 유일한 형식적 결과는 reject 결정의 오류율이 아니라 *score model의 추정 오차*(MMD bound)를 bound하며, 검출은 AUROC/FPR95로 보고된다. FedOV [4]는 각 client가 "unknown" 클래스를 출력하도록 학습시키고 open-set voting으로 앙상블하여 label skew 하의 one-shot FL을 다룬다. FedNovel [5]과 federated open-world semi-supervised learning [9]은 client 전반에서 novel 클래스를 발견·학습하며, client heterogeneity 자체가 OOD detection의 신호로 재해석되기도 했다(FOSTER) [10]. Noise-Resistant FedOSR [11]은 corruption 설정에 가장 가까워 Bayesian uncertainty와 label correction을 사용하지만, 위의 모든 방법과 마찬가지로 reject를 경험적으로 평가한다. 중앙집중 OSR은 compact accept/reject 영역을 계속 정교화하고 있으며(예: adversarial compact wrapping classifier [12]), federated heterogeneity는 이미지 외에서도(예: structural-entropy prototype aggregation 기반 그래프 [13]) 연구된다. 그러나 이들은 보장된 accept-오류율이 아니라 경험적 정확도나 검출 품질을 보고한다. 인접한 여러 흐름은 배치 신뢰성의 한 측면씩을 다룬다 — 통계적 heterogeneity 하의 federated 개인화·공정성 [14,15,16], 이질적 federation을 위한 client 클러스터링·휴리스틱 스케줄링 [17,18], heterogeneity 하의 통신 효율·프라이버시 보존 aggregation [19], reject-option classification [20], 비지도 outlier detection [21], data stream의 concept-drift detection [22] — 각각 heterogeneity, 기권, novelty, 분포 변화를 다루지만, 그 어느 것도 accept-예측 오류율에 대한 finite-sample bound를 인증하지 않는다. **accept된 예측에 대한 finite-sample 인증서를 제공하는 것은 없다.**

**Federated conformal / distribution-free uncertainty.** Lu et al. [6]은 Federated Conformal Prediction을 도입하여 exchangeability를 *partial exchangeability*(테스트 점이 확률 $\lambda_k$로 client $k$에 일치)로 대체하고, 프라이버시 보존 T-Digest quantile sketch와 함께 marginal coverage $1-\alpha \le \Pr(Y\in C) \le 1-\alpha + K/(N+K)$를 증명한다. Plassier et al. [7]은 importance weighting으로 label shift를 다루고, Rob-FCP [8]는 Byzantine 강건성을 더하며, generative·conditional 변형은 conditional coverage를 정교화한다. 이들은 모두 unknown reject나 selective risk가 아닌 **closed-set prediction-set coverage**를 인증한다. 우리는 partial-exchangeability 관점을 채택하되 *quantile*과는 다르게 동작하며 이들 연구가 다루지 않는 *risk*(binomial 범함수)를 인증한다.

**Conformal risk control 및 selective conformal prediction (중앙집중).** Conformal Risk Control [23]은 conformal coverage를 임의의 monotone risk로 일반화하며, distribution-free risk-controlling prediction set [24] 위에 구축된다. 2025–2026년의 빠르게 움직이는 흐름은 그 뒤 *사후-선택(post-selection)* risk를 인증한다. Selective Conformal Risk Control [25]은 2단계로 먼저 선택한 뒤 선택된 부분집합에서 risk를 제어하고, SCoRE [26]는 e-value를 통해 "trusted/positive" 사례들 사이의 risk를 제어한다. 이들은 강력하나 **중앙집중 / exchangeable**하다. 즉 federated heterogeneity, 미지의 client mixture, count-only aggregation — Fed-CORE가 구축된 축들 — 을 다루지 않는다. 우리는 이들을 drop-in baseline이 아니라 **중앙집중 oracle**(상한)으로 비교한다.

**Conformal open-set / novelty detection.** Good–Turing p-value 기반 conformal open-set classification [27]과 reject-option conformal classification [28]은 중앙집중적으로 reject를 인증한다. 유일한 분산형 항목인 *Decentralized Conformal Novelty Detection* [29]은 양자화된 surrogate score를 통해 **테스트 batch에 대한 전역 FDR**를 제어한다 — known-class 분류기가 없는 순수 novelty detection으로, accepted selective risk와는 다른 범함수이다. 우리는 이를 가장 가까운 이웃으로 사용하되 대상과 단일 고정 selector 인증이라는 점에서 차별화한다.

**위치 짓기.** 기존 selective conformal/risk-control 방법은 중앙집중/exchangeable 설정에서 사후-선택 risk를 인증하고, federated conformal 방법(FedOSS류 FedOSR [30]과 FCP [6] 포함)은 경험적 reject 품질이나 closed-set prediction-set coverage를 인증한다. Fed-CORE는 빠진 교집합을 채운다. **client heterogeneity와 deployment-mixture 불확실성 하의 federated open-set accepted-risk 인증.** 이는 federated 설정이 새롭게 요구하는 conditional selective-risk 인증서(및 그 mixture-강건 형태)를 제공하며, calibration 통계량은 score quantile이 아니라 conditional-binomial 비율이다.

---

## 3. Problem Setup

![Figure 1](experiments/fedcore/figs/fig0_problem_diagram.png)

**Figure 1. Fed-CORE가 인증하는 것 — 그리고 인증하지 않는 것.** federated open-set 모델 $\hat h$와 selector $A$는 deployment stream을 *accept*(known 클래스 예측)와 *reject*(unknown)로 분할한다. 이 stream에 대해 네 가지 서로 다른 양을 물을 수 있으며 이들은 *서로 교환 가능하지 않다*. AUROC/FPR95는 모든 점에 대한 unknown score의 *ranking*을 측정하고, federated conformal prediction은 known 클래스에 대한 *prediction-set coverage*(closed-set)를 보장하며, batch FDR는 테스트 batch에 대한 false novelty를 제어한다. **Fed-CORE는 $R_{\mathrm{sel}}(\lambda)=\Pr(\hat y\ne Y\mid \text{accept})$ — 실제로 행동에 옮기는 예측들 사이의 오류율 — 을 deployment mixture $\lambda$ 하에서 제어한다.** 집합의 coverage, ranking 품질, batch FDR는 이 accept-예측 오류율을 bound하지 못하지만, Fed-CORE는 federated 설정에서 이질적 client 간 calibration 데이터를 pooling하지 않고 그것을 해낸다.

**Client와 mixture.** $J$개의 client가 있고, client $j$는 $\mathcal{X}\times\mathcal{Y}$ 상의 데이터 분포 $P_j$를 가지며, $\mathcal{Y}=\{1,\dots,C\}\cup\{\textsf{unknown}\}$이다. deployment 데이터는 어떤 가중치 벡터 $\lambda\in\Delta^{J-1}$에 대해 mixture $Q_\lambda=\sum_{j=1}^J \lambda_j P_j$를 따른다. 우리는 $\lambda$가 **calibration 시점에 미지**이며 알려진 볼록 집합 $\Lambda\subseteq\Delta^{J-1}$(예: $\Lambda=\Delta^{J-1}$ 또는 client 데이터 비율 주변의 box)에만 제약된다고 허용한다.

**Open-set 결정.** federated 학습된 분류기 $\hat h$는 점 예측 $\hat y(x)\in\{1,\dots,C\}$를 산출한다. *selector* $A:\mathcal{X}\to\{0,1\}$는 accept 여부를 결정한다. $A(x)=1$은 "$\hat y(x)$를 known 클래스 예측으로 accept하여 행동에 옮김", $A(x)=0$은 "unknown으로 reject / 기권"을 뜻한다. $A$는 임의의 score $s(x)$(maximum softmax probability(MSP) [31], entropy, margin, energy [32])에 threshold를 둘 수 있으며, 보장은 어느 것을 쓰는지에 의존하지 않는다. 이러한 selector는 학습된 reject option을 갖는 selective classification [33]의 open-set 유사물이다.

**대상 risk.** 제어할 양은 deployment 하의 **accepted selective risk**이다:
$$
R_{\mathrm{sel}}(\lambda)\;=\;\Pr_{(X,Y)\sim Q_\lambda}\!\big(\hat y(X)\ne Y \,\big|\, A(X)=1\big)
\;=\;\frac{\sum_{j}\lambda_j\, m_j}{\sum_{j}\lambda_j\, a_j},
$$
여기서 $a_j=\Pr_{P_j}(A(X)=1)$은 client별 accept 비율이고 $m_j=\Pr_{P_j}(A(X)=1,\ \hat y(X)\ne Y)$은 client별 accepted-error mass이다(따라서 client별 selective risk는 $r_j=m_j/a_j$). 비율 형태는 mixture 하의 전확률 법칙에서 따라온다. **목표:** 미지의 deployment $\lambda\in\Lambda$에 대해 신뢰도 $1-\delta$로 $R_{\mathrm{sel}}(\lambda)\le\alpha$를 인증할 수 있을 때에만 $A$를 배치하며, 동시에 accept된 coverage $\mathrm{cov}(\lambda)=\sum_j\lambda_j a_j$를 최대화한다.

**신뢰 calibration 데이터와 분할.** 각 client는 작은 *신뢰(trusted) clean* calibration 표본을 보유하며, 이를 **proposal** fold와 **certification** fold로 분할한다. selector $A$는 (client 전반에서) proposal fold에서 선택되므로 certification fold와 *고정·독립*이다. certification fold에서 client $j$는 $P_j$로부터 $n_j$개의 i.i.d. 추출을 기여하고 두 정수를 보고한다:
$$
A_j=\sum_{i=1}^{n_j}\mathbf 1\{A(x_i)=1\},\qquad
K_j=\sum_{i=1}^{n_j}\mathbf 1\{A(x_i)=1,\ \hat y(x_i)\ne y_i\}.
$$
구성상 $A_j\sim\mathrm{Bin}(n_j,a_j)$, $K_j\sim\mathrm{Bin}(n_j,m_j)$이며 $K_j\le A_j$이고, 조건부로 $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$이다. client를 떠나야 하는 정보는 어떤 인증서를 쓰는지에 달려 있다(stratified는 client별 쌍, pooled는 합). Section 4.4의 수정된 privacy taxonomy를 참조하라.

**calibration은 레이블된 unknown을 포함해야 한다(명시적 가정).** *unknown reject*를 인증하려면 certification fold가 $Y=\textsf{unknown}$인, 그리고 **그렇게 레이블된** 점들을 포함해야 한다. unknown 클래스는 학습 중에는 보이지 않지만 이 작은 사후 **audit/calibration fold**에는 존재하고 레이블되어 있어야 한다. 따라서 "distribution-free"는 *calibration 분포 $Q_\lambda$에 대해*이지 전체 unknown 우주에 대한 것이 아니며, 보장은 audit fold에 표현된 unknown에 대해 성립한다. OSR 벤치마크에서는 이것이 구성상 성립한다(held-out 클래스를 학습에서 제외하고 unknown-레이블 calibration/test 예시로 사용). 배치에서는 작은 audit된 모니터링 집합에 해당한다. audit fold에 unknown이 없으면 인증서는 closed-set selective-risk 보장으로 약화된다. 우리는 양의 accept coverage $a_\lambda=\sum_j\lambda_j a_j>0$을 갖는 selector만 인증한다. coverage가 0인 selector는 배치 불가이다.

**Risk-buffered proposal.** 경험적 risk가 이미 $\alpha$에 놓여 있는(따라서 인증이 실패하는) selector를 인증하는 것을 피하기 위해, proposal fold는 $0<\gamma<1$인 경험적 buffer $\widehat R_{\mathrm{prop}}(A)\le\gamma\alpha$ 제약 하에서 $A$를 선택한다(기본 후보 $\gamma\in\{0.5,0.7,1.0\}$). 이는 중앙집중 framework로부터 계승된다.

---

## 4. Method

### 4.1 Clopper–Pearson primitives

$K\sim\mathrm{Bin}(n,p)$에 대해, 수준 $\varepsilon$에서의 일측 **상한** Clopper–Pearson 한계 [34]는
$$
U^+(K,n;\varepsilon)=\mathrm{BetaInv}\big(1-\varepsilon;\,K+1,\,n-K\big)\quad(\,=1\text{ if }K=n\,),
$$
이고 일측 **하한** 한계는
$$
L^-(K,n;\varepsilon)=\mathrm{BetaInv}\big(\varepsilon;\,K,\,n-K+1\big)\quad(\,=0\text{ if }K=0\,).
$$
이들은 모든 $p$에 대해 $\Pr\!\big(p\le U^+(K,n;\varepsilon)\big)\ge 1-\varepsilon$, $\Pr\!\big(p\ge L^-(K,n;\varepsilon)\big)\ge 1-\varepsilon$를 정확히, distribution-free로 만족한다.

### 4.2 Theorem 1 — Conditional selective-risk certificate (핵심, N1)

가장 날카로운 인증서는 client별 **conditional** selective risk $r_j=\Pr_{P_j}(\hat y(X)\ne Y\mid A(X)=1)=m_j/a_j$를 직접 다룬다. accept 수 $A_j$에 조건부로 accepted-error count는 정확히 $K_j\mid A_j\sim\mathrm{Bin}(A_j,r_j)$이므로, *관측된 accept 부분표본*에 대한 단일 Clopper–Pearson 상한이 **accept-비율 여유 없이** $r_j$를 bound한다:
$$
\bar r_j=U^+(K_j,A_j;\varepsilon)\qquad(\bar r_j:=1\text{ if }A_j=0).
$$
전역 risk를 **accept-재가중 볼록 결합**으로 쓰면 인증서가 투명해진다:
$$
R_{\mathrm{sel}}(\lambda)=\sum_j w_j(\lambda)\,r_j,\qquad
w_j(\lambda)=\frac{\lambda_j a_j}{\sum_\ell \lambda_\ell a_\ell},\quad \sum_j w_j(\lambda)=1 .
$$

**Theorem 1 (full simplex).** *$\varepsilon=\delta/J$로 두고, client 내 i.i.d. certification 표본과 certification fold에 독립인 selector $A$를 가정한다. $\Lambda=\Delta^{J-1}$, $\bar U_\Delta^{\,r}=\max_j \bar r_j$일 때,*
$$
\Pr\big(R_{\mathrm{sel}}(\lambda)\le \bar U_\Delta^{\,r}\ \text{ for all }\lambda\in\Lambda\big)\ \ge\ 1-\delta .
$$

이는 client당 **하나**의 사건만 쓰고(둘이 아님) accept 하한이 필요 없으므로, mass-ratio bound $\max_j \bar m_j/\underline a_j$(Appendix B에 유효하나 더 느슨한 baseline으로 보존)보다 균일하게 더 날카롭다. 해석은 그대로다 — **최악의 client가 기준을 정하며**, 어떤 client의 오류도 평균으로 상쇄될 수 없다 — 다만 상수가 더 작다.

**Theorem 1′ (bounded $\Lambda$, 강건 인증서 — 배치 권장).** *$\Lambda$가 알려진 진부분집합(예: 공개된 client 데이터 비율 주변의 box)일 때 worst-client 지배가 회피된다. 수준 $\varepsilon=\delta/3J$에서 $r_j\le\bar r_j$ 및 $a_j\in[\underline a_j,\bar a_j]$($\underline a_j=L^-(A_j,n_j;\varepsilon)$, $\bar a_j=U^+(A_j,n_j;\varepsilon)$)를 bound하고 다음을 둔다:*
$$
\bar U_{\Lambda}^{\,r,a}=\sup_{\lambda\in\Lambda,\ a_j\in[\underline a_j,\bar a_j]}\ \frac{\sum_j \lambda_j a_j \bar r_j}{\sum_j \lambda_j a_j}.
$$
*그러면 참 $\lambda^\star\in\Lambda$에 대해 $\Pr(R_{\mathrm{sel}}(\lambda^\star)\le \bar U_\Lambda^{\,r,a})\ge1-\delta$이다.* $a$에 대한 내부 supremum은 box 꼭짓점($a_j\in\{\underline a_j,\bar a_j\}$)에서 달성되며, 외부 supremum은 작은 linear-fractional program(Charnes–Cooper / Dinkelbach)이다.

**경계 사례(증명이 아니라 진술에 포함).** (i) *Coverage 0.* 우리는 $a_\lambda=\sum_j\lambda_j a_j>0$인 selector만 인증한다. $a_\lambda=0$이면 selector는 아무것도 accept하지 않아 **배치 불가**이다($R_{\mathrm{sel}}$ 미정의). (ii) *분모 bound 소멸.* $\inf_{\lambda\in\Lambda}\sum_j\lambda_j\underline a_j=0$이면(작은 $A_j$일 때 client가 accept해도 $\underline a_j=0$일 수 있음) 강건 인증서는 **infeasible**로 선언한다($\bar U=+\infty$). 그러한 client를 조용히 제외하지 않는다 — 그렇게 하면 worst-case 보장이 깨진다.

**증명 스케치(Theorem 1).** $A_j$에 조건부로 $K_j\sim\mathrm{Bin}(A_j,r_j)$이므로 각 $j$에 대해 $\Pr(r_j\le\bar r_j)\ge1-\varepsilon$이다(따라서 모든 $A_j$ 값에서 성립하므로 marginal하게도). $\varepsilon=\delta/J$로 $J$개 client에 대해 union을 취하면 사건 $E=\{\forall j:\ r_j\le\bar r_j\}$에 대해 $\Pr(E)\ge1-\delta$이다. $E$ 위에서 **임의의** $\lambda$에 대해 볼록 가중치 $w_j(\lambda)\ge0$, $\sum_j w_j=1$이므로 $R_{\mathrm{sel}}(\lambda)=\sum_j w_j r_j\le\sum_j w_j\bar r_j\le\max_j\bar r_j=\bar U_\Delta^{\,r}$이다. Theorem 1′의 경우 추가로 해당 사건 위에서 $a_j\in[\underline a_j,\bar a_j]$이고(이제 $\delta/3J$에서 $3J$개 사건), 허용 가능한 $(\lambda,a)$에 대한 $\sup$이 참값을 지배한다. $\square$

**비환원성(왜 이것이 corollary가 아닌 N1인가).** (a) pooling 데이터에 대한 중앙집중 CP가 아님: $\{r_j\}$가 다르므로 pooling이 무효이고 pooled accepted-error count는 binomial이 아니라 Poisson-binomial이다(Section 4.5; ablation은 Section 5). (b) Lu의 federated-conformal 인증서가 아님: 그것은 partial exchangeability 하에서 nonconformity score의 *quantile/coverage*를 제어하나, Fed-CORE는 client-mixture 불확실성 하의 **사후-선택 conditional 오류 비율**을 제어한다. calibration 통계량은 score quantile이 아니라 **conditional-binomial 비율** $K_j\mid A_j$이며, 이는 서로 다른 finite-sample 구성을 갖는 다른 범함수이다. (c) *"단지 client별 binomial CI의 Bonferroni union"도 아님.* 인증 대상은 *미지의* deployment mixture 하의 사후-선택 accepted-risk **비율** $R_{\mathrm{sel}}(\lambda)$이며 — client별 $r_j$는 accept-재가중 볼록 가중치로 결합되고, 배치 인증서(Thm 1′)는 독립 구간들의 max가 아니라 $(\lambda,a)$에 대한 *강건 linear-fractional program*이다. 기여는 이 대상을 형식화하고 *직관적* 해법(pool 후 단일 CP)이 anti-conservative함을 드러내는 데 있지, Clopper–Pearson 산술 자체가 아니다.

*경험적 점검(`exp_pooling_fail.py`, $J=5$, $\delta=0.1$).* conditional 인증서는 유효하며(coverage $0.98$–$1.00\ge0.90$) **mass-ratio baseline보다 더 날카롭다**(중앙값 simplex $\bar U^{\,r}\approx 0.37$ 대 $0.45$). bounded-$\Lambda$ box 버전(Thm 1′)은 in-box $\lambda^\star$에 대해 유효함을 유지하면서 $\bar U^{\,r,a}\approx 0.13$로 더 좁혀지며, 유효성과 $\Lambda$ 제한을 동기화하는 worst-client 지배를 모두 확인한다.

### 4.3 Theorem 2 — Federated feasibility (client별 accept-표본 threshold)

accepted-error가 0인 영역 $K_j=0$, $\Lambda=\Delta^{J-1}$에서 배치 조건 $\max_j\bar r_j\le\alpha$는, 구속하는 각 client에 대해 $U^+(0,A_j;\delta/J)\le\alpha$, 즉 $1-(\delta/J)^{1/A_j}\le\alpha$로 환원된다.

**Theorem 2.** *수준 $\alpha$에서 simplex 위 인증은, deployment mass가 무시할 수 없는 모든 client에 대해 **관측된 accept 수***
$$
A_j\ \ge\ \frac{\ln(J/\delta)}{-\ln(1-\alpha)}\ =\ \Omega\!\Big(\tfrac{\ln(J/\delta)}{\alpha}\Big)
$$
*를 요구한다. 기댓값 형태 $n_j a_j\gtrsim \ln(J/\delta)/\alpha$는 corollary로 따라온다.*

우리는 인증서가 실제로 소비하는 **관측된 $A_j$**에 대한 bound를 진술하며, 기댓값 형태는 corollary이다. 이는 중앙집중 $N_{\min}=\lceil\log\delta/\log(1-\alpha)\rceil$의 federated 유사물로, 조건이 *client별로* 성립해야 하고 $\log J$의 federation 페널티가 붙는다. 이는 client가 동시에 작고 고위험일 때 **인증-coverage 붕괴**를 예측하며, $\Lambda$를 box로 제한하면 worst-client 지배가 완화된다.

*경험적 점검.* 예측치 $\ln(J/\delta)/(-\ln(1-\alpha))\approx 78$ accept/client($J{=}5,\delta{=}0.1,\alpha{=}0.05$)는 시뮬레이션된 교차점(중앙값 $\bar U$가 $\approx 100$ accept/client 부근에서 $\alpha$ 아래로)과 일치한다. 실제 CIFAR는 메커니즘이 구속함을 확인한다. 실현 risk를 낮추려 buffer $\gamma$를 줄이면 accept 집합도 floor 아래로 굶주리게 되어(`cert_n` $500\to151$, $d{=}5$에서 client당 $\approx30<37$) UCB가 *오히려 상승*한다($0.185\to0.222$). 즉 인증 가능성은 운용점 튜닝이 아니라 feasibility가 지배한다(Section 5.1).

### 4.4 Privacy and communication (수정)

privacy 발자국은 **어떤** 인증서를 배치하는지에 달려 있다. "두 개의 secure-aggregated count만" 주장은 **pooled 인증서에만** 성립하며 stratified에는 성립하지 않는다. Table 1이 세 가지 체제를 요약한다.

**Table 1. 세 인증서 변형의 privacy taxonomy.**

| Certificate | 필요 통계량 | 서버가 알게 되는 것 | Privacy |
|---|---|---|---|
| Pooled (Prop. 3) | $(\sum_j A_j,\ \sum_j K_j)$ | 두 합계 | 표준 secure aggregation (sum-only) |
| Stratified (Thm 1/1′) | 모든 client별 $(A_j,K_j)$ | client별 count | privacy-light, sum-only **아님** |
| Grouped-stratified | group별 합 $(A_g,K_g)$ | group별 count | group 내 secure-aggregate; 조절 가능한 절충 |

Theorem 1은 **client별** count를 요구하므로 sum-only secure aggregation과 호환되지 *않는다* — 원래 주장에 대한 수정이다. 권장 절충은 **grouped-stratified 인증서**이다. client를 각 $\ge k$개로 이루어진 $G$개의 공개 stratum으로 분할하고, 각 stratum *내부*에서 count를 secure-aggregate한 뒤 $G$개 group에 대해 인증서를 돌린다. 이는 worst-group 보장을 유지하면서 $G$개의 aggregate 쌍만 공개한다. client별 count조차 federated conformal prediction의 client별 score *분포*(T-Digest)나 decentralized conformal novelty detection의 양자화된 score *함수*보다 훨씬 적게 누출한다. differentially private 변형은 count에 보정된 노이즈를 더하고 이를 흡수하도록 Clopper–Pearson 수준을 넓힌다.

### 4.5 Proposition 3 — matched-mixture calibration 하의 pooled 인증서

calibration 데이터 자체가 (크기 $n_j$의 고정 client별 stratum이 아니라) deployment mixture $Q_{\lambda^\star}$로부터 i.i.d.로 추출될 때, pooling은 worst-client 지배를 회피한다. $A=\sum_j A_j$, $K=\sum_j K_j$, $U_{\mathrm{pool}}=U^+(K,A;\delta)$로 둔다.

**Proposition 3 (matched-mixture).** *(a) i.i.d. mixture calibration과 (b) Lemma L 하에서 $\Pr(R_{\mathrm{sel}}(\lambda^\star)\le U_{\mathrm{pool}})\ge1-\delta$이며 $U_{\mathrm{pool}}\le\bar U_\Delta^{\,r}$이다.*

이는 **핵심 결과가 아니라 tightening**이며, theorem이 되기 전에 닫아야 할 두 gap이 있어 의도적으로 Proposition으로 진술된다.

*Gap 1 — Lemma L (해결됨).* Poisson-binomial accepted-error count에 적용된 binomial CP 상한이 그 평균에 대해 보수적으로 유지되는가(Poisson-binomial과 binomial tail에 대한 Hoeffding의 고전적 비교 [35] 참조)? **naive 전역** 지배 $P_{\mathrm{PB}}(S\le b)\le P_{\mathrm{Bin}(A,\bar r)}(S\le b)$가 *모든* $b$에 대해 성립한다는 것은 **거짓**이다(명시적 반례). 그러나 인증서가 사용하는 전부인 **운용 CP threshold** $b=k_\delta\le\mu$에서는 성립하므로 **Lemma L은 성립한다**. 자기완결적 2좌표 transfer 논증이 threshold 지배를 확립하며, 990개 구성에 대한 적대적 탐색은 최소 coverage $0.902/0.952/0.990\ge1-\delta$를 준다(예상대로 최악은 binomial 자신에서 달성). 무조건적 **Bernstein fallback**($\sum_i r_i(1-r_i)\le A\bar r(1-\bar r)$를 사용해 binomial 분산이 지배)이 약간 더 느슨하지만 가정 없는 대안으로 제공된다. 세부와 적대적 인증은 `experiments/fedcore/LEMMA_L_proof.md`에 있다.

*Gap 2 — roster-composition 결합.* 고정-stratum(비 i.i.d.-mixture) 추출 하에서 accept된 roster의 무작위 client 구성으로 인해 pooled 평균 $\bar r_A=\tfrac1A\sum_{\text{accepted }i}r_{c(i)}$가 목표 $R_{\mathrm{sel}}(\lambda^\star)$와 달라진다. 이를 닫으려면 가정 (a), stratified 유한모집단 보정, 또는 roster에 대한 명시적 조건화가 필요하다. Lu et al. [6]의 partial exchangeability는 이 accepted-risk 비율이 아니라 score rank/quantile에 작용하므로 그대로 이전되지 **않는다**.

**Theorem 1과 1′은 어느 gap 없이도 성립한다.** Proposition 3은 선택적 matched-mixture tightening이며 두 gap이 형식적으로 닫히기 전까지 theorem으로 승격되지 않는다.

### 4.6 Score-agnostic 보장과 deformation 현상 (N3)

Theorem 1, 1′(및 Proposition 3)의 보장은 전적으로 certification 분할 — *고정* selector 하의 $K_j\mid A_j$의 conditional-binomial 구조 — 에 의해 산출되므로, $A$를 정의하는 **임의의** score $s(\cdot)$에 대해 성립한다. score 품질은 *얼마나 많은 coverage*가 인증되는지에 영향을 줄 뿐(더 나은 score는 같은 risk에서 더 많이 accept), *risk가 제어되는지 여부*에는 결코 영향을 주지 않는다. 이것이 중요한 이유는, federated heterogeneity가 label corruption처럼 **confidence–correctness ranking을 변형**시키기 때문이다. 소수 known 클래스만 보유한 client에서는 그 클래스들이 진짜 unknown과 쉽게 혼동되므로, 어떤 단일 global score도 어딘가에서는 miscalibrated이다. Fed-CORE는 작은 신뢰 집합으로 이 변형을 *고치려* 하지 않고 *그 주위를 인증*한다. 동일한 신뢰-데이터 예산에서 인증이 수리(repair)를 능가함을 보이는 것이 핵심 경험적 주장이다(Section 5).

### 4.7 Accept된 예측은 무엇을 *위한* 것인가: 두 가지 인증된 use

인증서는 그 자체가 목적이 아니라 accept 집합의 두 가지 downstream use를 가능하게 하며, 이것이 본 논문의 "so what"이다.

**Use case A — safe automation / triage (재학습 없음).** accept $=$ 자동 처리, reject $=$ 사람에게 위임. 그러면 CertifiedCoverage@$\alpha$는 정확히 **보장 오류 $\le\alpha$에서 안전하게 자동화되는 작업량의 비율**이며, $1-\text{coverage}$는 사람 검토 부하이다. 이는 Theorem 1/1′에서 바로 읽히며 주된 배치 가치이다. 인증되지 않은 운용점(고정 confidence threshold나 FedOSR 모델의 기본값)은 $\alpha$를 *위반*하거나 같은 보장 risk에서 *더 적게* 자동화한다.

**Use case B — certified federated self-training (accept 집합을 소비).** 레이블 없는 client 데이터에 대한 accept 예측을 pseudo-label로 삼아 FedAvg에 되먹인다. 인증서는 그 **오염**(pseudo-label 오류율 $\le\alpha$, calibration 분포 기준)을 bound하므로, 모델은 naive self-training의 무한정 오염이 아니라 *증명 가능하게 bound된* 노이즈로 확장된다.

*self-training 루프 전반의 유효성 보존(미묘한 부분).* self-training은 라운드-$t$ 모델을 라운드 $t-1$에서 accept된 것에 의존하게 만들므로, **하나의 certification fold를 라운드 간 재사용하면** 인증서가 필요로 하는 독립성이 깨진다. 우리는 어려운 closed-loop 집중 문제를 **시간상 데이터 분할**로 우회한다. 신뢰 집합을 $T$개의 disjoint audit fold $\mathcal C^{(1)},\dots,\mathcal C^{(T)}$로 분할하고 라운드-$t$ selector를 신선한 fold $\mathcal C^{(t)}$에서 수준 $\delta/T$로 인증한다.

**Proposition 4 (라운드별 self-training 유효성).** *$\mathcal C^{(t)}$가 $(f_t,A_t)$와 독립이고(인덱스 $<t$인 fold와 레이블 없는 데이터만으로 $f_t,A_t$를 형성함으로써 보장), 각 라운드를 수준 $\delta/T$로 인증하면,*
$$
\Pr\big(\forall t\le T:\ R_{\mathrm{sel}}(A_t)\le \bar U^{(t)}\big)\ \ge\ 1-\delta .
$$
*따라서 주입되는 모든 pseudo-label batch는 $T$개 라운드 전체에 걸쳐 동시에 인증 오염 $\le\alpha$를 가진다.* fold 재사용 방식(closed-loop 적응성 논증이 필요)과 달리, 데이터 분할은 각 라운드를 Theorem 1/1′의 깨끗한 독립 적용으로 만든다. 대가는 feasibility이다. 각 라운드의 fold가 Theorem 2 threshold를 통과해야 하므로 $T$는 신뢰 집합 크기에 의해 제한된다 — 명시적 예산/효용 절충.

*경험적 검증(`run_selftrain_smoke.py` + 실제 CIFAR).* 계약은 직접 점검된다. $\delta/T$ 분할에서 **동시** unsafe rate는 $0.086\le\delta$인 반면, 라운드당 $\delta$를 쓰면(분할 없음) $0.386>\delta$로 부풀어 — 분할이 장식이 아니라 필요함을 확인한다. 실제 CIFAR self-training에서 **naive**(비인증) self-training은 pseudo-label 오염 $0.59$–$0.98\gg\alpha$를 주입하는 반면(인증서는 $U=1.0$으로 reject를 올바르게 권고하나 naive 규칙은 무시), **certified** self-training은 오염을 $\approx 0$으로 유지한다(Theorem-2-infeasible 라운드를 reject/중단). clean-data 정확도는 certified $>$ naive $>$ none 순서이다. 안전 계약은 실제 데이터에서 성립한다. *정확도* 상승(Use case B)은 작은 backbone이 안전 accept 여유를 거의 남기지 않아 현재로서는 미미하다(Section 5).

---

## 5. Experiments

### 5.1 실험 설정

비-IID Dirichlet 분할($d\in\{0.1,0.5,5\}$; 작을수록 더 이질적)로 FedAvg [36] 학습하고, 클래스를 held-out하여 테스트 시점 unknown으로 사용(표준 FedOSR open-set 분할)하며, corrupted-training 설정 연결을 위해 client 레이블을 선택적으로(symmetric/asymmetric) 오염시킨다. 데이터셋은 CIFAR-10/100(주), TinyImageNet, tabular FL 벤치마크(covtype)이다. 네 가지 사후 score — maximum softmax probability(MSP) [31], entropy, margin, energy [32] — 로 score-agnostic 주장을 검증한다. 표제 지표는 CertifiedCoverage@$\alpha$, 즉 $\bar U_\Lambda\le\alpha$를 인증하는 run들 사이의 accept coverage이다. FedOSR detector(FedPD/FedPD++ [1,2], FedOSS [30], FOOGD [3])는 경쟁자가 아니라 **base model**로 다룬다 — 이들은 인증서가 없고, Fed-CORE가 그 score를 사후 인증한다. 동일 대상을 인증하는 첫 방법으로서, 우리는 재해석된 가장 가까운 방법(federated CP [6], decentralized novelty FDR [29], 무효하거나 다른 범함수를 제어), 우리 자신의 변형(ablation), 중앙집중 oracle [25,26](상한)과 비교한다.

### 5.2 유효성과 비환원성

먼저 인증서가 유효하고, 명백한 대안인 pooling이 그렇지 않음을 확립한다.

*통제된 합성 연구.* ground-truth risk와 deployment mixture를 독립적으로 변화시킨 합성 client에서 모든 성질이 성립한다: heterogeneity 전반 경험적 coverage $\ge0.98$; tightness 순서 box $<$ simplex $<$ mass-ratio(모두 유효; pooled는 가장 좁지만 matched 외에서 무효); monotone CertifiedCoverage@$\alpha$ frontier; heterogeneity-붕괴 곡선이 Theorem-2 floor($\approx37$ accept/client)를 통과; 네 score 모두 유효(test_risk $\approx0.044\le\alpha$).

*Pooling은 anti-conservative(비환원성).* 자연스러운 지름길 — 모든 client의 accept된 audit 점을 pool하여 단일 Clopper–Pearson bound 적용 — 은 heterogeneity 하에서 실패한다.

![Figure 2](experiments/fedcore/figs/fig1_pooling_collapse.png)

**Figure 2. 왜 federated calibration을 pooling할 수 없는가.** deployment mixture가 calibration-matched $\lambda$에서 단일 고위험 client로 이동할 때 인증서의 경험적 coverage(저위험 client 4개 $a{=}0.7,r{=}0.02$ + 고위험 1개 $a{=}0.5,r{=}0.3$; $\delta{=}0.1$). naive pooled CP는 matched mixture에서만 유효하고 shift 하에서 $0$으로 붕괴한다($\approx0.072$를 인증하나 참 risk는 $0.165$–$0.30$); stratified conditional 인증서(Theorem 1)는 모든 mixture에 대해 $\ge1-\delta$를 유지하고, box-$\Lambda$ 인증서(Theorem 1′)는 box 내부에서 유효하다. pooled accepted-error count는 binomial이 아니라 Poisson-binomial이므로 단일 Clopper–Pearson bound는 anti-conservative하다.

*비인증 규칙은 unsafe(필요성).* 인증서가 없는 실무자에게는 두 선택지가 있고 둘 다 unsafe하다. Table 2는 임의의 유효한 방법이 $\le\delta=0.1$로 유지해야 하는 unsafe-deployment rate $\Pr(\text{deploy}\mid R_{\mathrm{sel}}>\alpha)$를 보고한다.

**Table 2. 필요성 — unsafe-deployment rate($\le\delta=0.1$ 필수).**

| 규칙 | unsafe-deploy rate | $\le\delta$? |
|---|---|---|
| naive 경험적 threshold(deploy iff $\hat r\le\alpha$), 경계에서 | 0.49 | ✗ |
| pooled CP, matched mixture | 0.07 | ✓ |
| pooled CP, shifted mixture | $\to0$ coverage (Figure 2) | ✗ |
| leaked split(certification fold에서 threshold 선택) | 0.18 | ✗ |
| **Fed-CORE (proper split)** | **0.00–0.03** | ✓ |

naive threshold는 finite-sample 노이즈를 무시하고 경계에서 약 절반의 경우 unsafe하게 배치한다; federated-CP 규칙은 accept risk가 아니라 quantile을 제어한다. proposal/certification 분할이 하중을 지탱한다: certification fold에서 threshold를 재탐색하면 배치율을 $99.8\%$로 부풀리나 unsafe rate를 $18.2\%\gg\delta$로 만든다 — 같은 fold에서 선택된 threshold의 다중검정을 인증서가 보정할 수 없기 때문이다. 올바른 분할과 conditional 인증서만이 unsafe rate를 $\le\delta$로 유지한다.

### 5.3 finite-sample feasibility law

*외견상 null과 그 원인.* 실제 CIFAR-10 ladder(12 runs, $\alpha=\delta=0.1$, 5 clients, small CNN)에서 CertifiedCoverage@$0.1$은 모든 run에서 $0$ — 정직하게 보고한다. 두 구별되는 모드가 이를 설명한다. **Mode 1**(극단 비-IID, $d{=}0.1$): 경험적 accept risk가 이미 $\alpha$를 초과하므로 어떤 방법도 안전 배치 불가, 인증서는 올바르게 거부한다. **Mode 2**(준-IID, $d{=}5$, test_risk $\approx0.08<\alpha$): 모델은 안전하나 얇은 client별 accept count가 상한을 $\alpha$ 아래로 끌어내리지 못한다 — Theorem-2 feasibility 붕괴이지 인증서 느슨함이 아니다. 시사적으로, 실현 risk를 낮추려 buffer $\gamma$를 줄이면 accept 집합도 굶주려(cert_n $500\to151$, $\approx30<37$/client) 상한이 *오히려 상승*($0.185\to0.222$)한다: 구속 레버는 운용점이 아니라 calibration 예산이다.

*계단(staircase).* $J{=}5$ client를 $G$개의 공개 group으로 재집계하면(grouped-stratified 인증서, Section 4.4) group별 accept count가 올라가 상한을 $\alpha$ 너머로 monotone하게 끌어내린다.

![Figure 3](experiments/fedcore/figs/F6_feasibility_law.png)

**Figure 3. feasibility law(Theorem 2), ResNet 5-seed band.** (a) client를 $G\in\{5,3,2,1\}$ group으로 합칠 때 worst-group certified-risk 상한 대 group별 accept count(로그축): count가 커질수록 상한이 떨어져 수백 accept 점 부근에서 $\alpha{=}0.10$을 통과하며, Theorem-2 floor($\approx37$/client)는 인증이 처음 가능해지는 지점을 표시한다. (b) 대응하는 CertifiedCoverage@$0.10$은 $0$에서 $\approx0.21$로 상승한다. 음영 band는 5 seed에 대한 $\pm1$ std이며, $\alpha{=}0.10$ 부근의 넓은 band가 우리가 정직하게 보고하는 seed-가변성이다. 이 그림은 "$\alpha{=}0.1$ null"을 정량적 법칙으로 바꾼다: 인증 가능성은 운용점이 아니라 $(\alpha-\hat r)^{-2}$ 요구에 대한 group별 표본 크기로 정해진다.

![Figure 4](experiments/fedcore/figs/F5_alpha_frontier.png)

**Figure 4. 실제-데이터 인증-coverage frontier(per-client $G{=}5$, box-$\Lambda$; CIFAR-10 $d{=}5$).** 가장 보수적인 per-client grouping에서 인증은 더 큰 risk target에서만 비자명해진다(SimpleCNN $\alpha{=}0.20$에서 $0.063$, $\alpha{=}0.25$에서 $0.193$; ResNet-18 $\alpha{=}0.25$에서 $0.316$). frontier가 monotone인 이유는 proposal측 proxy가 작은 안전 여유를 강제하기 때문이며, 공격적 운용점은 운으로 통과시키지 않고 보정된다(독립 certification fold가 유효성 보존). worst-group $G{=}2$로 합치면 같은 $\alpha$에서 달성 coverage가 올라간다(Section 5.4). 실용 운용점은 $\alpha{=}0.20$이며, $\alpha{=}0.10$은 feasibility 경계에 놓인다.

### 5.4 실제-데이터 인증 coverage

*Backbone.* CIFAR-stem ResNet-18에 **GroupNorm**을 원리적 FL normalization으로 채택한다(BatchNorm의 running statistics는 비-IID FedAvg에서 발산). GroupNorm은 $\hat r$를 낮추나 accept 집합도 함께 줄어 seed-가변 $\alpha{=}0.10$ 인증서를 강화하지 못한다; 중요한 사실은 강건하다 — 어떤 normalization에서도 false certificate가 없다.

*Headline.* $\alpha{=}0.20$에서 GroupNorm 인증서는 **다섯 seed 모두**에서 비자명하다: CertifiedCoverage $0.392\pm0.097$($d{=}5$), $0.353\pm0.130$($d{=}0.5$), false certificate $0/10$(Table 3a). 이는 현재 audit 예산 하의 finite-sample feasibility demonstration이지 safety target이 아니다. $\alpha{=}0.10$에서 worst-group 결과는 seed-가변($2$–$3/5$)으로 부차 결과로 보고한다. edge·negative cell(Table 3b)은 정확히 feasibility law가 붕괴를 예측하는 곳 — 극단 비-IID, corruption, 또는 $\hat r$가 $\alpha$에 가까운 backbone — 이다.

**Table 3a. 실제-데이터 인증 coverage — clean-data main results** (CIFAR-10, worst-group $G{=}2$ CertifiedCoverage@$\alpha$, fixed score MSP, seed 전반 mean$\pm$std; 괄호 $n_{\mathrm{cert}}/N$; ResNet-GroupNorm, cert_frac$=0.5$).

| backbone / cell | $\alpha{=}0.10$ (5-seed) | $\alpha{=}0.20$ (5-seed) | false certs |
|---|---|---|---|
| ResNet-GN $d{=}5$ clean | $0.077\pm0.097$ (2/5) | $\mathbf{0.392\pm0.097}$ (5/5) | 0 |
| ResNet-GN $d{=}0.5$ clean | $0.091\pm0.103$ (3/5) | $\mathbf{0.353\pm0.130}$ (5/5) | 0 |
| ResNet-BN $d{=}5$ clean | $0.106\pm0.098$ (3/5) | $\approx0.29$–$0.31$ (single-config) | 0 |

**Table 3b. Edge·negative 설정** (인증이 올바르게 vacuous한 경우, feasibility-law 이유).

| 설정 | CertCov@$\alpha$ | 인증되지 않는 이유 |
|---|---|---|
| ResNet $d{=}0.1$ clean | $0$ | 극단 비-IID; accept 집합 과소(Theorem-2 infeasible) |
| ResNet sym-0.35 | $0$ | corruption이 $\hat r>\alpha$로 올림(예: $d{=}0.5$에서 $0.167$) |
| SimpleCNN | $d{=}5$에서 $0.063$($\alpha{=}0.20$) | 더 느슨한 backbone; 높은 $\hat r$ |
| covtype (tabular FL) | fixed MSP에서 $0$; selection-optimistic으로 $0.10\pm0.17$ (2/5) | 선형 backbone $\hat r\approx0.14$–$0.24$로 $\alpha$ 근접 |

covtype은 동일 feasibility law를 그 경계에서 보이는 두 번째 도메인으로 보고하며, **안정적 양이 아니다**: fixed MSP에서 $0/5$를 인증하고 selection-optimistic best-of-scores 규칙 하에서만 비자명한 점에 이른다.

*우월성.* Table 4는 실용 운용점에서 Fed-CORE를 두 비인증 대안과 비교한다.

**Table 4. 우월성 — matched-risk 비교($d{=}5$, $\alpha{=}0.20$)** (실제 logit에 대한 사후). Fed-CORE는 동시에 안전하고 finite-sample 보장되는 유일한 방법이다.

| 방법 | accept coverage | 안전($\le\alpha$)? | 보장? |
|---|---|---|---|
| test-peeking oracle (MSP / energy) | $0.444$ / $0.431$ | test 레이블 사용 | ✗ |
| no-peek naive threshold | 준-IID에서 oracle과 일치; off-IID에서 $\alpha$ 위반(예: $d{=}0.5$에서 $0.201$) | ✗ | ✗ |
| **Fed-CORE** (worst-group $G{=}2$) | $0.286$ / $0.290$ (단일 구성); $0.392\pm0.097$ (5-seed) | ✓ | ✓ finite-sample |

test-peeking oracle은 더 높은 coverage에 이르나 test 레이블을 써 보장이 없고, no-peek naive threshold는 proposal과 test가 갈라지는 곳마다 $\alpha$를 위반한다. Fed-CORE는 oracle coverage의 상당 부분을 유지하면서 유일하게 안전·보장되는 선택지이다.

*Stress 축.* feasibility law는 세 축 — risk target $\alpha$(Figure 4), heterogeneity(Figure 5), corruption(Figure 6) — 을 가지며, 각 축은 $\hat r$를 $\alpha$ 너머로 밀거나 group별 count를 Theorem-2 floor 아래로 굶긴다.

![Figure 5](experiments/fedcore/figs/F7_hetero_collapse.png)

**Figure 5. Heterogeneity 축.** Dirichlet 농도 $d$가 작아질수록(더 비-IID) group별 accept count가 얇아지고 worst-group 인증서가 악화되어 극단($d{=}0.1$)에서 붕괴한다.

![Figure 6](experiments/fedcore/figs/F9_corruption_curve.png)

**Figure 6. Corruption 축.** $d\in\{0.5,5\}$에서 client측 학습-레이블 노이즈율에 대한 worst-group CertifiedCoverage@$0.20$. clean 데이터에서 비자명($d{=}5$에서 $0.31$, $d{=}0.5$에서 $0.13$)하나 노이즈율이 $\approx0.1$을 초과하면 $0$으로 붕괴한다 — 신뢰 calibration fold는 clean함에도. 원인은 축소판 thesis이다: corruption이 모델의 $\hat r$를 $\alpha$ 위로 올려 clean audit set이 인증할 안전한 것이 남지 않는다. 인증은 corrupted 모델이 더 이상 배치 불가임을 가리지 않고 드러낸다.

### 5.5 Score 및 base-model 의존성

*효율과 score-agnosticism.* 유효한 인증서 중 conditional 구성은 mass-ratio baseline보다 날카롭고 bounded-$\Lambda$ 형태는 더 날카롭다(Table 5). 보장은 모든 score에 대해 성립하며(Table 6), score는 인증되는 coverage 양만 바꾼다 — 유효성이 score 품질이 아니라 certification 분할에서 옴을 확인한다(Section 4.6).

**Table 5. 인증서 효율(중앙값 $\bar U$, 낮을수록 날카로움; $r_{\mathrm{bad}}{=}0.3$).**

| certificate | median $\bar U$ | matched 외 유효? |
|---|---|---|
| pooled (Prop. 3) | 0.073 | ✗ (matched only) |
| box-$\Lambda$ (Thm 1′) | 0.158 | ✓ |
| conditional simplex (Thm 1) | 0.383 | ✓ |
| mass-ratio baseline (App. B) | 0.473 | ✓ |

**Table 6. Score-agnostic 유효성(네 score; 모두 유효, coverage만 다름).**

| score | realized test_risk | CertCov@$\alpha$ |
|---|---|---|
| MSP / entropy / margin / energy | 0.042–0.046 ($\le\alpha$) | 0.64–0.66 |

*실제 FedOSR detector(FOOGD).* 위가 진짜 FedOSR detector가 아니라 FedAvg backbone의 MSP를 쓴다는 우려에 답하기 위해, FOOGD의 **native SM3D score** — 공유 backbone의 penultimate feature 위에서 federated denoising score matching으로 학습한 feature-공간 score model — 를 동일 인증 harness로 인증한다. backbone이 완전한 FOOGD-SAG 학습이 아니라 FedAvg이므로 이를 full이 아니라 **representative** head로 표기한다.

**Table 7. 실제 FedOSR base model 위에서의 인증** (FOOGD native SM3D score, FedAvg+MSP 대조군, 그리고 충실 재현한 full FOOGD-SAG; worst-group $G{=}2$, $d{=}5$; 인증된 모든 cell에서 $0/13$ false certificate).

| base model (kind) | $d$ | AUROC | CertCov@$0.20$ | 상태 |
|---|---|---|---|---|
| FOOGD–SM3D (representative) | $5$ | $0.689$ | $0.071\pm0.053$ (3/3) | 인증됨, multi-seed |
| FOOGD–SM3D (representative) | $0.5$ | $0.624$ | $0.014\pm0.020$ (1/3) | seed-가변 |
| FOOGD–SM3D–SAG (full) | $5$ | $0.467$ | $0$ (0/1) | 충실 재현; 예산 내 약함 |
| FedAvg+MSP (full, same backbone) | $5$ | $0.728$ | $0.350\pm0.077$ (3/3) | 대조군(harness 검증) |

**주된 실제-base-model 결과**는 representative FOOGD-SM3D head이다: Fed-CORE는 $\alpha{=}0.20$, $d{=}5$에서 그 native score를 세 seed 모두 인증($0.071\pm0.053$)하며, $0/13$ false certificate이다. 인증 coverage는 MSP 대조군보다 낮고 score 분리도(AUROC $0.69$ 대 $0.73$)를 따른다: 이 경량 구현에서 SM3D가 MSP보다 우수하다고 주장하지 않으며, 낮은 분리도는 invalid risk control이 아니라 낮은 certified coverage로 나타난다 — 실제 FedOSR detector에서의 score-agnostic thesis 그대로다. 대조군은 Section 5.4 표제($\approx0.35$)를 재현하여 harness를 검증한다.

*Full FedOSR 방법: 정직한 재현성 발견.* 우리는 full 파이프라인도 충실히 시도했다. **Full FOOGD-SAG**(WideResNet-40-2 + annealed denoising score matching + KSD/MMD Stein-증강 정규화, 동시 학습)는 방법을 재현하나 우리 6-known CIFAR open-set split·단일 GPU 예산에서 AUROC $0.467$($\approx$chance)에 그쳐 representative head보다 약하다. 원인은 FOOGD 파이프라인이 score-model 입력을 표준화하지 않고(representative 프로토콜의 결정적 fix), backbone을 동시 학습하며, 본 split의 semantic-shift가 아닌 covariate-shift OOD를 긴 sweep으로 겨냥하기 때문으로 보인다. 이를 single-seed honest negative로 보고하고 추가 GPU는 쓰지 않았다. **Full FedPD-PROSER**(WideResNet-28-10 + placeholder/dummy 분류기 + manifold-mixup)는 scratch에서 수렴하지 않았다(known-class 정확도 $0.26$ 정체, $\approx3.5$분/round) — PROSER는 closed-set 사전학습 모델 fine-tune용이라 dummy loss가 round 0부터 발산한다. 중단했고 **가짜 row를 만들지 않았다**. **FedOSS**(CIFAR loader 없는 의료영상 코드베이스)는 최고비용 옵션으로 defer한다. 시사점은 재현성의 현실이다: SOTA FedOSR 방법을 새 split에서 published 강도로 재현하는 것 자체가 비싸고 config-민감하다. 전 과정에서 Fed-CORE의 **validity는 모든 경우 성립**($0/13$ false certificate)하며, 진짜 native FedOSR score를 인증하는 robust한 vehicle은 representative 프로토콜 — 강한 공유 backbone + 방법의 진짜 score head — 이고, full 파이프라인 재현은 더 큰 compute 예산으로 남긴다.

### 5.6 Downstream use: certified self-training

인증서의 첫 downstream use인 safe automation은 CertifiedCoverage@$\alpha$ 그 자체이다(Section 4.7). 두 번째 use는 accept된 예측을 오염이 bound된 pseudo-label로 FedAvg에 되먹인다.

![Figure 7](experiments/fedcore/figs/F8_selftraining.png)

**Figure 7. Certified 대 naive federated self-training(Proposition 4, ResNet-GN $d{=}5$).** (a) 라운드별 실현 pseudo-label 오염: naive는 $0.19$–$0.67\gg\alpha$를 주입하며 증가하고, certified는 오염을 $\le\alpha$로 유지하며 Theorem-2-infeasible 라운드에서 중단한다. (b) downstream 정확도는 certified / naive / none 전반에서 비슷하다 — 보장은 정확도 상승이 아니라 오염 제어이다.

라운드별 disjoint audit fold로 $T{=}5$ 라운드에 걸쳐 Proposition-4 계약($\delta/T$)이 검증된다: 동시 unsafe rate가 분할 시 $0.086\le\delta$ 대 분할 없을 시 $0.386>\delta$(Table 8). 주장 이득은 오염 제어이며, 현재 작은 backbone은 안전 accept 여유가 거의 없어 정확도는 핵심이 아니다.

**Table 8. Proposition 4 — 라운드별 self-training 유효성(동시 unsafe rate).**

| 방식 | 동시 unsafe rate | $\le\delta$? |
|---|---|---|
| $\delta/T$ 분할 적용 | 0.086 | ✓ |
| 미적용(라운드당 $\delta$) | 0.386 | ✗ |

*방법-knob ablation.* 세 통제된 ablation이 인증서의 knob — calibration 예산, unknown 비율, client 수 — 을 탐침한다. unknown-비율 ablation은 보장의 조건이므로 본문 그림으로 승격하며(Figure 8, Section 6), calibration-예산·client-수 ablation은 Appendix C(Figure 9–10)에 둔다. 실제 CIFAR-10 logit에서 audit-예산 sweep은 calibration fold가 커질수록 worst-group 상한을 $0.58$에서 $0.18$로 끌어내리고($\alpha{=}0.10$이 최대 예산에서만 비자명, $2/5$), unknown-비율 sweep은 audit-대표성 붕괴를 재현한다(coverage $\rho{=}1$에서 $1.00$, unknown 과소대표 시 $0.005$로). 두 결과 모두 feasibility law와 audit-대표성 요구를 실제 federated logit에서 확인한다.

## 6. Limitations

stratified 인증서는 보수적이다(Clopper–Pearson 정확성 + union bound + worst-case mixture). box-$\Lambda$(Thm 1′)와 pooled(Prop. 3)는 tightness를 회복하나 각각 $\Lambda$에 대한 지식과 Section 4.5의 두 gap을 요구한다. 보장은 client별 conditional이 아니라 $Q_\lambda$에 대해 marginal이다.

**신뢰 calibration 가정(명시).** *unknown reject*를 인증하려면 certification fold가 **레이블된 unknown-클래스 점**($Y=\textsf{unknown}$)을 포함해야 한다. unknown 클래스는 학습 중에는 보이지 않지만 작은 사후 **audit/calibration fold**에는 존재하고 레이블되어야 한다. 따라서 "distribution-free"는 *calibration 분포 $Q_\lambda$에 대해*이지 전체 unknown 우주에 대한 것이 아니다. audit fold에 unknown이 없으면 보장은 OSR이 아니라 closed-set selective-risk 보장이다. OSR 벤치마크에서는 구성상 성립하며(held-out 클래스를 학습에서 제외, unknown-레이블 calibration/test 예시로 사용), 배치에서는 작은 audit된 모니터링 집합에 해당한다. Theorem 2는 그러한 audit 데이터가 얼마나 희소해질 때 인증이 infeasible해지는지를 정량화한다. **더 날카로운 요구(ablation A5, Figure 8):** audit fold는 deployment 비율 이상으로 unknown을 담아야 한다 — 과소대표는 인증서를 *anti-conservative*하게 만든다(coverage가 $1-\delta$ 아래로). 실무에서는 모니터링 집합이 라이브 스트림의 unknown 발생률을 과소표집하지 말고 추적해야 함을 뜻한다. 이는 주변적 ablation이 아니라 **보장의 조건**이므로 보충자료가 아닌 본문 그림으로 승격한다.

![Figure 8](experiments/fedcore/figs/FA5_unknown_proportion.png)

**Figure 8 (A5). audit set은 unknown을 대표해야 한다(보장의 조건).** deployment의 accept 중 unknown 비율을 $0.06$으로 고정하고 *calibration* unknown 비율을 변화시킨다. 참 deployment risk의 coverage는 calibration 비율이 deployment 비율을 **맞추거나 초과**할 때에만 $\ge1-\delta$이다($0.06$에서 $0.913$, $0.08$에서 $0.992$). **과소대표는 anti-conservative**하다($0.02$에서 $0.057$, $0.04$에서 $0.522$). CIFAR-10 logit에서의 실제-데이터 대응(Section 5.6)도 이 붕괴를 재현한다($\rho{=}1$에서 $1.00$, 과소대표 시 $0.005$로). 따라서 audit fold가 레이블된 unknown을 *포함*하는 것만으로는 부족하며, deployment 비율 이상으로 담아야 한다.

**Privacy.** Section 4.4에서 수정했듯 pooled 인증서만 sum-only이며, stratified 인증서는 client별(또는 group별) count를 필요로 한다. privacy 주장은 정보-흐름 진술이지, 노이즈 변형을 쓰지 않는 한 differential-privacy 보장이 아니다.

**Self-training use case (B).** 인증서가 보장하는 것은 주입되는 각 pseudo-label batch의 *오염*($\le\alpha$ per round, Prop. 4에 의해 $T$ 라운드 동시)이지, self-training이 반드시 정확도를 개선한다는 것이 **아니다**. 정확도 상승은 경험적 주장이다(bounded-noise 학습은 잘 동작하나 monotone 보장은 없음). 라운드별 audit-fold 분할은 feasibility를 적응성과 교환한다. $T$는 Theorem 2 fold별 threshold를 통해 신뢰 집합 크기로 상한된다. 형식적 closed-loop 유효성을 갖는 fold 재사용 방식($\delta/T$ 분할 회피)은 향후 과제로 남긴다.

---

## 7. Conclusion

Fed-CORE는 heterogeneity와 미지의 deployment mixture 하에서 성립하는 finite-sample distribution-free 보장으로 federated open-set 분류기의 accepted selective risk를 인증한다. 그 핵심은 naive pooling이 실패하는 곳에서 유효한 **stratified conditional selective-risk 인증서**(client별 conditional-binomial CP 한계의 $\max_j$, 강건 bounded-$\Lambda$ 형태 포함)이며, client별 feasibility threshold와 선택적 matched-mixture pooled tightening이 동반된다. 이 framework는 federated open-set recognition을 ranking 문제에서 *인증* 문제로 재구성한다. 작은 신뢰 audit set은 모델을 고치는 데가 아니라 그 예측 중 어떤 것을 안전하게 accept할 수 있는지 인증하는 데 쓰이며 — 그렇게 인증-안전한 예측은 보장-risk 자동화와, 증명 가능하게 bound된 오염으로 학습을 확장하는 certified federated self-training에 *사용*된다.

경험적으로, 합성 client와 실제 federated 벤치마크(CIFAR-10/100과 tabular FL task) 전반에서 인증서는 **명시된 audit-대표성 조건 하에서 시험된 모든 설정에서 유효했다 — 어떤 seed, data set, normalization에서도 false certificate 없음** — 그리고 naive pooling은 이론이 예측한 대로 mixture shift 하에서 붕괴했다. 인증 coverage는 **feasibility law**가 지배한다. $(\alpha-\hat r)^{-2}$로 스케일하는 group별 표본 threshold로, monotone grouped-stratified 계단과 heterogeneity·corruption 축으로 그려진다. 양의 결과는 CIFAR-10 ResNet-GroupNorm에서의 5-seed worst-group $\alpha{=}0.20$ 인증 coverage이다($d{=}5$에서 $0.392\pm0.097$, $d{=}0.5$에서 $0.353\pm0.130$, 둘 다 $5/5$, false certificate $0/10$). 두 번째 federated 도메인(tabular FL)은 동일 feasibility law를 그 경계에서 보이나 seed-가변이고 selection-optimistic scoring 하에서만 비자명하므로($0/5$ at fixed MSP), 두 번째 안정 양을 더하기보다 법칙을 보강한다. $\alpha{=}0.10$ worst-group 결과는 달성 가능하나 seed-가변($2$–$3/5$)이며 그대로 보고한다. 따라서 기여는 *대상과 그 finite-sample 인증서*, heterogeneity 하 pooled 무효성의 노출, 그리고 인증형 open-set 배치가 *언제* 가능한가의 특성화이지, 새로운 FedOSR 알고리즘이나 원시 정확도 상승이 아니다.

본 연구의 한계로부터 세 방향이 따른다. 첫째, matched-mixture pooled 결과(Proposition 3)는 pooled accepted-error 평균과 deployment risk 비율 사이의 roster-composition 결합 때문에 종속적으로 남는다. 이 결합을 stratified 유한모집단 보정으로 닫으면 pooled tightening을 theorem으로 승격하고 group별 audit 요구를 줄일 수 있다. 둘째, Theorem 2의 group별 표본 threshold가 경계 부근에서 $(\alpha-\hat r)^{-2}$로 증가하여 $\alpha=0.10$ 영역을 feasibility 경계에 둔다. worst-case Clopper–Pearson 폭을 경험-분산 폭으로 대체하는 Bernstein 또는 betting류 분산-적응 bound가 현실적 audit 예산에서 작은 risk target을 feasible 영역으로 들여올 것으로 기대된다. 셋째, grouped-stratified 인증서는 group별 count만 공개하므로, differentially private count-release 인증서가 여기서 제시한 정보-흐름 진술을 넘어 형식적 privacy 보장으로 가는 구체적 다음 단계가 된다.

---

## CRediT authorship contribution statement

Sanghoon Kim: Conceptualization, Methodology, Formal analysis, Software, Investigation, Writing – original draft, Writing – review and editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

This work was supported by [funding source to be completed].

## Data availability

The CIFAR-10 and CIFAR-100 data sets are publicly available. The code and the derived calibration counts used to reproduce the certificates will be made available on publication.

---

## Appendix A. Notation

| 기호 | 의미 |
|---|---|
| $J$, $j$ | client 수; client 인덱스 |
| $P_j$, $\lambda$, $\Lambda$ | client 분포; deployment mixture 가중치; 허용 가중치 집합 |
| $\hat h,\ \hat y(x)$ | federated 분류기; 그 known-클래스 점 예측 |
| $A(\cdot),\ s(\cdot)$ | accept/reject selector; 기저 score |
| $a_j,\ r_j,\ m_j$ | client별 accept 비율; conditional 오류; accepted-error mass $m_j=a_jr_j$ |
| $R_{\mathrm{sel}}(\lambda)$ | 전역 accepted selective risk $=\sum\lambda_j m_j/\sum\lambda_j a_j$ |
| $n_j,\ A_j,\ K_j$ | client별 certification 크기; accept 수; accepted-error 수 |
| $U^+,L^-$ | 일측 상/하한 Clopper–Pearson 한계 |
| $\bar r_j$ | conditional risk $r_j$에 대한 client별 상한 CP bound(주 인증서) |
| $\underline a_j,\bar a_j$ | $a_j$에 대한 client별 하/상한 CP bound(bounded-$\Lambda$ 인증서) |
| $\bar m_j$ | $m_j$에 대한 client별 상한(mass-ratio baseline, App. B) |
| $\bar U_\Delta^{\,r},\ \bar U_\Lambda^{\,r,a}$ | conditional 인증서(simplex; bounded-$\Lambda$); $\le\alpha$이면 배치 |
| $w_j(\lambda)$ | accept-재가중 mixture 가중치 $\lambda_j a_j/\sum_\ell\lambda_\ell a_\ell$ |
| $\alpha,\delta,\gamma$ | risk 허용오차; 인증서 신뢰도; proposal risk buffer |

## Appendix B. Mass-ratio baseline certificate (valid but looser)

원래 인증서는 $m_j$와 $a_j$를 따로 bound한다: $\bar m_j=U^+(K_j,n_j;\delta/2J)$, $\underline a_j=L^-(A_j,n_j;\delta/2J)$, $\bar U_\Lambda=\sup_{\lambda\in\Lambda}(\sum\lambda\bar m)/(\sum\lambda\underline a)$, simplex 닫힌 형태 $\max_j\bar m_j/\underline a_j$. 이는 유효하나(동일 monotone-transport 증명) accept-하한 여유와 $2J$중($J$중 대비) union을 따로 지불하므로 Section 4.2의 conditional 인증서보다 균일하게 더 느슨하다. sanity baseline으로만 보존하며, conditional 인증서가 주 결과이다.

## Appendix C. 보충 합성 ablation

세 가지 합성 ablation은 인증서의 knob을 직접 탐침하고 feasibility law를 확인한다. CIFAR-10 logit에서의 실제-데이터 대응은 Section 5.6에 요약되어 있다.

![Figure 9](experiments/fedcore/figs/FA4_calibration_budget.png)

**Figure 9 (A4). calibration 예산이 레버이다.** 모델 파라미터를 고정한 채 client별 calibration count를 키우면 중앙값 cert_ucb가 $\alpha{=}0.10$ 너머로 떨어지고($0.251\to0.080$) 인증 확률이 오른다($0.00\to0.98$). client당 $\approx320$ accept에서 비자명 인증으로 넘어간다 — 실용적 "audit 데이터가 얼마나 필요한가" 곡선이자 Theorem 2의 직접 확인.

![Figure 10](experiments/fedcore/figs/FJ_client_scaling.png)

**Figure 10 (J). client별 기아 / $\log J$ 페널티.** *총* calibration 예산을 고정하고 $J$ client에 나누면 worst-group cert_ucb가 상승한다($J:2\to20$에서 $0.064\to0.276$), $J{=}5$ 부근에서 $\alpha{=}0.10$을 통과 — Theorem 2의 client별 feasibility threshold가 작동: client가 많을수록 client별 fold가 얇아지고 worst-group bound가 느슨해진다. client 참여 subsampling도 동일하게 동작한다(client별 count를 낮춤).

## Appendix D. 전체 증명

이하에서 selector $A$는 proposal fold에서 고정되어 certification fold와 독립이며, 각 client 내 certification 추출은 $P_j$로부터 i.i.d.이다. $U^+(\cdot,\cdot;\varepsilon)$, $L^-(\cdot,\cdot;\varepsilon)$는 Section 4.1의 일측 Clopper–Pearson 한계로, $K\sim\mathrm{Bin}(n,p)$와 모든 $p$에 대해 $\Pr(p\le U^+(K,n;\varepsilon))\ge1-\varepsilon$, $\Pr(p\ge L^-(K,n;\varepsilon))\ge1-\varepsilon$를 만족한다.

**Lemma A (client별 bound의 conditional coverage).** *각 client $j$에 대해 $\Pr\big(r_j\le U^+(K_j,A_j;\varepsilon)\big)\ge1-\varepsilon$.*

*증명.* $A_j=a$에 조건부로 $K_j\mid A_j{=}a\sim\mathrm{Bin}(a,r_j)$이다. CP 보장으로 모든 $a$에 대해 $\Pr(r_j\le U^+(K_j,a;\varepsilon)\mid A_j{=}a)\ge1-\varepsilon$($a{=}0$일 때 $U^+:=1$). 모든 조건값에서 성립하므로 tower 성질로 marginal하게도 $\Pr(r_j\le U^+(K_j,A_j;\varepsilon))\ge1-\varepsilon$. $\square$

**Theorem 1 (full simplex).** *$\varepsilon=\delta/J$, $\bar U_\Delta^{\,r}=\max_j U^+(K_j,A_j;\delta/J)$일 때 $\Pr\big(R_{\mathrm{sel}}(\lambda)\le\bar U_\Delta^{\,r}\ \forall\lambda\in\Delta^{J-1}\big)\ge1-\delta$.*

*증명.* $\bar r_j=U^+(K_j,A_j;\delta/J)$, $E=\{\forall j:\ r_j\le\bar r_j\}$. Lemma A와 union bound로 $\Pr(E)\ge1-J\cdot(\delta/J)=1-\delta$. $\sum_j\lambda_j a_j>0$인 임의의 $\lambda$에 대해 $R_{\mathrm{sel}}(\lambda)=\sum_j w_j(\lambda)r_j$($w_j\ge0,\sum_j w_j=1$)이므로 $E$ 위에서 $R_{\mathrm{sel}}(\lambda)\le\sum_j w_j\bar r_j\le\max_j\bar r_j=\bar U_\Delta^{\,r}$. 이는 $\lambda$에 무관하므로 모든 $\lambda$에 동시 성립한다. $\square$

**Theorem 1′ (bounded $\Lambda$).** *$\varepsilon=\delta/3J$, $\bar r_j=U^+(K_j,A_j;\varepsilon)$, $\underline a_j=L^-(A_j,n_j;\varepsilon)$, $\bar a_j=U^+(A_j,n_j;\varepsilon)$, $\bar U_\Lambda^{\,r,a}=\sup_{\lambda\in\Lambda,\,a_j\in[\underline a_j,\bar a_j]}(\sum_j\lambda_j a_j\bar r_j)/(\sum_j\lambda_j a_j)$일 때 $\Pr(R_{\mathrm{sel}}(\lambda^\star)\le\bar U_\Lambda^{\,r,a})\ge1-\delta$.*

*증명.* $3J$개 사건 $\{r_j\le\bar r_j\},\{a_j\ge\underline a_j\},\{a_j\le\bar a_j\}$(각 $\ge1-\delta/3J$)의 교집합 $E'$는 $\Pr(E')\ge1-\delta$. $E'$ 위에서 참 $(\lambda^\star,a^\star)$가 feasible하고 분자에 $r_j\le\bar r_j$를 적용하면 $R_{\mathrm{sel}}(\lambda^\star)\le(\sum_j\lambda_j^\star a_j^\star\bar r_j)/(\sum_j\lambda_j^\star a_j^\star)\le\bar U_\Lambda^{\,r,a}$. 내부 supremum은 box 꼭짓점, 외부는 Charnes–Cooper로 푸는 linear-fractional program이다. $\square$

**Theorem 2 (feasibility threshold).** *$K_j=0$ 영역에서 $\max_j U^+(0,A_j;\delta/J)\le\alpha$는 $A_j\ge\ln(J/\delta)/(-\ln(1-\alpha))=\Omega(\ln(J/\delta)/\alpha)$를 요구한다.*

*증명.* $U^+(0,A_j;\varepsilon)=1-\varepsilon^{1/A_j}$($\varepsilon=\delta/J$). $1-(\delta/J)^{1/A_j}\le\alpha\Leftrightarrow\tfrac{1}{A_j}\ln(\delta/J)\ge\ln(1-\alpha)$, 두 로그가 음수이므로 $A_j\ge\ln(J/\delta)/(-\ln(1-\alpha))$. $-\ln(1-\alpha)\ge\alpha$이므로 $\Omega(\ln(J/\delta)/\alpha)$. 기댓값 형태는 $\mathbb E[A_j]=n_j a_j$로부터 따라온다. $\square$

**Proposition 4 (라운드별 self-training 유효성).** *$(f_t,A_t)$가 인덱스 $<t$인 fold·레이블 없는 데이터만으로 형성되어 $\mathcal C^{(t)}$와 독립이고 각 라운드를 $\delta/T$로 인증하면 $\Pr(\forall t\le T:\ R_{\mathrm{sel}}(A_t)\le\bar U^{(t)})\ge1-\delta$.*

*증명.* $\mathcal C^{(t)}$가 $(f_t,A_t)$와 독립이므로 $A_t$는 그 fold에 대해 고정 selector이고 Theorem 1/1′이 적용된다. $\delta/T$ 인증으로 $\Pr(R_{\mathrm{sel}}(A_t)>\bar U^{(t)})\le\delta/T$, union bound로 $\le\delta$, 여집합을 취해 결과를 얻는다. 따라서 각 주입 batch는 $T$ 라운드 전체에 걸쳐 동시에 오염 $\le\alpha$를 가진다. $\square$

## References

("NV" = author list not individually confirmed; "verify" = bibliographic details to be confirmed against the publisher record before camera-ready.)

[1] Yang et al., FedPD: Federated open set recognition with parameter disentanglement, in: ICCV, 2023.

[2] C. Yang, M. Zhu, Y. Liu, Y. Yuan, FedPD++: Enhanced federated open-set recognition with parameter disentanglement, Int. J. Comput. Vis. (2026).

[3] FOOGD: Federated collaboration for both OOD generalization and detection, in: NeurIPS, 2024, arXiv:2410.11397.

[4] E. Diao, J. Li, Z. He, Towards addressing label skews in one-shot federated learning (FedOV), in: ICLR, 2023.

[5] Wang, Liu, Guo, Dong, Wang, Huang, Zhu, Federated continual novel class learning (FedNovel), arXiv:2312.13500, 2023.

[6] K. Lu, Y. Yu, S.P. Karimireddy, M. Jordan, R. Raskar, Federated conformal predictors for distributed uncertainty quantification, in: ICML, 2023, arXiv:2305.17564.

[7] V. Plassier, M. Makni, A. Rubashevskii, E. Moulines, M. Panov, Conformal prediction for federated uncertainty quantification under label shift, in: ICML, 2023, arXiv:2306.05131.

[8] Certifiably Byzantine-robust federated conformal prediction (Rob-FCP), arXiv:2406.01960, 2024.

[9] Zhang et al., Towards unbiased training in federated open-world semi-supervised learning, in: ICML, 2023.

[10] Turning the curse of heterogeneity in FL into a blessing for OOD detection (FOSTER), in: ICLR, 2023 (author list NV).

[11] Gao, Liu, Qin, Ou, Noise-resistant federated open set recognition, in: KSEM, 2025.

[12] Adversarial compact wrapping classifier learning for open set recognition, Inf. Sci. (2024), PII S0020025524010284 (verify authors, volume, pages).

[13] Towards heterogeneous federated graph learning via structural entropy and prototype aggregation, Inf. Sci. 718 (2025) 122338 (verify authors).

[14] Personalized federated learning: A clustered distributed co-meta-learning approach, Inf. Sci. (2023), PII S0020025523010848 (verify authors, volume, pages).

[15] X. Li, S. Zhao, C. Chen, Z. Zheng, Heterogeneity-aware fair federated learning, Inf. Sci. 619 (2023) 968–986.

[16] Z. Pan, C. Li, F. Yu, S. Wang, X. Tang, J. Zhao, Balancing the trade-off between global and personalized performance in federated learning, Inf. Sci. 712 (2025) 122154.

[17] X. Yu, Z. Liu, W. Wang, Y. Sun, Clustered federated learning based on nonconvex pairwise fusion, Inf. Sci. 678 (2024) 120956.

[18] H. Yang, W. Xi, Z. Wang, Y. Shen, X. Ji, C. Sun, J. Zhao, FedRich: Towards efficient federated learning for heterogeneous clients using heuristic scheduling, Inf. Sci. 645 (2023) 119360.

[19] X. Zhou, G. Yang, Communication-efficient and privacy-preserving large-scale federated learning counteracting heterogeneity, Inf. Sci. 661 (2024) 120167.

[20] Exploiting reject option in classification for social discrimination control, Inf. Sci. (2017), PII S0020025517309830 (verify authors, volume, pages).

[21] Graph autoencoder-based unsupervised outlier detection, Inf. Sci. (2022), PII S0020025522006338 (verify authors, volume, pages).

[22] Concept drift detection with quadtree-based spatial mapping of streaming data, Inf. Sci. (2023), PII S0020025522015808 (verify authors, volume, pages).

[23] A.N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster, Conformal risk control, in: ICLR, 2024, arXiv:2208.02814.

[24] S. Bates, A.N. Angelopoulos, L. Lei, J. Malik, M.I. Jordan, Distribution-free, risk-controlling prediction sets, J. ACM 68 (2021) 1–34.

[25] Y. Xu, W. Guo, Z. Wei, Selective conformal risk control, arXiv:2512.12844, 2025.

[26] Conformal selective prediction with general risk control (SCoRE), arXiv:2603.24704 (e-value selective risk; centralized).

[27] Y. Xie, Y. Zhou, T. Liang, S. Favaro, M. Sesia, Conformal inference for open-set and imbalanced classification, arXiv:2510.13037, 2025.

[28] Classification with reject option: Distribution-free error guarantees via conformal prediction, Mach. Learn. Appl. (2025), PII S2666827025000477 (verify authors, volume, pages).

[29] Decentralized conformal novelty detection via quantized model exchange, arXiv:2605.08263, 2026.

[30] Zhu, Liao, Liu, Yuan, FedOSS: Federated open set recognition via inter-client discrepancy and collaboration, IEEE Trans. Med. Imaging (2023).

[31] D. Hendrycks, K. Gimpel, A baseline for detecting misclassified and out-of-distribution examples in neural networks, in: ICLR, 2017.

[32] W. Liu, X. Wang, J.D. Owens, Y. Li, Energy-based out-of-distribution detection, in: NeurIPS, 2020.

[33] Y. Geifman, R. El-Yaniv, SelectiveNet: A deep neural network with an integrated reject option, in: ICML, 2019.

[34] C.J. Clopper, E.S. Pearson, The use of confidence or fiducial limits illustrated in the case of the binomial, Biometrika 26 (1934) 404–413.

[35] W. Hoeffding, On the distribution of the number of successes in independent trials, Ann. Math. Statist. 27 (1956) 713–721.

[36] B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas, Communication-efficient learning of deep networks from decentralized data, in: AISTATS, 2017.
