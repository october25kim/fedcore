# Reference candidates to reach 30–40 (Information Sciences submission)

Current state of `Fed-CORE_draft.md`: **27 references, 8 from Information Sciences (30%).**
At the 40-reference ceiling, 8 INS already = 20%, so the INS share is satisfied across the
whole 30–40 range. To grow the list, draw from the two pools below; both are real papers
(INS pool = your uploaded PDFs, verified title/volume/year; foundational pool = standard
works this paper genuinely builds on). Add only what is genuinely relevant — no padding.

---

## Pool A — Information Sciences papers you uploaded (ready to cite)

All confirmed INS (ISSN 0020-0255). Two are already in the draft as [26], [27]. The rest are
ready to drop in; relevance is to Fed-CORE's federated-heterogeneity / detection / privacy
framing. "Slot" = where it would be cited in Section 2.

| Relevance | Citation | Slot |
|---|---|---|
| ALREADY [26] | X. Li, S. Zhao, C. Chen, Z. Zheng, Heterogeneity-aware fair federated learning, Inf. Sci. 619 (2023) 968–986. | heterogeneity |
| ALREADY [27] | X. Yu, Z. Liu, W. Wang, Y. Sun, Clustered federated learning based on nonconvex pairwise fusion, Inf. Sci. 678 (2024) 120956. | client grouping |
| High | Z. Pan, C. Li, F. Yu, et al., Balancing the trade-off between global and personalized performance in federated learning, Inf. Sci. 712 (2025) 122154. | personalization vs global |
| High | X. Zhou, G. Yang, Communication-efficient and privacy-preserving large-scale federated learning counteracting heterogeneity, Inf. Sci. 661 (2024) 120167. | privacy + heterogeneity |
| High | P. Ren, K. Qi, J. Li, T. Yan, Q. Dai, CosPer: An adaptive personalized approach for enhancing fairness and robustness of federated learning, Inf. Sci. 675 (2024) 120760. | personalization/robustness |
| High | H. Yang, W. Xi, Z. Wang, et al., FedRich: Towards efficient federated learning for heterogeneous clients using heuristic scheduling, Inf. Sci. 645 (2023) 119360. | client heterogeneity |
| High | S. Zhang, T. Xu, J. Zhu, et al., Privacy-preserving MTS anomaly detection for network devices through federated learning, Inf. Sci. 690 (2025) 121590. | federated anomaly/novelty + privacy |
| High | X. Zhou, Q. Zhi, Z. Liu, et al., FedPDA: Personalized federated learning based on attribute similarity migration, Inf. Sci. 720 (2025) 122553. | non-IID personalization |
| Med | W. Zhang, D. Deng, X. Wu, et al., An adaptive asynchronous federated learning framework for heterogeneous Internet of Things, Inf. Sci. 689 (2025) 121458. | systems heterogeneity |
| Med | F. Li, X. Chen, Z. Han, et al., Federated learning via reweighting information bottleneck with domain generalization, Inf. Sci. 677 (2024) 120825. | distribution shift / DG |
| Med | Q. Min, F. Luo, W. Dong, et al., Communication-efficient federated learning via personalized filter pruning, Inf. Sci. 678 (2024) 121030. | efficiency |
| Med | C. Chen, Z. Xu, W. Hu, Z. Zheng, J. Zhang, FedGL: Federated graph learning framework with global self-supervision, Inf. Sci. 657 (2024) 119976. | federated graph |
| Med | J. Li, J. Wang, Y. Hao, Federated dual averaging learning algorithm with delayed gradients for composite optimization, Inf. Sci. 689 (2025) 121223. | FL optimization |
| Low | J. Cai, B. Chen, J. Wen, et al., A joint vehicular device scheduling and uncertain resource management scheme for federated learning in Internet of Vehicles, Inf. Sci. 690 (2025) 121552. | edge/IoV (only if needed) |
| Low | H.-S. Kang, Z.-Y. Chai, Y.-L. Li, et al., Edge computing in Internet of Vehicles: A federated learning method based on Stackelberg dynamic game, Inf. Sci. 689 (2025) 121452. | edge/IoV |
| Low | Y. Cheng, Y. Hu, Federated learning with adaptive local aggregation for privacy-aware recommender systems in Internet of Vehicles, Inf. Sci. 710 (2025) 122100. | recsys (only if needed) |
| Low | Y. Li, B. Jin, X. Li, et al., FedRL-Hybrid: A federated hybrid reinforcement learning approach, Inf. Sci. 710 (2025) 122102. | federated RL |
| Low | X. Dong, J. Zeng, J. Wen, M. Gao, W. Zhou, SFL: A semantic-based federated learning method for POI recommendation, Inf. Sci. 679 (2024) 121057. | recsys |

Not INS (do NOT count toward the INS share; cite only if specifically useful):
- Z. Li, Z. Zhong, P. Zuo, H. Zhao, A personalized federated learning method based on the residual multi-head attention mechanism, J. King Saud Univ. Comput. Inf. Sci. 36 (2024) 102043.
- Federated learning with hyper-... , J. King Saud Univ. Comput. Inf. Sci. (2023).

## Pool B — foundational / method works this paper should cite (non-INS, real)

These are standard works Fed-CORE directly builds on; several are arguably must-cite.
Confirm exact pages/venue before camera-ready.

- **Clopper & Pearson, The use of confidence or fiducial limits illustrated in the case of the binomial, Biometrika 26 (1934) 404–413.** — the exact interval the certificate uses; SHOULD be cited regardless of count.
- B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas, Communication-efficient learning of deep networks from decentralized data (FedAvg), AISTATS 2017. — federated baseline we train.
- D. Hendrycks, K. Gimpel, A baseline for detecting misclassified and out-of-distribution examples (MSP), ICLR 2017. — the MSP score we use.
- W. Liu, X. Wang, J. Owens, Y. Li, Energy-based out-of-distribution detection, NeurIPS 2020. — the energy score we use.
- Y. Geifman, R. El-Yaniv, SelectiveNet: A deep neural network with an integrated reject option, ICML 2019. — selective classification.
- Y. Geifman, R. El-Yaniv, Selective classification for deep neural networks, NeurIPS 2017. — selective risk–coverage.
- A. Bendale, T. Boult, Towards open set deep networks (OpenMax), CVPR 2016. — OSR foundation.
- W. Scheirer, A. Rocha, A. Sapkota, T. Boult, Toward open set recognition, IEEE TPAMI 35 (2013) 1757–1772. — OSR definition.
- V. Vovk, A. Gammerman, G. Shafer, Algorithmic Learning in a Random World, Springer, 2005. — conformal prediction foundation.
- S. Bates, A. Angelopoulos, L. Lei, J. Malik, M. Jordan, Distribution-free, risk-controlling prediction sets (RCPS), J. ACM 68 (2021). — distribution-free risk control.

## Suggested plan

Current = 27 refs (8 INS, 30%).

- To **32 refs**: add Clopper–Pearson + FedAvg + MSP + energy (Pool B) + 1 high-relevance INS (Pool A). → 9 INS / 32 ≈ 28%.
- To **35 refs**: + SelectiveNet, OpenMax, RCPS. → 9 INS / 35 ≈ 26%.
- To **40 refs**: + 2 more high-relevance INS (Pool A) + Scheirer + Vovk + Geifman-2017. → 11 INS / 40 ≈ 28%.

Every plan stays ≥20% INS. Tell me the target count and I will insert the chosen entries,
cite each in the relevant Section-2 sentence, and rebuild the docx.

> Recommendation: add the four **must-cite** Pool-B items (Clopper–Pearson, FedAvg, MSP,
> energy) right away — the paper already uses all four and currently does not cite them.
