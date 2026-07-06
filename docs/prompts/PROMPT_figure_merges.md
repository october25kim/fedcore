# Claude Code prompt — figure merges + problem-diagram augmentation

Purpose. The reviewer asked for three figure changes that tighten the visual story:
1. **Merge Figure 3 (feasibility staircase) + Appendix Figure 9 (calibration budget)** into one
   feasibility-law figure — "grouping is a sample-size effect, and audit budget is a sample-size
   effect" should close in one panel set.
2. **Merge Figure 5 (heterogeneity) + Figure 6 (corruption)** into one "stress axes" figure.
3. **Augment Figure 1 (problem diagram)** to show where the proposal / certification folds enter.

All three are plotting/diagram-only (CPU; no retraining). **Recombine the existing, already-logged
panels — do not invent or re-fit any curve.** Overwrite the existing PNG paths so the manuscript
picks them up with no path change; the Mac side will then renumber/merge captions.

Paste the fenced block into Claude Code in the FedCORE repo.

```text
READ CLAUDE.md AND AGENTS.md FIRST. Plotting/diagram only — CPU, no GPU, no retraining, no new
numbers. Only recombine values already in the figure generators / CSVs. Match the existing
figs/*.png family (fonts, dpi=200, half/full-column sizing). Report which source produced each panel.

TASK A — merge feasibility staircase + calibration budget -> overwrite figs/F6_feasibility_law.png (+ .pdf).
  Find the generators that currently produce F6_feasibility_law.png (the G in {5,3,2,1} grouped
  staircase + CertifiedCoverage panel, ResNet 5-seed band) and FA4_calibration_budget.png
  (synthetic per-client calibration-budget sweep, from exp_ablation_extra.py). Combine them into
  ONE figure with three panels, same data:
    (a) worst-group certified-risk upper bound vs per-group accepted count (log x), G=5,3,2,1, with
        the alpha=0.10 line and the Theorem-2 floor (~37/client);
    (b) CertifiedCoverage@0.10 vs per-group accepted count (the coverage that appears at large counts);
    (c) calibration-budget sweep: median cert_ucb (left y) and P(certified) (right y) vs per-client
        calibration count (log x), alpha=0.10 line.
  Caption message: grouping and audit budget are the same sample-size lever of Theorem 2.

TASK B — merge heterogeneity + corruption stress axes -> overwrite figs/F7_hetero_collapse.png (+ .pdf).
  Combine the heterogeneity panel (current F7_hetero_collapse: certified outcome vs Dirichlet d,
  collapsing at d=0.1; SimpleCNN stress config) and the corruption panel (current F9_corruption_curve,
  from runs/corruption_curve.csv: worst-group CertCov@0.20 vs client-side train-label noise rate,
  d in {0.5,5}, collapsing once noise > ~0.1) into ONE figure with two panels:
    (a) heterogeneity axis (vary d);  (b) corruption axis (vary noise rate).
  Keep the SimpleCNN-stress label on (a) and "trusted calibration stays clean" note on (b).
  Both share the message: each axis pushes the model's r_hat past alpha or starves the per-group count.

TASK C — augment the problem diagram -> overwrite figs/fig0_problem_diagram.png (edit the SVG, re-export).
  figs/fig0_problem_diagram.svg is editable. Keep the existing content (base FedOSR model + selector
  A(x) -> accepted / rejected; the four quantities AUROC/FPR95, federated-CP coverage, batch FDR,
  Fed-CORE R_sel). ADD a small annotation showing where the trusted folds enter:
       proposal fold  ──selects the threshold of A(x)──┐
       certification fold ──certifies R_sel on a disjoint fold──┘
  i.e. an arrow/label into the selector A(x) box reading "proposal fold selects threshold" and a
  label into the risk-certification box reading "certification fold certifies risk (independent)".
  Re-export to PNG at the same canvas size (~780x460) and dpi.

REPORT (fixed format): 진단 요약 / 확인한 명령 / 핵심 결과 (각 PNG에 어떤 generator·CSV·panel을
넣었는지, 덮어쓴 파일 경로) / 판정 / 다음 행동. Confirm the three overwritten files exist:
figs/F6_feasibility_law.png (3 panels), figs/F7_hetero_collapse.png (2 panels), figs/fig0_problem_diagram.png (augmented).
Do NOT delete FA4_calibration_budget.png or F9_corruption_curve.png (kept as standalone sources);
just produce the merged composites at the paths above.
```

---

### After the PNGs land (Mac side — I will do this)
Once `F6_feasibility_law.png` (3-panel), `F7_hetero_collapse.png` (2-panel), and the augmented
`fig0_problem_diagram.png` are synced, I will, in `Fed-CORE_draft.md`:
- update **Figure 3** caption to describe the merged staircase + budget panels, and **remove the
  now-redundant Appendix-C calibration-budget figure** (Appendix C then holds only client-scaling);
- update **Figure 5** caption to the merged "stress axes (heterogeneity + corruption)" and **remove
  the separate corruption figure block**, renumbering the remaining figures (audit-representativeness
  and self-training shift up by one) so the main set is the reviewer's 7-figure package;
- update **Figure 1** caption to mention the proposal/certification folds shown in the diagram;
then rebuild with `bash build_docx.sh`. No content/number changes — only figure consolidation.

### Notes for Sanghoon
- 순수 plotting/SVG 편집입니다 — 새 학습·새 숫자 없음, 기존 패널 재조합만.
- 파일 경로를 그대로 덮어쓰게 해서, PNG가 들어오면 제가 caption 병합 + figure 재번호만 하면 됩니다.
- 결과가 동기화되면 알려 주세요. 제가 draft 통합(7-figure 패키지)을 마무리하겠습니다.
