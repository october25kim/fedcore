# wr-v4 — agent-computed manuscript artifacts (staging for v0.4.0)

Status: STAGING. Computed from the sealed wr-v3 primary release artifacts
(theorem_aligned_wr_450_v3, contract fedcore-headline-wr-v3) and from the
ratified Fed-ISIC terminal reconciliation. Owner must verify and re-run
through the sealed pipeline before tagging v0.4.0.

## artifacts/ (manuscript ver18 mapping)
- delta_sensitivity.csv        -> Table 4 (confidence sensitivity, H/S/B x {0.90,0.95,0.98,0.99})
- bounded_lambda_cells.csv     -> Table 7 (bounded-mixture sensitivity, per-cell, m in {250,500,1000,2000})
- officehome_traffic.csv       -> traffic-fold draws behind the confidence box (delta_lambda = 0.02)
- auroc_vs_cert.csv            -> Figure 5 + Section 5.5 statistics
                                  (Spearman 0.66, Pearson 0.57, 7/45 top-decile refused,
                                   13/225 at-or-below-median certified, quartile panels)
- rejectall_validity.csv       -> Section 5.3 reject-all / positive-subset validity sentence

## fedisic/ (Section 5.9, ratified terminal artifacts, copied verbatim)
- fedisic_terminal_cells.csv                   (50-cell roster, three independent statuses)
- final_no_training_summary.csv                (cascade 50->35->32->0; taxonomy 8/9/15)
- final_transition_matrix.csv                  (F1 fixed-target IUT: no cell flips)
- fedisic_eligible_32_risk_decomposition.csv   (per-cell refusal decomposition)

## scripts/
Generation + self-validation scripts. Each reproduces the released headline
(H 177 / ECA 0.0834, FS Office-Home 79 / 0.1024) before emitting new numbers.

## Publication checklist (owner)
1. Re-run scripts/ against the sealed wr-v3 inputs; diff against artifacts/.
2. Copy wr-v4/ into the GitHub checkout as paper/wr-v4/.
3. Update Data availability target: tree/v0.4.0/paper/wr-v4 (manuscript ver18 already points there).
4. git add paper/wr-v4 && git commit -m "paper/wr-v4: ver18 artifact release" && git tag v0.4.0 && git push origin main v0.4.0
