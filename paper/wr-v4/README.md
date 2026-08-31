# wr-v4 — sensitivity, transfer, and Fed-ISIC2019 artifacts (v0.4.2, final)

Status: FINAL. Every value used by the manuscript from this directory has been
verified against a clean checkout (see FINAL_QA.json, 13/13 pass) and against
the sealed wr-v3 primary release (contract fedcore-headline-wr-v3).

## artifacts/ (manuscript mapping)
- delta_sensitivity.csv            -> Table 4 (confidence sensitivity, H/S/B x {0.90,0.95,0.98,0.99})
- bounded_lambda_fixed_summary.csv -> Table 8 (bounded-mixture sensitivity; Theorem 5 simple-simultaneous
                                     family over the traffic-fold box; eps_r = dr/(3JM) per risk-side
                                     endpoint, eps_c = dc/(JM) for the acceptance LCB; per-alpha frozen families)
- officehome_traffic.csv           -> traffic-fold draws behind the confidence box (delta_lambda = 0.02)
- auroc_vs_cert.csv                -> Figure 5 + Section 5.5 statistics
- rejectall_validity_fixed.csv     -> Section 5.3 positive-subset validity (per-client denominator corrected)
- familywise_validity.csv          -> Section 5.3 count-level familywise repeated-audit validation
                                     (450 cells x 1,000 audits x 12 members; Holm/IUT and simple simultaneous)

## fedisic/ (Section 5.9, ratified terminal artifacts, copied verbatim)
- fedisic_terminal_cells.csv, final_no_training_summary.csv,
  final_transition_matrix.csv, fedisic_eligible_32_risk_decomposition.csv

## scripts/
Generation and verification scripts. Each self-validates against the released
headline (H 177 / ECA 0.0834; FS Office-Home 79 / 0.1024) before emitting new numbers.
A count-only replay of the certifier reproduces all 1,350 archived cell-procedure
decisions exactly (FINAL_QA.json).
