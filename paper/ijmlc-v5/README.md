# Fed-CORE: IJMLC v5 analysis snapshot

This directory publishes the existing analysis materials associated with
`FedCORE_IJMLC_Submission_Final_v5.docx`, as supplied on 18 November 2026.
It contains frozen aggregate counts, protocols, analysis code and recorded
outputs. It does not introduce a new experiment. NICO++ is outside this release.

## Reproduce the numerical analyses

Use Python 3.12 and the dependencies below, from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Saved aggregate counts -> all 8,100 primary H/S/B decisions.
python source_package/historical/code/reconstruct_ltt.py
# Saved counts -> seven risk statistics and candidate flags.
python source_package/historical/code/reconstruct_protection_statistics.py
# Saved outcomes -> Tables 3 and 6; independent designed-count replay.
python source_package/code/replay_remaining_tables.py
# PACS counts -> all candidate flags, selections and outcome rows.
python pacs_replay/replay_counts.py
# Archived simulation summaries -> the two different B.3 targets.
python diagnostics/b3/replay.py

# These read the original frozen source_package/results files.
python code/tv_comparison.py
python code/mechanism_analysis.py
# Run this last: it rewrites source_package/results for the matched comparison.
python source_package/code/replay_table8_matched.py
```

The commands use CPU only. They deliberately retain refusals, all attempted
cells, the original confidence allocations and deterministic selection rules.
Several original programs rewrite their output files. Use a fresh checkout to
repeat the hash checks after a replay; floating-point CSV serialization may
differ across dependency versions. `SHA256SUMS.txt` describes the published
snapshot before running the programs. Check it with `shasum -a 256 -c SHA256SUMS.txt`.

`requirements.txt` records the environment used for this upload's checks;
`requirements-original.txt` retains the original analysis environment.
The report and logs are in [verification/](verification/).

## Evidence map

| Manuscript material | Governing files | Replay scope |
| --- | --- | --- |
| Table 3: seven procedures on 450 cells | `source_package/historical/evidence/protection_recheck/results/` | Counts to risk statistics/flags; outcomes to table |
| Table 4: historical traffic-budget sensitivity | `../ijar-v37/artifacts/tables/officehome_traffic_summary.csv`; original per-cell traffic data retained elsewhere in this repository | Historical summary retained; this upload does not rerun every traffic budget |
| Table 5: matched Office-Home and TV comparison | `source_package/input/`, `source_package/results/`, `protocols/TV_PROTOCOL.json`, `results/tv/` | 100 cells, 12 candidates, exact rational/60-digit numerical checks |
| Table 6: PACS | `pacs_replay/audit/`, `source_package/pacs/`, `pacs_replay/fedcore_pacs/` | 30 models; 1,512,000 flags; 126,000 fixed-procedure selections; 162,000 outcomes; 16 contrasts |
| Figure 3: mechanism analyses | `source_package/e2/`, `results/mechanism/` | 256 designed states, 204 unique realized states, 1,152 one-factor pairs |
| Appendix B.3 / Figure 5 | `diagnostics/b3/` | Reaggregation of archived simulations and reservoir-truth checks |
| Primary H/S/B comparison, including Table 10 | `source_package/historical/inputs/results/theorem_aligned_wr_450_v3/` | All 2,700 cell/alpha cases and 8,100 procedure decisions |
| Other historical appendix diagnostics | `../ijar-v37/artifacts/`, `../wr-v4/`, `../wr-v3/`, `../v18/` | Retained historical materials; not all rerun by this upload |

The historical replay filename `replay_table8_matched.py` refers to an earlier
manuscript numbering. Its seven-procedure estimated-box result is part of the
current Table 5. Table numbers in original JSON files may likewise be historical.

## Results and interpretation retained

- At alpha 0.20, H and S certify 177/450 cells and B certifies 130/450.
  Their mean effective certified acceptance is respectively 0.0834240,
  0.0816593 and 0.0601824. H and S must not be conflated.
- The matched estimated-box comparison uses a common box acceptance objective.
  Its CP-IUT-H comparator is distinct from Table 4's budget-matched full-simplex
  acceptance objective, even when their certified-cell counts coincide.
- PACS remains **HOLD**. The unfavorable ResNet-18 contrast against
  GEOMETRY_DIRECT_CP is retained. These paired bootstrap intervals are
  conditional on the fixed models and empirical audit frames.
- All one-factor designed-state pairs have zero strict reversals. Unrestricted
  same-t pair comparisons are a different analysis and retain their original
  nonzero reversal counts. Designed states are not deployment frequencies.
- B.3's union-reservoir minimum is **0.953**; the separate certification-only
  simulation gives **0.958**. Both retain 443 non-reject-all selectors. The two
  empirical targets differ and cannot be substituted for one another.

## Scope and provenance

The frozen scientific source files were copied without changing their formulas.
The new `pacs_replay/replay_counts.py` is a portable invocation of the recovered
count-only PACS certifier and analysis modules against all saved audit tensors.
`diagnostics/b3/replay.py` extracts the numerical checks from the existing B.3
audit. `PROVENANCE.json` records the copied source hashes.

This is an aggregate-count and outcome replay release. It does not regenerate
images, training splits, predictor checkpoints or logits, and cannot establish
sampling independence or source-population validity from counts alone. The
existing ISIC source-overlap limitation and other historical exclusions remain
in force. No upstream FedPD/FedOSS/FOOGD training runners are added.

The supplied manuscript is preserved byte for byte. Its Code availability
paragraph predates this upload and mentions the withdrawn historical v0.4.2 tag;
use this directory and its Git commit instead. Replacement wording is provided
in [CODE_AVAILABILITY.md](CODE_AVAILABILITY.md). Do not interpret this snapshot
as a claim that every historical raw-data experiment has been reproduced.

Source code follows the repository's MIT license. Dataset providers retain
their terms; raw datasets are not redistributed here. See the repository's
[third-party notices](../../THIRD_PARTY_NOTICES.md).
