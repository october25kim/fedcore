# Reproduction boundary

After public tag `v0.8.0` is available, check it out and run:

```bash
python paper/ijar-v38/verify_release.py
make unit
```

The release verifier reproduces the manuscript-to-artifact map, registered
count-to-decision claims, matched-comparator aggregates, external fixed-frame
decisions, and PACS execution provenance. It does not silently convert a fixed
frame into a population sample or a released checkpoint into an independently
repeated training campaign.

## Core 450-condition analyses

Figure 2 uses 443 proposal-feasible target-matched validation conditions.
Seven proposal-infeasible reject-all placeholders remain only in the full
450-condition utility roster.

The same-target comparator package holds fixed the full-simplex credal set,
predictor, 12-member selector family, realized audit counts, alpha, risk and
coverage budgets, Holm correction, and coverage lower bound. Recompute its
published aggregates from:

- `artifacts/comparators/full_simplex_fixed_prior_betting/contract.json`;
- `artifacts/comparators/full_simplex_fixed_prior_betting/per_cell_eca.csv`;
- `artifacts/comparators/full_simplex_fixed_prior_betting/summary.csv`.

The registered result at alpha 0.20 is H 177/450 versus fixed-prior betting
81/450, with mean effective certified acceptance of 8.3424% versus 3.9676%.
The result is paired on a fixed roster and is not an independent population
comparison or a claim against all possible betting procedures.

## External fixed-frame evaluations

`artifacts/external/vlcs/` retains the full primary denominator and its `KILL`
decision. The protocol was locally frozen before training and prediction, but
was not externally timestamped. Raw images, logits, and checkpoints are not
redistributed.

`artifacts/external/domainnet/` retains the protocol sealed before image
download, decoding, and prediction, the 12,027-image source binding, complete
primary denominator, analysis code, and `KILL` / `NOT_TRANSFER` decision. The
1,024-draw analysis is secondary and descriptive. Raw images, logits, and
audit arrays are not redistributed.

Neither directory supports a positive untouched-population utility claim.
They are reproducible adverse fixed-frame analyses.

## PACS end-to-end package

`reproduction/pacs_v3/` binds the complete 30-model campaign to source commit
`fa016c199a7c846bbe20b64d54ab34164adf7647`, source tree
`8adbb6fea6b5f7eef4ea8e7f78b484103bf08024`, and scientific image
`sha256:62c35527e56f5a05c026598ffda355e2b2b80256d966e5d9202ba639a5bfde8a`.
The release carries five separately hashed assets: an exact source archive,
the immutable image split into two parts, 15 ResNet18 checkpoints, and 15
ConvNeXt-Tiny checkpoints.

Verify downloaded assets with:

```bash
cd paper/ijar-v38/reproduction/pacs_v3
./verify_release_assets.sh /path/to/downloaded/release-assets
```

Then follow `README_E2E.md`. Exact-identity replay requires five pinned inputs
listed in `INPUTS.sha256`. The raw PACS archive is not redistributed because a
single dataset-wide redistribution license was not established. The two
pretrained initialization files are available from the official TorchVision
URLs listed in `THIRD_PARTY_NOTICES.md`.

The included assets enable checkpoint audit and a new 30-cell rerun. This
release does not claim that an independent clean-machine rerun was executed.
The governing registered outcome remains `VALID_NEGATIVE`: 30/30 models were
accounted for, zero terminal model failures occurred, and H, S, and B each
certified three models.

## Exact-identity versus scientific reproduction

- `COUNT_TO_DECISION_REPLAY`: recompute certificates from released counts.
- `EXACT_IDENTITY_AUDIT`: verify source, image, inputs, checkpoints, receipts,
  denominator, and outcomes by hash.
- `PORTABLE_SCIENTIFIC_REPRODUCTION`: execute a new 30-cell campaign on
  compatible hardware using the pinned contract.

Only the first two levels are completed by the release itself. The third is
enabled, not claimed as already completed.
