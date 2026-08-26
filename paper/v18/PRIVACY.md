# Public-data and privacy boundary

Fed-CORE certification can be evaluated from aggregate client-level counts.
The public v18 release follows that boundary.

## Included

- Per-client audit sizes `n`.
- Per-client accepted counts `A`.
- Per-client accepted-error counts `K`.
- Non-sensitive totals for unique source and accepted-source accounting.
- Selector metadata, confidence allocations, decision outputs, summary
  statistics, seed manifests, and manuscript-ready Tables and Figures.
- Training and checkpoint metadata needed to interpret each experimental cell.
- Nine illustrative PathMNIST patches embedded in the non-machine-readable
  reference Figure 7 under the Dataset's attribution license.

## Excluded

- Standalone or machine-readable observation-level image arrays.
- Raw labels, logits, embeddings, checkpoints, and optimizer state.
- Per-example predictions or acceptance indicators.
- Source-level identifiers and the mapping between an observation and a
  client, label, prediction, or error event.
- Licensed Dataset contents.

The released `semantic_id` identifies an experimental cell, not a person or a
source observation. Uniqueness fields are counts only. No source-identity map is
released. Nevertheless, stratum-level counts may be sensitive in a real
deployment, especially for small strata. This research implementation does not
provide secure aggregation or differential privacy, and count release should not
be treated as a cryptographic privacy guarantee.

## Target interpretation

The validity and headline audit results target frozen, source-level
deduplicated empirical reservoirs. Repeated with-replacement draws are audit
replicates from those reservoirs. Duplicate draws do not create new source
labels, and the release does not interpret them that way.

This package therefore supports public verification of the certification
arithmetic while keeping raw and licensed data outside the repository.
