# Reproducing the WR-v3 numerical release

## Supported command

Run the following command from the repository root:

```bash
make reproduce-wr-v3
```

The command requires Python, NumPy, and SciPy. It does not require a GPU,
Dataset download, checkpoint, raw logit file, or record-level label.

## Verified chain

The public verifier performs the following checks:

1. It verifies the complete release file set against `SHA256SUMS`.
2. It checks the sealed WR-v3 contract and the primary and evaluation validation records.
3. It validates all 154,800 released count rows and their cellwise tensor hashes.
4. It independently reconstructs the H, S, and B procedures for all 450 cells and six risk targets.
5. It reproduces every per-cell selection, certificate decision, coverage lower bound, risk upper bound where applicable, and Holm adjusted p-value.
6. It reproduces the full sweep and the manuscript headline of 177, 177, and 130 certified cells.
7. It verifies that each primary draw size equals its client reservoir size and checks the released uniqueness accounting.
8. It confirms that the post-certification evaluation used the frozen H-selected set, had no source-ID overlap, and reproduces its component totals.

## Reproducibility boundary

This is a **count-to-decision release**. The public package reproduces the
statistical decisions from the exact released `(A, K, n)` records. It does not
retrain the models or reconstruct the count tensor from raw images, logits,
checkpoints, or record-level labels. Those inputs are excluded because of data
licensing, size, and privacy constraints.

The governing scripts document how the private raw artifacts produced the
released counts. They are provenance records rather than the public entry point.
The released hashes bind the numerical evidence used by the manuscript.

The primary claim is conditional on frozen empirical certification reservoirs.
It is not a claim about an unobserved future deployment population. The
post-certification evaluation is descriptive and is not a second certificate.
