# FedCORE IJAR v28 public release

This directory binds the IJAR manuscript claims to versioned count-level artifacts.
It contains no raw datasets, licensed images, checkpoints, or record-level logits.

## Central paired result

- Joint minus Bottleneck: 11.87 percentage points.
- Joint minus deficit greedy: 0.52 percentage points.
- Paired descriptive 95% t interval: [-0.13, 1.17] percentage points.
- Signs across nine exports: 3 improvements, 5 ties, and 1 loss.

The interval is descriptive and is not an equivalence test. New-image full-simplex
results are all zero and therefore do not rank allocation policies.

## Files

- `SAMPLING_CONTRACT.md`: target and sampling-law boundaries.
- `PROCEDURE_THEOREM_LEDGER.csv`: theorem-aligned, hybrid, and exploratory arms.
- `CLAIM_ARTIFACT.csv`: claim-to-evidence links.
- `central_finding_source.csv`: plotted paired means.
- `tinyimagenet_n3_s050_summary.csv` and `tinyimagenet_n3_s075_summary.csv`:
  completed Tiny-ImageNet aggregate rows underlying Table 10. These are stored
  aggregate verification outputs, not raw-training reproduction.
- `SHA256SUMS.txt`: package integrity.
- `verify_release.py`: checksum and claim-ledger verification using only the
  Python standard library.

## Verification

From the repository root, run:

```bash
python paper/ijar-v28/verify_release.py
make unit
FEDCORE_ALLOW_MISSING_ARTIFACTS=1 python tests/golden_check.py
```

The last command is an artifact-free partial gate when the licensed or large
frozen run files are not present. It does not claim to reproduce model training.

The versioned public release is tagged `v0.6.0`. Its theorem-facing APIs,
fail-closed numerical solver, procedure ledger, and aggregate result sources
were merged through pull request 5 after Python 3.10 and 3.12 CI passed.
