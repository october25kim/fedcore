# FedCORE IJAR v34 public release

This evidence release binds the IJAR v34 manuscript to the exact standalone
figures and aggregate/count-level sources used by seven Figures, seven numbered
Tables, two Abstract evidence statements, and eight retained quantitative prose
claims. It supersedes `paper/ijar-v33` only as the current manuscript binding.
Historical releases remain unchanged.

## Binding

- Manuscript version: v34
- Manuscript SHA-256: `3fe147f9670c1958da928cd6a45d8b03506098d83f613160c30f57b2e3964ded`
- Rendered PDF SHA-256: `745552f6927743a100b8ed66e8e28c79e8da276b431a5cac65ab1efe7d1b0ba5`
- Release tag: `v0.6.4`
- Release asset: `FedCORE_IJAR_v34_evidence_release.zip`
- Ledger: 24 unique manuscript elements in `CLAIM_ARTIFACT.csv`

## Scientific delta from v33

No new empirical observations were introduced. The v34 manuscript adds the
fixed-pooling reducibility boundary, makes the valid same-target H/S/B
confidence-allocation comparison primary, and restricts pooled calculations to
non-certifying mechanism diagnostics under audit-target mismatch. H and S use
the same full-simplex target as the valid conservative B reference and share the
same candidate family, confidence budget, and realized count tensor.

## Evidence boundary

The release reproduces count-to-decision calculations and compact aggregate
sources. It does not reproduce model training, licensed datasets, checkpoints,
record-level logits, or every record-level analysis. The 450 conditions are a
fixed roster. Empirical-reservoir, finite-frame, bootstrap, planning, and
source-disjoint results retain the scope restrictions in the manuscript and
ledger. A pooled fixed-quota calculation is not a valid comparator for the
protected worst-client target and is retained only as a mechanism diagnostic,
not a method ranking.

## Verification

From the repository root, run:

```bash
python paper/ijar-v34/verify_release.py
make unit
```

The release verifier checks all package checksums, the 7F+7T+2H+8C ledger,
the v34 manuscript/PDF binding, H/S/B decision and ECA parity, the exact
H-minus-B decomposition, the 443/7 validity split, the valid conservative
reference, and mechanism-only pooled framing.
