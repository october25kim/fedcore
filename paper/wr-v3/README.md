# Fed-CORE WR-v3 numerical release

This directory is the versioned numerical release for the manuscript titled
**Fed-CORE: Finite-Sample Certification for Federated Open-Set Recognition**.
It replaces the older `paper/v18` numbers for the current manuscript. The v18
directory remains available as a historical release and is not overwritten.

## What this release reproduces

The primary analysis uses one prespecified audit realization. Within each
client's frozen certification reservoir, the draw size equals the reservoir
size and indices are sampled independently and uniformly with replacement.
Repeated indices retain their multiplicity but are not treated as new source
records. The same realized index vectors and count tensor are shared by all 12
proposal-frozen selectors and the three procedures.

At `alpha = 0.20`, `delta_r = 0.05`, and `delta_c = 0.05`, the release reproduces:

| Procedure | Certified cells | Effective certified acceptance |
|---|---:|---:|
| H, Holm/IUT | 177 / 450 | 0.083424 |
| S, simple no-client-division | 177 / 450 | 0.081659 |
| B, conservative clientwise allocation | 130 / 450 | 0.060182 |

The release additionally checks a source-ID-disjoint evaluation fold that was
used only after the 177 H-selected policies had been frozen. This evaluation is
descriptive and does not modify or extend the certificates.

## One-command verification

From the repository root, install the package dependencies and run:

```bash
make reproduce-wr-v3
```

The verifier checks the exact release file set and hashes, reconstructs every H,
S, and B decision from 154,800 selector-indexed count rows, reproduces the full
risk-target sweep and manuscript headline, verifies matched sampling accounting,
and recomputes the post-certification diagnostic totals. Any mismatch terminates
with a nonzero exit status.

See [REPRODUCE.md](REPRODUCE.md) for the verification boundary and
[PRIVACY.md](PRIVACY.md) for the release and privacy scope. The exact primary
draw is summarized in [SAMPLING_CONTRACT.md](SAMPLING_CONTRACT.md).

## Directory map

```text
artifacts/primary/    WR-v3 count tensor, per-cell decisions, summaries, and uncertainty records
artifacts/postcert/   evaluation-only per-cell and aggregate diagnostics
governing/            sealed contracts and the original analysis programs
scripts/verify_release.py
```

The governing analysis programs require private raw inputs and are included for
provenance review. The public verifier is the supported count-to-decision entry
point.
