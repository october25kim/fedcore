# Prospective PACS H/S/B confirmation protocol

This directory freezes the amended scientific design for the first prospectively
sealed, independently trained-model confirmation attached to the FedCORE IJAR
v34 manuscript. It does **not** authorize training or journal submission.

## Status

- Scientific protocol: `SEALED_SCIENTIFIC_PROTOCOL_NOT_EXECUTED`
- PACS training: not started
- PACS inference: not started
- Execution gate: `HOLD_IMPLEMENTATION_BINDING`
- Submission: `submission_authorized=false`

The existing IJAR v34 evidence release at tag `v0.6.4` remains unchanged. The
public v1 protocol tag is preserved as a pre-outcome superseded record. Version
2 removes the PACS-specific proposal-coverage floor, separates primary decisions
from post-decision truth, makes scientific model failures schema-valid, freezes
exact Clopper-Pearson and Holm boundary conventions, and registers complete
candidate-level and secondary denominators. This protocol is a separate
prospective record. A future READY package requires the complete 30-model
experiment, independent replay, manuscript revision, a new public evidence
binding, and final package QA.

## Why PACS

PACS provides four real visual domains that serve as client strata. The frozen
matrix contains five open-set class splits, three training seeds, and two
backbones, for 30 fresh checkpoints. All 30 entries were `NOT_RUN` when this
protocol was sealed. The parent PACS preflight used only data inventory and
synthetic optimizer checks; it performed no PACS training or inference.

## Primary question

At `alpha=0.20`, `delta_r=0.05`, `delta_c=0.05`, and the full simplex over four
domains, does the v34 H procedure provide nontrivial certified coverage on
freshly trained models, and does removing the unnecessary client-count division
produce a practically visible S-minus-B gain under identical evidence?

The primary analysis uses exactly one registered audit per model: budget 512,
128 with-replacement draws per domain, and `replicate_id=000`. The remaining 99
audit repetitions and budgets 256 and 1,024 are secondary. They cannot replace
or rescue the primary result.

## Interpretation boundary

This is a prospective independent-model confirmation conditional on one exact,
hashed PACS empirical frame. The checkpoints are freshly trained, but the 30
models share source data, class splits, and pretrained weight files. The study
does not claim 30 independent population samples, unseen-population transport,
or PACS-wide clinical or safety validity.

## Verification

From the repository root, run:

```bash
python paper/ijar-v34-prospective-pacs-hsb-v2/verify_protocol.py
```

Expected terminal status:

```text
PASS_SCIENTIFIC_PROTOCOL_SEALED_HOLD_EXECUTION
```

The verifier independently regenerates the 30 training seeds and 2,000 audit
seeds, checks the full model product and common-random-number structure, checks
the exact H/S/B tails, boundary conventions, outcome gates, primary and
secondary row counts, candidate and cell monotonicity contracts, complete file
coverage, the non-circular bundle root, and the continuing training HOLD. It
launches no experiment and imports no machine-learning package.

The primary registered outputs contain 90 procedure rows, 1,080 candidate
decision rows, and 1,440 candidate-client count rows. The secondary grid is the
full Cartesian product of three budgets and 100 replicate IDs except the single
primary pair `(512, 0)`. It therefore contains 26,910 procedure rows, 322,920
candidate decision rows, and 430,560 count rows, including explicit placeholders
for any terminal scientific model failure.

## Next gate

The only permitted next step is a no-PACS-outcome implementation-binding pass:
create a clean isolated runner, bind every code and environment byte, complete
synthetic end-to-end and independent-replay checks, and publish a new
pre-outcome execution seal. GPU training remains prohibited until that gate is
PASS and the user explicitly authorizes execution.
