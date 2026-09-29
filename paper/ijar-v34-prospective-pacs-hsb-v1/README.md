# Prospective PACS H/S/B confirmation protocol

This directory freezes the scientific design for the first prospectively
sealed, independently trained-model confirmation attached to the FedCORE IJAR
v34 manuscript. It does **not** authorize training or journal submission.

## Status

- Scientific protocol: `SEALED_SCIENTIFIC_PROTOCOL_NOT_EXECUTED`
- PACS training: not started
- PACS inference: not started
- Execution gate: `HOLD_IMPLEMENTATION_BINDING`
- Submission: `submission_authorized=false`

The existing IJAR v34 evidence release at tag `v0.6.4` remains unchanged. This
protocol is a separate prospective record. A future READY package requires the
complete 30-model experiment, independent replay, manuscript revision, a new
public evidence binding, and final package QA.

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
python paper/ijar-v34-prospective-pacs-hsb-v1/verify_protocol.py
```

Expected terminal status:

```text
PASS_SCIENTIFIC_PROTOCOL_SEALED_HOLD_EXECUTION
```

The verifier checks the immutable file hashes, the 30-model denominator, all
registered RNG streams, the 12-member selector family, the H/S/B primary
contract, the outcome schema, and the continuing training HOLD. It launches no
experiment and imports no machine-learning package.

## Next gate

The only permitted next step is a no-PACS-outcome implementation-binding pass:
create a clean isolated runner, bind every code and environment byte, complete
synthetic end-to-end and independent-replay checks, and publish a new
pre-outcome execution seal. GPU training remains prohibited until that gate is
PASS and the user explicitly authorizes execution.
