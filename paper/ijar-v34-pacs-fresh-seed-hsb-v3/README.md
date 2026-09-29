# PACS H/S/B fresh-seed replication protocol v3

**Status: corrective scientific protocol sealed; implementation binding HOLD.**

This directory prospectively registers a fresh-seed FedCORE H/S/B replication
on a previously studied fixed PACS empirical frame. It does not authorize
training, inference, manuscript submission, or an untouched-dataset claim.

## Why v3 exists

The public v2 seal incorrectly stated that no PACS training, inference, or
scientific outcome existed before its seal. A read-only cross-host audit found
a completed 30-model PACS V1 study from 2026-09-17 and a later transfer probe
that reused the frozen PACS predictors. The v2 tag remains immutable and its
public release is marked `INVALID_PRESEAL_PROVENANCE / DO NOT EXECUTE`.

Version 3 discloses those outcomes and narrows the study label to:

> Prospectively registered fresh-seed replication on a previously studied
> fixed PACS empirical frame.

The v3 checkpoints and RNG streams will be fresh relative to V1. The dataset,
source frame, domain structure, class-split family, architectures, and
initialization families were previously studied. Consequently, this study is
not an independent source-population replication, external validation, or
evidence of unseen-population transport.

## Frozen scientific contract

The v2 H/S/B numerical contract is carried forward without changing its
selectors, candidate order, estimands, confidence budgets, audit budgets,
denominators, decision gates, or negative-result policy. The exact unexecuted
v2 training and audit seed manifests are retained byte-for-byte under the
frozen v2 seed namespace. `SCIENTIFIC_CONTRACT_CARRYFORWARD.json` records the
carry-forward hashes and permitted metadata-only differences.

The primary comparison uses 30 planned checkpoints, 12 proposal-frozen
candidates, `alpha=0.20`, `delta_r=delta_c=0.05`, the full simplex over four
PACS domains, and one budget-512 audit per model. H, S, and B must consume the
same count tensor. Any evidence-equality or monotonicity violation is
`INVALID`, not a scientific negative.

## Provenance and isolation

`PRIOR_OUTCOME_LEDGER.json` identifies both known pre-v3 outcome records.
`EXECUTION_ISOLATION_CONTRACT.json` prohibits mounting the legacy Fedcore
workspace or any prior PACS outputs. The future runtime may receive only the
five hash-bound input files as file-level read-only mounts and one dedicated
scientific output directory.

The planned execution root is:

```text
/home/sanghoon/Desktop/Workspace/Fedcore_HSB_v3_exec
```

The v3 runner, dedicated image, mount proof, resume logic, output schemas,
H/S/B parity, and independent replay have not yet been implementation-bound.

## Verification

From the repository root:

```bash
python paper/ijar-v34-pacs-fresh-seed-hsb-v3/verify_protocol.py
```

The expected state is
`PASS_CORRECTIVE_SCIENTIFIC_PROTOCOL_SEALED_HOLD_EXECUTION` with
`training_authorized=false` and `submission_authorized=false`.

## Next gate

The only permitted next stage is an isolated implementation-binding pass using
PACS-free synthetic data. PACS training and inference remain prohibited until
that stage produces a public `PASS_EXECUTION_READY` seal and the user gives a
separate explicit training authorization.
