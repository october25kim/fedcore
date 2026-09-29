# FedCORE PACS H/S/B v3 execution binding r1

This bundle binds the publicly sealed v3 scientific contract to a clean source
snapshot, a dedicated container image, and PACS-free synthetic integration
evidence. It does **not** authorize PACS training or inference.

## Passed scope

- the H, S, and B procedures consumed one identical count tensor per cell;
- the 30-cell synthetic denominator produced 360 proposal rows, 1,440 count
  rows, 1,080 candidate-decision rows, and 90 procedure rows;
- an independent implementation replayed all 1,080 candidate decisions with no
  mismatch;
- a seven-cell interrupted run resumed to the same artifact root as an
  uninterrupted run;
- terminal-model-failure, proposal-infeasible, and risk-refusal placeholders
  remained in their registered denominators;
- the dedicated container used no PACS input, no GPU device request, no network,
  and no legacy-workspace mount;
- the scientific output directory remained empty.

## Deliberate hold

The scientific runner has not been bound, the five scientific input files have
not been copied or mounted into the isolated root, and no PACS training or
inference has started. The current gate is therefore
`PASS_SYNTHETIC_IMPLEMENTATION_BINDING / HOLD_SCIENTIFIC_RUNNER_AND_INPUT_BINDING`.

The synthetic ECA values are test fixtures only. They are not PACS results and
must not appear as empirical evidence in the manuscript.

Run `python verify_binding.py` from this directory, or pass its full path from
the repository root, to verify the bundle root and every bound source file.
