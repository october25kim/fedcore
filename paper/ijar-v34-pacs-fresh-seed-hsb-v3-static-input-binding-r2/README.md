# FedCORE PACS H/S/B v3 static input and runtime binding r2

This bundle binds the sealed PACS v3 protocol to one public source snapshot,
one immutable container image, and exactly five staged scientific inputs. It
contains no PACS result and does not authorize scientific execution.

## Passed scope

- The source and staged copies of all five registered inputs have identical
  byte sizes and SHA-256 values.
- The isolated input directory contains exactly those five regular,
  nonsymlink files. Each staged file has mode `0444`.
- The static host plan fixes 30 model cells, the immutable image ID, the exact
  five read-only input mounts, one read-write output mount, and one restricted
  `/tmp` mount. Its `execution_allowed` field is `false`.
- A no-GPU probe ran with no network, a read-only root filesystem, no host
  devices, no NVIDIA runtime, and `NVIDIA_VISIBLE_DEVICES=void`. Writes to all
  five inputs and the image root failed. Writes to `/tmp` and `/output`
  succeeded and were removed. The scientific output directory remained empty.
- A separate CPU-only check verified all five hashes and strict-loaded the two
  pretrained state dictionaries into fresh four-class heads. It did not decode
  PACS records, inspect labels, run inference, or train a model.
- The exact public source commit passed 371 tests; 62 dependency-specific tests
  were skipped and no test failed.

## Deliberate hold

The public runner is a static, non-executing launch plan. The complete
training-to-finalization campaign is not wired into it, and artifact-level
readiness and run-authorization documents are not revalidated inside the
scientific container. The training entrypoint therefore fails closed. No
`PASS_EXECUTION_READY` or `RUN_AUTHORIZATION.json` exists.

The governing status is
`PASS_FIVE_INPUT_AND_STATIC_RUNTIME_BINDING / HOLD_END_TO_END_SCIENTIFIC_ORCHESTRATOR_AND_AUTHORIZATION`.

Run `python verify_binding.py` from this directory, or pass its full path from
the repository root, to verify every bundle artifact and bound source file.
