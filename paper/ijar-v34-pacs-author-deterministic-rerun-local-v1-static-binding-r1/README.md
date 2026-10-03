# FedCORE PACS H/S/B local static control binding r1

This bundle binds the local author-executed rerun to its own five staged
inputs, one empty output directory, and one fresh no-GPU mount probe. It
contains no PACS result and does not authorize scientific execution.

Permitted description of the eventual run:
`author-executed deterministic full-denominator rerun`.
It must not be described as independent replication, prospective
confirmation, untouched evaluation, or outcome-blind confirmation.

## Lineage

- Parent: `ijar-v34-pacs-fresh-seed-hsb-v3-static-input-binding-r2`, bundle
  root `362f4e9d…db752`. The parent is a historical record only. It is not
  the active local binding and was not modified.
- `PROTOCOL_ID` inside the generated files is the legacy scientific-engine
  identifier (`…-REPLICATION-HSB-v3`). It is not a claim of replication.
- Generated from source commit `f1e27127…` (`local-v1-path-rebind`), whose
  five path constants point at `/home/sanghoon/Desktop/Workspace/Fedcore_HSB_local_v1`.

## Contents and how each was produced

| File | Produced by |
|---|---|
| `INPUT_BINDING.json` | `python -m fedcore.experiments.v3_input_binding --receipt …` over the already-staged `inputs/` (five 0444 files, directory 0555) |
| `HOST_LAUNCH_PLAN.json` | `python -m fedcore.experiments.v3_scientific_runner plan`, with the r1 `EXECUTION_GATE.json` as the prebinding gate, the local output directory, and the probe image ID. Its `execution_allowed` field is `false`. |
| `IMAGE_INSPECT.json` | `docker image inspect` of the existing image `fedcore-v3-hsb:v4-repair-4fb2d58` (`sha256:943f4ca6…aef94`, revision label `4fb2d580…`). This is a **prebinding probe image only**, not the final execution image. |
| `MOUNT_PROBE.inspect.json`, `MOUNT_PROBE.stdout.txt` | raw `docker inspect` and stdout of the probe container `fedcore-local-v1-static-mount-probe-r1`. The probe used no network, a read-only root, `/tmp` as the only tmpfs, all capabilities dropped, `no-new-privileges`, user `1000:1000`, no GPU option, no `DeviceRequests`, no host devices, and `NVIDIA_VISIBLE_DEVICES=void`. It had five read-only input mounts and one read-write output mount, and its command was `python -m fedcore.experiments.v3_mount_probe`. |
| `MOUNT_RECEIPT.json` | `python -m fedcore.experiments.v3_mount_receipt` over the raw inspect, the launch plan, the image ID, and the stdout |
| `SEAL.json`, `SHA256SUMS.txt`, `verify_binding.py` | seal and verifier (standard library only) |

No PACS record was decoded, no label was inspected, torch was not imported,
CUDA was not initialized, and no training or inference ran. The output
directory was empty before and after the probe.

## Bundle root

As in r2, the root is the SHA-256 of the `<sha256>  <filename>\n` rows in
`SHA256SUMS.txt`, sorted by filename, with the `SEAL.json` row excluded.

Run `python verify_binding.py` from this directory. If the local staged inputs
exist on the host, the verifier also re-hashes them.

Status: `PASS_LOCAL_STATIC_BINDING_HOLD_RUNTIME_VALIDATION`.
