# FIDELITY — FedCORE S1 v2 run (2026-09-25 KST)

This run wrote only to new directories:
- `outputs/fedcore_s1v2_prereg_20260925/`
- `outputs/fedcore_s1v2_run_20260925/`
- ubuntu `data/domainnet_s1v2/`
- ubuntu `work/s1v2/`
- scratch `/tmp/s1v2_*`

Every V1 file was only read (sha256, `torch.load` into memory, import).

## 1. Frozen artifact sha256 (re-verified after the run, 21:52 KST)

| Artifact | Location | sha256 | Expected | Match |
|---|---|---|---|---|
| V1 PROTOCOL | Mac `fedcore_v1_prereg/PROTOCOL.json` | a8c944d5…7c50 | a8c944d57246d2fc46b175982cdada76f1c797da30090a35fbb344bad0cb7c50 | ✅ |
| V1 PROTOCOL | ubuntu `work/v1_pacs_preflight/PROTOCOL.json` | a8c944d5…7c50 | same | ✅ |
| V1 PROTOCOL | box `fedcore_s1_prereg_20260923/refs/V1_PROTOCOL.json`, v2 prereg `refs/V1_PROTOCOL.json` | a8c944d5…7c50 | same | ✅ |
| V1 RESULT_LOCK | Mac `fedcore_v1_result_lock_20260918/RESULT_LOCK.json` | 6f332e5c…353b | 6f332e5ced1f0a0f1cb0befed59146a726fb9dffbd1cfb78d324932b65ce353b | ✅ |
| V1 RESULT_LOCK | ubuntu `outputs/fedcore_v1_result_lock_20260918/RESULT_LOCK.json` | 6f332e5c…353b | same | ✅ |
| V1 RESULT_LOCK | box `uploads/RESULT_LOCK.json`, `fedcore_s1_prereg_20260923/refs/RESULT_LOCK.json` | 6f332e5c…353b | same | ✅ |
| V1 DECISION / CONTRASTS | ubuntu `outputs/fedcore_v1_run/` | 027b18bc… / 0019f098… | V1 RUN_STATE `output_sha256` | ✅ |
| S1 v1 PROTOCOL | Mac, box `fedcore_s1_prereg_20260923/S1_PROTOCOL.json` | e7dc3918…3173 | e7dc391888db2abc9993e19762ff84959098a10063bc67566d2f5cbb9d1a3173 | ✅ |
| S1 v1 prereg HASHES | Mac | 6/6 OK | — | ✅ |
| S1 v1 INVALID outputs HASHES | Mac, ubuntu, box `fedcore_s1_run_20260925/HASHES.sha256` | 13/13 OK on each host | — | ✅ |
| E2 lock tarball | Mac and box `fedcore_e2_count_state_20260918.tar.gz` | 53997114…c8ce (identical on both) | — | ✅ |

E2 directory HASHES.sha256 on Mac: 22/23 OK. The one "FAILED" line is the self-referential `HASHES.sha256` entry, which is a pre-existing file structure. On the box copy, a `.pyc` listed in the manifest is also absent, which dates from when that copy was extracted before this session. No file in either E2 directory has mtime ≥ 20:40 KST today.

## 2. No modification (mtime scan, files newer than 2026-09-25 20:40 KST)
- **Mac:** `fedcore_v1_prereg`, `fedcore_v1_result_lock_20260918`, `fedcore_s1_prereg_20260923`, `fedcore_s1_run_20260925`, `fedcore_e2_count_state_20260918`, `fedcore_tmlr_latex_20260919`, `fedcore_manuscript_revision_20260918`, `fedcore_v1_manuscript_plan_20260919`, `fedcore_tmlr_package_20260918` → **0 files**
- **ubuntu:** `fedcore/` (V1 code), `outputs/fedcore_v1_run`, `fedcore_v1_result_lock_20260918`, `fedcore_v1_full_run`, `fedcore_v1_ubuntu`, `fedcore_v1_manuscript_plan_20260919`, `fedcore_s1_run_20260925`, `work/v1_pacs_preflight` → **0 files**. The `fedcore/experiments` directory mtime is 2026-09-17 (no new files). All imports ran with `PYTHONDONTWRITEBYTECODE=1`.
- **box:** E2, S1 v1 prereg/run, TMLR LaTeX/package, manuscript revision → **0 files**
- The manuscript, LaTeX, and E2 lock were not opened or edited. The V1 PROTOCOL, RESULT_LOCK, and S1 v1 protocol and INVALID outputs were only hashed.

## 3. Code and checkpoint binding
- V1 code sha256: all 8 files match V1 `RUN_STATE.json` `code_sha256`. This includes runner `4442fa54…`, analysis `a8c7d454…`, certifier `e0e44645…`, adapter `ceff2d0e…`, and synthetic `ac23ebf1…` (`RUN_BINDING.json`).
- 30/30 checkpoint sha256 match V1 `RUN_ARTIFACTS` (`refs/V1_CHECKPOINT_SHA256.txt`).
- For all 30 models, the state digest after loading equals V1 `RUN_STATE` `state_sha256`. The runner aborts on any mismatch.
- The v2 runner imports the unmodified V1 adapter (proposals/counts/draws/transforms/infer), certifier (certify/choose), and analysis (assemble/outcome_rows/contrasts/bootstrap/truth_groups/decide/per_model_support). Only three things were re-specified, all as sealed in the protocol: `protocol_constants` (which hard-codes J=4 and the V1 seed), the domain list, and `per=B//6`.

## 4. Rule compliance
- No training, fine-tuning, or head change. The person head was kept, with all logits used. No V1 parameter was re-calibrated on DomainNet.
- No sealed choice changed after data or outputs were seen. The dry-check found no reason to change anything (transform max|Δ|=0, label map = protocol).
- Data: only the 6 official csr.bu.edu zips. The Sunhyeok copy on workstation4 was not used or touched.
- workstation4: only read-only `nvidia-smi` queries. GPU0 was busy with other users' jobs every time it was checked, so it was not used. GPUs 1–3 were not touched.
- V1 was not reinterpreted as CONFIRM. There is no claim that RATIO is superior in general and no claim of geometry-only causation.
- Reproducibility: re-running audit and analysis from the same logits in a separate directory gave byte-identical OUTCOMES, CONTRASTS, BOOTSTRAP, VALIDITY_GROUPS, PER_MODEL_DELTAS, secondary contrasts, and ECA tables.
