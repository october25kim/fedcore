# PR: test: enforce three CPU-script stdout goldens (+ follow-up notes)

**Branch:** `fix/golden-stdout-gate` → `main`
**Title (copy):** `test: enforce three CPU-script stdout goldens (+ follow-up notes)`

---

## Body (copy from here down)

### What
- Wire the three CPU-script stdout goldens (`exp_lemma_L`, `exp_pooling_fail`, `run_smoke`) into
  `tests/golden_check.py` so `make repro-check` actually enforces their output. These snapshots
  existed but were never diffed by any gate, even though the docstring claimed they were.
- Add tolerant comparison (whitespace-normalized, floats compared within `1e-6`, remaining text exact)
  plus **path canonicalization** (`_canon_paths`: any absolute path → `<ROOT>/<repo-relative>`,
  anchored on the repo top-level; non-repo paths untouched) so the goldens are **portable across clone
  locations** — the previous `run_smoke` "saved …smoke_results.csv" line baked in an absolute path and
  would fail from any other clone directory.
- Correct the `golden_check.py` docstring to match the now-true behavior.
- Record two follow-ups in `HANDOFF.md` (docs only): (1) lazy-import `torchvision` inside the training
  entry points so `--help` works outside the GPU container; (2) make `run_smoke` print a repo-relative
  path at the source (deferred — code change + golden re-snapshot; the gate canonicalizes paths for now).

### Why
Closes a verification-coverage gap: the scripts' output bit-identity was documented as enforced but
was not — the same class of gap that previously let a broken plotting entry point slip past a "green"
run. The underlying numeric goldens (`certificate_math`, `scores_selector`, `split_determinism`) were
already enforced; this adds the script-level stdout layer, made portable.

### Scope / safety
- **TEST-HARNESS + DOCS only.** No change to any experiment number, metric, schema name, RNG seed,
  threshold, split logic, or golden value. Golden files untouched (`run_smoke` mismatch was resolved by
  a comparator normalization, not a re-snapshot).
- `git diff --stat`: 2 files, `+105 / −5` (`tests/golden_check.py`, `HANDOFF.md`).

### Verification
- Adversarial sanity of the comparator: path-only diffs (incl. a foreign root) → 0 fail, suffix
  preserved; float drift `5e-5` detected; `5e-7` passes; text / line-count changes detected.
- `make repro-check` → **PASS**: 3 JSON goldens + 3 stdout checks reported `RAN/PASS`; artifact-dependent
  checks (`certify_frozen`, `*_agg`) still **LOUD-SKIP** in a bare clone (`runs/` artifacts gitignored).
- Torch entry `--help` (`run_cifar` / `run_foogd_cifar` / `run_selftrain_pkg`) verified `exit 0`
  **inside** the GPU pytorch container (torchvision 0.18.0); failure outside the container is by design
  (torchvision is container-provided).

### Commits
- `test: enforce the three CPU-script stdout goldens in golden_check` (`tests/golden_check.py`)
- `docs: record torchvision lazy-import and run_smoke path follow-ups` (`HANDOFF.md`)
