# HANDOFF.md — Fed-CORE runbook & state (2026-06-26)

Hand-off from the research/writing session (Mac, no GPU) to the **4070 Ubuntu**
execution environment (Claude Code / Codex via Cursor). Place `CLAUDE.md`,
`AGENTS.md`, this file, and `scripts/` at the **repo root** on the 4070 (this
folder is a synced copy, not a git repo). Have the agent read `CLAUDE.md` +
`AGENTS.md` first.

---

## 0. TL;DR — start here on the 4070

```bash
# 0) ensure the repo has these files at root: CLAUDE.md AGENTS.md HANDOFF.md
#    experiments/fedcore/  scripts/docker_cifar.sh
# 1) CPU sanity (proves the certificate path; no torch needed)
python experiments/fedcore/exp_lemma_L.py
python experiments/fedcore/exp_pooling_fail.py
python experiments/fedcore/run_smoke.py
# 2) GPU smoke -> real run
bash scripts/docker_cifar.sh         # cifar10, clean, seed 0 by default
```

---

## 1. Where we are

**Direction (decided):**
- SRCC / Paper 1 (centralized selective risk control after corrupted training)
  was **abandoned** — reason: *too incremental / novelty*. Its CP machinery is
  healthy (not an empirical failure) and is reused as a special case.
- **Flagship = Fed-CORE** (Federated Certified Open-Set Recognition). Chosen over
  the RC-OWPL pseudo-labeling idea because Fed-CORE's core is a **non-reducible
  theorem**, while RC-OWPL is a Learn-then-Test-style application that risks the
  same "too incremental" verdict.

**Done (this session):**
- Meta-analysis + adversarial novelty verification → `FedOSR_meta_analysis_and_novelty_brief.md`.
- Paper draft (Intro/RW/Method + Theorems 1–3 + proof sketches + experiment plan)
  → `Fed-CORE_draft.md`.
- Certificate module + two validations:
  - **Lemma L: SUPPORTED** numerically (worst-case coverage 0.919 ≥ 0.90; the
    binomial CP is conservative for the Poisson-binomial mean). → Theorem 3 holds
    under matched-λ; formal proof still TODO.
  - **Pooling-fail ablation: confirms non-reducibility.** Naive pooled CP coverage
    **collapses to 0%** under deployment-mixture shift toward a high-risk client,
    while the stratified Theorem-1 certificate stays valid (~1.0) for every mixture.
- FedOSR pipeline scaffold (score → risk-buffered selector → stratified
  certificate → metric schema). **Smoke validated** end-to-end on fake logits:
  box-Λ certifies 3/12, simplex 0/12 (robust-but-conservative). torch CIFAR path
  written + compiles; **not yet run on GPU**.

**Not done / open:**
1. **The central experiment:** run `run_cifar.py` on real CIFAR (GPU). This is the
   only thing that answers *is certified accepted coverage non-trivial at CIFAR
   scale under non-IID corruption?*
2. Formal proof of Lemma L.
3. Fill `Fed-CORE_draft.md` §5 with real CIFAR numbers (currently proposed/placeholder).

---

## 2. Run plan (priority order) — the next milestone

Mirror the project's original ladder, now in the FedOSR setting:

1. `run_smoke.py` (fake logits) — already green; re-run to confirm on the 4070.
2. `run_cifar.py` cifar10, **clean**, seed 0, dirichlet_alpha=0.1.
3. cifar10, **symmetric 35%** client-side label corruption, seed 0.
4. cifar10, **asymmetric 20%**, seed 0.
5. seeds 1, 2 for the above.
6. dirichlet_alpha sweep {0.1, 0.5, 5} → certified-coverage-collapse curve (Thm 2).
7. CIFAR-100.

For each: report `CertifiedCoverage@alpha` per (score, gamma, Lambda), plus the
full metric schema. Use the project's result format (진단/명령/핵심결과/판정/다음행동).

**Note on corruption (implemented).** Client-side label corruption is wired in
(`noise.py`; `--noise_type {symmetric,asymmetric} --noise_rate`, or `NOISE_TYPE`/
`NOISE_RATE` for `docker_cifar.sh`). It corrupts **training labels only**; the
trusted calibration/test folds stay clean by construction. Verified: symmetric
flips to a uniform other class, asymmetric flips `y -> (y+1) % n_known`, no
self-flips. Example for step 3:
`NOISE_TYPE=symmetric NOISE_RATE=0.35 bash scripts/docker_cifar.sh`.

---

## 3. What to watch / likely failure modes

- **Certified-coverage collapse.** If certified coverage is ~0 at dirichlet_alpha
  =0.1, check: (a) is any single client both small and high-risk (Thm 2 binding)?
  (b) try box-Λ instead of full simplex; (c) increase trusted calibration size per
  client; (d) smaller gamma. This collapse, characterized honestly, is itself a
  paper result — do not hide it.
- **Pooling temptation.** Do not "fix" a loose simplex bound by pooling — that is
  exactly the invalid move the ablation refutes. Use box-Λ (known client sizes).
- **Split leakage.** The single most dangerous bug. Assert disjointness of
  prop/cert/test indices and that test labels never touch proposal/certification.

---

## 4. File map (quick)

| File | Role |
|---|---|
| `Fed-CORE_draft.md` | paper draft; theorems 1–3, proof sketches, §5 experiment plan |
| `FedOSR_meta_analysis_and_novelty_brief.md` | landscape, gap, novelty verification, citations |
| `experiments/fedcore/certificates.py` | Thm 1 stratified + Thm 3 pooled certificates |
| `experiments/fedcore/exp_lemma_L.py` | Lemma L verification (CPU) |
| `experiments/fedcore/exp_pooling_fail.py` | non-reducibility ablation (CPU) |
| `experiments/fedcore/{config,scores,selector,certify,fedosr_split}.py` | numpy certification core |
| `experiments/fedcore/{models,fed_train}.py` | FedAvg + logit export (torch) |
| `experiments/fedcore/run_smoke.py` / `run_cifar.py` | fake-logit smoke / real CIFAR |
| `experiments/fedcore/README.md` | what each experiment proves + results |
| `scripts/docker_cifar.sh` | Docker wrapper for `run_cifar.py` |

---

## 5. Context this session has that a fresh agent won't

- The novelty defense rests on **Theorem 1 being non-reducible**; the decisive
  experiment is **ablation (iii)**: show naive pooled CP violates the target while
  Theorem 1 holds (already shown synthetically; reproduce on real CIFAR).
- The reviewer's strongest attack is "SRCC + federated CP glued together." The
  rebuttal is the Poisson-binomial invalidity of pooling + the partial-
  exchangeability risk UCB. Keep this front-and-center in writing.
- Target venue (journal): IEEE TIFS / TNNLS / Information Fusion; decide after the
  CIFAR numbers and the Lemma L proof are in hand.
