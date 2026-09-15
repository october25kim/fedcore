# Changelog

## v0.5.1 - 2026-09-15

First tag in which the Table 5 (T3 bounded-Lambda) result is reproducible from the
released tree. v0.5.0 contained no T3 artifact of any kind; the T3 module was committed
after that tag and its result files never reached git at all (see below).

### Added
- `results/fk_t3_uncertainty_e6/t3_corrected_fixed_traffic.py` - the **corrected** T3
  bounded-Lambda replay, which allocates `delta_r / (J * M)` and `delta_c / (2 * J * M)`.
  This is the implementation behind the 199-of-450 certification at alpha = 0.20 reported
  in the manuscript. It had never been distributed.
- `results/fk_t3_uncertainty_e6/imported_corrected/` - that replay's run outputs.
- `results/fk_t3_uncertainty_e6/reproduce_corrected_t3.py` - a runner that loads the
  replay with four asserted path substitutions, resolves the count tensor and the archived
  traffic realisation wherever they sit in this tree, and checks all six risk targets
  against their published values. Run it from the repository root in the project image; it
  exits non-zero on any mismatch.
- `results/t3_bounded_lambda/` - the superseded run and its pre-registration, which the
  previous commit intended to ship but did not (below), now carrying a `WITHDRAWN.md`
  marker. It is retained rather than deleted because its recorded traffic realisation is
  the pinned input the correction consumes.

### Fixed (release process)
- `.gitignore` excludes `/results/`, and `scripts/sync_dist.sh` copies allow-listed
  additions into the distribution working tree without `git add -f`. Every `results/`
  path on the allowlist was therefore copied and then silently ignored, so none of them
  has ever been committed or tagged. The commit "ship the T3 bounded-Lambda arm and its
  pre-registration" shipped the module and none of the artifacts. All allow-listed
  `results/` paths are force-added in this release.

### Unchanged
- No library code changed. `fedcore/` is byte-identical to the previous commit.
- The superseded run is not deleted and its numbers are not edited.

## v0.5.0 - 2026-09-13

### Removed (licensing correction)
- `fedcore/experiments/run_fedpd_cifar.py`, `run_selftrain_fedpd.py` - contained code
  the files themselves document as lifted verbatim from FedPD `tools/proser_federated.py`.
  FedPD is GPL-3.0 and this repository is MIT, so the MIT grant was not the author's to
  give. Relocated to a separate GPL-3.0 component; not redistributed here.
- `fedcore/experiments/run_fedoss_cifar.py`, `run_foogd_cifar.py`,
  `run_foogd_full_cifar.py`, `fedcore/experiments/foogd_score.py` - contained code
  documented as taken verbatim from FedOSS and FOOGD, neither of which publishes a
  license file. Absent a license, no redistribution right is established, so these are
  removed and not relocated.

### Withdrawn
Tags v0.2.0 through v0.4.2 shipped the files above under an MIT-only LICENSE. This was
inconsistent with the upstream terms and with this repository's own THIRD_PARTY_NOTICES.md,
which already stated that FedPD source must not be copied into an MIT-only distribution.
Those tags are withdrawn and should not be cited.

### Unaffected
No removed file participates in the count-to-decision path. The certification core, the
archived per-client count triples, and every certification number reported in the
manuscript are unchanged.

## 0.3.0

- Added the sealed WR-v3 count-to-decision release used by the current
  manuscript under `paper/wr-v3/`.
- Reproduced the 450-cell H/S/B headline of 177/177/130 from one prespecified
  with-replacement audit realization per client reservoir.
- Added source-ID-disjoint post-certification evaluation records for the 177
  frozen H-selected policies and an independent fail-closed verifier.
- Retained v0.2.0 and `paper/v18/` as a historical numerical release.

## 0.2.0

This release aligns the public certification implementation with the statistical
contract used by the v18 manuscript.

- A fixed full-simplex selector uses member-level risk and coverage tails with no
  additional division by the number of strata.
- A proposal-frozen simple family divides those tails by the number of family
  members only.
- The full-simplex Holm/IUT family procedure reports a fixed-alpha decision,
  raw and Holm-adjusted p-values, and a family-simultaneous coverage lower bound.
  It does not report a numerical risk upper confidence bound.
- Strict bounded-mixture certification uses simultaneous endpoints and a
  validated conservative positive-denominator solver that fails closed.
- Known-mixture pooled Clopper–Pearson certification requires an explicit
  matched-mixture i.i.d. sampling contract.
- The v18 paper package hash-binds the released benchmark count artifacts and
  provides a strict count-to-decision verification command.

Archived clientwise `/J` and familywise `/(MJ)` calculations remain available
only as explicitly labeled legacy comparison procedures.
