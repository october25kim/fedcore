# Changelog

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
