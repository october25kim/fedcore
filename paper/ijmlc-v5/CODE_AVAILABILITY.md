# Code availability

The analysis materials associated with this manuscript are available in the
Fed-CORE repository, under `paper/ijmlc-v5`:
https://github.com/october25kim/fedcore/tree/main/paper/ijmlc-v5.
The directory contains aggregate certification counts, frozen protocols,
matched-comparison and mechanism-analysis code, the complete PACS audit tensors
and outcomes, and CPU replay instructions. The Git commit identifying this
snapshot should be cited when referring to its results. Earlier tags, including
v0.4.2, do not identify this revision.

The upload verification replayed all 8,100 primary H/S/B decisions and all
1,512,000 PACS candidate-method risk flags, 126,000 fixed-procedure selections,
and 162,000 PACS outcome rows from the saved aggregate counts. The original
PACS HOLD decision and complete contrast family are retained. These checks
verify count-level analysis, not independent retraining or regeneration of
record-level logits. Dataset images, predictor checkpoints and record-level
logits are not redistributed in this snapshot.

This text replaces the earlier manuscript paragraph that deferred publication
of the complete analysis archive until article acceptance. The manuscript
copy in this directory is the unmodified supplied file, for provenance.
