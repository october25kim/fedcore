# Code availability

The following paragraph appears in the v7 manuscript.

Count-level data, frozen protocols, analysis code, and CPU replay instructions are publicly available at https://github.com/october25kim/fedcore. The repository documents the contents and reproduction scope of the primary certification, matched Office-Home comparisons, designed-count analyses, and PACS audit. Replay from the saved counts reproduced all 1,512,000 PACS risk flags, 126,000 fixed-procedure selections, and 162,000 outcome rows. These checks verify count-level analysis and do not constitute independent retraining of all thirty models. Raw datasets, model checkpoints, and record-level logits are not included in the repository.

## Snapshot identification

Cite the current Git commit when identifying this analysis snapshot. Earlier
tags, including v0.4.2, do not identify this revision. The evidence map and
additional replay limits are documented in [README.md](README.md).
