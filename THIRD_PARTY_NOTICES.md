# Third-party notices

Fed-CORE analysis source is licensed under [LICENSE](LICENSE). Dataset assets
and dependencies retain their own terms and are not relicensed by Fed-CORE.

The count-level replay uses NumPy, SciPy, pandas and mpmath. These packages are
installed separately using requirements.txt and are not vendored here.

The supplied manuscript contains illustrative PathMNIST tissue patches from
the MedMNIST Dataset, distributed under CC BY 4.0. Retain the benchmark and
source-data citations in the manuscript. See the provider's
[license and citation notice](https://github.com/MedMNIST/MedMNIST#license-and-citation).
Standalone dataset arrays, predictor checkpoints and record-level logits are
not included in this snapshot.

No FedPD, FedOSS or FOOGD training source or derived training runners are included.
The earlier repository tags v0.2.0 through v0.4.2 were withdrawn because their
runner distribution did not preserve the applicable upstream terms. Those tags
do not identify this analysis release; cite the current snapshot's Git commit.
