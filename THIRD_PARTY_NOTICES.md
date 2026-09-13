# Third-party notices

Fed-CORE source in this repository is licensed under `LICENSE`. Datasets,
Python packages, pretrained weights, and external baseline repositories are not
relicensed by Fed-CORE.

The core imports NumPy and SciPy. Optional paths use scikit-learn, pandas,
PyYAML, Matplotlib, Pillow, PyTorch, torchvision, MedMNIST, FLamby, and dataset
provider assets. Consult the license metadata of the exact installed versions
and the applicable dataset terms before redistribution.

FedPD, FedOSS, and FOOGD are not vendored, and as of v0.5.0 no runner that
embeds their source is distributed here. Six files that documented themselves as
containing verbatim upstream code were removed: two FedPD-derived PROSER runners
(FedPD is GPL-3.0, relocated to a separate GPL-3.0 component) and four
FedOSS/FOOGD-derived runners (no upstream license is published, so no
redistribution right is established). Tags v0.2.0 through v0.4.2 contained those
files under an MIT-only LICENSE and have been withdrawn; v0.5.0 is the first tag
intended for citation. Runners under `fedcore/experiments/`
expect separately obtained upstream checkouts. Preserve every upstream LICENSE
and NOTICE file; in particular, do not copy FedPD source into an MIT-only
distribution. Absence of `third_party/` means those optional runners are not a
self-contained reproduction.

The v18 reference Figure 7 embeds nine small illustrative patches from
PathMNIST, a MedMNIST Dataset distributed under CC BY 4.0. MedMNIST requests
citation of its benchmark and source-data papers. See the official license and
citation notice at <https://github.com/MedMNIST/MedMNIST#license-and-citation>.
No standalone Dataset arrays, model weights, or external baseline source are
included in this source snapshot.
