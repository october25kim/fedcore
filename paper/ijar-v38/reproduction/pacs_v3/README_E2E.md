# PACS Full-Campaign Training Reproduction

This package binds one complete 30-cell PACS training-and-certification campaign.

## Registered identity

- Source commit: `fa016c199a7c846bbe20b64d54ab34164adf7647`
- Source tree: `8adbb6fea6b5f7eef4ea8e7f78b484103bf08024`
- Scientific image ID: `sha256:62c35527e56f5a05c026598ffda355e2b2b80256d966e5d9202ba639a5bfde8a`
- Planned and evaluable models: 30/30
- Terminal failures: 0
- Primary decision rows: 90
- Registered result: `VALID_NEGATIVE`
- Reference hardware: one NVIDIA RTX 4070 Ti SUPER
- Measured campaign wall time: 25,161.207 seconds

## Release assets

The GitHub release carries five assets whose hashes are listed in `RELEASE_ASSETS.sha256`:

1. exact source archive;
2. immutable scientific container image, split into two parts below the per-asset size limit;
3. 15 ResNet18 checkpoints and their seals;
4. 15 ConvNeXt-Tiny checkpoints and their seals.

Reconstruct the image archive and load it as follows:

```bash
cat fedcore-pacs-v3-image-fa016c19.tar.gz.part-00 \
    fedcore-pacs-v3-image-fa016c19.tar.gz.part-01 \
    > fedcore-pacs-v3-image-fa016c19.tar.gz
gzip -t fedcore-pacs-v3-image-fa016c19.tar.gz
docker load -i fedcore-pacs-v3-image-fa016c19.tar.gz
```

The checkpoint inventory and complete count-to-decision evidence are under `artifacts/pacs/`.

## External inputs

Place exactly five files in an input directory and verify them with `INPUTS.sha256`. `PACS_mirror.zip`, `IMAGE_MANIFEST.csv`, and `FOLD_MANIFEST.csv` are identity-bound to the original run. The two pretrained files can be obtained from the official TorchVision URLs:

- https://download.pytorch.org/models/resnet18-f37072fd.pth
- https://download.pytorch.org/models/convnext_tiny-983f1562.pth

The PACS archive is not redistributed because a single dataset-wide redistribution license was not established. Obtain PACS through its upstream benchmark distributor and verify exact byte identity before claiming an exact replay. A scientifically comparable rerun with a different byte-level archive is not an exact-identity replay.

## Verification levels

`EXACT_IDENTITY_REPLAY` requires the exact source, image, five input hashes, 30 checkpoint hashes, 90 primary rows, and completion seal.

`PORTABLE_SCIENTIFIC_REPRODUCTION` requires a new 30-cell run on compatible hardware. The present study did not execute an independent clean-machine rerun and does not claim one.

The assets enable the rerun and checkpoint audit. They do not replace the adverse registered outcome: H, S, and B each certified three of 30 models.
