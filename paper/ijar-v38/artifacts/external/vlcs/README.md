# VLCS Adverse External Fixed-Frame Evaluation

Verdict: `KILL`

The protocol was locally frozen before training and prediction. It was not registered with an external timestamping service. The evaluation retained 30 models, 54,000 primary rows, 162,000 total rows, and 1,512,000 candidate-method flags.

ResNet18 produced zero registered utility contrast under both protection settings. ConvNeXt-Tiny produced a 1.591-percentage-point contrast at the half-simplex setting, but only a 0.164-point full-simplex contrast with a descriptive interval from 0.027 to 0.356 points. The latter fell below the predeclared 1.00-point minimum.

The source frame contains 10,720 images across four VLCS domains. Exact decoded-image overlap with the checked PACS, Office-Home, and ISIC frames was zero. Near-duplicate and ImageNet-pretraining overlap were not excluded. Therefore, this directory supports an adverse fixed-frame evaluation, not source-population transport or an independent analyst replication.

Raw images, checkpoints, and logits are excluded because redistribution permission was not established. The protocol, source and fold manifests, complete aggregate outcomes, denominator evidence, and decision are included.
