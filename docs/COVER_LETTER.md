# Cover Letter — Information Sciences

Dear Editor-in-Chief,

We are pleased to submit our manuscript, "Fed-CORE: Federated Certified
Open-Set Recognition via Selective Risk Control," for consideration for
publication in *Information Sciences*.

Federated learning systems are increasingly deployed in settings where test
inputs may belong to classes never seen during training. Existing federated
open-set recognition methods evaluate unknown rejection only through ranking
metrics such as AUROC, and federated conformal prediction certifies closed-set
prediction-set coverage; neither answers the question a deployer actually
faces: **among the predictions the system accepts and acts on, what is the
error rate, and can it be guaranteed to stay below a chosen tolerance?**

This manuscript introduces Fed-CORE, a post-hoc certification layer that
answers this question for any federated open-set model. Its contributions are:

1. **A new certification object.** We formalize the federated accepted
   selective risk under an unknown deployment mixture of heterogeneous
   clients, and show it is a different functional from prediction-set
   coverage, ranking metrics, and batch false discovery rate.

2. **A finite-sample, distribution-free certificate.** A stratified
   conditional-binomial construction certifies this risk from
   secure-aggregatable counts, with a mixture-robust deployment variant and a
   simultaneous certified-coverage lower bound. We prove that the natural
   alternative — pooling calibration data across clients — is
   anti-conservative under mixture shift, so the problem is not reducible to
   centralized selective risk control or to federated conformal prediction.

3. **A feasibility law.** A per-stratum sample-size threshold characterizes
   when certified open-set deployment is possible at all, and an
   information-theoretic converse shows the threshold binds every valid
   procedure, not only ours.

4. **A certification-first empirical study.** Across synthetic and
   CIFAR-10/100 real-logit experiments — including three reproduced federated
   open-set detectors (FedPD-PROSER, FedOSS, FOOGD) — certified deployments
   showed no held-out empirical violation under the stated audit assumptions;
   certified coverage tracked detector quality while validity was independent
   of it, and a full-simplex per-client certificate confirmed the positives do
   not depend on the grouping relaxation.

The paper reframes federated open-set recognition from a ranking problem into
a certification problem: a small trusted audit set is used not to repair the
model but to certify which of its predictions are safe to accept. We believe
this fits *Information Sciences*' scope at the intersection of machine
learning, uncertainty quantification, and trustworthy distributed systems.

This manuscript is original, has not been published previously, and is not
under consideration elsewhere. All authors have approved the submission and
declare no competing interests. Code and the derived calibration counts needed
to reproduce every certificate will be released upon publication.

Thank you for your consideration.

Sincerely,

Seoung Bum Kim (corresponding author)
Department of Industrial and Management Engineering, Korea University
145 Anam-ro, Seongbuk-gu, Seoul 02841, Republic of Korea
sbkim1@korea.ac.kr

On behalf of: Sanghoon Kim (dawonksh@korea.ac.kr)
