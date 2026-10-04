# Sampling contracts

1. Fixed-size population or empirical-reservoir certification freezes the
   model, selector family, audit sizes, mixture set, draw law, and tie rules
   before certification.
2. Proposal data freeze a single selector or finite selector family.
   Certification data support the deployment decision, and diagnostic test data
   do not select or certify a rule.
3. With-replacement reservoir draws target the frozen empirical reservoir and
   do not create new source labels or an unseen-population guarantee.
4. Full-simplex, strict-polytope, and family procedures retain their stated
   confidence allocations and positive-acceptance gates.
5. Empty positive-denominator domains, invalid endpoints, and failed numerical
   solver checks return `cannot certify`.
6. The registered PACS rerun retains every planned model and adverse outcome;
   fresh seeds on the same hashed frame do not create source-population independence.
