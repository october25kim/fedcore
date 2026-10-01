# FedCORE PACS H/S/B v3 workstation4 GPU 3 static binding r5

This pre-execution bundle binds the unchanged 30-cell PACS v3 protocol to
workstation4 physical GPU 3, exactly five read-only inputs, one empty
scientific output directory, and the imported sealed runtime lineage. Docker
exposes the selected physical GPU as `cuda:0` inside the container.

The device change is governed by
`../ijar-v34-pacs-fresh-seed-hsb-v3/PREEXECUTION_RESOURCE_AMENDMENT_WORKSTATION4_GPU3_R2.json`.
No PACS outcome was observed before rebinding. This bundle does not by itself
authorize scientific execution.

Run `python3 verify_binding.py` from this directory before use.

