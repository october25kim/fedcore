# FedCORE PACS H/S/B v3 workstation4 static binding r4

This pre-execution bundle binds the unchanged 30-cell PACS v3 protocol to
workstation4 GPU 0, exactly five read-only inputs, one empty scientific output
directory, and the imported sealed runtime lineage. It contains no PACS
outcome and does not by itself authorize scientific execution.

The host and operational watchdog changes are governed by
`../ijar-v34-pacs-fresh-seed-hsb-v3/PREEXECUTION_RESOURCE_AMENDMENT_WORKSTATION4_R1.json`.
The scientific frame, splits, seeds, architectures, H/S/B procedures,
denominators, and outcome gates are unchanged.

Run `python3 verify_binding.py` from this directory before using the bundle.

