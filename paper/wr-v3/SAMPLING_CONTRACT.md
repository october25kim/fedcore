# WR-v3 primary sampling contract

The primary target is the clientwise frozen certification-fold empirical
reservoir. It is not an unobserved future deployment population.

For each declared client `j`, let `N_reservoir,j` be the number of unique source
records in the frozen certification reservoir. The primary analysis uses exactly
one audit realization per experimental cell. It draws
`n_draw,j = N_reservoir,j` indices independently and uniformly with replacement
from that client's reservoir.

The random-number generator is NumPy PCG64 with
`SeedSequence([frozen_primary_audit_seed, client_id, 0])`. The same realized
client index vector is used for all 12 proposal-frozen selectors, all six risk
targets, and procedures H, S, and B. Repeated indices retain their multiplicity
as independent draws from the empirical reservoir. They are not interpreted as
additional source-level labels.

The proposal, certification, and evaluation roles remain disjoint. No evaluation
or test observation enters the primary draw, selector definition, candidate
ranking, or certification decision. Repeated-draw analyses elsewhere in the
manuscript study operating characteristics and are not primary cell certificates.

The machine-readable governing contract is `governing/PREREGISTRATION.json`.
