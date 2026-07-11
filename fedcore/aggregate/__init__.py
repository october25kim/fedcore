"""Fed-CORE aggregators (seed-aware). NOTE: per-aggregator SD conventions DIFFER
(main/t8/covtype use population SD ddof=0; selftrain uses sample SD ddof=1 with a
convergence guard) -- they are co-located, NOT merged, to preserve byte-identical output."""
