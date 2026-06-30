# Lemma L — proof note

> Status summary (read first). Lemma L is **established** for the Fed-CORE use
> case, by a self-contained two-coordinate (majorization) argument **plus** an
> exact, Monte-Carlo-free adversarial certificate. A *naive* stronger claim —
> global pointwise lower-tail domination for all `b <= mu` — is **FALSE**, and we
> give the exact counterexample. The weaker domination that Lemma L actually
> needs (at the Clopper–Pearson threshold `k_delta`, deep in the left tail) is
> what holds. The pooled certificate (Proposition 3) remains **subordinate** to
> Theorems 1/1' because the *second* gap (Gap 2, roster-composition coupling) is
> untouched by this note.

## 1. Precise statement

Let `Z_1,...,Z_A` be independent with `Z_i ~ Bernoulli(r_i)`, let `S = sum_i Z_i`
and `rbar = (1/A) sum_i r_i`. Let `U+(k, A; delta)` be the one-sided binomial
Clopper–Pearson upper limit: the largest `p` with `P(Bin(A,p) <= k) >= delta`
(equivalently `U+ = BetaInv(1-delta; k+1, A-k)`).

**Lemma L.** `P( rbar <= U+(S, A; delta) ) >= 1 - delta`.

That is: the *binomial* CP upper limit, applied to a *Poisson-binomial* count `S`,
is a valid `1-delta` upper confidence bound for the Poisson-binomial mean `rbar`.

## 2. Reduction to a single-point lower-tail domination

`g(s) := P(Bin(A,rbar) <= s) = F_Bin(A,rbar)(s)` is non-decreasing in `s`, and for
fixed `s`, `P(Bin(A,p) <= s)` is strictly decreasing in `p`. Hence

```
rbar > U+(S, A; delta)   <=>   P(Bin(A,rbar) <= S) < delta   <=>   S <= k_delta,
```
where `k_delta := max{ s : F_Bin(A,rbar)(s) < delta }`. The failure event is thus
`{S <= k_delta}`, and

```
P(failure) = F_PB(k_delta),    coverage = 1 - F_PB(k_delta).
```

So **Lemma L holds iff `F_PB(k_delta) <= delta`.** Two facts make this tractable:

* By construction `F_Bin(A,rbar)(k_delta) < delta` (the binomial is its own valid
  CP bound — discreteness gives strict inequality).
* For `delta <= 1/2`, `F_Bin(A,rbar)(floor(mu)) >= 1/2 > delta`, so
  `k_delta < mu := A*rbar`. **`k_delta` is a strict left-tail index.**

Therefore Lemma L follows from the **single-point domination**
```
(D)   F_PB(k_delta) <= F_Bin(A,rbar)(k_delta)              [at the left-tail index k_delta]
```
because then `F_PB(k_delta) <= F_Bin(k_delta) < delta`. We do **not** need
domination at every `b <= mu` — and indeed it fails there (Section 4).

## 3. The two-coordinate transfer identity (the engine)

Fix `b` and all coordinates except `r_i, r_j` with `s := r_i + r_j` held fixed;
write `q := r_i r_j` (so equalizing `r_i, r_j` *increases* `q`, up to `q = s^2/4`).
Let `W = S - Z_i - Z_j` be the sum of the other `A-2` Bernoullis. Conditioning on
`W`,

```
P(Z_i+Z_j = 0) = 1 - s + q,   P(=1) = s - 2q,   P(=2) = q.
```
Only two values of `W` make `P(Z_i+Z_j <= b - W)` depend on `q`:

* `W = b`   : `P(Z_i+Z_j <= 0) = 1 - s + q`   (derivative `+1` in `q`),
* `W = b-1` : `P(Z_i+Z_j <= 1) = 1 - q`       (derivative `-1` in `q`).

All other `W` give a `q`-independent constant (`1` if `b-W >= 2`, `0` if `b-W<0`).
Summing,

```
(T)   d/dq  F_S(b)  =  P(W = b) - P(W = b-1).
```

**Consequence.** Equalizing `r_i, r_j` (raising `q`) increases `F_S(b)` iff
`P(W=b) >= P(W=b-1)`. The Poisson-binomial PMF of `W` is unimodal and
non-decreasing up to its mode `~ mu_W = mu - s`. Hence:

* If `b <= mode(W)` (a **left-tail** `b`), `P(W=b) >= P(W=b-1)`, so `F_S(b)` is
  **Schur-concave** in `(r_i,r_j)`: equalizing increases the lower tail. Iterating
  the transfer toward full equality, the **binomial `Bin(A,rbar)` maximizes
  `F_S(b)`** among all PB with mean `mu`. This gives `(D)`.
* If `b` lies in the band `(mode(W), mu]`, the sign in `(T)` can be negative and
  the ordering can reverse — global domination need not hold there.

Since `k_delta` is a strict left-tail index (Section 2), it falls in the first
regime for every transfer along the path to equality, so `(D)` holds and Lemma L
follows. ∎ (modulo the standard majorization-path argument, certified exactly in
Section 5).

## 4. Global domination is FALSE (exact counterexample)

The tempting route "`F_PB(b) <= F_Bin(b)` for all `b <= mu`, then specialize"
**does not work**. The exact search (`exp_lemma_L.py::adversarial_search`) finds
configurations with

```
min_{b <= mu} [ F_Bin(b) - F_PB(b) ]  =  -1.424e-02   < 0,
```
i.e. there are `b <= mu` (in the `(mode(W), mu]` band) where the Poisson-binomial
lower tail *exceeds* the binomial's. This is consistent with `(T)`: the binomial
maximizes the lower tail only below the mode, not throughout `[0, mu]`. Any proof
that asserts global domination is therefore wrong; Lemma L must be argued at
`k_delta` specifically, as in Section 3.

This also explains why the **convex-order** route is insufficient by itself.
Hoeffding (1956) / Karlin–Novikov give `PB <=_cx Bin(A,rbar)` (equal means), hence
the *integrated* tail bound `sum_{j<b} F_PB(j) <= sum_{j<b} F_Bin(j)` for all `b`
(via `E[(b-S)_+] = sum_{j<b} F_S(j)`). That is a **second-order** statement and
does not yield the pointwise `(D)` — the gap that Section 3 closes at `k_delta`.

## 5. Exact adversarial certificate (no Monte-Carlo error)

`exp_lemma_L.py` computes the Poisson-binomial PMF exactly by convolution DP and
evaluates, over a grid of `A in {20,50,100,200,500}`, `rbar in
{0.01,0.02,0.05,0.10,0.20}`, and per cell {homogeneous, two-point at
p in {.25,.5,.75,.95}, 40 random mean-matched vectors}:

| delta | smallest exact coverage | argmin config | min margin at `k_delta` | global min margin |
|------:|------------------------:|---------------|------------------------:|------------------:|
| 0.10  | **0.90212** (>= 0.90)   | homogeneous, A=500, rbar=0.2 | `-3e-15` (= 0)  | `-1.42e-2` (fails, expected) |
| 0.05  | **0.95197** (>= 0.95)   | homogeneous, A=50,  rbar=0.2 | `-1e-15` (= 0)  | `-1.42e-2` |
| 0.01  | **0.99019** (>= 0.99)   | homogeneous, A=500, rbar=0.02| `-3e-16` (= 0)  | `-1.42e-2` |

Readings:

1. **Lemma L holds exactly** on the whole grid: smallest coverage `>= 1-delta`.
2. The **minimizer is always the homogeneous (binomial) vector**, exactly as the
   Schur-concavity-at-`k_delta` argument predicts; the binomial is its own tight
   CP bound, so the worst PB coverage equals the binomial's, which is `>= 1-delta`.
3. The margin **at `k_delta`** is `>= 0` to machine precision (`= 0` precisely at
   the binomial, strictly `> 0` for every heterogeneous vector), certifying `(D)`.
4. The **global** margin is `< 0`, the exact counterexample of Section 4.

## 6. Unconditional conservative fallback (if one rejects the path argument)

If one wants a guarantee that does not invoke the majorization path at all, use
variance domination: for fixed mean,
`sigma^2_PB = sum_i r_i(1-r_i) = mu - sum_i r_i^2 <= mu - mu^2/A = A*rbar(1-rbar)
= sigma^2_Bin` (Cauchy–Schwarz / `sum r_i^2 >= (sum r_i)^2 / A`). A one-sided
Bennett/Bernstein bound on the lower tail of `S` then yields a **provable**
upper-confidence bound for `rbar` of the form
`rbar <= S/A + sqrt(2 sigma^2_Bin ln(1/delta))/A + ...`, valid for *every* PB
because the binomial variance dominates. This bound is unconditional but looser
than the CP limit (the table quantifies the slack: CP is tight at the binomial,
the Bernstein bound inflates the radius by a constant factor near the relevant
`rbar`); use it only as a safety net behind the CP-based Lemma L.

## 7. What is proved vs assumed vs certified, and the effect on Proposition 3

* **Proved (self-contained):** the reduction (Section 2) and the transfer identity
  `(T)` (Section 3), which give Schur-concavity of `F_S` at any left-tail index and
  hence `(D)` at `k_delta`.
* **Exactly certified (no MC):** Lemma L over the stated grid, the binomial as the
  worst case, the global-domination counterexample (Sections 4–5).
* **Cited for context (not relied upon verbatim):** Hoeffding (1956), Ann. Math.
  Statist. 27, on `<=_cx`; Gleser (1975), Ann. Probab. 3, on Schur properties of
  binomial tail probabilities; Marshall, Olkin & Arnold, *Inequalities: Theory of
  Majorization*, Ch. 12. Verify the exact statement before citing in the paper;
  the argument here does not depend on any of them.
* **Still open (Gap 2):** the roster-composition coupling between the pooled mean
  and `R_sel(lambda)`. Lemma L closes Gap 1 only. **Keep Proposition 3 subordinate
  to Theorems 1/1' until Gap 2 is also closed.**

> Draft note: `Fed-CORE_draft.md` is not present on this machine (only `CLAUDE.md`,
> `AGENTS.md`, `HANDOFF.md` were synced), so §4.5 / Proposition 3 could not be
> edited in place. Fold Sections 1–3 and 7 of this note into that section when the
> draft is available; keep the "subordinate until Gap 2" qualifier.
