> **Historical.** This file records the derivations as they were developed. The statements
> and conditions that the paper uses are those of the manuscript and its supplement (Section S1), which supersede this
> file where they differ (e.g. the moment and density conditions of the square-root theorem).

# Theory: monotone thresholds, the √t law, and why a one-line rule is enough

Draft theory section for the AoAS paper. Each claim is tagged with the kernel check that backs it. The
checks are in `wolfram/proofs.wl`, and their outputs are in `results/proofs.json` (keys `T1…T4`) with the run log in
`results/proofs-run.json`. Kernel: **Wolfram 15.0.0 for Mac OS X ARM (64-bit) (May 26, 2026)**. Run it with
`wolframscript -file wolfram/proofs.wl` (about 25 s). All checks are deterministic: they use exact rationals, closed
forms and fixed-point iteration, with no sampling.

Status legend:
- **[proved]** — full proof here; the algebraic steps are kernel-checked.
- **[proved, sketch]** — the argument is complete but leans on a standard lemma that is cited rather than reproduced.
- **[numerical]** — supported only by deterministic computation.

## 0. Model and notation

- t = opportunities left; c = challenges held.
- At each opportunity a pair (g, p) ~ F_t arrives, independent of the past and of c.
  - g ≥ 0 is the stake: the WP gained if the call is overturned.
  - p ∈ [0,1] is the calibrated probability that a challenge succeeds.
- Retention: a success gains g and keeps the challenge; a failure loses it.
- With V_0(·) = V_t(0) = 0 and Δ_t(c) = V_t(c) − V_t(c−1):

  V_t(c) = V_{t−1}(c) + φ_t(Δ_{t−1}(c)),  φ_t(d):= E_t[(p g − (1−p) d)^+],

  and the optimal action is to challenge iff p ≥ Δ_{t−1}(c)/(g + Δ_{t−1}(c)). This cutoff was already checked in
  `check.wl` A.

Two consequences used throughout:

- **Δ_t(1) = V_t(1).** The option value of the last challenge is its whole value.
- **The square-increment identity.** Δ_t(1)² − Δ_{t−1}(1)² = κ_t(Δ_{t−1}(1)), where κ_t(d):= 2dφ_t(d) + φ_t(d)².
  Check: `T2b_square_increment` = 0.

## 1. Monotone thresholds

**Lemma 1 [proved].** Fix any law F. Then φ(d) is nonnegative and nonincreasing, with slopes in [−1, 0], and
ψ(d):= d + φ(d) is nondecreasing.

*Proof.* Pointwise in (p, g), the integrand j(x) = (pg − (1−p)x)^+ has slopes in {0, −(1−p)} ⊂ [−1, 0], and the
expectation preserves this. Checks: `T1a_phi_nonincreasing`, `T1a_psi_nondecreasing` and `T1a_phi_slope_in_[-1,0]`
all return `True` (Reduce over the reals with ForAll).

**Theorem 1 [proved].** Let (p, g) be exogenous: F_t may vary with t in any way, but not with c or with past actions.
Then for all t and c ≥ 1:

- (a) Δ_t(c) ≥ Δ_t(c+1) ≥ 0, so V_t is concave in c.
- (b) Δ_t(c) ≥ Δ_{t−1}(c), so Δ is nondecreasing in the opportunities left.

Hence the optimal cutoff Δ/(g+Δ) falls as the game runs down and rises as challenges run out. The cutoff is
increasing in Δ (`check.wl` B).

*Proof (induction on t).* Write a = Δ_{t−1}(c−1) ≥ b = Δ_{t−1}(c) ≥ e = Δ_{t−1}(c+1) ≥ 0. Then

  Δ_t(c) = ψ(b) − φ(a),  Δ_t(c+1) = ψ(e) − φ(b).

- (a): ψ(e) ≤ ψ(b), and φ(b) ≥ φ(a) ≥ 0, so Δ_t(c+1) ≤ Δ_t(c).
- Nonnegativity: Δ_t(c) ≥ φ(b) − φ(a) ≥ 0.
- (b): Δ_t(c) − Δ_{t−1}(c) = φ(b) − φ(a) ≥ 0.
- For c = 1 the φ(a) term is absent.

Checks: the abstract step, quantified over all reals that satisfy Lemma 1, returns `True` for
`T1b_concavity_step`, `T1b_nonneg_step`, `T1b_monotone_in_t_step` and `T1b_concavity_step_c1`. In addition, an exact
rational DP on 300 seeded random discrete laws satisfies both monotonicities in every case (`T1c_exact_random_laws`).
The laws cover i.i.d. and time-varying F, p and g dependent, retention and consumption, with T = 25 and C = 4. ∎

**What is new relative to Derman–Lieberman–Ross (1972) and Kleywegt–Papastavrou (1998, 2001).** In the stochastic
knapsack a unit is spent whenever the item is accepted. Here it is spent only on failure, so the continuation value
enters the gain as p(g + Δ) rather than g. The proof goes through because the gain is still 1-Lipschitz and
nonincreasing in Δ (Lemma 1). Under consumption the same induction applies with φ^cons(d) = E(pg − d)^+, which is
also checked in T1c.

**Where it fails (precise boundaries).**

1. *Laws that depend on c.* With p = 1/10 when c = 1 and p = 9/10 when c = 2 (g = 1), the DP gives Δ(2) = 3.61 >
   Δ(1) = 0.11 (`T1d`). Exogeneity is needed.
2. *The extra-inning top-up.* The calibrated DP (`src/sim.py`) refills c = 0 to 1 at each extra inning, which makes
   V(0) = V(1) at the last regulation opportunity. Concavity then fails there and only there: on the home side,
   Δ₁ = 0 < Δ₂ = 0.0002 at the final row (`T2i_calibrated_home`). Every other row of the calibrated DP satisfies
   both monotonicities, on both sides.

## 2. The √t law

**Theorem 2 [proved].** Under retention with c = 1, let q = 1 − p. Assume:

- Given g, q has a density f(·|g) near 0 with f(0+|g) = h(g).
- There is a local bound f(x|g) ≤ M(g) for x ≤ x₀, with E[g² M(g)] < ∞ and E[g] < ∞.
- K:= E[h(g) g²]/2 ∈ (0, ∞).

Then Δ_t(1)/√(2Kt) → 1, i.e. **Δ_t(1) ~ √(E[h(g) g²]·t)**.

*Proof.*

1. Δ_t(1) → ∞. Δ_t(1) is nondecreasing by Theorem 1. If it were bounded by L, every step would add at least
   φ(L) > 0, a contradiction. Here φ(L) > 0 because f(0+|g) > 0 on a set where g > 0.
2. dφ(d) → K. Substitute q = u g/(g+d):

   dφ(d) = E[ d g²/(g+d) ∫₀¹ (1−u) f(ug/(g+d) | g) du ] → E[g² h(g)]/2,

   by dominated convergence (the bound is g² M(g)). Kernel: `T2a_lim_d_phi_generic_density` = g²h/2.
3. Then κ(Δ) = 2Δφ(Δ) + φ(Δ)² → 2K. By the square-increment identity and Cesàro, Δ_t²/t → 2K. ∎

**The exponent is set by the mass of p near 1, not by E[pg].** If P(q ≤ x | g) ~ c(g) x^α, then
d^α φ(d) → K_α = E[c(g) g^{1+α}]/(1+α) (`T2a_lim_d^a_phi` gives g^{1+α}/(1+α) for c = 1). The fluid limit
Δ' = K_α Δ^{−α} gives

  Δ_t(1) ~ ((1+α) K_α t)^{1/(1+α)}  (`T2c_fluid_solution`).

| law of p near 1 | regime | check (t = 10⁶) |
|---|---|---|
| positive finite density (α = 1) | √t | Δ/√(2K₁t) = 0.9986, K₁ = E[h g²]/2 = 9/8 exactly (`T2d_alpha1`) |
| density vanishing linearly at 1 (α = 2) | t^{1/3} | Δ/(3K₂t)^{1/3} = 0.985, K₂ = E[g³]/3 = 83/144 (`T2d_alpha2`) |
| atom at p = 1 (α = 0) | linear | Δ/t → E[g; p = 1] = 11/120 = 0.0917 (`T2d_atom`) |

The condition in the issue, "E[pg] finite", is therefore not the right one. The √t law needs a positive, finite
density of calibrated confidence at certainty.

**Time-varying F [proved, sketch].** If K_s ∈ [K_lo, K_hi] ⊂ (0, ∞) and the convergence in step 2 is uniform in s,
the same argument with weighted Cesàro gives Δ_t(1)² / Σ_{s≤t} 2K_s → 1: the option value is the square root of the
cumulative squared-stake mass. Check: alternating stakes 1 and 3 give Δ²/Σg_s² = 0.9973 at t = 10⁶ (`T2h`).

**More challenges [proved, sketch; constants kernel-checked].** For c ≥ 2, Δ_t(c) ~ k_c √(2Kt), where

  k_1 = 1,  k_c² + k_c/k_{c−1} = 1.

So **k_2 = (√5 − 1)/2 = 0.618**: the second retained challenge is worth the golden-ratio conjugate of the first
(`T2e_k2_equals_GoldenRatio-1` = True). The first six constants are k_c = 1, 0.618, 0.477, 0.401, 0.351, 0.316,
and V_t(c)/√(2Kt) → Σk = 1, 1.618, 2.095, …

*Sketch.*

1. Δ_t(c) → ∞ by the same contradiction as in step 1 of Theorem 2.
2. y_t = Δ_t(c)²/(2Kt) follows a deterministic stochastic-approximation recursion with step 1/t and drift
   1 − √y/k_{c−1} − y + o(1). That drift is strictly decreasing (`T2e_drift_decreasing`) and has its unique root at
   k_c² (`T2e_drift_root`).
3. The standard Robbins–Monro / Chung lemma then gives y_t → k_c².

Numerical check (uniform p, g = 1): Δ_t(c)/√t at t = 10⁶ is 0.9990, 0.6170, 0.4763, 0.3997 (`T2e_uniform`), against
the limits 1, 0.618, 0.477, 0.401.

**Consumption contrast [proved].** Under consumption, Δ^cons_t(c) ≤ s:= ess sup(pg) for every t. The inductive step
x ≤ s, Y ≤ s ⇒ x + E(Y − x)^+ ≤ s is `T2f_consumption_bound_step` = True; with uniform p and g = 1,
Δ^cons at t = 10⁶ is 0.999998 ≤ 1. A consumed challenge is worth at most one stake; a retained one grows without
bound.

**Calibrated 2026 DP: what the √ fit actually reflects [numerical].** The calibrated Δ₁ runs from 0.002 to 0.02 WP,
while typical stakes g are 0.01–0.03. The game is therefore *pre-asymptotic* (Δ/g ≈ 0.1–2), not in the regime of
Theorem 2. What carries over exactly is the square-increment identity: Δ² grows by κ(Δ) per opportunity. For uniform
p,

  κ(d) = g²(2d+g)²/(4(d+g)²),

which is increasing and rises only from g²/4 to g² as d/g goes from 0 to ∞ (`T2g`). So Δ² is close to linear in n
even off-asymptote. In the calibrated DP, regressing Δ₁² on n gives **R² = 0.996 (home) and 0.999 (away)** (`T2i`).

The ratio Δ₂/Δ₁ is 0.52 at the first pitch, below the asymptotic 0.618. That is consistent with the pre-asymptotic
iteration, whose ratio climbs from 0.46 at t = 10 to 0.58 at t = 100 (`T2e_uniform`).

> **Paper wording:** "Δ² grows by a nearly constant increment per opportunity, so the option value scales like
> √(opportunities left)."
>
> Avoid: "Δ obeys the √t asymptotic law" — the calibrated constant 0.0031 is a fitted, pre-asymptotic quantity.

## 3. Flat optimum: why a one-line rule is enough

**Theorem 3 [proved].** Let π_δ be the rule "challenge iff p ≥ δ/(g+δ)" for a fixed δ > 0, and let c_t be the
challenges held under π_δ. Write D = Δ_{t−1}(c) for the optimal continuation and I_t(c, g) for the interval between
D/(g+D) and δ/(g+δ).

(i) **Exact regret identity:**

  V*_T(c₀) − V^δ_T(c₀) = Σ_t Σ_{c≥1} P^δ(c_t = c) · E_g ∫_{I_t(c,g)} |p(g+D) − D| dF_t(p|g).

(ii) **Quadratic bound.** If the conditional density of p on I_t is at most M, then

  regret ≤ (M/2) Σ_t Σ_c P^δ(c_t = c) · E_g[ g² (D − δ)² / ((g+D)(g+δ)²) ].

(iii) **Crude form.** Since g²/((g+D)(g+δ)²) ≤ 1/(4δ),

  regret ≤ (M/(8δ)) Σ_t E^δ[(Δ_{t−1}(c_t) − δ)²; c_t ≥ 1].

*Proof.*

- (i) is the performance-difference identity for a finite-horizon MDP with exogenous information. The Q*-gap of the
  wrong action at (t, c, p, g) is |p(g+D) − D|, and π_δ errs exactly on I_t.
- (ii) On I_t the integrand is linear in p and vanishes at D/(g+D). With density 1, the integral is exactly
  g²(D−δ)²/(2(g+D)(g+δ)²) on both sides of D (`T3a_regret_closed_form_both_sides` = {True, True}).
- (iii) is `T3a_g^2/((g+D)(g+de)^2) <= 1/(4 de)` = True. ∎

**Why the optimum is flat (envelope).** The unit-density integral underlying bound (ii), r(δ) = g²(D−δ)²/(2(g+D)(g+δ)²)
(the integral of the Q-gap over I against dp, not against dF), has r(D) = 0, r'(D) = 0 and r''(D) = g²/(g+D)³
(`T3a_envelope`). It is not the per-state regret of a general law of p (g = D = 1, δ = 2, density 2p: regret 11/324,
r = 1/36; `wolfram/regret.wl` -> `results/regret-check.json`, ), but M·r bounds that regret. Under a bounded
density, mis-setting the threshold costs nothing to first order, so the loss of the best constant rule is at most
quadratic in the spread of Δ around δ along the path. The minimising δ sits near the
occupation-weighted mean of Δ.

**Numerical illustration [numerical; stylized, not the calibrated model].** Setup (`T3b_setup`):

- T = 60 opportunities and c₀ = 2.
- Stakes g ∈ {3, 8, 20, 50}·10⁻³ WP, rising ×0.6→×1.8 over the game.
- p density 0.9·Beta(1,4) + 0.1·Beta(4,1).

Results:

- **The best constant δ = 0.0245 loses 7.6 %**, while Δ ranges over [0, 0.065] along the path.
- The regret identity matches V* − V^δ to 4·10⁻¹⁷ (`T3b_regret_identity`).
- The quadratic bound (ii) gives 15.3 %, so it is tight within a factor of 2.
- The crude bound (iii) is uninformative here (185 %). **Use the per-state form (ii) in the paper.**
- The loss curve is flat around the optimum: 9.3 % at δ = 0.02 and 9.2 % at δ = 0.03, but 35 % at δ = 0.01 and 23 %
  at δ = 0.05.

The paper's empirical 2 % for the calibrated DP is **numerical-only**: (ii) has not yet been evaluated on the
calibrated pools. Doing so needs `data/` and is a short addition to `src/sim.py`. The identity (i) makes the
decomposition exact by state (inning × c), which is a natural table for the paper.

## 4. Rule comparison: retention, consumption, NBA cap

Rules:

- **Retention** (MLB ABS): a success keeps the challenge.
- **Consumption**: every challenge is spent.
- **NBA cap**: one challenge; a success earns a second, consumable challenge (state A → B), so at most two are used.
  In A, challenge iff p(g + V_B) ≥ V_A.

Let s = ess sup(pg).

**Theorem 4 [proved].**

- (a) V^cons_t(1) ≤ V^NBA_t ≤ min(V^ret_t(1), V^cons_t(2)) and V^cons_t(c) ≤ V^ret_t(c).
- (b) V^cons_t(c) ≤ c·s and V^NBA_t ≤ 2s for all t.
- (c) V^ret_t(1) → ∞ whenever, for some g₀ > 0, P(p > 1−ε, g > g₀) > 0 for every ε (with g₀ = 0 the claim fails — q = g = U(0,1) keeps Δ_t(1) < 1; `wolfram/supplement.wl` S6). Under the conditions of Theorem 2,
  V^ret_t(1) ~ √(E[h g²] t). Hence one retained challenge eventually beats any fixed allotment of consumable ones.

*Proof.*

- (a) By coupling. The player with the richer rule copies the poorer player's policy and stops voluntarily where the
  poorer rule would have removed the challenge. This is feasible because the richer rule holds at least as many
  challenges on every path: after a success, retention-1 holds 1 ≥ NBA-B's 1; after a failure, consumption-2 holds
  1 ≥ NBA's 0.
- (b) At most c (resp. 2) challenges are used, each with conditional expected gain ≤ s.
- (c) Theorem 2, plus the contradiction argument in its step 1. ∎

**Numbers [numerical].**

Uniform p, g = 1 (s = 1); `T4_uniform_p_g1` has the full table, and the ordering in (a) holds at every reported t:

| t | cons 1 | NBA | ret 1 | cons 2 | ret 2 |
|---|---|---|---|---|---|
| 10 | 0.86 | 1.52 | 2.42 | 1.64 | 3.54 |
| 50 | 0.96 | 1.87 | 6.22 | 1.91 | 9.68 |
| 10⁴ | 1.00 | 2.00 | 99.0 | 2.00 | 159.8 |

One retained challenge beats 2, 3 and 5 consumable ones for all t ≥ 4, 10 and 27 respectively (`T4_ret1>consC`).

In the stylized MLB-like game of §3 (T = 60), in WP per team-game (`T4_T3b_law_T60`):

| rule | value |
|---|---|
| cons 1 | 0.032 |
| NBA | 0.048 |
| cons 2 | 0.055 |
| ret 1 | 0.066 |
| ret 2 | 0.098 |

Here retention of 2 is worth 77 % more than consumption of 2, and a single retained challenge already beats two
consumable ones.

## 5. Summary of status

| claim | status |
|---|---|
| Thresholds monotone in t and c (any exogenous, time-varying F) | proved (T1) |
| Monotonicity fails with c-dependent laws; with the extra-inning top-up only at the last regulation opportunity | counterexample (T1d) and calibrated check (T2i) |
| Δ_t(1) ~ √(E[h g²] t), exact constant | proved (T2) |
| Rate t^{1/(1+α)} set by the mass of p near 1 (atom ⇒ linear) | constants exact; rates proved for α = 1, numerical for α ≠ 1 |
| Time-varying: Δ² ~ Σ 2K_s | proved, sketch; numerical check |
| c ≥ 2: k_2 = (√5−1)/2, recursion for k_c | constants exact; convergence proved, sketch (SA lemma); numerical to 10⁶ |
| Consumption: Δ ≤ one stake | proved |
| Calibrated "√n" fit = near-linear Δ² growth, pre-asymptotic | numerical (R² 0.996 / 0.999) |
| Regret identity + quadratic (envelope) bound for constant-δ rules | proved (T3a); bound within 2× in the stylized game |
| 2 % loss of the one-line rule in the calibrated DP | **numerical-only** (bound not yet evaluated on calibrated pools) |
| Rule ordering, bounds, retention eventually dominates | proved (T4); crossovers numerical |
