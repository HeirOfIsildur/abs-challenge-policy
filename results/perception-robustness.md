## Perception robustness: σ_r and the wins-lost CI (HONZA-73)

> **Superseded numbers (HONZA-86, 18 Oct 2026).** The policy evaluation behind every wins figure below used a posterior
> calibration with a bug (running maximum over sparse signal bins) and a DP optimum that ignored the score. With both
> fixed, the main spec gives **0.77 [0.62, 0.91]** wins lost and the per-player model 0.90; see `results/headline.md`.
> The σ_r estimates, identification results and model comparisons in this report do not depend on either and stand.

Run 17–18 Oct 2026. Pre-registered decision rule (HONZA-73): flag "switch to TAS" if the final 95 % CI of wins
lost includes 0.25 wins/team-season or less, or is wider than 1.5 wins; otherwise confirm AoAS.

### Verdict

**Confirm AoAS.** Teams lose **0.65 [0.51, 0.81] wins per team-season** against the optimum. That is
about 20 wins league-wide. The CI comes from a full game bootstrap (B = 120) that refits the truth and
perception model, re-solves the DP and re-simulates in every replicate. The lower end is well above 0.25 and the
width is 0.30 wins, so neither TAS trigger fires.

**The headline is smaller than HONZA-71's 1.09.** Most of the drop comes from the missed-call (truth) model, not
from σ_r:

- HONZA-71's truth model had one offset per edge, so the top and bottom of the ABS zone shared a single vertical offset.
- The data reject this by 2,092 log-likelihood points. The ABS zone's bottom edge sits 1.0 in below the Statcast
  `sz_bot` line, and its top edge 0.75 in below the `sz_top` line.
- With one shared offset, the model predicted 434 more successful challenges than happened (6,114 vs 5,680). That
  inflated the pool of truly missed calls, and with it the optimum.
- A region-specific truth model plus a 0.14 % rate of rulings that contradict the tracking fixes the overturn
  calibration: predicted 50.0 % vs observed 48.9 % for batters, 58.8 % vs 58.0 % for catchers.

**σ_r is pinned.**

- **Bootstrap:** batter 3.11 [3.01, 3.22] in, catcher 2.03 [1.96, 2.11] in, about ±3.5 %.
- **Model independence:** rebuilding 2026 on the Stats API feed's own coordinates and zone removes the truth model
  entirely, because there σ_m → 0 and the feed reproduces the ABS rulings. On that rebuild σ_r is 3.10 / 1.88 in
  and wins lost is 0.51.
- **Break-even:** wins lost reaches 0.25 only if σ_r is about 17 % higher than estimated. That is five bootstrap
  half-widths away.

### 1. Player heterogeneity

`lik.py` with random effects is a bivariate normal player effect on (κ, log σ_r): a batter effect on called strikes
and a catcher effect on called balls. It uses a 5×5 Gauss–Hermite marginal likelihood and sits on top of the
BIC-best fixed part.

| | Batters (653, 104 with ≥ 15 challenges) | Catchers (111, 86 with ≥ 15 challenges) |
|---|---|---|
| SD of threshold shift κ_i | 1.22 in | 0.38 in |
| SD of log σ_r,i | 0.13 | 0.09 |
| corr(κ_i, log σ_r,i) | 0.51 | 0.39 |
| σ_r,i among regulars, EB posterior p5 / p50 / p95 | 2.60 / 2.96 / 3.39 in | 1.80 / 2.05 / 2.27 in |

- Heterogeneity is real: −650 log-likelihood against the same fixed part, and the best BIC of all models (58,029).
- It is mostly in *willingness to challenge* (κ), not in perception.
- With player-specific σ_r, the player's posterior is calibrated within their own noise level, and wins lost is
  **0.59**, against 0.66 for the same fixed part without player effects.

### 2. Out-of-sample checks

| Fit → evaluation | Batter σ_r | Catcher σ_r | Predicted vs observed |
|---|---|---|---|
| 2026 Mar–Jun (1,281 games) → Jul–Sep (1,148) | 3.34 | 2.09 | Overturn rate: batter 50.8 vs 50.2 %, catcher 59.1 vs 57.7 %. Challenges: batter 2,201 vs 2,272. **Catchers challenged 23 % more than predicted (2,397 vs 2,950): they learned during the season.** |
| 2026 Jul–Sep refit | 3.20 | 1.94 | — |
| Triple-A 2025 refit (2,194 games, 9,341 challenges) | 3.45 [3.35, 3.55] | 2.22 [2.17, 2.27] | Overturn rate 44.9 vs 45.0 %, 54.0 vs 53.8 % |
| Spring 2026 refit (443 games, 1,907 challenges) | 3.89 [3.63, 4.16] | 2.46 [2.34, 2.58] | Overturn rate 45.4 vs 45.7 %, 59.3 vs 59.5 % |
| 2026 rebuilt on feed geometry (σ_m → 0) | 3.10 [3.01, 3.18] | 1.88 [1.84, 1.92] | Wins lost 0.51 |

Realised wins on Jul–Sep, per team per 162 games:

- Mar–Jun parameters: opt − observed **0.49**.
- Jul–Sep parameters: **0.57**.
- Mar–Jun in-sample: **0.69**.

Observed realised wins rose from 3.21 to 3.49 per 162 games, so teams improved over the season.

Transported unchanged, the MLB 2026 parameters do *not* fit Triple-A or spring. These feeds measure location and
zone differently (the ABS system's own numbers), so the truth model does not transport. The perception noise does:
in both leagues, where the truth is known almost exactly, σ_r is 3.4–3.9 in for batters and 2.2–2.5 in for
catchers. Triple-A players and spring rosters are somewhat noisier than MLB regulars, which is the expected direction.

### 3. Alternative perception laws

ΔlogL is relative to the baseline (region truth + contamination, Gaussian, constant σ_r per side). Calibration
gives the overturn rate given a challenge, predicted vs observed. Wins lost = opt − observed, R = 10.

| Model | k | −ΔlogL | BIC | σ_r batter / catcher | Calibration (batter / catcher, %) | Wins lost |
|---|---|---|---|---|---|---|
| HONZA-71 truth (edge offsets), exact likelihood | 18 | +2,092 | 63,734 | 3.25 / 2.17 | 54.7 vs 48.9 / 61.9 vs 58.0 | 1.06 |
| Region truth, no contamination | 22 | +351 | 60,301 | 3.26 / 2.01 | 51.2 / 59.7 | 0.67 |
| **Baseline** | 23 | 0 | 59,612 | 3.27 / 2.02 | 50.0 / 58.8 | 0.60 |
| Logistic ε | 23 | +175 | 59,962 | 2.98 / 1.86 | 50.1 / 58.9 | 0.86 |
| t₁₀ ε | 23 | +123 | 59,857 | 3.05 / 1.89 | 50.1 / 58.9 | 0.75 |
| t₅ ε | 23 | +380 | 60,371 | 2.97 / 1.84 | 50.2 / 58.9 | 0.88 |
| t₃ ε | 23 | +869 | 61,349 | 3.22 / 1.98 | 50.2 / 59.0 | 0.63 |
| σ_r by edge | 25 | −76 | 59,484 | 3.42 / 2.13 | 50.0 / 58.9 | 0.62 |
| σ_r by region | 29 | −189 | 59,308 | 3.65 / 2.08 | 50.0 / 58.9 | 0.62 |
| σ_r slope in \|w\| | 25 | −11 | 59,614 | 3.53 / 2.01 | 50.0 / 58.8 | 0.51 |
| κ by region | 29 | −210 | 59,265 | 3.19 / 2.02 | 50.0 / 58.8 | 0.58 |
| **κ and σ_r by region (BIC-best fixed)** | 35 | −254 | 59,253 | 3.11 / 2.03 | 50.0 / 58.8 | **0.66** |
| κ, σ_r by region + slope | 37 | −265 | 59,256 | 3.37 / 2.04 | 50.0 / 58.8 | 0.55 |
| + player random effects | 41 | −903 | **58,029** | 3.04 / 2.04 | 50.0 / 58.8 | 0.59 |

(σ_r in the region models is the bottom-edge value; the other regions are multiples of it, listed in
`perception_variants.json`.)

- **Rejected:** heavier tails. Logistic and t₁₀/t₅/t₃ are all worse than the Gaussian by 123–869 log-likelihood
  points with the same parameter count. So is the HONZA-71 edge-only truth model.
- **Supported:** σ_r and the threshold varying by zone region. Batters are least willing to challenge high pitches
  (κ +1.6 in at the top edge) and catchers see top-edge pitches 18 % noisier. A σ_r that grows or shrinks with
  distance adds almost nothing (−11 for 2 parameters).
- **Not separately testable:** miscalibrated confidence (flat prior, overconfidence). A threshold on the signal is
  observationally equivalent under any monotone belief map, so challenge data cannot reject overconfidence. It also
  does not change the estimand: the optimum is the best policy given the same signal, whatever the players believe.

What the data *do* test is whether thresholds are stake-rational. A rule "challenge iff belief ≥ Δ/(g+Δ)" implies a
slope of 1 of logit(1−p*) on log g. The implied slope is:

- **Calibrated beliefs:** 0.07 for batters and catchers alike.
- **Flat-prior beliefs:** 0.19 for batters, 0.11 for catchers.

Under either belief map, players' thresholds barely respond to leverage. That under-response is the behavioural
source of the lost wins. If advice could only be delivered through players' own overconfident (flat-prior) beliefs,
the reachable gain would be 0.14 wins. This is a lower bound on implementability, not the headline.

All of these models leave the conclusion the same: every model the data do not reject gives 0.55–0.66 wins.

### 4. Identification

Kernel: Wolfram 15.0.0 for Mac OS X ARM (May 26, 2026). Script `wolfram/identification.wl`, output
`results/identification-check.json`. Within one side and region:

- P(challenge | w) = Φ((w−κ)/√(σ_m²+σ_r²)).
  - **I1:** σ_r ∂P/∂σ_m − σ_m ∂P/∂σ_r ≡ 0 (`FullSimplify` → 0). Non-challenged pitches identify only
    σ_t = √(σ_m²+σ_r²) and κ.
- P(challenge, call wrong | w) is the bivariate-normal orthant Φ₂(w/σ_m, (w−κ)/σ_t; ρ = σ_m/σ_t).
  - **I2:** it matches the integral to 8·10⁻²⁸.
- **I3:** at the fitted values, the Jacobian of [P(ch | w_j), P(ch, wrong | w_j)] over 8 values of w with respect to
  (σ_m, σ_r, κ) has **rank 3**.
  - Batter singular values 0.378 / 0.134 / 0.053; catcher 0.634 / 0.205 / 0.187.
  - The challenge-only block has rank 2. Its null vector is exactly the tangent of the circle σ_t = const:
    (−0.9979, 0.0646, 0) vs (−0.997909, 0.0646369, 0).
- **I4:** in words, the overturn rate of challenged pitches splits σ_t.
  - As σ_r → ∞, q(w) → Φ(w/σ_m), i.e. no selection.
  - As σ_r → 0 with κ > 0, q → 1.
  - At fixed σ_t, q(0) moves from 0.52 to 1.00 as σ_m goes from 0.1 to 3 in.

Known-truth recovery (`src/identify.py`, `results/identification.json`) simulates challenges and rulings from the
fitted baseline on the real 2026 opportunities. It refits from a deliberately wrong start (σ_r × 1.5, σ_m × 2).
- **Correct model, 8 replicates, all converged:** σ_r bias +0.4 % for batters and −0.04 % for catchers, with Monte
  Carlo SDs of 0.046 and 0.016 in. σ_m bias is within ±5 % in every region.
- **Misspecified: true ε ~ t₅, fitted Gaussian.** σ_r is biased up 22–24 %. Heavy tails do matter for σ_r, but the
  data reject them (Sec. 3).
- **Misspecified: true player heterogeneity at the fitted SDs, fitted homogeneous.** Batter σ_r is biased up 10 %
  and catcher σ_r 2.5 %. Ignoring heterogeneity overstates σ_r, which *lowers* wins lost, so the homogeneous
  headline is conservative. This matches the random-effects fit, where batter σ_r is 3.04 against 3.27.
- **Profile likelihood on the real data (baseline).** Moving batter σ_r by −5 % or +5 % costs 6.6 or 5.6
  log-likelihood points, and catcher σ_r 11.7 or 10.1. The 95 % LR interval is therefore within about ±3 %. The +17 %
  break-even of Sec. 5 would cost more than 40 points (batter +10 %: 20.7, +20 %: 71.5).

The likelihood also gives a direct, model-free check: in Triple-A, spring, and the 2026 feed rebuild, σ_m → 0, so
the split is trivial there. σ_r then comes from the challenge curve alone and agrees in magnitude with the joint fit.

### 5. Final number

| | Point | 95 % CI | Source |
|---|---|---|---|
| **Wins lost, opt − observed** | **0.65** (point fit 0.66) | **[0.51, 0.81]** | BIC-best fixed model, full bootstrap with refits, B = 120, R = 2 draws per replicate |
| Against the simulated fitted rule | 0.82 | [0.71, 0.91] | same |
| With player random effects | 0.59 | — | R = 10 (refitting random effects per replicate does not fit in memory; point is inside the CI above) |
| Feed geometry (no truth model) | 0.51 | — | R = 10 |
| Model spread (models not rejected) | 0.55–0.66 | — | Table in Sec. 3 |

**The rest of the HONZA-71 story survives on the BIC-best model:**

- Optimal 4.00 wins, best stake-only WP rule 3.90 (97.5 % of the optimum), observed 3.34, Savant 0.2 rule 2.84.
  Savant's rule is still worse than what teams did.
- Teams still under-challenge: 2.17 vs 2.83 challenges per team-game.
- By side: batters lose 0.38 (1.82 vs 1.44) and catchers/pitchers 0.28 (2.17 vs 1.90).
- Teams that challenge more get more out of it: the fitted-rule simulation gives 3.18 against 3.34 realised.
  - The model now under- rather than over-reproduces realised wins.
  - The cause is that teams pick higher-leverage pitches than κ linear in log g captures.
  - opt − simulated rule (0.82) is therefore the like-for-like loss; opt − observed (0.66) is the conservative one.

### Caveats

- The truth model is fitted to Statcast geometry. The feed geometry used by the ABS system itself gives σ_m ≈ 0. A
  paper version should consider switching the whole pipeline to feed coordinates; that gives 0.51 instead of 0.66.
- The random effects in the simulation use the full-sample EB posterior means and are not refitted per bootstrap
  replicate.
- Bootstrap CIs use R = 2 Monte Carlo draws per replicate. Monte Carlo SD of opt − observed per replicate is about
  0.02, which is negligible against the CI.
- The DP's slot pooling makes the optimum a lower bound, so wins lost is conservative.
- HONZA-71's `joint.py` likelihood used a 241-point grid. It clips challenged pitches far from the edge to
  log 1e-300, so it is inaccurate in the tails. `lik.py` replaces it with an exact split quadrature, validated
  against the bivariate-normal CDF.

### Files

- **Code:** `src/lik.py` (JAX likelihood), `robust_fit.py`, `identify.py`, `oos.py`, `fetch_ext.py`, `build_ext.py`,
  `feed_check.py`, `variant_wins.py`, `sr_breakeven.py`, `final_ci.py`; `sim.py` generalised to lik models.
- **Outputs:** `perception_variants.json`, `perception_calibration.csv`, `identification*.json`, `oos.json`,
  `feed_check.json`, `variant_wins.json`, `sr_breakeven.json`, `final_ci_kappa_sr_region.json`.
