## Cricket DRS replication of the retained-review model (HONZA-76)

Run 17–18 Oct 2026 on Cricsheet ball-by-ball data. The question is whether the MLB retained-challenge findings carry over to a second sport.

### Verdict

**Partly.** The method transfers and two of the four MLB findings replicate. The headline does not: batters in DRS leave almost nothing on the table.

| MLB finding | DRS (batting side) | Replicates? |
|---|---|---|
| Value curve is flat: a one-line rule p ≥ δ/(g+δ) captures ~96 % of the optimum (post-HONZA-88) | 99.8 % (Test), 98.8 % (ODI), 99.2 % (T20) | **yes** |
| The optimal threshold moves strongly with the state | With 1 review left, the LBW threshold falls 19 pp (Test), 15 pp (ODI) and 8 pp (T20) from the top order to the tail. Holding 1 vs 2 reviews moves it 17 / 13 / 7 pp at the start of the innings | **yes** (Tests similar to MLB's 25.7 pp score-averaged, 6.3 pp at a tied score; post-HONZA-89) |
| Option value ∝ √(opportunities left) | No. A linear fit beats √N in every format (R² 0.98–0.99 vs 0.92–0.94). The power exponent is 1.6–1.9, with 95 % CIs excluding 0.5 | **no** |
| Teams lose 0.62 [0.50, 0.75] wins per team-season, mostly by under-challenging (post-HONZA-89) | Teams are within 0.00–0.03 match points per 100 team-matches of the optimum (fitted rule, same draws) | **no** |
| Perception noise is the main referee risk | Same structure, and **better identified**: the umpire's-call share pins it from both sides in Tests and T20 | sharper |

The √t law reads as a regime result, not a universal law. In MLB there are many low-probability opportunities per retained challenge. In DRS a batting innings has on average 5–7 out decisions (median 5 in T20, 6 in ODIs, 7 in Tests; 90th percentile 7–10), and about 30 % of LBW out decisions would be overturned. With so few, high-hit-rate opportunities, the value of the last review is close to (expected reviewable errors ahead) × stake, which is linear or convex in N, not √N.

### Data and scope

- Men's Tests, ODIs and T20Is (Oct 2017 on) plus the IPL (2018 on). T20I and IPL are pooled as "T20". Only matches with at least one review are kept.
- **Allowances**, read off the data (only 1 review per side is ever taken with none left):
  - Tests: 2, then 3 per innings from Jul 2020.
  - ODIs and T20Is: 1, then 2 from Jul 2020.
  - IPL: 1, then 2 from 2022.
- A review that ends in umpire's call is kept.

| | Matches | Batting-side out decisions | Batting reviews | Fielding reviews |
|---|---|---|---|---|
| Test | 331 | 8,683 | 1,822 | 2,326 |
| ODI | 566 | 6,194 | 809 | 1,092 |
| T20 | 1,116 | 10,515 | 914 | 1,477 |

Batters review **~70–78 % of LBW out decisions** but only 1–3 % of caught decisions. Reviews succeed 35–38 % of the time.

### Method (mirrors the MLB pipeline)

1. **`parse.py` → `build.py`**: the ball-level state and reviews left per side, with every review's three-way outcome (overturned / failed but kept on umpire's call / lost).
   - On the batting side the opportunity set is **observed**: every LBW, caught or caught-and-bowled out decision, plus every overturned one.
   - On the fielding side, only reviews are observed.
2. **`wp.py`**: a win-probability engine per format, using gradient boosting with monotone constraints and an isotonic recalibration on held-out matches.
   - Tests use expected match points, win + ½ draw.
   - The stake g of a decision is the match-value swing of one more wicket at that state.
   - Held-out Brier: Test 0.128, ODI 0.162, T20 0.163, against 0.20–0.25 for a constant.
3. **`model.py`**: the selection model. The same Gaussian structure as the MLB perception model:
   - Each out decision has a margin X ~ N(μ_kind, 1) and is overturned iff X > 0.
   - The batter sees S = X + ε with ε ~ N(0, s²) and reviews iff the posterior p = P(X>0 | S) ≥ τ, a revealed threshold per cell (reviews left × wickets 0–2 / 3–5 / 6–9).
   - **Umpire's call is modelled as a margin band, −κ < X < 0.** A ball-tracking-free outcome therefore still carries information on how close the reviewed decisions were. This is what identifies s (see below).
   - The constant-rate alternative (kept with prob. ρ) is reported as a sensitivity.
4. **`dp.py`**: the DP.
   - State: wickets down and reviews left. Transition rates come from the real innings walks.
   - Review iff p(g + A) ≥ ℓ·ΔU, where ℓ is the posterior probability that the review is lost.
   - Policies (optimal, best stake-only rule, the fitted revealed rule, p ≥ 0.5) are simulated on the real decision streams, and the realised value of the real reviews is reported alongside.
   - Uncertainty: a match bootstrap that refits the selection model, re-solves the DP and re-simulates. B = 500 in every format (T20 re-run at B = 500 by the HONZA-89 audit, `audit/drs/audit_drs.md`).
5. **`wolfram/drs_check.wl`**: kernel checks. Output in `results/drs/check.json`.

### What is identified, and what is not

- **Identified from outcome rates:**
  - the review share, success rate and umpire's-call share in every state cell;
  - the LBW error rate, about 30 % of LBW out decisions would be overturned (Test 0.303, ODI 0.306, T20 0.290). Caught decisions: ≤ 0.1 %;
  - the umpire's-call band, κ ≈ 0.41–0.48 margin SD.
- **Perception noise s** (in units of the decision-margin SD), from the profile likelihood:

  | Format | ŝ | 95 % LR set |
  |---|---|---|
  | Test | 0.50 | ≈ [0.45, 0.85] (Δℓ = 2.5 at s = 1, 5.1 at s = 2) |
  | T20 | 0.46 | ≈ [0.42, 0.8] |
  | ODI | 0.81 | weak, [0.5, 2.0] |

  - Without the umpire's-call channel (constant-ρ model) the likelihood is flat above s ≈ 0.5: Δℓ < 1.4 up to s = 4 in Tests. Outcome rates alone bound s only from below. **The three-way outcome is what identifies perception.**
- **Not identified:**
  - **The fielding side.** Not-out appeals that were not reviewed leave no trace in the data, so the opportunity set, the optimum and the wins lost cannot be estimated. Reported descriptively only: fielding reviews succeed 20–24 % of the time, umpire's call 13–15 %, and the success rate barely moves with reviews left.
  - The Gaussian shape of the margin and signal distributions beyond the observed thresholds. The revealed thresholds are near 0 in most cells, so the counterfactual mostly re-weights decisions that teams already review.

### Results (batting side; match points per 100 team-matches; 95 % bootstrap CI)

| | Test | ODI | T20 |
|---|---|---|---|
| Optimal DP | 5.38 [4.74, 5.95] | 2.77 [2.36, 3.19] | 1.22 [1.12, 1.43] |
| Best one-line rule p ≥ δ/(g+δ) | 5.37 (δ = 0.001) | 2.74 (δ = 0.0075) | 1.21 (δ = 0.002) |
| — share of the optimum | **99.8 %** [99.4, 100] | **98.8 %** [96.4, 100] | **99.2 %** [98.9, 100] |
| Teams' fitted rule (simulated) | 5.35 | 2.77 | 1.21 |
| Observed (realised) | 5.14 [4.56, 5.75] | 2.71 [2.30, 3.10] | 1.26 [1.11, 1.42] |
| p ≥ 0.5 rule | 4.05 | 1.74 | 0.95 |
| **Lost vs fitted rule** | **0.027** [0.012, 0.068] | **0.000** [−0.014, 0.412] | **0.008** [−0.002, 0.014] |
| Lost vs realised | 0.23 [−0.17, 0.56] | 0.06 [−0.19, 0.45] | −0.04 [−0.08, 0.12] |
| Reviews per team-match: opt / rule / observed | 4.44 / 1.71 / 2.75 | 1.06 / 0.57 / 0.72 | 0.55 / 0.27 / 0.41 |

- **Validation.** The fitted rule reproduces realised behaviour in Tests: 2.74 vs 2.75 reviews, 0.971 vs 0.970 overturns and 0.575 vs 0.569 umpire's calls per team-match.
  - The gap between "fitted rule" and "realised" value is the stake of overturned decisions: mean g is 0.053 for real overturns vs 0.055 for all reviewable decisions.
  - "Lost vs fitted rule" (same draws) is therefore the clean comparison.
- **Review counts.** The optimum and the cheapest one-line rule review different numbers of decisions but earn the same. With s ≈ 0.5, batters' posteriors are nearly bimodal, so extra reviews of p ≈ 0 decisions are worth about nothing either way. MLB's "under-challenging" pattern has no counterpart on the batting side.

### Sensitivity (match points per 100 team-matches)

| Variant | Test: opt / lost vs rule / rule share | ODI | T20 |
|---|---|---|---|
| Main (band, ŝ) | 5.38 / 0.03 / 99.8 % | 2.77 / 0.00 / 98.8 % | 1.22 / 0.01 / 99.2 % |
| Upper end of the s 95 % set | 5.35 / 0.03 / 99.4 % (s = 0.7) | 3.29 / 0.62 / 95.9 % (s = 2.0) | 1.21 / 0.00 / 99.0 % (s = 0.7) |
| s = 2 (rejected in Test and T20: Δℓ 5.1 and 4.4) | 6.50 / 1.24 / 96.6 % | 3.29 / 0.62 / 95.9 % | 1.63 / 0.43 / 94.4 % |
| Constant-ρ umpire's call | 5.18 / 0.04 / 99.9 % | 2.79 / 0.03 / 99.3 % | 1.26 / 0.00 / 99.6 % |

- Wins lost would only become material, ≈ 0.4–1.2 points per 100 team-matches or 8–19 % of the optimum, if perception noise were large (s ≈ 2). The data reject that in Tests and T20, but not in ODIs.
- The one-line rule stays at 94–100 % of the optimum in every variant.

### Kernel checks (Wolfram 15.0.0 for Mac OS X ARM, 26 May 2026)

Script `wolfram/drs_check.wl`; output `results/drs/check.json`.

- **A.** Posterior X | S ~ N((μs² + S)/(1+s²), s²/(1+s²)) and marginal S ~ N(μ, 1+s²): `True`, `True`.
- **B.** The S* inversion of p ≥ τ: `True`. p is increasing in S: `True`.
- **C.** Orthant correlation 1/√(1+s²): `True`.
  - Python's Owen's-T bivariate normal matches the kernel's `BinormalDistribution` CDF to 1.1e−16.
  - Python's q and J(−κ) at the Test fit match direct 30-digit integration to 5e−17.
- **D.** The DP review rule with a kept umpire's call: the general form is `True`. The constant-ρ cutoff form has no counterexample (`False`). ∂p*/∂ρ = −ΔU(g+A)/(g+A+(1−ρ)ΔU)² < 0.
- **E.** Option value fits (calibrated DP):

  | | R² √N | R² linear | Power exponent [95 % CI] |
  |---|---|---|---|
  | Test | 0.939 | 0.989 | 1.58 [1.46, 1.71] |
  | ODI | 0.917 | 0.982 | 1.82 [1.56, 2.08] |
  | T20 | 0.926 | 0.986 | 1.90 [1.61, 2.19] |

### What this means for the paper

- For AoAS, DRS works as the **second application of the method**, not as a second "wins left on the table" headline. The method's parts carry over:
  - the selection-corrected perception model;
  - the calibrated DP with a three-way outcome;
  - policy evaluation with bootstrap uncertainty.
- Two findings are general:
  - the flat optimum and the one-line rule;
  - state-dependent thresholds.
- Two are context-specific:
  - **√t** holds in the many-opportunity, low-signal regime, which is baseball, not cricket batting;
  - **material losses** need either noisy perception or a binding allowance. Cricket batters perceive well and hold 2–3 reviews for 5–7 decisions per innings on average.
- New and useful for the MLB referee risk: **a richer outcome (umpire's call) identifies perception noise**. MLB has no analogue, so this sharpens the case for the MLB sensitivity analysis and is a contribution in its own right.

### Caveats

- The WP engine is our own and league-average: no team strength, no pitch or conditions. Test values are expected points with a ½ draw.
- The Gaussian margin model is parametric, and its shape beyond the observed thresholds is assumed.
- Decision kind is unrecorded for overturned decisions; it is integrated out.
- The DP pools stakes by wickets down. Overs left (limited overs) and declarations (Tests) enter only through the empirical transition rates.
- The counterfactual keeps the real decision stream fixed: a batter reprieved in simulation does not create new decisions. This is the same approximation as the MLB call stream.
- The fielding side is not analysed beyond descriptives (not identified).
- The best δ is tuned in-sample, which favours the simple rule.

### Audit (HONZA-89)

`audit/drs/audit_drs.md`. The independent recount from the raw JSON matches every count. One low-severity bug was fixed: the simulated fitted rule now uses its own reviews left; lost vs rule moves by ≤ 0.006. T20 was re-run at B = 500. Known-truth recovery (`results/drs/identification.json`) shows s is pinned from below but only weakly from above: 25–35 % of simulated replicates return ŝ > 1 when the truth is ≈ 0.5. WP recalibration and margin-model sensitivity: `results/drs/sensitivity.json`.

### Files

- Code: `src/drs/` (run order in the README).
- Outputs in `results/drs/`:
  - `summary.json`;
  - `model_<fmt>.json` (incl. profile likelihood in s) and `model_<fmt>_const.json`;
  - `dp_<fmt>.json` (values, thresholds, U(w,c), √t fits, bootstrap);
  - `wp_check.json`, `check.json`.
- Figure: `figures/drs_fig1.png`.
