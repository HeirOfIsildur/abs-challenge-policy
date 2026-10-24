## ABS retained challenges — decisive go/no-go test on 2026 MLB data 

Run 17–18 Oct 2026. Bars were fixed on before any data were looked at.

### Verdict

**GO by the pre-registered rule, because Bar 2 holds. Bar 1 holds only against Savant's rule.**

| Bar | Result (95 % CI, bootstrap by game) | Holds? |
|---|---|---|
| 1. Optimal beats Savant's 0.2 rule by ≥ 0.5 wins/team-season | **+1.24** [1.17, 1.30] | yes |
| 1. Optimal beats the stake-only rule by ≥ 0.5 wins/team-season | **+0.09** [0.05, 0.13] (best constant δ = 0.01 WP, tuned in-sample) | **no** |
| 2. Threshold gap early (1st) vs late (8th inning), same pitch, 1 left | **22.2 pp** [20.7, 23.5] | yes |
| 2. Same, 2 left | **27.9 pp** [26.5, 29.2] | yes |
| 2. Holding 1 vs 2 challenges (5th inning) | **18.5 pp** [17.9, 19.0] | yes |

Bar 1 needs the optimum to beat *both* rules, so Bar 1 fails. Bar 2 holds in every sensitivity variant (15–36 pp).

**How to read this:** the option value moves the optimal threshold a lot, but the value curve is flat around the optimum. A single leverage-scaled rule, challenge iff p ≥ δ/(g+δ) with g in win probability and δ ≈ 0.01, gets 98 % of the optimum. What costs wins is (i) pricing the stake in runs instead of win probability, and (ii) teams being too strict, above all while they still hold 2 challenges.

### Observed 2026 behaviour vs the optimum

Wins gained from challenges per team-season:

| Policy | Wins | Challenges per team-game |
|---|---|---|
| Optimal (DP) | 4.43 [4.29, 4.57] | 2.92 |
| Best stake-only WP rule | 4.34 [4.20, 4.49] | 2.34 |
| Teams' fitted rule (simulated) | 3.52 [3.39, 3.64] | 2.22 |
| **Observed 2026 (realised)** | **3.34** [3.18, 3.51] | 2.17 |
| Savant 0.2 rule | 3.19 [3.06, 3.33] | 1.45 |

- **Wins lost by the observed policy: 1.09** [0.95, 1.23] per team-season, about 33 wins league-wide. Against the simulated fitted rule, which uses the same information model, the loss is 0.92 [0.85, 0.98].
- Savant's published rule would do *worse* than what teams actually did.

By role (wins per team-season):

| Role | Observed | Optimal | Lost | Observed challenges | Overturn rate |
|---|---|---|---|---|---|
| Batter (called strikes) | 1.44 | 1.87 | 0.43 | 4,763 | 48.9 % |
| Catcher + pitcher (called balls) | 1.90 (catcher 1.86, pitcher 0.04) | 2.56 | 0.66 | 5,609 + 179 | 58.5 % / 39.7 % |

Teams under-challenge: 0.98 batter and 1.19 fielder challenges per team-game, against 1.27 and 1.65 under the optimum. Figure 1 shows the revealed thresholds. They have the right shape (falling with the inning, higher with 1 left) but sit 10–20 pp too high when the team holds 2.

### Sensitivity (point estimates, wins/team-season)

| Variant | opt − Savant | opt − stake-only | opt − observed | Bar 2 gaps (pp) |
|---|---|---|---|---|
| Main | 1.24 | 0.09 | 1.09 | 22 / 28 / 18 |
| Perception noise σ_r × 0.5 | 0.77 | 0.16 | 3.49 | 28 / 36 / 26 |
| σ_r × 1.5 | 1.45 | 0.07 | 0.05 | 19 / 24 / 15 |
| Truth model without the selection correction | 1.17 | 0.08 | 1.22 | 23 / 29 / 19 |
| Players use a flat prior (overconfident) | 0.35 | −0.15 | 0.57 | 21 / 36 / 20 |
| No extra-inning value in the DP | 1.14 | −0.02 | 0.99 | 22 / 28 / 18 |

- The opt − stake-only gap stays below 0.2 wins in every variant.
- **The wins-lost estimate depends on the estimated perception noise**: it ranges from 0.05 to 3.5 when σ_r is scaled by 1.5 or 0.5. The main calibration is supported by the fitted-rule simulation reproducing reality: 3.52 vs 3.34 wins, 2.22 vs 2.17 challenges per game, and overturn rates by role (below).

### Pipeline

Scripts are in `python/`, run in this order:

1. `fetch_challenges.py`: Stats API live feeds, 2,429 games, 714,483 pitches, **10,557 challenges** (exactly the published total). The per-game sums of successful and failed challenges reconcile with `gameData.absChallenges` on 4,455 of 4,456 team-games.
2. `build.py`: joins to Statcast (100 % match) and rebuilds the original call (flipped if overturned) and the challenger role (catcher 5,612, batter 4,766, pitcher 179). It also tracks challenges left per team. The inferred extra-inning rule (a team at 0 gets 1 at the start of each extra inning) means no challenge is ever made with 0 left. It computes the signed distance to the ABS zone at the plate midpoint, with the ball radius included.
3. `wp.py`: count-level WP and RE engine from 2026 pitch transitions. Checks: home WP is 0.636 tied in the bottom of the 9th, 0.839 up 1 in the top of the 9th, and 0.806 tied in the bottom of the 10th with the runner on 2nd.
4. `stakes.py`: g = WP swing of a flipped call (median 0.008; median challenged pitch 0.011) and RV in runs.
5. `truth.py`, `perception.py`, `joint.py`: the missed-call model P(ABS strike | location), probit by edge.
   - It is fitted **jointly with the challenge decision**, which corrects for challenged pitches being selected on the player's signal: μ_vert = −0.46 in, σ_m,vert = 0.85 in; μ_horiz = −0.19 in, σ_m,horiz = 0.52 in.
   - Perception noise: **batter σ_r = 3.2 in, catcher/pitcher 2.3 in**, consistent across edges.
   - The overturn AUC of the naive fit is 0.948. Predicted vs observed overturn rate: batter 0.488 vs 0.489, catcher 0.583 vs 0.585.
6. `sim.py`: the DP and the policy simulator.
   - Truth and signals are drawn from the fitted model.
   - The player's confidence is the Bayes posterior under the empirical prior.
   - DP state: half-inning slot, opportunity index and challenges left. Extras top-up included.
   - Values are summed over the real 2026 call stream; R = 10 draws.
   - The DP pools future stakes within each half-inning, so its value is a lower bound on the true optimum.
7. `analyze.py`, `boot_thresholds.py` (200 game bootstraps with the DP re-solved), `plots.py`.

Outputs are in `outputs/`: `res_*.json` (one file per variant), `bar2_boot.json`, `roles_observed.csv`, `fig1_threshold.png`, `fig2_wins.png`.

### Kernel checks (Wolfram 15.0.0 for Mac OS X ARM, 26 May 2026)

Script `wolfram/check.wl`; output `outputs/check.json`.

- **A.** Reduce gives challenge iff p ≥ Δ/(g+Δ). A search for any p, g, Δ where the two forms disagree returns `False`, so they are equivalent.
- **B.** ∂p*/∂Δ = g/(g+Δ)² > 0 and ∂p*/∂g = −Δ/(g+Δ)² < 0.
- **C.** Probit convolution, E Φ((w+η−κ)/σ_r) = Φ((w−κ)/√(σ_m²+σ_r²)): checked symbolically on the densities (`True`) and numerically to 4e−29.
- **D.** The selection-corrected likelihood term equals a bivariate-normal orthant probability. It matches the quadrature used in `joint.py` to 2e−23.
- **E.** Under a flat prior, P(wrong | s) = Φ(s/σ_r) (`True`).
- **F.** √t law on the calibrated DP: Δ₁ against expected opportunities left n gives Δ₁ ≈ −0.003 + 0.0031·√n with R² = 0.998, against 0.95 for a linear fit (home and away alike). The option value of a retained challenge scales like √(opportunities left) on real data too.

### Caveats

- The WP engine is our own: league-average transitions, no team or player quality.
- The Savant rule uses our RE-based RV, not Savant's RE288 table.
- The perception model is Gaussian with a constant σ per role. Heterogeneity across players would change the wins-lost number, not the bars.
- The game bootstrap holds the fitted model parameters fixed; Monte Carlo error with R = 10 is ≈ 0.01 wins.
- The stake-only δ is tuned in-sample, which favours the simple rule. That is conservative for the GO decision.
