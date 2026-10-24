## Independent adversarial audit of the full pipeline (HONZA-89)

Base: `honza-88-geometry` (PR #6). I read the code as a hostile referee hunting the next convention bug, then checked each
item numerically. Scripts in `audit/` test one claim each. `audit/independent.py` and `audit/drs/recount.py` do not
import `src/`. Kernel checks ran on Wolfram 15.0.0 for Mac OS X ARM (26 May 2026). The DRS part has its own table in
`audit/drs/audit_drs.md` and is summarised here.

### Bottom line

- **Bugs found: 7, none severe.**
  - MLB: 4 code bugs, all low (B1–B4), and 1 provenance bug (P1).
  - DRS: 1 code bug and 1 stale report, both low.
- **No convention bug of the HONZA-88 kind.** Units, the `plate_x` sign, L/R mirroring, the ball radius, `sz` per pitch,
  role and team attribution, the original call and the challenge counters all check out against independent numbers.
- **Headline after the fixes:** wins lost **0.64 [0.50, 0.78] → 0.62 [0.50, 0.75]** per team-season (19 wins
  league-wide). Most of the move is B3, the in-sample posterior calibration. The AoAS exit rule still passes: the lower
  end is 0.50 against a trigger of 0.25, and the width is 0.25 against a trigger of 1.5.
- **Flagged for Honza** (judgement calls, not changed):
  - R1: the win-indicator causal check disagrees with the RB estimate by −0.09 wins (medium).
  - R2: the CI is inflated by Monte Carlo noise.
  - R3: the DP is a restricted-state optimum.
  - R4: the simulated behavioural rule falls short of realised play by 7 %.
  - DRS: identification of s is fragile from above (medium).

### Fixes

| # | Bug | Evidence | Effect | Fix |
|---|---|---|---|---|
| B1 | 4 games played at venues **without ABS** were in the sample: the Mexico City series (25–26 Apr), Field of Dreams (13 Aug) and the Little League Classic (23 Aug). They count as 8 team-games of "never challenge". | `feed_games.has` is None in exactly these 4 games and False in 23 games that had ABS but no challenge. `has = True` ⇔ ≥ 1 challenge, on 2,402 / 2,402 games. | ≈ +0.007 wins lost; per-team-game challenge counts and the likelihood are slightly diluted. | `build.py` drops `has` = None; analysis sample is 2,425 games (`\NGamesABS`) |
| B2 | The OOS feeds dropped every ABS game **without a challenge**: 28 Triple-A, 4 spring and 23 MLB-feed games. | `build_ext.py` kept only `has` = True. | Biased the OOS challenge rate upward by ~1 %; Triple-A σ_r 3.448 → 3.457 in. | Keep `has` = False; drop only None |
| B3 | The player's calibrated posterior was fitted **in sample**. `sim.draw` fitted the isotonic map P(W \| s) on the same simulated truth W that then scores the policies, so the realised truth leaked into the decision. | `audit/crossfit_calibration.py` (R = 6) fits the map on an independent draw. Optimum −0.015, one-line rule −0.026, Savant −0.005 wins per team-season. Larger for the per-player model, whose 20 smaller calibration groups overfit more: 0.77 → 0.69. | Optimum 3.98 → 3.96; headline −0.02. `dp_checks` cross-fit only re-fitted the DP pools, so it could not see this. | `sim.draw` fits the map on an auxiliary draw |
| B4 | The extra-inning top-up was missed when an extra inning's first retained opportunity was in the bottom half, or came after an inning without opportunities. | 285 extra innings; the old flag fired in 281. | Negligible. | `sim.extra_starts`: first row of each extra inning |
| P1 | `results/roles_observed.csv` (`\NChBatter`, `\OverturnBatter`, …) was a HONZA-71 artefact that no script in the run order wrote. `plots.py` writes `figures/roles_observed.csv` from legacy `sim_main.pkl`. | Batter 4,763 / catcher 5,609 vs 4,766 / 5,612 in the feed (`audit/independent.json`). | `\NChBatter` and `\NChCatcher` were 3 too low. | `data_counts.py` writes it from the called-pitch table |
| D1 | DRS: the simulated "fitted rule" looked up its threshold with the *real* reviews left instead of the simulated ones. | `src/drs/dp.py` | Test lost-vs-rule 0.033 → 0.027; ODI 0.002 → 0.000. | fixed |
| D2 | DRS: `results/drs/report.md` quoted pre-rebuild MLB figures. | — | — | updated to the HONZA-89 numbers |

Also tidied:
- `feed_check.py`, `sr_breakeven.py` and `final_ci.py … kappa_sr_region` were missing from the README run order; added.
- The README now has an audit section.
- I merged `b19c683` (pruning of the retired `normal_edge` / `normal_region` variants) and applied the same pruning to
  the re-run files.

### Checklist (MLB, items 1–10)

| # | Item | Verdict | Evidence | Severity |
|---|---|---|---|---|
| 1a | feet vs inches | OK | `geometry.signed_distance` computes in ft and returns ×12 once. w, σ_r and κ are in inches throughout; tests in `tests/test_geometry.py`. | — |
| 1b | `plate_x` sign, catcher's view, L/R mirroring | OK | The distance uses \|px\|, so it is sign-free. `lik.region` puts inside at px < 0 for RHB, catcher's view. The rule reproduces L 99.60 % and R 99.70 %. | — |
| 1c | Ball radius applied once | OK | Only in `signed_distance`. A radius sweep is sharp at 1.45 in (37 mismatches vs 76 at 1.43 in and 93 at 1.47 in). | — |
| 1d | `sz` per pitch vs per batter; mid-season re-certification | OK | Per-pitch lines are used. 53.5 % / 27 % of a single height hold to 1e−8 on 99.9 % of pitches. 62 batters re-certified (max 0.36 in); none of the 37 mismatches involves one. | — |
| 2a | Challenger role | OK | ch_player = batter / pitcher / `fielder_2` gives 4,766 / 179 / 5,612, with 0 "other". Batters challenge only strikes (orig_strike mean 1.000) and fielders only balls (0.000). | — |
| 2b | Home/away, top/bottom | OK | `ch_team` is the batting team on 100 % of batter challenges and the fielding team on 100 % of fielder challenges. The engine raises home WP by +0.145 for a run in the bottom 5th and lowers it by 0.141 for a run in the top 5th. | — |
| 2c | Challenges left vs the feed's `usedSuccessful/usedFailed/remaining` | OK | Successful and failed counts match per team on every game except 822717. There, one ABS challenge is nested in a manager review's `additionalReviews` on a play whose pitch is ambiguous; it is left out (1 of 10,558). `remaining` is exactly max(0, 2 − usedFailed) on every game, so the feed ignores extra-inning grants. Our counter differs from it only in the 113 team-games that received a grant. usedFailed reaches 4, which proves grants exist. No challenge is made with 0 left. | info |
| 2d | Extra-inning grant vs the official rule (one per extra inning if none remain) | OK | `build.py` gives c = 0 → 1 at the start of every extra inning and never stacks. In the simulation, see B4. | — |
| 3 | Original call restored; swings excluded | OK | orig = final XOR overturned. Only codes C/B/*B count; all 10,557 challenged pitches are C or B. The independent rule reproduces 99.65 %. | — |
| 4a | WP calibration by inning and score | risk | `audit/wp_calibration.py`: Brier 0.1585, equal to Statcast's `home_win_exp` (corr 0.9996). With game-clustered SEs, 3 of 50 cells have \|z\| > 3. The engine has **no home-field advantage**: first pitch 0.500, realised home win 52.9 %, +2.4 pp in tied states. It is also slightly under-confident in blowouts (0.986 vs 0.995). g is a WP difference, so a level shift is second order. | low |
| 4b | g for a PA-ending flip | OK | `stakes.Vcount` handles ball 4 (forced advance, walk-off) and strike 3 (out; with 2 outs, half over). Bottom 9, bases loaded, 2 outs, tie, 3-2: g = 0.500 (walk-off vs extras). | — |
| 5a | Selection term | OK | Deterministic truth: P(ch, y \| w) = F((w−κ)/σ_r)·[(1−π)1{y=W} + π1{y≠W}], no challenge = 1 − F. Opportunities are pitches with left > 0 and the pre-pitch count. | — |
| 5b | Outcome leakage into the decision | OK (model) / **bug B3** (simulation) | The design X uses only pre-pitch state; the ruling enters only y. In the simulation, the posterior was fitted on the scoring truth (B3). | low |
| 5c | Isotonic calibration inside each bootstrap replicate | OK | `headline.one` refits the likelihood on the resample, and `sim.draw` runs on the resample frame. | — |
| 6a | DP state space | risk | The score-aware DP state is (half-inning slot, score difference at the half's start, opportunity index, c). Outs, bases and count enter only through the immediate g, and the extras block is a score-free stationary approximation. So the "optimum" is the best rule in a restricted state: a lower bound on the true optimum, which makes wins lost conservative. | low (R3) |
| 6b | Terminal condition | OK | No bottom 9th when the home team leads; after the 9th only d = 0 continues; c = 0 keeps the top-up value. | — |
| 6c | Cross-fit truly out of sample | **bug B3** | DP pools were cross-fitted but the posterior map was not. Fixed. | low |
| 6d | Same random numbers across policies | OK | Every policy runs on the same (W, s, pt) per draw. | — |
| 7a | Resampling unit = game; refit inside | OK | Games are drawn with replacement under new ids; the likelihood, calibration and every DP are refitted. The WP engine, g and P(runs \| slot) stay fixed (data-rich, but a stated simplification). | low |
| 7b | Percentile vs basic; B large enough | OK / risk | Percentile [0.502, 0.753]; basic and normal intervals agree within 0.01. The MC SE of the endpoints at B = 120 is 0.015 / 0.012. **R2:** each replicate uses R = 2 draws, and the per-draw SD of wins lost is 0.061 (`audit/mc_error.json`), so MC noise contributes about 0.043 of the bootstrap SD of 0.072. The CI is therefore conservative; R = 10 would shrink it. The point estimate's MC SE at R = 10 is 0.019. | low |
| 8a | Arora's card as published | OK | All 24 cells match his `tier1_card_2026.csv` to 5e−5. Bands, count class and terciles match `build_card`. Our tercile cell medians are within 0.9–1.0× of his in the mid and high g cells. | — |
| 8b | Namuduri's rule as in his code | OK | `abs_policy.py`: challenge iff p·dRE > (1−p)·C(t, k), with C taken at the start of the half-inning; his k = incorrect challenges spent maps to our c. Our version takes C after the first opportunity (immaterial). | — |
| 8c | Savant's RV on our engine vs RE288 | OK | Our run swing averages 0.142 against 0.150 from an RE288 table built from Statcast `delta_run_exp` (ratio 0.95, corr 0.82 over 288 cells). Our Savant rule is therefore marginally more conservative; the ranking is unaffected (8 % margin to the next policy). | low |
| 9 | Independent recomputation (`audit/independent.py`) | OK | See the table below. | — |
| 10 | Number provenance | **P1** fixed | Every macro reads a `results/` file. The writers are now all in the README run order (`roles_observed.csv` was the exception, P1). A clean clone of the branch plus `make -C paper check` regenerates `numbers.tex` byte for byte (0 pending). The full re-run from the raw parquet files (`audit/rerun.sh`) finishes with 265 macros and 0 pending. Still not done: a re-fetch from the APIs on a fresh machine. | — |

### Item 9: independent recomputation vs the pipeline

Independent script: raw parquet only, its own join, zone rule and bookkeeping. Stakes come from Statcast's own
`delta_home_win_exp` / `delta_run_exp` cell means. Perception is simulated with σ_r = 2.91 / 1.91 in and a pooled
calibration per side.

| Quantity | Independent | Pipeline | Tolerance | |
|---|---|---|---|---|
| Challenges (batter / catcher / pitcher) | 4,766 / 5,612 / 179 | 4,766 / 5,612 / 179 | exact | ✅ |
| Overturn rate (batter / catcher / pitcher) | 48.8 / 58.6 / 39.7 % | 48.9 / 58.6 / 39.7 % | exact | ✅ |
| Challenges per team-game | 2.177 | 2.175 | ±1 % | ✅ |
| Rule reproduces rulings | 99.65 % | 99.65 % | exact | ✅ |
| Observed wins per team-season | 3.18 (batter 1.32, fielder 1.86) | 3.34 (1.44, 1.90) | ±10 % (different engine) | ✅ (−5 %) |
| Savant 0.2 rule | 2.03, 1.00 challenges/team-game | 2.32, 0.93 | ±15 % | ✅ (−12 %) |
| Savant ÷ observed | 0.64 | 0.69 | ranking | ✅ Savant below teams' actual play |
| One-line rule δ = 0.008 | 3.71, 2.16 challenges/team-game | 3.79, 2.16 | ±10 % | ✅ |

### Risks flagged for Honza (not changed)

- **R1, medium: binary vs RB causal estimate.**
  - The win-indicator version of the causal estimate is below the RB version by −0.086 [−0.129, −0.040] wins for the
    optimum. It was already −0.061 [−0.106, −0.019] before this audit, so the fixes did not cause it.
  - Under `causal.py`'s own martingale argument the two should agree when the WP engine is the simulator's value
    function. The B-world calibration (−0.012, slope 1.00) is fine, so the gap comes from how the simulator responds to
    a flip.
  - Measured in simulated wins, wins lost are 0.48 rather than 0.56 (`\BinaryLost`); still well above 0.25.
  - The paper quotes "causal agrees (−0.003)", the RB difference. It should report the binary number too or explain the
    gap. Next step: find which simulator component fails to match the engine.
- **R2, low: CI inflated by MC noise.** See 7b. The CI is conservative.
- **R3, low: restricted-state DP.** See 6a. Wins lost is a lower bound.
- **R4, low–medium: the behavioural model falls short of realised play.**
  - The simulated fitted rule earns 3.11 wins per team-season against 3.34 realised (CIs overlap), at the same
    challenge rate (2.12 vs 2.18).
  - The model's version of the teams is therefore about 7 % less skilled than the real teams.
  - Wins lost against the same-model teams would be 0.85 instead of 0.62. The headline compares with *realised* play,
    which is the conservative choice. The paper should say which one it quotes and why.
- **R5, low: the WP engine has no home-field term** (4a).
- **Data note:** 6 of the 10,557 challenges have no Statcast location. They count against the challenge budget but not
  in the likelihood.

### DRS (items 11–14, `audit/drs/audit_drs.md`)

- **11. Reconstruction: OK.**
  - The independent recount from raw Cricsheet JSON matches the pipeline exactly on every count.
  - The review allowance matches the ICC/IPL rules half-year by half-year.
  - 2 of 10,440 reviews were taken with none left.
- **12. Model.**
  - The band likelihood and the three-way DP are OK; the kernel check matches to 7e−17 (`wolfram/drs_audit.wl`).
  - Bug D1 fixed (see Fixes).
  - **Medium risk:** s is identified only weakly from above. In known-truth recovery, 25–35 % of replicates return
    ŝ > 1 when the truth is about 0.5. §7's "bounds perception noise from both sides" should be softened.
  - Low risk: the ODI and T20 WP engines are over-confident in the shoulders. Recalibrating changes the loss by ≤ 0.02.
- **13. T20 at B = 500.**
  - Optimum 1.222 [1.12, 1.43]; lost vs rule 0.008 [−0.002, 0.014].
  - "No material loss" holds under the constant-rate margin model and the recalibrated engine.
  - It fails if s ≈ 2. That is inside ODI's 95 % set: loss 0.62 per 100 team-matches.
- **14. Provenance: OK.** Every DRS macro and `tab_drs` cell is generated from `results/drs/summary.json`.
  - Prose bug, flagged only: "at most about seven out decisions" should read "on average five to seven"
    (maximum 13 / 11 / 11).

### What moved (`audit/numbers_diff.md`: 138 of 264 macros changed, 15 by more than 10 %)

- **Headline:**
  - wins lost 0.64 [0.50, 0.78] → **0.62 [0.50, 0.75]**;
  - batters 0.32 → 0.31; catchers/pitchers 0.32;
  - optimum 3.98 → 3.96;
  - shares: rule 95.8 %, Savant 58.5 %, teams 84.3 %.
- **Per-player model:** 0.77 [0.64, 0.92] → 0.69 [0.57, 0.83] (B3).
- **Variant spread** of models the data don't reject: 0.54–0.58 (was 0.54–0.61).
- **σ_r break-even:** wins lost reaches 0.25 at +15 % σ_r (was +16 %).
- **Causal:** RB vs path −0.006 → −0.003.
- **Threshold gaps:** 6.3 / 25.7 / 16.1 pp (were 6.5 / 25.7 / 16.2). √n R² 0.999.
- **Unchanged:** σ_r 2.91 / 1.91 in, π 0.33 %, the geometry results, theory, and the competitor ranking.
- **Large relative moves on small numbers:** `RecovBias*` (re-drawn known-truth recovery, still within 0.5 %),
  `LostRuns` 0.07 → 0.04 and `LostPooled` 0.07 → 0.09.

### Verified vs open

- **Verified:**
  - geometry and units;
  - challenge and role reconstruction against the feed's own counters;
  - original-call logic;
  - the likelihood's selection term;
  - bootstrap mechanics;
  - competitor rules against their authors' artefacts;
  - counts and rates recomputed independently;
  - observed and Savant value within 5–12 % on an independent engine;
  - number provenance and a clean-clone paper build;
  - the DRS reconstruction.
- **Open:**
  - R1 (binary causal gap);
  - R4 (which teams baseline the headline uses);
  - the DRS s upper bound;
  - a full re-fetch on a fresh machine;
  - the `main.tex` prose (2,425 ABS games, the DRS "seven decisions", the softened DRS identification claim), left
    for HONZA-77.

## Follow-up (HONZA-100, 18 Oct 2026)

- **R1, fixed.** The binary − RB gap came from the HONZA-86 simulator in `causal.py`. It drew in-play outcomes without the count, drew events without outs and bases, and had no baserunning between pitches, so a flipped call moved its win probability by only 96.4 % of the engine's g. Pricing the branches with that simulator's exact g predicts −0.077 of the measured −0.086. The corrected simulator reaches 99.3 % of g, and binary − RB is −0.024 [−0.041, −0.006] for the optimum (−0.003 on average over nine policies). The headline keeps path-sum / RB. Details: `results/causal_gap.md`.
- **R2, done.** The bootstrap now uses R = 10 per replicate (B = 120, same resamples and refits): wins lost 0.62 [0.52, 0.74], SD 0.060, against [0.50, 0.75] and SD 0.069 at R = 2. The point estimate uses R = 40, Monte Carlo SE 0.008. The exit rule passes.
- **R5, closed.** An engine with one PA kernel per batting team (home WP 0.527 at the first pitch) changes g by −1.0 % on both the optimum's branches and the observed successes; wins lost moves by −0.005.
