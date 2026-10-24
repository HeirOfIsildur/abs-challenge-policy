## Audit R1: why the binary causal estimate sat below the RB one 

### Verdict

**A simulator bug, now fixed, not a property of the game or of the headline accounting.** The pitch simulator in `src/causal.py` moved a team's win probability by only **95.4 %** of the engine's g when a call was flipped. The binary estimator values the branch flip in simulated wins, so it inherited that shortfall: +0.000 [-0.048, +0.048] wins per team-season for the optimum (run, R = 4, K = 300). Pricing every branch with the old simulator's own exact g predicts **-0.079**, which accounts for the whole measured gap with no free parameter. The corrected simulator reaches 98.8 % of g. On it binary − RB is -0.008 [-0.025, +0.009] for the optimum (R = 10, K = 1000), against -0.010 predicted, and it averages -0.011 over the nine policies (was -0.010).

**The headline keeps the path-sum accounting priced in engine g.** The RB causal estimate agrees with it (+0.000 [-0.002, +0.002]). Measured in simulated wins (binary), teams lose 0.68 [0.57, 0.80], against 0.69 [0.58, 0.81] for RB; before the fix the binary figure was 0.68 [0.56, 0.81]. The reasons to keep the engine are:

1. The engine is the calibrated object. Its PA kernel is the empirical end-of-PA distribution from every 2026 pitch, and it matches realised results as well as Statcast's own win expectancy (`audit/wp_calibration.json`). The simulator is a generative approximation of the same data, used only to replay games after a flip.
2. With the simulator fixed, the two estimators differ by -0.008 for the optimum. -0.010 of that is the residual simulator mismatch predicted by the exact chain, and the rest is within the CI half-width (0.017). Across the nine policies the difference averages -0.011. The binary estimator adds win-indicator noise and no information.
3. Any residual mismatch between the simulator and the engine biases the binary estimate and not the path-sum. The residual is 1.2 % of g, predicted -0.010 wins.

### Where the gap came from

`src/causal_gap.py` writes every simulator as a pitch-level Markov chain and computes its PA kernel, its WP table and its g exactly, with no Monte Carlo. The simulator itself (`causal.mc_win`, K = 2000 paths from 400 branch states) reproduces its chain's flip swing to +0.0003 ± 0.0005 (old) and -0.0004 ± 0.0005 (new). The decomposition is therefore exact up to the second-order term (g mismatch at the team's later flips, A − A').

Predicted binary − RB for the optimum, wins per team-season, adding one ingredient at a time (105 branch flips per team-season, mean engine g 0.0204):

| Simulator | Events | In-play outcome | Between-pitch baserunning | g / engine g | Predicted binary − RB | Step |
|---|---|---|---|---|---|---|
| sim  | by count | by the PA's starting outs/bases | none | 95.4 % | -0.079 | — |
| kc | by count | by count, outs, bases | none | — | -0.044 | +0.035 |
| kcob | by count, outs, bases | by count, outs, bases | none | — | -0.064 | -0.020 |
| run  | by count, outs, bases | by count, outs, bases | yes (1.0 % of pitches) | 98.8 % | -0.010 | +0.054 |
| engine (`wp.py`) | empirical end-of-PA kernel by count, outs, bases | | (included) | 100 % | 0 | +0.010 |

Measured binary − RB: +0.000 [-0.048, +0.048] with the simulator, -0.008 [-0.025, +0.009] with the corrected one.

Reading the steps:
- **Count-dependent contact** (sim → kc) closes 44 % of the gap. Batters ahead in the count hit for more damage (runs scored on the ball in play, averaged over the 24 base-out states: 0-0 0.476, 0-2 0.446, 1-2 0.462, 2-0 0.510, 3-1 0.520). A count-blind in-play kernel compresses the value of moving from a strike count to a ball count. The old kernel was also indexed by the state at the PA's first pitch, and its runs included runners who moved during the PA.
- **Events by outs and bases** (kc → kcob) moves the other way (-26 %): pitch mix and swing rates depend on the base-out state.
- **Baserunning between pitches** (kcob → run) closes 68 %. Steals, wild pitches and pickoffs happen on about 1 % of pitches. Their chance accumulates over the extra pitches a ball buys, and the engine's empirical kernel contains them.

By count of the branch pitch, the old simulator under-priced early-count flips most (g / engine g − 1):

| Count | n | sim | kc | kcob | run |
|---|---|---|---|---|---|
| 0-0 | 3005 | -11.0 % | +1.6 % | -0.4 % | +1.3 % |
| 0-1 | 1027 | -8.6 % | -0.2 % | -0.5 % | +1.3 % |
| 0-2 | 617 | -1.2 % | -4.9 % | -4.7 % | -2.2 % |
| 1-0 | 1414 | -9.8 % | -3.4 % | -6.5 % | -4.5 % |
| 1-1 | 1027 | -8.3 % | +0.3 % | -0.4 % | +1.2 % |
| 1-2 | 965 | -1.9 % | -2.9 % | -3.3 % | -0.9 % |
| 2-0 | 740 | +4.5 % | +3.1 % | -0.7 % | +2.2 % |
| 2-1 | 783 | -4.5 % | -1.5 % | -1.8 % | +0.7 % |
| 2-2 | 1105 | -2.5 % | -3.6 % | -3.8 % | -1.2 % |
| 3-0 | 337 | -1.7 % | -6.7 % | -6.8 % | -3.2 % |
| 3-1 | 547 | -0.9 % | -2.8 % | -3.3 % | +0.2 % |
| 3-2 | 1024 | -1.2 % | -2.8 % | -3.4 % | -0.4 % |

By inning and margin, the shortfall is spread over the whole game (share of the old gap in the last column):

| Inning | Margin | n | engine g | sim | run | share of old gap |
|---|---|---|---|---|---|---|
| 1-3 | tied | 3576 | 0.0206 | -3.5 % | -0.3 % | 27 % |
| 1-3 | 1-2 | 2429 | 0.0208 | -3.5 % | -0.5 % | 18 % |
| 1-3 | 3+ | 670 | 0.0148 | -3.2 % | -0.1 % | 3 % |
| 4-6 | tied | 681 | 0.0250 | -3.2 % | +0.1 % | 6 % |
| 4-6 | 1-2 | 1898 | 0.0226 | -3.4 % | -0.4 % | 16 % |
| 4-6 | 3+ | 1129 | 0.0130 | -4.2 % | -1.3 % | 6 % |
| 7-8 | tied | 212 | 0.0272 | -6.1 % | -0.6 % | 4 % |
| 7-8 | 1-2 | 705 | 0.0296 | -4.4 % | -0.7 % | 10 % |
| 7-8 | 3+ | 696 | 0.0106 | -2.3 % | -1.2 % | 2 % |
| 9 | tied | 53 | 0.0290 | -10.1 % | +0.6 % | 2 % |
| 9 | 1-2 | 184 | 0.0351 | -3.2 % | +0.8 % | 2 % |
| 9 | 3+ | 226 | 0.0057 | -1.0 % | -2.6 % | 0 % |
| 10+ | tied | 69 | 0.0379 | -10.9 % | -4.7 % | 3 % |
| 10+ | 1-2 | 49 | 0.0361 | -5.3 % | +0.2 % | 1 % |
| 10+ | 3+ | 14 | 0.0222 | -4.5 % | -2.9 % | 0 % |

### The other candidates

- **WP calibration of the simulator (levels).** From 400 branch states, the simulated home win rate after the original call matches the engine WP to +0.0003 ± 0.0005 (old) and +0.0004 ± 0.0005 (new) in every WP decile (`calibration` in `results/causal_gap.json`). This is why the audit's B-world check looked fine: the defect was in the **swing** of a flip, not in the level. The B-world offset of about −0.012 is the team giving up its challenges while the opponent keeps its own. It is the same at home (-0.0145) and away (-0.0151).
- **Missing home-field term.** Neither the engine nor the simulator has one, so it cannot separate them. Re-estimating the engine with one PA kernel per batting team gives a home WP of 0.527 at the first pitch and changes g on the optimum's branch flips by -1.0 % and on the observed successes by -1.1 %. The first-order effect on wins lost is -0.003 (audit R5 closed: negligible).
- **Extra innings.** Both use the runner on second and the same top-up rule. Branches in extra innings contribute -0.003 of the old -0.079.
- **The branch point (first successful challenge only).** Path-sum and both causal estimators share the recorded path up to the branch by construction, so it cannot create a binary − RB difference. Its own effect is RB − path = +0.000 [-0.002, +0.002].
- **Common random numbers.** Worlds A, A′ and B share seeds per (branch, k), which lowers the variance of A − A′. Both estimators are means of i.i.d. continuations, so CRN cannot bias them.
- **K / R Monte Carlo bias.** Both estimators are linear in sample means, hence unbiased in K and R. The old gap's CI excluded 0, and the exact chain reproduces it (-0.079 predicted vs +0.000 measured), so it was not noise.

### Recommended manuscript wording (for )

*§5 (causal check), one paragraph:* "Summing g along the recorded game ignores that, after a flipped call, the rest of the game would have been different. We therefore replay every team-game from its first successful challenge in a pitch-level simulator: events and in-play outcomes by count, outs and bases, plus baserunning between pitches, all estimated from 2026 Statcast. The flipped call is compared with the original one, holding everything else fixed (common random numbers). The simulator reproduces the engine's win probabilities in every decile, and a flip moves its win probability by \SimGShareRun\,\% of g. Valued this way, the optimum's value changes by \CausalMinusPath{} \CausalMinusPathCI{} wins per team-season relative to the path-sum, and teams lose \CausalLost{} \CausalLostCI. Using simulated win indicators instead of g gives \BinaryLost{} \BinaryLostCI; the difference, \BinaryMinusRB{} \BinaryMinusRBCI, is about the size the remaining \SimGShareRunGap\,\% mismatch between simulator and engine predicts (\GapPredRun). A simpler simulator that ignores the count when drawing contact and has no baserunning moves win probability by only \SimGShare\,\% of g, and for exactly that reason its win-indicator estimate differs from g-based accounting by \BinaryMinusRBSim{} (Supplement~S\ref{sec:causal-gap})."

*Supplement section "Engine versus simulator":* the decomposition table above; the count and inning tables; the home-field paragraph; the Monte Carlo validation of the exact chains.
