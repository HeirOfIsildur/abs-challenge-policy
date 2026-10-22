## Headline table (rebuilt on corrected geometry)

Main specification: kappa_sr_region_g (BIC-best homogeneous model) on raw Statcast 2026 geometry with a deterministic ABS truth, sim.draw with isotonic calibration cross-fitted on an auxiliary draw, no-ABS games dropped. Point estimate from the full-sample fit (R = 40 simulation draws, Monte Carlo SE 0.008); 95 % CIs from a game bootstrap that refits the truth and perception model, re-solves every DP and re-simulates in each of B = 397 replicates (R = 10 draws each). Units: wins per team-season (162 games); challenges per team-game. Path-sum accounting; the causal re-simulation agrees (below).

**Wins lost against the optimum: 0.69 [0.58, 0.81] per team-season**, about 21 wins league-wide. Batters lose 0.39 [0.32, 0.49], catchers/pitchers 0.30 [0.22, 0.38].

**AoAS exit rule (switch to TAS if the 95 % CI of wins lost includes <= 0.25 or is wider than 1.5): passes.**

Monte Carlo: the bootstrap used R = 2 draws per replicate before; its CI [0.50, 0.75] (SD 0.069) becomes [0.58, 0.81] (SD 0.057) at R = 10, same resamples and refits.

| Policy | Wins per team-season | Share of optimum | Wins lost vs optimum | Challenges per team-game | Batter / fielder wins |
|---|---|---|---|---|---|
| Clairvoyant ceiling (knows the future (p, g) of its team-game) | 4.85 [4.68, 5.05] | 120.3 % [118.6, 121.9] | -0.82 [-0.89, -0.76] | 3.41 [3.38, 3.46] | 0.00 / 4.85 |
| Optimum (score-aware DP) | 4.03 [3.88, 4.19] | 100.0 % [100.0, 100.0] | 0.00 [0.00, 0.00] | 2.71 [2.67, 2.75] | 1.83 / 2.20 |
| Namuduri runs rule (rebuilt on our model) | 3.95 [3.80, 4.11] | 97.9 % [97.4, 98.7] | 0.08 [0.05, 0.11] | 2.98 [2.94, 3.02] | 1.78 / 2.17 |
| Pooled DP (original 'optimum') | 3.92 [3.77, 4.08] | 97.2 % [96.4, 98.0] | 0.11 [0.08, 0.15] | 2.68 [2.65, 2.73] | 1.78 / 2.14 |
| One-line rule, delta = 0.008 | 3.85 [3.69, 3.99] | 95.4 % [94.4, 96.1] | 0.19 [0.16, 0.23] | 2.21 [2.15, 2.27] | 1.73 / 2.12 |
| One-line rule, delta = 0.01 | 3.76 [3.60, 3.94] | 93.3 % [91.9, 94.1] | 0.27 [0.25, 0.32] | 1.90 [1.84, 1.96] | 1.67 / 2.10 |
| Arora card, published thresholds | 3.80 [3.65, 3.94] | 94.3 % [92.9, 94.9] | 0.23 [0.20, 0.29] | 2.59 [2.54, 2.64] | 1.69 / 2.11 |
| Arora card, refit on our DP | 3.76 [3.61, 3.89] | 93.3 % [91.9, 93.8] | 0.27 [0.24, 0.33] | 2.52 [2.47, 2.56] | 1.67 / 2.09 |
| Fitted behavioural rule (simulated) | 3.31 [3.16, 3.45] | 82.0 % [80.3, 83.5] | 0.72 [0.66, 0.81] | 2.15 [2.12, 2.19] | 1.43 / 1.88 |
| Savant 0.2 rule | 2.47 [2.33, 2.66] | 61.2 % [59.5, 63.9] | 1.56 [1.47, 1.62] | 1.02 [0.99, 1.08] | 0.96 / 1.51 |
| Observed (realised) | 3.34 [3.18, 3.50] | 82.8 % [79.8, 85.4] | 0.69 [0.58, 0.81] | 2.17 [2.13, 2.22] | 1.44 / 1.90 |
| Savant form, run cost tuned (savant_c_0.08) | 3.71 [3.57, 3.88] | 92.1 % [91.0, 93.3] | 0.32 [0.27, 0.36] | 2.44 [2.38, 2.50] | 1.64 / 2.08 |

One-line rule p ≥ δ/(g+δ): best δ = 0.008; with δ re-selected in each replicate it gets 95.3 % [94.4, 96.1] of the optimum.

### Per-player model

Player random effects on (κ, log σ_r) (the best-BIC model); each player's posterior is calibrated within their own noise level. Optimum 4.10, wins lost **0.77** [0.66, 0.92] (game bootstrap with EB redraws of the player effects, no refit, B = 150; the same no-refit bootstrap gives [0.56, 0.82] for the main spec). Best one-line δ = 0.008, 95.4 % of the optimum.

### Model spread (all perception variants, same corrected pipeline)

| Variant | Wins lost | Wins lost vs pooled DP |
|---|---|---|
| sm_edge | 0.64 | 0.54 |
| sm_region | 0.65 | 0.55 |
| normal | 0.65 | 0.55 |
| logistic | 0.89 | 0.77 |
| t10 | 0.85 | 0.74 |
| t5 | 1.00 | 0.89 |
| t3 | 1.19 | 1.07 |
| sr_edge | 0.66 | 0.55 |
| sr_region | 0.69 | 0.58 |
| sr_slope | 0.63 | 0.52 |
| sr_region_slope | 0.64 | 0.54 |
| kappa_region | 0.67 | 0.57 |
| kappa_sr_region | 0.70 | 0.58 |
| kappa_sr_region_slope | 0.66 | 0.54 |
| kappa_sr_region_g | 0.71 | 0.59 |
| re | 0.78 | 0.67 |
| main_logistic | 0.92 | 0.81 |
| main_t10 | 0.89 | 0.78 |
| main_t5 | 1.04 | 0.92 |
| main_t3 | 1.23 | 1.10 |
| main_lapse | 0.86 | 0.75 |
| main_glow | 0.71 | 0.59 |
| main_sr_g | 0.74 | 0.64 |
| linear_lapse | 0.85 | 0.74 |
| re_lapse | 0.92 | 0.82 |

### Causal wins vs path-sum

Anchored re-simulation (`src/causal.py`; R = 10 draws, K = 1000 continuations per world per branch). Game-bootstrap CIs.

Simulator: `run` (events and in-play outcomes by count, outs and bases, plus baserunning between pitches; its flip swing is 99.2 % of the engine's g). The original simulator's swing was 96.7 % of g, which predicts a binary − RB gap of -0.075 for the optimum; it measured -0.015 [-0.059, +0.033] (audit R1, `results/causal_gap.md`).

| Policy | Path-sum | Causal (RB) | Causal − path | Binary − RB | Causal − observed |
|---|---|---|---|---|---|
| Optimum (score-aware DP) | 4.04 [3.89, 4.19] | 4.04 [3.89, 4.19] | +0.000 [-0.001, +0.002] | -0.030 [-0.048, -0.013] | 0.70 [0.59, 0.81] |
| Namuduri runs rule (rebuilt on our model) | 3.95 [3.80, 4.10] | 3.95 [3.81, 4.10] | +0.002 [+0.000, +0.003] | -0.019 [-0.037, -0.001] | 0.61 [0.51, 0.73] |
| Pooled DP (original 'optimum') | 3.93 [3.79, 4.08] | 3.93 [3.79, 4.08] | +0.004 [+0.002, +0.005] | -0.025 [-0.043, -0.006] | 0.59 [0.48, 0.71] |
| One-line rule, delta = 0.008 | 3.86 [3.71, 4.00] | 3.86 [3.71, 4.00] | +0.001 [-0.001, +0.002] | -0.024 [-0.042, -0.007] | 0.52 [0.41, 0.63] |
| One-line rule, delta = 0.01 | 3.76 [3.61, 3.91] | 3.76 [3.61, 3.91] | +0.001 [-0.001, +0.003] | -0.024 [-0.041, -0.006] | 0.42 [0.32, 0.53] |
| Arora card, published thresholds | 3.82 [3.69, 3.95] | 3.82 [3.69, 3.96] | +0.002 [+0.000, +0.004] | -0.003 [-0.021, +0.014] | 0.48 [0.38, 0.59] |
| Arora card, refit on our DP | 3.78 [3.64, 3.91] | 3.78 [3.64, 3.91] | +0.001 [-0.001, +0.003] | -0.019 [-0.036, -0.002] | 0.44 [0.34, 0.55] |
| Fitted behavioural rule (simulated) | 3.31 [3.18, 3.44] | 3.31 [3.18, 3.44] | +0.001 [-0.000, +0.003] | -0.007 [-0.025, +0.009] | -0.03 [-0.13, 0.07] |
| Savant 0.2 rule | 2.47 [2.35, 2.59] | 2.47 [2.35, 2.59] | +0.001 [-0.000, +0.003] | -0.007 [-0.025, +0.010] | -0.87 [-0.97, -0.76] |

### Flat optimum on the calibrated pools (Theorem 3 of theory.md)

Exact regulation-only score-aware MDP. Best constant δ = 0.008: loss 6.0 % of V*, local quadratic (envelope) approximation 6.4 %, sup-density bound (ii) 45.5 % (holds, loose). Regret identity (i) reproduces V* − V^δ to 6.7e-17 at every δ.

### Optimum robustness

Score clip ±6 vs ±10: 4.037 vs 4.038. 2-fold cross-fit (DP on half the games, evaluated on the other half): 4.032 vs 4.037 in-sample.

Kernel checks (`wolfram/headline.wl`, 15.0.0 for Mac OS X ARM (64-bit) (May 26, 2026)): T1_regret_D_gt_delta, T1_regret_D_lt_delta, T1_envelope_r'(D)=0, T2_causal_decomposition, T2_rb_difference, T2_rb_difference_signed, T2_rb_own_only_misses_opponent, T3_runs_rule, T3_stake_rule — all True.
