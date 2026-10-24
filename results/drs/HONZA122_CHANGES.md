# HONZA-122: DRS referee fixes

Run 21 Oct 2026 on a fresh Cricsheet download (`src/drs/fetch.py`). The pipeline was re-run end to end in README order: parse, wp, build, model band/const, `dp.py all 10 500 2`, report, plots, then `sensitivity.py 10`, `identify.py 20` and `wolframscript -file wolfram/drs_check.wl`. The bootstrap uses **B = 500 in every format, as before**, with 2 workers.

**Data vintage.** The old "≥ 1 review" rule gives exactly the old match counts on the new download (Test 331, ODI 566, T20 1,116). So every old → new difference below comes from the fixes, not from new data. The old outputs are kept in `data/drs_old_honza122/`.

Units: match points per 100 team-matches unless stated. Brackets are 95 % match-bootstrap intervals.

## 1. Outcome-dependent sample selection

**Changes**
- `src/drs/parse.py` now writes `data/drs_matches.parquet`, a per-match table with:
  - event, season, city, teams;
  - whether a TV umpire is listed;
  - the number of recorded reviews;
  - Cricsheet's own flag `info.missing = ["reviews"]`.
- `src/drs/build.py:drs_sample` defines the DRS sample:
  - A match with a recorded review is kept.
  - A match with zero reviews is kept if another match in the same competition, event and season, between the same teams or at the same city, has a recorded review. DRS is set per series, tournament and host.
  - A match flagged `missing: reviews` is excluded whatever the rest says.
- Counts, plus a zero-truncated-Poisson check of the zero-review share, go to `results/drs/sample.json`. They are copied into `summary.json` under the new key `sample`.

**Counts**

| | Test | ODI | T20 (T20I + IPL) |
|---|---|---|---|
| Reviewed matches (old sample) | 331 | 566 | 1,116 |
| Zero-review DRS matches added | 0 | 57 | 215 (105 T20I + 110 IPL) |
| Excluded: Cricsheet flags reviews missing | 0 | 1 | 13 |
| Zero-review matches judged non-DRS | 16 (Zimbabwe/Ireland hosts) | 365 | 2,365 |
| Zero-review share: observed vs ZT-Poisson implied | 0 / 0 | 9.2 % / 4.0 % | 16.2 % / 16.8 % |

- The ODI excess of zero-review matches suggests that some of the 57 added ODIs may not have had DRS. T20 shows no excess.
- `summary.json` `matches` counts matches with at least one batting-side decision: 331 / 622 / 1,326. `sample.matches` gives 331 / 623 / 1,331.

## 2. Perception noise s: continuous profile-likelihood set, multi-start free fit

**Changes**
- `src/drs/model.py:fit_full`:
  - The grid profile is kept.
  - The free fit now uses `fit_multi` and `free_starts`: the 3 best grid thetas, each with log s moved by −0.25, 0 and +0.25.
  - The 95 % set comes from `brentq` on the profile, 2(ℓ̂ − ℓ_p(s)) = χ²₁(0.95), bracketed by the grid. Each profile point is itself a 2-start fit.
  - New keys: `s_95`, `s_95_open`, `profile_ends` and `profile_extra`.
- `src/drs/dp.py`:
  - `s_lo` and `s_hi` now use the endpoint fits.
  - The bootstrap refits use multiple starts.
- `src/drs/report.py` reads `model.s_95`.
- `src/drs/identify.py` checks coverage exactly, lo ≤ truth ≤ hi. The old check used a 15 % grid slack.

**Results**
- The old Test ŝ = 0.499 had stopped at its start value. The new fit lowers the negative log-likelihood by 0.13 on identical data.

| | Test | ODI | T20 |
|---|---|---|---|
| ŝ | 0.499 → **0.469** | 0.809 → **0.806** | 0.463 → **0.449** |
| 95 % set | [0.5, 0.7] → **[0.434, 0.844]** | [0.5, 2.0] → **[0.372, 2.284]** | [0.5, 0.7] → **[0.396, 0.855]** |
| s_hi used in sensitivity | 0.7 → 0.84 | 2.0 → 2.28 | 0.7 → 0.86 |
| κ̂ | 0.48 → 0.478 | 0.413 → 0.405 | 0.408 → 0.400 |
| Wrong LBW out decisions | 30.3 → 30.3 % | 30.6 → 29.0 % | 29.0 → 27.8 % |
| Constant-rate model ŝ | 0.50 → 0.46 | 0.43 → 0.41 | 0.50 → 0.45 |
| Constant-rate model 95 % set (new) | [0.43, 7.42] | [0.36, 1.20] | [0.40, 4.32] |
| Known truth, R = 20: median ŝ | 0.500 → 0.496 | 0.700 → 0.777 | 0.600 → 0.644 |
| Known truth: replicates with ŝ > 1 | 5 → 4 | 6 → 8 | 7 → 3 |
| Known truth: LR coverage | 90 → 90 % | 95 → 95 % | 95 → 90 % |

- All the main-model sets are closed.
- In the known-truth study, 2–3 replicates per format have an upper end at the bound (s = 20).
- The old medians were themselves stuck on grid values.

## 3. WP engine: held-out recalibration diagnostics

**Changes**
- `src/drs/wp.py`: `brier_recal` and `calibration_recal` are now cross-fitted.
  - The held-out matches are split in two halves.
  - The isotonic map is fitted on one half and scored on the other, in both directions.
  - The old in-sample number is kept as `brier_recal_insample`.
- The production map used for the stakes is unchanged: isotonic on the whole held-out fold. So the stakes g are identical.
- I also made the engine load one format at a time, for memory.

| Brier | Test | ODI | T20 |
|---|---|---|---|
| Raw engine (held-out) | 0.128 | 0.162 | 0.163 |
| Recalibrated, old (in-sample) | 0.1265 | 0.1595 | 0.1613 |
| Recalibrated, cross-fitted (new) | **0.1365** | **0.1622** | **0.1630** |

- **Flag:** held out, the isotonic map does not improve on the raw engine. It is neutral in ODI and T20 and worse in Tests, where only 135 held-out matches are available.
- The paper should stop claiming the recalibration improves calibration, or should drop it. The existing `recal` sensitivity row bounds its effect.

## 4. DP without the extra opportunity after an overturn

**Changes**
- `src/drs/dp.py:solve_dp(extra=False)`:
  - An overturn moves on to w+1, the same as the replay on the recorded stream.
  - The value is U(w,c) = a_w E max{U(w+1,c), p g + U(w+1,c) − ℓ ΔU} + b_w U(w+1,c), so the rule is "review iff p g ≥ ℓ ΔU".
- `run_policy` policy 4 evaluates it as `opt_nx` on the same draws, in the main fit, every variant and every bootstrap replicate.
- The rule is checked in the kernel: `D_noextra` and `D_noextra_is_A0` are both True.
- New keys: `summary.per100_no_extra`, `sqrt_law_no_extra`, and `sensitivity.*.opt_nx`.

| | Test | ODI | T20 |
|---|---|---|---|
| Optimum, extra opportunity | 5.380 | 2.498 | 1.089 |
| Optimum, no extra (`opt_nx`) | 5.382 [4.844, 5.927] | 2.495 [2.171, 3.066] | 1.089 [0.946, 1.225] |
| opt − opt_nx | −0.002 [−0.007, 0.004] | 0.003 [−0.005, 0.006] | 0.000 [0.000, 0.001] |
| Lost vs realised (no extra) | 0.238 [−0.118, 0.635] | 0.034 [−0.174, 0.516] | 0.026 [−0.067, 0.102] |
| Lost vs fitted rule (no extra) | 0.032 [0.011, 0.075] | 0.021 [−0.010, 0.464] | 0.002 [−0.002, 0.010] |
| One-line share of `opt_nx` | 99.8 % | 98.6 % | 99.5 % |
| Reviews per team-match, `opt_nx` | 4.17 | 0.99 | 0.47 |
| Power of Δ₁, no extra | 1.59 | 1.84 | 1.95 |

Conclusion: the mismatch between the DP and the replay is immaterial, at most 0.003 per 100 team-matches.

## 5. Cross-fitted constant δ

**Changes**
- `src/drs/dp.py:crossfit_delta`:
  - The matches are split at the median match date.
  - δ is chosen on one half and scored on the other, in both directions; the two halves are pooled.
  - It is computed in the main fit, every variant and every bootstrap replicate.
- New keys: `per100.stake_share_crossfit` (with a CI) and `crossfit_delta` (per-half δ and shares).

| | Test | ODI | T20 |
|---|---|---|---|
| In-sample share | 99.8 % [99.4, 100.1] | 98.4 % [96.4, 100.0] | 99.5 % [99.0, 100.1] |
| **Cross-fitted share** | **99.7 % [98.7, 100.0]** | **96.8 % [94.2, 99.8]** | **98.9 % [97.2, 99.9]** |
| δ chosen: early half / late half | 0.001 / 0.0005 | 0.0075 / 0.002 | 0.002 / 0.0002 |

## Every summary.json number used by `make_numbers.py drs()` and `supplement_numbers.py` (old → new)

| key | Test | ODI | T20 |
|---|---|---|---|
| matches | 331 → 331 | 566 → 622 | 1,116 → 1,326 |
| decisions | 8,683 → 8,683 | 6,194 → 6,738 | 10,515 → 12,246 |
| bootstrap_reps | 500 → 500 | 500 → 500 | 500 → 500 |
| model.s | 0.499 → 0.469 | 0.809 → 0.806 | 0.463 → 0.449 |
| model.s_95 | [0.5, 0.7] → [0.434, 0.844] | [0.5, 2.0] → [0.372, 2.284] | [0.5, 0.7] → [0.396, 0.855] |
| model.kappa | 0.48 → 0.478 | 0.413 → 0.405 | 0.408 → 0.400 |
| model.lbw_error_rate | 0.303 → 0.303 | 0.306 → 0.290 | 0.290 → 0.278 |
| model.rho | 0.36 → 0.36 | 0.313 → 0.313 | 0.31 → 0.31 |
| per100.opt | 5.375 [4.739, 5.946] → 5.380 [4.843, 5.926] | 2.767 [2.36, 3.188] → 2.498 [2.171, 3.067] | 1.222 [1.12, 1.434] → 1.089 [0.946, 1.225] |
| per100.stake (best δ) | 5.365 (0.001) → 5.370 (0.001) | 2.735 (0.0075) → 2.459 (0.005) | 1.212 (0.002) → 1.084 (0.001) |
| per100.stake_share | 0.998 [0.994, 1.001] → 0.998 [0.994, 1.001] | 0.988 [0.964, 1.001] → 0.984 [0.964, 1.000] | 0.992 [0.989, 1.001] → 0.995 [0.990, 1.001] |
| per100.obs_rule | 5.348 → 5.350 | 2.767 → 2.474 | 1.214 → 1.088 |
| per100.observed | 5.144 [4.559, 5.753] → 5.144 [4.551, 5.745] | 2.705 [2.304, 3.097] → 2.461 [2.118, 2.848] | 1.263 [1.11, 1.42] → 1.063 [0.923, 1.213] |
| per100.conf50 | 4.045 → 4.127 | 1.743 → 1.548 | 0.949 → 0.854 |
| per100.lost_vs_rule | 0.027 [0.012, 0.068] → 0.030 [0.010, 0.074] | 0.000 [−0.014, 0.412] → 0.023 [−0.011, 0.465] | 0.008 [−0.002, 0.014] → 0.002 [−0.002, 0.010] |
| per100.lost_vs_observed | 0.232 [−0.169, 0.561] → 0.237 [−0.120, 0.634] | 0.062 [−0.188, 0.449] → 0.037 [−0.175, 0.522] | −0.042 [−0.083, 0.121] → 0.026 [−0.066, 0.102] |
| reviews/team-match: opt / stake / observed | 4.44 / 1.71 / 2.75 → 4.42 / 1.65 / 2.75 | 1.06 / 0.57 / 0.72 → 1.02 / 0.58 / 0.65 | 0.55 / 0.27 / 0.41 → 0.49 / 0.25 / 0.35 |
| thresholds move, c1, w0 → w8 (pp) | 18.6 → 18.2 | 15.4 → 14.7 | 7.8 → 7.0 |
| thresholds, c1 vs c2 at w0 (pp) | 17.0 → 16.9 | 13.4 → 12.7 | 7.4 → 6.8 |
| sqrt_law.power | 1.58 → 1.59 | 1.82 → 1.85 | 1.90 → 1.95 |
| sqrt_law R² √N / N | 0.939 / 0.989 → 0.938 / 0.988 | 0.917 / 0.982 → 0.920 / 0.983 | 0.926 / 0.986 → 0.927 / 0.985 |
| sqrt_law.N[0] | 6.65 → 6.65 | 5.36 → 5.33 | 4.65 → 4.58 |
| sensitivity.s_hi: s, opt / lost_rule / share | 0.7, 5.346 / 0.026 / 0.994 → 0.84, 5.379 / 0.059 / 0.990 | 2.0, 3.289 / 0.620 / 0.959 → 2.28, 3.329 / 0.912 / 0.925 | 0.7, 1.214 / 0.003 / 0.990 → 0.86, 1.097 / 0.017 / 0.989 |
| sensitivity.s2.0: dnll, opt / lost_rule / share | 5.08, 6.496 / 1.243 / 0.966 → 5.21, unchanged | 1.6, 3.289 / 0.620 / 0.959 → 1.57, 3.097 / 0.652 / 0.943 | 4.41, 1.629 / 0.431 / 0.944 → 4.46, 1.564 / 0.517 / 0.946 |
| sensitivity.const: opt / lost_rule / share | 5.175 / 0.042 / 0.999 → 5.179 / 0.043 / 1.000 | 2.788 / 0.028 / 0.993 → 2.559 / 0.027 / 0.996 | 1.260 / 0.004 / 0.996 → 1.078 / 0.000 / 0.998 |
| bowl success / uc | see `summary.json` (the fielding-review sample is unchanged) | | |

Power exponents of the last review's value against out decisions left, from the kernel fit (`check.json` E_*):

| | old | new |
|---|---|---|
| Test | 1.58 [1.46, 1.71] | **1.59 [1.46, 1.71]** |
| ODI | 1.82 [1.56, 2.08] | **1.85 [1.54, 2.16]** |
| T20 | 1.90 [1.61, 2.19] | **1.95 [1.64, 2.26]** |

All exclude 0.5, so the √N law still fails in every format.

`sensitivity.json`, opt / lost_vs_rule / share, old → new:

| | Test | ODI | T20 |
|---|---|---|---|
| recal | 5.707 / 0.027 / 0.997 → 5.714 / 0.032 / 0.998 | 2.560 / −0.002 / 0.990 → 2.316 / 0.026 / 0.982 | 1.160 / 0.008 / 0.991 → 1.032 / 0.002 / 0.995 |
| g×1.5 | 8.063 / 0.041 / 0.997 → 8.071 / 0.045 / 0.998 | 4.150 / 0.000 / 0.989 → 3.747 / 0.035 / 0.984 | 1.833 / 0.011 / 0.992 → 1.634 / 0.002 / 0.995 |
| g×0.67 | 3.601 / 0.018 / 0.997 → 3.605 / 0.020 / 0.998 | 1.854 / 0.000 / 0.989 → 1.674 / 0.016 / 0.984 | 0.819 / 0.005 / 0.993 → 0.730 / 0.001 / 0.995 |

**Wolfram.** `wolfram/drs_check.wl` was re-run on Mathematica 15.0.0 (Mac OS X ARM). Every check is True; the new checks are D_noextra and D_noextra_is_A0. The C_* numerical checks are unchanged because they are pinned to the old Test parameters.

## Not done / needs follow-up outside my allowed paths

- **Paper text and tables.** `paper/make_numbers.py` and `paper/supplement_numbers.py` were not touched.
  - The JSON layout is backward compatible: only keys were added.
  - Labels now wrong: "Matches with a review" should read "DRS matches" (zero-review matches included). "95 % likelihood-ratio set on the grid" should read "profile-likelihood set".
  - The paper should add the cross-fitted share, the `opt_nx` sensitivity and the sample counts.
- **`audit/drs/recount.json`** (the supplement's data table) still uses the old ≥ 1-review rule. `audit/drs/recount.py` needs the same sample rule.
- **README.md** is not updated: the new sample rule, the `plots.py [out]` argument, and the note that T20 is now B = 500.
- **Figure.** `figures/drs_fig1.png` was regenerated by mistake and then restored to HEAD. The new figure is `results/drs/drs_fig1.png`. Copy it over with `python src/drs/plots.py` when the paper is rebuilt.
- **`results/drs/report.md`** (prose) was not rewritten. Its numbers are now superseded by `summary.json` and this file.
- The production isotonic map was kept, even though its cross-fitted Brier is no better (item 3). That decision is left to the authors.
- The DP's per-team-match normalisation still counts only matches with at least one batting decision: 1 ODI and 5 T20 matches without one are left out (negligible).
- The ODI zero-review excess (item 1) means up to about 30 of the 57 added ODIs may be non-DRS. There is no field in Cricsheet that settles this.
