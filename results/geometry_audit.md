## Geometry audit (HONZA-88; round-ball corners HONZA-120)

### HONZA-120 update: the ball is round at the zone corners

The GPT-6-Astra pre-submission review found that `geometry.signed_distance` grew each zone edge by the ball radius
with **square corners**, so a ball diagonally off a corner counted as touching the zone. The correct distance is the
radius plus the signed distance of the ball's centre to the unexpanded rectangle (rounded corners); guarded by
`tests/test_geometry.py::test_corner_contact_is_round`. Effects (all from `results/geometry_audit.json`):

- 353 of 371,199 called pitches change label (26 of them challenged).
- Reproduction of the 10,551 challenge rulings rises from 99.65 % (37 mismatches) to **99.90 % (11 mismatches)**;
  the projected Stats API feed from 99.73 % to 99.97 % (3), Triple-A 2025 to 99.95 %, spring 2026 to 100 %.
- The 11 mismatches: **7 tracking disagreements** (Statcast and the live feed differ by inches; the feed agrees with the
  ruling every time), **3 boundary ties** within 0.005 in of an edge (1 ruled ball, 2 ruled strike: no lean), and
  1 tie in Statcast that the feed places 0.76 in outside. The "27-vs-2 lean of boundary ties" and the conclusion that
  ABS is slightly stricter than touching (below) were artefacts of the square corners and are withdrawn; the
  near-edge mismatch rate (0.15 % within 0.5 in) is now the same as beyond 1 in (0.13 %).
- Ball-radius sweep (mismatches): 127 at 1.40 in, 57 at 1.43, 28 at 1.44, **11 at 1.45**, 39 at 1.46, 67 at 1.47,
  136 at 1.50: still sharply minimised at the official radius.
- The mislabel rate π of the main model falls accordingly; every downstream number was rebuilt (HONZA-120).

The rest of this file is the HONZA-88 audit on the square-corner rule; its plane and projection findings are unchanged.

`src/geometry_audit.py` → `results/geometry_audit.json`, `results/geometry_mismatches.csv`. The guardrails are in `tests/test_geometry.py`, and the convention is the `SOURCE_PLANE_Y_FT` constant in `src/geometry.py`.

### Verdict

In 2026, Statcast's `plate_x`/`plate_z` already give the ball's position in the ABS evaluation plane, the plate midpoint at y = 17/24 ft. Before HONZA-88, `build.py` treated them as front-of-plate values and projected each pitch a second time. That lowered every pitch by 0.98 in on average (SD 0.31). The "ABS bottom 1.02 in / top 0.82 in below Statcast" offsets in the old truth model came from this double projection, not from the ABS system.

With raw coordinates, the published rule reproduces **99.65 %** of the 10,551 challenge rulings (37 mismatches). The rule uses a 17 in plate, Statcast `sz_top`/`sz_bot` and a 1.45 in ball radius. The best constant shift of either zone line is **0.00 in**. With the old projected coordinates the rule reproduced 85.0 %, and the best-fitting shifts were −1.00 in at the bottom and −0.75 in at the top.

### 1. Which plane each source reports

I anchored the 9-parameter trajectory on the reported release point, which comes from the same fit, and found the y at which it passes through the reported (`plate_x`, `plate_z`). In the table below, Δz is the mean (SD) of trajectory z minus reported `plate_z`, in inches.

| Source | Pitches | Δz at front (y = 17/12 ft) | Δz at midpoint (y = 17/24 ft) | Median y* (ft) |
|---|---|---|---|---|
| Statcast 2024 (one day) | 4,386 | +0.003 (0.035) | −0.972 (0.301) | 1.41 |
| Statcast 2025 (one day) | 8,639 | +0.003 (0.035) | −0.978 (0.307) | 1.41 |
| **Statcast 2026 regular season** | 714,115 | +0.967 (0.299) | **+0.003 (0.035)** | **0.71** |
| Statcast 2026 postseason | 9,462 | +0.943 (0.308) | +0.003 (0.035) | 0.71 |
| Stats API feed, MLB 2026 (pZ) | 10,541 challenged | — | — | 1.42 |

Statcast moved from the front of the plate to the midpoint between 2025 and 2026. The Stats API live feed (`pitchData.coordinates.pX/pZ`) stayed at the front of the plate. Its trajectory parameters and `sz` lines are identical to Statcast's: the trajectory agrees on 99.9 % of pitches and the `sz` lines on 99.96 %.

On the 10,551 challenged pitches, the feed's `pZ` is +0.95 in above Statcast (SD 0.42). After projecting the feed to the midpoint, the difference is +0.010 in, and 99 % of pitches agree within 0.010 in.

### 2. Documentation

There is explicit documentation. The Baseball Savant CSV documentation (https://baseballsavant.mlb.com/csv-docs) says, for `plate_x` and for `plate_z`:

> "Through 2025, this was front-of-plate. From 2026 on, this is middle-of-plate to align with the ABS system."

For `sz_top`/`sz_bot`:

> "From 2026 on, this is the top [bottom] of the batter's ABS-defined strike zone."

The Savant changelog of 19 Feb 2026 (https://baseballsavant.mlb.com/changelog/2026-02-19-new-strike-zone-images) contrasts "Strike zone measured at front of plate and dependent on batter's stance" (through 2025) with "Strike zone measured at midpoint of plate and based on batter's height" (2026). The HONZA-79 note that no document states the `plate_z` convention was wrong; the CSV documentation states it.

### 3. Rule reproduction (challenged pitches; the ruling is observed)

| Geometry | Share reproduced | Mismatches |
|---|---|---|
| **Statcast raw `plate_x/plate_z`, ball radius 1.45 in** | **99.65 %** | 37 |
| Statcast raw, no ball radius | 65.6 % | 3,627 |
| Statcast projected to the midpoint again (pre-HONZA-88) | 85.0 % | 1,580 |
| Feed raw (front of plate) | 85.1 % | 1,577 |
| Feed projected to the midpoint | 99.73 % | 29 |

Varying the ball radius gives 138 mismatches at 1.40 in, 76 at 1.43 in, **37 at 1.45 in** and 93 at 1.47 in.

By edge: "inside/outside" is the side nearest the batter or away from him (the `lik.region` labels). "Arm/glove" is relative to the pitcher's throwing hand.

| Edge | n | Raw | Projected (old) | Raw mismatches |
|---|---|---|---|---|
| Bottom | 3,620 | 99.59 % | 74.8 % | 15 |
| Top | 1,764 | 99.83 % | 85.0 % | 3 |
| Inside (lik "arm") | 1,845 | 99.67 % | 92.4 % | 6 |
| Outside (lik "glove") | 3,322 | 99.61 % | 92.1 % | 13 |
| Pitcher's arm side | 2,710 | 99.63 % | 90.6 % | 10 |
| Pitcher's glove side | 2,457 | 99.63 % | 94.0 % | 9 |

The old projection hurt the bottom edge most (74.8 %). That edge is where the 1 in vertical drop matters.

By batter side: L 99.60 % (old 86.0 %), R 99.70 % (old 84.1 %).

By month, raw: Mar 100 %, Apr 99.82 %, May 99.47 %, Jun 99.59 %, Jul 99.76 %, Aug 99.64 %, Sep 99.58 %. The old projection gave 84–87 % in every month. There is no drift over the season.

By challenger: batter 99.64 %, catcher 99.64 %, pitcher 100 %.

The external feeds use the front plane in every sample, projected to the midpoint. On raw front-of-plate coordinates they reproduce 84–86 % of rulings; projected, they reproduce:

| Sample | Challenges | Projected to midpoint |
|---|---|---|
| MLB 2026 | 10,557 | 99.73 % |
| Triple-A 2025 | 9,341 | 99.53 % |
| Spring 2026 | 1,907 | 99.74 % |
| Postseason 2026 | 156 | 100 % |

Postseason Statcast raw reproduces 100 % as well. The Triple-A and spring OOS tables (`build_ext.py`) therefore keep the projection, which is correct for feed coordinates.

### 4. Zone lines vs height

Statcast 2026 `sz_top`/`sz_bot` are exactly 53.5 % / 27 % of one height on every pitch: sz_top/0.535 and sz_bot/0.27 agree to 10⁻⁸ in on 99.9 % of pitches. Against the whole-inch listed height (Stats API `people`, 662 batters), the lines are off by:

| Line | Mean | SD | Max |
|---|---|---|---|
| Top | +0.001 in | 0.155 in | 0.27 in |
| Bottom | +0.001 in | 0.078 in | — |

94.4 % of batters are within 0.25 in on both lines. The maximum, 0.2675 in, equals 0.535 × 0.5 in: the certified height is measured more finely than the listed height, which is rounded to the inch.

Within a batter the zone is constant: the median within-batter SD is 0. 62 batters have more than one value over the season (largest range 0.36 in), consistent with re-certification. None of the 37 mismatches comes from a zone that differs from the batter's season median.

### 5. The 37 mismatches, one by one (`results/geometry_mismatches.csv`)

For every mismatch I re-read the live feed: pitch description, play description, review flag, challenger, and start and end time of the pitch.

| Class | n | What it is |
|---|---|---|
| Boundary tie | 29 | Statcast and feed agree to < 0.02 in, and the pitch is 0.002–0.33 in from the boundary. 27 sit just *inside* by our rule and were ruled balls; 2 sit just outside and were ruled strikes. Within 0.5 in of the boundary, 1.1 % of the 2,723 challenged pitches mismatch; beyond 1 in, 0.13 %. |
| Tracking disagreement | 7 | Statcast's location differs from the live feed's by 1.6–11.2 in. The live-feed location agrees with the ruling in **all 7**; Statcast is right while the feed is wrong in none of the 11 challenged pitches where the two disagree by > 0.5 in. Statcast's record was evidently revised after the live ABS call. 6 are far from the boundary in Statcast (4.7–9.5 in), so the ruling looks absurd in Statcast and natural in the feed. |
| Other (boundary tie in Statcast, feed disagrees) | 1 | Statcast −0.0006 in, feed −0.76 in, ruled strike. |
| Mislabel (recorded outcome contradicts both sources) | **0** | Every recorded call text and review flag is consistent with the play-by-play description. |

Nothing clusters in time: mismatches are spread over April–September, and there are none in March. They are split 17 batter and 20 catcher challenges. The mismatch share does not differ by challenger, edge or batter side.

The 27-vs-2 lean of the boundary ties suggests ABS is very slightly stricter than "touching by a 1.45 in ball". That lean is confined to the last 0.33 in. A larger effective radius or zone shift does not fit (see the radius sweep above). It is an order of magnitude below the old 1 in artefact and is absorbed by the mislabel rate π.

### 6. Consequences for the model

- The truth model becomes deterministic: W = 1{w > 0} with a ruling mislabel rate π. The fitted π is 0.33 % (`lik.spec_of`, `det=True`). The 7 tracking disagreements and 29 ties are what π absorbs.
- The old smooth truth model, refitted on the corrected geometry (`sm_region`), collapses. σ_m is 0.000–0.025 in, |μ| ≤ 0.010 in, and π = 0.26 %. It gains 22.9 log-likelihood points over the deterministic model with 8 more parameters, all on the boundary σ_m → 0, where χ² does not apply. That gain comes from the near-boundary ties, and BIC favours the deterministic model by 53.
  - Without π (`sm_edge`), σ_m inflates to 0.32–0.42 in to accommodate the 7 far-off cases, and its fit is 1,277 log-likelihood points worse.
- The HONZA-73 "2,092 log-likelihood points for region offsets" was the model undoing the double projection.
