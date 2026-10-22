# Data

No raw data are committed to this repository. All inputs are public and are re-downloaded by the fetch scripts into
`data/` (git-ignored):

| Source | Used for | Fetch script |
|---|---|---|
| Baseball Savant Statcast (MLB Advanced Media) | pitch locations and calls, 2026 regular season and postseason, 2026 spring training, 2025 Triple-A | `src/fetch_statcast.py`, `src/fetch_ext.py`, `src/postseason.py fetch` |
| MLB Stats API live game feeds | ABS challenge events, challenger role, challenges left, game state | `src/fetch_challenges.py`, `src/fetch_ext.py`, `src/postseason.py fetch` |
| Cricsheet (cricsheet.org) | ball-by-ball data and DRS reviews, men's Tests / ODIs / T20Is since Oct 2017, IPL since 2018 | `src/drs/fetch.py` |

MLB data are © MLB Advanced Media and are not redistributed here; use them under MLB's terms. Cricsheet data are
distributed by Cricsheet under the Open Data Commons Attribution License (ODC-By); we do not redistribute them either.

The committed files in `results/` are derived aggregates (model fits, simulation summaries, bootstrap intervals) from
which the manuscript is built. `results/geometry_mismatches.csv` lists the challenge rulings that the published zone rule
does not reproduce (game and pitch identifiers with coordinates), and `results/roles_observed.csv` holds
challenge counts by role. `results/manifest.json` records the build commit (in the authors' working repository) and the checksums of the raw inputs used for the paper, so a
re-fetch can be compared against them: public feeds are occasionally revised.
