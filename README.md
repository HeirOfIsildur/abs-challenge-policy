# ABS challenges: wins left on the table

Code, results and manuscript sources for

> Janoušek, J. and Pešta, M. (2026). *ABS challenges: wins left on the table.* Manuscript submitted to the
> *Annals of Applied Statistics*.

In 2026 Major League Baseball began letting teams challenge ball–strike calls against an automated ball–strike (ABS)
system, and a team keeps every challenge that succeeds. The paper asks when a team should challenge and what getting it
wrong costs. The published zone rule, applied to the public 2026 Statcast pitch locations, reproduces 99.95 % of the
reviewed calls, so the correct call is known for every called pitch, challenged or not. The challenge decision is
then a binary-response model on all called pitches whose slope measures the challenger's perception noise. Combined
with a score-aware dynamic program and an evaluation of policies on the real call stream:

- the one-line rule *challenge iff p ≥ 0.008 / (g + 0.008)*, with p the calibrated probability that the call is wrong
  and g the win-probability stake, attains 95.4 % of the dynamic-programming benchmark;
- against that benchmark a team leaves **0.69 [0.58, 0.81]** wins per season on the table under the main model, and
  0.69–0.92 across the models that fit the data at least as well;
- most of the loss is in *which* pitches teams challenge, not how often.

The same method is applied to cricket's Decision Review System in the Supplementary Material.

Every number, table and figure in the paper is generated from the committed files in `results/`
(`paper/make_numbers.py`, `paper/supplement_numbers.py`, `paper/make_figures.py`); nothing is typed by hand.

## Data

The raw data are public but are **not redistributed** here (see `DATA.md`):

- MLB Statcast pitch data (Baseball Savant) and MLB Stats API live game feeds, 2026 regular season and postseason,
  2026 spring training and 2025 Triple-A;
- Cricsheet ball-by-ball data (men's Tests, ODIs and T20Is since October 2017; IPL since 2018).

The `src/fetch_*.py` and `src/drs/fetch.py` scripts re-download them into `data/`, which is git-ignored.
`results/manifest.json` records the commit, the pinned package versions and checksums of the raw inputs used for the
paper, since public feeds can be revised.

## Setup

Python 3.12+ and, for the symbolic checks, Wolfram Mathematica (`wolframscript`). LaTeX (`latexmk`, `pdftotext`) to
build the manuscript.

```bash
uv venv && uv pip install -r requirements-lock.txt   # or requirements.txt for unpinned versions
make test                                            # unit tests (tests/): geometry guardrails, simulator, causal check
```

## Run order

Run everything from the repository root. Fetch the data first:

```bash
python src/fetch_statcast.py data 2026        # Savant Statcast, all pitches
python src/fetch_challenges.py data 2026      # MLB Stats API live feeds, ABS challenge fields
python src/fetch_ext.py data spring2026 1 S 2026-02-01 2026-03-31   # out-of-sample: 2026 spring training
python src/fetch_ext.py data aaa2025 11 R 2025-03-01 2025-10-15     # out-of-sample: 2025 Triple-A
python src/postseason.py fetch                # 2026 postseason feeds and Statcast
python src/drs/fetch.py data                  # Cricsheet zips -> data/cricsheet/
```

`audit/rerun.sh` then rebuilds every result in `results/` in dependency order (several hours on 10 cores; the step
arguments are the ones used for the paper). The main steps:

```bash
python src/wp.py                              # count-level win-probability / run-expectancy engine
python src/build.py data                      # called-pitch table: original call, role, challenges left, ABS-zone distance
python src/stakes.py                          # g (win-probability swing of a flipped call) and RV (runs)
python src/truth.py && python src/perception.py   # ABS truth and the selection-corrected perception model
python src/robust_fit.py                      # all perception / truth variants -> results/perception_variants.json
python src/headline.py point 40               # all policies at the point estimate, with Monte Carlo SE
python src/headline.py boot 400 10 5          # game bootstrap with model refits -> results/headline_boot.json
python src/headline.py noref 150 2 4          # per-player model interval
python src/joint_boot.py joint 100 4 4        # joint bootstrap (engine + perception from one resample)
python src/joint_boot.py block 100 4 4        # block bootstrap by home team x month
python src/wp_boot.py 100 10 3                # engine re-estimated on game resamples
python src/causal.py 10 1000                  # re-simulation from the first divergence vs path-sum accounting
python src/headline_report.py                 # -> results/headline.{json,md}
```

Further checks (all in `audit/rerun.sh`): out-of-sample fits (`src/oos.py`), known-truth recovery (`src/identify.py`),
threshold structure (`src/refresh_thresholds.py`), the postseason (`src/postseason.py build|analyze`), the geometry
audit (`src/geometry_audit.py`, `results/geometry_audit.md`) and independent recomputations that do not import `src/`
(`audit/*.py`). The bootstrap coverage study is run separately: `python src/boot_coverage.py run 50 50 4 6`
(-> `results/coverage.json`, ~1.3 GB of memory per worker).

Symbolic and exact checks run in Mathematica, each writing a JSON file to `results/`:

```bash
for f in check proofs identification identification_det headline regret supplement; do
  wolframscript -file wolfram/$f.wl
done
wolframscript -file wolfram/drs_check.wl      # cricket DRS
```

`theory.md` states the theory (monotone thresholds, √t law, flat-optimum bound, rule comparison) that these files check.

### Cricket DRS (`src/drs/`, outputs in `results/drs/`)

```bash
python src/drs/parse.py               # ball-level state + reviews
python src/drs/wp.py                  # WP / expected-points engine per format, stake g of every out decision
python src/drs/build.py               # review opportunities, reviews left, three-way outcome
python src/drs/model.py all band      # selection-corrected perception model (umpire's call as a margin band)
python src/drs/model.py all const     # sensitivity: umpire's call at a constant rate
python src/drs/dp.py all 10 500       # DP + policy simulation + match bootstrap
python src/drs/report.py && python src/drs/plots.py
```

## Manuscript (`paper/`)

IMS `imsart` class (support files from VTeX, `paper/IMSART-LICENSE`).

```bash
python src/data_counts.py                     # sample sizes and challenges by role
make -C paper PY=python                       # numbers + tables + figures + main.pdf and supplement.pdf
make -C paper check PY=python                 # fails on a pending number, an undefined reference, an overfull box or a lint hit
make check                                    # tests + manuscript check
```

`make -C paper check` only needs the committed `results/`, not the raw data.

## Layout

| Path | Contents |
|---|---|
| `src/` | pipeline (MLB ABS); `src/drs/` cricket DRS |
| `results/` | every output the paper reads; `headline.json` is the source of the headline numbers |
| `audit/` | independent recomputations and the full rebuild script `rerun.sh` |
| `wolfram/` | Mathematica checks of the algebra and identification |
| `tests/` | unit tests |
| `paper/` | manuscript and Supplementary Material sources and PDFs |
| `figures/` | pipeline figures |

## Citation

See `CITATION.cff`, or cite the paper above.

## License

Code: MIT (`LICENSE`). The `imsart` class files in `paper/` keep their own license (`paper/IMSART-LICENSE`).
