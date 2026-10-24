# ABS challenges: an accounting index of wins left on the table

Code, results and manuscript sources for Janoušek & Pešta, *ABS challenges: an accounting index of wins left on the
table* (submitted to the *Annals of Applied Statistics*). The paper asks how much MLB teams leave on the table, in
expected wins per team-season, by how they used the 2026 automated ball–strike (ABS) challenge system, and what a simple
decision rule would recover.

Every number, table and figure in `paper/main.tex` and `paper/supplement.tex` is a macro or an `\input` written by
`paper/make_numbers.py` from the files in `results/`; nothing in the manuscript is typed by hand.

## What is where

| Path | Contents |
|---|---|
| `src/` | the MLB pipeline: data fetch and build, geometry, WP engine and stakes, decision models (`lik.py`, `robust_fit.py`), simulation and DP (`sim.py`, `dp_score.py`, `policies.py`), the full policy cross-fit (`policy_crossfit.py`), its bootstrap and coverage study (`coverage_pcf.py`), the primary model's recovery study (`recovery_primary.py`), the scoring-engine check (`scoring_bias.py`), the quadrature check (`quadrature_check.py`), robustness checks (`extra_checks.py`: DP monotonicity, far-miss non-challenges, near-edge balance) |
| `src/drs/` | the exploratory cricket DRS application |
| `wolfram/` | Mathematica scripts for the exact algebra (identification, the theorems, the intervention model, honest scoring) |
| `results/` | every output the manuscript reads (JSON, CSV, Markdown reports) |
| `paper/` | manuscript (`main.tex`, `supplement.tex`, IMS `imsart` class), number and figure generators |
| `tests/` | geometry and simulation guardrails (`make test`) |
| `theory.md` | notes on the retained-challenge theory |
| `rerun.sh` | the full rebuild in dependency order |

Raw data are not redistributed (`DATA.md`): MLB Statcast (Baseball Savant), the MLB Stats API and Cricsheet are public, and the
`fetch_*` scripts download them into `data/`. `results/manifest.json` records the commit, package versions and checksums
of the raw inputs, since public feeds can be revised.

## Setup

```bash
uv venv && uv pip install -r requirements-lock.txt      # Python 3.13; exact versions used for the paper
python src/fetch_statcast.py data 2026                   # Statcast, every 2026 pitch
python src/fetch_challenges.py data 2026                 # Stats API live feeds with the ABS challenge fields
python src/build.py data                                 # called-pitch table, challenges left, ABS-zone distance
make test
```

The Mathematica checks need `wolframscript` (the paper used Wolfram 15.0).

## Rebuilding the results

`./rerun.sh <stage>` runs the pipeline in dependency order; `./rerun.sh all` runs everything.

| Stage | What it does | Cost |
|---|---|---|
| `fits` | WP engine, stakes, truth and decision models, every model variant | ~1 h |
| `points` | earlier estimators (stakes-only cross-fit), stakes validation, intervention grid | ~1 h |
| `pcf` | **the headline**: full policy cross-fit of every model at least as good as the comparator by BIC, with the (ρ, ℓ) grid | ~10 CPU h |
| `chainA`–`chainF` | bootstraps of the earlier estimator, mechanism, causal re-simulation, out-of-sample and robustness checks | hours each |
| `chainG` | refit bootstraps of the full policy cross-fit (B = 400; the primary model refits 11 × 11-node likelihoods in every fold) | several hundred CPU h |
| `chainH` | coverage study of the full policy cross-fit on the 2026 game stream | several hundred CPU h |
| `report` | Mathematica checks, headline report, manifest, `make -C paper check` | minutes |

The long stages resume from row files in `data/`, which record the configuration they were produced with, and accept
`[shard/nshard]` arguments so that a cluster can run them in parallel (one row file per shard; the summaries read them
all). For the paper, `pcf`, `chainG` and `chainH` were run on a Slurm cluster with `OMP_NUM_THREADS=1` and at most 32
worker processes per node (about 4.5 GB per worker for the player-effects model).

## The estimator in one paragraph

The games are split into K = 5 folds. For each fold, the WP engine and the stakes the policies decide on, the DP's run
laws, the decision model (and the players' posterior effects), the belief maps and the DP's continuation values are
learned on the other folds and applied unchanged to the held-out games; every flip and every observed challenge on the
held-out games is valued by an engine fitted to those games alone (`src/policy_crossfit.py`). The bootstrap repeats
the whole procedure, refits included, on game resamples with folds drawn over the original games
(`policy_crossfit.py boot`), and `src/coverage_pcf.py` checks its coverage on simulated seasons that keep the real game
stream. Point estimates: `results/policy_crossfit.json`; intervals: `results/pcf_boot_<model>.json`; coverage:
`results/coverage_pcf.json`.

## Manuscript

```bash
make -C paper PY=$PWD/.venv/bin/python          # numbers, tables, figures, main.pdf and supplement.pdf
make -C paper check                             # fails on a pending number, an undefined reference, an overfull box or a lint hit
make -C paper submission                        # single-file main-flat.tex for upload
```

## Licence and citation

Code: MIT (`LICENSE`). The `imsart` class files in `paper/` are distributed under their own licence
(`paper/IMSART-LICENSE`). Data sources and their terms: `DATA.md`. How to cite: `CITATION.cff`.
