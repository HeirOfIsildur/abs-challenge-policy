#!/bin/bash
# The full MLB rebuild behind the paper's numbers, in dependency order, from the fetched raw data (src/fetch_*.py) to
# the compiled paper. The cricket DRS pipeline is separate (src/drs/, see README). The long bootstrap and coverage stages
# (chainA-chainH, hours each; run on a cluster for the paper -- the primary model's bootstrap and coverage study are
# several hundred CPU hours) resume from their row files in data/, which record the configuration they were produced
# with; a sharded run ([shard/nshard] arguments of src/policy_crossfit.py and src/coverage_pcf.py) writes one row file
# per shard, and the summaries read them all. src/geometry_audit.py also needs data/called_2026_oldgeom.parquet, an earlier
# build kept for its historical plane and projection comparisons; the ruling agreement reported in the paper is
# src/geometry_current.py on the current frame.
# usage:./rerun.sh fits|points|pcf|chainA..chainH|report|all   (long stages print to data/rerun/<stage>.log)
set -e; cd "$(dirname "$0")"; PY=.venv/bin/python; L=data/rerun; mkdir -p $L
step(){ echo "=== $(date +%T) $*"; "$@"; }
fits(){                                   # ~30 min (the 11 x 11 player-effects fits dominate)
  step $PY src/wp.py; step $PY src/build.py data; step $PY src/stakes.py; step $PY src/truth.py; step $PY src/perception.py; step $PY src/joint.py
  rm -f data/extra_runs.npy data/half_runs.npy
  step $PY src/robust_fit.py
  step $PY src/robust_fit.py main_state re_lapse_state_gh11
  step $PY src/join_audit.py; step $PY src/re_calibration.py
}
points(){                                 # cross-fitted point estimates, full-sample point, stakes validation, grid
  rm -f results/crossfit.json
  step $PY src/crossfit.py point 8 10 kappa_sr_region_g
  step $PY src/crossfit.py point 4 10 kappa_sr_region_g 2
  step $PY src/crossfit.py point 4 10 re_lapse_gh11
  step $PY src/crossfit.py decomp 10
  step $PY src/headline.py point 40
  step $PY src/stakes_validation.py 10
  step $PY src/intervention.py 6
  step $PY src/mechanism_boot.py point 20 4
}
pcf(){                                    # the headline: full policy cross-fit (src/policy_crossfit.py), point estimates
  rm -f results/policy_crossfit.json
  step $PY src/policy_crossfit.py point re_lapse_gh11 8 16 8 grid
  step $PY src/policy_crossfit.py point kappa_sr_region_g 8 16 8 grid
  for v in main_state main_lapse main_time main_time_lapse re_gh11 re_lapse_state_gh11; do
    step $PY src/policy_crossfit.py point $v $([ $v = main_state ] && echo 8 || echo 4) 16 4
  done
  # review checks: competitor and re-tuned rules, belief cells by exact count, training-only noise quintiles, K
  step $PY src/policy_crossfit.py point re_lapse_gh11 8 16 8 ext; step $PY src/policy_crossfit.py point kappa_sr_region_g 8 16 8 ext
  step $PY src/policy_crossfit.py point re_lapse_gh11 4 16 4 calfine; step $PY src/policy_crossfit.py point kappa_sr_region_g 4 16 4 calfine
  step $PY src/policy_crossfit.py point re_lapse_gh11 4 16 4 trainq5
  PCF_NAUX=8 step $PY src/policy_crossfit.py point re_lapse_gh11 4 16 4; PCF_NAUX=8 step $PY src/policy_crossfit.py point re_lapse_gh11 4 16 4 calfine
  PCF_K=2 step $PY src/policy_crossfit.py point kappa_sr_region_g 4 16 4; PCF_K=10 step $PY src/policy_crossfit.py point kappa_sr_region_g 4 16 4
  step $PY src/policy_crossfit.py collect -
  step $PY src/scoring_bias.py 3; step $PY src/quadrature_check.py
  for m in dp_mono lapse edge tail; do step $PY src/extra_checks.py $m; done
  for i in $(seq 0 11); do step $PY src/recovery_primary.py run $i 1; done; step $PY src/recovery_primary.py summary
}
chainG(){                                 # refit bootstraps of the full policy cross-fit (primary: ~1.5 CPU h per replicate)
  step $PY src/policy_crossfit.py boot kappa_sr_region_g 400 8 32; step $PY src/policy_crossfit.py bootsum kappa_sr_region_g
  step $PY src/policy_crossfit.py boot re_lapse_gh11 400 8 32; step $PY src/policy_crossfit.py bootsum re_lapse_gh11
}
chainH(){                                 # coverage of the full policy cross-fit on the 2026 stream (src/coverage_pcf.py)
  for s in kappa_sr_region_g main_lapse re_lapse_gh11; do
    step $PY src/coverage_pcf.py truth $s 16 20
    for i in $(seq 0 9); do step $PY src/coverage_pcf.py truthK $s kappa_sr_region_g 1 8 $i; done
  done
  for i in $(seq 0 15); do step $PY src/coverage_pcf.py truthK re_lapse_gh11 re_lapse_gh11 1 8 $i; done
  step $PY src/coverage_pcf.py truth re_gh11 16 20
  for i in $(seq 0 11); do step $PY src/coverage_pcf.py truthK re_gh11 re_lapse_gh11 1 8 $i; done
  step $PY src/coverage_pcf.py run kappa_sr_region_g kappa_sr_region_g 50 50 4 32
  step $PY src/coverage_pcf.py run main_lapse kappa_sr_region_g 30 50 4 32
  step $PY src/coverage_pcf.py run re_lapse_gh11 kappa_sr_region_g 30 50 4 32
  step $PY src/coverage_pcf.py run re_lapse_gh11 re_lapse_gh11 30 30 8 32
  step $PY src/coverage_pcf.py run re_gh11 re_lapse_gh11 12 20 8 32
  step $PY src/coverage_pcf.py summary
}
chainA(){ step $PY src/joint_boot.py joint 400 10 4; step $PY src/joint_boot.py block 100 4 4; }
chainB(){ step $PY src/joint_boot.py joint 400 2 4 re_lapse_gh11; }
chainC(){ step $PY src/coverage_stream.py run kappa_sr_region_g:50:20,main_lapse:30:10,re_lapse_gh11:30:10 50 20 2 4; }
chainD(){ step $PY src/mechanism_boot.py boot 60 4 2 3; }
chainE(){
  step $PY src/causal.py 10 1000
  step $PY src/causal.py 4 300 sim
  step $PY src/causal_gap.py 2000 400 cached; step $PY src/causal_gap.py md
}
chainF(){
  for t in spring2026 aaa2025 mlb2026feed; do step $PY src/build_ext.py $t; done
  step $PY src/label_agreement.py
  # the historical plane / projection comparison needs an earlier build (data/called_2026_oldgeom.parquet); skipped without it
  [ -f data/called_2026_oldgeom.parquet ] && step $PY src/geometry_audit.py; step $PY src/geometry_current.py; step $PY src/feed_counters.py data
  step $PY src/data_counts.py; step $PY src/sample_flow.py
  step $PY src/postseason.py build; step $PY src/postseason.py analyze 20 2000
  step $PY src/identify.py 20; step $PY src/feed_check.py 10; step $PY src/variant_wins.py; step $PY src/isotonic_check.py
  step $PY src/oos.py 4
  step $PY src/oos.py frozen 20; step $PY src/oos.py frozen 20 engine; step $PY src/oos.py frozen 20 world_test
  step $PY src/headline.py variants 20; step $PY src/headline.py lapse 20
  step $PY src/behaviour_gap.py 10; step $PY src/opt_slope.py 4; step $PY src/flat_bound.py 3; step $PY src/dp_checks.py 40
  step $PY src/sr_breakeven.py 10; step $PY src/refresh_thresholds.py 10 10 4
  step $PY src/robustness_checks.py months; step $PY src/robustness_checks.py moot; step $PY src/robustness_checks.py pclip 10
  step $PY src/robustness_checks.py wp_shrink 10; step $PY src/robustness_checks.py lapse_state
  step $PY src/crossfit_calibration.py 6; step $PY src/mechanism.py 6
  step $PY src/wp_calibration.py; step $PY src/independent.py
}
report(){
  step wolframscript -file wolfram/intervention.wl; step wolframscript -file wolfram/honest_scoring.wl
  step $PY src/headline_report.py; step $PY src/manifest.py
  step make -C paper PY=$PWD/$PY check
}
case "$1" in
  fits|points|pcf|report) "$1" ;;
  chainA|chainB|chainC|chainD|chainE|chainF|chainG|chainH) "$1" > $L/$1.log 2>&1 ;;
  all) fits; points; pcf; chainA; chainB; chainC; chainD; chainE; chainF; chainG; chainH; report ;;
  *) echo "usage: $0 fits|points|pcf|chainA..chainH|report|all"; exit 1 ;;
esac
echo "=== $(date +%T) $1 DONE"
