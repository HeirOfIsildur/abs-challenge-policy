#!/bin/bash
# Full rebuild after the five-referee review: mis-joins and
# position-player pitches out, signed stakes, half-start score, random opportunity count + exact extra innings in the
# DP, conditional calibration, observable opportunity set, main spec kappa_sr_region_g, first-divergence causal check,
# joint and block bootstraps. Chain A (headline) and chain B (checks) run concurrently after the fits.
set -e; cd "$(dirname "$0")/.."; PY=.venv/bin/python; L=data/rerun_logs
mkdir -p $L
step(){ echo "=== $(date +%T) $*"; "$@"; }
step $PY src/wp.py; step $PY src/build.py data; step $PY src/stakes.py; step $PY src/truth.py; step $PY src/perception.py
rm -f data/extra_runs.npy
step $PY src/robust_fit.py
rm -f results/headline_boot.json results/headline_noref.json data/thresholds_score_boot.json data/wp_boot_rows.json \
      data/joint_boot_rows.json data/block_boot_rows.json
chainB(){
  step $PY src/causal.py 10 1000
  step $PY src/refresh_thresholds.py 10 10 4
  for t in spring2026 aaa2025 mlb2026feed; do step $PY src/build_ext.py $t; done
  step $PY src/label_agreement.py
  step $PY src/postseason.py build
  step $PY src/geometry_audit.py
  step $PY src/feed_counters.py data
  step $PY src/data_counts.py
  step $PY audit/independent.py 5
  step $PY src/identify.py 20
  step $PY src/oos.py 4
  step $PY src/feed_check.py 10
  step $PY src/variant_wins.py
  step $PY src/isotonic_check.py
  step $PY src/behaviour_gap.py 10
  step $PY src/headline.py variants 20
  step $PY src/headline.py lapse 20
  step $PY src/opt_slope.py 4
  step $PY src/flat_bound.py 3
  step $PY src/dp_checks.py 40
  step $PY src/sr_breakeven.py 10
  step $PY src/postseason.py analyze 20 2000
  step $PY audit/crossfit_calibration.py 6
  step $PY src/mechanism.py 6
  step $PY src/causal.py 4 300 sim
}
chainB > $L/chainB.log 2>&1 &
BPID=$!
step $PY src/headline.py point 40
step $PY src/headline.py boot 400 10 5
step $PY src/headline.py noref 150 2 4
step $PY src/joint_boot.py joint 100 4 4
step $PY src/joint_boot.py block 100 4 4
step $PY src/wp_boot.py 100 10 3
wait $BPID
step $PY src/causal_gap.py 2000 400 cached; step $PY src/causal_gap.py md
for f in identification_det identification supplement check headline regret proofs; do step wolframscript -file wolfram/$f.wl; done
step $PY src/headline_report.py; step $PY src/wp_boot.py 100 10 2
step $PY src/joint_boot.py joint 100 4 1; step $PY src/joint_boot.py block 100 4 1   # re-summarise against the new headline boot
step $PY src/headline.py point 40      # again after the random-effects posterior files were separated per variant
step $PY src/sample_flow.py; step $PY src/manifest.py
echo "=== $(date +%T) REBUILD DONE"
# results/re_quadrature.json: re refitted with gh = 7 (lik.fit(lik.prepare(m, players=True), dict(spec_of("re"), gh=7))),
# compared with the gh = 5 fit of robust_fit.py
