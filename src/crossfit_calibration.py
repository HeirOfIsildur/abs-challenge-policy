"""is the player's calibrated posterior an in-sample fit?  sim.draw fits the isotonic map
P(W | s bin, side, region) on the same simulated truth W that then scores the policies.  Here the map is fitted on an
independent draw (A) and applied to the signals of draw B; DPs are re-solved on the cross-fitted posterior.
usage: python src/crossfit_calibration.py [R] -> results/crossfit_calibration.json"""
import json, sys
import numpy as np
sys.path.insert(0, "src")
import sim, policies as P, dp_score
from headline import spec_variant, MAIN

R = int(sys.argv[1]) if len(sys.argv) > 1 else 3
v = spec_variant(MAIN)
m, _ = sim.load(v); m = dp_score.add_dstart(m); m = sim.opportunities(m)
m["new_extra"] = (m.inning >= 10) & m.top & (m.groupby("game_pk").inning.diff().fillna(0) > 0)
kin = sim.kin_index(m).astype(np.int64); ng = int(m.gidx.max() + 1); isoff = (m.side == "off").values
base = (m.gidx.values, m.is_home_team.values, m.new_extra.values, m.slot.values.astype(np.int64), kin,
        m.inning.values.astype(np.int64), m.balls.values.astype(np.int64), m.strikes.values.astype(np.int64), m.dbin.values.astype(np.int64))


def maps(W, s):
    out = {}
    for side in ("off", "def"):
        for e in np.unique(m.pgrp.values):
            k = ((m.side == side) & (m.pgrp == e)).values
            b = np.clip(np.floor((s[k] + 15) / 0.1).astype(int), 0, 300)
            f = sim.monotone_calibration(np.bincount(b, weights=W[k], minlength=301), np.bincount(b, minlength=301))
            ok = np.flatnonzero(~np.isnan(f))            # bins unseen in draw A: carry the nearest fitted value
            out[(side, e)] = np.interp(np.arange(301), ok, f[ok])
    return out


def apply(mp, s):
    pt = np.zeros(len(m))
    for (side, e), f in mp.items():
        k = ((m.side == side) & (m.pgrp == e)).values
        pt[k] = f[np.clip(np.floor((s[k] + 15) / 0.1).astype(int), 0, 300)]
    return pt


def value(W, s, pt):
    D4h, _ = dp_score.solve_dp_score(m, pt, True); D4a, _ = dp_score.solve_dp_score(m, pt, False)
    z = np.zeros((19, 80, 3)); zc = np.zeros((19, 3))
    o = {}
    for name, (pc, dc) in dict(opt_score=(7, 0.0), stake_0008=(2, 0.008), savant=(1, 0.0)).items():
        val, n, *_ = P.run_path(*base, W, pt, s, m.g.values, m.RV.values, sim.kappa_by_count(m), isoff, pc, z, z, zc, zc,
                                np.zeros((4, 3, 2, 2)), 0.0, 1.0, dc, D4h, D4a, ng)
        o[name] = float(val.sum() / 30)
    return o


rng = np.random.default_rng(11); rows = []
for r in range(R):
    Wa, sa, _ = sim.draw(m, rng, v)
    Wb, sb, ptb_in = sim.draw(m, rng, v)
    ptb_x = apply(maps(Wa, sa), sb)
    a, b = value(Wb, sb, ptb_in), value(Wb, sb, ptb_x)
    rows.append(dict(in_sample=a, cross_fit=b, max_abs_pt_diff=float(np.abs(ptb_in - ptb_x).max()),
                     mean_abs_pt_diff=float(np.abs(ptb_in - ptb_x).mean())))
    print(r, rows[-1], flush=True)
summ = {k: {p: float(np.mean([x[k][p] for x in rows])) for p in rows[0][k]} for k in ("in_sample", "cross_fit")}
summ["diff_in_minus_cross"] = {p: summ["in_sample"][p] - summ["cross_fit"][p] for p in summ["in_sample"]}
json.dump(dict(R=R, rows=rows, summary=summ), open("results/crossfit_calibration.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
