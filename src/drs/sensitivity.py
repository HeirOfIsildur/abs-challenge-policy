"""Audit sensitivities of the DRS "no material loss" finding (item 13).
  recal   stakes from a recalibrated engine: logit V' = a + b logit V fitted on realised points in the analysis era
          (one ball per over), g' = V'(V0) - V'(V0 - g); b < 1 means the engine is over-confident.
  g_x1.5  / g_x0.67: every stake scaled (a level error in the engine's wicket value).
  const   umpire's call at a constant rate (model_<fmt>_const.json) -- already in dp_<fmt>.json, repeated for one table.
  iters   DP fixed point solved with 400 instead of 60 iterations (convergence of U(w, c)).
Value in match points per 100 team-matches; lost = opt - observed and opt - obs_rule on the same draws.
usage: python src/drs/sensitivity.py [R] [fmt...]  ->  results/drs/sensitivity.json"""
import json, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
sys.path.insert(0, "src/drs")
import dp
from dp import evaluate, summarise   # summarise skips evaluate's private "_" keys

R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
FMTS = sys.argv[2:] or ["Test", "ODI", "T20"]

def recal_map(fmt):
    b = pd.read_parquet("data/drs_balls_g.parquet", columns=["fmt", "dl", "pts", "V0", "balls"])
    b = b[(b.fmt == fmt) & ~b.dl & b.pts.notna() & (b.balls % 6 == 0)]
    x = logit(b.V0.clip(1e-4, 1 - 1e-4).values); y = b.pts.values
    f = lambda t: -np.mean(y * np.log(expit(t[0] + t[1] * x)) + (1 - y) * np.log(1 - expit(t[0] + t[1] * x)))
    a, k = minimize(f, [0.0, 1.0]).x
    return float(a), float(k)

def row(d, ev, P, seed=1):
    out, U, a, b = evaluate(d, ev, P, R, seed); per, rev, best = summarise(out, d.match.nunique()); H = 100
    return dict(opt=per["opt"] * H, stake=per[best] * H, obs_rule=per["obs_rule"] * H, observed=per["observed"] * H,
                lost_vs_observed=(per["opt"] - per["observed"]) * H, lost_vs_rule=(per["opt"] - per["obs_rule"]) * H,
                stake_share=per[best] / per["opt"], reviews_opt=rev["opt"], reviews_obs=rev["observed"])

if __name__ == "__main__":
    o = pd.read_parquet("data/drs_opps.parquet"); o = o[o.side == "bat"]
    ev_all = pd.read_parquet("data/drs_innings.parquet"); res = {}
    for fmt in FMTS:
        d = o[o.fmt == fmt].sort_values("oid").reset_index(drop=True); ev = ev_all[ev_all.fmt == fmt]
        M = json.load(open(f"results/drs/model_{fmt}.json")); C = json.load(open(f"results/drs/model_{fmt}_const.json"))
        r = {"main": row(d, ev, M)}
        a, k = recal_map(fmt); V = lambda v: expit(a + k * logit(np.clip(v, 1e-4, 1 - 1e-4)))
        r["recal"] = dict(row(d.assign(g=np.maximum(V(d.V0) - V(d.V0 - d.g), 0)), ev, M), a=a, b=k)
        for m in (1.5, 0.67): r[f"g_x{m}"] = row(d.assign(g=d.g * m), ev, M)
        r["const"] = row(d, ev, C)
        it0 = dp.solve_dp.__defaults__; dp.solve_dp.__defaults__ = (400, True); r["iters400"] = row(d, ev, M); dp.solve_dp.__defaults__ = it0
        res[fmt] = r
        for k_, v in r.items(): print(fmt, k_, {x: round(y, 3) for x, y in v.items()}, flush=True)
    json.dump(res, open("results/drs/sensitivity.json", "w"), indent=1)
