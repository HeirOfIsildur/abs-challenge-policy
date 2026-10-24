"""HONZA-106 (referee comment 3): why does the fitted behavioural rule, simulated, earn less than teams realised, and
does that misfit touch the headline (optimum minus realised play)?
(1) By stake quintile and side: observed vs model-expected challenges, overturns and value (sum of g over overturns),
    the expectation taken pitch by pitch under the main model at the observed challenges left.
(2) The main model against kappa_sr_region_g (criterion with hinges in log g, lik.G_KNOTS; fit by robust_fit.py):
    sigma_r, the optimum, the simulated behavioural rule and the realised value, all policies on the same R draws.
usage: python src/behaviour_gap.py [R] -> results/behaviour_gap.json"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, "src")
import lik, sim
from final_ci import spec_of
from headline import MAIN, evaluate_all, spec_variant, summarise

FLEX = "kappa_sr_region"          # HONZA-122: the main criterion is now piecewise in log g; compare with the linear one

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    m, _ = sim.load(dict(theta=f"data/th_{MAIN}.npy", spec=spec_of(MAIN)))
    m = m[lik.eligible_mask(m, "off") | lik.eligible_mask(m, "def")].copy()
    W = (m.w.values > 0).astype(float); pi = m.pi.values
    m["pc"] = norm.cdf((m.w.values - m.kappa.values) / m.sr.values)
    m["ps"] = m.pc * ((1 - pi) * W + pi * (1 - W))
    m["chal"] = m.ch.astype(bool).astype(float)
    m["ov"] = (m.ch.astype(bool) & m.ch_over.fillna(False).astype(bool)).astype(float)
    m["q"] = pd.qcut(m.g, 5, labels=False)
    m["v_obs"] = m.g * m.ov; m["v_pred"] = m.g * m.ps
    t = m.groupby(["side", "q"]).agg(g_mean=("g", "mean"), ch_obs=("chal", "sum"), ch_pred=("pc", "sum"), ov_obs=("ov", "sum"),
                                     ov_pred=("ps", "sum"), v_obs=("v_obs", "sum"), v_pred=("v_pred", "sum")).reset_index()
    for c in ("v_obs", "v_pred"):
        t[c] = t[c] / 30                                                 # wins per team-season
    pv = json.load(open("results/perception_variants.json"))
    fits = {nm: dict(nll=pv[nm]["nll"], k=pv[nm]["k"], bic=pv[nm]["bic"], off_sr=pv[nm]["params"]["off_sr"],
                     def_sr=pv[nm]["params"]["def_sr"]) for nm in (MAIN, FLEX)}
    ev = {}
    for nm in (MAIN, FLEX):
        mm, _, out, _ = evaluate_all(spec_variant(nm), R=R, seed=1)
        s = summarise(out, int(mm.gidx.max() + 1))
        ev[nm] = {k: dict(wins=s[k]["wins"], n=s[k]["n"], batter=s[k]["batter"], fielder=s[k]["fielder"])
                  for k in ("opt_score", "observed", "obs_rule", "stake_0.008")}
        print(nm, {k: round(v["wins"], 3) for k, v in ev[nm].items()}, flush=True)
    json.dump(dict(R=R, flex=FLEX, knots_log_g=list(lik.G_KNOTS), quintiles=t.to_dict(orient="records"), fits=fits,
                   evaluation=ev, expected_total_value=float(t.v_pred.sum()), observed_total_value=float(t.v_obs.sum())),
              open("results/behaviour_gap.json", "w"), indent=1, default=float)
