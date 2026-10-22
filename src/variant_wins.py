"""Wins lost (opt - observed) under every fitted perception variant, the sigma_r x 0.5 / x 1.5 sensitivity
on the new baseline, player-specific sigma_r (random effects), and the confidence-calibration test.

Calibration test.  A threshold on the signal s is observationally equivalent under any monotone belief map, so the
choice data alone cannot tell a calibrated player from an overconfident one.  What they can test is consistency
with a stake-based rule: a player who challenges iff belief >= D/(g+D) has an implied belief threshold p* with
log((1-p*)/p*) = log g - log D, i.e. slope 1 on log g once D's dependence on the state is absorbed by controls.
We compute the implied belief at the fitted threshold, p*(kappa), under (i) the calibrated posterior P(W | s)
(empirical prior, as in sim.draw) and (ii) a flat-prior, overconfident belief Phi(s / sr), and regress
logit(1 - p*) on log g + state controls.
usage: python src/variant_wins.py [R] [variant ...]  -> results/variant_wins.json"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, "src")
import sim
from final_ci import spec_of


def run(variant, R, **extra):
    spec = spec_of(variant)
    v = dict(theta=f"data/th_{variant}.npy", spec=spec, re=bool(spec.get("re")), **extra)
    m, out, _ = sim.evaluate(v, R=R)
    tot = {k: x[0].sum() / 30 for k, x in out.items()}
    side = {f"{k}_{nm}": float(out[k][0][:, :, sd].sum() / 30) for k in ("opt", "observed", "obs_rule") for sd, nm in ((1, "batter"), (0, "fielder"))}
    nch = {f"n_{k}": float(out[k][1].sum() / (2 * len(out[k][1]))) for k in ("opt", "observed", "obs_rule")}
    return dict(opt=tot["opt"], observed=tot["observed"], obs_rule=tot["obs_rule"], lost=tot["opt"] - tot["observed"],
                lost_vs_rule=tot["opt"] - tot["obs_rule"], best_stake=max(x for k, x in tot.items() if k.startswith("stake_")),
                savant=tot["savant"], **side, **nch)


def calibration_test(variant="normal", seed=3):
    import statsmodels.api as smx
    spec = spec_of(variant)
    m, _ = sim.load(dict(theta=f"data/th_{variant}.npy", spec=spec))
    m = sim.opportunities(m)
    W, s, _ = sim.draw(m, np.random.default_rng(seed), {})
    out = {}
    for side in ("off", "def"):
        k = (m.side == side).values & (m.left > 0).values
        e = m[k]; kap = e.kappa.values
        pc = np.zeros(k.sum())
        cal = sim.cal_groups(m); ce = cal[k]
        for grp in np.unique(ce):                     # The posterior's calibration cells (sim.cal_groups)
            j = ce == grp; jj = (m.side == side).values & (cal == grp)
            order = np.argsort(s[jj]); ss = s[jj][order]; cum = np.cumsum(W[jj][order].astype(float))
            # calibrated P(W | s = kappa): share of truly wrong calls among the 2,000 simulated signals around kappa
            idx = np.searchsorted(ss, kap[j]); lo = np.clip(idx - 1000, 0, len(ss) - 1); hi = np.clip(idx + 1000, 0, len(ss) - 1)
            pc[j] = (cum[hi] - cum[lo]) / np.maximum(hi - lo, 1)
        pf = norm.cdf(kap / e.sr.values)
        X = smx.add_constant(pd.DataFrame(dict(log_g=np.log(np.clip(e.g.values, 1e-4, None)), left1=(e.left == 1).astype(float),
                                               inn=np.clip(e.inning.values, 1, 10) / 9, s2=(e.strikes == 2).astype(float),
                                               b3=(e.balls == 3).astype(float))))
        res = {}
        for nm, p in (("calibrated", pc), ("flat_prior", pf)):
            p = np.clip(p, 1e-3, 1 - 1e-3)
            f = smx.OLS(np.log((1 - p) / p), X).fit()
            res[nm] = dict(slope_log_g=float(f.params["log_g"]), r2=float(f.rsquared), mean_p_threshold=float(p.mean()))
        out[side] = res
    return out


if __name__ == "__main__":
    # The manuscript uses only the calibration test, now on the main specification (it was the baseline);
    # the per-variant wins lost come from headline.py variants (score-aware DP), not from the pooled sim.evaluate here
    from headline import MAIN
    path = "results/variant_wins.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    res["calibration_test"] = calibration_test(MAIN); res["calibration_test_variant"] = MAIN
    print(res["calibration_test"], flush=True)
    json.dump(res, open(path, "w"), indent=1)
