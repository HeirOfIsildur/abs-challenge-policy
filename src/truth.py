"""Missed-call model: P(ABS strike | measured signed distance to the ABS zone) fitted on the 10,557 challenged
pitches (where the ABS ruling is observed). Probit Phi((d - mu)/sigma_m), optionally by edge."""
import sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

m = pd.read_parquet("data/called_2026.parquet")
c = m[m.ch & m.dist_in.notna()].copy()
c["abs_strike"] = np.where(c.ch_over, ~c.orig_strike, c.orig_strike)

def fit(d, y):
    nll = lambda th: -np.sum(y * norm.logcdf((d - th[0]) / np.exp(th[1])) + (1 - y) * norm.logcdf(-(d - th[0]) / np.exp(th[1])))
    r = minimize(nll, [0, 0], method="Nelder-Mead")
    return r.x[0], np.exp(r.x[1])

mu, sg = fit(c.dist_in.values, c.abs_strike.values.astype(float))
print(f"all: mu={mu:.3f} in, sigma_m={sg:.3f} in, n={len(c)}")
for e in ["vert", "horiz"]:
    s = c[c.edge == e]; print(e, [round(v, 3) for v in fit(s.dist_in.values, s.abs_strike.values.astype(float))], len(s))
# front-of-plate distance (no midpoint correction) for comparison
# calibration table
c["p_strike"] = norm.cdf((c.dist_in - mu) / sg)
c["bin"] = pd.cut(c.dist_in, [-20, -3, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 3, 20])
print(c.groupby("bin", observed=True).agg(n=("abs_strike", "size"), obs=("abs_strike", "mean"), pred=("p_strike", "mean")).round(3))
# overturn prediction by role
c["p_wrong"] = np.where(c.orig_strike, 1 - c.p_strike, c.p_strike)
print(c.groupby("role").agg(n=("ch_over", "size"), overturn=("ch_over", "mean"), pred=("p_wrong", "mean")).round(3))
from sklearn.metrics import roc_auc_score, brier_score_loss
print("AUC overturn:", round(roc_auc_score(c.ch_over, c.p_wrong), 4), "Brier:", round(brier_score_loss(c.ch_over, c.p_wrong), 4))
pd.Series(dict(mu=mu, sigma_m=sg)).to_json("data/truth_params.json")
# miss rate of umpires overall (all called pitches)
m["p_strike"] = norm.cdf((m.dist_in - mu) / sg)
m["p_wrong"] = np.where(m.orig_strike, 1 - m.p_strike, m.p_strike)
print("expected wrong calls per game:", round(m.p_wrong.sum() / m.game_pk.nunique(), 2),
      "| of called strikes:", round(m[m.orig_strike].p_wrong.mean(), 4), "| of called balls:", round(m[~m.orig_strike].p_wrong.mean(), 4))
