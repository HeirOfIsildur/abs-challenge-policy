"""Perceived-confidence model, revealed by challenges.
Truth: ABS strike iff d_true > 0, d_true = d_meas - mu_e + eta, eta ~ N(0, sm_e^2) (edge-specific, fitted on the
challenged pitches). Wrongness w = -(d_true) for called strikes, +d_true for called balls; w_meas = signed.
Player signal s = w_true + eps, eps ~ N(0, sr^2); challenge iff s > kappa(state).
=> P(challenge | w_meas, state) = Phi((w_meas - kappa(state)) / sqrt(sm_e^2 + sr_role^2)).
Fitted by ML on eligible opportunities (team has >= 1 challenge left; batter on called strikes, fielders on
called balls) with |w_meas| < 8 in."""
import json, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

def fit_truth(c):
    out = {}
    for e in ("vert", "horiz"):
        s = c[c.edge == e]; d = s.dist_in.values; y = s.abs_strike.values.astype(float)
        nll = lambda th: -np.sum(y * norm.logcdf((d - th[0]) / np.exp(th[1])) + (1 - y) * norm.logcdf(-(d - th[0]) / np.exp(th[1])))
        r = minimize(nll, [0, 0], method="Nelder-Mead"); out[e] = (r.x[0], np.exp(r.x[1]))
    return out

def prep(m, tp):
    m = m.copy()
    mu = 0.0     # Raw Statcast 2026 geometry, the ABS boundary sits on the published lines (no offset)
    m["sm"] = np.where(m.edge == "vert", tp["vert"][1], tp["horiz"][1])
    dc = m.dist_in - mu
    m["w"] = np.where(m.orig_strike, -dc, dc)               # >0 means measured on the wrong side
    m["p_model"] = norm.cdf(m.w / m.sm)                       # analyst's P(call wrong | tracking)
    m["side"] = np.where(m.orig_strike, "off", "def")         # who could challenge
    m["left"] = np.where(m.orig_strike,
                         np.where(m.top, m.left_away, m.left_home),
                         np.where(m.top, m.left_home, m.left_away))
    return m

def design(e):
    lg = np.log(np.clip(e.g.values, 1e-4, None))
    X = np.column_stack([np.ones(len(e)), lg, (e.left == 1).astype(float), np.clip(e.inning.values, 1, 10) / 9,
                         (e.strikes == 2).astype(float), (e.balls == 3).astype(float)])
    return X

def fit_side(e):
    """kappa = X beta ; sigma_tot_edge = sqrt(sm_e^2 + sr^2)"""
    X = design(e); w = e.w.values; sm = e.sm.values; y = e.ch.values.astype(float)
    def nll(th):
        b, sr = th[:-1], np.exp(th[-1]); st = np.sqrt(sm ** 2 + sr ** 2)
        z = (w - X @ b) / st
        return -np.sum(y * norm.logcdf(z) + (1 - y) * norm.logcdf(-z))
    r = minimize(nll, np.r_[np.zeros(X.shape[1]), 0.0], method="BFGS", options=dict(maxiter=2000))
    return r.x[:-1], float(np.exp(r.x[-1])), r

if __name__ == "__main__":
    m = pd.read_parquet("data/called_2026_g.parquet")
    c = m[m.ch].copy(); c["abs_strike"] = np.where(c.ch_over, ~c.orig_strike, c.orig_strike)
    tp = fit_truth(c); print("truth (mu, sigma_m) by edge:", {k: tuple(round(x, 3) for x in v) for k, v in tp.items()})
    m = prep(m, tp)
    out = dict(truth=tp)
    for side in ("off", "def"):
        e = m[(m.side == side) & (m.left > 0) & (m.w.abs() < 8)]
        b, sr, r = fit_side(e)
        print(side, "n=", len(e), "challenges=", int(e.ch.sum()), "sigma_r=", round(sr, 3), "in; kappa coefs",
              dict(zip(["const", "log g", "left==1", "inning/9", "2 strikes", "3 balls"], b.round(3))), "conv", r.success)
        out[side] = dict(beta=b.tolist(), sr=sr)
        # check: sigma_r by edge separately
        for ed in ("vert", "horiz"):
            b2, sr2, _ = fit_side(e[e.edge == ed]); print("   edge", ed, "sigma_r", round(sr2, 3))
    json.dump(out, open("data/perception.json", "w"))
    m.to_parquet("data/called_2026_w.parquet")
