"""Joint ML of the truth (ABS) model and the perception/challenge model, correcting the selection of challenged
pitches on the player's signal.  w_true = w_meas + eta (eta ~ N(0, sm_e^2)); W = 1{w_true > 0};
s = w_true + eps (eps ~ N(0, sr_side^2)); challenge iff s > kappa = X beta_side (eligible opportunities).
Challenged: P(ch=1, W=y | w) = int phi(eta) 1{(w+eta>0)=y} Phi((w+eta-kappa)/sr) d eta  (grid quadrature).
Not challenged: Phi(-(w-kappa)/sqrt(sm^2+sr^2))."""
import json, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from perception import design

Z = np.linspace(-6, 6, 241); PHI = norm.pdf(Z) * (Z[1] - Z[0])

def unpack(th):
    return dict(mu=dict(vert=th[0], horiz=th[1]), sm=dict(vert=np.exp(th[2]), horiz=np.exp(th[3])),
                off=dict(beta=th[4:10], sr=np.exp(th[10])), deff=dict(beta=th[11:17], sr=np.exp(th[17])))

def side_terms(e, P, key):
    isv = (e.edge.values == "vert")
    mu = np.where(isv, P["mu"]["vert"], P["mu"]["horiz"]); sm = np.where(isv, P["sm"]["vert"], P["sm"]["horiz"])
    dc = e.dist_in.values - mu
    w = np.where(e.orig_strike.values, -dc, dc)
    kap = e.X @ P[key]["beta"]; sr = P[key]["sr"]
    ch = e.ch.values
    ll = np.empty(len(e))
    nc = ~ch
    ll[nc] = norm.logcdf(-(w[nc] - kap[nc]) / np.sqrt(sm[nc] ** 2 + sr ** 2))
    wc, kc, sc = w[ch], kap[ch], sm[ch]
    eta = sc[:, None] * Z[None, :]
    wt = wc[:, None] + eta
    pr = norm.cdf((wt - kc[:, None]) / sr) * PHI[None, :]
    y = e.wrong.values[ch]
    p1 = (pr * (wt > 0)).sum(1); p0 = (pr * (wt <= 0)).sum(1)
    ll[ch] = np.log(np.clip(np.where(y, p1, p0), 1e-300, None))
    return ll.sum()

def fit(off, de, th0):
    nll = lambda th: -(side_terms(off, unpack(th), "off") + side_terms(de, unpack(th), "deff"))
    r = minimize(nll, th0, method="L-BFGS-B", options=dict(maxiter=500))
    return unpack(r.x), r

def eligible(m):
    e = m[(m.left > 0) & (m.w.abs() < 8)].copy()
    e["wrong"] = e.ch_over.values  # observed only for challenged pitches
    e.X = None
    return e

def prepare(m):
    off = m[(m.side == "off") & (m.left > 0) & (m.w.abs() < 8)].copy()
    de = m[(m.side == "def") & (m.left > 0) & (m.w.abs() < 8)].copy()
    for e in (off, de):
        e["wrong"] = e.ch_over.values
        object.__setattr__(e, "X", design(e))
    return off, de

if __name__ == "__main__":
    m = pd.read_parquet("data/called_2026_w.parquet")
    pj = json.load(open("data/perception.json"))
    off, de = prepare(m)
    th0 = np.r_[pj["truth"]["vert"][0], pj["truth"]["horiz"][0], np.log(pj["truth"]["vert"][1]), np.log(pj["truth"]["horiz"][1]),
                pj["off"]["beta"], np.log(pj["off"]["sr"]), pj["def"]["beta"], np.log(pj["def"]["sr"])]
    P, r = fit(off, de, th0)
    print("converged:", r.success, r.message, "nll", round(r.fun, 2))
    print("truth mu:", {k: round(v, 3) for k, v in P["mu"].items()}, "sigma_m:", {k: round(v, 3) for k, v in P["sm"].items()})
    for k in ("off", "deff"):
        print(k, "sigma_r", round(P[k]["sr"], 3), "beta", np.round(P[k]["beta"], 3))
    json.dump(dict(theta=r.x.tolist()), open("data/joint.json", "w"))
