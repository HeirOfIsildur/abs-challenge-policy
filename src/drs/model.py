"""Selection model for batting-side reviews (no ball tracking: identification rests on outcome rates).

Same Gaussian structure as the MLB perception model.  An out decision of kind k (lbw / caught) has a latent
margin X ~ N(mu_k, 1); the decision is overturned on review iff X > 0 (that is the "truth").  The batter sees
S = X + eps, eps ~ N(0, s^2) -- s is the perception noise in units of the spread of decision margins -- and holds
the Bayes posterior p = P(X > 0 | S) = Phi(m(S)/sqrt(v)), v = s^2/(1+s^2), m(S) = (mu_k s^2 + S)/(1+s^2).
p is increasing in S, so "review iff p >= tau" is "review iff S >= S*(tau)", S* = (1+s^2) sqrt(v) Phi^-1(tau) - mu_k s^2.
The team reviews iff p >= tau_c, a revealed threshold per cell c = (reviews left, wickets down 0-2 / 3-5 / 6-9).
Umpire's call (lbw only): mode "band" (main) -- a failed review is kept iff the margin is marginal, -kappa < X < 0,
so the umpire's-call share carries information on how close reviewed decisions are; mode "const" -- kept w.p. rho
(closed form), independent of X.  A failed caught review is always lost.
Likelihood per opportunity (kind unknown for overturned decisions -> mixture over kinds with share pi_lbw), with
J(t) = P(X > t, S >= S*) a bivariate normal orthant (corr 1/sqrt(1+s^2)) and q = P(S >= S*):
  none    pi_k (1 - q)
  success sum_k pi_k J(0)
  uc      pi_lbw (J(-kappa) - J(0))        [const: rho (q - J(0))]
  lost    pi_k (q - J(-kappa)), caught: q - J(0)   [const: (1 - rho)(q - J(0))]
Identified directly: review share, success rate and umpire's-call share in every cell.  As tau moves across cells
these trace one curve; its shape pins s (see the profile likelihood written to the json).  The free fit uses several
starts and the 95 % set for s has continuous profile-likelihood endpoints (s_95, root-finding).
usage: python src/drs/model.py [fmt|all] [band|const]  ->  results/drs/model_<fmt>[_const].json"""
import json, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, owens_t
from scipy.stats import norm

WB = [0, 3, 6, 10]
OUT = {"none": 0, "success": 1, "uc": 2, "lost": 3}
NC = 9

def cells(d):
    return ((d.c.clip(1, 3) - 1) * 3 + np.searchsorted(WB, d.wkts.clip(0, 9), side="right") - 1).values

def sstar(tau, mu, s):
    v = s * s / (1 + s * s)
    return (1 + s * s) * np.sqrt(v) * norm.ppf(tau) - mu * s * s

def bvn(h, k, r):
    """P(U < h, V < k), standard bivariate normal with correlation r, exact via Owen's T (checked in wolfram/drs_check.wl)"""
    h, k = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float))
    sq = np.sqrt(1 - r * r)
    with np.errstate(divide="ignore", invalid="ignore"):
        ah = np.where(h == 0, np.inf, (k - r * h) / (h * sq)); ak = np.where(k == 0, np.inf, (h - r * k) / (k * sq))
    th = np.where(np.isinf(ah), 0.25 * np.sign(k - r * h) * (h == 0), owens_t(h, np.where(np.isinf(ah), 0, ah)))
    tk = np.where(np.isinf(ak), 0.25 * np.sign(h - r * k) * (k == 0), owens_t(k, np.where(np.isinf(ak), 0, ak)))
    delta = np.where((h * k < 0) | ((h * k == 0) & (h + k < 0)), 0.5, 0.0)
    return 0.5 * (norm.cdf(h) + norm.cdf(k)) - th - tk - delta

def probs(tau, mu, s, t=0.0):
    """q = P(review), J = P(review & X > t) for thresholds tau (array), margin mean mu, perception noise s"""
    S = sstar(tau, mu, s); sd = np.sqrt(1 + s * s)
    q = norm.sf((S - mu) / sd)
    return q, np.clip(bvn(np.full_like(S, mu - t), (mu - S) / sd, 1 / sd), 0, q)

def unpack(th, s_fixed=None):
    s = np.exp(th[2]) if s_fixed is None else s_fixed
    return dict(mu=th[:2], s=s, pi=expit(th[3]), kappa=np.exp(th[4]), tau=expit(th[5:]))

def cell_probs(P, mode, rho):
    """per kind: (P none, P success, P uc, P lost) by cell"""
    res = []
    for j in range(2):
        q, J0 = probs(P["tau"], P["mu"][j], P["s"])
        if j == 1: res.append((1 - q, J0, 0 * q, q - J0)); continue
        if mode == "band":
            _, Jk = probs(P["tau"], P["mu"][j], P["s"], -P["kappa"]); Jk = np.maximum(Jk, J0)
            res.append((1 - q, J0, Jk - J0, q - Jk))
        else:
            res.append((1 - q, J0, rho * (q - J0), (1 - rho) * (q - J0)))
    return res

def nll(th, n, mode, rho, s_fixed=None):
    """n[k, cell, outcome] counts; k = 0 lbw, 1 caught, 2 unknown kind (successes)"""
    P = unpack(th, s_fixed); pk = [P["pi"], 1 - P["pi"]]
    cp = cell_probs(P, mode, rho); L = lambda x: np.log(np.maximum(x, 1e-300))
    ll = sum((n[j, :, o] * L(pk[j] * cp[j][o])).sum() for j in range(2) for o in (0, 2, 3) if n[j, :, o].any())
    ll += (n[2, :, 1] * L(pk[0] * cp[0][1] + pk[1] * cp[1][1])).sum()
    return -ll

def counts(d):
    n = np.zeros((3, NC, 4))
    k = np.where(d.outcome == "success", 2, np.where(d.kind == "lbw", 0, 1))
    np.add.at(n, (k, cells(d), d.outcome.map(OUT).values), 1)
    return n

def fit(d, th0=None, s_fixed=None, mode="band"):
    lb = d[(d.kind == "lbw") & d.outcome.isin(["uc", "lost"])]
    rho = float((lb.outcome == "uc").mean())
    n = counts(d)
    if th0 is None: th0 = np.r_[-0.5, -3.0, np.log(1.0), logit(0.2), np.log(0.3), np.full(NC, logit(0.1))]
    th0 = np.array(th0, float).copy()
    if s_fixed is not None: th0[2] = np.log(s_fixed)
    kb = (np.log(1e-3), np.log(5)) if mode == "band" else (th0[4], th0[4])
    r = minimize(nll, th0, args=(n, mode, rho, s_fixed), method="L-BFGS-B",
                 bounds=[(-6, 3), (-6, 3), (np.log(0.05), np.log(20)), (-8, 8), kb] + [(-9, 9)] * NC, options=dict(maxiter=5000))
    P = unpack(r.x, s_fixed)
    return dict(mode=mode, theta=r.x.tolist(), rho=rho, mu=P["mu"].tolist(), s=float(P["s"]), pi_lbw=float(P["pi"]),
                kappa=float(P["kappa"]), tau=P["tau"].tolist(), nll=float(r.fun), conv=bool(r.success),
                err_rate=[float(norm.cdf(m)) for m in P["mu"]])

def fit_check(d, P):
    """observed vs fitted review share, success rate and umpire's-call share (lbw failures) by cell"""
    Pu = dict(P, tau=np.array(P["tau"]), mu=np.array(P["mu"])); pk = [P["pi_lbw"], 1 - P["pi_lbw"]]
    cp = cell_probs(Pu, P["mode"], P["rho"])
    Q = sum(pk[j] * (1 - cp[j][0]) for j in range(2)); J = sum(pk[j] * cp[j][1] for j in range(2))
    UC = cp[0][2] / np.maximum(cp[0][2] + cp[0][3], 1e-12)
    c = cells(d); rows = []
    for s in range(NC):
        g = d[c == s]
        if not len(g): continue
        rv = g.outcome != "none"; lf = g[(g.kind == "lbw") & g.outcome.isin(["uc", "lost"])]
        rows.append(dict(cell=s, c=s // 3 + 1, wkts=f"{WB[s % 3]}-{WB[s % 3 + 1] - 1}", n=len(g), tau=round(Pu["tau"][s], 3),
                         rev_obs=round(rv.mean(), 4), rev_fit=round(Q[s], 4),
                         succ_obs=round((g.outcome == "success").sum() / max(rv.sum(), 1), 3), succ_fit=round(J[s] / Q[s], 3),
                         uc_obs=round((lf.outcome == "uc").mean(), 3) if len(lf) else None, uc_fit=round(UC[s], 3)))
    return rows

S_GRID = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0]
S_MIN, S_MAX = 0.05, 20.0           # bounds of s in fit()
CHI2_95 = 1.920729410347062         # chi2_1(0.95) / 2

def fit_multi(d, starts, s_fixed=None, mode="band"):
    """best of several local fits (one L-BFGS run can stop at its start value)"""
    return min((fit(d, th, s_fixed, mode) for th in starts), key=lambda x: x["nll"])

def free_starts(thetas, k=3):
    """starts for the free fit: each theta with log s moved off its value both ways, and the theta itself"""
    out = []
    for th in thetas[:k]:
        for ds in (-0.25, 0.0, 0.25):
            t = np.array(th, float).copy(); t[2] = np.clip(t[2] + ds, np.log(S_MIN), np.log(S_MAX)); out.append(t)
    return out

def fit_full(d, mode="band"):
    """profile in s on a grid (warm-started both directions); free fit from several starts; continuous 95 % profile-LR
    endpoints by root-finding on the profile, 2 (l_p(s) - l_hat) = chi2_1(0.95)."""
    from scipy.optimize import brentq
    prof = {}
    th = None
    for sf in S_GRID:
        prof[sf] = fit(d, th, sf, mode); th = prof[sf]["theta"]
    for sf in S_GRID[::-1]:
        x = fit(d, th, sf, mode)
        if x["nll"] < prof[sf]["nll"]: prof[sf] = x
        th = prof[sf]["theta"]
    order = sorted(prof, key=lambda k: prof[k]["nll"])
    P = fit_multi(d, free_starts([prof[k]["theta"] for k in order]), None, mode)
    best = prof[order[0]]
    if P["nll"] > best["nll"]: P = dict(best, s=best["s"])
    th_hat = np.array(P["theta"])
    cache = {}
    def prof_at(sf):
        if sf not in cache:
            near = min(prof, key=lambda k: abs(np.log(k / sf)))
            cache[sf] = fit_multi(d, [th_hat, np.array(prof[near]["theta"])], sf, mode)
        return cache[sf]
    g = lambda ls: prof_at(float(np.exp(ls)))["nll"] - P["nll"] - CHI2_95
    ends, open_ = {}, {}
    for side, grid in (("lo", [x for x in S_GRID[::-1] if x < P["s"]] + [S_MIN]), ("hi", [x for x in S_GRID if x > P["s"]] + [S_MAX])):
        a, out = np.log(P["s"]), None
        for x in grid:
            if g(np.log(x)) > 0: out = brentq(g, *sorted((a, np.log(x))), xtol=1e-4); break
            a = np.log(x)
        open_[side] = out is None
        ends[side] = float(np.exp(out)) if out is not None else float(grid[-1])
    P["s_95"] = [ends["lo"], ends["hi"]]; P["s_95_open"] = [open_["lo"], open_["hi"]]
    P["profile_ends"] = {k: dict(prof_at(v), s=v) for k, v in ends.items()}
    P["check"] = fit_check(d, P)
    P["profile_s"] = [dict(x, s=sf) for sf, x in prof.items()]
    P["profile_extra"] = [dict(x, s=sf) for sf, x in sorted(cache.items())]
    return P

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    mode = sys.argv[2] if len(sys.argv) > 2 else "band"
    o = pd.read_parquet("data/drs_opps.parquet"); o = o[o.side == "bat"]
    for fmt in (["Test", "ODI", "T20"] if which == "all" else [which]):
        d = o[(o.fmt == fmt) & (o.c > 0)].reset_index(drop=True)  # with no review left there is no choice
        P = fit_full(d, mode)
        json.dump(P, open(f"results/drs/model_{fmt}{'' if mode == 'band' else '_const'}.json", "w"), indent=1)
        print(fmt, mode, "n", len(d), "rho", round(P["rho"], 3), "kappa", round(P["kappa"], 3), "mu", np.round(P["mu"], 3),
              "err", np.round(P["err_rate"], 3), "s", round(P["s"], 3), "pi_lbw", round(P["pi_lbw"], 3), "conv", P["conv"])
        print(pd.DataFrame(P["check"]).to_string(index=False))
        print("profile dnll:", [(x["s"], round(x["nll"] - P["nll"], 2)) for x in P["profile_s"]])
        print("s 95 % profile-LR set:", np.round(P["s_95"], 3), "open ends:", P["s_95_open"])
