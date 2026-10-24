"""Calibrated DP + policy simulation for batting-side DRS reviews, on the real stream of out decisions.

Per replicate: every out decision gets X ~ N(mu_kind, 1), S = X + s eps; truth W = 1{X > 0}; the batter's
posterior p = P(X > 0 | S) and l = P(review lost | S): a failed lbw review is kept (umpire's call) iff
-kappa < X < 0 (band model; "const" model: w.p. rho), so l = P(X < -kappa | S) for lbw and 1 - p for caught.
Overturned decisions have no recorded kind: it is drawn from its posterior given an overturn.
DP (one batting innings, state = wickets down w and reviews left c): from w the next event is an out decision
(prob a_w), a non-reviewable wicket (b_w) or the end of the innings; a decision draws (g, kind, p) from the pool
of decisions at w.  U(w,c) = expected stake recovered by reviews from here:
  U(w,c) = a_w E max{ U(w+1,c), p (g + U(w,c)) + (1-p) U(w+1,c) - l dU } + b_w U(w+1,c),  dU = U(w+1,c) - U(w+1,c-1)
-> review iff p (g + A) >= l dU,  A = U(w,c) - U(w+1,c); with l = (1-rho)(1-p) this is p >= D/(g+A+D),
D = (1-rho) dU (wolfram/drs_check.wl).
Policies on the same draws: opt (DP), stake (p >= d/(g+d), best constant d in match value), conf50 (p >= 0.5),
obs_rule (fitted revealed thresholds tau(c, wickets)), plus the realised value of the real reviews.
Value = sum of g over successful reviews, in match value (P(win) / Test expected points) per team-match.
HONZA-122 additions:
  opt_nx  the DP solved without the extra opportunity after a successful review (an overturn moves on to w+1 as the
          replay on the recorded stream does): U(w,c) = a_w E max{U(w+1,c), p g + U(w+1,c) - l dU} + b_w U(w+1,c),
          review iff p g >= l dU.  DP and evaluation then share one process; reported next to opt (key opt_nx).
  crossfit  the constant delta of the one-line rule chosen on the earlier half of the matches (by date) and scored on
          the later half, and vice versa (key crossfit_delta).
  s_lo / s_hi use the continuous 95 % profile-likelihood endpoints of model.py; free fits in the bootstrap use several starts.
usage: python src/drs/dp.py [fmt|all] [R] [B] [workers]  ->  results/drs/dp_<fmt>.json
(workers > 1: bootstrap replicates in parallel, replicate j seeded by SeedSequence(11).spawn(B)[j])"""
import json, sys
import numpy as np, pandas as pd
from numba import njit
from scipy.stats import norm
sys.path.insert(0, "src/drs")
from model import fit_multi, free_starts, probs, cells

D_GRID = np.array([0.0002, 0.0005, 0.001, 0.002, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12])
CMAX = 3

def rates(ev):
    t = ev.groupby(["w", "ev"]).size().unstack(fill_value=0).reindex(range(10), fill_value=0)
    n = t.sum(axis=1).clip(lower=1)
    return (t.get("opp", 0) / n).values, (t.get("other", 0) / n).values

def draw(d, P, rng):
    mu = np.array(P["mu"]); s = P["s"]
    kind = np.where(d.kind.values == "lbw", 0, np.where(d.kind.values == "caught", 1, -1))
    unk = kind < 0
    if unk.any():   # kind of an overturned decision, from its posterior given an overturn
        # HONZA-130 (Sol review): at the row's own threshold tau(reviews left, wickets) -- the cell of the fitted
        # likelihood (model.nll mixes the kinds of a success cell by cell) -- not at the mean of the nine thresholds
        tau = np.asarray(P["tau"], float)[cells(d)[unk]]
        pl = P["pi_lbw"] * probs(tau, mu[0], s)[1]; pc = (1 - P["pi_lbw"]) * probs(tau, mu[1], s)[1]
        kind[unk] = (rng.random(unk.sum()) > pl / (pl + pc)).astype(int)
    m = mu[kind]
    X = m + rng.standard_normal(len(d)); S = X + s * rng.standard_normal(len(d))
    v = s * s / (1 + s * s); mS = (m * s * s + S) / (1 + s * s)
    p = norm.cdf(mS / np.sqrt(v))
    if P.get("mode", "band") == "band":
        lost = np.where(kind == 0, norm.cdf((-P["kappa"] - mS) / np.sqrt(v)), 1 - p)
        uc = (kind == 0) & (X < 0) & (X > -P["kappa"])
    else:
        lost = np.where(kind == 0, (1 - P["rho"]) * (1 - p), 1 - p)
        uc = (kind == 0) & (rng.random(len(d)) < P["rho"])
    return kind, X > 0, p, lost, uc

def solve_dp(w, g, p, lost, a, b, iters=60, extra=True):
    """U[w, c] for w = 0..10, c = 0..CMAX (U[10] = U[:, 0] = 0).  extra=False: no extra opportunity after an overturn"""
    U = np.zeros((11, CMAX + 1))
    for ww in range(9, -1, -1):
        k = w == ww
        gg, pp, ll = g[k], p[k], lost[k]
        for c in range(1, CMAX + 1):
            U1, dU = U[ww + 1, c], U[ww + 1, c] - U[ww + 1, c - 1]
            u = U1
            if not extra:
                e = np.maximum(U1, U1 + pp * gg - ll * dU).mean() if len(gg) else U1
                U[ww, c] = a[ww] * e + b[ww] * U1; continue
            for _ in range(iters):
                e = np.maximum(U1, pp * (gg + u) + (1 - pp) * U1 - ll * dU).mean() if len(gg) else U1
                u = a[ww] * e + b[ww] * U1
            U[ww, c] = u
    return U

@njit(cache=True)
def run_policy(inn, w, c0, g, lost, W, p, uc, U, tauc, pol, dconst, mi, nm):
    """pol 0 opt, 1 stake(dconst), 2 conf50, 3 obs_rule, 4 opt_nx (U = no-extra DP, A = 0).  returns value and reviews per match"""
    val = np.zeros(nm); nrev = np.zeros(nm)
    cur = -1; c = 0
    for i in range(len(g)):
        if inn[i] != cur:
            cur = inn[i]; c = c0[i]
        if c == 0: continue
        ww = min(w[i], 9); cc = min(c, 3)
        if pol == 0:
            dU = U[ww + 1, cc] - U[ww + 1, cc - 1]; A = U[ww, cc] - U[ww + 1, cc]
            go = p[i] * (g[i] + A) >= lost[i] * dU
        elif pol == 4:
            go = p[i] * g[i] >= lost[i] * (U[ww + 1, cc] - U[ww + 1, cc - 1])
        elif pol == 1:
            go = p[i] >= dconst / (g[i] + dconst)
        elif pol == 2:
            go = p[i] >= 0.5
        else:   # revealed rule at the policy's own state (reviews left in this simulation, not the real ones)
            go = p[i] >= tauc[(cc - 1) * 3 + (0 if ww < 3 else 1 if ww < 6 else 2)]
        if go:
            nrev[mi[i]] += 1
            if W[i]:
                val[mi[i]] += g[i]
            elif not uc[i]:
                c -= 1
    return val, nrev

def evaluate(d, ev, P, R, seed):
    """d: all batting-side decisions of one format in stream order (incl. c = 0); returns per-match arrays"""
    a, b = rates(ev)
    ms = d.match.unique(); mi = pd.Series(np.arange(len(ms)), ms)[d.match].values.astype(np.int64); nm = len(ms)
    inn = pd.factorize(d.key)[0].astype(np.int64)
    c0 = d.A.values.astype(np.int64)                    # every innings starts with the full allowance
    w = d.wkts.clip(0, 9).values.astype(np.int64); g = d.g.values
    tauc = np.array(P["tau"], float)                    # by cell (reviews left 1-3) x (wickets 0-2 / 3-5 / 6-9), as model.cells
    rng = np.random.default_rng(seed)
    out = {}; Us = []; Unx = []
    for _ in range(R):
        kind, W, p, lost, uc = draw(d, P, rng)
        U = solve_dp(w, g, p, lost, a, b); Us.append(U)
        U2 = solve_dp(w, g, p, lost, a, b, extra=False); Unx.append(U2)
        v, n = run_policy(inn, w, c0, g, lost, W, p, uc, U2, tauc, 4, 0.0, mi, nm)
        o = out.setdefault("opt_nx", [np.zeros(nm), np.zeros(nm)]); o[0] += v / R; o[1] += n / R
        pols = {"opt": (0, 0.0), "conf50": (2, 0.0), "obs_rule": (3, 0.0)}
        for dc in D_GRID: pols[f"stake_{dc}"] = (1, dc)
        for name, (pc, dc) in pols.items():
            v, n = run_policy(inn, w, c0, g, lost, W, p, uc, U, tauc, pc, dc, mi, nm)
            o = out.setdefault(name, [np.zeros(nm), np.zeros(nm)]); o[0] += v / R; o[1] += n / R
    rv = (d.outcome != "none").values
    ov, on = np.zeros(nm), np.zeros(nm)
    np.add.at(ov, mi[rv], (d.g * (d.outcome == "success")).values[rv]); np.add.at(on, mi[rv], 1)
    out["observed"] = [ov, on]
    out["_Unx"] = np.mean(Unx, axis=0)
    return out, np.mean(Us, axis=0), a, b

def crossfit_delta(out, d):
    """one-line rule with delta chosen on one half of the matches (split at the median match date), scored on the other"""
    ms = d.match.unique(); dt = d.groupby("match").date.min().reindex(ms).values
    o = np.argsort(dt, kind="stable"); h = np.zeros(len(ms), bool); h[o[: len(ms) // 2]] = True     # True = earlier half
    keys = [f"stake_{dc}" for dc in D_GRID]
    val = lambda k, m: float(out[k][0][m].sum())
    r = dict(split_date=str(np.datetime_as_string(dt[o[len(ms) // 2]], "D")))
    num = 0.0
    for nm_, fit_h in (("early_to_late", h), ("late_to_early", ~h)):
        k = max(keys, key=lambda k: val(k, fit_h)); ev_h = ~fit_h
        r[nm_] = dict(delta=float(k.split("_")[1]), share=val(k, ev_h) / val("opt", ev_h),
                      share_vs_opt_nx=val(k, ev_h) / val("opt_nx", ev_h), n_fit=int(fit_h.sum()), n_eval=int(ev_h.sum()))
        num += val(k, ev_h)
    r["share"] = num / val("opt", np.ones(len(ms), bool))      # pooled out-of-half share of the optimum
    return r

def summarise(out, nm):
    per = {k: float(v[0].sum() / (2 * nm)) for k, v in out.items() if not k.startswith("_")}     # per team-match
    rev = {k: float(v[1].sum() / (2 * nm)) for k, v in out.items() if not k.startswith("_")}
    best = max((k for k in per if k.startswith("stake_")), key=per.get)
    return per, rev, best

def thresholds(U, P, g_ref):
    """optimal p* at the reference stake for every (w, c): smallest p with p (g + A) >= l(p) dU"""
    s = P["s"]; v = s * s / (1 + s * s); T = {}
    for j, kname in enumerate(("lbw", "caught")):
        mS = (P["mu"][j] * s * s + np.linspace(-12, 12, 4001)) / (1 + s * s)
        p = norm.cdf(mS / np.sqrt(v))
        if kname == "caught": l = 1 - p
        elif P.get("mode", "band") == "band": l = norm.cdf((-P["kappa"] - mS) / np.sqrt(v))
        else: l = (1 - P["rho"]) * (1 - p)
        T[kname] = [[float(p[np.argmax(p * (g_ref + U[w, c] - U[w + 1, c]) >= l * (U[w + 1, c] - U[w + 1, c - 1]))])
                     for c in range(1, CMAX + 1)] for w in range(10)]
    return T

def sqrt_law(U, a, b):
    """Delta_1(w) = U(w,1) - U(w,0) against expected decisions left N(w) (no reviews): sqrt vs linear fit"""
    N = np.zeros(11)
    for w in range(9, -1, -1): N[w] = a[w] * (1 + N[w + 1]) + b[w] * N[w + 1]
    y = U[:10, 1]; n = N[:10]
    def r2(x):
        X = np.c_[np.ones_like(x), x]; beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()), beta.tolist()
    k = y > 1e-4                                         # power law log D1 = a + beta log N
    beta = np.polyfit(np.log(n[k]), np.log(y[k]), 1)[0] if k.sum() > 2 else np.nan
    return dict(N=n.tolist(), delta1=y.tolist(), delta2=(U[:10, 2] - U[:10, 1]).tolist(),
                sqrt=r2(np.sqrt(n)), linear=r2(n), power=float(beta))

def boot_once(d, ev, M, R, rng):
    idx = d.groupby("match").indices; ms = np.array(list(idx)); pick = rng.choice(ms, len(ms))
    db = d.iloc[np.concatenate([idx[m] for m in pick])].copy()
    tag = np.repeat([f"{m}_{j}" for j, m in enumerate(pick)], [len(idx[m]) for m in pick])
    db["key"] = tag + "_" + db.inn.astype(str).values; db["match"] = tag; db = db.reset_index(drop=True)
    Pb = fit_multi(db[db.c > 0].reset_index(drop=True), free_starts([M["theta"]]))
    eg = ev.groupby("match").indices
    evb = ev.iloc[np.concatenate([eg[m] for m in pick])]
    out, U, a, b = evaluate(db, evb, Pb, R, int(rng.integers(1 << 30)))
    per, rev, best = summarise(out, len(pick))
    return dict(per=per, rev=rev, best=best, s=Pb["s"], crossfit=crossfit_delta(out, db))

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    NW = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    o = pd.read_parquet("data/drs_opps.parquet"); o = o[o.side == "bat"]
    ev_all = pd.read_parquet("data/drs_innings.parquet")
    for fmt in (["Test", "ODI", "T20"] if which == "all" else [which]):
        d = o[o.fmt == fmt].sort_values("oid").reset_index(drop=True)
        ev = ev_all[ev_all.fmt == fmt]
        M = json.load(open(f"results/drs/model_{fmt}.json"))
        variants = {"main": M, "s_lo": dict(M["profile_ends"]["lo"], mode="band"),       # continuous 95 % profile-LR endpoints
                    "s_hi": dict(M["profile_ends"]["hi"], mode="band"),
                    "s2.0": dict(next(x for x in M["profile_s"] if x["s"] == 2.0), mode="band")}
        try: variants["const"] = json.load(open(f"results/drs/model_{fmt}_const.json"))
        except FileNotFoundError: pass
        res = {}
        for vn, P in variants.items():
            out, U, a, b = evaluate(d, ev, P, R, 1)
            per, rev, best = summarise(out, d.match.nunique())
            res[vn] = dict(s=P["s"], dnll=P["nll"] - M["nll"] if P.get("mode") == "band" else None, value=per, reviews=rev,
                           best_stake=best, U=U.tolist(), a=a.tolist(), b=b.tolist(),
                           thresholds=thresholds(U, P, float(np.median(d.g[d.outcome != "none"]))), sqrt_law=sqrt_law(U, a, b),
                           crossfit_delta=crossfit_delta(out, d), U_nx=out["_Unx"].tolist(), sqrt_law_nx=sqrt_law(out["_Unx"], a, b))
            sl = res[vn]["sqrt_law"]
            print(fmt, vn, "s", round(P["s"], 2), {k: round(per[k] * 100, 2) for k in ("opt", best, "obs_rule", "observed", "conf50")},
                  "rev", {k: round(rev[k], 2) for k in ("opt", best, "obs_rule", "observed")},
                  "R2 sqrt/lin", round(sl["sqrt"][0], 3), round(sl["linear"][0], 3), "beta", round(sl["power"], 2),
                  "opt_nx", round(per["opt_nx"] * 100, 2), "xfit share", round(res[vn]["crossfit_delta"]["share"], 3), flush=True)
        if NW > 1:
            from concurrent.futures import ProcessPoolExecutor
            rngs = [np.random.default_rng(q) for q in np.random.SeedSequence(11).spawn(B)]
            with ProcessPoolExecutor(NW) as ex:
                res["boot"] = list(ex.map(boot_once, *zip(*[(d, ev, M, max(R // 2, 2), r) for r in rngs])))
        else:
            rng = np.random.default_rng(11)
            res["boot"] = [boot_once(d, ev, M, max(R // 2, 2), rng) for _ in range(B)]
        json.dump(res, open(f"results/drs/dp_{fmt}.json", "w"), indent=1)
