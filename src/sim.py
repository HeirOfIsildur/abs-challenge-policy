"""Calibrated DP + policy simulation on the 2026 called-pitch stream.

Per replicate: eta ~ N(0, sm_edge) -> truth W = 1{w + eta > 0} (main specs: sm = 0, and the ruling flips W
with the fitted mislabel rate pi); signal s = w + eta + eps, eps ~ N(0, sr_side);
player's posterior p~ = E[W | s, side, edge] (Bayes under the empirical prior, estimated by binning s).
DP (per team: home/away; slots = 18 regulation half-innings + extra block): opportunities in a slot are i.i.d.
draws from the pooled (g, p~) of that slot; Delta_k(c) = V_k(c) - V_k(c-1); challenge iff p~ >= Delta/(g+Delta).
Policies (all on the same p~ draws): opt (DP), savant (p~ >= 0.2/(0.2+RV)), stake (p~ >= d/(g+d), best constant
d in WP), conf50 (p~ >= 0.5), obs_rule (fitted kappa rule: s > X beta).  Value = sum of g over successful
challenges; extra-inning top-up rule applied.  Output: per-game x team values for bootstrap."""
import json, sys
import numpy as np, pandas as pd
from numba import njit
from scipy.stats import norm
sys.path.insert(0, "src")
from perception import design
from joint import unpack

def load(variant):
    m = variant["frame"].copy() if variant.get("frame") is not None else pd.read_parquet("data/called_2026_w.parquet")
    if variant.get("games") is not None:                                 # subset of games (out-of-sample split)
        m = m[m.game_pk.isin(set(variant["games"]))]
    m = m.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    P = unpack(np.array(json.load(open("data/joint.json"))["theta"]))
    if variant.get("naive_truth"):
        tp = json.load(open("data/perception.json"))["truth"]
        P["mu"] = dict(vert=tp["vert"][0], horiz=tp["horiz"][0]); P["sm"] = dict(vert=tp["vert"][1], horiz=tp["horiz"][1])
    if variant.get("theta"):
        m = apply_lik(m, variant)
    else:
        isv = m.edge.values == "vert"
        mu = np.where(isv, P["mu"]["vert"], P["mu"]["horiz"])
        m["sm"] = np.where(isv, P["sm"]["vert"], P["sm"]["horiz"])
        dc = m.dist_in.values - mu
        m["w"] = np.where(m.orig_strike.values, -dc, dc)
        m["sr"] = np.where(m.side == "off", P["off"]["sr"], P["deff"]["sr"])
        X = design(m)
        m["kappa"] = np.where(m.side == "off", X @ P["off"]["beta"], X @ P["deff"]["beta"])
        for c in (1, 2):                                  # criterion at the count the simulated team holds
            Xc = design(m.assign(left=c))
            m[f"kappa{c}"] = np.where(m.side == "off", Xc @ P["off"]["beta"], Xc @ P["deff"]["beta"])
        m["pgrp"] = m.edge.values
    m["sr"] = m.sr * variant.get("sr_mult", 1.0)
    m["is_home_team"] = np.where(m.side == "off", ~m.top, m.top)       # whose opportunity
    m["half"] = (~m.top).astype(int)
    m["slot"] = np.where(m.inning >= 10, 18, (np.clip(m.inning, 1, 9) - 1) * 2 + m.half)
    m["new_extra"] = extra_starts(m)
    m["gidx"] = m.game_pk.astype("category").cat.codes
    return m, P

def extra_starts(m):
    """first row of every extra inning in m (m sorted within game): where a team at 0 challenges gets 1.
    the old flag (first top-half row whose inning differs from the previous row) missed extra innings whose
    first retained opportunity is in the bottom half or follows an inning without opportunities (4 of 285 in 2026)."""
    return ((m.inning >= 10) & ~m.duplicated(["game_pk", "inning"])).values


def apply_lik(m, variant):
    """row-level truth offset/scale, perception noise and threshold from a lik.py fit.
    variant: theta (path to.npy), spec (lik spec), optional re (True: per-player effects from data/re_<side>.parquet;
    "draw": a fresh draw from each player's EB posterior), theta_override (array, e.g. a bootstrap draw)."""
    import lik
    spec = variant["spec"]
    th = np.asarray(variant["theta_override"]) if variant.get("theta_override") is not None else np.load(variant["theta"])
    P = lik.unpack(th, spec)
    reg = lik.region(m)
    if spec.get("truth_region"):
        mu = np.asarray(P["mu"])[reg]; sm = np.exp(np.asarray(P["lsm"]))[reg]
    else:
        isv = m.edge.values == "vert"
        mu = np.where(isv, float(P["mu"][0]), float(P["mu"][1])); sm = np.exp(np.where(isv, float(P["lsm"][0]), float(P["lsm"][1])))
    dc = m.dist_in.values - mu
    m["w"] = np.where(m.orig_strike.values, -dc, dc); m["sm"] = sm
    # ruling mislabel rate pi: the ABS ruling disagrees with the tracked truth with probability pi
    m["pi"] = float(1 / (1 + np.exp(-float(P["lpi"])))) if "lpi" in P else 0.0
    X = lik.design(m)
    # kappa(x) also at left = 1 and left = 2, so that a simulated team applies the fitted rule with the
    # challenges it holds in the simulation, not the recorded count (m.kappa keeps the recorded count, for the data)
    Xc = {c: lik.design(m.assign(left=c)) for c in (1, 2)}
    sr = np.zeros(len(m)); kap = np.zeros(len(m)); kapc = {1: np.zeros(len(m)), 2: np.zeros(len(m))}
    for side in ("off", "def"):
        k = (m.side == side).values; q = P[side]
        lsr = np.full(k.sum(), float(q["lsr"]))
        if "edge" in q: lsr += float(q["edge"]) * (m.edge.values[k] != "vert")
        if "region" in q: lsr += np.asarray(q["region"])[reg[k]]
        if "slope" in q: lsr += float(q["slope"]) * (np.abs(m.w.values[k]) - 1)
        kk = X[k] @ np.asarray(q["beta"])
        shift = {c: Xc[c][k] @ np.asarray(q["beta"]) - kk for c in (1, 2)}
        if "kreg" in q: kk += np.asarray(q["kreg"])[reg[k]]
        if "kg" in q: kk += lik.gfeat(m[k]) @ np.asarray(q["kg"])
        if "kgl" in q: kk += lik.glow(m[k]) * float(q["kgl"])                       # low-stake hinge
        if "srg" in q: lsr += float(q["srg"]) * (np.log(np.clip(m.g.values[k], 1e-4, None)) - lik.LG_MID)
        if "kt" in q or "srt" in q:                                                  # time-varying model
            TF = lik.tfeat(m[k])
            if "kt" in q: kk += TF @ np.asarray(q["kt"])
            if "srt" in q: lsr += float(q["srt"]) * TF[:, 0]
        if "ks" in q: kk += lik.sfeat(m[k]) @ np.asarray(q["ks"])                  # game-state terms
        if variant.get("re"):
            tag = variant.get("re_tag", "re")             # posterior file of this random-effects variant
            post = pd.read_parquet(f"data/re_{side}.parquet" if tag == "re" else f"data/{tag}_{side}.parquet").set_index("player")
            pid = m[lik.PLAYER[side]].values[k]
            a = post.a.reindex(pid).fillna(0).values; b = post.b.reindex(pid).fillna(0).values
            if variant.get("re") == "draw":
                # one draw per player per replicate from the player's EB posterior (it was one per pitch,
                # which averaged the player-effect uncertainty away), with an independent stream for each side
                rng = np.random.default_rng([variant.get("re_seed", 0), 0 if side == "off" else 1])
                za = pd.Series(rng.standard_normal(len(post)), index=post.index)
                zb = pd.Series(rng.standard_normal(len(post)), index=post.index)
                if "cov_ab" in post:
                    # a JOINT draw from the player's bivariate posterior (Cholesky of
                    # [[sd_a^2, cov], [cov, sd_b^2]]); the independent marginal draws ignored the correlation
                    sa_, sb_ = post.sd_a.values, post.sd_b.values
                    r_ = np.clip(post.cov_ab.values / np.maximum(sa_ * sb_, 1e-12), -0.999, 0.999)
                    ea_ = pd.Series(sa_ * za.values, index=post.index)
                    eb_ = pd.Series(sb_ * (r_ * za.values + np.sqrt(1 - r_ ** 2) * zb.values), index=post.index)
                else:
                    ea_ = za * post.sd_a; eb_ = zb * post.sd_b
                a = a + ea_.reindex(pid).fillna(0).values
                b = b + eb_.reindex(pid).fillna(0).values
            kk += a; lsr += b
        sr[k] = np.exp(lsr); kap[k] = kk
        for c in (1, 2):
            kapc[c][k] = kk + shift[c]
    m["sr"] = sr; m["kappa"] = kap; m["kappa1"] = kapc[1]; m["kappa2"] = kapc[2]
    # the lapse-rate model's behavioural rule challenges with probability (1 - lambda) when
    # s > kappa; the simulated rule ignored lambda. m.lapse carries lambda by side (0 without a lapse term).
    m["lapse"] = 0.0
    for side in ("off", "def"):
        if "llapse" in P[side]:
            m.loc[(m.side == side).values, "lapse"] = float(1 / (1 + np.exp(-float(P[side]["llapse"]))))
    # groups within which the player's posterior P(W | s) is calibrated: region (truth scale) x the player's own noise
    # level (random effects only; a noise level that depends on the unknown distance |w| is not known to the player)
    g = pd.Series(reg).astype(str).values
    if variant.get("re"):
        q5 = np.zeros(len(m), int)
        for side in ("off", "def"):
            k = (m.side == side).values
            q5[k] = pd.qcut(np.log(sr[k]), 5, labels=False, duplicates="drop")
        g = np.char.add(np.char.add(g.astype(str), "_"), q5.astype(str))
    m["pgrp"] = g
    return m


def monotone_calibration(num, den):
    """P(W | signal bin): weighted isotonic (pool-adjacent-violators) fit of num/den, the monotone MLE.
    replaces a running maximum, which let one sparse extreme bin (e.g. 1 wrong call out of 2 in the clipped
    s < -15 bin) lift every higher bin and set p >= 0.5 for a whole zone region in some draws."""
    from sklearn.isotonic import IsotonicRegression
    j = np.flatnonzero(den > 0)
    fit = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0).fit(j, num[j] / den[j], sample_weight=den[j])
    return fit.predict(np.arange(len(den)))


def unit_noise(rng, n, law="normal"):
    """n draws of the perception-noise law F with unit variance (lik.cdf_unit): normal, logistic or t<nu>.
    truth_signal always drew Gaussian noise, so the logistic and t variants were
    evaluated with Gaussian signals at their fitted scales. The normal law keeps the same generator calls as before."""
    if law == "normal":
        return rng.standard_normal(n)
    if law == "logistic":
        return rng.logistic(0.0, np.sqrt(3.0) / np.pi, n)
    nu = int(law[1:])
    return rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)


def law_of(variant):
    return (variant.get("spec") or {}).get("law", "normal") if variant else "normal"


def truth_signal(m, rng, law="normal", rho=1.0):
    """truth W and the PLAYER's signal s = w + eta + sqrt(rho) sr e. rho: the share of the fitted decision
    noise sr^2 that is perceptual, i.e. that a change of decision rule cannot remove; rho = 1 (the primary
    intervention) is the fitted model's reading, and draws the same random numbers as before."""
    eta = rng.standard_normal(len(m)) * m.sm.values
    W = (m.w.values + eta) > 0
    s = m.w.values + eta + unit_noise(rng, len(m), law) * m.sr.values * np.sqrt(rho)
    if "pi" in m and m.pi.values.max() > 0:              # ruling mislabel; not seen by the player's signal
        W = W ^ (rng.random(len(m)) < m.pi.values)
    return W, s


N_AUX = 2


def draw(m, rng, variant, return_maps=False):
    """truth W, signal s and the player's calibrated posterior pt = P(W | s bin, side, group).
    the calibration map is fitted on an independent auxiliary draw of (W, s) on the same pitches, not on the
    W that then scores the policies: a player's belief is learned from experience, and the in-sample fit leaked the
    realised truth into the decision (+0.015 wins per team-season for the optimum, results/crossfit_calibration.json).
    return_maps=True also returns the per-cell calibration maps {"side|cell": f[301]} (signal
    bin b = clip(floor((s + 15) / 0.1), 0, 300) -> posterior), so that causal.py can read a pool pitch's posterior in
    the cell of the simulated state (stake tercile). Same random numbers and the same (W, s, pt) either way."""
    law = law_of(variant)
    rho = float(variant.get("rho", 1.0))
    if rho < 1.0 and law != "normal":
        raise ValueError("rho < 1 splits Gaussian decision noise; not defined for the law " + law)
    W, s = truth_signal(m, rng, law, rho)                # s: the player's signal (perceptual noise only)
    pt = np.zeros(len(m)); maps = {}
    if variant.get("flat_prior"):
        pt = norm.cdf(s / m.sr.values)
    else:
        # auxiliary draws: the experience the belief is learned from (two draws, since the calibration cells
        # are finer); on the player's own (perceptual) signal
        aux = [truth_signal(m, rng, law, rho) for _ in range(N_AUX)]
        grp = m["cal"].values if "cal" in m else cal_groups(m)
        side_v = m.side.values
        for gname in np.unique(np.char.add(side_v.astype(str), np.char.add("|", grp))):
            sd, e = gname.split("|", 1)
            k = (side_v == sd) & (grp == e)
            if not k.any():
                continue
            num = np.zeros(301); den = np.zeros(301)
            for Wa, sa in aux:
                ba = np.clip(np.floor((sa[k] + 15) / 0.1).astype(int), 0, 300)
                num += np.bincount(ba, weights=Wa[k], minlength=301); den += np.bincount(ba, minlength=301)
            f = monotone_calibration(num, den)
            ok = np.flatnonzero(~np.isnan(f))             # signal bins empty in the auxiliary draws: nearest fitted value
            f = np.interp(np.arange(301), ok, f[ok])
            pt[k] = f[np.clip(np.floor((s[k] + 15) / 0.1).astype(int), 0, 300)]
            maps[gname] = f
    if rho < 1.0:
        # the behavioural rule decides on the player's signal plus the removable criterion noise, so that
        # its total decision noise is the fitted sr; s returned is that behavioural signal (obs_rule uses it)
        s = s + rng.standard_normal(len(m)) * m.sr.values * np.sqrt(1.0 - rho)
    if return_maps:
        return W, s, pt, (maps if not variant.get("flat_prior") else None)
    return W, s, pt

def solve_dp(m, pt, team_home, nrep_pool=None, extra_value=True):
    """Return Delta[slot, k, c] (c=1,2) for one team side, k = opportunity index within slot (capped)."""
    g = m.g.values; isht = m.is_home_team.values == team_home
    ngames = m.game_pk.nunique()
    slots = list(range(19))
    pools, nh = {}, {}
    for sl in slots:
        k = isht & (m.slot.values == sl)
        pools[sl] = (g[k], pt[k]); nh[sl] = k.sum() / ngames
    # extras: pool of the extra block; expected number of extra opportunities per game reaching extras
    games_extra = m[m.inning >= 10].game_pk.nunique()
    nh[18] = (isht & (m.slot.values == 18)).sum() / max(games_extra, 1)
    q_extra = games_extra / ngames
    KMAX = 80
    Delta = np.zeros((19, KMAX, 3))
    def step(V, sl):
        gg, pp = pools[sl]
        D1, D2 = V[1] - V[0], V[2] - V[1]
        n = len(gg)
        a1 = np.maximum(0, pp * gg - (1 - pp) * D1).mean() if n else 0
        a2 = np.maximum(0, pp * gg - (1 - pp) * D2).mean() if n else 0
        return np.array([V[0], V[1] + a1, V[2] + a2])
    # extras block: top-up makes c=0 behave like c=1 at each extra inning start; continuation ~ stationary
    Vx = np.zeros(3)
    if extra_value and nh[18] > 0:
        for _ in range(30):
            V = np.array([Vx[1], Vx[1], Vx[2]]) * 0.5  # ~half of extra innings end tied -> another inning (top-up)
            n = int(round(nh[18] * 0.5))  # per extra inning, about half the per-game extra opportunities
            for _k in range(max(n, 1)):
                V = step(V, 18)
            Vx = V
    V = q_extra * Vx if extra_value else np.zeros(3)
    V = np.array([q_extra * Vx[1], q_extra * Vx[1], q_extra * Vx[2]]) if extra_value else np.zeros(3)
    V[0] = q_extra * Vx[1] if extra_value else 0.0
    # c = 0 in regulation: only value is the extra-inning top-up
    Vzero = V[0]
    for sl in range(17, -1, -1):
        n = int(round(nh[sl]))
        seq = []
        for k in range(n - 1, -1, -1):
            seq.append((k, V[1] - V[0], V[2] - V[1]))
            V = step(V, sl); V[0] = Vzero
        for k, d1, d2 in seq:
            if k < KMAX:
                Delta[sl, k, 1], Delta[sl, k, 2] = d1, d2
        n_cap = min(n, KMAX)
        Delta[sl, n_cap:, 1] = Delta[sl, max(n_cap - 1, 0), 1]; Delta[sl, n_cap:, 2] = Delta[sl, max(n_cap - 1, 0), 2]
    # extra slot thresholds: stationary approx
    Delta[18, :, 1] = Vx[1] * 0.5 - 0; Delta[18, :, 2] = (Vx[2] - Vx[1]) * 0.5
    return Delta, V

@njit(cache=True)
def run_policy(gidx, is_home_team, new_extra, slot, kin, W, pt, s, g, RV, kappa, Dh, Da, pol, dconst, ngames, isoff):
    """pol: 0 opt, 1 savant, 2 stake(dconst), 3 conf50, 4 obs_rule(s > kappa). returns value[game, team(0 away,1 home)], n, nwin"""
    val = np.zeros((ngames, 2, 2)); nch = np.zeros((ngames, 2, 2)); nwin = np.zeros((ngames, 2, 2))
    c = np.array([2, 2]); cur = -1
    for i in range(len(g)):
        if gidx[i] != cur:
            cur = gidx[i]; c[0] = 2; c[1] = 2
        if new_extra[i]:
            for t in range(2):
                if c[t] == 0: c[t] = 1
        t = 1 if is_home_team[i] else 0
        if c[t] == 0 or pt[i] <= 0.0: continue
        if pol == 0:
            D = Dh[slot[i], min(kin[i], Dh.shape[1] - 1), min(c[t], 2)] if t == 1 else Da[slot[i], min(kin[i], Da.shape[1] - 1), min(c[t], 2)]
            go = pt[i] * (g[i] + D) >= D
        elif pol == 1:
            go = pt[i] >= 0.2 / (0.2 + RV[i])
        elif pol == 2:
            go = pt[i] >= dconst / (g[i] + dconst)
        elif pol == 3:
            go = pt[i] >= 0.5
        else:
            go = s[i] > kappa[i, min(c[t], 2)]
        if go:
            sd = 1 if isoff[i] else 0
            nch[cur, t, sd] += 1
            if W[i]:
                val[cur, t, sd] += g[i]; nwin[cur, t, sd] += 1
            else:
                c[t] -= 1
    return val, nch, nwin

def kappa_by_count(m):
    """[n, 3]: the fitted criterion with c = 0 (unused), 1 and 2 challenges held, for the simulated behavioural rule."""
    return np.ascontiguousarray(np.column_stack([m.kappa.values, m.kappa1.values, m.kappa2.values]).astype(float))

def kin_index(m):
    """opportunity index of the team within the half-inning (it ran on across extra innings, which share
    slot 18, and counted only |w| < 8 pitches, a truth-based filter the player cannot apply)."""
    return m.groupby([m.game_pk.values, m.is_home_team.values, m.inning.values, m.top.values]).cumcount().values


def opportunity_mask(m):
    """challenge opportunities for the simulated policies: every located
    called pitch except those thrown by position players (not challengeable under MLB's definition). The |w| < 8 in
    window is a fitting trim of the likelihood only; it used to define the policies' opportunity set as well, which
    let the policies screen on the true distance."""
    return ~m.pos_pitcher.values if "pos_pitcher" in m else np.ones(len(m), bool)


def opportunities(m):
    return m[opportunity_mask(m)].reset_index(drop=True)


def stake_cuts(m):
    """stake tercile cut points by side (cal_groups); a frame can borrow another's (postseason: regular-season cuts)."""
    return {side: np.quantile(m.g.values[(m.side == side).values], [1 / 3, 2 / 3]) for side in ("off", "def")
            if (m.side == side).any()}


def cal_groups(m, cuts=None):
    """calibration cells of the player's posterior: zone region (x the player's own
    noise quintile for the random-effects model) x count class (two strikes / three balls / other) x stake tercile
    within the side. Both the count and the stake are known to the player."""
    cnt = np.where(m.strikes.values == 2, "s2", np.where(m.balls.values == 3, "b3", "o"))
    gq = np.zeros(len(m), int); cuts = cuts or stake_cuts(m)
    for side in ("off", "def"):
        k = (m.side == side).values
        if k.any():
            gq[k] = np.searchsorted(cuts[side], m.g.values[k], side="right")
    return np.char.add(np.char.add(np.char.add(m.pgrp.values.astype(str), "_" ), cnt), np.char.add("_", gq.astype(str)))

def evaluate(variant, R=10, seed=1, stake_grid=None):
    m, P = load(variant)
    m = opportunities(m)
    m["new_extra"] = extra_starts(m)
    kin = kin_index(m)
    ng = m.gidx.max() + 1
    rng = np.random.default_rng(seed)
    out = {}; Ds = []
    for r in range(R):
        W, s, pt = draw(m, rng, variant)
        Dh, _ = solve_dp(m, pt, True, extra_value=not variant.get("no_extra"))
        Da, _ = solve_dp(m, pt, False, extra_value=not variant.get("no_extra"))
        Ds.append((Dh, Da))
        args = (m.gidx.values, m.is_home_team.values, m.new_extra.values, m.slot.values.astype(np.int64), kin.astype(np.int64),
                W, pt, s, m.g.values, m.RV.values, kappa_by_count(m), Dh, Da)
        isoff = (m.side == 'off').values
        pols = {"opt": (0, 0.0), "savant": (1, 0.0), "conf50": (3, 0.0), "obs_rule": (4, 0.0)}
        for d in (stake_grid if stake_grid is not None else [0.002, 0.004, 0.006, 0.008, 0.01, 0.0125, 0.015, 0.02, 0.03]):
            pols[f"stake_{d}"] = (2, d)
        for name, (pc, dc) in pols.items():
            v, n, w = run_policy(*args, pc, dc, ng, isoff)
            if name not in out: out[name] = [np.zeros_like(v), np.zeros_like(n), np.zeros_like(w)]
            out[name][0] += v / R; out[name][1] += n / R; out[name][2] += w / R
    # observed (real) realized value
    mm, _ = load(variant)
    obs = np.zeros((ng, 2, 2)); obsn = np.zeros((ng, 2, 2))
    ch = mm[mm.ch]
    gmap = dict(zip(m.game_pk.unique(), range(ng)))
    gi = ch.game_pk.map(gmap).values; ti = ch.is_home_team.values.astype(int)
    si = (ch.side == 'off').values.astype(int)
    np.add.at(obs, (gi, ti, si), (ch.g * ch.ch_over).values); np.add.at(obsn, (gi, ti, si), 1)
    out["observed"] = [obs, obsn, None]
    return m, out, Ds

if __name__ == "__main__":
    import pickle
    variant = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    tag = sys.argv[2] if len(sys.argv) > 2 else "main"
    m, out, Ds = evaluate(variant)
    per = {k: v[0].sum() / 30 for k, v in out.items()}
    nn = {k: v[1].sum() / (2 * len(v[1])) for k, v in out.items()}
    for k in per: print(f"{k:14s} wins/team-season {per[k]:.3f}  challenges/team-game {nn[k]:.2f}")
    pickle.dump(dict(out={k: (v[0], v[1]) for k, v in out.items()}, Ds=Ds, games=m.game_pk.unique()), open(f"data/sim_{tag}.pkl", "wb"))
