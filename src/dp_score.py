"""Score-aware calibrated DP (HONZA-86; opportunity law and extra innings rebuilt in HONZA-122).

sim.solve_dp pools every opportunity of a half-inning slot regardless of the score, so its continuation value
Delta(slot, k, c) is the same in a blowout and in a tie game.  The immediate stake g does depend on the score, so the
pooled rule over-values a token when the game is lopsided (g tiny, Delta average) and under-values it when the game
is close.  A run-ratio rule p RV > (1-p) C does not have this defect, and on the 2026 stream it beat the pooled
"optimum", which an optimum cannot allow.

State: half-inning slot (18 regulation halves, then the extra-inning halves), home score differential d at the START
of the half-inning (clipped to +-DM), opportunity index k within the half, challenges held c.
  V_end(slot, d, c)    = sum_r P(r | slot) V_start(slot + 1, d -+ r, c)   (runs r by the batting team; game over after
                         the top of the 9th if the home team leads, and after the 9th if d != 0; tie -> extra innings)
  within the half      : the number N of the team's opportunities is random, with the empirical law of N in the
                         half-innings observed at (slot, d); after k opportunities another one comes with the hazard
                         h_k = P(N > k | N >= k), and (g, p) are i.i.d. draws from the pool of (slot, d).
                         C_k(c) = h_k U_k(c) + (1 - h_k) V_end(c),
                         U_k(c) = E max{C_{k+1}(c), p g + p C_{k+1}(c) + (1 - p) C_{k+1}(c - 1)}
  Delta(slot, d, k, c) = C_{k+1}(c) - C_{k+1}(c-1): the threshold at opportunity k is p >= Delta/(g + Delta).
HONZA-122 (Fable, Astra, Sol reviews): N was its rounded mean (64,192 of 257,668 opportunities fell beyond it), the
extra-inning block was a score-free heuristic with a hard-coded continuation factor 0.5, and d was the score at the
first called pitch, not at the start of the half (different in 477 half-innings).
Extra innings (exact stationary DP): every extra inning starts tied, with a runner on second and the top-up (a team
at 0 challenges gets 1). X(c) = Y(topup(c)); Y = top half (pool and N law of extra top halves), ending in
sum_r Px(r) B_{-r}(c); B_d = bottom half (pool and N law of extra bottom halves), ending in Px(-d) X(c) (the game goes
on only if the home team ties exactly; otherwise it is over). Px = runs in an extra top half (runner on 2nd), from
2026 Statcast. Frame encoding: extra rows have slot 18; top halves carry dbin = topx() = DM + 1 (a lead the home team
cannot hold at the start of an extra top half), bottom halves dbin = d + DM with d <= 0."""
import numpy as np, pandas as pd
from numba import njit

DM = 6; ND = 2 * DM + 1; KMAX = 80; RMAX = 8
def topx():
    """dbin of an extra-inning top half: DM + 1, read at call time (dp_checks.py changes DM)."""
    return DM + 1


def half_runs():
    """P(runs scored in the half-inning | slot 0..17) from 2026 Statcast, cached."""
    try:
        return np.load("data/half_runs.npy")
    except FileNotFoundError:
        P = half_runs_from(pd.read_parquet("data/statcast_2026.parquet", columns=["game_pk", "inning", "inning_topbot",
                                           "at_bat_number", "pitch_number", "bat_score", "post_bat_score"]))
        np.save("data/half_runs.npy", P)
        return P


def half_runs_from(sc):
    """P(runs scored in the half-inning | slot 0..17) from a Statcast frame (all of 2026, or a game resample)."""
    sc = sc[sc.inning <= 9].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    h = sc.groupby(["game_pk", "inning", "inning_topbot"]).agg(a=("bat_score", "first"), b=("post_bat_score", "last")).reset_index()
    h["slot"] = (h.inning - 1) * 2 + (h.inning_topbot == "Bot")
    h["r"] = (h.b - h.a).clip(0, RMAX)
    P = np.zeros((18, RMAX + 1))
    np.add.at(P, (h.slot.values, h.r.values.astype(int)), 1)
    return P / P.sum(1, keepdims=True)


def extra_runs():
    """P(runs scored in an extra-inning TOP half) (runner on second; not truncated by a walk-off), cached."""
    try:
        return np.load("data/extra_runs.npy")
    except FileNotFoundError:
        sc = pd.read_parquet("data/statcast_2026.parquet", columns=["game_pk", "inning", "inning_topbot", "at_bat_number",
                                                                    "pitch_number", "bat_score", "post_bat_score"])
        sc = sc[(sc.inning >= 10) & (sc.inning_topbot == "Top")].sort_values(["game_pk", "at_bat_number", "pitch_number"])
        h = sc.groupby(["game_pk", "inning"]).agg(a=("bat_score", "first"), b=("post_bat_score", "last"))
        r = (h.b - h.a).clip(0, RMAX).astype(int).values
        P = np.bincount(r, minlength=RMAX + 1) / len(r)
        np.save("data/extra_runs.npy", P)
        return P


def add_dstart(m):
    """home score differential at the start of each half-inning (build.py d_half_start; frames without it fall back to
    the score at the first called pitch), clipped; adds m.dbin = d + DM, and TOPX for extra-inning top halves."""
    if "d_half_start" in m:
        first = m.d_half_start.astype(int)
    else:
        first = m.groupby(["game_pk", "inning", "top"]).score_diff_home.transform("first")
    m["dbin"] = (np.clip(first.values, -DM, DM) + DM).astype(np.int64)
    m.loc[(m.inning >= 10).values & m.top.values, "dbin"] = topx()
    return m


def hazard(counts):
    """h_k = P(N > k | N >= k), k = 0..KMAX-1, from the observed per-half-inning counts N (h = 0 where no half reaches k)."""
    n = np.minimum(np.asarray(counts, int), KMAX)
    ge = np.array([(n >= k).sum() for k in range(KMAX + 1)], float)
    return np.where(ge[:-1] > 0, ge[1:] / np.maximum(ge[:-1], 1), 0.0)


def half_value(step, h, Vend):
    """backward recursion within one half-inning: returns C_0 (value at the start of the half) and
    Delta[k, c] = C_{k+1}(c) - C_{k+1}(c-1) (k = 0..KMAX-1). step: a stepper(gg, pp) (HONZA-132: compiled with numba
    via _half_value; identical arithmetic)."""
    return _half_value(step.gg, step.pp, np.asarray(h, float), np.asarray(Vend, float), KMAX)


@njit(cache=True)
def _half_value(gg, pp, h, Vend, kmax):
    C = Vend.copy(); Dk = np.zeros((kmax, 3)); n = len(gg)
    for k in range(kmax - 1, -1, -1):
        Dk[k, 1] = C[1] - C[0]; Dk[k, 2] = C[2] - C[1]
        S = C.copy()
        if n > 0:
            D1 = C[1] - C[0]; D2 = C[2] - C[1]; a1 = 0.0; a2 = 0.0
            for j in range(n):
                x1 = pp[j] * gg[j] - (1 - pp[j]) * D1
                x2 = pp[j] * gg[j] - (1 - pp[j]) * D2
                if x1 > 0: a1 += x1
                if x2 > 0: a2 += x2
            S[1] = C[1] + a1 / n; S[2] = C[2] + a2 / n
        C = h[k] * S + (1 - h[k]) * Vend
    return C, Dk


def stepper(gg, pp):
    """Bellman step over one opportunity drawn from the pool (gg, pp): V -> E max(stay, challenge). The pool rides
    along as step.gg / step.pp for the compiled recursion (_half_value)."""
    gg = np.ascontiguousarray(gg, float); pp = np.ascontiguousarray(pp, float)
    def step(V):
        if len(gg) == 0:
            return V
        D1, D2 = V[1] - V[0], V[2] - V[1]
        a1 = np.maximum(0, pp * gg - (1 - pp) * D1).mean(); a2 = np.maximum(0, pp * gg - (1 - pp) * D2).mean()
        return np.array([V[0], V[1] + a1, V[2] + a2])
    step.gg, step.pp = gg, pp
    return step


# HONZA-130 (GPT-6-Astra review) / HONZA-132 (Sol, Astra): the half-inning universe of the count law used to be the
# opportunity frame, so a half-inning with no located, eligible called pitch for either side (189 of 43,132 Statcast
# halves, 8 in extra innings) was missing from the hazards instead of counting as N = 0. FULL_HALVES (default since
# HONZA-132) adds every Statcast 2026 half-inning of the frame's games with n = 0. On a bootstrap resample, which
# renumbers games, the frame's orig_game column (final_ci.resample, joint_boot.resample_frame) maps each resampled
# game to its 2026 game, so a game drawn twice contributes its empty halves twice.
FULL_HALVES = True


def statcast_halves():
    """every 2026 Statcast half-inning: game, inn, top, slot, d (dbin at the start of the half, as add_dstart), cached."""
    global _HALVES
    try:
        return _HALVES
    except NameError:
        pass
    sc = pd.read_parquet("data/statcast_2026.parquet", columns=["game_pk", "inning", "inning_topbot", "at_bat_number",
                                                                "pitch_number", "home_score", "away_score"])
    h = sc.sort_values(["game_pk", "at_bat_number", "pitch_number"]).groupby(["game_pk", "inning", "inning_topbot"]).first().reset_index()
    top = h.inning_topbot.eq("Top").values; inn = h.inning.values.astype(int)
    slot = np.where(inn >= 10, 18, (np.clip(inn, 1, 9) - 1) * 2 + (~top).astype(int))
    d = (np.clip((h.home_score - h.away_score).values.astype(int), -DM, DM) + DM).astype(np.int64)
    d[(inn >= 10) & top] = topx()
    _HALVES = pd.DataFrame(dict(game=h.game_pk.values, inn=inn, top=top, slot=slot, d=d))
    return _HALVES


def solve_dp_score(m, pt, team_home):
    """Delta[slot(19), dbin(ND), k(KMAX), c(3)] for one team side; m must carry dbin (add_dstart on the full frame)."""
    P = half_runs(); Px = extra_runs(); TOPX = topx(); ND = 2 * DM + 1
    g = m.g.values; isht = m.is_home_team.values == team_home
    sl_ = m.slot.values; db = m.dbin.values
    # the team's opportunity count in every observed half-inning (zeros included), by (slot, dbin)
    key = pd.DataFrame(dict(game=m.game_pk.values, inn=m.inning.values, top=m.top.values, slot=sl_, d=db, own=isht))
    hk = key.groupby(["game", "inn", "top"]).agg(slot=("slot", "first"), d=("d", "first"), n=("own", "sum")).reset_index()
    if FULL_HALVES:                                  # half-innings without any opportunity count as n = 0
        orig = m.orig_game.values if "orig_game" in m else m.game_pk.values
        gm = pd.DataFrame(dict(game=m.game_pk.values, orig=orig)).drop_duplicates("game")
        sh = statcast_halves().rename(columns={"game": "orig"}).merge(gm, on="orig")
        miss = sh.merge(hk[["game", "inn", "top"]], on=["game", "inn", "top"], how="left", indicator=True)
        miss = miss[miss._merge == "left_only"].drop(columns="_merge").assign(n=0)
        hk = pd.concat([hk, miss[hk.columns]], ignore_index=True)
    H = {(s, d): hazard(grp.n.values) for (s, d), grp in hk.groupby(["slot", "d"])}
    sel = np.flatnonzero(isht)
    pools = {(int(s), int(d)): (g[idx.values], pt[idx.values]) for (s, d), idx in pd.Series(sel).groupby([sl_[sel], db[sel]])}
    empty = (np.zeros(0), np.zeros(0)); zero = np.zeros(KMAX)
    # extra halves: one pool and one N law for the top halves, one for the bottom halves (all bottom dbins pooled)
    xb = sel[(sl_[sel] == 18) & (db[sel] != TOPX)]
    st_top = stepper(*pools.get((18, TOPX), empty)); st_bot = stepper(g[xb], pt[xb])
    h_top = hazard(hk[(hk.slot == 18) & hk.top].n.values); h_bot = hazard(hk[(hk.slot == 18) & ~hk.top].n.values)
    dvals = np.arange(-DM, DM + 1)
    # extra innings: fixed point of X(c) = Y(topup(c))
    X = np.zeros(3)
    for _ in range(500):
        pxd = lambda d: Px[-d] if -d <= RMAX else 0.0       # P(home ties exactly from -d down)
        B = {d: half_value(st_bot, h_bot, pxd(d) * X)[0] for d in range(-DM, 1)}
        Etop = sum(Px[r] * B[max(-r, -DM)] for r in range(RMAX + 1))
        Y, _ = half_value(st_top, h_top, Etop)
        Xn = np.array([Y[1], Y[1], Y[2]])
        done = np.max(np.abs(Xn - X)) < 1e-13
        X = Xn
        if done:
            break
    Delta = np.zeros((19, ND, KMAX, 3))
    _, Delta[18, TOPX] = half_value(st_top, h_top, Etop)
    for d in range(-DM, 1):
        _, Delta[18, d + DM] = half_value(st_bot, h_bot, (Px[-d] if -d <= RMAX else 0.0) * X)
    Vstart = np.zeros((ND, 3))                      # start of the next slot
    for s in range(17, -1, -1):
        if s == 17:                                  # after the 9th: only a tie goes on, into extra innings
            Vn = np.zeros((ND, 3)); Vn[DM] = X
        else:
            Vn = Vstart.copy()
            if s == 16:                              # after the top of the 9th: home ahead -> no bottom half
                Vn[dvals > 0] = 0.0
        sgn = 1 if s % 2 == 1 else -1                # bottom half: home bats, d rises
        Vend = np.zeros((ND, 3))
        for r in range(RMAX + 1):
            Vend += P[s, r] * Vn[np.clip(dvals + sgn * r, -DM, DM) + DM]
        Vs = np.zeros((ND, 3))
        for d in range(ND):
            Vs[d], Delta[s, d] = half_value(stepper(*pools.get((s, d), empty)), H.get((s, d), zero), Vend[d])
        Vstart = Vs
    return Delta, Vstart
