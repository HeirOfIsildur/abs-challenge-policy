"""Policy zoo for one decision function shared by the path-sum evaluator (this file) and the causal
re-simulation (causal.py), plus the two public competitors rebuilt on our calibrated DP.

Policies (code: rule; p = calibrated posterior pt, g = WP stake of a flip, RV = run stake, c = challenges held):
  0 opt       p(g + D) >= D, D = Delta[slot, k, c] from sim.solve_dp
  1 savant    p >= 0.2 / (0.2 + RV)                          (Savant/Tango breakeven, runs)
  2 stake     p >= d / (g + d), constant d in WP               (our one-line rule)
  3 conf50    p >= 0.5
  4 obs_rule  s > kappa                                        (fitted behavioural rule)
  5 card      p > card[inning band, g tercile, count class, c] (Arora's 48-threshold card)
  6 runs_dp   p RV > (1 - p) C[slot, c], C = run-valued token value at the start of the half-inning
              (Namuduri: same DP, objective = runs, half-inning-level continuation value)
  7 opt_score p(g + D) >= D, D = Delta[slot, d, k, c] from dp_score.solve_dp_score (score-aware optimum)
Arora's card (github.com/AyanArora29/use-it-or-lose-it, code/engine/dp.py::build_card): inning bands 1-3/4-6/7-8/9+,
g terciles over all opportunities, count class "PA-ending" iff 3 balls or 2 strikes, tokens 1/2; each cell holds the
median break-even p* = D/(g + D) over the opportunities in it.  We build it two ways: "card_refit" takes p* from our
DP (the card as a rule class, re-estimated on our model), "card_pub" uses Arora's published 2026 thresholds
(results/2026-09-28/tier1_card_2026.csv) with our g terciles.
Namuduri (github.com/Cowshal/mlb-abs-challenge-analysis, src/abs_policy.py): challenge iff p dRE > (1-p) C(t, k),
C from a run-objective DP evaluated at the start of half-inning t; rebuilt here with sim.solve_dp on RV."""
import numpy as np, pandas as pd
from numba import njit

POL = dict(opt=0, savant=1, stake=2, conf50=3, obs_rule=4, card=5, runs_dp=6, opt_score=7)
CARD_PUB = {  # Arora, tier1_card_2026.csv (2026-09-28): (inn_band, lev, cnt) -> (pstar_t1, pstar_t2)
    ("1-3", "high", 1): (0.3360, 0.2161), ("1-3", "high", 0): (0.4649, 0.3200), ("1-3", "low", 1): (0.7550, 0.6370),
    ("1-3", "low", 0): (0.7778, 0.6575), ("1-3", "mid", 1): (0.5686, 0.4197), ("1-3", "mid", 0): (0.6138, 0.4650),
    ("4-6", "high", 1): (0.2748, 0.1646), ("4-6", "high", 0): (0.4025, 0.2575), ("4-6", "low", 1): (0.6632, 0.5210),
    ("4-6", "low", 0): (0.7400, 0.6044), ("4-6", "mid", 1): (0.4684, 0.3122), ("4-6", "mid", 0): (0.5627, 0.4013),
    ("7-8", "high", 1): (0.1795, 0.0905), ("7-8", "high", 0): (0.3087, 0.1683), ("7-8", "low", 1): (0.5747, 0.4046),
    ("7-8", "low", 0): (0.6843, 0.5216), ("7-8", "mid", 1): (0.3505, 0.2037), ("7-8", "mid", 0): (0.4596, 0.2880),
    ("9+", "high", 1): (0.0800, 0.0320), ("9+", "high", 0): (0.1687, 0.0793), ("9+", "low", 1): (0.4975, 0.2809),
    ("9+", "low", 0): (0.5889, 0.3685), ("9+", "mid", 1): (0.1914, 0.0603), ("9+", "mid", 0): (0.2877, 0.1455)}
BANDS = ["1-3", "4-6", "7-8", "9+"]; LEVS = ["low", "mid", "high"]


@njit(cache=True)
def inn_band(inn):
    return 0 if inn <= 3 else (1 if inn <= 6 else (2 if inn <= 8 else 3))


@njit(cache=True)
def decide(pol, pt, s, kappa, g, RV, slot, kin, c, inn, balls, strikes, D, Cn, card, q1, q2, dconst, D4, dbin):
    if pt <= 0.0:
        return False
    if pol == 7:
        d = D4[slot, dbin, min(kin, D4.shape[2] - 1), min(c, 2)]
        return pt * (g + d) >= d
    if pol == 0:
        d = D[slot, min(kin, D.shape[1] - 1), min(c, 2)]
        return pt * (g + d) >= d
    if pol == 1:                                  # Savant: fixed run cost 0.2, or a tuned constant (dconst > 0)
        cr = dconst if dconst > 0.0 else 0.2
        return pt >= cr / (cr + RV)
    if pol == 2:
        return pt >= dconst / (g + dconst)
    if pol == 3:
        return pt >= 0.5
    if pol == 4:
        return s > kappa
    if pol == 5:
        lev = 0 if g < q1 else (1 if g < q2 else 2)
        cnt = 1 if (balls == 3 or strikes == 2) else 0
        return pt > card[inn_band(inn), lev, cnt, min(c, 2) - 1]
    C = Cn[slot, min(c, 2)]
    return pt * RV > (1.0 - pt) * C


@njit(cache=True)
def run_path(gidx, is_home_team, new_extra, slot, kin, inning, balls, strikes, dbin, W, pt, s, g, RV, kappa, isoff,
             pol, Dh, Da, Cnh, Cna, card, q1, q2, dconst, D4h, D4a, ngames):
    """run_path2 with the flips scored at the stake the policy decides on (gv = g)."""
    return run_path2(gidx, is_home_team, new_extra, slot, kin, inning, balls, strikes, dbin, W, pt, s, g, g, RV, kappa,
                     isoff, pol, Dh, Da, Cnh, Cna, card, q1, q2, dconst, D4h, D4a, ngames)


@njit(cache=True)
def run_path2(gidx, is_home_team, new_extra, slot, kin, inning, balls, strikes, dbin, W, pt, s, g, gv, RV, kappa, isoff,
              pol, Dh, Da, Cnh, Cna, card, q1, q2, dconst, D4h, D4a, ngames):
    """g is the stake the policy DECIDES on, gv the stake a successful flip is SCORED at (cross-fitted
    stakes: g from the engine fitted without the pitch's game fold, gv from the engine of its own fold).
    Path-sum evaluation on the recorded stream (same accounting as sim.run_policy). kappa: [n, 3], the fitted
    criterion by challenges held (sim.kappa_by_count), so obs_rule uses the count the simulated team holds.
    Returns value/challenges/successes [game, team(0 away, 1 home), side(0 def, 1 off)] and, per (game, team),
    the row of the team's first successful challenge (-1 if none) and the challenges it then held -- the branch point
    for causal.py."""
    val = np.zeros((ngames, 2, 2)); nch = np.zeros((ngames, 2, 2)); nwin = np.zeros((ngames, 2, 2))
    first = np.full((ngames, 2), -1, np.int64); firstc = np.zeros((ngames, 2), np.int64)
    c = np.array([2, 2]); cur = -1
    for i in range(len(g)):
        if gidx[i] != cur:
            cur = gidx[i]; c[0] = 2; c[1] = 2
        if new_extra[i]:
            for t in range(2):
                if c[t] == 0: c[t] = 1
        t = 1 if is_home_team[i] else 0
        if c[t] == 0:
            continue
        D = Dh if t == 1 else Da; Cn = Cnh if t == 1 else Cna; D4 = D4h if t == 1 else D4a
        if decide(pol, pt[i], s[i], kappa[i, min(c[t], 2)], g[i], RV[i], slot[i], kin[i], c[t], inning[i], balls[i], strikes[i],
                  D, Cn, card, q1, q2, dconst, D4, dbin[i]):
            sd = 1 if isoff[i] else 0
            nch[cur, t, sd] += 1
            if W[i]:
                val[cur, t, sd] += gv[i]; nwin[cur, t, sd] += 1
                if first[cur, t] < 0:
                    first[cur, t] = i; firstc[cur, t] = c[t]
            else:
                c[t] -= 1
    return val, nch, nwin, first, firstc


def card_cells(m):
    """cell coordinates of each row: inning band, g tercile (cut points q1, q2), count class."""
    q1, q2 = np.quantile(m.g.values, [1 / 3, 2 / 3])
    ib = np.select([m.inning.values <= 3, m.inning.values <= 6, m.inning.values <= 8], [0, 1, 2], 3)
    lev = np.where(m.g.values < q1, 0, np.where(m.g.values < q2, 1, 2))
    cnt = ((m.balls.values == 3) | (m.strikes.values == 2)).astype(int)
    return ib, lev, cnt, float(q1), float(q2)


def card_refit(m, kin, D4h, D4a):
    """Arora's card rebuilt on our score-aware DP: cell median of p* = D/(g + D) for c = 1, 2.
    Array [band, lev, cnt, c-1]."""
    ib, lev, cnt, q1, q2 = card_cells(m)
    hm = m.is_home_team.values; sl = m.slot.values; db = m.dbin.values; k = np.minimum(kin, D4h.shape[2] - 1)
    card = np.full((4, 3, 2, 2), 0.5)
    for c in (1, 2):
        D = np.where(hm, D4h[sl, db, k, c], D4a[sl, db, k, c])
        ps = np.where(m.g.values > 0, D / (m.g.values + D), 1.0)
        df = pd.DataFrame(dict(ib=ib, lev=lev, cnt=cnt, ps=ps)).groupby(["ib", "lev", "cnt"]).ps.median()
        for (a, b, e), v in df.items():
            card[a, b, e, c - 1] = v
    return card, q1, q2


def card_published(m):
    _, _, _, q1, q2 = card_cells(m)
    card = np.full((4, 3, 2, 2), 0.5)
    for (bd, lv, cn), (p1, p2) in CARD_PUB.items():
        card[BANDS.index(bd), LEVS.index(lv), cn] = (p1, p2)
    return card, q1, q2


def runs_token_values(m, pt, solve_dp):
    """Namuduri's C(t, c): run-valued marginal token value from the same DP with g -> RV, taken at the start of
    each half-inning (k = 0) and applied to every opportunity in it."""
    mr = m.copy(); mr["g"] = m.RV.values
    out = []
    for home in (True, False):
        D, _ = solve_dp(mr, pt, home)
        out.append(np.ascontiguousarray(D[:, 0, :]))
    return out[0], out[1]


@njit(cache=True)
def clairvoyant(gidx, is_home_team, topup, W, pt, g, ngames):
    """clairvoyant2 with the flips scored at the decision stake."""
    return clairvoyant2(gidx, is_home_team, topup, W, pt, g, g, ngames)


@njit(cache=True)
def clairvoyant2(gidx, is_home_team, topup, W, pt, g, gv, ngames):
    """a clairvoyant comparator for the DP benchmark. For every team-game the policy
    knows the whole future sequence of (p, g) of its own opportunities (not the truths) and solves the exact
    finite-horizon DP over it, with the retention rule and the extra-inning top-up; it is then played on the realised
    truths. It is NOT a certified upper bound for the benchmark: the benchmark's pooled DP and the
    recorded stream are different processes, and the posteriors p are only calibrated within cells. topup[i]: an
    extra inning started since the team's previous opportunity (a team at 0 gets 1). Rows must be in game order.
    Returns value/challenges [game, team]."""
    n = len(g)
    val = np.zeros((ngames, 2)); nch = np.zeros((ngames, 2))
    idx = np.empty(n, np.int64)
    i0 = 0
    while i0 < n:
        i1 = i0
        while i1 < n and gidx[i1] == gidx[i0]:
            i1 += 1
        for t in range(2):
            m = 0
            for i in range(i0, i1):
                if (1 if is_home_team[i] else 0) == t:
                    idx[m] = i; m += 1
            # backward: V[j, c] = value before opportunity j holding c (top-up applied first)
            V = np.zeros((m + 1, 3)); go = np.zeros((m, 3), np.bool_)
            for j in range(m - 1, -1, -1):
                i = idx[j]; U = np.zeros(3)
                U[0] = V[j + 1, 0]
                for c in range(1, 3):
                    ch = pt[i] * g[i] + pt[i] * V[j + 1, c] + (1 - pt[i]) * V[j + 1, c - 1]
                    go[j, c] = ch > V[j + 1, c]
                    U[c] = ch if go[j, c] else V[j + 1, c]
                for c in range(3):
                    cc = 1 if (topup[i] and c == 0) else c
                    V[j, c] = U[cc]
            c = 2
            for j in range(m):
                i = idx[j]
                if topup[i] and c == 0:
                    c = 1
                if c > 0 and go[j, c]:
                    nch[gidx[i], t] += 1
                    if W[i]:
                        val[gidx[i], t] += gv[i]
                    else:
                        c -= 1
        i0 = i1
    return val, nch


def topup_flags(m):
    """per row: an extra inning started since the same team's previous opportunity in the game (or this is the team's
    first opportunity and it is in extra innings)."""
    t = m.is_home_team.values.astype(int)
    prev = pd.Series(m.inning.values).groupby([m.game_pk.values, t]).shift(1).values
    return (m.inning.values >= 10) & (np.isnan(prev) | (m.inning.values > np.nan_to_num(prev)))


@njit(cache=True)
def first_divergence(gidx, is_home_team, new_extra, slot, kin, inning, balls, strikes, dbin, W, pt, s, g, RV, kappa,
                     pol, Dh, Da, Cnh, Cna, card, q1, q2, dconst, D4h, D4a, recsucc, ngames):
    """the first pitch at which the policy's game and the recorded game part:
    the first row where exactly one of {the policy flips the call, the team flipped it in 2026} holds. Before it the
    two paths coincide, so the policy is evaluated on the recorded path exactly. Returns per (game, team): the row
    (-1: never), its type (+1 the policy flips a call the team left standing, -1 the team flipped a call the policy
    leaves standing), the challenges the policy then holds, and the value of the matched flips before it."""
    div = np.full((ngames, 2), -1, np.int64); typ = np.zeros((ngames, 2), np.int64); dc = np.zeros((ngames, 2), np.int64)
    pre = np.zeros((ngames, 2))
    c = np.array([2, 2]); cur = -1
    for i in range(len(g)):
        if gidx[i] != cur:
            cur = gidx[i]; c[0] = 2; c[1] = 2
        if new_extra[i]:
            for t in range(2):
                if c[t] == 0: c[t] = 1
        t = 1 if is_home_team[i] else 0
        if div[cur, t] >= 0:
            continue
        go = False
        if c[t] > 0:
            D = Dh if t == 1 else Da; Cn = Cnh if t == 1 else Cna; D4 = D4h if t == 1 else D4a
            go = decide(pol, pt[i], s[i], kappa[i, min(c[t], 2)], g[i], RV[i], slot[i], kin[i], c[t], inning[i], balls[i],
                        strikes[i], D, Cn, card, q1, q2, dconst, D4, dbin[i])
        ps = go and W[i]
        if ps != recsucc[i]:
            div[cur, t] = i; typ[cur, t] = 1 if ps else -1; dc[cur, t] = c[t]
            continue
        if ps:
            pre[cur, t] += g[i]
        elif go:
            c[t] -= 1
    return div, typ, dc, pre


@njit(cache=True)
def path_trace(gidx, is_home_team, new_extra, slot, kin, inning, balls, strikes, dbin, W, pt, s, g, RV, kappa,
               pol, Dh, Da, Cnh, Cna, card, q1, q2, dconst, D4h, D4a):
    """one policy played along the stream, row by row: succ[i] (the row's team flips the
    call), fail[i] (it challenges and loses) and held[i, t] (challenges team t holds before row i, top-up applied).
    The paired causal contrast (causal.py) branches the policy against the SIMULATED behavioural rule at the first row
    where their successes differ, with both inventories carried from these traces."""
    n = len(g)
    succ = np.zeros(n, np.bool_); fail = np.zeros(n, np.bool_); held = np.zeros((n, 2), np.int64)
    c = np.array([2, 2]); cur = -1
    for i in range(n):
        if gidx[i] != cur:
            cur = gidx[i]; c[0] = 2; c[1] = 2
        if new_extra[i]:
            for t in range(2):
                if c[t] == 0: c[t] = 1
        held[i, 0] = c[0]; held[i, 1] = c[1]
        t = 1 if is_home_team[i] else 0
        if c[t] == 0:
            continue
        D = Dh if t == 1 else Da; Cn = Cnh if t == 1 else Cna; D4 = D4h if t == 1 else D4a
        if decide(pol, pt[i], s[i], kappa[i, min(c[t], 2)], g[i], RV[i], slot[i], kin[i], c[t], inning[i], balls[i],
                  strikes[i], D, Cn, card, q1, q2, dconst, D4, dbin[i]):
            if W[i]:
                succ[i] = True
            else:
                fail[i] = True; c[t] -= 1
    return succ, fail, held
