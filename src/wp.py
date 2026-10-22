"""Count-level win-probability / run-expectancy engine from 2026 Statcast.
Kernel T[count, outs, bases] -> (outs_end, bases_end, runs) to the end of the PA (empirical, 2026 pitches).
WP_home[inn(1..10=extra), half, outs, bases, diff] at PA start by backward induction; extra innings use the
runner-on-2nd rule and a stationary fixed point. Writes data/wp_engine.npz."""
import numpy as np, pandas as pd


SC_COLS = ["game_pk", "at_bat_number", "pitch_number", "inning", "inning_topbot", "outs_when_up", "on_1b", "on_2b", "on_3b",
           "balls", "strikes", "bat_score", "post_bat_score"]


def build_T(by_half=False, sc=None, verbose=True):
    """empirical PA kernel T[count, outs, bases, outs_end, bases_end, runs] from 2026 Statcast; by_half: one kernel
    per batting team [2, ...] (away, home), each shrunk the same way (home-field check); sc: a Statcast
    frame with SC_COLS (e.g. a game-bootstrap resample), default all 2026 pitches."""
    if sc is None:
        sc = pd.read_parquet("data/statcast_2026.parquet", columns=SC_COLS)
    sc = sc.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    sc["bases"] = sc.on_1b.notna() * 1 + sc.on_2b.notna() * 2 + sc.on_3b.notna() * 4
    sc["half"] = (sc.inning_topbot == "Bot").astype(int)
    pa = sc.groupby(["game_pk", "at_bat_number"]).agg(inning=("inning", "first"), half=("half", "first"),
            outs=("outs_when_up", "first"), bases=("bases", "first"), end_score=("post_bat_score", "last")).reset_index()
    nxt = pa.groupby("game_pk").shift(-1)
    same = (nxt.inning == pa.inning) & (nxt.half == pa.half)
    pa["outs_end"] = np.where(same, nxt.outs, 3).astype(int)
    pa["bases_end"] = np.where(same, nxt.bases, 0).astype(int)
    sc = sc.merge(pa[["game_pk", "at_bat_number", "outs_end", "bases_end", "end_score"]], on=["game_pk", "at_bat_number"])
    sc["runs"] = (sc.end_score - sc.bat_score).clip(0, 4).astype(int)
    sc = sc[(sc.balls <= 3) & (sc.strikes <= 2) & (sc.outs_end >= sc.outs_when_up)]
    sc["cnt"] = sc.balls * 3 + sc.strikes
    if by_half:
        return np.stack([_kernel(sc[sc.half == h], verbose) for h in (0, 1)])
    return _kernel(sc, verbose)


def _kernel(sc, verbose=True):
    T = np.zeros((12, 3, 8, 4, 8, 5))
    np.add.at(T, (sc.cnt, sc.outs_when_up, sc.bases, sc.outs_end, sc.bases_end, sc.runs), 1)
    # shrink sparse cells toward the count-pooled-over-bases kernel is awkward; instead pool with 0-0 kernel weight 20
    n = T.sum(axis=(3, 4, 5), keepdims=True)
    T0 = T.sum(axis=0, keepdims=True); T0 = T0 / T0.sum(axis=(3, 4, 5), keepdims=True)
    T = (T + 20 * T0) / (n + 20)
    if verbose: print("min cell n:", int(n.min()), "median:", int(np.median(n)))
    return T


def solve(T, verbose=True, runner=True):
    """WP_home [inn-1, half, outs(4), bases, diff] and RE[outs(4), bases] for a PA kernel T (wp.py layout); T may carry
    a leading half axis [2, ...] (away batting, home batting: a home-field term). runner: extra half-innings
    start with a runner on second (regular season); runner=False for the postseason."""
    b0 = 2 if runner else 0
    Th = (lambda h: T[h]) if T.ndim == 7 else (lambda h: T)
    DMAX = 20; ND = 2 * DMAX + 1
    def clampd(d): return np.clip(d, -DMAX, DMAX)

    def solve_half(Wnext_half_start, inn, half, extra=False):
        """W[outs, bases, diff] at PA start (count 0-0) for one half-inning, given value at the end of the half
        (function of diff). Outs non-decreasing; within same outs, solve by fixed-point iteration."""
        W = np.zeros((4, 8, ND)); W[3] = Wnext_half_start[None, :]
        sgn = 1 if half == 1 else -1
        diffs = np.arange(-DMAX, DMAX + 1)
        for o in (2, 1, 0):
            Wo = np.full((8, ND), 0.5)
            for _ in range(200):
                new = np.zeros_like(Wo)
                for b in range(8):
                    k = Th(half)[0, o, b]  # (outs_end, bases_end, runs)
                    for oe in range(o, 4):
                        for r in range(5):
                            dd = clampd(diffs + sgn * r) + DMAX
                            if oe == o:
                                v = Wo[:, dd]
                            else:
                                v = W[oe][:, dd]
                            if half == 1 and inn >= 9:   # walk-off: home ahead in bottom 9+ ends the game
                                v = np.where((diffs + r > 0)[None, :], 1.0, v)
                            new[b] += (k[oe, :, r][:, None] * v).sum(0)
                if np.max(np.abs(new - Wo)) < 1e-9: Wo = new; break
                Wo = new
            W[o] = Wo
        return W

    diffs = np.arange(-DMAX, DMAX + 1)
    # extra innings (stationary): W_extra_top_start(diff=0) via fixed point
    WP = {}
    def end_after_bottom(diff, cont):  # after a complete bottom half of inning >= 9
        return np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, cont))
    cont = 0.5
    for it in range(60):
        end_bot = end_after_bottom(diffs, cont)
        Wb = solve_half(end_bot, 10, 1)
        start_bot = np.where(diffs > 0, 1.0, Wb[0, b0])  # runner on 2nd (regular season); home ahead -> no bottom
        start_bot = np.where(diffs > 0, 1.0, start_bot)
        Wt = solve_half(start_bot, 10, 0)
        new = Wt[0, b0, DMAX]
        if abs(new - cont) < 1e-10: break
        cont = new
    WP[(10, 0)], WP[(10, 1)] = Wt, Wb
    if verbose: print("home WP at start of extra innings (tie):", round(cont, 4))
    nxt_start = Wt[0, b0]  # value at start of top 10 (runner on 2nd in the regular season), function of diff
    nxt_start = np.where(diffs != 0, (diffs > 0) * 1.0, nxt_start)
    for inn in range(9, 0, -1):
        if inn == 9:
            end_bot = np.where(diffs > 0, 1.0, np.where(diffs < 0, 0.0, nxt_start))
        else:
            end_bot = nxt_start
        Wb = solve_half(end_bot, inn, 1)
        sb = Wb[0, 0]
        if inn == 9: sb = np.where(diffs > 0, 1.0, sb)
        Wt = solve_half(sb, inn, 0)
        WP[(inn, 0)], WP[(inn, 1)] = Wt, Wb
        nxt_start = Wt[0, 0]
    if verbose: print("home WP at first pitch:", round(WP[(1, 0)][0, 0, DMAX], 4))

    # run expectancy RE[outs, bases] at PA start (to end of inning), and by count
    RE = np.zeros((4, 8))
    for o in (2, 1, 0):
        R = np.zeros(8)
        for _ in range(500):
            new = np.zeros(8)
            for b in range(8):
                k = (T if T.ndim == 6 else T.mean(0))[0, o, b]
                for oe in range(o, 4):
                    for r in range(5):
                        new[b] += (k[oe, :, r] * (r + (R if oe == o else RE[oe]))).sum()
            if np.max(abs(new - R)) < 1e-12: break
            R = new
        RE[o] = R
    if verbose: print("RE24 (0 outs):", RE[0].round(3))
    Wall = np.stack([np.stack([WP[(i, h)] for h in (0, 1)]) for i in range(1, 11)])  # [inn-1, half, outs(4), bases, diff]
    return Wall, RE, DMAX


if __name__ == "__main__":
    T = build_T()
    Wall, RE, DMAX = solve(T)
    np.savez("data/wp_engine.npz", T=T, W=Wall, RE=RE, DMAX=DMAX)
    Wp, REp, _ = solve(T, runner=False)                  # Postseason engine, no runner in extra innings
    np.savez("data/wp_engine_post.npz", T=T, W=Wp, RE=REp, DMAX=DMAX)
