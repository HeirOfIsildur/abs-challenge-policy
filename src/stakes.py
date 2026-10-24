"""Stake of each called pitch: g = max(s (WP_home(after ball) - WP_home(after strike)), 0) with s = +1 when the home
team bats and -1 otherwise -- the batting team's WP swing of a flip, signed, with the non-monotone engine cells set to
zero (HONZA-122) -- the same for either challenger; RV = max(RE(after ball) - RE(after strike), 0) in runs."""
import numpy as np, pandas as pd
z = np.load("data/wp_engine.npz"); T, W, RE, D = z["T"], z["W"], z["RE"], int(z["DMAX"])


def use_engine(path):
    """switch the WP/RE engine (e.g. data/wp_engine_post.npz: extra innings without the runner on second)."""
    global T, W, RE, D
    z = np.load(path); T, W, RE, D = z["T"], z["W"], z["RE"], int(z["DMAX"])
WALK = {}
for b in range(8):  # forced advance on a walk
    f1, f2, f3 = b & 1, b & 2, b & 4
    if not f1: WALK[b] = (b | 1, 0)
    elif not f2: WALK[b] = (b | 3, 0)
    elif not f3: WALK[b] = (7, 0)
    else: WALK[b] = (7, 1)

def Wpa(inn, half, o, b, diff):
    """home WP at PA start (o may be 3 = half over); walk-off for bottom 9+"""
    if half == 1 and inn >= 9 and diff > 0: return 1.0
    i = min(inn, 10) - 1
    return W[i, half, o, b if o < 3 else 0, np.clip(diff, -D, D) + D]

def REpa(o, b):
    return 0.0 if o >= 3 else RE[o, b]

def Vcount(inn, half, o, b, diff, balls, strikes):
    if balls == 4:
        nb, r = WALK[b]; sg = 1 if half else -1
        return Wpa(inn, half, o, nb, diff + sg * r), r + REpa(o, nb)
    if strikes == 3:
        return Wpa(inn, half, o + 1, b, diff), REpa(o + 1, b)
    k = T[balls * 3 + strikes, o, b]; sg = 1 if half else -1
    i = min(inn, 10) - 1; wv = 0.0; rv = 0.0
    for oe in range(o, 4):
        for r in range(5):
            dd = diff + sg * r
            if half == 1 and inn >= 9:
                col = np.full(8, 1.0) if dd > 0 else W[i, half, oe, :, np.clip(dd, -D, D) + D]
            else:
                col = W[i, half, oe, :, np.clip(dd, -D, D) + D]
            if oe == 3: col = np.full(8, col[0])
            wv += (k[oe, :, r] * col).sum()
            rv += (k[oe, :, r] * (r + (RE[oe] if oe < 3 else 0))).sum()
    return wv, rv

def state_table():
    """HONZA-132: Vcount for EVERY state at once (vectorised; the per-state loop of add_stakes took ~19 s per engine,
    which the cross-fitted bootstrap calls ten times per replicate). Returns (WV, RV) of shape
    [inn 1..10 (10 = extras), half, outs 0..2, bases, diff in [-(D+4), D+4], balls 0..4, strikes 0..3], exactly
    Vcount(inn, half, o, b, diff, balls, strikes) for balls < 4 or strikes < 3 (the score axis is extended by the 4-run
    maximum of a plate appearance, so clipping the input diff to +-(D+4) is exact)."""
    DX = D + 4; diffs = np.arange(-DX, DX + 1); ND = len(diffs)
    WV = np.zeros((10, 2, 3, 8, ND, 5, 4)); RVt = np.zeros_like(WV)
    REx = np.vstack([RE[:3], np.zeros((1, 8))])                       # RE at outs 3 = 0
    K = T[:12].reshape(4, 3, 3, 8, 4, 8, 5)                           # [balls, strikes, o, b, oe, be, r]
    rr = np.arange(5)
    RVk = np.einsum("lsobxyr,xyr->lsob", K, rr[None, None, :] + REx[:, :, None])
    for i in range(10):
        inn = i + 1
        for h in range(2):
            sg = 1 if h else -1
            walkoff = (h == 1 and inn >= 9)
            dd = diffs[:, None] + sg * rr[None, :]                     # [ND, r]
            Wc = W[i, h][:, :, np.clip(dd, -D, D) + D]                 # [oe 4, be 8, ND, r]
            Wc[3] = Wc[3, 0][None]                                     # half over: base state irrelevant
            if walkoff:
                Wc = np.where((dd > 0)[None, None], 1.0, Wc)
            WV[i, h, :, :, :, :4, :3] = np.einsum("lsobxyr,xydr->obdls", K, Wc)
            RVt[i, h, :, :, :, :4, :3] = np.broadcast_to(np.transpose(RVk, (2, 3, 0, 1))[:, :, None], (3, 8, ND, 4, 3))
            def Wpa(o, b, d):
                v = W[i, h, o, (b if o < 3 else 0), np.clip(d, -D, D) + D]
                return np.where(d > 0, 1.0, v) if walkoff else v
            for o in range(3):
                for b in range(8):
                    nb, r = WALK[b]
                    WV[i, h, o, b, :, 4, :] = Wpa(o, nb, diffs + sg * r)[:, None]
                    RVt[i, h, o, b, :, 4, :] = r + RE[o, nb]
                    WV[i, h, o, b, :, :, 3] = (Wpa(o + 1, b, diffs) if o + 1 < 3 else Wpa(3, 0, diffs))[:, None]
                    RVt[i, h, o, b, :, :, 3] = RE[o + 1, b] if o + 1 < 3 else 0.0
    return WV, RVt


def add_stakes(m):
    """g = |WP_home(after ball) - WP_home(after strike)| and RV in runs for every row of m (pre-pitch state)."""
    m = m.copy()
    m["half"] = (~m.top).astype(int)
    m["diff"] = m.score_diff_home.astype(int)
    keys = ["inning", "half", "outs_when_up", "bases", "diff", "balls", "strikes"]
    m[keys] = m[keys].astype(int)        # HONZA-132: state columns can arrive as float (unmatched rows of the join)
    WV, RVt = state_table()              # HONZA-132: vectorised (was a loop of Vcount calls over the unique states)
    ii = np.clip(m.inning.values, 1, 10) - 1; hh = m.half.values; oo = m.outs_when_up.values; bb = m.bases.values
    dd = np.clip(m["diff"].values, -(D + 4), D + 4) + D + 4; bl = m.balls.values; st = m.strikes.values
    m["wp_ball"] = WV[ii, hh, oo, bb, dd, bl + 1, st]; m["wp_strike"] = WV[ii, hh, oo, bb, dd, bl, st + 1]
    m["re_ball"] = RVt[ii, hh, oo, bb, dd, bl + 1, st]; m["re_strike"] = RVt[ii, hh, oo, bb, dd, bl, st + 1]
    # HONZA-122 (Astra, Sol reviews): signed stakes oriented to the batting team. The empirical count kernels can make a
    # ball worth less than a strike to the batting team (a non-monotone engine cell); the absolute value turned those
    # into positive gains. A flip is now worth max(signed swing, 0); g_signed keeps the raw contrast for the audit.
    sg = np.where(m.half.values == 1, 1.0, -1.0)
    m["g_signed"] = sg * (m.wp_ball - m.wp_strike)
    m["g"] = m.g_signed.clip(lower=0.0)
    m["RV"] = (m.re_ball - m.re_strike).clip(lower=0.0)
    m["wp_home_now"] = np.nan
    return m


if __name__ == "__main__":
    m = pd.read_parquet("data/called_2026.parquet")
    m = add_stakes(m[m.dist_in.notna()])
    print("non-monotone engine cells (ball worse than strike for the batting team):", int((m.g_signed < 0).sum()),
          "rows, min", round(float(m.g_signed.min()), 5), "| challenged:", int((m.ch & (m.g_signed < 0)).sum()),
          "| negative RV contrasts:", int((m.re_ball < m.re_strike).sum()))
    print(m[["g", "RV"]].describe(percentiles=[.1, .5, .9, .99]).round(4))
    print("challenged pitches g:", m[m.ch].g.describe(percentiles=[.25, .5, .75]).round(4).to_dict())
    print("check vs Statcast delta_run_exp on called pitches (corr |.|):", np.corrcoef(m.RV, m.delta_run_exp.abs().fillna(0))[0, 1].round(3))
    m.to_parquet("data/called_2026_g.parquet")
