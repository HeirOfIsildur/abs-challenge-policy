"""Stake of each called pitch: g = |WP_home(after ball) - WP_home(after strike)| (WP swing of a flip, same for
either challenger), and RV = |RE(after ball) - RE(after strike)| in runs (Savant/RE288-style)."""
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

def add_stakes(m):
    """g = |WP_home(after ball) - WP_home(after strike)| and RV in runs for every row of m (pre-pitch state)."""
    m = m.copy()
    m["half"] = (~m.top).astype(int)
    m["diff"] = m.score_diff_home.astype(int)
    keys = ["inning", "half", "outs_when_up", "bases", "diff", "balls", "strikes"]
    u = m[keys].drop_duplicates().reset_index(drop=True)
    res = []
    for inn, h, o, b, d, bl, st in u.itertuples(index=False):
        wb, rb = Vcount(inn, h, o, b, d, bl + 1, st)
        ws, rs = Vcount(inn, h, o, b, d, bl, st + 1)
        res.append((wb, ws, rb, rs))
    u[["wp_ball", "wp_strike", "re_ball", "re_strike"]] = np.array(res)
    m = m.merge(u, on=keys)
    # Signed stakes oriented to the batting team. The empirical count kernels can make a
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
