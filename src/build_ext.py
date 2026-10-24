"""Called-pitch tables for the external validation feeds (Triple-A 2025, MLB spring 2026), in the same layout as
data/called_2026_w.parquet so that lik.prepare / sim can use them.  Geometry and the challenges-left tracking
follow build.py; stakes g use the 2026 MLB WP engine (stakes.add_stakes), an approximation for Triple-A.
usage: python src/build_ext.py <tag>  -> data/called_<tag>_w.parquet"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from stakes import add_stakes
import geometry as G


def challenges_left(f):
    """start 2 per team, -1 on a failed challenge, a team at 0 gets 1 at the start of each extra inning (build.py)."""
    f = f.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    lh, la = [], []
    for pk, gdf in f.groupby("game_pk", sort=False):
        L = {"h": 2, "a": 2}; hid = gdf.home_id.iat[0]; last_inn = 0
        for inn, top, ch, over, team in zip(gdf.inning, gdf.top, gdf.ch, gdf.ch_over, gdf.ch_team):
            if inn >= 10 and top and inn != last_inn:
                for k in L:
                    if L[k] == 0:
                        L[k] = 1
            last_inn = inn
            lh.append(L["h"]); la.append(L["a"])
            if ch and not over:
                L["h" if team == hid else "a"] -= 1
    f["left_home"] = lh; f["left_away"] = la
    return f


def geometry(m):
    """signed distance (in) to the ABS zone with the ball radius, nearest edge. Stats API feed pX/pZ are at the front
    of the plate, so they are projected to the midpoint (src/geometry.py, results/geometry_audit.md)."""
    m["px_mid"], m["pz_mid"] = G.abs_location(m, "feed")
    m["dist_in"], m["edge"] = G.signed_distance(m.px_mid, m.pz_mid, m.sz_top, m.sz_bot)
    return m


if __name__ == "__main__":
    tag = sys.argv[1]
    f = pd.read_parquet(f"data/ext_{tag}.parquet"); g = pd.read_parquet(f"data/ext_games_{tag}.parquet")
    f = f.merge(g[["game_pk", "date", "home_id", "away_id", "has"]], on="game_pk")
    f = f[f.has.notna()]            # HONZA-89: has = False is an ABS game without a challenge; only None means no ABS
    f = challenges_left(f)
    f["left_ch"] = np.where(f.ch_team == f.home_id, f.left_home, f.left_away)
    print(tag, "challenges made with 0 left:", int(((f.left_ch <= 0) & f.ch).sum()), "of", int(f.ch.sum()))
    m = f[f.code.isin(["C", "B", "*B"]) & f.plate_x.notna() & f.sz_top.notna() & f.vy0.notna()].copy()
    m["final_strike"] = m.code.eq("C")
    m["orig_strike"] = np.where(m.ch_over, ~m.final_strike, m.final_strike)
    m = geometry(m)
    m["score_diff_home"] = m.home_score - m.away_score
    m["inning"] = m.inning.astype(int); m["bases"] = m.bases.astype(int); m["outs_when_up"] = m.outs_when_up.astype(int)
    m["balls"] = m.balls.clip(0, 3).astype(int); m["strikes"] = m.strikes.clip(0, 2).astype(int)
    m = add_stakes(m)
    m["side"] = np.where(m.orig_strike, "off", "def")
    m["left"] = np.where(m.orig_strike, np.where(m.top, m.left_away, m.left_home), np.where(m.top, m.left_home, m.left_away))
    m["w"] = np.where(m.orig_strike, -m.dist_in, m.dist_in)        # preliminary (mu = 0), only for the |w| < 8 filter
    m["fielder_2"] = -1
    m = m.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    print(tag, "called pitches:", len(m), "challenged:", int(m.ch.sum()), "overturn rate:", round(m[m.ch].ch_over.mean(), 3),
          "| batter side challenges:", int((m.ch & m.orig_strike).sum()), "fielder side:", int((m.ch & ~m.orig_strike).sum()))
    m.to_parquet(f"data/called_{tag}_w.parquet")
