"""Build the 2026 called-pitch table: feed (challenges) x Statcast (location, state), with the original call,
challenger role, ABS-zone signed distance from raw Statcast 2026 plate_x/plate_z (already at the plate midpoint, src/geometry.py),
and challenges left per team before each pitch.
usage: python build.py <datadir>   -> <datadir>/called_2026.parquet, <datadir>/pa_2026.parquet"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import geometry as G

D = Path(sys.argv[1])
f = pd.read_parquet(D / "feed_pitches_2026.parquet")
g = pd.read_parquet(D / "feed_games_2026.parquet")
sc = pd.read_parquet(D / "statcast_2026.parquet")
f["ch"] = f.ch.fillna(False).astype(bool)
f["ch_over"] = f.ch_over.fillna(False).astype(bool)
# GameData.absChallenges is absent (has = None) in 4 games at venues without ABS (Mexico City series
# 25-26 Apr, Field of Dreams 13 Aug, Little League Classic 23 Aug); has = False marks ABS games without a challenge.
# Keeping the 4 no-ABS games counted 8 team-games of "never challenge" against the observed policy.
no_abs = set(g.game_pk[g.has.isna()])
print("games without ABS (dropped):", sorted(no_abs))
g = g[~g.game_pk.isin(no_abs)]
f = f.merge(g[["game_pk", "home_id", "away_id"]], on="game_pk")
f["bat_id"] = np.where(f.top, f.away_id, f.home_id)
f["fld_id"] = np.where(f.top, f.home_id, f.away_id)

# --- challenges left per team before each pitch (start 2, -1 on failure; extra-inning top-up inferred) ---
f = f.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
left_h, left_a, rows = [], [], []
for pk, gdf in f.groupby("game_pk", sort=False):
    L = {"h": 2, "a": 2}; hid = gdf.home_id.iat[0]
    last_inn = 0
    for inn, top, ch, over, team in zip(gdf.inning, gdf.top, gdf.ch, gdf.ch_over, gdf.ch_team):
        if inn >= 10 and top and inn != last_inn:      # start of each extra inning
            for k in L:
                if L[k] == 0:
                    L[k] = 1
        last_inn = inn
        left_h.append(L["h"]); left_a.append(L["a"])
        if ch and not over:
            k = "h" if team == hid else "a"
            L[k] -= 1
f["left_home"] = left_h; f["left_away"] = left_a
f["left_ch"] = np.where(f.ch_team == f.home_id, f.left_home, f.left_away)
print("challenges made with 0 left (rule check):", int(((f.left_ch <= 0) & f.ch).sum()), "of", int(f.ch.sum()))
print("extra-inning challenges:", int((f.ch & (f.inning >= 10)).sum()))

# --- join Statcast ---
keep = ["game_pk", "at_bat_number", "pitch_number", "plate_x", "plate_z", "sz_top", "sz_bot", "vx0", "vy0", "vz0",
        "ax", "ay", "az", "stand", "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b", "home_score",
        "away_score", "fielder_2", "umpire", "description", "delta_run_exp", "home_win_exp", "bat_win_exp"]
m = f.merge(sc[keep], on=["game_pk", "at_bat_number", "pitch_number"], how="left", indicator=True)
print("feed pitches matched to Statcast:", (m._merge == "both").mean().round(4))
m = m.drop(columns="_merge")

# --- final and original call ---
m["final_strike"] = m.code.eq("C")
called = m.code.isin(["C", "B", "*B"])
m = m[called].copy()
m["orig_strike"] = np.where(m.ch_over, ~m.final_strike, m.final_strike)
# The feed-Statcast join on (game, at-bat, pitch number) attaches the wrong Statcast pitch
# when the two sources number pitches differently; the Statcast description then contradicts the feed's call code
# (e.g. "foul" on a called ball). Statcast reports the final call, so an overturned pitch agrees with the feed code.
m["misjoin"] = ~((m.code.eq("C") & m.description.eq("called_strike"))
                 | (m.code.isin(["B", "*B"]) & m.description.isin(["ball", "blocked_ball"])))
# role of the challenger
m["role"] = np.select([~m.ch, m.ch_player == m.batter, m.ch_player == m.pitcher, m.ch_player == m.fielder_2],
                      [None, "batter", "pitcher", "catcher"], "other")
print(m[m.ch].role.value_counts())
print("batters challenge strikes only:", m[m.ch & (m.role == "batter")].orig_strike.mean().round(4),
      "| fielders challenge balls only:", 1 - m[m.ch & m.role.isin(["catcher", "pitcher"])].orig_strike.mean().round(4))

# --- ABS zone geometry (src/geometry.py). Statcast 2026 plate_x/plate_z are already in the ABS plane (plate midpoint):
# no projection. The geometry fix removed a front->midpoint projection that lowered every pitch by 0.98 in (results/geometry_audit.md).
assert G.SOURCE_PLANE_Y_FT["statcast2026"] == G.Y_MID_FT
m["px_mid"], m["pz_mid"] = G.abs_location(m, "statcast2026")      # = plate_x, plate_z (names kept for lik.region)
m["dist_in"], m["edge"] = G.signed_distance(m.px_mid, m.pz_mid, m.sz_top, m.sz_bot)
m.loc[m.plate_x.isna() | m.sz_top.isna(), "dist_in"] = np.nan
m.loc[m.misjoin, "dist_in"] = np.nan        # A mis-joined row carries another pitch's location
m["mid_shift_z_in"] = 12 * (m.pz_mid - m.plate_z)                  # 0 by construction (guarded in tests/test_geometry.py)

# count BEFORE the pitch: Statcast balls/strikes are pre-pitch
m["bases"] = m.on_1b.notna().astype(int) + 2 * m.on_2b.notna().astype(int) + 4 * m.on_3b.notna().astype(int)
m["score_diff_home"] = m.home_score - m.away_score

# MLB's challenge-opportunity definition excludes pitches thrown by position players.
# A position player pitching = a pitcher with >= 30 plate appearances as a batter and < 400 pitches in 2026 (54
# players, at most 159 pitches each; the only two-way regular, with 1,335 pitches, stays a pitcher).
npit = sc.groupby("pitcher").size()
npa = sc.drop_duplicates(["game_pk", "at_bat_number"]).groupby("batter").size()
pos_p = set(npit.index[(npa.reindex(npit.index).fillna(0).values >= 30) & (npit.values < 400)])
m["pos_pitcher"] = m.pitcher.isin(pos_p)
# Home score differential at the START of the half-inning (first pitch of the half in
# Statcast, called or not); the score at the first called pitch can already include runs scored earlier in the half
sh = sc.sort_values(["game_pk", "at_bat_number", "pitch_number"]).groupby(["game_pk", "inning", "inning_topbot"]).first()
sh = (sh.home_score - sh.away_score).rename("d_half_start").reset_index()
sh["top"] = sh.inning_topbot.eq("Top")
m = m.merge(sh[["game_pk", "inning", "top", "d_half_start"]], on=["game_pk", "inning", "top"], how="left")
m["d_half_start"] = m.d_half_start.fillna(m.score_diff_home).astype(int)
fc = m.sort_values(["game_pk", "at_bat_number", "pitch_number"]).groupby(["game_pk", "inning", "top"]).first()
print("mis-joined rows:", int(m.misjoin.sum()), "| position-player pitches:", int(m.pos_pitcher.sum()),
      "| half-innings whose first called pitch is not at the half's start score:",
      int((fc.score_diff_home != fc.d_half_start).sum()), "of", len(fc))
m.to_parquet(D / "called_2026.parquet")
print("called pitches:", len(m), "with location:", m.dist_in.notna().sum(), "challenged:", int(m.ch.sum()))
print("mean shift front->mid (in):", m.mid_shift_z_in.describe().round(2).to_dict())

# PA table for the WP engine
sc2 = sc.sort_values(["game_pk", "at_bat_number", "pitch_number"])
pa = sc2.groupby(["game_pk", "at_bat_number"]).last().reset_index()
pa.to_parquet(D / "sc_lastpitch_2026.parquet")
