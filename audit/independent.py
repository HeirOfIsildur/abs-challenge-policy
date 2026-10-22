"""Audit item 9: independent recomputation from the raw parquet files. Does NOT import src/.
Own join, own zone rule, own challenge bookkeeping, own simplified engine (stakes from Statcast's own
delta_home_win_exp / delta_run_exp cell means), own minimal perception simulation for the Savant rule.
usage: python audit/independent.py [R] -> audit/independent.json"""
import json, sys
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

R = int(sys.argv[1]) if len(sys.argv) > 1 else 5
f = pd.read_parquet("data/feed_pitches_2026.parquet")
gm = pd.read_parquet("data/feed_games_2026.parquet")
sc = pd.read_parquet("data/statcast_2026.parquet", columns=[
    "game_pk", "at_bat_number", "pitch_number", "plate_x", "plate_z", "sz_top", "sz_bot", "description", "inning",
    "inning_topbot", "outs_when_up", "on_1b", "on_2b", "on_3b", "balls", "strikes", "home_score", "away_score",
    "batter", "pitcher", "fielder_2", "delta_home_win_exp", "delta_run_exp"])
res = {}
abs_games = set(gm.game_pk[gm.has.notna()])                  # has = None: no ABS at the venue
res["games_total"] = int(len(gm)); res["games_with_abs"] = len(abs_games)
f = f[f.game_pk.isin(abs_games)].copy()
f["ch"] = f.ch.fillna(False).astype(bool); f["ch_over"] = f.ch_over.fillna(False).astype(bool)
x = f.merge(sc, on=["game_pk", "at_bat_number", "pitch_number"], how="inner", suffixes=("", "_sc"))
ch = x[x.ch]
role = np.select([ch.ch_player == ch.batter_sc, ch.ch_player == ch.pitcher_sc, ch.ch_player == ch.fielder_2],
                 ["batter", "pitcher", "catcher"], "other")
res["challenges_feed"] = int(f.ch.sum()); res["challenges_joined"] = int(len(ch))
res["by_role"] = {r: dict(n=int((role == r).sum()), overturn=float(ch.ch_over[role == r].mean()))
                  for r in ["batter", "catcher", "pitcher", "other"] if (role == r).any()}
res["overturn_all"] = float(ch.ch_over.mean())
res["challenges_per_team_game"] = float(f.ch.sum() / (2 * len(abs_games)))

# --- own zone rule on raw Statcast 2026 coordinates (ABS plane), ball radius 1.45 in, plate 17 in
c = x[x.code.isin(["C", "B", "*B"]) & x.plate_x.notna() & x.sz_top.notna()].copy()
c["final_strike"] = c.code.eq("C")
c["orig_strike"] = c.final_strike ^ c.ch_over
r_ = 1.45 / 12; hw = 17 / 24
# round ball: r_ + signed distance of the centre to the unexpanded rectangle, so corners are rounded
dx = hw - c.plate_x.abs(); dz = np.minimum(c.plate_z - c.sz_bot, c.sz_top - c.plate_z)
inside = (dx >= 0) & (dz >= 0)
dist = 12 * (r_ + np.where(inside, np.minimum(dx, dz), -np.hypot(np.clip(-dx, 0, None), np.clip(-dz, 0, None))))
c["w"] = np.where(c.orig_strike, -dist, dist)                 # > 0: the original call is wrong under ABS
cc = c[c.ch]
res["rule_reproduces_rulings"] = float(((cc.w > 0) == cc.ch_over).mean())

# --- simplified engine: WP / run swing of a ball vs a strike = difference of Statcast's own per-pitch deltas,
# averaged over called balls and called strikes in the same (inning<=9|extra, half, outs, bases, count, diff) cell
c["bases"] = c.on_1b.notna() * 1 + c.on_2b.notna() * 2 + c.on_3b.notna() * 4
c["half"] = (c.inning_topbot == "Bot").astype(int)
c["d"] = (c.home_score - c.away_score).clip(-4, 4)
c["inn"] = c.inning.clip(1, 10)
cp = c[c.description.isin(["ball", "called_strike"])]
def swing(keys, col):
    t = cp.groupby(keys + ["description"])[col].mean().unstack("description")
    return (t["ball"] - t["called_strike"]).rename("sw")
fine = ["inn", "half", "outs_when_up", "bases", "balls", "strikes", "d"]
coarse = ["inn", "half", "outs_when_up", "balls", "strikes", "d"]
for col, name in (("delta_home_win_exp", "g"), ("delta_run_exp", "RV")):
    a = c.join(swing(fine, col), on=fine)["sw"]; b = c.join(swing(coarse, col), on=coarse)["sw"]
    c[name] = a.fillna(b).fillna(0).abs().values
c["home_bats"] = c.half == 1
c["team_home"] = np.where(c.orig_strike, c.home_bats, ~c.home_bats)   # whose opportunity
c = c.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
ng = c.game_pk.nunique()
obs = c[c.ch]
res["observed_wins_per_team_season"] = float((obs.g * obs.ch_over).sum() / 30)
res["observed_wins_batter"] = float((obs.g * obs.ch_over)[obs.orig_strike].sum() / 30)
res["observed_wins_fielder"] = float((obs.g * obs.ch_over)[~obs.orig_strike].sum() / 30)

# --- minimal perception simulation (sigma_r from the main fit, deterministic truth, pooled calibration per side)
SIG = {True: 2.91, False: 1.91}          # batter (orig strike) / catcher (orig ball), results/headline
gk = c.game_pk.values; th = c.team_home.values.astype(int); inn = c.inning.values; top = (c.half.values == 0)
newx = np.r_[False, (inn[1:] >= 10) & top[1:] & ((inn[1:] != inn[:-1]) | (gk[1:] != gk[:-1]))]
newg = np.r_[True, gk[1:] != gk[:-1]]
W = c.w.values > 0; g = c.g.values; RV = c.RV.values; os_ = c.orig_strike.values

def play(go_fn, p):
    val = 0.0; n = 0; left = [2, 2]
    for i in range(len(c)):
        if newg[i]: left = [2, 2]
        if newx[i]:
            left = [max(l, 1) if l == 0 else l for l in left]
        t = th[i]
        if left[t] == 0 or not go_fn(i, p[i]): continue
        n += 1
        if W[i]: val += g[i]
        else: left[t] -= 1
    return val / 30, n / (2 * ng)

rng = np.random.default_rng(7); acc = {"savant": [], "stake_0.008": []}
for rr in range(R):
    s = c.w.values + rng.standard_normal(len(c)) * np.where(os_, SIG[True], SIG[False])
    p = np.zeros(len(c))
    for side in (True, False):
        k = os_ == side
        bins = np.clip(np.floor((s[k] + 15) / 0.25), 0, 120)
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(bins, W[k].astype(float))
        p[k] = iso.predict(bins)
    acc["savant"].append(play(lambda i, pi: pi >= 0.2 / (0.2 + RV[i]), p))
    acc["stake_0.008"].append(play(lambda i, pi: pi >= 0.008 / (g[i] + 0.008), p))
res["sim_R"] = R
for k, v in acc.items():
    v = np.array(v); res[f"{k}_wins_per_team_season"] = float(v[:, 0].mean()); res[f"{k}_challenges_per_team_game"] = float(v[:, 1].mean())
res["savant_over_observed"] = res["savant_wins_per_team_season"] / res["observed_wins_per_team_season"]
json.dump(res, open("audit/independent.json", "w"), indent=1)
print(json.dumps(res, indent=1))
