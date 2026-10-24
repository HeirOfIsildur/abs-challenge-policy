"""HONZA-89 audit item 4 / 8: WP engine calibration against realised results, home/away sign, and our run stake RV
against an RE288 table estimated directly from Statcast delta_run_exp.  Imports the engine only through stakes.Wpa /
stakes.Vcount (the objects under audit).  usage: python src/wp_calibration.py -> results/wp_calibration.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import stakes as S

cols = ["game_pk", "at_bat_number", "pitch_number", "inning", "inning_topbot", "outs_when_up", "on_1b", "on_2b", "on_3b",
        "balls", "strikes", "home_score", "away_score", "post_home_score", "post_away_score", "home_win_exp",
        "description", "delta_run_exp", "type"]
sc = pd.read_parquet("data/statcast_2026.parquet", columns=cols).sort_values(["game_pk", "at_bat_number", "pitch_number"])
fin = sc.groupby("game_pk").agg(h=("post_home_score", "last"), a=("post_away_score", "last"))
fin["home_win"] = (fin.h > fin.a).astype(float)
pa = sc.groupby(["game_pk", "at_bat_number"]).first().reset_index()
pa = pa.merge(fin[["home_win"]], left_on="game_pk", right_index=True)
pa["half"] = (pa.inning_topbot == "Bot").astype(int)
pa["bases"] = pa.on_1b.notna() * 1 + pa.on_2b.notna() * 2 + pa.on_3b.notna() * 4
pa["diff"] = (pa.home_score - pa.away_score).astype(int)
pa["wp"] = [S.Wpa(i, h, o, b, d) for i, h, o, b, d in pa[["inning", "half", "outs_when_up", "bases", "diff"]].itertuples(index=False)]
res = {"n_pa": len(pa), "n_games": int(fin.shape[0]), "home_win_rate": float(fin.home_win.mean())}
res["brier_engine"] = float(((pa.wp - pa.home_win) ** 2).mean())
res["brier_statcast_home_win_exp"] = float(((pa.home_win_exp - pa.home_win) ** 2).mean())
res["corr_engine_statcast"] = float(np.corrcoef(pa.wp, pa.home_win_exp)[0, 1])
pa["bin"] = pd.cut(pa.wp, np.linspace(0, 1, 11), include_lowest=True)
cal = pa.groupby("bin", observed=True).agg(n=("wp", "size"), pred=("wp", "mean"), obs=("home_win", "mean"))
res["home_field_gap_tied_states"] = float((pa[pa["diff"] == 0].home_win - pa[pa["diff"] == 0].wp).mean())
res["calibration_by_decile"] = cal.round(4).reset_index().astype({"bin": str}).to_dict("records")
pa["inn"] = pa.inning.clip(1, 10)
pa["dcls"] = pd.cut(pa["diff"], [-99, -3, -1, 0, 2, 99], labels=["<=-3", "-2..-1", "0", "1..2", ">=3"])
t = pa.groupby(["inn", "dcls"], observed=True).agg(n=("wp", "size"), pred=("wp", "mean"), obs=("home_win", "mean")).reset_index()
pa["res"] = pa.home_win - pa.wp            # game-clustered SE: PAs of one game share the outcome
gc = pa.groupby(["inn", "dcls", "game_pk"], observed=True).res.sum().pow(2).groupby(level=[0, 1], observed=True).sum()
t = t.merge(gc.rename("ss").reset_index(), on=["inn", "dcls"])
t["se"] = np.sqrt(t.ss) / t.n; t["z"] = (t.obs - t.pred) / t.se.clip(1e-4)
res["by_inning_score"] = t.round(4).astype({"dcls": str}).to_dict("records")
res["max_abs_z_by_inning_score"] = float(t.z.abs().max())
res["n_cells_abs_z_gt_3"] = int((t.z.abs() > 3).sum())
# home/away sign: tied, top of 1st, nobody on -> home WP from engine vs realised home win share
res["wp_first_pitch_engine"] = float(S.Wpa(1, 0, 0, 0, 0))
# a home run in the bottom of an inning must raise home WP; a run in the top must lower it
res["sign_check"] = dict(bot5_up1_vs_tie=float(S.Wpa(5, 1, 0, 0, 1) - S.Wpa(5, 1, 0, 0, 0)),
                         top5_up1_vs_tie=float(S.Wpa(5, 0, 0, 0, 1) - S.Wpa(5, 0, 0, 0, 0)))
# RV vs RE288 from Statcast: mean delta_run_exp of called balls minus called strikes per (count, outs, bases)
c = sc[sc.description.isin(["ball", "called_strike"])].copy()
c["bases"] = c.on_1b.notna() * 1 + c.on_2b.notna() * 2 + c.on_3b.notna() * 4
c["is_ball"] = c.description.eq("ball")
r = c.groupby(["balls", "strikes", "outs_when_up", "bases", "is_ball"]).delta_run_exp.agg(["mean", "size"]).unstack("is_ball")
r = r.dropna()
r["re288_swing"] = r[("mean", True)] - r[("mean", False)]
r["n"] = r[("size", True)] + r[("size", False)]
ours = []
for (b, s, o, bs) in r.index:
    _, rb = S.Vcount(1, 0, o, bs, 0, b + 1, s); _, rs = S.Vcount(1, 0, o, bs, 0, b, s + 1)
    ours.append(rb - rs)
r["ours"] = ours
w = r.n.values
res["rv_vs_re288"] = dict(cells=int(len(r)), weighted_mean_ours=float(np.average(r.ours, weights=w)),
                         weighted_mean_re288=float(np.average(r.re288_swing, weights=w)),
                         weighted_corr=float(np.corrcoef(r.ours, r.re288_swing)[0, 1]),
                         weighted_mean_abs_diff=float(np.average(np.abs(r.ours - r.re288_swing), weights=w)),
                         ratio_ours_to_re288=float(np.average(r.ours, weights=w) / np.average(r.re288_swing, weights=w)))
json.dump(res, open("results/wp_calibration.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("by_inning_score", "calibration_by_decile")}, indent=1))
print(cal.round(3))
print(t[t.z.abs() > 2.5].round(3))
