"""the count-based feed x Statcast join (src/join.py) against the old pitch-number join, and the
half-innings the DP's count law used to miss. -> results/join_audit.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
K = ["game_pk", "at_bat_number", "pitch_number"]
f = pd.read_parquet("data/feed_pitches_2026.parquet")
sc = pd.read_parquet("data/statcast_2026.parquet", columns=K + ["balls", "strikes", "description"])
c = f[f.code.isin(["C", "B", "*B"])].copy()
isk = c.code.eq("C").values
c["pb"] = np.where(isk, c.balls_after, c.balls_after - 1); c["ps"] = np.where(isk, c.strikes_after - 1, c.strikes_after)
old = c.merge(sc, on=K, how="left")
isk = old.code.eq("C")
desc_ok = (isk & old.description.eq("called_strike")) | (~isk & old.description.isin(["ball", "blocked_ball"]))
cnt_ok = (old.balls == old.pb) & (old.strikes == old.ps)
m = pd.read_parquet("data/called_2026.parquet", columns=K + ["dist_in", "renumbered", "misjoin", "ch"])
fd = pd.read_parquet("data/called_mlb2026feed_w.parquet", columns=K + ["dist_in"]).rename(columns={"dist_in": "dfeed"})
x = m.merge(fd, on=K, how="left"); d = (x.dist_in - x.dfeed).abs()
ren = x.renumbered.values
import dp_score
m2 = pd.read_parquet("data/called_2026_w.parquet", columns=["game_pk", "inning", "top", "pos_pitcher"]); m2 = m2[~m2.pos_pitcher]   # the DP frame: sim.opportunities
sh = dp_score.statcast_halves(); sh = sh[sh.game.isin(m2.game_pk.unique())]
have = m2.drop_duplicates(["game_pk", "inning", "top"]).rename(columns={"game_pk": "game", "inning": "inn"})
miss = sh.merge(have, on=["game", "inn", "top"], how="left", indicator=True)
miss = miss[miss._merge == "left_only"]
out = dict(old_join=dict(called=int(len(old)), description_ok_count_inconsistent=int((desc_ok & ~cnt_ok).sum()),
                         description_bad=int((~desc_ok).sum())),
           count_join=dict(called=int(len(m)), unmatched=int(m.misjoin.sum()), located=int(m.dist_in.notna().sum()),
                           renumbered=int(ren.sum()), renumbered_challenged=int((ren & m.ch.values).sum()),
                           renumbered_max_abs_diff_vs_feed_in=float(np.nanmax(d[ren])),
                           renumbered_median_abs_diff_vs_feed_in=float(np.nanmedian(d[ren])),
                           challenges=int(m.ch.sum()), challenges_located=int((m.ch & m.dist_in.notna()).sum())),
           halves=dict(statcast_halves=int(len(sh)), without_opportunity_frame_rows=int(len(miss)),
                       extra_innings=int((miss.inn >= 10).sum())))
json.dump(out, open("results/join_audit.json", "w"), indent=1)
print(json.dumps(out, indent=1))
