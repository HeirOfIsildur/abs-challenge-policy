"""Sample sizes quoted in the manuscript  -> results/data_counts.json, and the observed challenges by role
-> results/roles_observed.csv (that file used to be a artefact that no script in the run order wrote).
usage: python src/data_counts.py   (needs data/ from fetch_challenges.py, build.py and perception.py)"""
import json
import numpy as np, pandas as pd

g = pd.read_parquet("data/feed_games_2026.parquet")
p = pd.read_parquet("data/feed_pitches_2026.parquet", columns=["ch", "ch_over"])
c = pd.read_parquet("data/called_2026_w.parquet", columns=["ch", "ch_over", "role", "g", "game_pk", "inning"])
out = dict(games=len(g), games_abs=int(g.has.notna().sum()), pitches=len(p), challenges=int(p.ch.fillna(False).astype(bool).sum()),
           overturned=int(p.ch_over.fillna(False).astype(bool).sum()), called_pitches=len(c), called_challenged=int(c.ch.sum()),
           games_extra=int((c.groupby("game_pk").inning.max() >= 10).sum()), games_called=int(c.game_pk.nunique()))
json.dump(out, open("results/data_counts.json", "w"), indent=1)
print(out)
a = pd.read_parquet("data/called_2026.parquet", columns=["ch", "ch_over", "role"])     # every challenge, with or without a location
a = a[a.ch]; cw = c[c.ch]
rows = [dict(role=r, n=int((a.role == r).sum()), overturn=round(float(a.ch_over[a.role == r].mean()), 6),
             wins_gained_per_team=round(float((cw.g * cw.ch_over)[cw.role == r].sum() / 30), 4),
             mean_g=round(float(cw.g[cw.role == r].mean()), 4)) for r in ("batter", "catcher", "pitcher")]
pd.DataFrame(rows).to_csv("results/roles_observed.csv", index=False)
print(pd.DataFrame(rows))
