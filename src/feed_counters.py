"""(supplement S3): our challenges-left reconstruction (build.py) against the live feed's own counters.

The feed's gameData.absChallenges gives usedSuccessful / usedFailed / remaining per team and game. We recount
successes and failures from the pitch-level reviews, rebuild challenges left with build.py's rule (start 2, -1 on a
failure, a team at 0 gets 1 at the start of each extra inning, no stacking), and compare. ABS games only (has not None).
Also counts the challenges outside the |distance| < 8 in window used by the likelihood and the evaluation (lik.prepare).
usage: python src/feed_counters.py [datadir]  -> results/feed_counters.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

D = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
g = pd.read_parquet(D / "feed_games_2026.parquet")
f = pd.read_parquet(D / "feed_pitches_2026.parquet")
g = g[g.has.notna()]
f = f[f.game_pk.isin(g.game_pk)].sort_values(["game_pk", "at_bat_number", "pitch_number"])
f["ch"] = f.ch.fillna(False).astype(bool); f["ch_over"] = f.ch_over.fillna(False).astype(bool)
f = f.merge(g[["game_pk", "home_id", "away_id"]], on="game_pk")

rows, left_at_ch, extra_innings, grants = [], [], 0, 0
for pk, gd in f.groupby("game_pk", sort=False):
    hid, aid = gd.home_id.iat[0], gd.away_id.iat[0]
    L = {hid: 2, aid: 2}; S = {hid: 0, aid: 0}; F = {hid: 0, aid: 0}; got = {hid: 0, aid: 0}
    last = 0
    for inn, top, ch, over, team in zip(gd.inning, gd.top, gd.ch, gd.ch_over, gd.ch_team):
        if inn >= 10 and top and inn != last:
            extra_innings += 1
            for k in L:
                if L[k] == 0:
                    L[k] = 1; got[k] += 1; grants += 1
        last = inn
        if ch:
            left_at_ch.append(L[team])
            if over:
                S[team] += 1
            else:
                F[team] += 1; L[team] -= 1
    gr = g[g.game_pk == pk].iloc[0]
    for k, side in ((hid, "home"), (aid, "away")):
        rows.append(dict(game_pk=pk, side=side, ours_s=S[k], ours_f=F[k], ours_left=L[k], granted=got[k],
                         feed_s=gr[f"{side}_usedSuccessful"], feed_f=gr[f"{side}_usedFailed"], feed_rem=gr[f"{side}_remaining"]))
r = pd.DataFrame(rows)
ok_s, ok_f = r.ours_s == r.feed_s, r.ours_f == r.feed_f
bad = r[~(ok_s & ok_f)]
out = dict(
    games=int(g.game_pk.nunique()), team_games=len(r), challenges=int(f.ch.sum()),
    team_games_counts_match=int((ok_s & ok_f).sum()), games_counts_mismatch=sorted(int(x) for x in bad.game_pk.unique()),
    mismatch_detail=bad.to_dict("records"),
    feed_remaining_is_max0_2_minus_failed=float((r.feed_rem == np.maximum(0, 2 - r.feed_f)).mean()),
    max_used_failed=int(r.feed_f.max()),
    team_games_with_grant=int((r.granted > 0).sum()), grants=int(grants), extra_innings=int(extra_innings),
    ours_left_differs_from_feed=int((r.ours_left != r.feed_rem).sum()),
    ours_left_differs_from_feed_without_grant=int(((r.ours_left != r.feed_rem) & (r.granted == 0)).sum()),
    challenges_with_zero_left=int(sum(x <= 0 for x in left_at_ch)),
    extra_inning_challenges=int((f.ch & (f.inning >= 10)).sum()))
c = pd.read_parquet(D / "called_2026.parquet", columns=["ch", "dist_in"])
c = c[c.ch]
out["challenges_beyond_8in"] = int((c.dist_in.abs() >= 8).sum()); out["challenges_no_location"] = int(c.dist_in.isna().sum())
json.dump(out, open("results/feed_counters.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "mismatch_detail"}, indent=1))
