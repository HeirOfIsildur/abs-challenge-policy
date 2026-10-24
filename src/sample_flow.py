"""one sample-flow table from the feed to the likelihood and evaluation samples.
usage: python src/sample_flow.py  -> results/sample_flow.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import lik

if __name__ == "__main__":
    c = pd.read_parquet("data/called_2026.parquet")
    w = pd.read_parquet("data/called_2026_w.parquet")
    pos = w.pos_pitcher.values
    out = dict(
        called_feed=int(len(c)), challenges_feed=int(c.ch.sum()), overturned_feed=int((c.ch & c.ch_over).sum()),
        no_location=int((c.plate_x.isna() | c.sz_top.isna()).sum()),
        # challenges without a Statcast location (10,557 - 7 mis-joined - 6 = 10,544)
        no_location_challenges=int((c.ch & (c.plate_x.isna() | c.sz_top.isna())).sum()),
        misjoined_located=int((c.misjoin & c.plate_x.notna() & c.sz_top.notna()).sum()),
        misjoined_challenges=int((c.misjoin & c.ch & c.plate_x.notna()).sum()),
        located=int(len(w)), located_challenges=int(w.ch.sum()), located_overturned=int((w.ch & w.ch_over).sum()),
        position_player_pitches=int(pos.sum()), position_player_challenges=int((pos & w.ch).sum()),
        position_players=int(w[pos].pitcher.nunique()),
        nonmonotone_cells=int((w.g_signed < 0).sum()), nonmonotone_challenged=int((w.ch & (w.g_signed < 0)).sum()),
        nonmonotone_min=float(w.g_signed.min()),
        evaluation_opportunities=int((~pos).sum()),
        outside_fit_window=int(((w.w.abs() >= 8) & ~pos).sum()),
        # located challenges at least 8 in from the zone edge (outside the |w| < 8 fit window)
        outside_fit_window_challenges=int(((w.w.abs() >= 8) & w.ch & ~pos).sum()),
        likelihood={s: dict(n=int(lik.eligible_mask(w, s).sum()), challenges=int((lik.eligible_mask(w, s) & w.ch.values).sum()))
                    for s in ("off", "def")},
        half_innings_restarted=int(w.groupby(["game_pk", "inning", "top"]).apply(
            lambda d: d.sort_values(["at_bat_number", "pitch_number"]).score_diff_home.iat[0] != d.d_half_start.iat[0]).sum()))
    print(json.dumps(out, indent=1)); json.dump(out, open("results/sample_flow.json", "w"), indent=1)
