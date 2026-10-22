"""How often would the ABS label of an UNCHALLENGED pitch change with the tracking source?
The 99.90 % rule reproduction is measured on challenged pitches, where the ruling is observed. For every called pitch
the Stats API live feed carries its own coordinates and zone limits (build_ext.py mlb2026feed, projected to the
plate midpoint); a pitch whose ABS label differs between Statcast and the feed has no well-defined known truth.
The share of such pitches, by distance band and for challenged / unchallenged pitches, is a direct estimate of the
label uncertainty on the pitches the likelihood treats as known.
usage: python src/label_agreement.py  -> results/label_agreement.json"""
import json
import numpy as np, pandas as pd

KEY = ["game_pk", "at_bat_number", "pitch_number"]

if __name__ == "__main__":
    sc = pd.read_parquet("data/called_2026_w.parquet", columns=KEY + ["dist_in", "orig_strike", "ch", "ch_over", "side", "left", "w"])
    fd = pd.read_parquet("data/called_mlb2026feed_w.parquet", columns=KEY + ["dist_in"]).rename(columns={"dist_in": "dist_feed"})
    m = sc.merge(fd, on=KEY, how="inner")
    m = m[m.dist_in.notna() & m.dist_feed.notna()]
    m["differ"] = (m.dist_in > 0) != (m.dist_feed > 0)
    m["band"] = pd.cut(m.dist_in.abs(), [0, 0.25, 0.5, 1, 2, 4, 100], right=False)
    elig = (m.left > 0) & (m.w.abs() < 8)
    out = dict(pitches=int(len(m)), share_differ=float(m.differ.mean()), n_differ=int(m.differ.sum()),
               challenged=dict(n=int(m.ch.sum()), share_differ=float(m[m.ch].differ.mean())),
               unchallenged=dict(n=int((~m.ch).sum()), share_differ=float(m[~m.ch].differ.mean())),
               likelihood_sample=dict(n=int(elig.sum()), share_differ=float(m[elig].differ.mean())),
               abs_diff_in=dict(zip(["p50", "p90", "p99"], np.percentile((m.dist_in - m.dist_feed).abs(), [50, 90, 99]).round(4).tolist())),
               by_band=[dict(band=str(b), n=int(len(d)), share_differ=float(d.differ.mean()))
                        for b, d in m.groupby("band", observed=True)],
               note="label = ABS strike iff signed distance > 0 (geometry.signed_distance, round ball); Statcast 2026 "
                    "plate_x/plate_z at the plate midpoint vs the live feed's pX/pZ projected to the midpoint")
    print(json.dumps(out, indent=1))
    json.dump(out, open("results/label_agreement.json", "w"), indent=1)
