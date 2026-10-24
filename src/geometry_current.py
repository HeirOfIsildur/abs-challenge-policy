"""the ruling-agreement audit on the CURRENT count-joined analysis
frame (data/called_2026_w.parquet, every challenge located), not on the pre-frame that src/geometry_audit.py
keeps for its historical plane and projection comparisons. Same rule (geometry.abs_strike: round ball, midpoint plane,
raw Statcast coordinates) and same ABS ruling (the call after the challenge).
usage: python src/geometry_current.py  -> results/geometry_current.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import geometry as G

m = pd.read_parquet("data/called_2026_w.parquet")
c = m[m.ch & m.plate_x.notna()].copy()
abs_s = np.where(c.ch_over, ~c.orig_strike.astype(bool), c.orig_strike.astype(bool))
pred = G.abs_strike(c.plate_x, c.plate_z, c.sz_top, c.sz_bot)
d, _ = G.signed_distance(c.plate_x, c.plate_z, c.sz_top, c.sz_bot)
mis = c[pred != abs_s]
out = dict(n=int(len(c)), n_challenges=int(m.ch.sum()), share=float((pred == abs_s).mean()), mismatches=int(len(mis)),
           mismatch_abs_d_in=sorted(float(abs(x)) for x in np.asarray(d)[pred != abs_s]),
           mismatch_keys=mis[["game_pk", "at_bat_number", "pitch_number"]].astype(int).values.tolist())
json.dump(out, open("results/geometry_current.json", "w"), indent=1)
print(json.dumps(out, indent=1))
