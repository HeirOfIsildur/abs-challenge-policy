"""Game-bootstrap CI for the Bar-2 threshold gaps (re-solve the DP on resampled games). usage: boot_thresholds.py B"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from sim import load, draw, solve_dp
B = int(sys.argv[1]); variant = {}
m0, _ = load(variant); m0 = m0[m0.w.abs() < 8].reset_index(drop=True)
gref = float(m0[m0.ch].g.median())
games = m0.game_pk.unique(); by = {g: d for g, d in m0.groupby("game_pk")}
rng = np.random.default_rng(11); rows = []
for b in range(B):
    pick = rng.choice(games, len(games))
    mb = pd.concat([by[g].assign(game_pk=i) for i, g in enumerate(pick)], ignore_index=True)
    W, s, pt = draw(mb, rng, variant)
    r = {}
    for nm, home in (("h", True), ("a", False)):
        D, _ = solve_dp(mb, pt, home)
        p = lambda Dv: Dv / (gref + Dv)
        r[nm] = (p(D[0, 0, 1]) - p(D[14, 0, 1]), p(D[0, 0, 2]) - p(D[14, 0, 2]), p(D[8, 0, 1]) - p(D[8, 0, 2]))
    rows.append(np.mean([r["h"], r["a"]], axis=0) * 100)
rows = np.array(rows)
out = {k: [round(float(np.mean(rows[:, i])), 1), *np.round(np.percentile(rows[:, i], [2.5, 97.5]), 1).tolist()]
       for i, k in enumerate(["early_vs_late_c1_pp", "early_vs_late_c2_pp", "c1_vs_c2_mid_pp"])}
print(json.dumps(out)); json.dump(out, open("data/bar2_boot.json", "w"))
