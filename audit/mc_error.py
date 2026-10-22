"""Audit item 7: Monte Carlo error of the point estimate (R simulation draws of truth/signal at the full-sample
fit). 40 single-draw evaluations with different seeds -> SD per draw, SE at R = 10.
usage: python audit/mc_error.py [n] -> audit/mc_error.json"""
import json, sys
import numpy as np
sys.path.insert(0, "src")
from headline import evaluate_all, spec_variant, summarise, MAIN

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rows = []
for sd in range(100, 100 + n):
    m, _, out, _ = evaluate_all(spec_variant(MAIN), R=1, seed=sd)
    s = summarise(out, int(m.gidx.max() + 1))
    rows.append(dict(seed=sd, lost=s["opt_score"]["wins"] - s["observed"]["wins"], opt=s["opt_score"]["wins"],
                     rule=s["stake_0.008"]["wins"], savant=s["savant"]["wins"]))
l = np.array([r["lost"] for r in rows]); sh = np.array([r["rule"] / r["opt"] for r in rows])
res = dict(n=n, lost_mean=float(l.mean()), lost_sd_per_draw=float(l.std(ddof=1)), lost_se_R10=float(l.std(ddof=1) / np.sqrt(10)),
           rule_share_mean=float(sh.mean()), rule_share_sd_per_draw=float(sh.std(ddof=1)), rows=rows)
json.dump(res, open("audit/mc_error.json", "w"), indent=1)
print({k: v for k, v in res.items() if k != "rows"})
