"""Bars, bootstrap CIs (by game), thresholds, role split. usage: python analyze.py <tag> [B]"""
import json, pickle, sys
import numpy as np, pandas as pd
tag = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
d = pickle.load(open(f"data/sim_{tag}.pkl", "rb")); out = d["out"]
ng = len(d["games"]); rng = np.random.default_rng(7)
stakes = [k for k in out if k.startswith("stake_")]
tot = {k: out[k][0].sum(axis=(1, 2)) for k in out}          # per game, both teams
best_stake = max(stakes, key=lambda k: tot[k].sum())
per_ts = lambda x: x.sum() / 30
idx = rng.integers(0, ng, size=(B, ng))
def ci(x):
    bs = np.array([x[i].sum() / 30 for i in idx]); return per_ts(x), np.percentile(bs, [2.5, 97.5])
res = {}
for name, x in [("opt", tot["opt"]), ("savant", tot["savant"]), ("best_stake", tot[best_stake]), ("obs_rule_sim", tot["obs_rule"]),
                ("observed", tot["observed"]), ("opt-savant", tot["opt"] - tot["savant"]), ("opt-best_stake", tot["opt"] - tot[best_stake]),
                ("opt-observed", tot["opt"] - tot["observed"]), ("opt-obs_rule_sim", tot["opt"] - tot["obs_rule"]),
                ("best_stake-observed", tot[best_stake] - tot["observed"])]:
    e, (lo, hi) = ci(x); res[name] = (round(e, 3), round(lo, 3), round(hi, 3))
# stake-only: re-select best d inside each bootstrap (honest optimism control)
bs = []
S = np.stack([tot[k] for k in stakes])
for i in idx:
    s = S[:, i].sum(1); bs.append((tot["opt"][i].sum() - s.max()) / 30)
res["opt-best_stake(reselected)"] = (round(res["opt-best_stake"][0], 3), *np.round(np.percentile(bs, [2.5, 97.5]), 3))
res["best_stake_d"] = best_stake
# side split (0 = defence: catcher/pitcher on called balls; 1 = offence: batter on called strikes)
side = {}
for k in ("opt", "savant", best_stake, "observed", "obs_rule"):
    for sd, nm in ((1, "batter"), (0, "fielder")):
        x = out[k][0][:, :, sd].sum(1); side[f"{k}:{nm}"] = round(per_ts(x), 3)
        side[f"{k}:{nm}:n_per_team_game"] = round(out[k][1][:, :, sd].sum() / (2 * ng), 3)
res["side"] = side
# thresholds (Bar 2): p* = D/(g+D) at reference g
Ds = d["Ds"]
m = pd.read_parquet("data/called_2026_w.parquet")
gref = float(m[m.ch].g.median()); res["g_ref"] = round(gref, 5)
def pstar(D, g): return D / (g + D)
th = []
for Dh, Da in Ds:
    for D in (Dh, Da):
        th.append(dict(early_c2=pstar(D[0, 0, 2], gref), early_c1=pstar(D[0, 0, 1], gref),
                       late8_c2=pstar(D[14, 0, 2], gref), late8_c1=pstar(D[14, 0, 1], gref),
                       mid5_c2=pstar(D[8, 0, 2], gref), mid5_c1=pstar(D[8, 0, 1], gref),
                       D1_early=D[0, 0, 1], D1_mid=D[8, 0, 1], D1_late=D[14, 0, 1], D2_early=D[0, 0, 2]))
th = pd.DataFrame(th).mean()
res["thresholds_at_gref"] = th.round(4).to_dict()
res["bar2_early_vs_late_c1_pp"] = round(100 * (th.early_c1 - th.late8_c1), 1)
res["bar2_early_vs_late_c2_pp"] = round(100 * (th.early_c2 - th.late8_c2), 1)
res["bar2_c1_vs_c2_mid_pp"] = round(100 * (th.mid5_c1 - th.mid5_c2), 1)
json.dump(res, open(f"data/res_{tag}.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
