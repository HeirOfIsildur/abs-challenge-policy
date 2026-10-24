"""robustness of the score-aware optimum (dp_score.py).
(a) score-state clip DM = 6 (main) vs 10;  (b) 2-fold cross-fit: DP pools and token values estimated on the odd
(even) games, policies evaluated on the other half, so the optimum carries no in-sample optimism.
usage: python src/dp_checks.py [R]  -> results/dp_checks.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import sim, policies as P, dp_score
from headline import spec_variant, MAIN

POLS = dict(opt_score=(7, 0.0), opt=(0, 0.0), runs_dp=(6, 0.0), stake_0008=(2, 0.008), stake_001=(2, 0.01))


def frame(DM):
    dp_score.DM, dp_score.ND = DM, 2 * DM + 1
    v = spec_variant(MAIN)
    m, _ = sim.load(v); m = dp_score.add_dstart(m); m = sim.opportunities(m)
    m["new_extra"] = sim.extra_starts(m)
    return v, m, sim.kin_index(m).astype(np.int64)


def evaluate(m, kin, W, s, pt, fit_mask, eval_mask):
    """solve all DPs on rows fit_mask, evaluate on eval_mask; wins per team-season scaled to a full season."""
    mf = m[fit_mask].reset_index(drop=True); ptf = pt[fit_mask]
    Dh, _ = sim.solve_dp(mf, ptf, True); Da, _ = sim.solve_dp(mf, ptf, False)
    D4h, _ = dp_score.solve_dp_score(mf, ptf, True); D4a, _ = dp_score.solve_dp_score(mf, ptf, False)
    Cnh, Cna = P.runs_token_values(mf, ptf, sim.solve_dp)
    me = m[eval_mask].reset_index(drop=True); k = kin[eval_mask]
    gi = me.game_pk.astype("category").cat.codes.values.astype(np.int64); ng = int(gi.max() + 1)
    I = lambda x: np.ascontiguousarray(np.asarray(x).astype(np.int64))
    base = (gi, me.is_home_team.values, me.new_extra.values, I(me.slot), k, I(me.inning), I(me.balls), I(me.strikes), I(me.dbin))
    out = {}
    for name, (pc, dc) in POLS.items():
        v, n, _, _, _ = P.run_path(*base, W[eval_mask], pt[eval_mask], s[eval_mask], me.g.values, me.RV.values, sim.kappa_by_count(me),
                                   (me.side == "off").values, pc, Dh, Da, Cnh, Cna, np.zeros((4, 3, 2, 2)), 0.0, 1.0, dc, D4h, D4a, ng)
        out[name] = float(v.sum() / 30 * m.game_pk.nunique() / ng)
    return out


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    res = {}
    for DM in (6, 10):
        v, m, kin = frame(DM)
        rng = np.random.default_rng(3)
        odd = (m.game_pk.values % 2 == 1)
        acc = {"in_sample": [], "cross_fit": []}
        for r in range(R):
            W, s, pt = sim.draw(m, rng, v)
            allm = np.ones(len(m), bool)
            acc["in_sample"].append(evaluate(m, kin, W, s, pt, allm, allm))
            a = evaluate(m, kin, W, s, pt, odd, ~odd); b = evaluate(m, kin, W, s, pt, ~odd, odd)
            acc["cross_fit"].append({k: (a[k] + b[k]) / 2 for k in a})
        res[f"DM{DM}"] = {k: pd.DataFrame(x).mean().round(4).to_dict() for k, x in acc.items()}
        print(DM, res[f"DM{DM}"], flush=True)
    json.dump(dict(R=R, variant=MAIN, results=res), open("results/dp_checks.json", "w"), indent=1)
