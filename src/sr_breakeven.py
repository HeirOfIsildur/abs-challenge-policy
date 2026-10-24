"""Wins lost on the main model (headline.MAIN, BIC-best fixed effects) when both sides' sigma_r is scaled by a
multiplier, holding everything else at the fit: how far would sigma_r have to be off for the headline to reach the
0.25 wins/team-season line of the decision rule?  evaluated on the score-aware optimum (opt_score, as the
headline), not the pooled DP of.  also records the simulated behavioural rule s > kappa(x) (wins and
challenges per team-game) at each multiplier, so the multiplier at which the simulated rule earns what teams realised can
be read off.  usage: python src/sr_breakeven.py [R] -> results/sr_breakeven.json"""
import json, sys
sys.path.insert(0, "src")
from headline import evaluate_all, spec_variant, summarise, MAIN

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    res = {}
    for mult in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4):
        m, _, out, _ = evaluate_all(spec_variant(MAIN, sr_mult=mult), R=R, seed=1)
        s = summarise(out, int(m.gidx.max() + 1))
        res[str(mult)] = dict(lost=s["opt_score"]["wins"] - s["observed"]["wins"], opt=s["opt_score"]["wins"],
                              observed=s["observed"]["wins"], lost_pooled_dp=s["opt"]["wins"] - s["observed"]["wins"],
                              obs_rule=s["obs_rule"]["wins"], obs_rule_n=s["obs_rule"]["n"], opt_n=s["opt_score"]["n"])
        print(mult, res[str(mult)], flush=True)
        json.dump(res, open("results/sr_breakeven.json", "w"), indent=1)
