"""Headline table for the DRS replication: values per 100 team-matches (match points: wins, Tests win + draw/2),
95 % match-bootstrap intervals (selection model refitted, DP re-solved), perception-noise sensitivity, threshold
movement, sqrt-law fit, fielding-side descriptives.
S_95 is the continuous profile-likelihood set (model.py); new keys (added, nothing renamed): sample (DRS match
sample incl. zero-review matches, build.py), model.s_95_open, per100_no_extra (DP without the extra opportunity after an
overturn, opt_nx), per100.stake_share_crossfit / crossfit_delta (delta chosen on the other half of the matches by date),
sqrt_law_no_extra.
usage: python src/drs/report.py  ->  results/drs/summary.json"""
import json
import numpy as np, pandas as pd

def ci(x):
    if not len(x): return None
    return [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]

S = {}
o = pd.read_parquet("data/drs_opps.parquet")
sample = json.load(open("results/drs/sample.json"))
for fmt in ("Test", "ODI", "T20"):
    r = json.load(open(f"results/drs/dp_{fmt}.json")); M = json.load(open(f"results/drs/model_{fmt}.json"))
    m = r["main"]; v = m["value"]; bs = m["best_stake"]; H = 100
    B = r["boot"]; S_boot = len(B)
    f = lambda k: np.array([b["per"][k] for b in B]) * H
    bopt, bobs, brule = f("opt"), f("observed"), f("obs_rule")
    bstake = np.array([b["per"][b["best"]] for b in B]) * H
    T = np.array(m["thresholds"]["lbw"])                         # [w][c]
    sens = {k: dict(s=round(x["s"], 2), dnll=None if x["dnll"] is None else round(x["dnll"], 2),
                    opt=round(x["value"]["opt"] * H, 3), obs_rule=round(x["value"]["obs_rule"] * H, 3),
                    stake=round(x["value"][x["best_stake"]] * H, 3), stake_share=round(x["value"][x["best_stake"]] / x["value"]["opt"], 3),
                    lost_vs_observed=round((x["value"]["opt"] - x["value"]["observed"]) * H, 3),
                    lost_vs_rule=round((x["value"]["opt"] - x["value"]["obs_rule"]) * H, 3),
                    reviews_opt=round(x["reviews"]["opt"], 2), opt_nx=round(x["value"]["opt_nx"] * H, 3),
                    stake_share_crossfit=round(x["crossfit_delta"]["share"], 3))
            for k, x in r.items() if k != "boot"}
    bat = o[(o.fmt == fmt) & (o.side == "bat")]; bowl = o[(o.fmt == fmt) & (o.side == "bowl")]
    nm = o[o.fmt == fmt].match.nunique()
    L = bat[(bat.kind == "lbw") & (bat.c > 0)]
    S[fmt] = dict(
        matches=int(nm), bootstrap_reps=S_boot, decisions=int(len(bat)), bat_reviews=int((bat.outcome != "none").sum()), bowl_reviews=int(len(bowl)),
        sample=sample[fmt],
        model=dict(s=round(M["s"], 3), s_95=[round(x, 3) for x in M["s_95"]], s_95_open=M["s_95_open"],
                   lbw_error_rate=round(M["err_rate"][0], 3), caught_error_rate=round(M["err_rate"][1], 4),
                   kappa=round(M["kappa"], 3), rho=round(M["rho"], 3),
                   lbw_review_share_failed_known=round(float((L.outcome != "none").mean()), 3)),
        per100=dict(opt=[round(v["opt"] * H, 3), ci(bopt)], stake=[round(v[bs] * H, 3), ci(bstake)], best_delta=float(bs.split("_")[1]),
                    obs_rule=[round(v["obs_rule"] * H, 3), ci(brule)], observed=[round(v["observed"] * H, 3), ci(bobs)],
                    conf50=round(v["conf50"] * H, 3),
                    lost_vs_observed=[round((v["opt"] - v["observed"]) * H, 3), ci(bopt - bobs)],
                    lost_vs_rule=[round((v["opt"] - v["obs_rule"]) * H, 3), ci(bopt - brule)],
                    stake_share=[round(v[bs] / v["opt"], 3), ci(bstake / bopt)],
                    stake_share_crossfit=[round(m["crossfit_delta"]["share"], 3), ci(np.array([b["crossfit"]["share"] for b in B]))]),
        crossfit_delta=m["crossfit_delta"],
        per100_no_extra=dict(opt=[round(v["opt_nx"] * H, 3), ci(f("opt_nx"))],
                             lost_vs_observed=[round((v["opt_nx"] - v["observed"]) * H, 3), ci(f("opt_nx") - bobs)],
                             lost_vs_rule=[round((v["opt_nx"] - v["obs_rule"]) * H, 3), ci(f("opt_nx") - brule)],
                             stake_share=[round(v[bs] / v["opt_nx"], 3), ci(bstake / f("opt_nx"))],
                             opt_minus_opt_nx=[round((v["opt"] - v["opt_nx"]) * H, 3), ci(bopt - f("opt_nx"))],
                             reviews_opt=round(m["reviews"]["opt_nx"], 2)),
        sqrt_law_no_extra=dict(delta1_pct=np.round(np.array(m["sqrt_law_nx"]["delta1"]) * 100, 3).tolist(),
                               R2_sqrt=round(m["sqrt_law_nx"]["sqrt"][0], 3), R2_linear=round(m["sqrt_law_nx"]["linear"][0], 3),
                               power=round(m["sqrt_law_nx"]["power"], 2)),
        reviews_per_team_match=dict(opt=round(m["reviews"]["opt"], 2), stake=round(m["reviews"][bs], 2),
                                    obs_rule=round(m["reviews"]["obs_rule"], 2), observed=round(m["reviews"]["observed"], 2)),
        thresholds_lbw_pct=dict(c1=np.round(T[:, 0] * 100, 1).tolist(), c2=np.round(T[:, 1] * 100, 1).tolist(),
                                c3=np.round(T[:, 2] * 100, 1).tolist(),
                                move_c1_w0_to_w8=round((T[0, 0] - T[8, 0]) * 100, 1), c1_vs_c2_w0=round((T[0, 0] - T[0, 1]) * 100, 1)),
        revealed_tau=[round(t, 3) for t in M["tau"]],
        sqrt_law=dict(N=np.round(m["sqrt_law"]["N"], 2).tolist(), delta1_pct=np.round(np.array(m["sqrt_law"]["delta1"]) * 100, 3).tolist(),
                      R2_sqrt=round(m["sqrt_law"]["sqrt"][0], 3), R2_linear=round(m["sqrt_law"]["linear"][0], 3),
                      power=round(m["sqrt_law"]["power"], 2)),
        sensitivity=sens,
        bowl=dict(per_team_match=round(len(bowl) / (2 * nm), 2),
                  success=round((bowl.outcome == "success").mean(), 3), uc=round((bowl.outcome == "uc").mean(), 3),
                  success_by_c={int(c): round((g.outcome == "success").mean(), 3) for c, g in bowl[bowl.c > 0].groupby("c")}))
json.dump(S, open("results/drs/summary.json", "w"), indent=1)
for fmt, x in S.items():
    print(fmt, json.dumps({k: x[k] for k in ("matches", "decisions", "bat_reviews", "bowl_reviews", "model", "per100", "per100_no_extra",
                                             "reviews_per_team_match")}))
    print("  thresholds", x["thresholds_lbw_pct"]["move_c1_w0_to_w8"], x["thresholds_lbw_pct"]["c1_vs_c2_w0"], "sqrt", x["sqrt_law"]["R2_sqrt"],
          x["sqrt_law"]["R2_linear"], x["sqrt_law"]["power"])
    for k, s in x["sensitivity"].items(): print("  ", k, s)
    print("  bowl", x["bowl"])
