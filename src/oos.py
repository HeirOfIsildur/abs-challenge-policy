"""Out-of-sample checks of the baseline perception model (deterministic truth + ruling mislabel pi).
 (a) temporal: fit on 2026 games through 30 June, predict 1 July - end of season: challenges, overturn rate given a
     challenge and expected realised wins (sum of g over successful challenges) by side; compare with a refit on the
     test half; wins lost on the test half with train-half vs test-half parameters (DP + simulation, R draws).
 (b) external: Triple-A 2025 and MLB spring 2026 (data/called_<tag>_w.parquet from build_ext.py): the MLB 2026
     parameters transported as they are, and a refit on each feed (sigma_r stability).
usage: python src/oos.py [R]  -> results/oos.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import lik, sim

MAIN = "kappa_sr_region_g"        # BIC-best homogeneous model (criterion piecewise-linear in log g)
SPEC = lik.spec_of(MAIN)          # The main specification (was the baseline "normal")


def frames(m):
    """eligible rows per side in lik.prepare order."""
    out = {}
    for side in ("off", "def"):
        e = m[lik.eligible_mask(m, side)]
        out[side] = e.iloc[np.argsort(~e.ch.values, kind="stable")]
    return out


def score(m, th):
    data = lik.prepare(m); fr = frames(m); out = {}
    ngames = m.game_pk.nunique()
    for side in ("off", "def"):
        pch, p1 = lik.predict(data, th, SPEC, side); e = fr[side]; c = e.ch.values; g = e.g.values
        q = np.clip(p1[c] / pch[c], 1e-6, 1 - 1e-6); y = e.ch_over.values[c]
        out[side] = dict(games=ngames, ch_obs=int(c.sum()), ch_pred=round(float(pch.sum()), 1),
                         over_obs=round(float(y.mean()), 4),
                         # The overturn rate among the MODEL's challenges; the mean of p1/pch over the
                         # observed challenges is pi or 1 - pi pitch by pitch under the deterministic truth (a geometry
                         # check), kept as over_pred_given_observed_challenges
                         over_pred=round(float(p1.sum() / pch.sum()), 4),
                         over_pred_given_observed_challenges=round(float(q.mean()), 4),
                         wins_obs_per_team_game=round(float((g * (c & e.ch_over.values)).sum() / (2 * ngames)), 5),
                         wins_pred_per_team_game=round(float((g * p1).sum() / (2 * ngames)), 5),
                         logscore_overturn=round(float(np.mean(np.where(y, np.log(q), np.log(1 - q)))), 4),
                         brier_overturn=round(float(np.mean((q - y) ** 2)), 4))
    return out


def wins_lost(games, th, R, delta=None):
    """score-aware DP benchmark - observed and - simulated fitted rule on a game subset, per team per 162 games
    (headline.evaluate_all); best stake constant of the subset, and the value of a frozen constant delta."""
    from headline import evaluate_all
    _, _, out, _ = evaluate_all(dict(theta="override", theta_override=th, spec=SPEC, games=games), R=R, seed=1)
    tot = {k: v[0].sum() for k, v in out.items()}; n = 2 * len(games)
    f = lambda x: round(float(x) / n * 162, 4)
    stakes = {k: tot[k] for k in tot if k.startswith("stake_")}; best = max(stakes, key=stakes.get)
    res = dict(opt=f(tot["opt_score"]), observed=f(tot["observed"]), obs_rule_sim=f(tot["obs_rule"]),
               opt_minus_observed=f(tot["opt_score"] - tot["observed"]), opt_minus_obs_rule_sim=f(tot["opt_score"] - tot["obs_rule"]),
               best_stake=best, best_stake_share=round(float(stakes[best] / tot["opt_score"]), 4),
               rule_0008_share=round(float(tot["stake_0.008"] / tot["opt_score"]), 4))
    if delta is not None:
        k = f"stake_{delta}"
        res.update(frozen_delta=delta, frozen_share=round(float(tot[k] / tot["opt_score"]), 4),
                   frozen_gap_closed=round(float((tot[k] - tot["observed"]) / (tot["opt_score"] - tot["observed"])), 4))
    return res


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    res = {}
    m = pd.read_parquet("data/called_2026_w.parquet")
    dates = pd.read_parquet("data/feed_games_2026.parquet")[["game_pk", "date"]]
    m = m.merge(dates, on="game_pk")
    tr, te = m[m.date <= "2026-06-30"], m[m.date >= "2026-07-01"]
    th_full = np.load(f"data/th_{MAIN}.npy")
    fits = {}
    for name, d in (("train_marjun", tr), ("test_julsep", te)):
        r, _ = lik.fit(lik.prepare(d), SPEC, th0=th_full); fits[name] = r.x
        res[f"fit_{name}"] = dict(games=int(d.game_pk.nunique()), **{k: v for k, v in lik.readable(r.x, SPEC).items() if "beta" not in k})
        print(name, res[f"fit_{name}"], flush=True)
    res["temporal_predict_test_with_train"] = score(te, fits["train_marjun"])
    res["temporal_predict_test_with_test"] = score(te, fits["test_julsep"])
    res["temporal_insample_train"] = score(tr, fits["train_marjun"])
    print(json.dumps(res["temporal_predict_test_with_train"]), flush=True)
    gte = te.game_pk.unique()
    # Cross-fitted constant -- chosen on Mar-Jun with Mar-Jun parameters, then
    # frozen and scored on Jul-Sep (with Mar-Jun parameters, as a team would have to)
    res["wins_lost_train_train_params"] = wins_lost(tr.game_pk.unique(), fits["train_marjun"], R)
    d_train = float(res["wins_lost_train_train_params"]["best_stake"].split("_")[1])
    res["wins_lost_test_train_params"] = wins_lost(gte, fits["train_marjun"], R, delta=d_train)
    res["wins_lost_test_test_params"] = wins_lost(gte, fits["test_julsep"], R, delta=d_train)
    print({k: res[k] for k in res if k.startswith("wins_lost")}, flush=True)
    json.dump(res, open("results/oos.json", "w"), indent=1)
    for tag in ("spring2026", "aaa2025"):
        x = pd.read_parquet(f"data/called_{tag}_w.parquet")
        res[f"{tag}_transport_mlb2026"] = score(x, th_full)
        r, _ = lik.fit(lik.prepare(x), SPEC, th0=th_full)
        res[f"{tag}_refit"] = dict(converged=bool(r.success), **{k: v for k, v in lik.readable(r.x, SPEC).items() if "beta" not in k})
        res[f"{tag}_refit_score"] = score(x, r.x)
        cov = np.linalg.inv(lik.hessian(lik.make_nll(lik.prepare(x), SPEC), r.x)); nt = lik.n_truth(SPEC)
        ns = lik.n_side(SPEC)
        for i, side in ((nt + 6, "off"), (nt + ns + 6, "def")):
            sr = res[f"{tag}_refit"][f"{side}_sr"]; se = float(np.sqrt(cov[i, i]))
            res[f"{tag}_refit"][f"{side}_sr_ci95"] = [sr * np.exp(-1.96 * se), sr * np.exp(1.96 * se)]
        print(tag, res[f"{tag}_refit"], json.dumps(res[f"{tag}_transport_mlb2026"]), flush=True)
        json.dump(res, open("results/oos.json", "w"), indent=1)
