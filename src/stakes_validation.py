"""HONZA-132 (Sol #4, Astra #3): independent validation of the stakes -- the ball-vs-strike win-probability contrasts g
that drive every value in the paper. Level checks (Brier, correlation with Statcast WP) do not validate a contrast.

(1) Statcast's own contrast. Statcast's home_win_exp is an exact function of the pre-pitch state (inning, half, outs,
    bases, score, count: zero spread within the 44,560 observed cells), so its ball-vs-strike contrast can be read
    off its own table: WE after a ball = WE(count + ball) for b < 3, and the observed WE after ball four at that state
    (home_win_exp + delta_home_win_exp of the walks there) for b = 3; WE after a strike likewise (strike three: every
    third strike at that state). g_sc = batting-team sign x (WE_ball - WE_strike). Compared with our g by leverage,
    inning and score cell; Statcast's WE is rounded to 0.001, which adds noise of that order.
(2) Held-out engines. The games are split at random in two halves; engines fitted on each half give two independent
    contrasts for every pitch (agreement by leverage decile).
(3) Outcome test near the zone edge. Within 2 in of the ABS edge the call is decided by fractions of an inch of
    location, so given the state, ball vs strike is close to as good as random. On held-out games (engine fitted on
    the other half), Y = 1{the batting team wins} satisfies E[Y | X, ball] - E[Y | X, strike] = g(X) if the engine's
    contrast is right: regress Y - WPs(X) on [1, WPs(X), B g(X)] (WPs = the engine's batting-team WP after a strike,
    B = 1{the call that stood was a ball}); the slope on B g should be 1. Game-clustered SEs. By leverage tercile,
    and for Statcast's g_sc.
(4) Policy values under other yardsticks. (a) The whole analysis on Statcast's stakes (perception refit on g_sc, DP,
    every policy, scored on g_sc); (b) decisions on our stakes, every flip scored at Statcast's signed contrast.
usage: python src/stakes_validation.py [R]  -> results/stakes_validation.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")

KEY = ["game_pk", "at_bat_number", "pitch_number"]
STRIKE_DESC = ["called_strike", "swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt",
               "bunt_foul_tip"]
S = ["inning", "half", "outs_when_up", "bases", "d"]


def statcast_states():
    sc = pd.read_parquet("data/statcast_2026.parquet", columns=KEY + ["inning", "inning_topbot", "outs_when_up", "on_1b",
                         "on_2b", "on_3b", "balls", "strikes", "home_score", "away_score", "home_win_exp",
                         "delta_home_win_exp", "description", "post_home_score", "post_away_score", "events"])
    sc["bases"] = sc.on_1b.notna() * 1 + sc.on_2b.notna() * 2 + sc.on_3b.notna() * 4
    sc["half"] = (sc.inning_topbot == "Bot").astype(int)
    sc["d"] = (sc.home_score - sc.away_score).astype(int)
    return sc


def state_frame(m):
    return pd.DataFrame(dict(inning=m.inning.values, half=(~m.top.values).astype(int), outs_when_up=m.outs_when_up.values,
                             bases=m.bases.values, d=m.score_diff_home.values.astype(int), balls=m.balls.values,
                             strikes=m.strikes.values))


def statcast_after(m, sc):
    """Statcast's home WE after a ball and after a strike at every row's pre-pitch state (NaN where Statcast never
    visited the needed after-state)."""
    we = sc.groupby(S + ["balls", "strikes"]).home_win_exp.first()
    walk = sc[(sc.balls == 3) & sc.description.isin(["ball", "blocked_ball", "pitchout"]) & sc.events.isin(["walk"])]
    wb4 = (walk.home_win_exp + walk.delta_home_win_exp).groupby([walk[c] for c in S]).median()
    k3 = sc[(sc.strikes == 2) & sc.description.isin(STRIKE_DESC) & sc.events.isin(["strikeout", "strikeout_double_play"])]
    ws3 = (k3.home_win_exp + k3.delta_home_win_exp).groupby([k3[c] for c in S + ["balls"]]).median()
    x = state_frame(m)
    kb = pd.MultiIndex.from_arrays([x[c] for c in S] + [x.balls + 1, x.strikes])
    ks = pd.MultiIndex.from_arrays([x[c] for c in S] + [x.balls, x.strikes + 1])
    web = np.where(x.balls < 3, we.reindex(kb).values, wb4.reindex(pd.MultiIndex.from_arrays([x[c] for c in S])).values)
    wes = np.where(x.strikes < 2, we.reindex(ks).values,
                   ws3.reindex(pd.MultiIndex.from_arrays([x[c] for c in S] + [x.balls])).values)
    return web, wes


def statcast_stakes(m, sc):
    web, wes = statcast_after(m, sc)
    return np.where(m.top.values, -1.0, 1.0) * (web - wes)


def wins(sc):
    last = sc.sort_values(KEY).groupby("game_pk").last()
    return (last.post_home_score > last.post_away_score).astype(float)


def ols_cluster(y, X, cl):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    codes = pd.factorize(cl)[0]
    U = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(U, codes, X * e[:, None])
    V = XtXi @ (U.T @ U) @ XtXi
    return b, np.sqrt(np.diag(V))


def edge_test(m, g, wps, hw, band=2.0):
    """slope of Y - WPs on B g (see (3)); g contrast (batting team), wps batting-team WP after a strike."""
    k = (m.dist_in.abs().values < band) & np.isfinite(g) & np.isfinite(wps)
    Y = np.where(~m.top.values, hw, 1 - hw)
    B = (~m.final_strike.values).astype(float)          # the call that stood (after any challenge): the game went on from it
    X = np.column_stack([np.ones(k.sum()), wps[k], B[k] * g[k]])
    b, se = ols_cluster((Y - wps)[k], X, m.game_pk.values[k])
    return dict(n=int(k.sum()), slope=float(b[2]), se=float(se[2]), ci=[float(b[2] - 1.96 * se[2]), float(b[2] + 1.96 * se[2])])


def main(R=10):
    import crossfit as CF, lik
    from headline import evaluate_all, summarise, spec_variant, MAIN
    out = {}
    m = pd.read_parquet("data/called_2026_w.parquet")
    sc = statcast_states()
    web, wes = statcast_after(m, sc)
    m["g_sc"] = np.where(m.top.values, -1.0, 1.0) * (web - wes)
    ok = np.isfinite(m.g_sc.values); ch = m.ch.values
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    out["statcast"] = dict(rows=int(len(m)), with_statcast_contrast=int(ok.sum()), share=float(ok.mean()),
                           corr=c(m.g_signed[ok], m.g_sc[ok]), mean_ours=float(m.g_signed[ok].mean()),
                           mean_statcast=float(m.g_sc[ok].mean()),
                           ratio_of_means=float(m.g_sc[ok].mean() / m.g_signed[ok].mean()),
                           challenged=dict(n=int((ok & ch).sum()), corr=c(m.g_signed[ok & ch], m.g_sc[ok & ch]),
                                           ratio_of_means=float(m.g_sc[ok & ch].mean() / m.g_signed[ok & ch].mean())),
                           negative_ours=int((m.g_signed < 0).sum()), negative_statcast=int((m.g_sc[ok] < 0).sum()))
    dec = pd.qcut(m.g[ok], 10, labels=False, duplicates="drop")
    out["statcast"]["by_leverage_decile"] = [dict(decile=int(q), n=int(len(d)), mean_ours=float(d.g_signed.mean()),
                                                  mean_statcast=float(d.g_sc.mean()), corr=c(d.g_signed, d.g_sc))
                                             for q, d in m[ok].groupby(dec)]
    inn = np.select([m.inning <= 3, m.inning <= 6, m.inning <= 8], ["1-3", "4-6", "7-8"], "9+")
    sd = np.select([m.score_diff_home.abs() == 0, m.score_diff_home.abs() <= 2, m.score_diff_home.abs() <= 4],
                   ["tie", "1-2", "3-4"], "5+")
    out["statcast"]["by_inning_score"] = [dict(inning=a, score=b, n=int(len(d)), mean_ours=float(d.g_signed.mean()),
                                               mean_statcast=float(d.g_sc.mean()), corr=c(d.g_signed, d.g_sc))
                                          for (a, b), d in m[ok].groupby([inn[ok], sd[ok]])]
    print(json.dumps({k: v for k, v in out["statcast"].items() if not k.startswith("by_")}, indent=1), flush=True)

    # (2) + (3): half-sample engines; each pitch's contrast from the engine of the OTHER half
    hw = wins(sc).reindex(m.game_pk.values).values
    sc_e = CF.statcast()
    rows, tests = [], []
    dec_all = pd.qcut(m.g, 10, labels=False, duplicates="drop").values
    for s in range(5):
        rng = np.random.default_rng(7000 + s)
        games = np.unique(m.game_pk.values); fold = dict(zip(games, rng.permutation(len(games)) % 2))
        fm = m.game_pk.map(fold).values; fs = sc_e.game_pk.map(fold).values
        gA, gB = (CF.stakes_under(CF.engine(sc_e[fs == f]), m) for f in (0, 1))
        rows.append(dict(split=s, corr_halves=c(gA.g_signed, gB.g_signed),
                         by_decile=[dict(decile=int(q), corr=c(gA.g_signed.values[dec_all == q], gB.g_signed.values[dec_all == q]),
                                         mean_A=float(gA.g_signed.values[dec_all == q].mean()),
                                         mean_B=float(gB.g_signed.values[dec_all == q].mean())) for q in range(10)]))
        gho = np.where(fm == 0, gB.g_signed.values, gA.g_signed.values)
        wps = np.where(fm == 0, gB.wp_strike.values, gA.wp_strike.values)
        tests.append((gho, np.where(~m.top.values, wps, 1 - wps)))         # batting team's WP after a strike
    o = [edge_test(m, g, w, hw) for g, w in tests]
    terc = pd.qcut(m.g, 3, labels=False).values
    gho, wps = tests[0]
    wps_sc = np.where(~m.top.values, wes, 1 - wes)
    out["edge_test"] = dict(band_in=2.0, note="slope of (Y - batting-team WP after a strike) on 1{ball} g; 1 = contrast right",
                            ours_held_out=dict(splits=o, slope_mean=float(np.mean([x["slope"] for x in o])),
                                               se_typical=float(np.mean([x["se"] for x in o]))),
                            ours_by_leverage_tercile=[edge_test(m[terc == t], gho[terc == t], wps[terc == t], hw[terc == t])
                                                      for t in range(3)],
                            statcast=edge_test(m, m.g_sc.values, wps_sc, hw))
    out["half_engines"] = dict(splits=rows, corr_mean=float(np.mean([r["corr_halves"] for r in rows])))
    print(json.dumps(out["edge_test"], indent=1), flush=True)
    json.dump(out, open("results/stakes_validation.json", "w"), indent=1)

    # (4) policy values under Statcast's yardstick, on the rows where Statcast's contrast exists
    v = spec_variant(MAIN); spec = v["spec"]
    pols = ["bench", "opt_score", "stake_0.008", "runs_dp", "card_refit", "card_pub", "savant", "obs_rule", "observed",
            "clairvoyant"]
    tab = lambda s: dict(lost=s["bench"]["wins"] - s["observed"]["wins"], wins={k: s[k]["wins"] for k in pols},
                         share={k: s[k]["wins"] / s["bench"]["wins"] for k in pols}, best_stake=s["best_stake"])
    msc = m[ok].copy(); msc["g"] = np.clip(msc.g_sc.values, 0, None); msc["g_signed"] = msc.g_sc.values
    r, _ = lik.fit(lik.prepare(msc), spec, th0=np.load(v["theta"]), maxiter=60)
    mm, _, o4, _ = evaluate_all(dict(v, theta_override=r.x, frame=msc), R=R, seed=1)
    sa = summarise(o4, int(mm.gidx.max() + 1))
    mm, _, o5, _ = evaluate_all(dict(v, frame=m[ok].assign(gv=m.g_sc.values[ok])), R=R, seed=1)
    sb = summarise(o5, int(mm.gidx.max() + 1))
    mm, _, o6, _ = evaluate_all(dict(v, frame=m[ok].copy()), R=R, seed=1)
    sc_ = summarise(o6, int(mm.gidx.max() + 1))
    P = lik.readable(r.x, spec)
    out["policies"] = dict(R=R, rows=int(ok.sum()), statcast_everything=tab(sa), ours_decide_statcast_score=tab(sb),
                           ours_same_rows=tab(sc_), statcast_fit=dict(off_sr=P["off_sr"], def_sr=P["def_sr"]))
    print(json.dumps(out["policies"], indent=1), flush=True)
    json.dump(out, open("results/stakes_validation.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
