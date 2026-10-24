"""further model and design checks.
  dp_mono   the monotonicities of Theorem 2 in the score-aware DP actually used (random horizons, score transitions,
            extra-inning top-up): share of computed entries, and of opportunities by visit weight, where the marginal
            value of a challenge rises with the opportunity index or where the first marginal exceeds the second's
            complement (Delta(1) < Delta(2)); comparator at its fitted parameters, one simulated world.
  lapse     what predicts not challenging a call wrong by more than 3 in (a challenge held): rates and a logit on the
            stake, the count, outs, the inning, the score and runners, by side.
  edge      the local-randomisation assumption of the near-edge outcome check: balance of pre-pitch covariates between
            balls and strikes within 2 in of the edge (standardised differences), for the call that stood and for the
            umpire's original call; and the held-out slope with the original call.
usage: python src/extra_checks.py dp_mono|lapse|edge  -> results/extra_checks.json (keys merged)"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
OUT = "results/extra_checks.json"


def save(key, val):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res[key] = val; json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(val, indent=1)[:3000])


def dp_mono():
    import sim, dp_score
    from headline import spec_variant, MAIN
    v = spec_variant(MAIN)
    m, _ = sim.load(v); m = dp_score.add_dstart(m); m = sim.opportunities(m)
    m["new_extra"] = sim.extra_starts(m); m["cal"] = sim.cal_groups(m)
    kin = sim.kin_index(m)
    W, s, pt = sim.draw(m, np.random.default_rng(20261024), v)
    out = {}
    for home in (True, False):
        D, _ = dp_score.solve_dp_score(m, pt, home)
        sel = (m.is_home_team.values == home)
        k = np.minimum(kin[sel], dp_score.KMAX - 1)
        visits = np.zeros(D.shape[:3]); np.add.at(visits, (m.slot.values[sel], m.dbin.values[sel], k), 1)
        d1, d2 = D[..., 1], D[..., 2]
        rise = np.zeros(D.shape[:3], bool); rise[:, :, :-1] = d1[:, :, 1:] > d1[:, :, :-1] + 1e-12
        conc = d1 < d2 - 1e-12
        vis = visits > 0
        out["home" if home else "away"] = dict(
            entries_visited=int(vis.sum()),
            share_entries_rise_in_k=float(rise[vis].mean()), share_entries_delta1_lt_delta2=float(conc[vis].mean()),
            share_opportunities_rise_in_k=float(visits[rise].sum() / visits.sum()),
            share_opportunities_delta1_lt_delta2=float(visits[conc].sum() / visits.sum()),
            regulation_only=dict(share_opportunities_rise_in_k=float(visits[:18][rise[:18]].sum() / visits[:18].sum()),
                                 share_opportunities_delta1_lt_delta2=float(visits[:18][conc[:18]].sum() / visits[:18].sum())))
    save("dp_monotonicity", out)


def lapse():
    import lik
    from headline import spec_variant, MAIN
    import sim
    m, _ = sim.load(spec_variant(MAIN))
    out = {}
    for side in ("off", "def"):
        e = m[lik.eligible_mask(m, side) | ((m.side == side) & (m.left > 0) & (m.w > 3)).values]
        e = e[(e.side == side) & (e.left > 0) & (e.w > 3) & ~e.pos_pitcher]
        y = e.ch.values.astype(float)
        X = pd.DataFrame(dict(const=1.0, log_g=np.log(np.clip(e.g.values, 1e-4, None)), two_strikes=(e.strikes == 2).astype(float),
                              three_balls=(e.balls == 3).astype(float), two_outs=(e.outs == 2).astype(float),
                              late=(e.inning >= 7).astype(float), abs_score=np.minimum(e.score_diff_home.abs(), 6) / 6,
                              risp=(e.on_2b.notna() | e.on_3b.notna()).astype(float), one_left=(e.left == 1).astype(float),
                              w_over4=(e.w > 4).astype(float)))
        import statsmodels.api as sm
        fit = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds=dict(groups=pd.factorize(e.game_pk)[0]))
        gq = pd.qcut(e.g, 4, labels=False, duplicates="drop")
        out[side] = dict(n=int(len(e)), challenge_rate=float(y.mean()),
                         rate_by_stake_quartile=[float(y[gq == q].mean()) for q in range(int(gq.max()) + 1)],
                         rate_one_left=float(y[e.left.values == 1].mean()), rate_two_left=float(y[e.left.values == 2].mean()),
                         logit=dict(coef={c: float(fit.params[c]) for c in X}, se={c: float(fit.bse[c]) for c in X}))
    save("lapse_far_misses", out)


def edge():
    import crossfit as CF
    from stakes_validation import wins, statcast_states, edge_test
    m = pd.read_parquet("data/called_2026_w.parquet")
    k = m.dist_in.abs().values < 2.0
    cov = pd.DataFrame(dict(balls=m.balls, strikes=m.strikes, outs=m.outs, inning=m.inning.clip(1, 10),
                            abs_score=m.score_diff_home.abs().clip(0, 6), risp=(m.on_2b.notna() | m.on_3b.notna()).astype(float),
                            on_1b=m.on_1b.notna().astype(float), home_batting=(~m.top).astype(float),
                            batter_right=(m.stand == "R").astype(float), log_g=np.log(np.clip(m.g, 1e-4, None))))
    bal = {}
    for lab, call in (("final_call", m.final_strike.values), ("original_call", m.orig_strike.values)):
        a, b = cov[k & ~call], cov[k & call]
        bal[lab] = {c: float((a[c].mean() - b[c].mean()) / np.sqrt((a[c].var() + b[c].var()) / 2)) for c in cov}
    sc = statcast_states(); hw = wins(sc).reindex(m.game_pk.values).values
    sc_e = CF.statcast(); slopes = []
    for s in range(5):
        rng = np.random.default_rng(7000 + s)
        games = np.unique(m.game_pk.values); fold = dict(zip(games, rng.permutation(len(games)) % 2))
        fm = m.game_pk.map(fold).values; fs = sc_e.game_pk.map(fold).values
        gA, gB = (CF.stakes_under(CF.engine(sc_e[fs == f]), m) for f in (0, 1))
        gho = np.where(fm == 0, gB.g_signed.values, gA.g_signed.values)
        wps = np.where(fm == 0, gB.wp_strike.values, gA.wp_strike.values); wps = np.where(~m.top.values, wps, 1 - wps)
        slopes.append(edge_test(m.assign(final_strike=m.orig_strike), gho, wps, hw))
    save("edge_checks", dict(band_in=2.0, standardised_differences=bal,
                             original_call_slope=dict(slope_mean=float(np.mean([x["slope"] for x in slopes])),
                                                      se_typical=float(np.mean([x["se"] for x in slopes])), splits=slopes)))


def tail():
    """the far tail that identifies the lapse rate: eligible opportunities (a challenge held, |w| < 8 in, no position-
    player pitching) by miss margin, their challenge rates, and the share of wrong calls within 1 and 2 in."""
    import lik, sim
    from headline import spec_variant, MAIN
    m, _ = sim.load(spec_variant(MAIN))
    out = {}
    for side in ("off", "def"):
        e = m[lik.eligible_mask(m, side)]
        bins = [0, 1, 2, 3, 4, 5, 8]
        b = pd.cut(e.w, bins)
        out[side] = dict(by_bin=[dict(bin=str(k), n=int(len(d)), rate=float(d.ch.mean())) for k, d in e.groupby(b, observed=True)],
                         n_beyond_3=int((e.w > 3).sum()), n_beyond_4=int((e.w > 4).sum()),
                         wrong_calls=int((e.w > 0).sum()), share_wrong_within_1=float((e.w[e.w > 0] <= 1).mean()),
                         share_wrong_within_2=float((e.w[e.w > 0] <= 2).mean()))
    save("tail", out)


if __name__ == "__main__":
    dict(dp_mono=dp_mono, lapse=lapse, edge=edge, tail=tail)[sys.argv[1]]()
