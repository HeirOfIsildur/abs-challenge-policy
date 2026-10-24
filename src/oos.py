"""HONZA-73 step 2: out-of-sample checks of the baseline perception model (HONZA-88: deterministic truth + ruling mislabel pi).
 (a) temporal: fit on 2026 games through 30 June, predict 1 July - end of season: challenges, overturn rate given a
     challenge and expected realised wins (sum of g over successful challenges) by side; compare with a refit on the
     test half; wins lost on the test half with train-half vs test-half parameters (DP + simulation, R draws).
 (b) external: Triple-A 2025 and MLB spring 2026 (data/called_<tag>_w.parquet from build_ext.py): the MLB 2026
     parameters transported as they are, and a refit on each feed (sigma_r stability).
HONZA-130 (Opus / Sol reviews): (a) re-estimated the calibration cells, the belief maps and the DP on the test half
     itself (only theta and the constant delta came from Mar-Jun). It is kept as the "retrospective test-half
     benchmark"; frozen() is the genuinely frozen cross-fit: every component of the policy is estimated on Mar-Jun
     (optionally also the WP engine, stakes and run laws; HONZA-132: the perception fit then uses the training-engine
     stakes) and applied unchanged to the Jul-Sep call stream.
usage: python src/oos.py [R]  -> results/oos.json
       python src/oos.py frozen [R] [engine] [world_test]  -> results/oos.json key "frozen_crossfit" (other keys untouched)"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import lik, sim

MAIN = "kappa_sr_region_g"        # HONZA-122: BIC-best homogeneous model (criterion piecewise-linear in log g)
SPEC = lik.spec_of(MAIN)          # HONZA-122 (Sol review): the main specification (was the baseline "normal")


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
                         # HONZA-122: the overturn rate among the MODEL's challenges; the mean of p1/pch over the
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



# ---------------------------------------------------------------------------------------------------------------------
# HONZA-130 (Opus / Sol reviews): genuinely frozen temporal cross-fit.
TRAIN_END, TEST_START = "2026-06-30", "2026-07-01"


def train_theta(half="train_marjun", frame=None):
    """Mar-Jun (or Jul-Sep: half="test_julsep") refit of the main model, exactly as in __main__. frame: the called-pitch
    frame whose stakes the fit uses (HONZA-132, Sol review: with the training WP engine the perception model is refitted
    on the TRAINING-engine stakes, so that nothing in the frozen pipeline sees the full-season engine); default the
    full-season frame. Not cached (HONZA-132: the old cache was reused on existence alone)."""
    m = (frame if frame is not None else pd.read_parquet("data/called_2026_w.parquet"))
    m = m.drop(columns=[c for c in ("date",) if c in m]).merge(       # HONZA-132: build.py now carries the date
        pd.read_parquet("data/feed_games_2026.parquet")[["game_pk", "date"]], on="game_pk")
    d = m[m.date <= TRAIN_END] if half == "train_marjun" else m[m.date >= TEST_START]
    r, _ = lik.fit(lik.prepare(d), SPEC, th0=np.load(f"data/th_{MAIN}.npy"))
    return r.x


def prep(games, th, frame=None, cuts=None):
    """the opportunity frame of evaluate_all for a game subset; calibration cells with the given stake cut points."""
    import dp_score
    m, _ = sim.load(dict(theta="override", theta_override=th, spec=SPEC, games=games, frame=frame))
    m = dp_score.add_dstart(m)
    m = sim.opportunities(m)
    m["new_extra"] = sim.extra_starts(m)
    m["cal"] = sim.cal_groups(m, cuts)
    return m


def base_args(m):
    return (m.gidx.values, m.is_home_team.values, m.new_extra.values, m.slot.values.astype(np.int64),
            sim.kin_index(m).astype(np.int64), m.inning.values.astype(np.int64), m.balls.values.astype(np.int64),
            m.strikes.values.astype(np.int64), m.dbin.values.astype(np.int64))


def apply_maps(m, s, maps):
    """the player's belief on new pitches from FROZEN per-cell maps ("side|cell" -> f[301], sim.draw binning). A cell
    never seen in training falls back to the mean map of its side (count returned)."""
    pt = np.zeros(len(m)); miss = 0
    key = np.char.add(m.side.values.astype(str), np.char.add("|", m.cal.values.astype(str)))
    side_mean = {sd: np.mean([f for k, f in maps.items() if k.startswith(sd + "|")], axis=0) for sd in ("off", "def")}
    for k in np.unique(key):
        idx = key == k
        f = maps.get(k)
        if f is None:
            f = side_mean[k.split("|", 1)[0]]; miss += int(idx.sum())
        pt[idx] = f[np.clip(np.floor((s[idx] + 15) / 0.1).astype(int), 0, 300)]
    return pt, miss


def run_split(base, W, pt, g_dec, g_val, pol, dconst, D4h, D4a, isoff, ng):
    """policies.run_path for pol 2 (stake) / 7 (opt_score) when the stake the player DECIDES on (g_dec, from the
    training-half WP engine) differs from the stake the flip is SCORED at (g_val, the full-season yardstick that also
    scores the observed challenges). With g_dec = g_val it is policies.run_path."""
    import policies as P
    z1 = np.zeros((19, 80, 3)); z2 = np.zeros((19, 3)); zc = np.zeros((4, 3, 2, 2)); n = len(W)
    if np.array_equal(g_dec, g_val):
        v, nn, _, _, _ = P.run_path(*base, W, pt, np.zeros(n), g_val, np.zeros(n), np.zeros((n, 3)), isoff, pol,
                                    z1, z1, z2, z2, zc, 0.0, 0.0, dconst, D4h, D4a, ng)
        return v, nn
    return _run_split(*base, W, pt, g_dec, g_val, isoff, pol, dconst, D4h, D4a, ng)


def _njit_run_split():
    from numba import njit
    import policies as P
    decide = P.decide

    @njit(cache=False)
    def f(gidx, is_home_team, new_extra, slot, kin, inning, balls, strikes, dbin, W, pt, g_dec, g_val, isoff, pol,
          dconst, D4h, D4a, ngames):
        val = np.zeros((ngames, 2, 2)); nch = np.zeros((ngames, 2, 2))
        z1 = np.zeros((19, 80, 3)); z2 = np.zeros((19, 3)); zc = np.zeros((4, 3, 2, 2))
        c = np.array([2, 2]); cur = -1
        for i in range(len(W)):
            if gidx[i] != cur:
                cur = gidx[i]; c[0] = 2; c[1] = 2
            if new_extra[i]:
                for t in range(2):
                    if c[t] == 0: c[t] = 1
            t = 1 if is_home_team[i] else 0
            if c[t] == 0:
                continue
            D4 = D4h if t == 1 else D4a
            if decide(pol, pt[i], 0.0, 0.0, g_dec[i], 0.0, slot[i], kin[i], c[t], inning[i], balls[i], strikes[i],
                      z1, z2, zc, 0.0, 0.0, dconst, D4, dbin[i]):
                sd = 1 if isoff[i] else 0
                nch[cur, t, sd] += 1
                if W[i]:
                    val[cur, t, sd] += g_val[i]
                else:
                    c[t] -= 1
        return val, nch
    return f


def _run_split(*a):
    global _RS
    if "_RS" not in globals():
        _RS = _njit_run_split()
    return _RS(*a)


def observed_total(games, th):
    """realised value of the actual challenges on a game subset (headline.evaluate_all 'observed'), full-season g."""
    mm, _ = sim.load(dict(theta="override", theta_override=th, spec=SPEC, games=games))
    ch = mm[mm.ch]
    return float((ch.g * ch.ch_over).sum())


def train_engine():
    """WP engine, stakes and DP run laws rebuilt from Mar-Jun Statcast only (joint_boot.py recipe). Returns the
    called-pitch frame with training-engine stakes and the (half_runs, extra_runs) laws; module state is restored."""
    import wp, stakes, dp_score
    sc = pd.read_parquet("data/statcast_2026.parquet", columns=sorted(set(wp.SC_COLS) | {"home_score", "away_score",
                                                                                         "post_bat_score", "game_date"}))
    sc = sc[pd.to_datetime(sc.game_date) <= TRAIN_END].drop(columns="game_date")
    T = wp.build_T(sc=sc, verbose=False); Wwp, RE, D = wp.solve(T, verbose=False)
    keep = (stakes.T, stakes.W, stakes.RE, stakes.D)
    stakes.T, stakes.W, stakes.RE, stakes.D = T, Wwp, RE, int(D)
    try:
        m = pd.read_parquet("data/called_2026_w.parquet")
        cols = ["wp_ball", "wp_strike", "re_ball", "re_strike", "g", "g_signed", "RV", "wp_home_now", "half", "diff"]
        me = stakes.add_stakes(m.drop(columns=[c for c in cols if c in m]))
    finally:
        stakes.T, stakes.W, stakes.RE, stakes.D = keep
    P = dp_score.half_runs_from(sc)
    x = sc[(sc.inning >= 10) & (sc.inning_topbot == "Top")].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    h = x.groupby(["game_pk", "inning"]).agg(a=("bat_score", "first"), z=("post_bat_score", "last"))
    rx = (h.z - h.a).clip(0, dp_score.RMAX).astype(int).values
    Px = np.bincount(rx, minlength=dp_score.RMAX + 1) / max(len(rx), 1)
    return me, P, Px


def frozen(R=8, seed=130, engine=False, world="train"):
    """HONZA-130 (Opus / Sol reviews): the frozen temporal cross-fit. On Mar-Jun: perception theta, stake-tercile cut
    points, per-cell isotonic belief maps (sim.draw on the TRAINING pitches: auxiliary draws there), the score-aware
    DP continuation values D4h/D4a (dp_score.solve_dp_score on training pools and hazards) and the one-line constant
    (best of headline.STAKES on the training draws). engine=True also rebuilds the WP engine, the stakes the player
    decides on, and the DP run laws from Mar-Jun Statcast (scoring stays on the full-season stakes, the same yardstick
    as the observed value). On Jul-Sep, per draw r: truth and signal from the training theta (sim.draw on the test
    pitches), beliefs from the TRAINING maps by cell (training cut points), and the frozen DP / frozen rule run along the
    recorded test stream. The same draw also gives the retrospective benchmark (maps and DP re-estimated on the test
    half, as evaluate_all does), so the two are paired. Totals in wins per team-season (per team per 162 games).
    world="test": the Jul-Sep truths and signals come from the Jul-Sep refit instead (perception drift: the frozen
    beliefs are then miscalibrated to the extent the halves differ; the retrospective maps are fitted to that world)."""
    import dp_score, policies as P
    from headline import STAKES
    feed = pd.read_parquet("data/feed_games_2026.parquet")[["game_pk", "date"]]
    called = pd.read_parquet("data/called_2026_w.parquet", columns=["game_pk"]).game_pk.unique()
    feed = feed[feed.game_pk.isin(called)]
    gtr = feed.game_pk[feed.date <= TRAIN_END].unique(); gte = feed.game_pk[feed.date >= TEST_START].unique()
    hr0, xr0 = dp_score.half_runs, dp_score.extra_runs
    fr_eng = None
    if engine:
        fr_eng, Ptr, Pxtr = train_engine()
    th = train_theta(frame=fr_eng)                         # HONZA-132: on the training-engine stakes if engine
    variant = dict(spec=SPEC)
    try:
        mtr = prep(gtr, th, frame=fr_eng)                      # training half (training-engine stakes if engine)
        cuts = sim.stake_cuts(mtr)
        thw = th if world == "train" else train_theta("test_julsep")
        mte = prep(gte, thw)                                   # test half, full-season stakes: the world and the yardstick
        if engine:
            key = ["game_pk", "at_bat_number", "pitch_number"]
            gd = mte[key].merge(fr_eng[key + ["g"]], on=key, how="left").g.values
            assert len(gd) == len(mte) and not np.isnan(gd).any()
        else:
            gd = mte.g.values
        mte["cal"] = sim.cal_groups(mte.assign(g=gd), cuts)    # frozen cells: training cut points on the player's stakes
        mte_retro = mte.copy(); mte_retro["cal"] = sim.cal_groups(mte)   # retrospective: test-half cells and cut points
        btr, bte = base_args(mtr), base_args(mte)
        ngtr, ngte = int(mtr.gidx.max() + 1), int(mte.gidx.max() + 1)
        isoff_tr, isoff_te = (mtr.side == "off").values, (mte.side == "off").values
        rtr = np.random.default_rng([seed, 0]); rte = np.random.default_rng([seed, 1])
        # pass 1: training draws -> frozen maps, frozen DP, training value of every stake constant
        frz, tr_stake = [], {d: 0.0 for d in STAKES}; tr_opt = 0.0; tr_lap = []
        if engine:
            dp_score.half_runs, dp_score.extra_runs = (lambda: Ptr), (lambda: Pxtr)
        for r in range(R):
            Wt, st, ptt, maps = sim.draw(mtr, rtr, variant, return_maps=True)
            D4h, _ = dp_score.solve_dp_score(mtr, ptt, True); D4a, _ = dp_score.solve_dp_score(mtr, ptt, False)
            frz.append((maps, D4h, D4a))
            v, _ = run_split(btr, Wt, ptt, mtr.g.values, mtr.g.values, 7, 0.0, D4h, D4a, isoff_tr, ngtr); tr_opt += v.sum() / R
            for d in STAKES:
                v, _ = run_split(btr, Wt, ptt, mtr.g.values, mtr.g.values, 2, d, D4h, D4a, isoff_tr, ngtr)
                tr_stake[d] += v.sum() / R
        dp_score.half_runs, dp_score.extra_runs = hr0, xr0
        d_tr = max(STAKES, key=lambda d: tr_stake[d])
        # pass 2: test draws, frozen objects applied unchanged; retrospective benchmark on the same (W, s)
        acc = {k: [0.0, 0.0] for k in ("frozen_dp", "frozen_rule", "frozen_rule_0008", "retro_dp", "retro_rule")}
        per_draw, miss = [], 0
        for r in range(R):
            maps, D4h, D4a = frz[r]
            W, s, pt_retro = sim.draw(mte_retro, rte, variant)          # test truths/signals (training theta) + test maps
            pt, mi = apply_maps(mte, s, maps); miss += mi
            R4h, _ = dp_score.solve_dp_score(mte, pt_retro, True); R4a, _ = dp_score.solve_dp_score(mte, pt_retro, False)
            row = {}
            for name, (pp, pol, dc, Dh_, Da_, gdec) in dict(
                    frozen_dp=(pt, 7, 0.0, D4h, D4a, gd), frozen_rule=(pt, 2, d_tr, D4h, D4a, gd),
                    frozen_rule_0008=(pt, 2, 0.008, D4h, D4a, gd),
                    retro_dp=(pt_retro, 7, 0.0, R4h, R4a, mte.g.values),
                    retro_rule=(pt_retro, 2, 0.008, R4h, R4a, mte.g.values)).items():
                v, n = run_split(bte, W, pp, gdec, mte.g.values, pol, dc, Dh_, Da_, isoff_te, ngte)
                acc[name][0] += v.sum() / R; acc[name][1] += n.sum() / R; row[name] = float(v.sum())
            per_draw.append(row)
            print("frozen draw", r, {k: round(v / (2 * ngte) * 162, 3) for k, v in row.items()}, flush=True)
    finally:
        dp_score.half_runs, dp_score.extra_runs = hr0, xr0
    obs = observed_total(gte, th)
    f = lambda x: round(float(x) / (2 * ngte) * 162, 4)
    tot = {k: v[0] for k, v in acc.items()}
    fd, fr, rd, rr = tot["frozen_dp"], tot["frozen_rule"], tot["retro_dp"], tot["retro_rule"]
    pse = lambda x, y: round(float(np.std([(d[x] - d[y]) / (2 * ngte) * 162 for d in per_draw], ddof=1) / np.sqrt(R)), 4)
    lost_draw = np.array([(d["frozen_dp"] - obs) / (2 * ngte) * 162 for d in per_draw])
    return dict(
        R=R, seed=seed, world=("Jul-Sep truths/signals from the Mar-Jun theta" if world == "train" else
                               "Jul-Sep truths/signals from the Jul-Sep refit (beliefs still frozen on Mar-Jun)"), engine=("Mar-Jun Statcast: WP engine, decision stakes, half/extra-inning run laws" if engine
                                else "full-season WP engine, stakes and run laws (not frozen)"),
        train_games=int(len(gtr)), test_games=int(ngte), test_opportunities=int(len(mte)),
        frozen_delta=d_tr, train_insample_stake_share={str(d): round(float(tr_stake[d] / tr_opt), 4) for d in STAKES},
        test_rows_in_cells_unseen_in_training=int(miss // R),
        wins=dict(frozen_dp=f(fd), frozen_rule=f(fr), frozen_rule_0008=f(tot["frozen_rule_0008"]), observed=f(obs),
                  retro_dp_same_draws=f(rd), retro_rule_0008_same_draws=f(rr)),
        challenges_per_team_game={k: round(float(v[1] / (2 * ngte)), 3) for k, v in acc.items()},
        frozen_rule_share_of_retro=round(float(fr / rd), 4), frozen_dp_share_of_retro=round(float(fd / rd), 4),
        frozen_rule_share_of_frozen_dp=round(float(fr / fd), 4),
        frozen_rule_gap_closed_vs_frozen_dp=round(float((fr - obs) / (fd - obs)), 4),
        frozen_rule_gap_closed_vs_retro=round(float((fr - obs) / (rd - obs)), 4),
        wins_lost_vs_frozen_dp=f(fd - obs), wins_lost_vs_frozen_dp_mc_se=round(float(lost_draw.std(ddof=1) / np.sqrt(R)), 4),
        wins_lost_vs_retro_same_draws=f(rd - obs),
        # paired over draws (same truths and signals): what freezing costs the DP and the rule, with MC SEs
        frozen_minus_retro_dp=f(fd - rd), frozen_minus_retro_dp_mc_se=pse("frozen_dp", "retro_dp"),
        frozen_rule_minus_retro_rule_0008=f(fr - rr), frozen_rule_minus_retro_rule_0008_mc_se=pse("frozen_rule", "retro_rule"))


def frozen_main(R, engine=False, world="train"):
    res = json.load(open("results/oos.json"))
    out = res.get("frozen_crossfit", {})
    out[("with_train_engine" if engine else "no_engine") + ("" if world == "train" else "_world_testfit")] = frozen(R, engine=engine, world=world)
    # the retrospective test-half benchmark and the half-season losses already in this file (wins_lost_* keys:
    # main model kappa_sr_region_g, score-aware DP opt_score, R draws of __main__), recorded so the text can cite them
    ex = {k: res[k] for k in ("wins_lost_train_train_params", "wins_lost_test_train_params", "wins_lost_test_test_params")}
    out["retrospective_test_half_benchmark"] = dict(
        model=MAIN, dp="score-aware (opt_score, dp_score.solve_dp_score)",
        note=("theta (and delta) from Mar-Jun; calibration cells, stake cut points, belief maps and DP re-estimated on "
              "the half it is evaluated on; full-season WP engine and run laws"),
        opt=ex["wins_lost_test_train_params"]["opt"], observed=ex["wins_lost_test_train_params"]["observed"],
        frozen_share=ex["wins_lost_test_train_params"]["frozen_share"],
        frozen_gap_closed=ex["wins_lost_test_train_params"]["frozen_gap_closed"])
    out["halfseason_losses_existing"] = dict(
        first_half_own_params=ex["wins_lost_train_train_params"]["opt_minus_observed"],
        second_half_first_half_params=ex["wins_lost_test_train_params"]["opt_minus_observed"],
        second_half_own_params=ex["wins_lost_test_test_params"]["opt_minus_observed"],
        model=MAIN, dp="score-aware (opt_score)")
    res["frozen_crossfit"] = out
    json.dump(res, open("results/oos.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "frozen":       # HONZA-130
    frozen_main(int(sys.argv[2]) if len(sys.argv) > 2 else 8, engine="engine" in sys.argv[3:],
                world="test" if "world_test" in sys.argv[3:] else "train")
elif __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    res = {}
    m = pd.read_parquet("data/called_2026_w.parquet")
    dates = pd.read_parquet("data/feed_games_2026.parquet")[["game_pk", "date"]]
    m = m.drop(columns=[c for c in ("date",) if c in m]).merge(dates, on="game_pk")
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
    # HONZA-122 (Opus, Sol, Astra reviews): cross-fitted constant -- chosen on Mar-Jun with Mar-Jun parameters, then
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
