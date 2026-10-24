"""HONZA-130: analyses requested by the five referees. Each subcommand writes its own key of results/referee130.json
(read-modify-write under a file lock; other keys are kept).

All policy evaluations reuse the headline machinery: sim.load / dp_score.add_dstart / sim.opportunities /
sim.cal_groups for the frame, sim.draw for (truth, signal, calibrated belief), dp_score.solve_dp_score for the benchmark
and policies.run_path for the path-sum accounting (wins per team-season = sum / 30, as headline.summarise). With
seed 1 the draws are the first R draws of headline.evaluate_all(seed=1).

usage: python src/referee130.py <cmd> [args]
  decomp [R]          A  gap decomposition inside the teams' own rule class s > kappa(x) (intercept / stake terms / both),
                         tuned on draws 0..R/2-1, evaluated on draws R/2..R-1 (and in-sample)
  slopes [R]          B  threshold-slope regressions by stake range (log g > -5 vs <= -5), teams and optimum; kappa scale
  pclip [R]           E  belief clipped to [pi, 1 - pi] vs unclipped, same draws
  months              F  wins lost by calendar month from data/headline_point.pkl (R = 40 point estimate)
  moot                G  called pitches whose call was moot (a baserunning out on the same pitch ended the inning)
  wp_shrink [R]       D  WP-engine shrinkage sensitivity (weight 0, 100; toward the 0-0 table at 20)
  crossfit [R] [S]    C  optimiser's curse: stakes / engine from the other half of the games
  re_quad <gh>        H  random-effects model refitted with gh-point quadrature (warm start); + calibration table
  lapse_state         I  lapse rate by stake group (lik spec key lapse_low)"""
import fcntl, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "src")

OUT = "results/referee130.json"
SEASON_TEAM_GAMES = 161.7          # 2 x 2,425 games / 30 teams


def save(key, val):
    """merge val under key into results/referee130.json (file-locked)."""
    with open(OUT + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.load(open(OUT)) if os.path.exists(OUT) else {}
        d[key] = val
        tmp = OUT + ".tmp"
        json.dump(d, open(tmp, "w"), indent=1, default=float)
        os.replace(tmp, OUT)
        fcntl.flock(lk, fcntl.LOCK_UN)
    print(json.dumps({key: val}, indent=1, default=float)[:6000], flush=True)


# ----------------------------------------------------------------------------------------------- common machinery
def context(frame=None, variant=None):
    """the evaluate_all frame (main model unless variant given) and the run_path state arrays."""
    import sim, dp_score
    from headline import spec_variant, MAIN
    v = dict(variant or spec_variant(MAIN))
    if frame is not None:
        v["frame"] = frame
    m, _ = sim.load(v)
    m = dp_score.add_dstart(m)
    m = sim.opportunities(m)
    m["new_extra"] = sim.extra_starts(m)
    m["cal"] = sim.cal_groups(m)
    kin = sim.kin_index(m).astype(np.int64)
    ng = int(m.gidx.max() + 1)
    base = (m.gidx.values, m.is_home_team.values, m.new_extra.values, m.slot.values.astype(np.int64), kin,
            m.inning.values.astype(np.int64), m.balls.values.astype(np.int64), m.strikes.values.astype(np.int64),
            m.dbin.values.astype(np.int64))
    return dict(v=v, m=m, kin=kin, ng=ng, base=base, isoff=(m.side == "off").values, K=sim.kappa_by_count(m))


Z19 = np.zeros((19, 80, 3)); ZC = np.zeros((19, 3)); ZCARD = np.zeros((4, 3, 2, 2))


def run(ctx, W, s, pt, pol, dconst=0.0, K=None, D4h=None, D4a=None, g=None):
    """policies.run_path with unused arguments zeroed; returns value, challenges [game, team, side]."""
    import policies as P
    m = ctx["m"]
    D4h = Z19[:, None] if D4h is None else D4h; D4a = Z19[:, None] if D4a is None else D4a
    gd = m.g.values if g is None else g
    gv = m.gv.values if (g is None and "gv" in m) else gd        # HONZA-132: cross-fitted frames score at gv
    v, n, _, _, _ = P.run_path2(*ctx["base"], W, pt, s, gd, gv, m.RV.values,
                                ctx["K"] if K is None else K, ctx["isoff"], pol, Z19, Z19, ZC, ZC, ZCARD, 0.0, 0.0,
                                dconst, D4h, D4a, ctx["ng"])
    return v, n


def observed(ctx, gcol="g"):
    """realised value of the recorded challenges, per [game, team, side] (as headline.evaluate_all)."""
    import sim
    m, ng = ctx["m"], ctx["ng"]
    mm, _ = sim.load(ctx["v"])
    obs = np.zeros((ng, 2, 2)); obsn = np.zeros((ng, 2, 2))
    ch = mm[mm.ch]
    gmap = dict(zip(m.game_pk.unique(), range(ng)))
    gi = ch.game_pk.map(gmap).values; ti = ch.is_home_team.values.astype(int); si = (ch.side == "off").values.astype(int)
    np.add.at(obs, (gi, ti, si), (ch[gcol] * ch.ch_over).values); np.add.at(obsn, (gi, ti, si), 1)
    return obs, obsn


def worlds(ctx, R, seed=1, dp=True, clip=None):
    """R draws (W, s, pt, D4h, D4a) with the evaluate_all random stream (seed)."""
    import sim, dp_score
    rng = np.random.default_rng(seed); out = []
    for r in range(R):
        W, s, pt = sim.draw(ctx["m"], rng, ctx["v"])
        if clip is not None:
            pt = clip(pt)
        d = dict(W=W, s=s, pt=pt)
        if dp:
            d["D4h"], _ = dp_score.solve_dp_score(ctx["m"], pt, True); d["D4a"], _ = dp_score.solve_dp_score(ctx["m"], pt, False)
        out.append(d)
    return out


def wins(v):
    return float(v.sum() / 30)


def per_team_game(n, ng):
    return float(n.sum() / (2 * ng))


# ----------------------------------------------------------------------------------------------- A: decomposition
def stake_part(m, th=None):
    """the stake terms of the fitted criterion kappa(x): beta_logg log g + hinges kg (lik.design / lik.gfeat), by row.
    th: the main model's parameters (default the point fit; HONZA-132: a bootstrap refit)."""
    import lik
    from final_ci import spec_of
    from headline import MAIN
    spec = spec_of(MAIN); P = lik.unpack(np.load(f"data/th_{MAIN}.npy") if th is None else np.asarray(th), spec)
    lg = np.log(np.clip(m.g.values, 1e-4, None)); G = lik.gfeat(m)
    sp = np.zeros(len(m)); dk = np.zeros(len(m))
    for side in ("off", "def"):
        k = (m.side == side).values; q = P[side]
        b1 = float(q["beta"][1]); kg = np.asarray(q["kg"])
        sp[k] = b1 * lg[k] + G[k] @ kg
        # d kappa / d log g (0 where log g is clipped at log 1e-4)
        dk[k] = (b1 + (lg[k][:, None] > np.asarray(lik.G_KNOTS)[None, :]) @ kg) * (m.g.values[k] > 1e-4)
    return sp, dk, lg


def decomp(R=20, only=None, ctx=None, th=None, store=True, seed=1):
    """HONZA-132: ctx / th / store / seed let mechanism_boot.py run it on a bootstrap resample (returns the result)."""
    ctx = ctx or context(); m = ctx["m"]; ng = ctx["ng"]
    t = time.time(); dr = worlds(ctx, R, seed=seed); print("draws", round(time.time() - t), flush=True)
    obs = observed(ctx, "gv" if "gv" in ctx["m"] else "g")
    sp, _, lg = stake_part(m, th)
    isoff = ctx["isoff"]
    # centre the stake part (and log g) within side over the opportunities, so that scaling it is a pure re-ranking of
    # pitches around the side's average criterion (the level is carried by the intercept)
    spc = sp.copy(); lgc = lg.copy()
    for k in (isoff, ~isoff):
        spc[k] -= sp[k].mean(); lgc[k] -= lg[k].mean()
    K0 = ctx["K"]
    half = R // 2
    IN, OUTS = list(range(half)), list(range(half, R))

    def kappa(p, form):
        a = np.where(isoff, p["a_off"], p["a_def"])
        if form == "scale":
            add = a + np.where(isoff, p["f_off"] - 1, p["f_def"] - 1) * spc
        else:                                         # linear: stake terms replaced by b (log g - mean)
            add = a - spc + np.where(isoff, p["b_off"], p["b_def"]) * lgc
        return np.ascontiguousarray(K0 + add[:, None])

    def evalp(p, form, idx):
        K = kappa(p, form); v = n = 0
        for r in idx:
            vv, nn = run(ctx, dr[r]["W"], dr[r]["s"], dr[r]["pt"], 4, K=K)
            v = v + vv / len(idx); n = n + nn / len(idx)
        return v, n

    def obj(p, form, idx):
        return wins(evalp(p, form, idx)[0])

    def search(free, form, idx, start):
        p = dict(start); best = obj(p, form, idx)
        grids = dict(a_off=np.arange(-4, 4.01, 0.25), a_def=np.arange(-4, 4.01, 0.25), f_off=np.arange(-1, 12.01, 0.5),
                     f_def=np.arange(-1, 12.01, 0.5), b_off=np.arange(-6, 2.01, 0.25), b_def=np.arange(-6, 2.01, 0.25))
        for sweep in range(4):
            changed = False
            for k in free:
                step = grids[k][1] - grids[k][0]
                cand = grids[k] if sweep == 0 else p[k] + step * np.arange(-4, 4.01, 1) / (2 ** min(sweep, 2))
                vals = []
                for x in cand:
                    q = dict(p, **{k: float(x)}); vals.append(obj(q, form, idx))
                j = int(np.argmax(vals))
                if vals[j] > best + 1e-9:
                    best = vals[j]; p[k] = float(cand[j]); changed = True
            if not changed and sweep > 0:
                break
        return p, best

    b_teams = {}
    for side, k in (("off", isoff), ("def", ~isoff)):             # teams' own average d kappa / d log g, for the linear form
        b_teams[side] = float(np.polyfit(lg[k], sp[k], 1)[0])
    start_s = dict(a_off=0.0, a_def=0.0, f_off=1.0, f_def=1.0)
    start_l = dict(a_off=0.0, a_def=0.0, b_off=b_teams["off"], b_def=b_teams["def"])
    # benchmarks on the same draws
    bench = {}
    for name, idx in (("in", IN), ("out", OUTS), ("all", list(range(R)))):
        vo = no = vs = ns = 0
        for r in idx:
            d = dr[r]
            a, b = run(ctx, d["W"], d["s"], d["pt"], 7, D4h=d["D4h"], D4a=d["D4a"]); vo = vo + a / len(idx); no = no + b / len(idx)
            a, b = run(ctx, d["W"], d["s"], d["pt"], 2, dconst=0.008); vs = vs + a / len(idx); ns = ns + b / len(idx)
        bench[name] = dict(opt_score=(wins(vo), per_team_game(no, ng)), stake_0008=(wins(vs), per_team_game(ns, ng)))
    W_obs = wins(obs[0]); n_obs = per_team_game(obs[1], ng)
    res = dict(R=R, R_tune=half, R_eval=R - half, form_note=(
        "rule s > kappa(x) + a_side + (f_side - 1) * stake_c(x) [scale] or kappa(x) + a_side - stake_c(x) + b_side * "
        "(log g - mean) [linear]; stake_c = beta_logg log g + hinges kg, centred within side over opportunities; "
        "kappa(x) at the challenges the simulated team holds (sim.kappa_by_count)"),
               observed=dict(wins=W_obs, n=n_obs), teams_linear_slope_in_per_logg=b_teams, rows={})
    cases = [("fitted_rule", "scale", []), ("intercept_only", "scale", ["a_off", "a_def"]),
             ("stake_scale_only", "scale", ["f_off", "f_def"]), ("both_scale", "scale", ["a_off", "a_def", "f_off", "f_def"]),
             ("stake_linear_only", "linear", ["b_off", "b_def"]), ("both_linear", "linear", ["a_off", "a_def", "b_off", "b_def"])]
    prev = (json.load(open(OUT)).get("decomp", {}).get("rows", {}) if os.path.exists(OUT) else {}) if only else {}
    for name, form, free in cases:
        if only and name not in only:
            res["rows"][name] = prev[name]; continue
        t = time.time()
        start = start_s if form == "scale" else start_l
        if name.startswith("both"):
            # the joint objective is flat and the coordinate search can stall when it moves the intercept first: start
            # from each single-component optimum (and their combination) and keep the best in-sample result
            sk = "stake_scale_only" if form == "scale" else "stake_linear_only"
            starts = [res["rows"][sk]["params"], dict(res["rows"][sk]["params"], a_off=res["rows"]["intercept_only"]["params"]["a_off"],
                                                       a_def=res["rows"]["intercept_only"]["params"]["a_def"])]
            cands = [search(free, form, IN, st) for st in starts]
            p = max(cands, key=lambda c: c[1])[0]
        else:
            p, _ = search(free, form, IN, start) if free else (dict(start), None)
        row = dict(params=p)
        for lab, idx in (("in_sample", IN), ("out_of_sample", OUTS)):
            v, n = evalp(p, form, idx); Wv = wins(v); O = bench["in" if lab == "in_sample" else "out"]["opt_score"][0]
            row[lab] = dict(wins=Wv, batter=float(v[:, :, 1].sum() / 30), fielder=float(v[:, :, 0].sum() / 30),
                            share_opt=Wv / O, gap_closed=(Wv - W_obs) / (O - W_obs), n=per_team_game(n, ng),
                            n_batter=float(n[:, :, 1].sum() / (2 * ng)), n_fielder=float(n[:, :, 0].sum() / (2 * ng)),
                            lost_vs_opt=O - Wv)
        # full-sample re-optimisation (all R draws) for reference
        if free:
            pa, _ = search(free, form, list(range(R)), p)
            v, n = evalp(pa, form, list(range(R))); O = bench["all"]["opt_score"][0]
            row["tuned_on_all"] = dict(params=pa, wins=wins(v), share_opt=wins(v) / O, gap_closed=(wins(v) - W_obs) / (O - W_obs),
                                       n=per_team_game(n, ng))
        res["rows"][name] = row
        print(name, round(time.time() - t), p, {k: round(x, 3) for k, x in row["out_of_sample"].items()}, flush=True)
    for lab, key in (("in_sample", "in"), ("out_of_sample", "out"), ("all", "all")):
        O, nO = bench[key]["opt_score"]; S, nS = bench[key]["stake_0008"]
        res[f"benchmarks_{lab}"] = dict(opt_score=dict(wins=O, n=nO), stake_0008=dict(wins=S, n=nS, share_opt=S / O,
                                        gap_closed=(S - W_obs) / (O - W_obs)), lost=O - W_obs)
    if store:
        save("decomp", res)
    return res


# ----------------------------------------------------------------------------------------------- B: slopes
def slopes(R=4, ctx=None, th=None, store=True, seed=1):
    """HONZA-132: ctx / th / store / seed for mechanism_boot.py; controls "count_fe" (Astra #9) replace the two count
    indicators by the twelve count cells."""
    import dp_score
    from postseason import cal_maps, apply_maps
    ctx = ctx or context(); m = ctx["m"]
    maps = cal_maps(m, np.random.default_rng(10 + seed), n=5)     # as postseason.main (teams' revealed p*)
    rng = np.random.default_rng(seed); D = np.zeros(len(m))       # as opt_slope.main
    import sim
    for r in range(R):
        _, _, pt = sim.draw(m, rng, ctx["v"])
        D4h, _ = dp_score.solve_dp_score(m, pt, True); D4a, _ = dp_score.solve_dp_score(m, pt, False)
        c = np.clip(m.left.values, 1, 2); k = np.minimum(ctx["kin"], D4h.shape[2] - 1)
        D += np.where(m.is_home_team.values, D4h[m.slot.values, m.dbin.values, k, c], D4a[m.slot.values, m.dbin.values, k, c]) / R
    sp, dk, lg = stake_part(m, th)

    def inv_map(side, grp, p):
        """smallest signal (bin lower edge, in) at which the calibrated belief reaches p, cell by cell."""
        out = np.full(len(p), np.nan)
        for e in np.unique(grp):
            k = grp == e; mp = maps[(side, e)]
            j = np.searchsorted(np.maximum.accumulate(mp), p[k])
            out[k] = np.where(j < len(mp), -15 + 0.1 * np.minimum(j, len(mp) - 1), np.nan)
        return out

    def reg(y, x, controls):
        lgx = np.log(np.clip(x.g.values, 1e-4, None))
        cols = [np.ones(len(x)), lgx]
        if controls in ("full", "no_count"):
            cols += [(x.left == 1).astype(float), np.clip(x.inning.values, 1, 10) / 9]
        if controls == "full":
            cols += [(x.strikes == 2).astype(float), (x.balls == 3).astype(float)]
        if controls == "count_fe":                                # HONZA-132 (Astra #9): exact count cells
            cols += [(x.left == 1).astype(float), np.clip(x.inning.values, 1, 10) / 9]
            cnt = (x.balls.values * 3 + x.strikes.values).astype(int)
            cols += [(cnt == c).astype(float) for c in range(1, 12)]
        ok = np.isfinite(y)
        return float(np.linalg.lstsq(np.column_stack(cols)[ok], y[ok], rcond=None)[0][1])

    out = dict(R_opt=R, note=("logit scale: logit(1 - p*) on log g (postseason.implied_slope / opt_slope.py); controls full = "
                              "challenges left, inning, two strikes, three balls; no_count = without the two count indicators; "
                              "none = log g only. Ranges: high = log g > -5 (top ~60 % of eligible pitches), low = log g <= -5. "
                              "kappa scale (belief-map free): d kappa / d log g of the fitted criterion averaged over the "
                              "eligible pitches, and OLS of kappa on log g; s_opt: signal threshold (in) at which the "
                              "calibrated belief reaches the optimum's p* = D/(g + D)"), sides={})
    for side in ("off", "def"):
        e = (m.side == side).values & (m.left.values > 0)
        x = m[e]; g = np.clip(x.g.values, 1e-4, None); lx = np.log(g)
        grp = x.cal.values
        p_team = np.clip(apply_maps(maps, side, grp, x.kappa.values), 1e-3, 1 - 1e-3)
        p_opt = np.clip(D[e] / (g + D[e]), 1e-3, 1 - 1e-3)
        y_team = np.log((1 - p_team) / p_team); y_opt = np.log((1 - p_opt) / p_opt)
        s_opt = inv_map(side, grp, D[e] / (x.g.values + D[e] + 1e-12))
        S = dict(n=int(e.sum()), share_high=float((lx > -5).mean()))
        for rng_name, k in (("all", np.ones(len(x), bool)), ("high", lx > -5), ("low", lx <= -5)):
            xx = x[k]
            r = dict(n=int(k.sum()))
            for ctl in ("full", "count_fe", "no_count", "none"):
                r[f"teams_logit_{ctl}"] = reg(y_team[k], xx, ctl)
                r[f"opt_logit_{ctl}"] = reg(y_opt[k], xx, ctl)
                r[f"teams_kappa_ols_{ctl}"] = reg(x.kappa.values[k], xx, ctl)
                r[f"opt_signal_threshold_ols_{ctl}"] = reg(s_opt[k], xx, ctl)
            r["teams_dkappa_dlogg_mean"] = float(dk[e][k].mean())
            r["mean_pstar_team"] = float(p_team[k].mean()); r["mean_pstar_opt"] = float(p_opt[k].mean())
            S[rng_name] = r
        out["sides"][side] = S
    if store:
        save("slopes", out)
    return out


# ----------------------------------------------------------------------------------------------- E: pclip
def pclip(R=10):
    from headline import STAKES
    ctx = context(); m = ctx["m"]; ng = ctx["ng"]
    pi = m.pi.values
    import sim, dp_score
    rng = np.random.default_rng(1)
    obs = observed(ctx); Wobs = wins(obs[0])
    acc = {f: {} for f in ("raw", "clip")}; cnt = dict(p1=0.0, p0=0.0, p_gt_1mpi=0.0, p_lt_pi=0.0, p1_wrong=0.0)
    for r in range(R):
        W, s, pt = sim.draw(m, rng, ctx["v"])
        cnt["p1"] += (pt >= 1.0).sum() / R; cnt["p0"] += (pt <= 0.0).sum() / R
        cnt["p_gt_1mpi"] += (pt > 1 - pi).sum() / R; cnt["p_lt_pi"] += (pt < pi).sum() / R
        cnt["p1_wrong"] += ((pt >= 1.0) & ~W).sum() / R
        for f, p in (("raw", pt), ("clip", np.clip(pt, pi, 1 - pi))):
            D4h, _ = dp_score.solve_dp_score(m, p, True); D4a, _ = dp_score.solve_dp_score(m, p, False)
            pols = dict(opt_score=(7, 0.0), obs_rule=(4, 0.0), **{f"stake_{d}": (2, d) for d in STAKES})
            for name, (pc, dc) in pols.items():
                v, n = run(ctx, W, s, p, pc, dconst=dc, D4h=D4h, D4a=D4a)
                a = acc[f].setdefault(name, [0.0, 0.0]); a[0] += wins(v) / R; a[1] += per_team_game(n, ng) / R
        print(r, {f: round(acc[f]["opt_score"][0] * R / (r + 1) - Wobs, 3) for f in acc}, flush=True)
    res = dict(R=R, pi=float(pi[0]), n_opportunities=len(m), counts_per_draw=cnt, observed=Wobs)
    for f in acc:
        a = acc[f]; best = max((k for k in a if k.startswith("stake_")), key=lambda k: a[k][0])
        res[f] = dict(opt_score=a["opt_score"][0], lost=a["opt_score"][0] - Wobs, obs_rule=a["obs_rule"][0],
                      stake_0008=a["stake_0.008"][0], share_0008=a["stake_0.008"][0] / a["opt_score"][0], best_stake=best,
                      best_share=a[best][0] / a["opt_score"][0], n_opt=a["opt_score"][1], n_obs_rule=a["obs_rule"][1])
    res["change_lost"] = res["clip"]["lost"] - res["raw"]["lost"]
    res["change_share_0008"] = res["clip"]["share_0008"] - res["raw"]["share_0008"]
    save("pclip", res)


# ----------------------------------------------------------------------------------------------- F: months
def months():
    import pickle
    from headline import MAIN
    res = pickle.load(open("data/headline_point.pkl", "rb"))[MAIN]
    out, games = res["out"], np.asarray(res["games"])
    dates = pd.read_parquet("data/feed_games_2026.parquet")[["game_pk", "date"]].drop_duplicates("game_pk").set_index("game_pk").date
    d = pd.to_datetime(dates.reindex(games).values)
    assert not pd.isna(d).any()
    lost_g = (out["opt_score"][0] - out["observed"][0]).sum(axis=(1, 2))          # per game, both teams
    opt_g = out["opt_score"][0].sum(axis=(1, 2)); obs_g = out["observed"][0].sum(axis=(1, 2))
    rule_g = out["stake_0.008"][0].sum(axis=(1, 2))
    mon = np.where(d.month <= 4, "Mar/Apr", d.strftime("%b"))
    asb = pd.Timestamp("2026-07-14")                              # 2026 All-Star Game (Philadelphia)
    rng = np.random.default_rng(130)

    def block(k):
        n = int(k.sum()); sc = SEASON_TEAM_GAMES / (2 * n)
        idx = np.flatnonzero(k); bs = [lost_g[rng.choice(idx, n)].sum() * sc for _ in range(1000)]
        return dict(games=n, lost_per_team_season=float(lost_g[k].sum() * sc), boot_se=float(np.std(bs, ddof=1)),
                    opt_per_team_season=float(opt_g[k].sum() * sc), observed_per_team_season=float(obs_g[k].sum() * sc),
                    observed_share_opt=float(obs_g[k].sum() / opt_g[k].sum()), rule_share_opt=float(rule_g[k].sum() / opt_g[k].sum()))
    r = dict(R=res["mc"]["R"], note="per team-season equivalent = month sum x 161.7 / (2 x games in the month); boot_se = game "
             "bootstrap within the block at the point draws (no refit)", months={})
    for mo in ["Mar/Apr", "May", "Jun", "Jul", "Aug", "Sep"]:
        r["months"][mo] = block(mon == mo)
    r["first_half_pre_asb"] = block(d < asb); r["second_half_post_asb"] = block(d >= asb)
    med = np.sort(d)[len(d) // 2]
    r["median_date"] = str(med.date() if hasattr(med, "date") else med)
    r["first_half_games"] = block(d < med); r["second_half_games"] = block(d >= med)
    r["season"] = block(np.ones(len(d), bool))
    x = np.array([r["months"][k]["lost_per_team_season"] for k in r["months"]]); t = np.arange(len(x))
    r["linear_trend_per_month"] = float(np.polyfit(t, x, 1)[0])
    save("months", r)


# ----------------------------------------------------------------------------------------------- G: moot
BASERUN_OUT = r"^(caught_stealing|pickoff|other_out)"


def feed_out_index(pk_abs):
    """for each (game, at_bat_number): index of the last pitch event and the playIndex / out number of every runner
    out on the play, from the MLB Stats API live feed (cached in data/rerun130/moot_feed.json)."""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    path = "data/rerun130/moot_feed.json"
    cache = json.load(open(path)) if os.path.exists(path) else {}
    S = requests.Session()

    def one(pk):
        for k in range(5):
            try:
                d = S.get(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live", timeout=60).json(); break
            except Exception:
                time.sleep(2 * (k + 1))
        else:
            return pk, None
        res = {}
        for p in d["liveData"]["plays"]["allPlays"]:
            ab = p["about"]["atBatIndex"] + 1
            pit = [i for i, e in enumerate(p["playEvents"]) if e.get("isPitch")]
            outs = [(r["details"].get("playIndex"), r["movement"].get("outNumber"), r["details"].get("eventType"))
                    for r in p.get("runners", []) if r["movement"].get("isOut")]
            res[str(ab)] = dict(last_pitch=pit[-1] if pit else None, n_events=len(p["playEvents"]), outs=outs,
                                event=p["result"].get("eventType"))
        return pk, res
    todo = sorted({int(pk) for pk, _ in pk_abs if str(pk) not in cache})
    with ThreadPoolExecutor(8) as ex:
        for pk, r in ex.map(one, todo):
            if r is not None:
                cache[str(pk)] = r
    json.dump(cache, open(path, "w"))
    return cache


def moot():
    m = pd.read_parquet("data/called_2026_w.parquet")
    f = pd.read_parquet("data/feed_pitches_2026.parquet", columns=["game_pk", "at_bat_number", "pitch_number"])
    last = f.groupby(["game_pk", "at_bat_number"]).pitch_number.max().rename("last_pn").reset_index()
    x = m.merge(last, on=["game_pk", "at_bat_number"], how="left")
    called = x.dist_in.notna().values & ~x.pos_pitcher.astype(bool).values
    cand = called & (x.pitch_number == x.last_pn).values & x.pa_event.fillna("").str.match(BASERUN_OUT).values
    c = x[cand].copy()
    feed = feed_out_index(list(zip(c.game_pk, c.at_bat_number)))
    same, later, third = [], [], []
    for r in c.itertuples():
        p = feed.get(str(int(r.game_pk)), {}).get(str(int(r.at_bat_number)))
        if p is None:
            same.append(None); later.append(None); third.append(None); continue
        o3 = [o for o in p["outs"] if o[1] == 3]
        third.append(bool(o3))
        same.append(bool(o3) and o3[0][0] == p["last_pitch"])
        later.append(bool(o3) and o3[0][0] is not None and p["last_pitch"] is not None and o3[0][0] > p["last_pitch"])
    c["third_out"] = third; c["same_pitch"] = same; c["after_pitch"] = later
    o = x[called]; G = float(o.g.sum()); Gch = float(o[o.ch].g.sum())

    def stats(k):
        y = c[k]
        return dict(n=int(len(y)), events=y.pa_event.value_counts().to_dict(), n_challenged=int(y.ch.fillna(False).sum()),
                    n_overturned=int((y.ch.fillna(False) & y.ch_over.fillna(False)).sum()), stake=float(y.g.sum()),
                    share_of_total_stake=float(y.g.sum() / G),
                    stake_challenged=float(y[y.ch.fillna(False).astype(bool)].g.sum()),
                    share_of_challenged_stake=float(y[y.ch.fillna(False).astype(bool)].g.sum() / Gch),
                    stake_overturned=float(y[(y.ch.fillna(False) & y.ch_over.fillna(False)).astype(bool)].g.sum()),
                    mean_g=float(y.g.mean()) if len(y) else None)
    sp = c.same_pitch.fillna(False).astype(bool); ap = c.after_pitch.fillna(False).astype(bool)
    res = dict(n_called_located=int(called.sum()), stake_total=G, mean_g_all=float(o.g.mean()),
               candidates_last_pitch_of_baserunning_out_pa=int(cand.sum()), feed_missing=int(c.same_pitch.isna().sum()),
               moot_same_pitch=stats(sp), third_out_after_the_pitch=stats(ap),
               examples=c[sp][["game_pk", "at_bat_number", "pitch_number", "call", "pa_event", "ch", "ch_over", "g"]].head(8).astype(str).to_dict("records"),
               note=("candidates: located called pitch (no position player pitching) that is the last pitch of a PA whose feed "
                     "event is a baserunning out (caught stealing, pickoff, other out). moot_same_pitch: the third out was "
                     "made on that very pitch (runner playIndex = last pitch event, MLB live feed): the count was wiped "
                     "on the same play, so the ball/strike call could not matter. third_out_after_the_pitch: the third out "
                     "came on a later non-pitch event (pickoff between pitches) -- the call was live when made and is "
                     "not moot ex ante. Statcast marks these PAs truncated_pa."))
    save("moot", res)


# ----------------------------------------------------------------------------------------------- D: WP-engine shrinkage
def engine_frame(T, sc_games=None):
    """stakes of every called pitch under the PA kernel T (wp.solve -> stakes globals -> stakes.add_stakes)."""
    import wp, stakes
    W, RE, D = wp.solve(T, verbose=False)
    stakes.T, stakes.W, stakes.RE, stakes.D = T, W, RE, int(D)
    m = pd.read_parquet("data/called_2026_w.parquet")
    cols = ["wp_ball", "wp_strike", "re_ball", "re_strike", "g", "g_signed", "RV", "wp_home_now", "half", "diff"]
    keep = m[["game_pk", "at_bat_number", "pitch_number"]].copy()
    mb = stakes.add_stakes(m.drop(columns=[c for c in cols if c in m]))
    mb = keep.merge(mb, on=["game_pk", "at_bat_number", "pitch_number"], how="left")   # add_stakes merges: restore order
    return mb


def nonmono(mb):
    k = mb.g_signed < 0
    keys = ["inning", "half", "outs_when_up", "bases", "diff", "balls", "strikes"]
    return dict(rows=int(k.sum()), rows_opportunity=int((k & ~mb.pos_pitcher.astype(bool)).sum()),
                challenged=int((k & mb.ch).sum()), distinct_state_cells=int(mb.loc[k, keys].drop_duplicates().shape[0]),
                distinct_count_outs_bases=int(mb.loc[k, ["balls", "strikes", "outs_when_up", "bases"]].drop_duplicates().shape[0]),
                min_g_signed=float(mb.g_signed.min()))


def evaluate_frame(mb, R, seed=1):
    """opt_score, stake grid, obs_rule and observed on a frame (perception parameters fixed at the main fit)."""
    from headline import STAKES
    ctx = context(frame=mb); ng = ctx["ng"]
    obs = observed(ctx); Wobs = wins(obs[0])
    acc = {}
    for d in worlds(ctx, R, seed):
        pols = dict(opt_score=(7, 0.0), obs_rule=(4, 0.0), **{f"stake_{x}": (2, x) for x in STAKES})
        for name, (pc, dc) in pols.items():
            v, n = run(ctx, d["W"], d["s"], d["pt"], pc, dconst=dc, D4h=d["D4h"], D4a=d["D4a"])
            a = acc.setdefault(name, [0.0, 0.0]); a[0] += wins(v) / R; a[1] += per_team_game(n, ng) / R
    best = max((k for k in acc if k.startswith("stake_")), key=lambda k: acc[k][0])
    O = acc["opt_score"][0]
    return dict(opt_score=O, observed=Wobs, lost=O - Wobs, share_0008=acc["stake_0.008"][0] / O, best_stake=best,
                best_share=acc[best][0] / O, obs_rule=acc["obs_rule"][0], n_opt=acc["opt_score"][1])


def wp_shrink(R=10):
    import wp
    res = dict(R=R, note=("PA kernel pooled with a target table, weight w pitches (wp._kernel): main = all-count pooled "
                          "table (T.sum(axis=0)) at w = 20. Engine, stakes g and RV recomputed; perception parameters fixed at the main "
                          "fit (kappa re-evaluated at the new g); DP laws (half-inning runs) unchanged"), engines={})
    for name, kw in (("main_pooled_w20", dict()), ("pooled_w0", dict(shrink_w=0)), ("pooled_w100", dict(shrink_w=100)),
                     ("zero_zero_w20", dict(shrink_to="00"))):
        t = time.time()
        T = wp.build_T(verbose=False, **kw)
        mb = engine_frame(T)
        r = dict(nonmonotone=nonmono(mb))
        if name == "main_pooled_w20":
            m0 = pd.read_parquet("data/called_2026_w.parquet")
            r["max_abs_diff_g_vs_stored"] = float(np.nanmax(np.abs(mb.g.values - m0.g.values)))
            r["nonmonotone_stored_frame"] = nonmono(m0)
        r.update(evaluate_frame(mb, R))
        res["engines"][name] = r
        print(name, round(time.time() - t), {k: v for k, v in r.items() if k != "nonmonotone"}, r["nonmonotone"], flush=True)
        save("wp_shrink", res)


# ----------------------------------------------------------------------------------------------- C: cross-fit stakes
def half_engine(sc):
    """PA kernel, WP/RE engine, half-inning and extra top-half run laws from a Statcast subset (as joint_boot.job)."""
    import wp, dp_score
    T = wp.build_T(sc=sc, verbose=False); W, RE, D = wp.solve(T, verbose=False)
    P = dp_score.half_runs_from(sc)
    x = sc[(sc.inning >= 10) & (sc.inning_topbot == "Top")].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    h = x.groupby(["game_pk", "inning"]).agg(a=("bat_score", "first"), z=("post_bat_score", "last"))
    rx = (h.z - h.a).clip(0, dp_score.RMAX).astype(int).values
    Px = np.bincount(rx, minlength=dp_score.RMAX + 1) / max(len(rx), 1)
    return dict(T=T, W=W, RE=RE, D=int(D), P=P, Px=Px)


def stakes_with(eng, m):
    import stakes
    stakes.T, stakes.W, stakes.RE, stakes.D = eng["T"], eng["W"], eng["RE"], eng["D"]
    cols = ["wp_ball", "wp_strike", "re_ball", "re_strike", "g", "g_signed", "RV", "wp_home_now", "half", "diff"]
    keep = m[["game_pk", "at_bat_number", "pitch_number"]].copy()
    mb = stakes.add_stakes(m.drop(columns=[c for c in cols if c in m]))
    return keep.merge(mb, on=["game_pk", "at_bat_number", "pitch_number"], how="left")


def crossfit(R=10, S=2):
    import dp_score, wp
    m_all = pd.read_parquet("data/called_2026_w.parquet")
    sc = pd.read_parquet("data/statcast_2026.parquet", columns=sorted(set(wp.SC_COLS) | {"home_score", "away_score", "post_bat_score"}))
    games = np.array(sorted(m_all.game_pk.unique()))
    hr0, ex0 = dp_score.half_runs, dp_score.extra_runs
    res = dict(R=R, note=("games split at random into halves A/B (seed 130 + split). Same-sample: decisions (DP pools, "
                          "thresholds, calibration cells) and scoring on the stakes g of the half's own engine (PA kernel + WP "
                          "engine + run laws from that half's Statcast games). Cross-fit: decisions on the other half's engine "
                          "and run laws, scoring on the own-half g. Observed scored with the own-half g. Same (truth, signal) "
                          "draws in both versions. bias = lost_same - lost_cross (optimism of selecting and scoring on one g)"),
               splits=[])
    for sp in range(S):
        rng = np.random.default_rng(130 + sp)
        A = set(rng.choice(games, len(games) // 2, replace=False)); B = set(games) - A
        engs = {h: half_engine(sc[sc.game_pk.isin(H)]) for h, H in (("A", A), ("B", B))}
        row = dict(split=sp)
        for ev_half, other in (("A", "B"), ("B", "A")):
            H = A if ev_half == "A" else B
            mh = m_all[m_all.game_pk.isin(H)]
            g_own = stakes_with(engs[ev_half], mh); g_oth = stakes_with(engs[other], mh)
            r = {}
            for ver, dec, laws in (("same", g_own, engs[ev_half]), ("cross", g_oth, engs[other])):
                dp_score.half_runs = (lambda P=laws["P"]: P); dp_score.extra_runs = (lambda Px=laws["Px"]: Px)
                try:
                    ctx = context(frame=dec)
                    # scoring stakes of the own half, aligned to ctx rows
                    key = ["game_pk", "at_bat_number", "pitch_number"]
                    gs = ctx["m"][key].merge(g_own[key + ["g"]], on=key, how="left").g.values
                    ctx_obs = dict(ctx, v=dict(ctx["v"], frame=g_own))
                    obs = observed(ctx_obs); Wobs = wins(obs[0])
                    vo = vs = 0.0; no = 0.0
                    for d in worlds(ctx, R, seed=sp + 1):
                        v, n = run(ctx, d["W"], d["s"], d["pt"], 7, D4h=d["D4h"], D4a=d["D4a"], g=None)
                        # re-score with the own-half g: run_path adds g[i]; rerun with scoring g via the selection
                        v2 = rescore(ctx, d, gs)
                        vo += wins(v2) / R; no += per_team_game(n, ctx["ng"]) / R
                        v3 = rescore(ctx, d, gs, pol=2, dconst=0.008); vs += wins(v3) / R
                finally:
                    dp_score.half_runs, dp_score.extra_runs = hr0, ex0
                # per team-season of the half: x 2 (each half is about half a season)
                r[ver] = dict(opt_score=vo, observed=Wobs, lost=vo - Wobs, stake_0008=vs, share_0008=vs / vo, n_opt=no,
                              games=len(H), lost_full_season_equiv=(vo - Wobs) * len(games) / len(H))
            r["bias_same_minus_cross"] = r["same"]["lost"] - r["cross"]["lost"]
            r["bias_full_season_equiv"] = r["same"]["lost_full_season_equiv"] - r["cross"]["lost_full_season_equiv"]
            row[f"eval_{ev_half}"] = r
            print(sp, ev_half, {k: (round(v["lost_full_season_equiv"], 3) if isinstance(v, dict) else round(v, 4)) for k, v in r.items()}, flush=True)
        row["bias_full_season_equiv_mean"] = float(np.mean([row[f"eval_{h}"]["bias_full_season_equiv"] for h in "AB"]))
        res["splits"].append(row)
        b = np.array([x["bias_full_season_equiv_mean"] for x in res["splits"]])
        res["bias_mean"] = float(b.mean()); res["bias_sd_over_splits"] = float(b.std(ddof=1)) if len(b) > 1 else None
        allb = np.array([x[f"eval_{h}"]["bias_full_season_equiv"] for x in res["splits"] for h in "AB"])
        res["bias_all_halves"] = allb.tolist()
        res["lost_same_mean"] = float(np.mean([x[f"eval_{h}"]["same"]["lost_full_season_equiv"] for x in res["splits"] for h in "AB"]))
        res["lost_cross_mean"] = float(np.mean([x[f"eval_{h}"]["cross"]["lost_full_season_equiv"] for x in res["splits"] for h in "AB"]))
        save("crossfit_stakes", res)


def rescore(ctx, d, gs, pol=7, dconst=0.0):
    """value of the policy's challenges (decided on ctx's g) scored with stakes gs: the path is the decision path, so
    run the selection (mechanism._sel) and sum gs over the successful selected pitches."""
    from mechanism import _sel
    m = ctx["m"]
    sel = _sel(*ctx["base"], d["W"], d["pt"], d["s"], m.g.values, m.RV.values, ctx["K"], pol, Z19, Z19, ZC, ZC, ZCARD,
               0.0, 0.0, dconst, d["D4h"], d["D4a"])
    v = np.zeros((ctx["ng"], 2, 2)); k = sel & d["W"]
    np.add.at(v, (m.gidx.values[k], m.is_home_team.values[k].astype(int), ctx["isoff"][k].astype(int)), gs[k])
    return v


# ----------------------------------------------------------------------------------------------- H: RE quadrature
def re_quad(gh=9, evaluate=0):
    import lik
    from final_ci import spec_of
    m = pd.read_parquet("data/called_2026_w.parquet")
    data = lik.prepare(m, players=True)
    spec = dict(spec_of("re"), gh=gh)
    th0 = np.load(f"data/th_re_gh{gh}.npy") if os.path.exists(f"data/th_re_gh{gh}.npy") else np.load("data/th_re.npy")
    th_store = np.load("data/th_re.npy")
    t = time.time()
    import jax, jax.numpy as jnp
    from scipy.optimize import minimize
    r, nll = lik.fit(data, spec, th0=th0, verbose=True)
    # polish: L-BFGS-B stops on the relative-reduction rule with gradients ~0.1 in the flat random-effect directions
    vg = jax.jit(jax.value_and_grad(nll))
    f = lambda th: tuple(np.asarray(v, float) for v in vg(jnp.asarray(th)))
    r = minimize(f, r.x, jac=True, method="L-BFGS-B", options=dict(maxiter=500, maxcor=50, ftol=1e-15, gtol=1e-3))
    print("polished", r.fun, r.message, r.nit, flush=True)
    g = np.asarray(jax.grad(nll)(jnp.asarray(r.x)))
    th0 = th_store
    P = lik.readable(r.x, spec)
    out = dict(gh=gh, nll=float(r.fun), nll_at_start=float(nll(jnp.asarray(th0))), converged=bool(r.success), message=str(r.message),
               nit=int(r.nit), secs=round(time.time() - t), max_abs_grad=float(np.abs(g).max()),
               off_sr=P["off_sr"], def_sr=P["def_sr"], off_re=P["off_re"], def_re=P["def_re"])
    np.save(f"data/th_re_gh{gh}.npy", r.x)
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cur = d.get("re_quad", {}); cur[f"gh{gh}"] = out
    cur["stored_gh5_theta_nll_under_gh"] = cur.get("stored_gh5_theta_nll_under_gh", {}); cur["stored_gh5_theta_nll_under_gh"][str(gh)] = out["nll_at_start"]
    save("re_quad", cur)


def re_wins(gh=9, R=10):
    """wins lost under the random-effects model with the gh-point fit (EB posteriors recomputed with that fit, tagged
    data/re_gh<gh>_<side>.parquet) vs the stored gh = 5 fit, same seeds (headline.evaluate_all, R draws)."""
    import lik
    from final_ci import spec_of
    from headline import evaluate_all, summarise, spec_variant
    m = pd.read_parquet("data/called_2026_w.parquet")
    data = lik.prepare(m, players=True)
    spec = dict(spec_of("re"), gh=gh); th = np.load(f"data/th_re_gh{gh}.npy")
    for side in ("off", "def"):
        lik.re_posterior(data, th, side, spec).to_parquet(f"data/re_gh{gh}_{side}.parquet")
    out = {}
    for name, v in ((f"gh{gh}", dict(theta=f"data/th_re_gh{gh}.npy", spec=spec, re=True, re_tag=f"re_gh{gh}")),
                    ("gh5_stored", spec_variant("re"))):
        mm, _, o, _ = evaluate_all(v, R=R, seed=1)
        sm = summarise(o, int(mm.gidx.max() + 1))
        out[name] = dict(opt_score=sm["opt_score"]["wins"], observed=sm["observed"]["wins"],
                         lost=sm["opt_score"]["wins"] - sm["observed"]["wins"], share_0008=sm["stake_0.008"]["wins"] / sm["opt_score"]["wins"],
                         best_stake=sm["best_stake"])
        print(name, out[name], flush=True)
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cur = d.get("re_quad", {}); cur[f"wins_R{R}"] = out
    save("re_quad", cur)


def calib_table():
    """population-marginal predicted vs observed challenge counts (and overturn rates) of every model in
    results/perception_variants.json (robust_fit.py: lik.predict with the random effects integrated out)."""
    pv = json.load(open("results/perception_variants.json"))
    rows = {}
    for name, v in pv.items():
        c = v.get("calibration") or {}
        r = dict(nll=v.get("nll"), k=v.get("k"), bic=v.get("bic"))
        for side in ("off", "def"):
            cs = c.get(side, {})
            for f in ("ch_obs", "ch_pred", "over_obs", "over_pred", "chi2_challenge", "bins_challenge", "chi2_overturn"):
                r[f"{side}_{f}"] = cs.get(f)
            if cs.get("ch_pred") is not None:
                r[f"{side}_ch_pred_minus_obs"] = cs["ch_pred"] - cs["ch_obs"]
        r["total_ch_obs"] = (r["off_ch_obs"] or 0) + (r["def_ch_obs"] or 0)
        r["total_ch_pred"] = (r["off_ch_pred"] or 0) + (r["def_ch_pred"] or 0)
        rows[name] = r
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cur = d.get("re_quad", {}); cur["calibration_table"] = rows
    save("re_quad", cur)


# ----------------------------------------------------------------------------------------------- I: lapse by stake
def lapse_state():
    import lik, jax, jax.numpy as jnp
    from final_ci import spec_of
    m = pd.read_parquet("data/called_2026_w.parquet")
    data = lik.prepare(m)
    out = {}
    base = dict(spec_of("main_lapse"))
    th_b = np.load("data/th_main_lapse.npy")
    rb, nb = lik.fit(data, base, th0=th_b)
    n = sum(data[s]["n"] for s in data)
    out["main_lapse"] = dict(nll=float(rb.fun), k=len(rb.x), bic=float(2 * rb.fun + len(rb.x) * np.log(n)),
                             lapse={s: float(lik.readable(rb.x, base)[f"{s}_lapse"]) for s in ("off", "def")})
    spec = dict(base, lapse_low=True)
    # warm start: insert 0 after each side's llapse (lapse_low follows the other side terms in lik.unpack)
    th0 = lik.init(spec)
    # map: rebuild by unpacking the base and re-packing with the new term at 0
    th0 = repack_with(th_b, base, spec)
    r, nll = lik.fit(data, spec, th0=th0)
    P = lik.unpack(jnp.asarray(r.x), spec)
    H = np.asarray(jax.hessian(nll)(jnp.asarray(r.x))); V = np.linalg.inv(H)
    sig = lambda x: float(1 / (1 + np.exp(-x)))
    res = dict(nll=float(r.fun), k=len(r.x), bic=float(2 * r.fun + len(r.x) * np.log(n)), converged=bool(r.success),
               n=n, low_threshold_log_g=lik.LG_MID, low_share={s: float((data[s]["LG"] < 0).mean()) for s in data})
    k0 = len(rb.x)
    for s in ("off", "def"):
        q = P[s]; a, b = float(q["llapse"]), float(q["llapse_low"])
        ia, ib = idx_of(spec, s, "llapse"), idx_of(spec, s, "llapse_low")
        res[s] = dict(lapse_high_stake=sig(a), lapse_low_stake=sig(a + b), logit_shift_low=b, se_logit_shift=float(np.sqrt(V[ib, ib])),
                      z=b / float(np.sqrt(V[ib, ib])), sr=float(np.exp(q["lsr"])))
    res["delta_bic_vs_main_lapse"] = res["bic"] - out["main_lapse"]["bic"]
    res["lr_stat"] = 2 * (out["main_lapse"]["nll"] - res["nll"])
    out["main_lapse_low"] = res
    out["note"] = ("lambda_side = sigmoid(llapse + llapse_low * 1{log g <= -5}); log g = -5 is the first hinge of the main "
                   "criterion, at about the 40th percentile of the eligible pitches (lik.G_KNOTS). BIC with n = eligible rows.")
    np.save("data/th_main_lapse_low.npy", r.x)
    save("lapse_state", out)


def idx_of(spec, side, key):
    import lik
    th = np.arange(lik.n_truth(spec) + 2 * lik.n_side(spec), dtype=float)
    return int(lik.unpack(th, spec)[side][key])


def repack_with(th_b, base, spec):
    import lik
    th = lik.init(spec)
    nb = lik.n_truth(base); ns_b = lik.n_side(base); ns = lik.n_side(spec)
    th[:nb] = th_b[:nb]
    for j in range(2):
        th[nb + j * ns: nb + j * ns + ns_b] = th_b[nb + j * ns_b: nb + (j + 1) * ns_b]
        th[nb + j * ns + ns_b:nb + (j + 1) * ns] = 0.0
    return th


def lapse_grad():
    """max |gradient| of the stored lapse_low fit (the trust-region fit reported success = False)."""
    import lik, jax, jax.numpy as jnp
    from final_ci import spec_of
    spec = dict(spec_of("main_lapse"), lapse_low=True)
    nll = lik.make_nll(lik.prepare(pd.read_parquet("data/called_2026_w.parquet")), spec)
    g = np.asarray(jax.grad(nll)(jnp.asarray(np.load("data/th_main_lapse_low.npy"))))
    d = json.load(open(OUT)); cur = d["lapse_state"]; cur["main_lapse_low"]["max_abs_grad"] = float(np.abs(g).max())
    save("lapse_state", cur)


def report():
    d = json.load(open(OUT)); L = []; f = lambda x, k=3: f"{x:.{k}f}"
    A = d.get("decomp")
    if A:
        b = A["benchmarks_out_of_sample"]; L.append(f"A decomp (R={A['R']}: tune on draws 0-{A['R_tune']-1}, evaluate on the other {A['R_eval']}; "
              f"observed {f(A['observed']['wins'])} wins/team-season, {f(A['observed']['n'],2)} ch/team-game; out-of-sample opt_score "
              f"{f(b['opt_score']['wins'])} ({f(b['opt_score']['n'],2)} ch), one-line rule d=0.008 share {f(b['stake_0008']['share_opt'])}, "
              f"gap closed {f(b['stake_0008']['gap_closed'])}):")
        for n, r in A["rows"].items():
            o = r["out_of_sample"]; i = r["in_sample"]
            L.append(f"  {n:18s} params {json.dumps({k: round(v, 3) for k, v in r['params'].items()})}: OOS wins {f(o['wins'])}, share {f(o['share_opt'])}, "
                     f"gap closed {f(o['gap_closed'])}, {f(o['n'],2)} ch/team-game (in-sample {f(i['wins'])}, gap closed {f(i['gap_closed'])})")
    B = d.get("slopes")
    if B:
        L.append("B slopes (logit(1-p*) on log g; full controls / no count controls / none; high = log g > -5):")
        for sd, S in B["sides"].items():
            for rg in ("all", "high", "low"):
                x = S[rg]
                L.append(f"  {sd} {rg:4s} n={x['n']}: teams {f(x['teams_logit_full'])} / {f(x['teams_logit_no_count'])} / {f(x['teams_logit_none'])}; "
                         f"opt {f(x['opt_logit_full'])} / {f(x['opt_logit_no_count'])} / {f(x['opt_logit_none'])}; teams dkappa/dlog g "
                         f"{f(x['teams_dkappa_dlogg_mean'])} in (OLS full {f(x['teams_kappa_ols_full'])}); opt signal threshold OLS full {f(x['opt_signal_threshold_ols_full'])} in")
    E = d.get("pclip")
    if E:
        c = E["counts_per_draw"]
        L.append(f"E pclip (R={E['R']}, pi={E['pi']:.5f}): per draw p=1 on {c['p1']:.0f} pitches ({c['p1_wrong']:.1f} of them right calls), "
                 f"p=0 on {c['p0']:.0f}, p<pi on {c['p_lt_pi']:.0f} of {E['n_opportunities']}; lost raw {f(E['raw']['lost'],4)} vs clipped "
                 f"{f(E['clip']['lost'],4)} (change {E['change_lost']:.1e}); share d=0.008 {f(E['raw']['share_0008'],4)} vs {f(E['clip']['share_0008'],4)}; best d {E['raw']['best_stake']} both")
    F = d.get("months")
    if F:
        L.append("F months (R=40 point draws; lost per team-season equivalent, game-bootstrap SE):")
        L.append("  " + "; ".join(f"{k} {f(v['lost_per_team_season'],2)} ({f(v['boot_se'],2)}, {v['games']} g)" for k, v in F["months"].items()))
        L.append(f"  pre-ASB (to 13 Jul) {f(F['first_half_pre_asb']['lost_per_team_season'],2)} ({f(F['first_half_pre_asb']['boot_se'],2)}), "
                 f"post-ASB {f(F['second_half_post_asb']['lost_per_team_season'],2)} ({f(F['second_half_post_asb']['boot_se'],2)}); halves by median date "
                 f"{f(F['first_half_games']['lost_per_team_season'],2)} / {f(F['second_half_games']['lost_per_team_season'],2)}; trend {f(F['linear_trend_per_month'])}/month; "
                 f"observed share of opt by month " + ", ".join(f"{k} {f(v['observed_share_opt'],2)}" for k, v in F["months"].items()))
    G = d.get("moot")
    if G:
        a, b = G["moot_same_pitch"], G["third_out_after_the_pitch"]
        L.append(f"G moot: {a['n']} called pitches with the inning-ending baserunning out on the same pitch ({json.dumps(a['events'])}), "
                 f"{a['n_challenged']} challenged ({a['n_overturned']} overturned), stake {f(a['stake'],2)} = {100*a['share_of_total_stake']:.3f} % of total stake "
                 f"(challenged-stake share {100*a['share_of_challenged_stake']:.4f} %); a further {b['n']} last pitches before an inning-ending pickoff/CS on a later "
                 f"non-pitch event ({b['n_challenged']} challenged, not moot ex ante), stake share {100*b['share_of_total_stake']:.3f} %")
    D = d.get("wp_shrink")
    if D:
        L.append(f"D wp_shrink (R={D['R']}, perception fixed):")
        for n, r in D["engines"].items():
            nm = r["nonmonotone"]
            L.append(f"  {n:16s} lost {f(r['lost'])} (opt {f(r['opt_score'])}, obs {f(r['observed'])}), share d=0.008 {f(r['share_0008'])}, best {r['best_stake']} "
                     f"({f(r['best_share'])}); non-monotone rows {nm['rows']} ({nm['challenged']} challenged), distinct state cells {nm['distinct_state_cells']}, "
                     f"distinct (count,outs,bases) {nm['distinct_count_outs_bases']}")
    C = d.get("crossfit_stakes")
    if C:
        L.append(f"C crossfit (R={C['R']}, {len(C['splits'])} random game splits x 2 directions, engines from half the games): lost same-sample "
                 f"{f(C['lost_same_mean'])} vs cross-fit {f(C['lost_cross_mean'])} (full-season equivalents); bias {f(C['bias_mean'])} "
                 f"(SD over splits {f(C['bias_sd_over_splits'])}; halves {', '.join(f(x) for x in C['bias_all_halves'])})")
    H = d.get("re_quad")
    if H:
        for k in sorted(x for x in H if x.startswith("gh")):
            r = H[k]
            L.append(f"H {k}: nll {f(r['nll'],2)} (stored gh5 theta under this rule {f(r['nll_at_start'],2)}), max|grad| {r['max_abs_grad']:.2g}, sr off/def "
                     f"{f(r['off_sr'])}/{f(r['def_sr'])}, criterion SD off/def {f(r['off_re']['sd_kappa'])}/{f(r['def_re']['sd_kappa'])}, log sr SD "
                     f"{f(r['off_re']['sd_logsr'])}/{f(r['def_re']['sd_logsr'])}")
        for k in [x for x in H if x.startswith("wins_")]:
            L.append(f"H {k}: " + "; ".join(f"{n} lost {f(v['lost'])} share {f(v['share_0008'])}" for n, v in H[k].items()))
        T = H.get("calibration_table", {})
        if T:
            L.append("H calibration (population-marginal predicted vs observed challenges, off | def):")
            L.append("  " + "; ".join(f"{n} {r['off_ch_pred']:.0f}/{r['off_ch_obs']} | {r['def_ch_pred']:.0f}/{r['def_ch_obs']}" for n, r in T.items()))
    I = d.get("lapse_state")
    if I:
        r = I["main_lapse_low"]
        L.append(f"I lapse_state: lambda high/low stake off {f(r['off']['lapse_high_stake'])}/{f(r['off']['lapse_low_stake'])} (logit shift {f(r['off']['logit_shift_low'],2)}, "
                 f"z {f(r['off']['z'],2)}), def {f(r['def']['lapse_high_stake'])}/{f(r['def']['lapse_low_stake'])} (z {f(r['def']['z'],2)}); LR {f(r['lr_stat'],2)} on 2 df, "
                 f"dBIC vs main_lapse {f(r['delta_bic_vs_main_lapse'],1)} (main_lapse lambda {f(I['main_lapse']['lapse']['off'])}/{f(I['main_lapse']['lapse']['def'])})"
                 + (f"; max|grad| {r['max_abs_grad']:.2g}" if "max_abs_grad" in r else ""))
    txt = "\n".join(L) + "\n"
    os.makedirs("data/rerun130", exist_ok=True)
    open("data/rerun130/analyses.done", "w").write(txt); print(txt)


if __name__ == "__main__":
    cmd = sys.argv[1]; a = sys.argv[2:]
    if cmd == "decomp": decomp(int(a[0]) if a else 20, a[1].split(",") if len(a) > 1 else None)
    elif cmd == "slopes": slopes(int(a[0]) if a else 4)
    elif cmd == "pclip": pclip(int(a[0]) if a else 10)
    elif cmd == "months": months()
    elif cmd == "moot": moot()
    elif cmd == "wp_shrink": wp_shrink(int(a[0]) if a else 10)
    elif cmd == "crossfit": crossfit(int(a[0]) if a else 10, int(a[1]) if len(a) > 1 else 2)
    elif cmd == "re_quad": re_quad(int(a[0]) if a else 9)
    elif cmd == "calib_table": calib_table()
    elif cmd == "lapse_grad": lapse_grad()
    elif cmd == "report": report()
    elif cmd == "re_wins": re_wins(int(a[0]) if a else 9, int(a[1]) if len(a) > 1 else 10)
    elif cmd == "lapse_state": lapse_state()
    else: raise SystemExit(__doc__)
