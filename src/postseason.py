"""Postseason out-of-sample preview (2026 playoffs so far).

Everything is fitted on the 2026 regular season only (BIC-best model kappa_sr_region refitted on the
raw-coordinate geometry with deterministic truth, data/th_kappa_sr_region.npy; isotonic belief map learned on
an auxiliary draw; score-aware DP).  No playoff row enters any fit except the explicitly labelled
playoff-only refits in checks 2b and 3.
  fetch   : Stats API live feeds of the final postseason games (F wild card, D division series, L LCS, W World Series)
            with the series context, plus Savant Statcast for the same games -> data/ext_post2026.parquet,
            data/ext_games_post2026.parquet, data/statcast_post2026.parquet
  build   : called-pitch table in the layout of data/called_2026_w.parquet (Statcast location/state/catcher joined to
            the feed's challenge fields, original call, challenger role, challenges left with the extra-inning grant,
            ABS-zone geometry, WP stakes g from the regular-season engine) -> data/called_post2026_w.parquet
  analyze : (1) sample, (2) perception transfer, (3) stakes response, (4) learning, (5) wins/WP lost
            -> results/postseason_preview.{json,md}
  report  : results/postseason_preview.md from the json
usage: python src/postseason.py fetch | build | analyze [R] [B] | report"""
import io, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, "src")

TAG = "post2026"
ROUND = dict(F="Wild Card", D="Division Series", L="LCS", W="World Series")
BEST_OF = dict(F=3, D=5, L=7, W=7)
SC_URL = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfGT=F%7CD%7CL%7CW%7C&hfSea=2026%7C"
          "&player_type=pitcher&type=details&game_date_gt={a}&game_date_lt={b}")
MAIN = "kappa_sr_region_g"        # BIC-best homogeneous model (criterion piecewise-linear in log g)
ALLSTAR = "2026-07-14"            # All-Star Game: first half = regular-season games before it


# ----------------------------------------------------------------------------------------------------------- fetch
def schedule(start="2026-09-28", end="2026-11-15"):
    from fetch_ext import get, API
    d = get(f"{API}/v1/schedule?sportId=1&gameType=F,D,L,W&startDate={start}&endDate={end}")
    rows = []
    for dd in d["dates"]:
        for g in dd["games"]:
            if g["status"]["codedGameState"] != "F":
                continue
            h, a = g["teams"]["home"], g["teams"]["away"]
            rows.append(dict(game_pk=g["gamePk"], game_type=g["gameType"], date=dd["date"], home_id=h["team"]["id"],
                             away_id=a["team"]["id"], home_name=h["team"]["name"], away_name=a["team"]["name"],
                             home_runs=h.get("score"), away_runs=a.get("score"), home_win=bool(h.get("isWinner")),
                             series_game=g.get("seriesGameNumber"), games_in_series=g.get("gamesInSeries")))
    return pd.DataFrame(rows).drop_duplicates("game_pk").sort_values(["date", "game_pk"]).reset_index(drop=True)


def series_state(s):
    """wins of the home and away team in the series before each game (series = round x team pair)."""
    s = s.copy()
    s["pair"] = [tuple(sorted(x)) for x in zip(s.home_id, s.away_id)]
    hw, aw = {}, {}
    for _, d in s.groupby(["game_type", "pair"], sort=False):
        wins = {}
        for i, r in d.sort_values(["date", "game_pk"]).iterrows():
            hw[i] = wins.get(r.home_id, 0); aw[i] = wins.get(r.away_id, 0)
            win = r.home_id if r.home_win else r.away_id
            wins[win] = wins.get(win, 0) + 1
    s["home_series_w"] = pd.Series(hw); s["away_series_w"] = pd.Series(aw)
    return s.drop(columns="pair")


def statcast_post(start="2026-09-28", end="2026-11-15"):
    import datetime as dt
    UA = {"User-Agent": "Mozilla/5.0 (research script)"}
    d, e, parts = dt.date.fromisoformat(start), dt.date.fromisoformat(end), []
    while d <= e:
        b = min(d + dt.timedelta(days=4), e)
        for k in range(4):
            try:
                raw = urllib.request.urlopen(urllib.request.Request(SC_URL.format(a=d, b=b), headers=UA), timeout=300).read()
                if raw.strip():
                    parts.append(pd.read_csv(io.BytesIO(raw), low_memory=False))
                break
            except Exception as ex:
                print("retry", d, ex, flush=True); time.sleep(10 * (k + 1))
        d = b + dt.timedelta(days=1)
    return pd.concat(parts, ignore_index=True).drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])


def fetch():
    from fetch_ext import game_rows
    s = series_state(schedule())
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(game_rows, s.game_pk.tolist()))
    bad = [pk for pk, r, _ in res if r is None]
    df = pd.DataFrame([row for _, r, _ in res if r for row in r])
    gm = pd.DataFrame([g for _, _, g in res if g]).drop(columns=["date", "home_id", "away_id"]).merge(s, on="game_pk")
    df.to_parquet(f"data/ext_{TAG}.parquet"); gm.to_parquet(f"data/ext_games_{TAG}.parquet")
    sc = statcast_post(); sc.to_parquet(f"data/statcast_{TAG}.parquet")
    print(TAG, len(s), "final games", gm.game_type.value_counts().to_dict(), len(df), "feed pitches", int(df.ch.sum()),
          "challenges; Statcast pitches:", len(sc), "games:", sc.game_pk.nunique(), "failed fetches:", len(bad), flush=True)


# ----------------------------------------------------------------------------------------------------------- build
def build():
    from build_ext import challenges_left
    import geometry as G
    from stakes import add_stakes
    f = pd.read_parquet(f"data/ext_{TAG}.parquet"); g = pd.read_parquet(f"data/ext_games_{TAG}.parquet")
    sc = pd.read_parquet(f"data/statcast_{TAG}.parquet")
    print("games with ABS challenges enabled:", int(g.has.fillna(False).sum()), "of", len(g))
    f = f.merge(g[["game_pk", "date", "home_id", "away_id", "has", "game_type"]], on="game_pk")
    f = f[f.has.notna()]                                            # None = no ABS, False = ABS game without a challenge
    f["ch"] = f.ch.fillna(False).astype(bool); f["ch_over"] = f.ch_over.fillna(False).astype(bool)
    f = challenges_left(f)                                           # start 2, -1 per failure, extra-inning grant
    f["left_ch"] = np.where(f.ch_team == f.home_id, f.left_home, f.left_away)
    print("challenges made with 0 left (rule check):", int(((f.left_ch <= 0) & f.ch).sum()), "of", int(f.ch.sum()),
          "| extra-inning challenges:", int((f.ch & (f.inning >= 10)).sum()))
    # Statcast location, pre-pitch state and catcher as in build.py (the feed parser assumes the regular-season runner
    # on second in extras, which the postseason does not use)
    keep = ["plate_x", "plate_z", "sz_top", "sz_bot", "vx0", "vy0", "vz0", "ax", "ay", "az", "balls", "strikes",
            "outs_when_up", "on_1b", "on_2b", "on_3b", "home_score", "away_score", "fielder_2", "umpire", "description"]
    f = f.drop(columns=[c for c in keep + ["bases"] if c in f.columns])
    m = f.merge(sc[["game_pk", "at_bat_number", "pitch_number"] + keep], on=["game_pk", "at_bat_number", "pitch_number"],
                how="left", indicator=True)
    called = m.code.isin(["C", "B", "*B"])
    print("called pitches matched to Statcast:", round(float((m._merge[called] == "both").mean()), 4),
          "| challenged matched:", round(float((m._merge[called & m.ch] == "both").mean()), 4))
    m = m[called & (m._merge == "both") & m.plate_x.notna() & m.vy0.notna() & m.sz_top.notna()].drop(columns="_merge").copy()
    # Drop feed-Statcast mis-joins (the Statcast description contradicts the feed's call code; build.py)
    mj = ~((m.code.eq("C") & m.description.eq("called_strike")) | (m.code.isin(["B", "*B"]) & m.description.isin(["ball", "blocked_ball"])))
    print("mis-joined rows dropped:", int(mj.sum())); m = m[~mj].copy()
    m["final_strike"] = m.code.eq("C")
    m["orig_strike"] = np.where(m.ch_over, ~m.final_strike, m.final_strike)
    m["role"] = np.select([~m.ch, m.ch_player == m.batter, m.ch_player == m.pitcher, m.ch_player == m.fielder_2],
                          [None, "batter", "pitcher", "catcher"], "other")
    # raw Statcast 2026 coordinates are already in the ABS plane (the postseason Statcast plane test in
    # results/geometry_audit.json gives a mid-plane residual of 0.003 +- 0.035 in): no projection
    m["px_mid"], m["pz_mid"] = G.abs_location(m, "statcast2026")
    m["dist_in"], m["edge"] = G.signed_distance(m.px_mid, m.pz_mid, m.sz_top, m.sz_bot)
    print("ABS rule on raw coordinates reproduces", round(float((((m.dist_in > 0) == ~m.orig_strike) == m.ch_over)[m.ch].mean()), 4),
          "of postseason challenge rulings")
    m["bases"] = m.on_1b.notna().astype(int) + 2 * m.on_2b.notna().astype(int) + 4 * m.on_3b.notna().astype(int)
    m["score_diff_home"] = (m.home_score - m.away_score).astype(int)
    for c in ("inning", "outs_when_up"):
        m[c] = m[c].astype(int)
    m["balls"] = m.balls.clip(0, 3).astype(int); m["strikes"] = m.strikes.clip(0, 2).astype(int)
    # The postseason has no runner on second in extra innings; stakes come from the
    # engine solved without it (wp.py -> data/wp_engine_post.npz), same 2026 PA kernel
    import stakes
    stakes.use_engine("data/wp_engine_post.npz")
    m = add_stakes(m)
    stakes.use_engine("data/wp_engine.npz")
    m["side"] = np.where(m.orig_strike, "off", "def")
    m["left"] = np.where(m.orig_strike, np.where(m.top, m.left_away, m.left_home), np.where(m.top, m.left_home, m.left_away))
    m["w"] = np.where(m.orig_strike, -m.dist_in, m.dist_in)          # preliminary, replaced by sim.apply_lik
    m = m.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    print("called pitches:", len(m), "challenged:", int(m.ch.sum()), "roles:", m[m.ch].role.value_counts().to_dict(),
          "| batters challenge strikes only:", round(float(m[m.ch & (m.role == "batter")].orig_strike.mean()), 4))
    m.to_parquet(f"data/called_{TAG}_w.parquet")


# --------------------------------------------------------------------------------------------------------- analyze
def load_frames():
    """regular season (with official date) and postseason called-pitch tables; w for the |w| < 8 eligibility filter
    of lik.prepare as in the regular-season fit (deterministic truth: w = signed distance on the wrong side)."""
    rs = pd.read_parquet("data/called_2026_w.parquet")
    rs = rs.merge(pd.read_parquet("data/feed_games_2026.parquet")[["game_pk", "date"]], on="game_pk")
    po = pd.read_parquet(f"data/called_{TAG}_w.parquet")
    po = po.merge(pd.read_parquet(f"data/ext_games_{TAG}.parquet")[["game_pk", "home_series_w", "away_series_w"]], on="game_pk")
    po["w"] = np.where(po.orig_strike.values, -po.dist_in.values, po.dist_in.values)
    po["L"] = series_leverage(po)
    return rs, po


def series_win_prob(a, b, n):
    """P(a team with a wins beats one with b wins to n wins), each game a coin flip."""
    from math import comb
    if a >= n: return 1.0
    if b >= n: return 0.0
    need, other = n - a, n - b
    return sum(comb(need - 1 + k, k) * 0.5 ** (need + k) for k in range(other))


def series_leverage(po):
    """L = P(series | win this game) - P(series | lose it): d series-WP / d game-WP, the same for both teams."""
    n = po.game_type.map(BEST_OF).values // 2 + 1
    a, b = po.home_series_w.values, po.away_series_w.values
    return np.array([series_win_prob(x + 1, y, k) - series_win_prob(x, y + 1, k) for x, y, k in zip(a, b, n)])


def ctx():
    import lik
    from final_ci import spec_of
    spec = spec_of(MAIN); th = np.load(f"data/th_{MAIN}.npy")
    nt, ns = lik.n_truth(spec), lik.n_side(spec)
    idx = {s: dict(b0=nt + k * ns, blg=nt + k * ns + 1, lsr=nt + k * ns + 6) for k, s in enumerate(("off", "def"))}
    return spec, th, idx, np.load(f"data/cov_{MAIN}.npy")


def score(m, th, spec):
    """observed vs predicted challenges and overturn rates by side (oos.score with the main spec) and z statistics.
    With the deterministic truth the overturn probability of a pitch that WAS challenged is ~1{w > 0}, so
    over_pred / logscore_overturn / brier_overturn check geometry and the mislabel rate, not perception.  The
    perception tests are the challenge counts, the challenge log score per eligible pitch (skill against a constant
    rate equal to the model's own mean, i.e. discrimination) and over_rate_model = sum p(ch, wrong) / sum p(ch), the
    overturn rate the model expects among its own predicted challenges, without conditioning on which were made."""
    import lik
    from oos import frames
    data = lik.prepare(m); fr = frames(m); out = {}; ng = m.game_pk.nunique()
    for side in ("off", "def"):
        pch, p1 = lik.predict(data, th, spec, side); e = fr[side]; c = e.ch.values
        q = np.clip(p1[c] / pch[c], 1e-6, 1 - 1e-6); y = e.ch_over.values[c]
        out[side] = dict(games=ng, eligible=int(len(e)), ch_obs=int(c.sum()), ch_pred=round(float(pch.sum()), 1),
                         ch_z=round(float((c.sum() - pch.sum()) / np.sqrt((pch * (1 - pch)).sum())), 2),
                         ch_per_team_game_obs=round(float(c.sum() / (2 * ng)), 3), ch_per_team_game_pred=round(float(pch.sum() / (2 * ng)), 3),
                         over_n=int(y.sum()), over_obs=round(float(y.mean()), 4), over_pred=round(float(q.mean()), 4),
                         over_z=round(float((y.sum() - q.sum()) / np.sqrt((q * (1 - q)).sum())), 2),
                         logscore_overturn=round(float(np.mean(np.where(y, np.log(q), np.log(1 - q)))), 4),
                         brier_overturn=round(float(np.mean((q - y) ** 2)), 4),
                         over_rate_model=round(float(p1.sum() / pch.sum()), 4),
                         over_rate_model_z=round(float((y.sum() - c.sum() * p1.sum() / pch.sum())
                                                       / np.sqrt(c.sum() * (p1.sum() / pch.sum()) * (1 - p1.sum() / pch.sum()))), 2))
        pc = np.clip(pch, 1e-9, 1 - 1e-9); r0 = pc.mean()
        ls = np.mean(np.where(c, np.log(pc), np.log(1 - pc))); l0 = np.mean(np.where(c, np.log(r0), np.log(1 - r0)))
        out[side].update(logscore_challenge=round(float(ls), 5), logscore_challenge_null=round(float(l0), 5),
                         challenge_skill=round(float(1 - ls / l0), 4))
    return out


def refit(m, free, spec, th, ycol=None):
    """playoff-only refit of the parameters at positions `free`, all others held at their regular-season values.
    Returns estimate, Wald covariance and the negative log-likelihood."""
    import jax, jax.numpy as jnp, lik
    from scipy.optimize import minimize
    nll = lik.make_nll(lik.prepare(m), spec); free = np.asarray(free); base = jnp.asarray(th)
    f = lambda x: nll(base.at[free].set(x))
    vg = jax.jit(jax.value_and_grad(f)); H = jax.jit(jax.hessian(f))
    r = minimize(lambda x: tuple(np.asarray(v, float) for v in vg(jnp.asarray(x))), th[free], jac=True, method="BFGS")
    return r.x, np.linalg.inv(np.asarray(H(jnp.asarray(r.x)))), float(r.fun), bool(r.success)


def ci_exp(x, v):
    return [round(float(np.exp(x - 1.96 * np.sqrt(v))), 3), round(float(np.exp(x + 1.96 * np.sqrt(v))), 3)]


def ci_lin(x, v):
    return [round(float(x - 1.96 * np.sqrt(v)), 3), round(float(x + 1.96 * np.sqrt(v)), 3)]


def cal_maps(m, rng, n=5):
    """the regular-season calibration P(W | signal bin) by (side, posterior group), averaged over n auxiliary draws
    (sim.draw, the belief is learned from a draw independent of the one that scores the policies)."""
    import sim
    maps = {}
    for _ in range(n):
        W, s = sim.truth_signal(m, rng)
        for side in ("off", "def"):
            grp = m.cal.values if "cal" in m else m.pgrp.values    # Sim.cal_groups cells
            for e in np.unique(grp):
                k = (m.side == side).values & (grp == e)
                if not k.any():
                    continue
                b = np.clip(np.floor((s[k] + 15) / 0.1).astype(int), 0, 300)
                mp = sim.monotone_calibration(np.bincount(b, weights=W[k], minlength=301), np.bincount(b, minlength=301))
                ok = np.flatnonzero(~np.isnan(mp))           # bins outside the draw's range: nearest fitted value
                mp = np.interp(np.arange(301), ok, mp[ok])
                maps[(side, e)] = maps.get((side, e), 0) + mp / n
    return maps


def apply_maps(maps, side, grp, s):
    b = np.clip(np.floor((np.asarray(s) + 15) / 0.1).astype(int), 0, 300)
    out = np.zeros(len(b))
    for e in np.unique(grp):
        k = grp == e
        out[k] = maps[(side, e)][b[k]]
    return out


def implied_slope(e, kap, maps, side, gcol="g"):
    """slope of logit(1 - p*) on log g with the state controls of the threshold, p* = calibrated belief at kappa;
    a stake-rational rule p >= D/(g + D) has slope 1 (variant_wins.calibration_test, with the isotonic map)."""
    p = np.clip(apply_maps(maps, side, (e.cal if "cal" in e else e.pgrp).values, kap), 1e-3, 1 - 1e-3)
    X = np.column_stack([np.ones(len(e)), np.log(np.clip(e[gcol].values, 1e-4, None)), (e.left == 1).astype(float),
                         np.clip(e.inning.values, 1, 10) / 9, (e.strikes == 2).astype(float), (e.balls == 3).astype(float)])
    return float(np.linalg.lstsq(X, np.log((1 - p) / p), rcond=None)[0][1]), float(p.mean())


def kappa_rows(e, th, spec, side):
    import lik
    q = lik.unpack(th, spec)[side]
    k = lik.design(e) @ np.asarray(q["beta"]) + np.asarray(q["kreg"])[lik.region(e)]
    if "kg" in q: k = k + lik.gfeat(e) @ np.asarray(q["kg"])          # Main criterion is piecewise in log g
    if "kgl" in q: k = k + lik.glow(e) * float(q["kgl"])
    return k


def eligible(ms, side):
    return ms[(ms.side == side).values & (ms.left > 0).values].copy()


def stakes_check(rs, po, spec, th, idx, cov, maps_rs, mrs, mpo, rng):
    """(3) kappa = X beta: refit the intercept and the log g coefficient on playoff challenges (sigma_r, truth and
    the other threshold terms at their regular-season values); implied threshold slope on log g, regular season vs
    playoffs, with parametric CIs; series-leverage response."""
    out = {}
    free = [idx[s][k] for s in ("off", "def") for k in ("b0", "blg")]
    x, V, nll_wp, ok = refit(po, free, spec, th)
    thp = th.copy(); thp[free] = x
    draws = rng.multivariate_normal(x, V, 400)
    for j, side in enumerate(("off", "def")):
        i_b, i_l = 2 * j, 2 * j + 1
        r = dict(beta_logg_regular=round(float(th[idx[side]["blg"]]), 3),
                 beta_logg_regular_ci95=ci_lin(th[idx[side]["blg"]], cov[idx[side]["blg"], idx[side]["blg"]]),
                 beta_logg_playoff=round(float(x[i_l]), 3), beta_logg_playoff_ci95=ci_lin(x[i_l], V[i_l, i_l]),
                 intercept_shift_playoff=round(float(x[i_b] - th[idx[side]["b0"]]), 3), refit_converged=ok)
        er, ep = eligible(mrs, side), eligible(mpo, side)
        r["implied_slope_regular"], r["mean_pstar_regular"] = implied_slope(er, er.kappa.values, maps_rs, side)
        r["implied_slope_playoff_regular_params"], _ = implied_slope(ep, ep.kappa.values, maps_rs, side)
        r["implied_slope_playoff_refit"], r["mean_pstar_playoff_refit"] = implied_slope(ep, kappa_rows(ep, thp, spec, side), maps_rs, side)
        sl = []
        for d in draws:
            t2 = th.copy(); t2[free] = d
            sl.append(implied_slope(ep, kappa_rows(ep, t2, spec, side), maps_rs, side)[0])
        r["implied_slope_playoff_refit_ci95"] = [round(float(v), 3) for v in np.quantile(sl, [0.025, 0.975])]
        out[side] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
    # series-WP version: stake g * L (L = series leverage of the game).  Within a game L is a constant, so the
    # game-WP optimum is also the series-WP optimum and a stake-rational threshold does not depend on L; a response
    # to L is a cross-game (attention/urgency) effect.  (a) refit on log(g L) instead of log g, (b) observed vs
    # regular-season-predicted challenges by leverage group.
    pos = po.copy(); pos["g"] = po.g * po.L
    xs, Vs, nll_s, _ = refit(pos, free, spec, th)
    out["series"] = dict(leverage_by_round={ROUND[k]: [round(float(v), 3) for v in np.quantile(po[po.game_type == k].drop_duplicates("game_pk").L, [0, 0.5, 1])]
                                            for k in sorted(po.game_type.unique())},
                         nll_game_wp=round(nll_wp, 2), nll_series_wp=round(nll_s, 2),
                         beta_logg_series={s: [round(float(xs[2 * j + 1]), 3), ci_lin(xs[2 * j + 1], Vs[2 * j + 1, 2 * j + 1])] for j, s in enumerate(("off", "def"))})
    hi = po.L >= 0.5                      # deciding games and games one win from clinching / elimination
    out["series"]["by_leverage"] = {nm: dict(games=int(po[k].game_pk.nunique()), **{s: {kk: v[kk] for kk in ("ch_obs", "ch_pred", "ch_z", "over_obs", "over_pred")}
                                                                                   for s, v in score(po[k], th, spec).items()})
                                    for nm, k in (("L>=0.5", hi), ("L<0.5", ~hi))}
    return out


def transfer_check(rs, po, spec, th, idx, cov):
    """(2) regular-season parameters held fixed -> playoff challenges and overturns; (2b) sigma_r refit on playoffs."""
    import lik
    out = dict(transfer=score(po, th, spec))
    # the most recent behaviour: the full model refitted on second-half regular-season games only
    h2 = rs[rs.date >= ALLSTAR].drop(columns="date")
    r, _ = lik.fit(lik.prepare(h2), spec, th0=th, maxiter=60)
    out["transfer_second_half_fit"] = dict(converged=bool(r.success), **score(po, r.x, spec))
    out["second_half_fit_sr"] = {s: round(float(np.exp(r.x[idx[s]["lsr"]])), 3) for s in ("off", "def")}
    for name, keys in (("sr_only", ("lsr",)), ("sr_and_intercept", ("lsr", "b0"))):
        free = [idx[s][k] for s in ("off", "def") for k in keys]
        x, V, nll, ok = refit(po, free, spec, th)
        r = dict(converged=ok, nll=round(nll, 2))
        for j, side in enumerate(("off", "def")):
            i = free.index(idx[side]["lsr"]); ir = idx[side]["lsr"]
            r[side] = dict(sr_regular=round(float(np.exp(th[ir])), 3), sr_regular_ci95=ci_exp(th[ir], cov[ir, ir]),
                           sr_playoff=round(float(np.exp(x[i])), 3), sr_playoff_ci95=ci_exp(x[i], V[i, i]))
            if "b0" in keys:
                i0 = free.index(idx[side]["b0"]); r[side]["intercept_shift"] = round(float(x[i0] - th[idx[side]["b0"]]), 3)
                r[side]["intercept_shift_ci95"] = ci_lin(x[i0] - th[idx[side]["b0"]], V[i0, i0])
        out[f"refit_{name}"] = r
    return out


def sample_table(rs, po):
    """(1) games, called pitches, challenges by role and round (regular season for reference)."""
    out = {}
    for nm, d in [(ROUND[k], po[po.game_type == k]) for k in sorted(po.game_type.unique(), key="FDLW".index)] + [("Postseason", po), ("Regular season", rs)]:
        c = d[d.ch]; ng = d.game_pk.nunique()
        out[nm] = dict(games=int(ng), called=int(len(d)), challenges=int(len(c)), per_team_game=round(len(c) / (2 * ng), 3),
                       overturn=round(float(c.ch_over.mean()), 3),
                       **{f"ch_{r}": int((c.role == r).sum()) for r in ("batter", "catcher", "pitcher", "other")},
                       **{f"over_{r}": (round(float(c[c.role == r].ch_over.mean()), 3) if (c.role == r).any() else None) for r in ("batter", "catcher", "pitcher")})
    return out


def learning_check(rs, po, spec, th):
    """(4) challenge and overturn rates by role: first half, second half, postseason; observed and as predicted by the
    full-season model (obs/pred isolates behaviour from the opportunity mix)."""
    out = {}
    for nm, d in (("first_half", rs[rs.date < ALLSTAR]), ("second_half", rs[rs.date >= ALLSTAR]), ("postseason", po)):
        c = d[d.ch]; ng = d.game_pk.nunique(); sc = score(d, th, spec)
        out[nm] = dict(games=int(ng), **{f"{r}_per_team_game": round(float((c.role == r).sum() / (2 * ng)), 3) for r in ("batter", "catcher", "pitcher")},
                       **{f"{r}_overturn": round(float(c[c.role == r].ch_over.mean()), 3) for r in ("batter", "catcher")},
                       model={s: {k: sc[s][k] for k in ("eligible", "ch_obs", "ch_pred", "ch_z", "over_obs", "over_pred", "over_z")} for s in sc})
        out[nm]["model"]["off"]["ch_per_100_eligible"] = round(100 * sc["off"]["ch_obs"] / sc["off"]["eligible"], 2)
        out[nm]["model"]["def"]["ch_per_100_eligible"] = round(100 * sc["def"]["ch_obs"] / sc["def"]["eligible"], 2)
    return out


def wins_check(rs, po, spec, R, B, seed=1):
    """(5) observed vs optimal policy on playoff games with the regular-season calibration and score-aware DP
    (path-sum accounting as headline.evaluate_all), per team-game in game WP; game bootstrap at fixed parameters."""
    import sim, dp_score, policies as P
    V = dict(theta=f"data/th_{MAIN}.npy", spec=spec)
    prep = []
    for fr in (rs.drop(columns="date"), po):
        m, _ = sim.load(dict(V, frame=fr))
        m = dp_score.add_dstart(m); m = sim.opportunities(m)
        m["new_extra"] = sim.extra_starts(m)
        m["gidx"] = m.game_pk.astype("category").cat.codes
        prep.append(m)
    mrs, mpo = prep
    cuts = sim.stake_cuts(mrs)                       # Calibration cells with regular-season stake terciles
    mrs["cal"] = sim.cal_groups(mrs, cuts); mpo["cal"] = sim.cal_groups(mpo, cuts)
    rng = np.random.default_rng(seed)
    z3, zc, zcard = np.zeros((19, 80, 3)), np.zeros((19, 3)), np.zeros((4, 3, 2, 2))
    pols = {"opt_score": (7, 0.0), "stake_0.008": (2, 0.008), "obs_rule": (4, 0.0)}
    acc = {f: {} for f in ("rs", "po")}
    for r in range(R):
        # auxiliary regular-season draw -> the belief map (sim.draw); the same map is applied to playoff signals, and
        # the regular-season DP is solved on regular-season posteriors
        maps = cal_maps(mrs, rng, n=1)
        sims = {}
        for f, m in (("rs", mrs), ("po", mpo)):
            W, s = sim.truth_signal(m, rng)
            pt = np.zeros(len(m))
            for side in ("off", "def"):
                k = (m.side == side).values
                pt[k] = apply_maps(maps, side, m.cal.values[k], s[k])
            sims[f] = (W, s, pt)
        D4h, _ = dp_score.solve_dp_score(mrs, sims["rs"][2], True); D4a, _ = dp_score.solve_dp_score(mrs, sims["rs"][2], False)
        for f, m in (("rs", mrs), ("po", mpo)):
            W, s, pt = sims[f]; ng = int(m.gidx.max() + 1)
            kin = sim.kin_index(m).astype(np.int64)
            base = (m.gidx.values, m.is_home_team.values, m.new_extra.values, m.slot.values.astype(np.int64), kin,
                    m.inning.values.astype(np.int64), m.balls.values.astype(np.int64), m.strikes.values.astype(np.int64),
                    m.dbin.values.astype(np.int64))
            for name, (pc, dc) in pols.items():
                v, n, _, _, _ = P.run_path(*base, W, pt, s, m.g.values, m.RV.values, sim.kappa_by_count(m), (m.side == "off").values,
                                           pc, z3, z3, zc, zc, zcard, 0.0, 0.0, dc, D4h, D4a, ng)
                a = acc[f].setdefault(name, [np.zeros_like(v), np.zeros_like(n)])
                a[0] += v / R; a[1] += n / R
    out = {}
    for f, frame, m in (("rs", rs.drop(columns="date"), mrs), ("po", po, mpo)):
        mm, _ = sim.load(dict(V, frame=frame)); ng = int(m.gidx.max() + 1)
        gmap = dict(zip(np.sort(m.game_pk.unique()), range(ng)))
        ch = mm[mm.ch & mm.game_pk.isin(gmap)]
        obs, obsn = np.zeros((ng, 2, 2)), np.zeros((ng, 2, 2))
        np.add.at(obs, (ch.game_pk.map(gmap).values, ch.is_home_team.values.astype(int), (ch.side == "off").values.astype(int)), (ch.g * ch.ch_over).values)
        np.add.at(obsn, (ch.game_pk.map(gmap).values, ch.is_home_team.values.astype(int), (ch.side == "off").values.astype(int)), 1)
        acc[f]["observed"] = [obs, obsn]
        per_game = {k: v[0].sum(1) for k, v in acc[f].items()}           # [game, side] summed over the two teams
        res = {}
        for k, (v, n) in acc[f].items():
            res[k] = dict(wp_per_team_game=round(float(v.sum() / (2 * ng)), 5), batter=round(float(v[:, :, 1].sum() / (2 * ng)), 5),
                          fielder=round(float(v[:, :, 0].sum() / (2 * ng)), 5), challenges_per_team_game=round(float(n.sum() / (2 * ng)), 3))
        lost = per_game["opt_score"] - per_game["observed"]
        res["opt_minus_observed"] = dict(wp_per_team_game=round(float(lost.sum() / (2 * ng)), 5),
                                         batter=round(float(lost[:, 1].sum() / (2 * ng)), 5), fielder=round(float(lost[:, 0].sum() / (2 * ng)), 5),
                                         share_of_opt_captured=round(float(per_game["observed"].sum() / per_game["opt_score"].sum()), 3))
        if f == "po":
            bt = np.random.default_rng(7)
            L = mpo.groupby("gidx").L.first().values
            bs = {k: [] for k in ("lost", "lost_batter", "lost_fielder", "share", "series_lost")}
            for _ in range(B):
                j = bt.integers(0, ng, ng)
                bs["lost"].append(lost[j].sum() / (2 * ng)); bs["lost_batter"].append(lost[j, 1].sum() / (2 * ng))
                bs["lost_fielder"].append(lost[j, 0].sum() / (2 * ng))
                bs["share"].append(per_game["observed"][j].sum() / per_game["opt_score"][j].sum())
                bs["series_lost"].append((lost[j].sum(1) * L[j]).sum() / (2 * ng))
            q = lambda a: [round(float(x), 5) for x in np.quantile(a, [0.025, 0.975])]
            res["opt_minus_observed"].update(ci95=q(bs["lost"]), batter_ci95=q(bs["lost_batter"]), fielder_ci95=q(bs["lost_fielder"]),
                                             share_ci95=q(bs["share"]), series_wp_per_team_game=round(float((lost.sum(1) * L).sum() / (2 * ng)), 5),
                                             series_wp_ci95=q(bs["series_lost"]), bootstrap_B=B)
        res["games"] = ng
        out[f] = res
    out["R"] = R
    return out, mrs, mpo


def main(R=20, B=2000):
    import sim
    spec, th, idx, cov = ctx()
    rs, po = load_frames()
    res = dict(note="PREVIEW: 2026 postseason through 17 Oct (World Series not started); all model parameters fitted on the "
                    "2026 regular season only; intervals are wide by construction.", model=MAIN)
    res["sample"] = sample_table(rs, po); print(json.dumps(res["sample"]), flush=True)
    res["wins"], mrs, mpo = wins_check(rs, po, spec, R, B); print(json.dumps(res["wins"]), flush=True)
    res["perception_transfer"] = transfer_check(rs, po, spec, th, idx, cov); print(json.dumps(res["perception_transfer"]), flush=True)
    res["perception_transfer"]["regular_season_insample"] = score(rs, th, spec)
    maps = cal_maps(mrs, np.random.default_rng(11), n=5)
    res["stakes_response"] = stakes_check(rs, po, spec, th, idx, cov, maps, mrs, mpo, np.random.default_rng(5)); print(json.dumps(res["stakes_response"]), flush=True)
    res["learning"] = learning_check(rs, po, spec, th); print(json.dumps(res["learning"]), flush=True)
    json.dump(res, open("results/postseason_preview.json", "w"), indent=1)
    report(res)


def report(res):
    """results/postseason_preview.md from the json."""
    S, T, K, Lr, Wn = (res[k] for k in ("sample", "perception_transfer", "stakes_response", "learning", "wins"))
    SIDE = dict(off="Batter (called strikes)", **{"def": "Catcher/pitcher (called balls)"})
    f5 = lambda x: f"{x:.4f}"
    ci = lambda v, f=f5: f"[{f(v[0])}, {f(v[1])}]"
    L = ["# Postseason out-of-sample preview", "",
         "**Preview, small n.** 2026 postseason games final through 17 Oct 2026 (Wild Card, Division Series, LCS so far; "
         "the World Series has not started). Every model parameter, the belief calibration and the score-aware DP are "
         "from the 2026 regular season only (main spec `kappa_sr_region` on the corrected geometry); "
         "playoff data enter only the explicitly labelled playoff-only refits. Generated by `python src/postseason.py analyze`.", "",
         "## 1. Sample", "",
         "| | games | called pitches | challenges | per team-game | batter | catcher | pitcher | overturn | batter ovt. | catcher ovt. |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in S.items():
        L.append(f"| {k} | {v['games']} | {v['called']:,} | {v['challenges']:,} | {v['per_team_game']:.2f} | {v['ch_batter']} | "
                 f"{v['ch_catcher']} | {v['ch_pitcher']} | {v['overturn']:.3f} | {v['over_batter']} | {v['over_catcher']} |")
    L += ["", "The raw-coordinate ABS rule reproduces every postseason challenge ruling; no challenge was made with 0 left.", "",
          "## 2. Perception transfer (regular-season parameters held fixed)", "",
          "| side | eligible | challenges obs | pred | z | model overturn rate (own predictions) | obs overturn | z | challenge log-score skill |",
          "|---|---|---|---|---|---|---|---|---|"]
    for lab, blk in (("full season fit", T["transfer"]), ("second-half fit", T["transfer_second_half_fit"])):
        for s in ("off", "def"):
            v = blk[s]
            L.append(f"| {SIDE[s]}, {lab} | {v['eligible']:,} | {v['ch_obs']} | {v['ch_pred']} | {v['ch_z']:+.2f} | {v['over_rate_model']:.3f} | "
                     f"{v['over_obs']:.3f} | {v['over_rate_model_z']:+.2f} | {v['challenge_skill']:.3f} |")
    ins = T["regular_season_insample"]
    L += ["", f"Regular-season in-sample challenge log-score skill: batter {ins['off']['challenge_skill']:.3f}, catcher {ins['def']['challenge_skill']:.3f}. "
          f"Overturn given an actual challenge (geometry check, the truth is deterministic): log score "
          f"{T['transfer']['off']['logscore_overturn']:.4f} / {T['transfer']['def']['logscore_overturn']:.4f}, Brier "
          f"{T['transfer']['off']['brier_overturn']:.4f} / {T['transfer']['def']['brier_overturn']:.4f} (batter / catcher).", "",
          "**2b. σ_r refitted on the playoffs alone** (everything else at regular-season values; Wald 95 % CIs):", "",
          "| refit | side | σ_r regular season | σ_r playoffs | κ intercept shift |", "|---|---|---|---|---|"]
    for nm, lab in (("refit_sr_only", "σ_r only"), ("refit_sr_and_intercept", "σ_r + κ intercept")):
        for s in ("off", "def"):
            v = T[nm][s]
            sh = f"{v['intercept_shift']:+.2f} {ci(v['intercept_shift_ci95'], lambda x: f'{x:+.2f}')}" if "intercept_shift" in v else "—"
            L.append(f"| {lab} | {SIDE[s]} | {v['sr_regular']:.2f} {ci(v['sr_regular_ci95'], lambda x: f'{x:.2f}')} | "
                     f"{v['sr_playoff']:.2f} {ci(v['sr_playoff_ci95'], lambda x: f'{x:.2f}')} | {sh} |")
    L += ["", "## 3. Stakes response", "",
          "κ intercept and log g coefficient refitted on playoff challenges (σ_r and the other threshold terms fixed). "
          "Implied threshold slope = slope of logit(1 − p*) on log g with state controls, p* = calibrated belief at κ; "
          "a stake-rational rule has slope 1.", "",
          "| side | κ log-g coef. regular | playoffs | implied slope regular | playoffs (regular params) | playoffs (refit) |",
          "|---|---|---|---|---|---|"]
    f3 = lambda x: f"{x:+.3f}"
    for s in ("off", "def"):
        v = K[s]
        L.append(f"| {SIDE[s]} | {v['beta_logg_regular']:+.3f} {ci(v['beta_logg_regular_ci95'], f3)} | {v['beta_logg_playoff']:+.3f} "
                 f"{ci(v['beta_logg_playoff_ci95'], f3)} | {v['implied_slope_regular']:.3f} | {v['implied_slope_playoff_regular_params']:.3f} | "
                 f"{v['implied_slope_playoff_refit']:.3f} {ci(v['implied_slope_playoff_refit_ci95'], lambda x: f'{x:.2f}')} |")
    se = K["series"]
    L += ["", "**Playoff stakes.** Challenges do not carry over between games, so series win probability = "
          "P(series | lose) + L · P(win game), with L = P(series | win) − P(series | lose) fixed within a game (coin-flip "
          "series model). Maximising series WP within a game is therefore exactly maximising game WP: the regular-season "
          "game-WP DP is the right optimum in the playoffs, and the regular-season WP stakes understate playoff stakes only "
          "by a per-game factor L that rescales g and the token value Δ together. A stake-rational threshold therefore does "
          "not move with L; any response to L is a cross-game urgency effect.", "",
          f"- Series leverage L by round (min / median / max): " + "; ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" for k, v in se["leverage_by_round"].items()) + ".",
          f"- Refit on log(g·L) instead of log g: −log-lik {se['nll_series_wp']:.2f} vs {se['nll_game_wp']:.2f} (same parameter count); "
          f"coefficient batter {se['beta_logg_series']['off'][0]:+.3f}, catcher {se['beta_logg_series']['def'][0]:+.3f}.",
          "- Observed vs regular-season-predicted challenges by leverage:", ""]
    L += ["| games | n | batter obs / pred (z) | catcher obs / pred (z) |", "|---|---|---|---|"]
    for k, v in se["by_leverage"].items():
        L.append(f"| {k} | {v['games']} | {v['off']['ch_obs']} / {v['off']['ch_pred']} ({v['off']['ch_z']:+.2f}) | "
                 f"{v['def']['ch_obs']} / {v['def']['ch_pred']} ({v['def']['ch_z']:+.2f}) |")
    L += ["", "## 4. Learning", "",
          "| period | games | batter / team-game | catcher / team-game | batter per 100 eligible | catcher per 100 eligible | batter obs / pred (z) | catcher obs / pred (z) | batter ovt. | catcher ovt. |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in Lr.items():
        mo, md = v["model"]["off"], v["model"]["def"]
        L.append(f"| {k.replace('_', ' ')} | {v['games']} | {v['batter_per_team_game']:.2f} | {v['catcher_per_team_game']:.2f} | "
                 f"{mo['ch_per_100_eligible']:.2f} | {md['ch_per_100_eligible']:.2f} | {mo['ch_obs']} / {mo['ch_pred']} ({mo['ch_z']:+.1f}) | "
                 f"{md['ch_obs']} / {md['ch_pred']} ({md['ch_z']:+.1f}) | {v['batter_overturn']:.3f} | {v['catcher_overturn']:.3f} |")
    L += ["", "Predicted = the full-season regular-season model applied to that period's opportunities, so obs/pred isolates "
          "behaviour from the opportunity mix.", "",
          f"## 5. Wins / WP lost (descriptive preview, R = {Wn['R']} draws)", "",
          "Path-sum evaluation with the regular-season belief map and score-aware DP, in game WP per team-game.", "",
          "| policy | regular season | postseason |", "|---|---|---|"]
    for k, lab in (("opt_score", "optimum (score-aware DP)"), ("stake_0.008", "one-line rule d = 0.008"), ("obs_rule", "fitted behavioural rule (simulated)"), ("observed", "observed")):
        L.append(f"| {lab} | {Wn['rs'][k]['wp_per_team_game']:.4f} ({Wn['rs'][k]['challenges_per_team_game']:.2f} ch.) | "
                 f"{Wn['po'][k]['wp_per_team_game']:.4f} ({Wn['po'][k]['challenges_per_team_game']:.2f} ch.) |")
    lr, lp = Wn["rs"]["opt_minus_observed"], Wn["po"]["opt_minus_observed"]
    L += [f"| **optimum − observed** | {lr['wp_per_team_game']:.4f} (×162 = {162 * lr['wp_per_team_game']:.2f} wins/season) | "
          f"**{lp['wp_per_team_game']:.4f}** {ci(lp['ci95'])} |",
          f"| batter / fielder | {lr['batter']:.4f} / {lr['fielder']:.4f} | {lp['batter']:.4f} {ci(lp['batter_ci95'])} / {lp['fielder']:.4f} {ci(lp['fielder_ci95'])} |",
          f"| share of optimum captured | {lr['share_of_opt_captured']:.3f} | {lp['share_of_opt_captured']:.3f} {ci(lp['share_ci95'], lambda x: f'{x:.2f}')} |",
          f"| in series WP (Σ L · game WP lost) | — | {lp['series_wp_per_team_game']:.4f} {ci(lp['series_wp_ci95'])} |",
          "", f"Postseason CIs: game bootstrap (B = {lp['bootstrap_B']}) at fixed regular-season parameters; they omit "
          "parameter uncertainty, which is small next to the 33-game sampling noise."]
    open("results/postseason_preview.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fetch":
        fetch()
    elif mode == "build":
        build()
    elif mode == "analyze":
        main(int(sys.argv[2]) if len(sys.argv) > 2 else 20, int(sys.argv[3]) if len(sys.argv) > 3 else 2000)
    elif mode == "report":
        report(json.load(open("results/postseason_preview.json")))
