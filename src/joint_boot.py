"""HONZA-122: two bootstraps the referees asked for, both refitting the main perception model in every replicate.

joint (Opus, Sol, Astra, Fable): one game resample drives EVERY fitted component -- the PA kernel and WP / RE engine
       (wp.build_T, wp.solve), the stakes g and RV of every called pitch (stakes.add_stakes), the half-inning and
       extra-inning run laws of the DP (dp_score), the perception fit (lik.fit) and the policy evaluation
       (headline.evaluate_all). The headline bootstrap (headline.py boot) holds the engine fixed; wp_boot.py
       resamples the engine at fixed perception parameters; this is the joint version.
block (Sonnet): resample blocks of games (home team x calendar month, about 200 blocks) instead of games, to allow
       for dependence within a team's homestand / month (same catchers, pitchers, umpire crews, criterion drift);
       engine fixed as in the headline bootstrap.
usage: python src/joint_boot.py <joint|block> <B> [R] [workers]  -> data/<mode>_boot_rows.json (resumable),
       results/<mode>_boot.json"""
import json, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, "src")

KEEP = ["bench", "opt_score", "stake_0.008", "runs_dp", "card_pub", "savant", "obs_rule", "observed", "clairvoyant"]


def resample_frame(by, pick):
    return pd.concat([by[g].assign(game_pk=i, orig_game=g) for i, g in enumerate(pick)], ignore_index=True)


def run_laws(scb):
    """the score-aware DP's half-inning and extra-inning run laws from a (resampled) Statcast frame."""
    import dp_score
    P = dp_score.half_runs_from(scb); dp_score.half_runs = lambda: P
    x = scb[(scb.inning >= 10) & (scb.inning_topbot == "Top")].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    h = x.groupby(["game_pk", "inning"]).agg(a=("bat_score", "first"), z=("post_bat_score", "last"))
    rx = (h.z - h.a).clip(0, dp_score.RMAX).astype(int).values
    Px = np.bincount(rx, minlength=dp_score.RMAX + 1) / max(len(rx), 1); dp_score.extra_runs = lambda: Px


def job(args):
    """one replicate. HONZA-132: every replicate (joint and block) resamples the Statcast games with the called pitches,
    takes the DP run laws from the resample, and CROSS-FITS the stakes inside the resample (crossfit.crossfit_frame:
    decisions and the perception fit on out-of-fold engines, every flip scored at the in-fold engine's signed stake),
    as the headline point estimate (crossfit.py). variant: the main model, or a random-effects model whose player
    effects are re-estimated (EB posteriors under the refit) and drawn jointly once per replicate (re_seed = b)."""
    mode, b, R = args[:3]
    variant = args[3] if len(args) > 3 else None
    import lik, wp, crossfit as CF
    from headline import evaluate_all, summarise, MAIN
    from final_ci import spec_of
    variant = variant or MAIN
    t = time.time()
    spec = spec_of(variant); is_re = bool(spec.get("re"))
    m = pd.read_parquet("data/called_2026_w.parquet")
    by = {g: d for g, d in m.groupby("game_pk")}; games = np.array(sorted(by))
    rng = np.random.default_rng((3000 if mode == "joint" else 4000) + b)
    if mode == "joint":
        pick = rng.choice(games, len(games))
    else:
        gm = pd.read_parquet("data/feed_games_2026.parquet")[["game_pk", "date", "home_id"]]
        gm = gm[gm.game_pk.isin(games)]
        gm["block"] = gm.home_id.astype(str) + "_" + pd.to_datetime(gm.date).dt.month.astype(str)
        blocks = {k: d.game_pk.values for k, d in gm.groupby("block")}; keys = np.array(sorted(blocks))
        pick = np.concatenate([blocks[k] for k in rng.choice(keys, len(keys))])
    mb = resample_frame(by, pick)
    sc = pd.read_parquet("data/statcast_2026.parquet", columns=sorted(set(wp.SC_COLS) | {"home_score", "away_score", "post_bat_score"}))
    bys = {g: d for g, d in sc.groupby("game_pk")}
    scb = pd.concat([bys[g].assign(game_pk=i) for i, g in enumerate(pick) if g in bys], ignore_index=True)
    run_laws(scb)
    mb = CF.crossfit_frame(mb, scb[wp.SC_COLS], np.random.default_rng([3000 + b, 1]))
    th = np.load(f"data/th_{variant}.npy")
    if is_re and PARAMETRIC_RE:
        # HONZA-132: parametric bootstrap of the hyperparameters for the player-effects model (an 11 x 11-node refit
        # takes ~30 min): theta_b ~ N(theta_hat, H^-1) with the finite-difference Hessian of the exact gradient
        # (robust_fit.py -> data/cov_<variant>.npy); the stream is the game resample with cross-fitted stakes, and the
        # player effects are re-estimated (EB) on it under theta_b and drawn jointly below
        from scipy.optimize import OptimizeResult
        thb = np.random.default_rng([3000 + b, 2]).multivariate_normal(th, np.load(f"data/cov_{variant}.npy"))
        r = OptimizeResult(x=thb, success=True)
    else:
        r, _ = lik.fit(lik.prepare(mb, players=is_re), spec, th0=th, maxiter=60)
    var = dict(theta="override", theta_override=r.x, spec=spec, frame=mb)
    tag = None
    if is_re:
        data = lik.prepare(mb, players=True); tag = f"jb_{variant}_{b}_{os.getpid()}"
        for side in ("off", "def"):
            lik.re_posterior(data, r.x, side, spec).to_parquet(f"data/{tag}_{side}.parquet")
        var.update(re="draw", re_seed=b, re_tag=tag)
    try:
        mm, _, out, _ = evaluate_all(var, R=R, seed=b)
    finally:
        if tag:
            for side in ("off", "def"):
                os.remove(f"data/{tag}_{side}.parquet")
    sm = summarise(out, int(mm.gidx.max() + 1))
    P0 = lik.readable(r.x, spec)
    pols = [k for k in sm if not k.startswith("best_")]
    return dict(b=b, variant=variant, ok=bool(r.success), secs=round(time.time() - t, 1), games=int(len(pick)),
                lost=sm["bench"]["wins"] - sm["observed"]["wins"],
                lost_opt_score=sm["opt_score"]["wins"] - sm["observed"]["wins"],
                wins={k: sm[k]["wins"] for k in pols}, n={k: sm[k]["n"] for k in pols},
                batter={k: sm[k]["batter"] for k in pols}, fielder={k: sm[k]["fielder"] for k in pols},
                best_stake=sm["best_stake"], sr=[P0["off_sr"], P0["def_sr"]])


SEASON_GAMES = 2425        # ABS regular-season games in data/called_2026_w.parquet
PARAMETRIC_RE = True       # HONZA-132: player-effects variants use the parametric hyperparameter bootstrap (job)


def freq_matched(r):
    """HONZA-132 (Astra minor 6, Sol #10): the one-line rule constrained to the teams' challenge rate -- the constant
    delta at which the rule makes as many challenges per team-game as the observed teams (linear interpolation over
    the stake grid, whose challenge rate falls with delta), and its value."""
    ks = sorted([k for k in r["n"] if k.startswith("stake_")], key=lambda k: float(k.split("_")[1]))
    n = np.array([r["n"][k] for k in ks]); w = np.array([r["wins"][k] for k in ks]); d = np.array([float(k.split("_")[1]) for k in ks])
    n0 = r["n"]["observed"]
    if not (n.min() <= n0 <= n.max()):
        return dict(delta=None, wins=None)
    o = np.argsort(n)
    return dict(delta=float(np.interp(n0, n[o], d[o])), wins=float(np.interp(n0, n[o], w[o])))


def intervals(x, point):
    """percentile, basic and normal 95 % intervals of a bootstrap sample x around the point estimate."""
    lo, hi = np.percentile(x, [2.5, 97.5]); sd = float(np.std(x, ddof=1))
    return dict(percentile=[float(lo), float(hi)], basic=[float(2 * point - hi), float(2 * point - lo)],
                normal=[float(point - 1.96 * sd), float(point + 1.96 * sd)], sd=sd, mean=float(np.mean(x)),
                bias=float(np.mean(x) - point))


def summary(rows, mode, R, variant=None):
    """HONZA-130 (Sol review): a block resample has 2,323-2,567 games but headline.summarise divides by 30 teams, so
    the season quantities (lost, wins) of each replicate are rescaled to the 2,425-game season (x SEASON_GAMES /
    games) before the percentiles; the per-team-game counts n are already per game. Joint replicates resample games
    one by one (games = 2,425), so their factor is exactly 1.
    HONZA-132: the point estimate is the cross-fitted one (results/crossfit.json); percentile, basic and normal
    intervals for the loss and the shares; the frequency-matched one-line rule (freq_matched)."""
    from headline import MAIN
    variant = variant or MAIN
    ok = [r for r in rows if r["ok"]]
    sc = lambda r: SEASON_GAMES / r.get("games", SEASON_GAMES)
    lost = np.array([r["lost"] * sc(r) for r in ok])
    cf = json.load(open("results/crossfit.json")).get(variant, {}) if os.path.exists("results/crossfit.json") else {}
    point = cf.get("lost", float(np.mean(lost)))
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    keep = [k for k in KEEP if k in ok[0]["wins"]]
    share = lambda k: np.array([r["wins"][k] / r["wins"]["bench"] for r in ok])
    fm = [freq_matched(r) for r in ok]
    fm_ok = [(f, r) for f, r in zip(fm, ok) if f["wins"] is not None]
    gap_fm = np.array([(f["wins"] - r["wins"]["observed"]) / (r["wins"]["bench"] - r["wins"]["observed"]) for f, r in fm_ok])
    gap_0008 = np.array([(r["wins"]["stake_0.008"] - r["wins"]["observed"]) / (r["wins"]["bench"] - r["wins"]["observed"]) for r in ok])
    return dict(mode=mode, variant=variant, B=len(ok), B_failed=len(rows) - len(ok), R=R, point=point,
                point_source="results/crossfit.json (cross-fitted stakes, S splits)" if cf else "bootstrap mean (no point file)",
                lost=intervals(lost, point), lost_ci=pct(lost),
                shares={k: dict(mean=float(share(k).mean()), ci=pct(share(k))) for k in keep if k != "bench"},
                gap_closed_stake_0008=dict(mean=float(gap_0008.mean()), ci=pct(gap_0008)),
                freq_matched=dict(B=len(fm_ok), delta=pct([f["delta"] for f, _ in fm_ok]),
                                  gap_closed=dict(mean=float(gap_fm.mean()), ci=pct(gap_fm))),
                n=dict(observed=pct([r["n"]["observed"] for r in ok]), bench=pct([r["n"]["bench"] for r in ok])),
                sr=dict(off=pct([r["sr"][0] for r in ok]), deff=pct([r["sr"][1] for r in ok])), rows=rows)


def CF_K():
    import crossfit
    return crossfit.K


def config(mode, R, variant):
    """HONZA-132 (Sol #14): a resumed run must have the same configuration and input data (size and mtime of the
    analysis frame and the variant's point fit); rows from another configuration are never mixed in."""
    st = lambda f: [os.path.getsize(f), int(os.path.getmtime(f))]
    import crossfit
    return dict(mode=mode, R=R, variant=variant, crossfit_K=CF_K(), folds=crossfit.FOLDS, frame=st("data/called_2026_w.parquet"),
                theta=st(f"data/th_{variant}.npy"))


if __name__ == "__main__":
    from headline import MAIN
    mode = sys.argv[1]; B = int(sys.argv[2]); R = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    variant = sys.argv[5] if len(sys.argv) > 5 else MAIN
    sfx = "" if variant == MAIN else f"_{variant}"
    path = f"data/{mode}_boot_rows{sfx}.json"
    cfg = config(mode, R, variant)
    st = json.load(open(path)) if os.path.exists(path) else dict(R=R, config=cfg, rows=[])
    if st.get("config") != cfg:
        raise SystemExit(f"{path}: stored configuration {st.get('config')} differs from {cfg}; move the file to start over")
    done = {r["b"] for r in st["rows"]}
    with mp.get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(job, [(mode, b, R, variant) for b in range(B) if b not in done]):
            st["rows"].append(r); json.dump(st, open(path, "w"))
            print(mode, variant, len(st["rows"]), r["b"], r["secs"], round(r["lost"], 3), flush=True)
    res = summary(st["rows"], mode, st["R"], variant)
    json.dump(res, open(f"results/{mode}_boot{sfx}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
