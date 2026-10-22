"""Two bootstraps the referees asked for, both refitting the main perception model in every replicate.

joint: one game resample drives EVERY fitted component -- the PA kernel and WP / RE engine
       (wp.build_T, wp.solve), the stakes g and RV of every called pitch (stakes.add_stakes), the half-inning and
       extra-inning run laws of the DP (dp_score), the perception fit (lik.fit) and the policy evaluation
       (headline.evaluate_all). The headline bootstrap (headline.py boot) holds the engine fixed; wp_boot.py
       resamples the engine at fixed perception parameters; this is the joint version.
block: resample blocks of games (home team x calendar month, about 200 blocks) instead of games, to allow
       for dependence within a team's homestand / month (same catchers, pitchers, umpire crews, criterion drift);
       engine fixed as in the headline bootstrap.
usage: python src/joint_boot.py <joint|block> <B> [R] [workers]  -> data/<mode>_boot_rows.json (resumable),
       results/<mode>_boot.json"""
import json, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, "src")

KEEP = ["opt_score", "stake_0.008", "runs_dp", "card_pub", "savant", "obs_rule", "observed", "clairvoyant"]


def resample_frame(by, pick):
    return pd.concat([by[g].assign(game_pk=i) for i, g in enumerate(pick)], ignore_index=True)


def job(args):
    mode, b, R = args
    import lik, wp, stakes, dp_score
    from headline import evaluate_all, summarise, MAIN
    from final_ci import spec_of
    t = time.time()
    spec = spec_of(MAIN)
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
    if mode == "joint":
        sc = pd.read_parquet("data/statcast_2026.parquet", columns=sorted(set(wp.SC_COLS) | {"home_score", "away_score", "post_bat_score"}))
        bys = {g: d for g, d in sc.groupby("game_pk")}
        scb = pd.concat([bys[g].assign(game_pk=i) for i, g in enumerate(pick) if g in bys], ignore_index=True)
        T = wp.build_T(sc=scb, verbose=False); W, RE, D = wp.solve(T, verbose=False)
        stakes.T, stakes.W, stakes.RE, stakes.D = T, W, RE, int(D)
        P = dp_score.half_runs_from(scb); dp_score.half_runs = lambda: P
        x = scb[(scb.inning >= 10) & (scb.inning_topbot == "Top")].sort_values(["game_pk", "at_bat_number", "pitch_number"])
        h = x.groupby(["game_pk", "inning"]).agg(a=("bat_score", "first"), z=("post_bat_score", "last"))
        rx = (h.z - h.a).clip(0, dp_score.RMAX).astype(int).values
        Px = np.bincount(rx, minlength=dp_score.RMAX + 1) / max(len(rx), 1); dp_score.extra_runs = lambda: Px
        cols = ["wp_ball", "wp_strike", "re_ball", "re_strike", "g", "g_signed", "RV", "wp_home_now", "half", "diff"]
        mb = stakes.add_stakes(mb.drop(columns=[c for c in cols if c in mb]))
    th = np.load(f"data/th_{MAIN}.npy")
    r, _ = lik.fit(lik.prepare(mb), spec, th0=th, maxiter=60)
    mm, _, out, _ = evaluate_all(dict(theta="override", theta_override=r.x, spec=spec, frame=mb), R=R, seed=b)
    sm = summarise(out, int(mm.gidx.max() + 1))
    P0 = lik.readable(r.x, spec)
    return dict(b=b, ok=bool(r.success), secs=round(time.time() - t, 1), games=int(len(pick)),
                lost=sm["opt_score"]["wins"] - sm["observed"]["wins"], wins={k: sm[k]["wins"] for k in KEEP},
                n={k: sm[k]["n"] for k in KEEP}, sr=[P0["off_sr"], P0["def_sr"]])


def summary(rows, mode, R):
    ok = [r for r in rows if r["ok"]]
    lost = np.array([r["lost"] for r in ok])
    hb = json.load(open("results/headline_boot.json"))
    bl = np.array([r["summary"]["opt_score"]["wins"] - r["summary"]["observed"]["wins"] for r in hb["rows"] if r.get("ok", True)])
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    share = lambda k: np.array([r["wins"][k] / r["wins"]["opt_score"] for r in ok])
    return dict(mode=mode, B=len(ok), B_failed=len(rows) - len(ok), R=R, lost=dict(mean=float(lost.mean()), sd=float(lost.std(ddof=1)), ci=pct(lost)),
                game_refit_boot=dict(B=len(bl), sd=float(bl.std(ddof=1)), ci=pct(bl)),
                sd_ratio_to_game_refit=float(lost.std(ddof=1) / bl.std(ddof=1)),
                shares={k: dict(mean=float(share(k).mean()), ci=pct(share(k))) for k in KEEP if k != "opt_score"},
                sr=dict(off=pct([r["sr"][0] for r in ok]), deff=pct([r["sr"][1] for r in ok])), rows=rows)


if __name__ == "__main__":
    mode = sys.argv[1]; B = int(sys.argv[2]); R = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    path = f"data/{mode}_boot_rows.json"
    st = json.load(open(path)) if os.path.exists(path) else dict(R=R, rows=[])
    done = {r["b"] for r in st["rows"]}
    with mp.get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(job, [(mode, b, R) for b in range(B) if b not in done]):
            st["rows"].append(r); json.dump(st, open(path, "w"))
            print(mode, len(st["rows"]), r["b"], r["secs"], round(r["lost"], 3), flush=True)
    res = summary(st["rows"], mode, st["R"])
    json.dump(res, open(f"results/{mode}_boot.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
