"""Uncertainty from the WP / RE engine, which the refit bootstrap holds fixed.

Replicate b resamples the 2026 Statcast games with replacement, re-estimates the PA kernel and solves the WP / RE engine
(wp.build_T, wp.solve) and the half-inning run distribution of the score-aware DP (dp_score.half_runs_from), recomputes
g and RV for every called pitch (stakes.add_stakes) and re-evaluates every headline policy at the point parameters of
the main spec (headline.evaluate_all, R draws, the same seed in every replicate so that Monte Carlo noise cancels in
the differences). b = -1 is the original engine on the same seed.
The extra spread is combined with the refit bootstrap (results/headline_boot.json) by adding the engine deviation
lost_j - lost_base to every refit replicate (all pairs; the two sources are independent resamples).
usage: python src/wp_boot.py <B> [R] [workers]  -> data/wp_boot_rows.json (resumable), results/wp_boot.json"""
import json, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, "src")

MAIN = "kappa_sr_region_g"        # BIC-best homogeneous model (criterion piecewise-linear in log g)
ROWS = "data/wp_boot_rows.json"
OUT = "results/wp_boot.json"
STAKE_COLS = ["wp_ball", "wp_strike", "re_ball", "re_strike", "g", "RV", "wp_home_now", "half", "diff"]
KEEP = ["opt_score", "stake_0.008", "runs_dp", "card_pub", "savant", "obs_rule", "observed"]


def job(args):
    b, R = args
    import wp, stakes, dp_score
    from headline import evaluate_all, summarise, spec_variant
    t = time.time()
    m = pd.read_parquet("data/called_2026_w.parquet")
    if b >= 0:
        sc = pd.read_parquet("data/statcast_2026.parquet", columns=wp.SC_COLS)
        by = {g: d for g, d in sc.groupby("game_pk")}; games = np.array(sorted(by))
        pick = np.random.default_rng(2000 + b).choice(games, len(games))
        scb = pd.concat([by[g].assign(game_pk=i) for i, g in enumerate(pick)], ignore_index=True)
        T = wp.build_T(sc=scb, verbose=False); W, RE, D = wp.solve(T, verbose=False)
        stakes.T, stakes.W, stakes.RE, stakes.D = T, W, RE, int(D)          # stakes reads the engine from module globals
        P = dp_score.half_runs_from(scb)
        dp_score.half_runs = lambda: P                                       # the score-aware DP's run distribution
        key = ["game_pk", "at_bat_number", "pitch_number"]
        new = stakes.add_stakes(m.drop(columns=STAKE_COLS))[key + ["g", "RV"]]
        m = m.drop(columns=["g", "RV"]).merge(new, on=key, how="left", validate="1:1")
        assert m.g.notna().all()
    mm, _, out, _ = evaluate_all(dict(spec_variant(MAIN), frame=m), R=R, seed=1)
    sm = summarise(out, int(mm.gidx.max() + 1))
    ch = m[m.ch]
    return dict(b=b, secs=round(time.time() - t, 1), lost=sm["opt_score"]["wins"] - sm["observed"]["wins"],
                wins={k: sm[k]["wins"] for k in KEEP}, g_med_challenged=float(ch.g.median()), g_mean=float(m.g.mean()))


def summarise_rows(rows, R):
    base = [r for r in rows if r["b"] < 0][0]; reps = [r for r in rows if r["b"] >= 0]
    lost = np.array([r["lost"] for r in reps]); dev = lost - base["lost"]
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    hb = json.load(open("results/headline_boot.json")); hl = json.load(open("results/headline.json"))
    bl = np.array([r["summary"]["opt_score"]["wins"] - r["summary"]["observed"]["wins"] for r in hb["rows"] if r.get("ok")])
    q = lambda xs, a: np.sort(xs)[min(len(xs) - 1, max(0, int(round(a * (len(xs) - 1)))))]   # as paper/make_numbers.py
    comb = (bl[:, None] + dev[None, :]).ravel()
    share = lambda r, k: r["wins"][k] / r["wins"]["opt_score"]
    return dict(variant=MAIN, B=len(reps), R=R, seed="1 in every replicate (common random numbers)",
                base=dict(lost=base["lost"], wins=base["wins"]),
                lost=dict(mean=float(lost.mean()), sd=float(lost.std(ddof=1)), ci=pct(lost)),
                dev=dict(mean=float(dev.mean()), sd=float(dev.std(ddof=1)), ci=pct(dev)),
                refit=dict(B=len(bl), sd=float(bl.std(ddof=1)), ci=[float(q(bl, 0.025)), float(q(bl, 0.975))],
                           point=hl["wins_lost"]["point"]),
                combined=dict(sd=float(np.hypot(bl.std(ddof=1), dev.std(ddof=1))), ci=[float(q(comb, 0.025)), float(q(comb, 0.975))],
                              how="refit replicate + engine deviation, all B_refit x B_wp pairs"),
                sd_ratio=float(dev.std(ddof=1) / bl.std(ddof=1)),
                shares={k: dict(base=share(base, k), sd=float(np.std([share(r, k) for r in reps], ddof=1)),
                                ci=pct([share(r, k) for r in reps])) for k in KEEP if k != "opt_score"},
                g_med_challenged=dict(base=base["g_med_challenged"], ci=pct([r["g_med_challenged"] for r in reps])),
                rows=rows)


if __name__ == "__main__":
    B = int(sys.argv[1]); R = int(sys.argv[2]) if len(sys.argv) > 2 else 10; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    st = json.load(open(ROWS)) if os.path.exists(ROWS) else dict(R=R, rows=[])
    done = {r["b"] for r in st["rows"]}
    with mp.get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(job, [(b, R) for b in range(-1, B) if b not in done]):
            st["rows"].append(r); json.dump(st, open(ROWS, "w"))
            print(len(st["rows"]), r["b"], r["secs"], round(r["lost"], 3), flush=True)
    res = summarise_rows(st["rows"], st["R"])
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
