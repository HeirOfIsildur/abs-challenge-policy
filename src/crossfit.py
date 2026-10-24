"""cross-fitted stakes for the headline value contrast.

A policy that selects pitches on estimated stakes and is then scored on the same estimates is scored optimistically:
it picks the pitches whose stake happens to be over-estimated. Here every pitch gets two stakes from WP engines fitted
on disjoint sets of games. The games are split at random into K = 5 folds; a pitch in fold k
  DECIDES on g  = the stake from the engine fitted on the other K - 1 folds (clipped at 0, as stakes.add_stakes),
  is SCORED at gv = the signed stake from the engine fitted on fold k alone (signed: clipping a noisy engine's
                    contrasts at zero would bias the score up).
The perception model, the calibration cells, the DP pools and every policy see only g; every successful flip -- of
the benchmark, of the other policies and of the observed challenges -- is valued at gv (headline.evaluate_all reads
the gv column). The decision engine of a pitch is fitted without that pitch's game and the scoring engine on that
pitch's fold alone, so the two engines share no games, which removes the selection on estimated stakes (in a
resample, folds are drawn over the original games, so copies of one game never straddle folds); the perception fit, the beliefs and the DP still use every fold (only the stakes are cross-fitted);
what is left is the noise of a 1/K-season engine, which averages out in the sum (gv is unbiased up to the engine's
non-linearity) and is part of the bootstrap. K = 5 rather than 2: the benchmark then decides on engines fitted to 80 %
of the games, nearly as accurate as the full-season engine (with K = 2 the half-season decision engine degrades the
benchmark's decisions and biases the gap down by several hundredths, results/crossfit.json "K2" rows). The pooling
weight of every fold engine is scaled to its share of the games (SHRINK_W x share). The DP run laws (half-inning runs) stay those of the whole sample: they
are not stakes and select nothing.
usage: python src/crossfit.py point <S> [R] [variants] [K]  -> results/crossfit.json (S random K-fold splits per variant;
       K other than 5 is stored under "<variant>_K<k>")"""
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "src")

STAKE_COLS = ["wp_ball", "wp_strike", "re_ball", "re_strike", "g", "g_signed", "RV", "wp_home_now", "half", "diff", "gv"]
K = 5
FOLDS = "orig_game"              # resampled copies of a game share a fold (crossfit_frame)
SHRINK_W = 20                     # wp._kernel's pooling weight (pitches) for the full-season engine


def engine(sc, shrink_w=SHRINK_W):
    import wp
    T = wp.build_T(sc=sc, verbose=False, shrink_w=shrink_w); W, RE, D = wp.solve(T, verbose=False)
    return T, W, RE, int(D)


def stakes_under(eng, m):
    """g, g_signed and RV of every row of m under the engine eng (stakes.add_stakes; module state restored)."""
    import stakes
    keep = (stakes.T, stakes.W, stakes.RE, stakes.D)
    stakes.T, stakes.W, stakes.RE, stakes.D = eng
    try:
        x = stakes.add_stakes(m.drop(columns=[c for c in STAKE_COLS if c in m]).assign(_row=np.arange(len(m))))
    finally:
        stakes.T, stakes.W, stakes.RE, stakes.D = keep
    x = x.sort_values("_row")
    assert len(x) == len(m) and (x._row.values == np.arange(len(m))).all()
    return x


def crossfit_frame(m, sc, rng, k=K):
    """m: called-pitch frame, sc: the Statcast frame of the same games (same game_pk labels, wp.SC_COLS). Returns m
    with g / g_signed / RV from the out-of-fold engines, gv from the in-fold engines, and fold."""
    # in a resample every drawn copy of a game has its own game_pk; the folds are drawn
    # over the ORIGINAL games (orig_game) so that all copies of a game share a fold and no game is in both the
    # decision and the scoring engine of a pitch. Without resampled copies (orig_game absent) this is the old split.
    if "orig_game" in m:
        key = m.drop_duplicates("game_pk").set_index("game_pk")["orig_game"]
        origs = np.unique(key.values)
        fold_orig = dict(zip(origs, rng.permutation(len(origs)) % k))
        fold_of = {g: fold_orig[o] for g, o in key.items()}
    else:
        games = np.unique(m.game_pk.values)
        fold_of = dict(zip(games, rng.permutation(len(games)) % k))
    fm = m.game_pk.map(fold_of).values; fs = sc.game_pk.map(fold_of).values
    parts = []
    for f in range(k):
        rows = m[fm == f]
        # the pooling weight scales with the fold's share of the games, so that every fold engine shrinks its count
        # cells as much as the full-season engine does (a fixed weight attenuates a small engine's contrasts)
        out_eng = engine(sc[(fs != f) & ~pd.isna(fs)], SHRINK_W * (k - 1) / k); in_eng = engine(sc[fs == f], SHRINK_W / k)
        dec = stakes_under(out_eng, rows); val = stakes_under(in_eng, rows)
        dec["gv"] = val.g_signed.values; dec["fold"] = f
        parts.append(dec.set_index(rows.index))
    return pd.concat(parts).loc[m.index].drop(columns="_row")


def statcast(cols=None):
    import wp
    return pd.read_parquet("data/statcast_2026.parquet", columns=cols or wp.SC_COLS)


def point(names, S, R, k=K):
    """S random 2-fold splits of the 2026 games per variant: refit on the decision stakes, evaluate every policy with
    the flips scored at the in-fold stakes."""
    import lik
    from headline import evaluate_all, summarise, spec_variant
    m0 = pd.read_parquet("data/called_2026_w.parquet"); sc = statcast()
    path = "results/crossfit.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name in names:
        rows = []
        for s in range(S):
            t = time.time()
            mb = crossfit_frame(m0, sc, np.random.default_rng(5000 + s), k)
            v = spec_variant(name); spec = v["spec"]
            if spec.get("re"):
                # the player-effects model keeps its full-sample hyperparameters (an 11 x 11-node refit takes
                # ~30 min per split); the players' posteriors are recomputed on the cross-fitted frame below
                from scipy.optimize import OptimizeResult
                r = OptimizeResult(x=np.load(v["theta"]), success=True)
            else:
                r, _ = lik.fit(lik.prepare(mb), spec, th0=np.load(v["theta"]), maxiter=60)
            var = dict(v, theta_override=r.x, frame=mb)
            if spec.get("re"):           # player effects: EB posteriors under the refit, on this frame
                data = lik.prepare(mb, players=True); tag = f"cf_{name}_{os.getpid()}"
                for side in ("off", "def"):
                    lik.re_posterior(data, r.x, side, spec).to_parquet(f"data/{tag}_{side}.parquet")
                var.update(re_tag=tag, re="draw", re_seed=s)     # one joint draw of the player effects per split
            mm, _, out, dr = evaluate_all(var, R=R, seed=s)
            sm = summarise(out, int(mm.gidx.max() + 1))
            row = dict(split=s, ok=bool(r.success), secs=round(time.time() - t, 1),
                       lost=sm["bench"]["wins"] - sm["observed"]["wins"],
                       lost_opt_score=sm["opt_score"]["wins"] - sm["observed"]["wins"],
                       wins={k: sm[k]["wins"] for k in sm if not k.startswith("best_")},
                       n={k: sm[k]["n"] for k in sm if not k.startswith("best_")},
                       batter={k: sm[k]["batter"] for k in sm if not k.startswith("best_")},
                       fielder={k: sm[k]["fielder"] for k in sm if not k.startswith("best_")},
                       best_stake=sm["best_stake"], best_savant_c=sm.get("best_savant_c"),
                       lost_per_draw=[d["totals"]["bench"] - sm["observed"]["wins"] for d in dr])
            rows.append(row)
            print(name, s, row["secs"], round(row["lost"], 4), flush=True)
        lost = np.array([r["lost"] for r in rows])
        res = json.load(open(path)) if os.path.exists(path) else {}      # re-read: other writers (decomp) may have added keys
        res[name if k == K else f"{name}_K{k}"] = dict(S=S, R=R, K=k, lost=float(lost.mean()),
                         lost_sd_over_splits=float(lost.std(ddof=1)) if S > 1 else None,
                         lost_se=float(lost.std(ddof=1) / np.sqrt(S)) if S > 1 else None,
                         wins={k: float(np.mean([r["wins"][k] for r in rows])) for k in rows[0]["wins"]},
                         n={k: float(np.mean([r["n"][k] for r in rows])) for k in rows[0]["n"]},
                         batter={k: float(np.mean([r["batter"][k] for r in rows])) for k in rows[0]["batter"]},
                         fielder={k: float(np.mean([r["fielder"][k] for r in rows])) for k in rows[0]["fielder"]},
                         rows=rows)
        json.dump(res, open(path, "w"), indent=1)
    return res


def decomp(R=10, seeds=(5000, 5001)):
    """where the cross-fitting effect comes from. On a split, with common random numbers, the main model is
    evaluated four ways: decide on the full-season stakes or on the out-of-fold engine's, and score every flip at the
    full-season stakes or at the in-fold engine's signed contrast (decision stakes: perception refitted on them).
    full/full is the uncorrected estimate, out/in the cross-fitted one; full/in isolates the selection optimism
    (decisions as in the uncorrected estimate, scored by an engine that did not see those games), out/full the cost of
    deciding on a (K-1)/K-season engine."""
    import lik
    from headline import evaluate_all, summarise, spec_variant, MAIN
    m0 = pd.read_parquet("data/called_2026_w.parquet"); sc = statcast()
    v = spec_variant(MAIN); spec = v["spec"]
    res = {}
    for seed in seeds:
        mb = crossfit_frame(m0, sc, np.random.default_rng(seed))
        out = {}
        for dec in ("full", "out"):
            fr = m0.copy() if dec == "full" else mb.drop(columns=["gv"]).copy()
            r, _ = lik.fit(lik.prepare(fr), spec, th0=np.load(v["theta"]), maxiter=60)
            for scr in ("full", "in"):
                f2 = fr.assign(gv=(m0.g.values if scr == "full" else mb.gv.values))
                mm, _, o, _ = evaluate_all(dict(v, theta_override=r.x, frame=f2), R=R, seed=1)
                sm = summarise(o, int(mm.gidx.max() + 1))
                out[f"{dec}_{scr}"] = dict(lost=sm["bench"]["wins"] - sm["observed"]["wins"], bench=sm["bench"]["wins"],
                                           observed=sm["observed"]["wins"])
                print(seed, dec, scr, round(out[f"{dec}_{scr}"]["lost"], 4), flush=True)
        res[str(seed)] = out
    keys = ["full_full", "full_in", "out_full", "out_in"]
    avg = {k: float(np.mean([res[s][k]["lost"] for s in res])) for k in keys}
    st = json.load(open("results/crossfit.json")) if os.path.exists("results/crossfit.json") else {}
    st["decomp"] = dict(R=R, seeds=list(seeds), K=K, splits=res, mean_lost=avg,
                        selection_optimism=avg["full_full"] - avg["full_in"],
                        decision_engine_cost=avg["full_in"] - avg["out_in"])
    json.dump(st, open("results/crossfit.json", "w"), indent=1)
    print(json.dumps(st["decomp"]["mean_lost"], indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "decomp":
        decomp(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    if sys.argv[1] == "point":
        S = int(sys.argv[2]); R = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        names = sys.argv[4].split(",") if len(sys.argv) > 4 else ["kappa_sr_region_g"]
        point(names, S, R, int(sys.argv[5]) if len(sys.argv) > 5 else K)
