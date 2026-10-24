"""the full policy cross-fit (outer split), the headline estimator.

crossfit.py cross-fits only the stakes: the perception fit, the players' beliefs and the DP still see every game. Here
EVERY component of the benchmark and of the rules is learned on the other folds and applied, unchanged, to held-out
games, as the temporal check does for Mar-Jun -> Jul-Sep (oos.frozen). The games are split at random into K = 5
folds (over the original games in a resample, so copies of a game share a fold). For held-out fold k:
  learned on the other K - 1 folds ("training games"):
    the WP engine and the stakes g every decision uses (crossfit.engine; also for the held-out pitches),
    the DP run laws (half-inning and extra-inning runs),
    the perception model theta (lik.fit on the training pitches with their training-engine stakes); for the
      player-effects model also the players' EB posteriors (lik.re_posterior on the training pitches),
    the stake tercile cut points of the calibration cells and the per-cell belief maps (sim.draw on the training
      pitches: the auxiliary experience the belief is learned from),
    the score-aware DP continuation values (dp_score.solve_dp_score on training pools and hazards), re-solved for the
      lapses the benchmark keeps (lambda),
    the one-line rule's best constant on the training games (the published 0.008 is also run);
  held-out fold k, per draw r: truths and signals from the training theta (the fitted world, as oos.frozen world =
    "train"), the player's belief = the training map of the pitch's cell applied to its signal, the frozen DP and
    rules run along the recorded held-out stream; every successful flip and every observed challenge is scored at the
    signed stake of the engine fitted on fold k alone (gv, as crossfit.py).
The K held-out folds cover the season once, so the totals are per team-season (/ 30), as headline.summarise. The
estimate is averaged over S random splits. The (rho, lambda) grid of intervention.py is evaluated on the same fold
fits (cells), the primary intervention being rho = 1, lambda = 0.
usage: python src/policy_crossfit.py point <model> <S> [R] [workers] [grid]   -> results/policy_crossfit.json
       python src/policy_crossfit.py boot <model> <B> [R] [workers] [shard/nshard] -> data/pcf_boot_rows_<model>[_<shard>].json
       python src/policy_crossfit.py bootsum <model>                               -> results/pcf_boot_<model>.json
       python src/policy_crossfit.py collect -                                     -> results/policy_crossfit.json"""
import json, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, "src")

K = int(os.environ.get("PCF_K", 5))       # K = 2 and 10 as a sensitivity (environment)
RHOS = [1.0, 0.75, 0.5, 0.25]
LAMS = [0.0, 0.5, 1.0]
RES = "results/policy_crossfit.json"
SC_EXTRA = {"home_score", "away_score", "post_bat_score"}


def _naux():
    """auxiliary draws for the belief maps (sim.N_AUX, default 2); environment PCF_NAUX overrides it."""
    import sim
    if os.environ.get("PCF_NAUX"):
        sim.N_AUX = int(os.environ["PCF_NAUX"])
    return sim.N_AUX


def folds_of(m, rng, k=K):
    """game_pk -> fold, drawn over the original games when the frame is a resample (crossfit.crossfit_frame)."""
    if "orig_game" in m:
        key = m.drop_duplicates("game_pk").set_index("game_pk")["orig_game"]
        origs = np.unique(key.values)
        fo = dict(zip(origs, rng.permutation(len(origs)) % k))
        return {g: fo[o] for g, o in key.items()}
    games = np.unique(m.game_pk.values)
    return dict(zip(games, rng.permutation(len(games)) % k))


def run_laws_from(sc):
    """(half-inning run law by slot, extra-inning top-half run law) from a Statcast frame (joint_boot.run_laws)."""
    import dp_score
    P = dp_score.half_runs_from(sc)
    x = sc[(sc.inning >= 10) & (sc.inning_topbot == "Top")].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    h = x.groupby(["game_pk", "inning"]).agg(a=("bat_score", "first"), z=("post_bat_score", "last"))
    rx = (h.z - h.a).clip(0, dp_score.RMAX).astype(int).values
    return P, np.bincount(rx, minlength=dp_score.RMAX + 1) / max(len(rx), 1)


def with_laws(laws, f, *a):
    import dp_score
    keep = dp_score.half_runs, dp_score.extra_runs
    dp_score.half_runs, dp_score.extra_runs = (lambda: laws[0]), (lambda: laws[1])
    try:
        return f(*a)
    finally:
        dp_score.half_runs, dp_score.extra_runs = keep


def prep_rows(M, sel):
    """the rows sel of the loaded frame M as a policy frame (whole games, own gidx)."""
    x = M[sel].reset_index(drop=True)
    x["gidx"] = x.game_pk.astype("category").cat.codes
    return x


def base_args(x):
    return (x.gidx.values, x.is_home_team.values, x.new_extra.values, x.slot.values.astype(np.int64),
            x.kin.values.astype(np.int64), x.inning.values.astype(np.int64), x.balls.values.astype(np.int64),
            x.strikes.values.astype(np.int64), x.dbin.values.astype(np.int64))


def run_rule(x, W, pt, s, gv, pol, dconst, D4h, D4a, kap, Cnh=None, Cna=None, card=None, q1=0.0, q2=0.0):
    """value and challenges of policy pol along the stream x (policies.run_path2: decide on x.g, score at gv).
    Cnh / Cna: the runs-objective rule's token values (pol 6); card, q1, q2: a 48-cell card (pol 5)."""
    import policies as P
    z1 = np.zeros((19, 80, 3)); z2 = np.zeros((19, 3)); zc = np.zeros((4, 3, 2, 2))
    ng = int(x.gidx.max() + 1)
    v, n, _, _, _ = P.run_path2(*base_args(x), W, pt, s, x.g.values, gv, x.RV.values, kap, (x.side == "off").values,
                                pol, z1, z1, z2 if Cnh is None else Cnh, z2 if Cna is None else Cna,
                                zc if card is None else card, q1, q2, dconst, D4h, D4a, ng)
    return v, n


def fold_fit(m, sc, fold_of, k, name, th0, tag):
    """everything fitted on the training folds of held-out fold k: the frame with training-engine stakes (all rows)
    and the held-out rows' in-fold signed stake gv, the training theta, the run laws and the players' posteriors."""
    import lik, crossfit as CF, wp
    from final_ci import spec_of
    spec = spec_of(name); is_re = bool(spec.get("re"))
    fm = m.game_pk.map(fold_of).values; fs = sc.game_pk.map(fold_of).values
    tr_sc = sc[(fs != k) & ~pd.isna(fs)]
    out_eng = CF.engine(tr_sc[wp.SC_COLS], CF.SHRINK_W * (K - 1) / K)
    in_eng = CF.engine(sc[fs == k][wp.SC_COLS], CF.SHRINK_W / K)
    mk = CF.stakes_under(out_eng, m.assign(g_true=m.g.values)).drop(columns="_row")   # g_true: the frame's own stakes
    te = fm == k
    mk["gv"] = mk.g.values
    mk.loc[te, "gv"] = CF.stakes_under(in_eng, m[te]).g_signed.values
    mk["fold"] = fm
    laws = run_laws_from(tr_sc)
    t = time.time()
    r, _ = lik.fit(lik.prepare(mk[~te], players=is_re), spec, th0=th0, maxiter=60)
    fit_secs = time.time() - t
    if is_re:
        data = lik.prepare(mk[~te], players=True)
        for side in ("off", "def"):
            lik.re_posterior(data, r.x, side, spec).to_parquet(f"data/{tag}_{side}.parquet")
    return dict(frame=mk, theta=r.x, ok=bool(r.success), laws=laws, spec=spec, is_re=is_re, tag=tag if is_re else None,
                fit_secs=round(fit_secs, 1))


def load_fold(ff, j):
    """the full frame under the training fit (player effects: joint draw j from the training posteriors), with the
    opportunity set, start-of-half score, extra-inning starts, kin and calibration cells (training cut points)."""
    import sim, dp_score
    var = dict(theta="override", theta_override=ff["theta"], spec=ff["spec"], frame=ff["frame"])
    if ff["is_re"]:
        var.update(re="draw", re_seed=j, re_tag=ff["tag"])
    M, _ = sim.load(var)
    M = dp_score.add_dstart(M)
    M = sim.opportunities(M)
    M["new_extra"] = sim.extra_starts(M)
    M["kin"] = sim.kin_index(M)
    tr = (M.fold != ff["k"]).values
    if ff.get("train_q5") and ff["is_re"]:
        # the players' noise quintiles of the belief cells with cut points from the
        # training rows only (sim.apply_lik cuts them over the whole frame: transductive use of a covariate)
        import lik
        reg = pd.Series(lik.region(M)).astype(str).values; q5 = np.zeros(len(M), int); lsr = np.log(M.sr.values)
        for side in ("off", "def"):
            k = (M.side == side).values
            cuts = np.quantile(lsr[k & tr], [0.2, 0.4, 0.6, 0.8])
            q5[k] = np.searchsorted(cuts, lsr[k], side="right")
        M["pgrp"] = np.char.add(np.char.add(reg.astype(str), "_"), q5.astype(str))
    M["cal"] = sim.cal_groups(M, sim.stake_cuts(M[tr]))
    if ff.get("cal_fine"):
        # belief cells by the exact count (12 cells) instead of three count classes
        parts = pd.Series(M["cal"].values).str.rsplit("_", n=2, expand=True)
        cnt = M.balls.astype(int).astype(str).values + M.strikes.astype(int).astype(str).values
        M["cal"] = (parts[0] + "_c" + cnt + "_" + parts[2]).values
    return M, var


ZD = (np.zeros((19, 13, 80, 3)), np.zeros((19, 13, 80, 3)))
A_GRID = list(np.round(np.arange(-3.0, 1.01, 0.25), 2))     # intercept shifts of kappa (in)
B_GRID = list(np.round(np.arange(-2.0, 0.51, 0.125), 3))    # added slope of kappa on log g (in per unit log g)


def shifted(x, kap, ab):
    """the teams' criterion by challenges held, shifted per side by a + b (log g - LG_MID)."""
    import lik
    lg = np.log(np.clip(x.g.values, 1e-4, None)) - lik.LG_MID
    off = (x.side == "off").values
    a = np.where(off, ab["off"][0], ab["def"][0]); b = np.where(off, ab["off"][1], ab["def"][1])
    return np.ascontiguousarray(kap + (a + b * lg)[:, None])


def evaluate_fold(ff, R, seed, cells, J):
    """R draws on held-out fold ff["k"] (player effects: J joint draws, R // J truths each). cells: (rho, lam) pairs.
    Returns the mean over draws of every policy's value and challenges on the fold, the per-draw values and the
    observed value (real challenges at gv). ff["world"] (coverage_pcf truth only): a generating-model variant whose
    truths, signals and lapses replace the training fit's on the held-out games, every flip then scored at the frame's
    own stakes g_true -- the value, in the true world, of the benchmark learned on the training folds."""
    import sim, dp_score
    from headline import STAKES
    from oos import apply_maps
    law = (ff["spec"] or {}).get("law", "normal")
    rhos = sorted({c[0] for c in cells}, reverse=True)
    _naux()
    per_draw = []; n_unseen = 0
    nj = J if ff["is_re"] else 1
    reps = max(R // nj, 1)
    fr = ff["frame"]; te = (fr.fold == ff["k"]).values & fr.ch.values
    obs = float((fr.gv.values[te] * fr.ch_over.values[te]).sum()); n_obs = int(te.sum())
    for j in range(nj):
        M, var = load_fold(ff, j)
        tr = (M.fold != ff["k"]).values
        Mtr, Mte = prep_rows(M, tr), prep_rows(M, ~tr)
        ng_te = int(Mte.gidx.max() + 1)
        kap_tr, kap_te = sim.kappa_by_count(Mtr), sim.kappa_by_count(Mte)
        lap_tr, lap_te = Mtr.lapse.values, Mte.lapse.values
        gsc = Mte.gv.values
        if ff.get("world") is not None:
            Mw, _ = sim.load(dict(ff["world"], frame=ff["frame"]))
            Mw = sim.opportunities(Mw)
            assert len(Mw) == len(M) and (Mw.game_pk.values == M.game_pk.values).all()
            ww = Mw[~tr].reset_index(drop=True)
            Mte = Mte.assign(w=ww.w.values, sm=ww.sm.values, sr=ww.sr.values, pi=ww.pi.values)
            lap_te = ww.lapse.values; gsc = Mte.g_true.values
        for r in range(reps):
            row = {}
            for rho in rhos:
                rng = np.random.default_rng([seed, ff["k"], j, r, int(rho * 100)])
                vr = dict(var, rho=rho)
                # training: the experience the beliefs are learned from, the DP pools, the training-best constant
                Wt, st, ptt, maps = sim.draw(Mtr, rng, vr, return_maps=True)
                u_tr = rng.random(len(Mtr))
                # held-out: truths and the player's (perceptual) signal from the training theta; the frozen maps
                W, sp = sim.truth_signal(Mte, rng, law, rho)
                pt, miss_ = apply_maps(Mte, sp, maps); n_unseen += miss_
                sb = sp + rng.standard_normal(len(Mte)) * Mte.sr.values * np.sqrt(1.0 - rho) if rho < 1 else sp
                u = rng.random(len(Mte))
                s_obs = np.where(u < lap_te, -1e9, sb)        # the behavioural rule lapses (headline.evaluate_all)
                v, n = run_rule(Mte, W, pt, s_obs, gsc, 4, 0.0, *ZD, kap_te)
                row[(rho, "obs_rule")] = (v.sum(), n.sum())
                for lam in sorted({c[1] for c in cells if c[0] == rho}):
                    ptb_tr = np.where(u_tr < (1 - lam) * lap_tr, 0.0, ptt)
                    D4h, _ = with_laws(ff["laws"], dp_score.solve_dp_score, Mtr, ptb_tr, True)
                    D4a, _ = with_laws(ff["laws"], dp_score.solve_dp_score, Mtr, ptb_tr, False)
                    ptb = np.where(u < (1 - lam) * lap_te, 0.0, pt)
                    v, n = run_rule(Mte, W, ptb, sp, gsc, 7, 0.0, D4h, D4a, kap_te)
                    row[(rho, lam, "bench")] = (v.sum(), n.sum())
                    if rho == 1.0 and lam == 0.0:
                        # the rules, played by the benchmark's player (same lapses); frozen constants
                        trv = {}
                        for d in STAKES:
                            vt, _ = run_rule(Mtr, Wt, ptb_tr, st, Mtr.g.values, 2, d, D4h, D4a, kap_tr); trv[d] = vt.sum()
                            v, n = run_rule(Mte, W, ptb, sp, gsc, 2, d, D4h, D4a, kap_te)
                            row[(rho, lam, f"stake_{d}")] = (v.sum(), n.sum())
                        dtr = max(STAKES, key=lambda d: trv[d])
                        row[(rho, lam, "stake_frozen")] = row[(rho, lam, f"stake_{dtr}")]
                        row[(rho, lam, "d_train")] = (dtr, 0.0)
                        v, n = run_rule(Mte, W, ptb, sp, gsc, 1, 0.0, D4h, D4a, kap_te)
                        row[(rho, lam, "savant")] = (v.sum(), n.sum())
                        if ff.get("ext"):
                            # the public competitors in the same cross-fit, every
                            # tuned object learned on the training folds
                            import policies as P
                            from headline import SAVANT_C
                            Cnh, Cna = P.runs_token_values(Mtr, ptb_tr, sim.solve_dp)
                            v, n = run_rule(Mte, W, ptb, sp, gsc, 6, 0.0, D4h, D4a, kap_te, Cnh=Cnh, Cna=Cna)
                            row[(rho, lam, "runs_dp")] = (v.sum(), n.sum())
                            for nm_, (cd, q1, q2) in (("card_refit", P.card_refit(Mtr, Mtr.kin.values, D4h, D4a)),
                                                      ("card_pub", P.card_published(Mtr))):
                                v, n = run_rule(Mte, W, ptb, sp, gsc, 5, 0.0, D4h, D4a, kap_te, card=cd, q1=q1, q2=q2)
                                row[(rho, lam, nm_)] = (v.sum(), n.sum())
                            trs = {}
                            for c in SAVANT_C:
                                vt, _ = run_rule(Mtr, Wt, ptb_tr, st, Mtr.g.values, 1, c, D4h, D4a, kap_tr); trs[c] = vt.sum()
                                v, n = run_rule(Mte, W, ptb, sp, gsc, 1, c, D4h, D4a, kap_te)
                                row[(rho, lam, f"savant_c_{c}")] = (v.sum(), n.sum())
                            row[(rho, lam, "savant_tuned")] = row[(rho, lam, f"savant_c_{max(SAVANT_C, key=lambda c: trs[c])}")]
                            # the teams' own rule class s > kappa(x), re-tuned on the training folds: only the intercept,
                            # only the slope on log g, or both, per side (coordinate search), then frozen on fold k
                            s_tr = np.where(u_tr < lap_tr, -1e9, st)
                            # "both" is a coordinate search: slope, then intercept given it, then slope again
                            for nm_, steps in (("retune_intercept", ["a"]), ("retune_slope", ["b"]), ("retune_both", ["b", "a", "b"])):
                                ab = {"off": (0.0, 0.0), "def": (0.0, 0.0)}
                                for sd_ in ("off", "def"):
                                    for st_ in steps:
                                        best = None
                                        for x_ in (A_GRID if st_ == "a" else B_GRID):
                                            t_ab = dict(ab)
                                            t_ab[sd_] = (x_, ab[sd_][1]) if st_ == "a" else (ab[sd_][0], x_)
                                            vt, _ = run_rule(Mtr, Wt, ptt, s_tr, Mtr.g.values, 4, 0.0, *ZD, shifted(Mtr, kap_tr, t_ab))
                                            if best is None or vt.sum() > best[0]:
                                                best = (vt.sum(), t_ab[sd_])
                                        ab[sd_] = best[1]
                                v, n = run_rule(Mte, W, pt, s_obs, gsc, 4, 0.0, *ZD, shifted(Mte, kap_te, ab))
                                row[(rho, lam, nm_)] = (v.sum(), n.sum())
            per_draw.append(row)
    keys = per_draw[0].keys()
    tot = {kk: [float(np.mean([d[kk][0] for d in per_draw])), float(np.mean([d[kk][1] for d in per_draw]))] for kk in keys}
    return dict(tot=tot, per_draw=[{kk: d[kk][0] for kk in keys} for d in per_draw], observed=obs, n_obs=n_obs, ng=ng_te,
                unseen_cell_rows_per_draw=n_unseen / max(len(per_draw) * len(rhos), 1))


def one_split(m, sc, name, seed, R, cells, J=4, th0=None, keep_theta=False, world=None, ext=False, cal_fine=False,
              train_q5=False):
    """the full policy cross-fit on one random K-fold split of the frame m (Statcast sc of the same games): season
    totals per team-season of every policy, the observed value and the wins lost (primary benchmark - observed)."""
    fold_of = folds_of(m, np.random.default_rng([seed, 99]))
    th0 = np.load(f"data/th_{name}.npy") if th0 is None else th0
    folds = []; ok = True; thetas = []; secs = []
    for k in range(K):
        tag = f"pcf_{name}_{seed}_{k}_{os.getpid()}"
        ff = fold_fit(m, sc, fold_of, k, name, th0, tag); ff["k"] = k; ff["world"] = world
        ff["ext"] = ext; ff["cal_fine"] = cal_fine; ff["train_q5"] = train_q5
        try:
            folds.append(evaluate_fold(ff, R, seed, cells, J))
        finally:
            if ff["tag"]:
                for side in ("off", "def"):
                    p = f"data/{ff['tag']}_{side}.parquet"
                    if os.path.exists(p):
                        os.remove(p)
        ok &= ff["ok"]; thetas.append(ff["theta"].tolist()); secs.append(ff["fit_secs"])
    keys = [kk for kk in folds[0]["tot"] if kk[-1] != "d_train"]
    ng = sum(f["ng"] for f in folds)
    obs = sum(f["observed"] for f in folds) / 30
    out = {"|".join(str(x) for x in kk): dict(wins=sum(f["tot"][kk][0] for f in folds) / 30,
                                              n=sum(f["tot"][kk][1] for f in folds) / (2 * ng)) for kk in keys}
    nd = min(len(f["per_draw"]) for f in folds)          # draws are paired across folds by index
    lost_draws = [sum(f["per_draw"][i][(1.0, 0.0, "bench")] for f in folds) / 30 - obs for i in range(nd)]
    res = dict(seed=seed, ok=bool(ok), observed=obs, n_observed=sum(f["n_obs"] for f in folds) / (2 * ng), games=ng,
               policies=out, lost=out["1.0|0.0|bench"]["wins"] - obs, lost_draws=lost_draws,
                      d_train=[f["tot"][(1.0, 0.0, "d_train")][0] for f in folds], fit_secs=secs,
               unseen_cell_rows=float(sum(f.get("unseen_cell_rows_per_draw", 0) for f in folds)))
    if keep_theta:
        res["thetas"] = thetas
    return res


def statcast_frame():
    import wp
    return pd.read_parquet("data/statcast_2026.parquet", columns=sorted(set(wp.SC_COLS) | SC_EXTRA))


def cells_for(name, grid):
    from final_ci import spec_of
    lapse = bool(spec_of(name).get("lapse"))
    if not grid:
        return [(1.0, 0.0)]
    return [(rho, lam) for rho in RHOS for lam in (LAMS if lapse else [0.0])]


def point_job(args):
    name, s, R, grid = args[:4]
    flags = args[4] if len(args) > 4 else {}
    t = time.time()
    m = pd.read_parquet("data/called_2026_w.parquet"); sc = statcast_frame()
    r = one_split(m, sc, name, 7000 + s, R, cells_for(name, grid), keep_theta=True, **flags)
    r.update(split=s, secs=round(time.time() - t, 1))
    return r


def boot_job(args):
    """one bootstrap replicate: a game resample (with its Statcast games), then the full policy cross-fit on it --
    every component refitted inside the replicate, the perception hyperparameters included (warm start at the point
    estimate), one random split per replicate, folds over the original games."""
    name, b, R = args
    from joint_boot import resample_frame
    t = time.time()
    m = pd.read_parquet("data/called_2026_w.parquet"); sc = statcast_frame()
    by = {g: d for g, d in m.groupby("game_pk")}; games = np.array(sorted(by))
    pick = np.random.default_rng(8000 + b).choice(games, len(games))
    mb = resample_frame(by, pick)
    bys = {g: d for g, d in sc.groupby("game_pk")}
    scb = pd.concat([bys[g].assign(game_pk=i) for i, g in enumerate(pick) if g in bys], ignore_index=True)
    r = one_split(mb, scb, name, 9000 + b, R, [(1.0, 0.0)], keep_theta=True)
    r.update(b=b, secs=round(time.time() - t, 1))
    return r


def config(name, R):
    st = lambda f: [os.path.getsize(f), int(os.path.getmtime(f))]
    return dict(model=name, R=R, K=K, folds="orig_game", frame=st("data/called_2026_w.parquet"), theta=st(f"data/th_{name}.npy"))


def summarise_point(rows):
    lost = np.array([r["lost"] for r in rows])
    pol = {k: dict(wins=float(np.mean([r["policies"][k]["wins"] for r in rows])),
                   n=float(np.mean([r["policies"][k]["n"] for r in rows]))) for k in rows[0]["policies"]}
    obs = float(np.mean([r["observed"] for r in rows]))
    bench = pol["1.0|0.0|bench"]["wins"]
    rules = {k.split("|")[-1]: dict(wins=v["wins"], n=v["n"], share=v["wins"] / bench, gap_closed=(v["wins"] - obs) / (bench - obs))
             for k, v in pol.items() if k.startswith("1.0|0.0|") and k != "1.0|0.0|bench"}
    ob = pol["1.0|obs_rule"]
    rules["obs_rule"] = dict(wins=ob["wins"], n=ob["n"], share=ob["wins"] / bench, gap_closed=(ob["wins"] - obs) / (bench - obs))
    grid = [dict(rho=float(k.split("|")[0]), lam=float(k.split("|")[1]), lost=v["wins"] - obs, bench=v["wins"], n=v["n"])
            for k, v in pol.items() if k.endswith("|bench")]
    return dict(S=len(rows), lost=float(lost.mean()),
                lost_sd_over_splits=float(lost.std(ddof=1)) if len(rows) > 1 else None,
                lost_se_splits=float(lost.std(ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else None,
                lost_mc_se_within=float(np.mean([np.std(r["lost_draws"], ddof=1) / np.sqrt(len(r["lost_draws"])) for r in rows])),
                bench=bench, observed=obs, n_bench=pol["1.0|0.0|bench"]["n"],
                n_observed=float(np.mean([r["n_observed"] for r in rows])), rules=rules, grid=grid,
                d_train=sorted({d for r in rows for d in r["d_train"]}), ok=all(r["ok"] for r in rows))


def summarise_rules_flat(pt_):
    """policies of a point summary in the "1.0|0.0|<rule>" layout of a replicate."""
    out = {f"1.0|0.0|{k}": dict(wins=v["wins"], n=v["n"]) for k, v in pt_["rules"].items() if k != "obs_rule"}
    out["1.0|0.0|bench"] = dict(wins=pt_["bench"], n=pt_["n_bench"])
    return out


def boot_summary(name):
    import glob
    from joint_boot import intervals
    seen = {}; cfgs = []
    for p in sorted(glob.glob(f"data/pcf_boot_rows_{name}*.json")):
        st_ = json.load(open(p)); cfgs.append(st_["config"])
        for r in st_["rows"]:
            seen.setdefault(r["b"], r)
    if any(c != cfgs[0] for c in cfgs):                 # one configuration only
        raise SystemExit(f"bootstrap shards of {name} have different configurations")
    rows = [seen[b] for b in sorted(seen)]
    ok = [r for r in rows if r["ok"]]
    pts = json.load(open(RES))
    point = (pts.get(name + "_grid") or pts[name])["lost"]          # the headline point runs carry the (rho, lambda) grid
    lost = np.array([r["lost"] for r in rows])
    gc = lambda r, k: (r["policies"][k]["wins"] - r["observed"]) / (r["policies"]["1.0|0.0|bench"]["wins"] - r["observed"])
    sh = lambda r, k: r["policies"][k]["wins"] / r["policies"]["1.0|0.0|bench"]["wins"]
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    rk = ("stake_0.008", "stake_frozen", "savant")
    out = dict(model=name, B=len(rows), B_ok=len(ok), point=point, lost=intervals(lost, point),
               lost_ok_only=intervals(np.array([r["lost"] for r in ok]), point) if len(ok) > 2 else None,
               gap_closed={k: dict(ci=pct([gc(r, f"1.0|0.0|{k}") for r in rows]),
                                   mean=float(np.mean([gc(r, f"1.0|0.0|{k}") for r in rows]))) for k in rk},
               share={k: dict(ci=pct([sh(r, f"1.0|0.0|{k}") for r in rows])) for k in rk},
               obs_rule_gap=dict(ci=pct([gc(r, "1.0|obs_rule") for r in rows])),
               bench=pct([r["policies"]["1.0|0.0|bench"]["wins"] for r in rows]), observed=pct([r["observed"] for r in rows]),
               secs_median=float(np.median([r["secs"] for r in rows])))
    # the one-line rule at the teams' challenge rate (joint_boot.freq_matched on the training-free constant
    # grid of each replicate) and the gap to the simulated behavioural rule
    from joint_boot import freq_matched
    def fm(r):
        x = dict(n={k.split("|")[-1]: v["n"] for k, v in r["policies"].items() if k.startswith("1.0|0.0|stake_0")},
                 wins={k.split("|")[-1]: v["wins"] for k, v in r["policies"].items() if k.startswith("1.0|0.0|stake_0")})
        x["n"]["observed"] = r["n_observed"]
        f = freq_matched(x)
        b = r["policies"]["1.0|0.0|bench"]["wins"]
        return None if f["wins"] is None else (f["wins"] - r["observed"]) / (b - r["observed"])
    fms = [x for x in map(fm, rows) if x is not None]
    pt_ = pts.get(name + "_grid") or pts[name]
    out["freq_matched"] = dict(B=len(fms), ci=pct(fms) if fms else None, point=fm(dict(policies={k: dict(wins=v["wins"], n=v["n"])
                               for k, v in summarise_rules_flat(pt_).items()}, n_observed=pt_["n_observed"], observed=pt_["observed"])))
    out["sim_rule_gap"] = dict(ci=pct([r["policies"]["1.0|0.0|bench"]["wins"] - r["policies"]["1.0|obs_rule"]["wins"] for r in rows]))
    json.dump(out, open(f"results/pcf_boot_{name}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    mode, name = sys.argv[1], sys.argv[2]
    if mode == "point":
        S = int(sys.argv[3]); R = int(sys.argv[4]) if len(sys.argv) > 4 else 8
        workers = int(sys.argv[5]) if len(sys.argv) > 5 else 1; opts = set(sys.argv[6:])
        grid = "grid" in opts
        flags = dict(ext="ext" in opts, cal_fine="calfine" in opts, train_q5="trainq5" in opts)   # sensitivities
        key = name + ("_grid" if grid else "") + ("_ext" if flags["ext"] else "") + ("_calfine" if flags["cal_fine"] else "") \
            + ("_trainq5" if flags["train_q5"] else "") \
            + (f"_K{K}" if K != 5 else "") + (f"_naux{os.environ['PCF_NAUX']}" if os.environ.get("PCF_NAUX") else "")
        path = f"data/pcf_point_rows_{key}.json"
        cfg = dict(config(name, R), grid=grid, **({k: v for k, v in flags.items() if v}))
        st = json.load(open(path)) if os.path.exists(path) else dict(config=cfg, rows=[])
        if st["config"] != cfg:
            raise SystemExit(f"{path}: configuration differs; move it to start over")
        done = {r["split"] for r in st["rows"]}
        jobs = [(name, s, R, grid, flags) for s in range(S) if s not in done]
        it = map(point_job, jobs) if workers == 1 else mp.get_context("spawn").Pool(workers, maxtasksperchild=1).imap_unordered(point_job, jobs)
        for r in it:
            st["rows"].append(r); json.dump(st, open(path, "w"))
            print(name, "split", r["split"], r["secs"], round(r["lost"], 4), flush=True)
        res = json.load(open(RES)) if os.path.exists(RES) else {}
        res[key] = dict(summarise_point(st["rows"]), R=R, K=K, rows=st["rows"])
        json.dump(res, open(RES, "w"), indent=1)
        print(json.dumps({k: v for k, v in res[key].items() if k != "rows"}, indent=1))
    elif mode == "boot":
        B = int(sys.argv[3]); R = int(sys.argv[4]) if len(sys.argv) > 4 else 8
        workers = int(sys.argv[5]) if len(sys.argv) > 5 else 4
        shard, nshard = (map(int, sys.argv[6].split("/")) if len(sys.argv) > 6 else (0, 1))
        path = f"data/pcf_boot_rows_{name}" + (f"_{shard}" if nshard > 1 else "") + ".json"
        cfg = config(name, R)
        st = json.load(open(path)) if os.path.exists(path) else dict(config=cfg, rows=[])
        if st["config"] != cfg:
            raise SystemExit(f"{path}: configuration differs; move it to start over")
        done = {r["b"] for r in st["rows"]}
        jobs = [(name, b, R) for b in range(B) if b % nshard == shard and b not in done]
        with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(boot_job, jobs):
                st["rows"].append(r); json.dump(st, open(path, "w"))
                print(name, "boot", len(st["rows"]), r["b"], r["secs"], round(r["lost"], 4), flush=True)
    elif mode == "bootsum":
        boot_summary(name)
    elif mode == "collect":          # rebuild results/policy_crossfit.json from every point-row file (jobs ran in parallel)
        import glob
        res = {}
        for p in sorted(glob.glob("data/pcf_point_rows_*.json")):
            key = p[len("data/pcf_point_rows_"):-len(".json")]
            st = json.load(open(p))
            res[key] = dict(summarise_point(st["rows"]), R=st["config"]["R"], K=st["config"].get("K", K), rows=st["rows"])
        json.dump(res, open(RES, "w"), indent=1)
        print({k: round(v["lost"], 4) for k, v in res.items()})
