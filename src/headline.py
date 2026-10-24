"""HONZA-86: every policy of the headline table on the same draws, the point estimate and the refit bootstrap.

evaluate_all(variant dict, R, seed) -> per-policy arrays value/challenges [game, team, side] (path-sum accounting,
as sim.evaluate), averaged over R draws of (truth, signal).  Policies: opt, stake_<d> grid (best d and the published
d = 0.01), savant, conf50, obs_rule, card_refit, card_pub, runs_dp (see policies.py), observed.
usage:
  python src/headline.py point [R]                 -> data/headline_point.pkl (point estimate, main spec + per-player,
                                                      with the Monte Carlo SE of wins lost over the R draws)
  python src/headline.py boot <B> [R] [workers]    -> results/headline_boot.json (game bootstrap with model refits; R = 10 per
                                                      replicate since HONZA-100, R = 2 before: results/headline_boot_R2.json;
                                                      same resamples and seeds as final_ci.py)
  python src/headline.py noref <B> [R] [workers]   -> results/headline_noref.json: game bootstrap without refits for the
                                                      per-player model (player effects redrawn from their EB posteriors
                                                      in every replicate) and, for comparison, the main spec
  python src/headline.py lapse [R]                -> results/lapse_wins.json: wins lost under the lapse-rate model (HONZA-114)
  python src/headline.py variants [R]              -> results/headline_variants.json: every HONZA-73 perception/truth
                                                      variant re-evaluated with all policies (wins lost model spread)"""
import json, os, pickle, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, "src")

MAIN = "kappa_sr_region_g"        # HONZA-122: BIC-best homogeneous model (criterion piecewise-linear in log g)
CO_PRIMARY = "re_lapse_gh11"      # HONZA-132: the BIC-best model, player effects + lapse (11 x 11 nodes), co-primary
STAKES = [0.004, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.0125, 0.015, 0.02]
SAVANT_C = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]   # run costs for Savant's p >= c/(c + RV)


def evaluate_all(variant, R=10, seed=1, keep_draws=False, keep_maps=False):
    """keep_maps (HONZA-130, with keep_draws): each kept draw also carries the player's per-cell calibration maps
    (sim.draw return_maps) under "maps"; the random numbers and every output are unchanged."""
    import sim, policies as P, dp_score
    lite = bool(variant.get("lite"))
    m, _ = sim.load(variant)
    m = dp_score.add_dstart(m)
    m = sim.opportunities(m)                         # HONZA-122: no truth-based |w| < 8 filter on the policies
    if variant.get("w8"):                            # attribution only (audit/honza122/attribution.py): the old filter
        m = m[m.w.abs() < 8].reset_index(drop=True)
    m["new_extra"] = sim.extra_starts(m)
    m["cal"] = m.pgrp.values if variant.get("cal_old") else sim.cal_groups(m)   # cal_old: attribution only
    topup = P.topup_flags(m)
    kin = sim.kin_index(m).astype(np.int64)
    ng = int(m.gidx.max() + 1)
    rng = np.random.default_rng(seed)
    isoff = (m.side == "off").values
    # HONZA-132: cross-fitted stakes -- a frame with column gv scores every flip (policies and observed challenges) at
    # gv, while every decision, pool and fit uses g (crossfit.py); without gv the two coincide
    gv = m.gv.values if "gv" in m else m.g.values
    base = (m.gidx.values, m.is_home_team.values, m.new_extra.values, m.slot.values.astype(np.int64), kin,
            m.inning.values.astype(np.int64), m.balls.values.astype(np.int64), m.strikes.values.astype(np.int64),
            m.dbin.values.astype(np.int64))
    out, draws = {}, []
    zcard = np.zeros((4, 3, 2, 2))
    for r in range(R):
        if keep_maps:
            W, s, pt, maps = sim.draw(m, rng, variant, return_maps=True)
        else:
            W, s, pt = sim.draw(m, rng, variant); maps = None
        D4h, _ = dp_score.solve_dp_score(m, pt, True); D4a, _ = dp_score.solve_dp_score(m, pt, False)
        if lite:
            # HONZA-132 (coverage study): only the benchmark, the fitted rule and the one-line rule
            Dh = Da = np.zeros((19, 80, 3)); Cnh = Cna = np.zeros((19, 3)); q1 = q2 = 0.0
            pols = dict(opt_score=(7, 0.0, zcard), obs_rule=(4, 0.0, zcard), **{"stake_0.008": (2, 0.008, zcard)})
        else:
            Dh, _ = sim.solve_dp(m, pt, True); Da, _ = sim.solve_dp(m, pt, False)
            Cnh, Cna = P.runs_token_values(m, pt, sim.solve_dp)
            cr, q1, q2 = P.card_refit(m, kin, D4h, D4a)
            cp, _, _ = P.card_published(m)
            pols = dict(opt_score=(7, 0.0, zcard), opt=(0, 0.0, zcard), savant=(1, 0.0, zcard), conf50=(3, 0.0, zcard), obs_rule=(4, 0.0, zcard),
                        card_refit=(5, 0.0, cr), card_pub=(5, 0.0, cp), runs_dp=(6, 0.0, zcard))
            for d in STAKES:
                pols[f"stake_{d}"] = (2, d, zcard)
            for c in SAVANT_C:                           # HONZA-122 (Sonnet, Astra): Savant's form with a tuned run cost
                pols[f"savant_c_{c}"] = (1, c, zcard)
        # HONZA-122 (Sol): the lapse-rate model's behavioural rule challenges only if it does not lapse
        lap = m.lapse.values if "lapse" in m else np.zeros(len(m))
        u_lap = np.random.default_rng([seed, r, 7]).random(len(m))
        lapsed = u_lap < lap
        s_obs = np.where(lapsed, -1e9, s) if lap.max() > 0 else s
        if lap.max() > 0:              # (Sol) the benchmark played by a player who still lapses: a lapsed pitch is skipped
            pols["opt_score_lapsed"] = (7, 0.0, zcard)
            # HONZA-130 (GPT-6-Astra review): re-solve the DP for a player who knows they lapse -- the pools carry the
            # lapsed opportunities as worthless (p = 0), so the continuation values price the lost opportunities
            ptl = np.where(lapsed, 0.0, pt)
            D4hl, _ = dp_score.solve_dp_score(m, ptl, True); D4al, _ = dp_score.solve_dp_score(m, ptl, False)
        # HONZA-132: "bench" = the benchmark of the stated intervention. A change of decision rule keeps the share 1 - lam
        # of the lapses (lam = variant["lam"], default 0: the primary intervention keeps every lapse), coupled to the
        # behavioural rule's lapses through the same uniforms; the DP is re-solved for that lapse rate. rho enters
        # through sim.draw. Without lapses bench = opt_score.
        lam = float(variant.get("lam", 0.0))
        if lap.max() > 0 and lam > 0:
            lapsed_b = u_lap < (1.0 - lam) * lap
            ptb = np.where(lapsed_b, 0.0, pt)
            D4hb, _ = dp_score.solve_dp_score(m, ptb, True); D4ab, _ = dp_score.solve_dp_score(m, ptb, False)
        elif lap.max() > 0:
            ptb, D4hb, D4ab = ptl, D4hl, D4al
        else:
            ptb, D4hb, D4ab = pt, D4h, D4a
        pols["bench"] = (7, 0.0, zcard)
        firsts, vals = {}, {}
        for name, (pc, dc, card) in pols.items():
            # HONZA-132: under a lapse model every implementable rule is played by the same player as the benchmark of
            # the stated intervention, so it skips the same lapsed pitches (ptb); opt_score stays the lapses-removed
            # reading and obs_rule lapses through s_obs. Without lapses ptb = pt.
            if name == "opt_score_lapsed":
                ptn = np.where(lapsed, 0.0, pt)
            elif name in ("opt_score", "obs_rule"):
                ptn = pt
            else:
                ptn = ptb
            d4h, d4a = (D4hl, D4al) if name == "opt_score_lapsed" else ((D4hb, D4ab) if name == "bench" else (D4h, D4a))
            v, n, w, first, firstc = P.run_path2(*base, W, ptn, s_obs if pc == 4 else s, m.g.values, gv, m.RV.values,
                                        sim.kappa_by_count(m), isoff, pc, Dh, Da, Cnh, Cna, card, q1, q2, dc, d4h, d4a, ng)
            if name not in out:
                out[name] = [np.zeros_like(v), np.zeros_like(n)]
            out[name][0] += v / R; out[name][1] += n / R
            firsts[name] = (first, firstc); vals[name] = v
        if not lite:
            cv, cn = P.clairvoyant2(m.gidx.values, m.is_home_team.values, topup, W, ptb, m.g.values, gv, ng)
            vz = np.zeros((ng, 2, 2)); nz = np.zeros((ng, 2, 2)); vz[:, :, 0] = cv; nz[:, :, 0] = cn   # side not split
            if "clairvoyant" not in out:
                out["clairvoyant"] = [np.zeros_like(vz), np.zeros_like(nz)]
            out["clairvoyant"][0] += vz / R; out["clairvoyant"][1] += nz / R; vals["clairvoyant"] = vz
        totals = {k: float(x.sum() / 30) for k, x in vals.items()}              # wins per team-season of this draw
        if keep_draws:
            draws.append(dict(W=W, s=s, pt=pt, Dh=Dh, Da=Da, D4h=D4h, D4a=D4a, Cnh=Cnh, Cna=Cna, card_refit=cr, card_pub=cp, q=(q1, q2),
                              pols=pols, firsts=firsts, vals=vals, totals=totals))
            if keep_maps:
                draws[-1]["maps"] = maps
        else:
            draws.append(dict(totals=totals))
    # observed: realised value of the actual challenges (ABS ruling), same per-game layout as sim.evaluate
    mm, _ = sim.load(variant)
    obs = np.zeros((ng, 2, 2)); obsn = np.zeros((ng, 2, 2))
    ch = mm[mm.ch]
    gmap = dict(zip(m.game_pk.unique(), range(ng)))
    gi = ch.game_pk.map(gmap).values; ti = ch.is_home_team.values.astype(int); si = (ch.side == "off").values.astype(int)
    gvo = ch.gv if "gv" in ch else ch.g
    np.add.at(obs, (gi, ti, si), (gvo * ch.ch_over).values); np.add.at(obsn, (gi, ti, si), 1)
    out["observed"] = [obs, obsn]
    return m, kin, out, draws


def summarise(out, ngames):
    """wins per team-season (30 teams), challenges per team-game, by side; best stake d."""
    tot = {k: v[0].sum() / 30 for k, v in out.items()}
    stakes = [k for k in out if k.startswith("stake_")]
    res = dict(best_stake=max(stakes, key=lambda k: tot[k]) if stakes else None)
    sav = [k for k in out if k.startswith("savant_c_")]
    if sav:
        res["best_savant_c"] = max(sav, key=lambda k: tot[k])
    for k, (v, n) in out.items():
        res[k] = dict(wins=float(tot[k]), batter=float(v[:, :, 1].sum() / 30), fielder=float(v[:, :, 0].sum() / 30),
                      n=float(n.sum() / (2 * ngames)), n_batter=float(n[:, :, 1].sum() / (2 * ngames)),
                      n_fielder=float(n[:, :, 0].sum() / (2 * ngames)))
    return res


def spec_variant(name, **kw):
    from final_ci import spec_of
    spec = spec_of(name)
    extra = dict(re_tag=name) if spec.get("re") else {}
    return dict(theta=f"data/th_{name}.npy", spec=spec, re=bool(spec.get("re")), **extra, **kw)


def one(args):
    """one refit-bootstrap replicate: identical resample, seed and refit to final_ci.one, all policies."""
    b, variant, R = args
    import lik
    from final_ci import spec_of, resample
    spec = spec_of(variant)
    m = pd.read_parquet("data/called_2026_w.parquet"); by = {g: d for g, d in m.groupby("game_pk")}; games = np.array(sorted(by))
    rng = np.random.default_rng(1000 + b)
    mb = resample(by, games, rng)
    th = np.load(f"data/th_{variant}.npy"); t = time.time()
    r, _ = lik.fit(lik.prepare(mb, players=bool(spec.get("re"))), spec, th0=th, maxiter=60)
    _, _, out, _ = evaluate_all(dict(theta="override", theta_override=r.x, spec=spec, frame=mb, re=bool(spec.get("re"))), R=R, seed=b)
    return dict(b=b, ok=bool(r.success), theta=r.x.tolist(), secs=round(time.time() - t, 1), summary=summarise(out, len(games)))


def one_noref(args):
    """game resample (same as final_ci) at fixed parameters; per-player model with a fresh EB draw of player effects."""
    b, variant, R = args
    from final_ci import resample
    m = pd.read_parquet("data/called_2026_w.parquet"); by = {g: d for g, d in m.groupby("game_pk")}; games = np.array(sorted(by))
    mb = resample(by, games, np.random.default_rng(1000 + b))
    extra = dict(re="draw", re_seed=b) if variant == "re" else {}
    _, _, out, _ = evaluate_all(dict(spec_variant(variant), frame=mb, **extra), R=R, seed=b)
    return dict(b=b, variant=variant, summary=summarise(out, len(games)))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "point":
        R = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        res = {}
        for name in (MAIN, CO_PRIMARY):
            t = time.time()
            if spec_variant(name)["spec"].get("re"):
                # HONZA-132 (Sol #7, Astra #6): the co-primary player-effects model integrates over the players' JOINT
                # posterior: 8 joint draws of the effects (sim.apply_lik re="draw"), R / 8 draws of truths and signals each
                parts = [evaluate_all(dict(spec_variant(name), re="draw", re_seed=k), R=max(R // 8, 1), seed=1 + k)
                         for k in range(8)]
                m, kin = parts[0][0], parts[0][1]
                out = {kk: [sum(p[2][kk][0] for p in parts) / 8, sum(p[2][kk][1] for p in parts) / 8] for kk in parts[0][2]}
                dr = [d for p in parts for d in p[3]]
            else:
                m, kin, out, dr = evaluate_all(spec_variant(name), R=R, seed=1)
            ng = int(m.gidx.max() + 1)
            obs = out["observed"][0].sum() / 30
            lost = np.array([d["totals"]["bench"] - obs for d in dr])              # HONZA-100: Monte Carlo SE of the point
            res[name] = dict(out={k: (x[0], x[1]) for k, x in out.items()}, games=m.game_pk.unique(), summary=summarise(out, ng),
                             mc=dict(R=R, lost_per_draw=lost.tolist(), lost_sd_per_draw=float(lost.std(ddof=1)),
                                     lost_mc_se=float(lost.std(ddof=1) / np.sqrt(R))))
            s = res[name]["summary"]
            print(name, round(time.time() - t), {k: round(s[k]["wins"], 3) for k in s if not k.startswith("best_")}, s["best_stake"], flush=True)
        pickle.dump(res, open("data/headline_point.pkl", "wb"))
    elif mode == "boot":
        B = int(sys.argv[2]); R = int(sys.argv[3]) if len(sys.argv) > 3 else 2; workers = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        path = "results/headline_boot.json"
        rows = json.load(open(path))["rows"] if os.path.exists(path) else []
        done = {r["b"] for r in rows}
        with mp.get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(one, [(b, MAIN, R) for b in range(B) if b not in done]):
                rows.append(r)
                json.dump(dict(variant=MAIN, R=R, B_done=len(rows), rows=rows), open(path, "w"))
                print(len(rows), r["b"], r["secs"], round(r["summary"]["opt_score"]["wins"] - r["summary"]["observed"]["wins"], 3), flush=True)
    elif mode == "noref":
        B = int(sys.argv[2]); R = int(sys.argv[3]) if len(sys.argv) > 3 else 2; workers = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        path = "results/headline_noref.json"
        rows = json.load(open(path))["rows"] if os.path.exists(path) else []
        done = {(r["variant"], r["b"]) for r in rows}
        jobs = [(b, v, R) for b in range(B) for v in ("re", MAIN) if (v, b) not in done]
        with mp.get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(one_noref, jobs):
                rows.append(r)
                json.dump(dict(R=R, rows=rows), open(path, "w"))
                if len(rows) % 20 == 0:
                    print(len(rows), flush=True)
    elif mode == "lapse":                       # HONZA-114: wins lost under the lapse-rate model (robust_fit.py main_lapse)
        R = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        m, _, out, _ = evaluate_all(spec_variant("main_lapse"), R=R, seed=1)
        s = summarise(out, int(m.gidx.max() + 1))
        res = {k: s[k] for k in ("opt_score", "observed", "stake_0.008", "obs_rule", "opt_score_lapsed") if k in s}
        res["lost"] = s["opt_score"]["wins"] - s["observed"]["wins"]; res["R"] = R
        json.dump(res, open("results/lapse_wins.json", "w"), indent=1)
        print(round(res["lost"], 3))
    elif mode == "variants":
        R = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        names = [v for v in ("sm_edge", "sm_region", "normal", "logistic", "t10", "t5", "t3", "sr_edge", "sr_region",
                             "sr_slope", "sr_region_slope", "kappa_region", "kappa_sr_region", "kappa_sr_region_slope",
                             "kappa_sr_region_g", "re", "main_logistic", "main_t10", "main_t5", "main_t3", "main_lapse",
                             "main_glow", "main_sr_g", "linear_lapse", "re_lapse")
                 if os.path.exists(f"data/th_{v}.npy")]
        res = {}
        if len(sys.argv) > 3:          # HONZA-130: re-evaluate only the named variants, merged into the stored file
            names = [v for v in sys.argv[3].split(",") if v in names]
            res = json.load(open("results/headline_variants.json"))["variants"]
        for nm in names:
            m, _, out, _ = evaluate_all(spec_variant(nm), R=R, seed=1)
            res[nm] = summarise(out, int(m.gidx.max() + 1))
            s = res[nm]; print(nm, round(s["opt_score"]["wins"] - s["observed"]["wins"], 3), round(s["opt"]["wins"] - s["observed"]["wins"], 3), flush=True)
            json.dump(dict(R=R, variants=res), open("results/headline_variants.json", "w"), indent=1)
