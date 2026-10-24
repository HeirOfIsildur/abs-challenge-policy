"""Bar-2 threshold gaps and the sqrt(n) / Delta^2 growth of the option value, recomputed on the score-aware
calibrated DP (dp_score.py) under the main spec (kappa_sr_region) with the isotonic posterior calibration.
These numbers came from the pooled DP of (analyze.py, boot_thresholds.py, results/delta_by_remaining.json).

Definitions (kept from analyze.py / boot_thresholds.py):
  reference pitch  g_ref = median stake g of the challenged pitches of 2026 (full sample, fixed in every replicate);
  threshold        p*(slot, d, c) = Delta / (g_ref + Delta), Delta = Delta(slot, d, k = 0, c), the continuation at the
                   first opportunity of the half-inning; averaged over the home and the away DP;
  gaps (pp)        early_vs_late_c1 = p*(top 1st, c=1) - p*(top 8th, c=1); early_vs_late_c2 the same with c=2;
                   c1_vs_c2_mid = p*(top 5th, c=1) - p*(top 5th, c=2).
Score state: "tied" uses d = 0 at the start of the half-inning; "occ" averages p* (and Delta) over d with the share
of real 2026 half-innings that start at d in that slot (the top of the 1st always starts tied, so it is the same in
both).  Score dependence: p* at the same pitch for a team leading by -3, 0, +3 (team perspective: home d, away -d).
sqrt(n): per side, Delta_1(slot, k = 0) for slots 0..17 against n = expected opportunities left in regulation from the
start of the slot (this one included): "model" n follows the DP's own score chain from d (rounded n(slot, d) per
half-inning, P(runs | slot), game-over rules; no extras); "pooled" n = sum over remaining slots of opportunities per
game (the definition).  Fits a + b sqrt(n), a + b n, Delta_1^2 = a + b n (R^2); Delta_2/Delta_1 at the first
pitch (top of the 1st, tied).
CI: game bootstrap with model refits, reusing the B = 120 refitted parameter vectors of results/headline_boot.json
(same resample and seed as headline.py boot / final_ci.py), DP re-solved on each resample with R draws.
usage: python src/refresh_thresholds.py [R_point=10] [R_boot=4] [workers=4]
  -> results/thresholds_score.json, figures/fig1_threshold_score.png, figures/fig_delta_sqrt.png
     (per-replicate rows cached in data/thresholds_score_boot.json)"""
import json, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, "src")

MAIN = "kappa_sr_region_g"        # BIC-best homogeneous model (criterion piecewise-linear in log g)
SL = dict(early=0, mid5=8, late8=14)
B_OWN = 200                       # replicates when no stored refit bootstrap exists          # top of the 1st, 5th, 8th
LEADS = (-3, 0, 3)
SQRT_VERSION = 130                # rows tagged v = 130 carry sqrt fits on the unrounded n (remaining)


def frame(variant):
    import sim, dp_score
    m, _ = sim.load(variant)
    m = dp_score.add_dstart(m)
    return sim.opportunities(m)


def solve(m, variant, R, seed):
    """mean over R draws of Delta[side(0 away, 1 home), slot, d, k, c]; the player's calibration maps of the first draw.
    the maps are sim.draw's own per-cell belief maps (sim.cal_groups: region x count
    class x stake tercile, fitted on the auxiliary draws), not a side x region refit on the scored draw; return_maps
    uses the same random numbers, so D is unchanged."""
    import sim, dp_score
    rng = np.random.default_rng(seed)
    D = np.zeros((2, 19, dp_score.ND, dp_score.KMAX, 3)); cal = None
    for r in range(R):
        if r == 0:
            W, s, pt, cal = sim.draw(m, rng, variant, return_maps=True)
        else:
            W, s, pt = sim.draw(m, rng, variant)
        for h in (0, 1):
            D[h] += dp_score.solve_dp_score(m, pt, bool(h))[0] / R
    return D, cal


def occupancy(m):
    """share of half-innings starting at each d, per slot (real stream / resample)."""
    import dp_score
    hk = m[["game_pk", "slot", "dbin"]].drop_duplicates(["game_pk", "slot"])
    w = np.zeros((19, dp_score.ND)); np.add.at(w, (hk.slot.values, hk.dbin.values), 1)
    return w / np.maximum(w.sum(1, keepdims=True), 1)


def remaining(m, home):
    """expected regulation opportunities left from the start of slot s at start differential d: 'model' [18, ND]
    (DP score chain) and 'pooled' [18] (definition).
    the opportunities per half-inning are the unrounded expectation E min(N, KMAX) =
    sum_k prod_{j<=k} h_j of the count law the DP uses (dp_score.hazard on the team's count in every observed
    half-inning at (slot, d), zeros included, as solve_dp_score), not the rounded mean count."""
    import dp_score
    DM, ND = dp_score.DM, dp_score.ND
    P = dp_score.half_runs(); dv = np.arange(-DM, DM + 1)
    isht = m.is_home_team.values == home; sl_, db = m.slot.values, m.dbin.values
    key = pd.DataFrame(dict(game=m.game_pk.values, inn=m.inning.values, top=m.top.values, slot=sl_, d=db, own=isht))
    hk = key.groupby(["game", "inn", "top"]).agg(slot=("slot", "first"), d=("d", "first"), n=("own", "sum")).reset_index()
    n = np.zeros((19, ND))
    for (s, d), grp in hk[hk.slot < 18].groupby(["slot", "d"]):
        n[int(s), int(d)] = float(np.cumprod(dp_score.hazard(grp.n.values)).sum())
    N = np.zeros((18, ND)); Nstart = np.zeros(ND)
    for s in range(17, -1, -1):
        Nn = Nstart.copy()
        if s == 16:
            Nn[dv > 0] = 0.0
        sgn = 1 if s % 2 == 1 else -1
        Nend = sum(P[s, r] * Nn[np.clip(dv + sgn * r, -DM, DM) + DM] for r in range(P.shape[1]))
        Nstart = n[s] + Nend; N[s] = Nstart
    ng = m.game_pk.nunique()
    per = np.bincount(sl_[isht], minlength=19)[:18] / ng
    return N, per[::-1].cumsum()[::-1]


def fit(x, y):
    X = np.column_stack([np.ones_like(x), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    return dict(a=float(b[0]), b=float(b[1]), R2=float(1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))))


def summarise(D, m, gref, cal=None, P=None):
    import dp_score
    DM = dp_score.DM
    w = occupancy(m)
    ps = lambda x: x / (gref + x)
    res = dict(g_ref=gref)
    # thresholds (home and away averaged, as analyze.py)
    th = {}
    for nm, s in SL.items():
        for c in (1, 2):
            Dk = D[:, s, :, 0, c]                                  # [side, d]
            th[f"{nm}_c{c}_tied"] = float(ps(Dk[:, DM]).mean())
            th[f"{nm}_c{c}_occ"] = float((ps(Dk) @ w[s]).mean())
            for L in LEADS:                                        # team lead L: home d = L, away d = -L
                th[f"{nm}_c{c}_lead{L:+d}"] = float((ps(Dk[1, DM + L]) + ps(Dk[0, DM - L])) / 2)
    res["thresholds"] = th
    gaps = {}
    for sc in ["tied", "occ"] + [f"lead{L:+d}" for L in LEADS]:
        e = "tied" if sc.startswith("lead") else sc                # 1st inning: always tied at the start
        gaps[f"early_vs_late_c1_pp_{sc}"] = 100 * (th[f"early_c1_{e}"] - th[f"late8_c1_{sc}"])
        gaps[f"early_vs_late_c2_pp_{sc}"] = 100 * (th[f"early_c2_{e}"] - th[f"late8_c2_{sc}"])
        gaps[f"c1_vs_c2_mid_pp_{sc}"] = 100 * (th[f"mid5_c1_{sc}"] - th[f"mid5_c2_{sc}"])
    res["gaps_pp"] = gaps
    # score dependence of the late threshold (the stake g_ref is held fixed)
    res["late8_c1_lead-3_minus_lead+3_pp"] = 100 * (th["late8_c1_lead-3"] - th["late8_c1_lead+3"])
    # sqrt(n) / Delta^2 growth
    sq = {}
    for h, nm in ((1, "home"), (0, "away")):
        Nm, Np = remaining(m, bool(h))
        d1t, d2t = D[h, :18, DM, 0, 1], D[h, :18, DM, 0, 2]
        d1o = (D[h, :18, :, 0, 1] * w[:18]).sum(1); d2o = (D[h, :18, :, 0, 2] * w[:18]).sum(1)
        no = (Nm * w[:18]).sum(1)
        rows = dict(tied_model=(Nm[:, DM], d1t), tied_pooled=(Np, d1t), occ_model=(no, d1o))
        r = {}
        for k, (n, d1) in rows.items():
            r[k] = dict(sqrt=fit(np.sqrt(n), d1), linear=fit(n, d1), sq_on_n=fit(n, d1 ** 2))
        r["D2_over_D1_first_pitch"] = float(d2t[0] / d1t[0])
        r["D1_first"], r["D2_first"] = float(d1t[0]), float(d2t[0])
        r["rows_tied"] = [[int(s), float(Nm[s, DM]), float(Np[s]), float(d1t[s]), float(d2t[s])] for s in range(18)]
        r["rows_occ"] = [[int(s), float(no[s]), float(d1o[s]), float(d2o[s])] for s in range(18)]
        r["monotone_tied"] = bool(np.all(np.diff(d1t) <= 1e-12) and np.all(d1t >= d2t - 1e-12))
        sq[nm] = r
    res["sqrt_n"] = sq
    # threshold-by-inning curves for the figure: side 'def' = fielding (home in the top), 'off' = batting
    curves = {}
    for side in ("def", "off"):
        for c in (1, 2):
            for sc in ("tied", "occ"):
                v = []
                for i in range(1, 10):
                    sh = 2 * (i - 1) + (1 if side == "off" else 0); sa = 2 * (i - 1) + (0 if side == "off" else 1)
                    if sc == "tied":
                        v.append(float((ps(D[1, sh, DM, 0, c]) + ps(D[0, sa, DM, 0, c])) / 2))
                    else:
                        v.append(float((ps(D[1, sh, :, 0, c]) @ w[sh] + ps(D[0, sa, :, 0, c]) @ w[sa]) / 2))
                curves[f"{side}_c{c}_{sc}"] = v
    res["curves"] = curves
    if cal is not None:
        res["revealed"] = revealed(m, cal, P, gref)
    return res


def revealed(m, cal, P, gref):
    """thresholds teams reveal: calibrated confidence at the signal cut-off kappa(state) of the fitted rule, at g_ref,
    1 ball / 1 strike, averaged over zone regions (weights: that side's opportunities), as plots.py.
    kappa(state) is the full fitted criterion, built with the analysis's own features
    (lik.design, the log g hinges lik.gfeat and the low-stake hinge lik.glow, as sim.apply_lik), not beta and kreg
    only; the belief is read from sim.draw's map of the cell (sim.cal_groups: region x count class x stake tercile)
    that a 1-1 pitch at g_ref falls in."""
    import lik, sim
    reg = lik.region(m); out = {}
    cuts = sim.stake_cuts(m)
    assert not any("_" in str(x) for x in np.unique(m.pgrp.values)), "revealed(): homogeneous model (no noise quintiles)"
    for side in ("def", "off"):
        q = P[side]; beta = np.asarray(q["beta"]); kreg = np.asarray(q.get("kreg", np.zeros(4)))
        k = (m.side == side).values
        wr = np.bincount(reg[k], minlength=4) / k.sum()
        tq = int(np.searchsorted(cuts[side], gref, side="right"))            # stake tercile of g_ref
        for c in (1, 2):
            e = pd.DataFrame(dict(g=np.full(9, gref), left=np.full(9, c), inning=np.arange(1, 10),
                                  strikes=np.ones(9, int), balls=np.ones(9, int)))
            kap0 = lik.design(e) @ beta
            if "kg" in q: kap0 = kap0 + lik.gfeat(e) @ np.asarray(q["kg"])
            if "kgl" in q: kap0 = kap0 + lik.glow(e) * float(q["kgl"])
            v = np.zeros(9)
            for r in range(4):
                f = cal[f"{side}|{r}_o_{tq}"]
                kap = kap0 + kreg[r]
                v += wr[r] * f[np.clip(np.floor((kap + 15) / 0.1).astype(int), 0, 300)]
            out[f"{side}_c{c}"] = [float(x) for x in v]
    return out


def one(args):
    b, theta, R, gref = args
    from final_ci import spec_of, resample
    import headline
    spec = spec_of(MAIN)
    m0 = pd.read_parquet("data/called_2026_w.parquet"); by = {g: d for g, d in m0.groupby("game_pk")}; games = np.array(sorted(by))
    mb = resample(by, games, np.random.default_rng(1000 + b))     # = headline.one / final_ci.one replicate b
    if theta is None:                                              # refit on the resample here
        import lik
        theta = lik.fit(lik.prepare(mb), spec, th0=np.load(f"data/th_{MAIN}.npy"), maxiter=60)[0].x
    variant = dict(theta="override", theta_override=np.asarray(theta), spec=spec, frame=mb, re=False)
    m = frame(variant)
    D, _ = solve(m, variant, R, seed=b)
    r = summarise(D, m, gref)
    return dict(b=b, v=SQRT_VERSION, gaps=r["gaps_pp"], thresholds=r["thresholds"], lead_diff=r["late8_c1_lead-3_minus_lead+3_pp"],
                curves=r["curves"],
                sqrt={s: {k: v for k, v in r["sqrt_n"][s].items() if not k.startswith("rows")} for s in ("home", "away")})


def ci(x):
    x = np.asarray(x, float)
    """[bootstrap mean, 2.5 %, 97.5 %] (6 significant digits)"""
    return [float(f"{v:.6g}") for v in (np.mean(x), *np.percentile(x, [2.5, 97.5]))]


def figures(res, boot):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    g = res["g_ref"]; cv = res["curves"]; rv = res.get("revealed", {})
    inn = np.arange(1, 10)
    # manuscript style (serif 9 pt like fig_policies, no title inside the plot; the caption carries it), PDF
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.4), sharey=True)
    for j, (side, lab) in enumerate((("def", "fielding (catcher)"), ("off", "batting (batter)"))):
        for c, col in ((2, "#2a6fdb"), (1, "#d1495b")):
            key = f"{side}_c{c}_tied"
            ax[j].plot(inn, cv[key], "-o", color=col, label=f"benchmark, {c} left (tied)")
            if boot:
                B = np.array([r["curves"][key] for r in boot])
                lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
                ax[j].fill_between(inn, lo, hi, color=col, alpha=.15, lw=0)
            ax[j].plot(inn, cv[f"{side}_c{c}_occ"], ":", color=col, lw=1.2, label=f"benchmark, {c} left (score-weighted)")
            if rv:
                ax[j].plot(inn, rv[f"{side}_c{c}"], "--", color=col, alpha=.7, label=f"teams (revealed), {c} left")
        ax[j].set_title(lab.capitalize(), fontsize=9); ax[j].set_xlabel("Inning (start of half-inning)"); ax[j].set_xticks(inn)
        for s_ in ("top", "right"):
            ax[j].spines[s_].set_visible(False)
    ax[0].set_ylabel(f"Confidence needed to challenge\n(stake g = {g:.4f} WP)")
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig("figures/fig1_threshold_score.pdf"); fig.savefig("figures/fig1_threshold_score.png", dpi=150)
    plt.close(fig); plt.rcParams.update(plt.rcParamsDefault)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (rk, fk, ttl) in zip(axs, (("rows_occ", "occ_model", "score-weighted (real 2026 score occupancy)"),
                                       ("rows_tied", "tied_model", "tied score at the start of the half-inning"))):
        for side, col in (("home", "#2a6fdb"), ("away", "#d1495b")):
            rows = np.array(res["sqrt_n"][side][rk]); n, d1 = rows[:, 1], rows[:, -2]
            f = res["sqrt_n"][side][fk]["sqrt"]
            ax.plot(np.sqrt(n), d1, "o", color=col, ms=4, label=side)
            xs = np.linspace(0, np.sqrt(n.max()), 50)
            ax.plot(xs, f["a"] + f["b"] * xs, "-", color=col, lw=1,
                    label=f"{side}: {f['a']:.4f} + {f['b']:.4f}" + r"$\sqrt{n}$" + f", R$^2$ = {f['R2']:.3f}")
        ax.set_xlabel(r"$\sqrt{n}$, n = expected opportunities left (regulation)"); ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=7)
    axs[0].set_ylabel(r"$\Delta_1$ (WP), first opportunity of the half-inning")
    fig.suptitle("Option value of one retained challenge, score-aware DP", fontsize=10)
    fig.tight_layout(); fig.savefig("figures/fig_delta_sqrt.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    Rb = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    import headline, lik
    from final_ci import spec_of
    m0 = pd.read_parquet("data/called_2026_w.parquet"); gref = float(m0[m0.ch].g.median())
    t = time.time()
    variant = headline.spec_variant(MAIN)
    m = frame(variant)
    D, cal = solve(m, variant, R, seed=1)
    Pm = lik.unpack(np.load(f"data/th_{MAIN}.npy"), spec_of(MAIN))
    res = summarise(D, m, gref, cal, Pm)
    print("point", round(time.time() - t, 1), {k: round(v, 1) for k, v in res["gaps_pp"].items()}, flush=True)
    # refit bootstrap on the stored refits
    # the engine-fixed refit bootstrap (results/headline_boot.json) is no longer run; without it every
    # replicate refits the main model on its resample itself (same resample rng 1000 + b), B = B_OWN
    hb = json.load(open("results/headline_boot.json")) if os.path.exists("results/headline_boot.json") else None
    assert hb is None or hb["variant"] == MAIN
    path = "data/thresholds_score_boot.json"
    rows = json.load(open(path))["rows"] if os.path.exists(path) else []
    if rows and json.load(open(path)).get("R") != Rb:
        rows = []
    done = {r["b"] for r in rows}
    jobs = ([(r["b"], r["theta"], Rb, gref) for r in hb["rows"] if r["ok"] and r["b"] not in done] if hb else
            [(b, None, Rb, gref) for b in range(B_OWN) if b not in done])
    if jobs:
        with mp.get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(one, jobs):
                rows.append(r); json.dump(dict(R=Rb, rows=rows), open(path, "w"))
                if len(rows) % 10 == 0:
                    print(len(rows), round(time.time() - t), flush=True)
    rows = sorted(rows, key=lambda r: r["b"])
    boot = dict(B=len(rows), R=Rb, method="game bootstrap with model refits (refitted inside each replicate, "
                "resample rng 1000+b), DP re-solved on each resample with R draws; "
                "g_ref fixed at the full-sample value; entries are [bootstrap mean, 2.5 %, 97.5 % percentile]")
    boot["gaps_pp"] = {k: ci([r["gaps"][k] for r in rows]) for k in rows[0]["gaps"]}
    boot["thresholds"] = {k: ci([r["thresholds"][k] for r in rows]) for k in rows[0]["thresholds"]}
    boot["late8_c1_lead-3_minus_lead+3_pp"] = ci([r["lead_diff"] for r in rows])
    boot["sqrt_n"] = {}
    # the gaps, thresholds and curves of a cached row do not depend on remaining; its sqrt fits do, so the
    # sqrt intervals use only rows computed with the unrounded n (cached pre-rows have no "v" tag)
    rows_all, rows = rows, [r for r in rows if r.get("v") == SQRT_VERSION]
    boot["sqrt_n_B"] = len(rows)
    for s in ("home", "away") if rows else ():
        o = {}
        for k in ("tied_model", "tied_pooled", "occ_model"):
            for f in ("sqrt", "linear", "sq_on_n"):
                for p in ("a", "b", "R2"):
                    o[f"{k}.{f}.{p}"] = ci([r["sqrt"][s][k][f][p] for r in rows])
        for k in ("D2_over_D1_first_pitch", "D1_first", "D2_first"):
            o[k] = ci([r["sqrt"][s][k] for r in rows])
        boot["sqrt_n"][s] = o
    if not rows:
        boot["sqrt_n_note"] = ("no bootstrap row computed with the unrounded n yet (cached rows predate the unrounded count); "
                               "delete data/thresholds_score_boot.json to recompute them")
    rows = rows_all
    res["bootstrap"] = boot
    res["spec"] = dict(variant=MAIN, dp="dp_score.solve_dp_score (score-aware, DM = 6 clip, random opportunity counts, exact extra-inning block)",
                       calibration="sim.draw isotonic", R_point=R, seed_point=1,
                       g_ref="median g of challenged pitches, data/called_2026_w.parquet")
    res["old_pooled_dp"] = dict(source="results/res_main.json, data/bar2_boot.json, results/check.json F, theory.md T2i",
                                early_vs_late_c1_pp=[22.2, 20.7, 23.5], early_vs_late_c2_pp=27.9, c1_vs_c2_mid_pp=18.5,
                                sqrt_fit="D1 = -0.003 + 0.0031 sqrt(n), R2 0.998 (linear 0.95)",
                                D1sq_on_n_R2=dict(home=0.996, away=0.999), D2_over_D1_first=0.52)
    res["notes"] = ["thresholds use Delta at k = 0 (first opportunity of the half-inning), home and away DPs averaged",
                    "occ = weighted by the share of real 2026 half-innings starting at each score differential in that slot; "
                    "the top of the 1st always starts tied",
                    "lead L is from the team's perspective (home d = L, away d = -L); the 8th inning is the top half "
                    "(home team fielding, away batting)",
                    "score differential clipped to +-6 (dp_score.DM); extra innings are an exact stationary block with "
                    "the top-up rule; n excludes extra-inning opportunities",
                    "Delta_1 at the last regulation opportunity can be ~0: with the extra-inning top-up c = 0 and c = 1 "
                    "are worth the same in extras"]
    json.dump(res, open("results/thresholds_score.json", "w"), indent=1)
    figures(res, rows)
    print(json.dumps(dict(gaps=res["gaps_pp"], boot=boot["gaps_pp"]), indent=1))
    print(round(time.time() - t), "s")
