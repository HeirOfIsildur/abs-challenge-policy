"""coverage of the refit-bootstrap interval for the wins lost.

Parametric simulation from the fitted main model (kappa_sr_region_g since ) on the real 2026 call stream. In season s every
called pitch keeps its tracked location; the ruling is W = 1{w > 0} flipped with the fitted mislabel rate pi, the signal
is s = w + sr e, and the team challenges iff s > kappa(x) with the challenges it holds in that simulated season (start
2, -1 per failure, extra-inning top-up; `left` in the design follows the simulated count). The simulated behaviour is
the fitted rule, so the true gap is
  gap* = V_opt(theta) - E[realised value of the fitted rule],
with V_opt from headline.evaluate_all at the true parameters (R_TRUE draws) and the expectation over N_TRUE simulated
seasons. In each season the analysis of the paper is repeated with a reduced bootstrap: refit (lik.fit), point
estimate opt_score - observed with R_POINT draws, B game-resample replicates that refit, re-solve and re-simulate with
R draws (headline.one with the season as data). Coverage of the percentile, basic and normal intervals with
Clopper-Pearson 95 % CIs.
usage:
  python src/boot_coverage.py run <S> <B> [R] [workers]   -> data/coverage_rows.json (resumable), results/coverage.json
  python src/boot_coverage.py summary [S]                 -> results/coverage.json from data/coverage_rows.json (first S seasons)"""
import json, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, "src")

MAIN = "kappa_sr_region_g"         # main model
R_POINT, R_TRUE, N_TRUE = 10, 40, 400
ROWS = "data/coverage_rows.json"
OUT = "results/coverage.json"


@njit(cache=True)
def _play(gidx, home, new_extra, elig, s, kap1, kap2, W):
    """the fitted rule s > kappa(left) along one simulated season; returns left, ch, ch_over per row."""
    n = len(s); left = np.zeros(n, np.int64); ch = np.zeros(n, np.bool_); ov = np.zeros(n, np.bool_)
    c = np.array([2, 2]); cur = -1
    for i in range(n):
        if gidx[i] != cur:
            cur = gidx[i]; c[0] = 2; c[1] = 2
        if new_extra[i]:
            for t in range(2):
                if c[t] == 0: c[t] = 1
        t = 1 if home[i] else 0
        left[i] = c[t]
        if c[t] == 0 or not elig[i]:
            continue
        if s[i] > (kap1[i] if c[t] == 1 else kap2[i]):
            ch[i] = True; ov[i] = W[i]
            if not W[i]:
                c[t] -= 1
    return left, ch, ov


COLS = ["game_pk", "at_bat_number", "pitch_number", "inning", "top", "batter", "ch", "ch_over", "left_home", "left_away",
        "sz_top", "sz_bot", "stand", "balls", "strikes", "fielder_2", "orig_strike", "px_mid", "pz_mid", "dist_in", "edge",
        "score_diff_home", "half", "g", "RV", "sm", "w", "p_model", "side", "left", "pos_pitcher", "d_half_start"]   # what lik / sim / policies read


def base_frame():
    m = pd.read_parquet("data/called_2026_w.parquet", columns=COLS)                  # pruned: 7 workers fit in memory
    return m.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def rule_inputs(m, th, spec):
    """row-level w, sr, pi and kappa at left = 1 and left = 2 under theta (sim.apply_lik)."""
    import sim
    v = dict(theta="override", theta_override=th, spec=spec)
    a1 = sim.apply_lik(m.assign(left=1), v); a2 = sim.apply_lik(m.assign(left=2), v)
    home = np.where(m.side == "off", ~m.top, m.top).astype(bool)
    return dict(w=a2.w.values, sr=a2.sr.values, pi=float(a2.pi.values[0]), kap1=a1.kappa.values, kap2=a2.kappa.values,
                home=home, gidx=m.game_pk.astype("category").cat.codes.values.astype(np.int64),
                new_extra=sim.extra_starts(m), elig=sim.opportunity_mask(m))   # the policies' opportunity set


def season(m, ri, seed):
    """one simulated season: m with left, ch, ch_over replaced, and the realised value of the fitted rule."""
    rng = np.random.default_rng(seed)
    W = (ri["w"] > 0) ^ (rng.random(len(m)) < ri["pi"])
    s = ri["w"] + ri["sr"] * rng.standard_normal(len(m))
    left, ch, ov = _play(ri["gidx"], ri["home"], ri["new_extra"], ri["elig"], s, ri["kap1"], ri["kap2"], W)
    f = m.copy(); f["left"] = left; f["ch"] = ch; f["ch_over"] = ov
    return f, float((m.g.values * ov).sum() / 30)


_CACHE = {}


def season_frame(s):
    if s not in _CACHE:
        from final_ci import spec_of
        m = base_frame(); th = np.load(f"data/th_{MAIN}.npy")
        _CACHE.clear(); _CACHE[s] = season(m, rule_inputs(m, th, spec_of(MAIN)), 5000 + s)[0]
    return _CACHE[s]


def job(args):
    """b = -1: point estimate on season s; b >= 0: one refit-bootstrap replicate of season s."""
    s, b, R = args
    import lik
    from final_ci import spec_of, resample
    from headline import evaluate_all, summarise
    spec = spec_of(MAIN); th0 = np.load(f"data/th_{MAIN}.npy"); t = time.time()
    f = season_frame(s)
    if b >= 0:
        by = {g: d for g, d in f.groupby("game_pk")}; games = np.array(sorted(by))
        f = resample(by, games, np.random.default_rng(100000 * (s + 1) + b))
    r, _ = lik.fit(lik.prepare(f), spec, th0=th0, maxiter=60)
    _, _, out, _ = evaluate_all(dict(theta="override", theta_override=r.x, spec=spec, frame=f, re=False),
                                R=R_POINT if b < 0 else R, seed=1 + max(b, 0))
    sm = summarise(out, f.game_pk.nunique()); P = lik.unpack(r.x, spec)
    return dict(s=s, b=b, ok=bool(r.success), secs=round(time.time() - t, 1), lost=sm["opt_score"]["wins"] - sm["observed"]["wins"],
                opt=sm["opt_score"]["wins"], observed=sm["observed"]["wins"], nch=int(f.ch.sum()),
                off_lsr=float(P["off"]["lsr"]), def_lsr=float(P["def"]["lsr"]))


def truth():
    """gap* = V_opt(theta) - E[realised fitted rule]; also the fitted rule's value as headline.evaluate_all scores it."""
    from final_ci import spec_of
    from headline import evaluate_all, summarise, spec_variant
    m = base_frame(); spec = spec_of(MAIN); th = np.load(f"data/th_{MAIN}.npy")
    ri = rule_inputs(m, th, spec)
    sims = [season(m, ri, 900000 + k) for k in range(N_TRUE)]
    val = np.array([v for _, v in sims]); nch = np.array([f.ch.sum() for f, _ in sims])
    mm, _, out, dr = evaluate_all(spec_variant(MAIN), R=R_TRUE, seed=1)
    sm = summarise(out, int(mm.gidx.max() + 1)); optd = np.array([d["totals"]["opt_score"] for d in dr])
    return dict(opt=sm["opt_score"]["wins"], opt_mc_se=float(optd.std(ddof=1) / np.sqrt(R_TRUE)), R=R_TRUE,
                rule_realised=float(val.mean()), rule_realised_sd=float(val.std(ddof=1)), n_seasons=N_TRUE,
                challenges_sim=float(nch.mean()), challenges_2026=int(m.ch.sum()),
                rule_engine=sm["obs_rule"]["wins"], observed_2026=sm["observed"]["wins"],
                gap=float(sm["opt_score"]["wins"] - val.mean()),
                gap_se=float(np.hypot(optd.std(ddof=1) / np.sqrt(R_TRUE), val.std(ddof=1) / np.sqrt(N_TRUE))))


def clopper(k, n):
    from scipy.stats import beta
    return [float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0, float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


def summary(rows, tr, S, B, R):
    q = lambda xs, a: sorted(xs)[min(len(xs) - 1, max(0, int(round(a * (len(xs) - 1)))))]   # as paper/make_numbers.py
    gap = tr["gap"]; per = []
    for s in sorted({r["s"] for r in rows if r["s"] < S}):
        pt = [r for r in rows if r["s"] == s and r["b"] < 0]
        bs = [r["lost"] for r in rows if r["s"] == s and 0 <= r["b"] < B and r["ok"]]
        if not pt or len(bs) < 0.9 * B:
            continue
        p = pt[0]["lost"]; lo, hi = q(bs, 0.025), q(bs, 0.975); sd = float(np.std(bs, ddof=1))
        ints = dict(percentile=(lo, hi), basic=(2 * p - hi, 2 * p - lo), normal=(p - 1.96 * sd, p + 1.96 * sd))
        per.append(dict(s=s, point=p, boot_sd=sd, B=len(bs), **{k: [float(a), float(b)] for k, (a, b) in ints.items()},
                        **{f"cover_{k}": bool(a <= gap <= b) for k, (a, b) in ints.items()}))
    n = len(per); pts = np.array([x["point"] for x in per])
    cov = {}
    for k in ("percentile", "basic", "normal"):
        c = sum(x[f"cover_{k}"] for x in per)
        cov[k] = dict(covered=c, rate=c / n if n else None, ci=clopper(c, n) if n else None,
                      miss_low=sum(x[k][1] < gap for x in per), miss_high=sum(x[k][0] > gap for x in per),
                      mean_width=float(np.mean([x[k][1] - x[k][0] for x in per])) if n else None)
    return dict(variant=MAIN, S=n, B=B, R=R, R_point=R_POINT, truth=tr, coverage=cov,
                point_mean=float(pts.mean()) if n else None, point_bias=float(pts.mean() - gap) if n else None,
                point_sd_across_seasons=float(pts.std(ddof=1)) if n > 1 else None,
                boot_sd_mean=float(np.mean([x["boot_sd"] for x in per])) if n else None,
                refits_failed=sum(not r["ok"] for r in rows if r["b"] < B and r["s"] < S), seasons=per)


if __name__ == "__main__":
    mode = sys.argv[1]
    st = json.load(open(ROWS)) if os.path.exists(ROWS) else dict(rows=[])
    if mode == "summary" and len(sys.argv) > 2:
        st["S"] = int(sys.argv[2])
    if mode == "run":
        S, B = int(sys.argv[2]), int(sys.argv[3]); R = int(sys.argv[4]) if len(sys.argv) > 4 else 4
        workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8
        st.update(S=S, B=B, R=R)
        if "truth" not in st:
            st["truth"] = truth(); json.dump(st, open(ROWS, "w"))
            print("truth", json.dumps(st["truth"]), flush=True)
        done = {(r["s"], r["b"]) for r in st["rows"]}
        jobs = [(s, b, R) for s in range(S) for b in range(-1, B) if (s, b) not in done]   # season-major: cache hits
        with mp.get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(job, jobs, chunksize=4):
                st["rows"].append(r)
                if len(st["rows"]) % 20 == 0:
                    json.dump(st, open(ROWS, "w"))
                    print(len(st["rows"]), r["s"], r["b"], r["secs"], round(r["lost"], 3), flush=True)
        json.dump(st, open(ROWS, "w"))
    json.dump(summary(st["rows"], st["truth"], st["S"], st["B"], st["R"]), open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in json.load(open(OUT)).items() if k != "seasons"}, indent=1))
