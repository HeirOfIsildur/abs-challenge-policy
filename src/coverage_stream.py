"""coverage of the headline interval for a POPULATION target, with resampled game
streams and under misspecification.

Target. The paper's estimand is the league-average gap per team-season between the pooled-model benchmark under the
primary intervention (change only the decision rule; keep the revealed information and the lapses) and the teams'
behaviour, for a season whose game stream is drawn from the population the 2026 games represent:
  gap* = E_stream[ V_bench(stream) ] - E_stream,behaviour[ realised value of the behaviour ],
with values at the TRUE stakes (the full-season 2026 engine) and the generating model's true parameters.
Generating models ("scenarios"): the main model (correctly specified), the main model + lapse, and player effects +
lapse (11 x 11 nodes; every player at his EB posterior mean). The ANALYSIS is always the paper's headline procedure
with the main model: cross-fitted stakes (crossfit.py) inside the season's games, refit of the main model, the
benchmark minus the observed (simulated) challenges valued at the in-fold stakes; interval from a joint bootstrap
over the season's games that resamples the Statcast games too, cross-fits and refits in every replicate
(as joint_boot.job). Reduced budget: B bootstrap replicates and R draws per evaluation.
Season s of a scenario: games resampled with replacement (a new stream), then the behaviour simulated along it:
truth W = 1{w > 0} flipped with pi, signal s = w + sigma_r e, challenge iff s > kappa(x, held) and no lapse, with the
challenges the simulated team holds (start 2, -1 per failure, extra-inning top-up).
usage: python src/coverage_stream.py run <scenarios> <S> <B> [R] [workers]  -> data/coverage_stream_rows.json
       python src/coverage_stream.py summary                                -> results/coverage_stream.json"""
import json, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
from numba import njit
sys.path.insert(0, "src")

ROWS = "data/coverage_stream_rows.json"
OUT = "results/coverage_stream.json"
ANALYSIS = "kappa_sr_region_g"
N_TRUTH, R_TRUTH, N_BEHAV = 16, 4, 20       # truth: streams, draws per stream; behaviour draws per stream


@njit(cache=True)
def _play(gidx, home, new_extra, elig, s, kap1, kap2, W, lapsed):
    """the behavioural rule s > kappa(held) (no challenge on a lapsed pitch) along one season."""
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
        if c[t] == 0 or not elig[i] or lapsed[i]:
            continue
        if s[i] > (kap1[i] if c[t] == 1 else kap2[i]):
            ch[i] = True; ov[i] = W[i]
            if not W[i]:
                c[t] -= 1
    return left, ch, ov


def gen_variant(scen):
    from headline import spec_variant
    v = spec_variant(scen)
    if v["spec"].get("re"):
        v["re"] = True; v["re_tag"] = scen           # true players: EB posterior means of the full-sample fit
    return v


def resample_with_statcast(f, sc, pick):
    """the called-pitch frame and the Statcast frame of the games pick (labels 0..n-1; orig_game = 2026 game)."""
    from joint_boot import resample_frame
    orig = f.drop_duplicates("game_pk").set_index("game_pk")["orig_game"] if "orig_game" in f else None
    byf = {g: d for g, d in f.groupby("game_pk")}; bys = {g: d for g, d in sc.groupby("game_pk")}
    fb = resample_frame(byf, pick)
    if orig is not None:                          # keep the 2026 game behind a resampled resample
        fb["orig_game"] = orig.reindex(pick).values[fb.game_pk.values]
    scb = pd.concat([bys[g].assign(game_pk=i) for i, g in enumerate(pick) if g in bys], ignore_index=True)
    return fb.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True), scb


def stream(seed):
    """a resampled game stream: called-pitch frame (orig_game kept) and the Statcast frame of the same games."""
    import wp
    m = pd.read_parquet("data/called_2026_w.parquet")
    sc = pd.read_parquet("data/statcast_2026.parquet", columns=sorted(set(wp.SC_COLS) | {"home_score", "away_score", "post_bat_score"}))
    games = np.unique(m.game_pk.values)
    return resample_with_statcast(m, sc, np.random.default_rng(seed).choice(games, len(games)))


def behave(f, scen, seed):
    """f with left / ch / ch_over replaced by the generating model's behaviour; returns (frame, realised value at the
    true stakes per team-season)."""
    import sim
    v = gen_variant(scen)
    a = {c: sim.apply_lik(f.assign(left=c), dict(v)) for c in (1, 2)}
    rng = np.random.default_rng(seed)
    w = a[2].w.values; sr = a[2].sr.values; pi = float(a[2].pi.values[0]); lap = a[2].lapse.values
    W = (w > 0) ^ (rng.random(len(f)) < pi)
    s = w + sr * rng.standard_normal(len(f))
    lapsed = rng.random(len(f)) < lap
    home = np.where(f.side == "off", ~f.top, f.top).astype(bool)
    left, ch, ov = _play(f.game_pk.astype("category").cat.codes.values.astype(np.int64), home, sim.extra_starts(f),
                         sim.opportunity_mask(f), s, a[1].kappa.values, a[2].kappa.values, W, lapsed)
    g = f.copy(); g["left"] = left; g["ch"] = ch; g["ch_over"] = ov
    return g, float((f.g.values * ov).sum() / 30)


def truth_job(scen, k):
    """one stream of the population truth: V_bench at the true parameters and true stakes, and the mean realised value
    of the behaviour over N_BEHAV behaviour draws."""
    from headline import evaluate_all, summarise
    t = time.time()
    f, _ = stream(800000 + k)
    _, _, out, _ = evaluate_all(dict(gen_variant(scen), frame=f, lite=True), R=R_TRUTH, seed=k)
    sm = summarise(out, f.game_pk.nunique())
    vals = [behave(f, scen, 900000 + 1000 * k + j)[1] for j in range(N_BEHAV)]
    return dict(kind="truth", scen=scen, k=k, bench=sm["bench"]["wins"], behaviour=float(np.mean(vals)),
                behaviour_sd=float(np.std(vals, ddof=1)), secs=round(time.time() - t, 1))


def analyse(f, scb, R, seed):
    """the headline procedure on one (resampled) season: cross-fit, refit the main model, bench - observed."""
    import lik, wp, crossfit as CF
    from headline import evaluate_all, summarise, spec_variant
    from joint_boot import run_laws
    run_laws(scb)
    mb = CF.crossfit_frame(f, scb[wp.SC_COLS], np.random.default_rng([seed, 1]))
    v = spec_variant(ANALYSIS)
    r, _ = lik.fit(lik.prepare(mb), v["spec"], th0=np.load(v["theta"]), maxiter=60)
    mm, _, out, _ = evaluate_all(dict(v, theta_override=r.x, frame=mb, lite=True), R=R, seed=seed)
    sm = summarise(out, int(mm.gidx.max() + 1))
    return sm["bench"]["wins"] - sm["observed"]["wins"], bool(r.success)


def season_job(scen, s, B, R):
    """season s of a scenario: the point estimate (b = -1) and B joint-bootstrap replicates (the season's games and
    their Statcast resampled again), all in one job so that the season is simulated once."""
    t0 = time.time()
    f0, scb0 = stream(100 + s)
    f0, _ = behave(f0, scen, 200 + s)
    rows = []
    games = np.unique(f0.game_pk.values)
    for b in range(-1, B):
        t = time.time()
        if b >= 0:
            f, scb = resample_with_statcast(f0, scb0, np.random.default_rng([300 + s, b]).choice(games, len(games)))
        else:
            f, scb = f0, scb0
        lost, ok = analyse(f, scb, R, 1 + max(b, 0) + 1000 * s)
        rows.append(dict(kind="season", scen=scen, s=s, b=b, lost=lost, ok=ok, secs=round(time.time() - t, 1)))
    print("season", scen, s, round(time.time() - t0), flush=True)
    return rows


def job(a):
    return [truth_job(a[1], a[2])] if a[0] == "truth" else season_job(*a[1:])


def clopper(k, n):
    from scipy.stats import beta
    return [float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0, float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


def summary(rows, plan=None):
    out = {}
    for scen in sorted({r["scen"] for r in rows}):
        tr = [r for r in rows if r["kind"] == "truth" and r["scen"] == scen]
        if not tr:
            continue
        bench = np.array([r["bench"] for r in tr]); beh = np.array([r["behaviour"] for r in tr])
        gap = float(bench.mean() - beh.mean())
        gap_se = float(np.std(bench - beh, ddof=1) / np.sqrt(len(tr))) if len(tr) > 1 else None
        per = []
        for s in sorted({r["s"] for r in rows if r["kind"] == "season" and r["scen"] == scen}):
            sel = [r for r in rows if r["kind"] == "season" and r["scen"] == scen and r["s"] == s]
            pt = [r for r in sel if r["b"] < 0]
            Bmax = (plan or {}).get(scen, {}).get("B", 10 ** 9)
            bs = np.array([r["lost"] for r in sel if 0 <= r["b"] < Bmax and r["ok"]])
            if not pt or len(bs) < min(10, Bmax):
                continue
            p = pt[0]["lost"]; lo, hi = np.percentile(bs, [2.5, 97.5]); sd = float(bs.std(ddof=1))
            ints = dict(percentile=(lo, hi), basic=(2 * p - hi, 2 * p - lo), normal=(p - 1.96 * sd, p + 1.96 * sd))
            per.append(dict(s=s, point=p, B=len(bs), boot_sd=sd, **{k: [float(a), float(b)] for k, (a, b) in ints.items()},
                            **{f"cover_{k}": bool(a <= gap <= b) for k, (a, b) in ints.items()}))
        n = len(per); pts = np.array([x["point"] for x in per]) if n else np.zeros(0)
        cov = {}
        for k in ("percentile", "basic", "normal"):
            c = sum(x[f"cover_{k}"] for x in per)
            cov[k] = dict(covered=c, rate=c / n if n else None, ci=clopper(c, n) if n else None,
                          miss_low=sum(x[k][1] < gap for x in per), miss_high=sum(x[k][0] > gap for x in per),
                          mean_width=float(np.mean([x[k][1] - x[k][0] for x in per])) if n else None)
        out[scen] = dict(truth=dict(gap=gap, gap_se=gap_se, bench=float(bench.mean()), behaviour=float(beh.mean()),
                                    streams=len(tr), R=R_TRUTH, behaviour_draws=N_BEHAV),
                         S=n, coverage=cov, point_mean=float(pts.mean()) if n else None,
                         point_bias=float(pts.mean() - gap) if n else None,
                         point_sd_across_seasons=float(pts.std(ddof=1)) if n > 1 else None,
                         boot_sd_mean=float(np.mean([x["boot_sd"] for x in per])) if n else None, seasons=per)
    return dict(analysis=ANALYSIS, scenarios=out)


if __name__ == "__main__":
    mode = sys.argv[1]
    st = json.load(open(ROWS)) if os.path.exists(ROWS) else dict(rows=[])
    if mode == "run":
        # scenarios as name[:S:B] (the misspecification scenarios can run with fewer seasons and replicates
        # than the main one); S, B: defaults
        S0, B0 = int(sys.argv[3]), int(sys.argv[4])
        R = int(sys.argv[5]) if len(sys.argv) > 5 else 2; workers = int(sys.argv[6]) if len(sys.argv) > 6 else 6
        plan = {}
        for tok in sys.argv[2].split(","):
            parts = tok.split(":")
            plan[parts[0]] = dict(S=int(parts[1]) if len(parts) > 1 else S0, B=int(parts[2]) if len(parts) > 2 else B0)
        import crossfit
        cfg = dict(R=R, analysis=ANALYSIS, n_truth=N_TRUTH, r_truth=R_TRUTH, plan=plan, folds=crossfit.FOLDS)
        old = st.get("config")
        if old and (old.get("R") != R or old.get("analysis") != ANALYSIS or old.get("folds") != crossfit.FOLDS):
            raise SystemExit(f"{ROWS}: stored configuration {old} differs from {cfg}")
        st["config"] = cfg
        done = {(r["kind"], r["scen"], r.get("k", r.get("s"))) for r in st["rows"]}
        jobs = [("truth", sc, k) for sc in plan for k in range(N_TRUTH) if ("truth", sc, k) not in done]
        Smax = max(p["S"] for p in plan.values())
        jobs += [("season", sc, s, plan[sc]["B"], R) for s in range(Smax) for sc in plan
                 if s < plan[sc]["S"] and ("season", sc, s) not in done]
        with mp.get_context("spawn").Pool(workers) as pool:
            for rs in pool.imap_unordered(job, jobs):
                st["rows"] += rs
                json.dump(st, open(ROWS, "w"))
                r = rs[0]; print(len(st["rows"]), r["kind"], r["scen"], r.get("s", r.get("k")), flush=True)
        json.dump(st, open(ROWS, "w"))
    res = summary(st["rows"], (st.get("config") or {}).get("plan")); res["config"] = st.get("config")
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "seasons"} for k, v in res["scenarios"].items()}, indent=1))
