"""coverage of the headline procedure (full policy cross-fit, policy_crossfit.py, with its refit bootstrap).

coverage_stream.py simulated each season by resampling the 2026 games with replacement, so a simulated season repeated
about a third of its games; with the folds kept over the original games its decision engines saw fewer distinct games
than the real season's, which biased the season point down ("the coverage artefact"). Here every simulated
season is the REAL 2026 game stream -- no game appears twice -- and only the behaviour is simulated along it from a
generating model (coverage_stream.behave: truth from the zone rule and the mislabel rate, signal and criterion from
the generating model's parameters, the challenges the simulated team holds, lapses). The target is the generating
model's gap on that stream under the primary intervention,
  gap* = V_bench(true parameters, full-season stakes) - E_behaviour[realised value of the behaviour],
so it is conditional on the 2026 stream; the interval, which resamples games, also carries stream variability, so
coverage of the conditional target is if anything conservative. The analysis is the paper's procedure for the
analysis model: the full policy cross-fit on the season (point) and B bootstrap replicates (games resampled with their
Statcast games, everything refitted inside, folds over the original games).
usage: python src/coverage_pcf.py run <scen> <analysis> <S | s0:S> <B> [R] [workers] [shard/nshard]
         -> data/covpcf_rows_<scen>_<analysis>[_<shard>].json
       python src/coverage_pcf.py truth <scen> [R] [behaviour draws]  -> data/covpcf_truth_<scen>.json
       python src/coverage_pcf.py truthK <scen> <analysis> <n> [R] [i0] -> data/covpcf_truthK_<scen>_<analysis>_<i0>.json
       python src/coverage_pcf.py summary                             -> results/coverage_pcf.json
The summary reports coverage of both targets: gap* above (the oracle: the generating model's benchmark at its true
parameters) and theta_K (truth_k: the benchmark learned on (K-1)/K of a simulated season, valued in the true world),
the quantity the cross-fit estimates; their difference is the cost of learning."""
import glob, json, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, "src")

OUT = "results/coverage_pcf.json"


def season(scen, s):
    """season s: the 2026 stream with behaviour simulated from scenario scen (seed 500 + s)."""
    from coverage_stream import behave
    m = pd.read_parquet("data/called_2026_w.parquet")
    f, _ = behave(m, scen, 500 + s)
    return f


def job(a):
    scen, ana, s, b, R = a
    import policy_crossfit as PC
    from joint_boot import resample_frame
    t = time.time()
    f = season(scen, s); sc = PC.statcast_frame()
    if b >= 0:
        by = {g: d for g, d in f.groupby("game_pk")}; games = np.array(sorted(by))
        pick = np.random.default_rng([600 + s, b]).choice(games, len(games))
        f = resample_frame(by, pick)
        bys = {g: d for g, d in sc.groupby("game_pk")}
        sc = pd.concat([bys[g].assign(game_pk=i) for i, g in enumerate(pick) if g in bys], ignore_index=True)
    r = PC.one_split(f, sc, ana, 10000 + 1000 * s + b, R, [(1.0, 0.0)])
    return dict(scen=scen, analysis=ana, s=s, b=b, lost=r["lost"], ok=r["ok"], secs=round(time.time() - t, 1))


def truth(scen, R=8, NB=20):
    """the target on the 2026 stream: the benchmark at the generating parameters (full-season stakes), averaged over R
    draws, minus the mean realised value of NB behaviour draws."""
    from coverage_stream import gen_variant, behave
    from headline import evaluate_all, summarise
    m = pd.read_parquet("data/called_2026_w.parquet")
    _, _, out, dr = evaluate_all(dict(gen_variant(scen), frame=m, lite=True), R=R, seed=4242)
    sm = summarise(out, m.game_pk.nunique())
    vals = [behave(m, scen, 700000 + j)[1] for j in range(NB)]
    res = dict(scen=scen, bench=sm["bench"]["wins"], bench_mc_se=float(np.std([d["totals"]["bench"] for d in dr], ddof=1) / np.sqrt(R)),
               behaviour=float(np.mean(vals)), behaviour_sd=float(np.std(vals, ddof=1)), R=R, NB=NB)
    res["gap"] = res["bench"] - res["behaviour"]
    json.dump(res, open(f"data/covpcf_truth_{scen}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def truth_k(scen, ana, n, R=8, i0=0):
    """the target of the cross-fit, theta_K, on the 2026 stream: n seasons (behaviour from scen, seeds 900 + i), on each
    the benchmark learned on the training folds exactly as in the analysis (model ana) and valued in the TRUE world of
    scen (its truths, signals and lapses on the held-out games, flips at the full-season stakes), minus the mean realised
    value of the behaviour (data/covpcf_truth_<scen>.json). One split per season."""
    import policy_crossfit as PC
    from coverage_stream import gen_variant
    path = f"data/covpcf_truthK_{scen}_{ana}_{i0}.json"
    st = json.load(open(path)) if os.path.exists(path) else dict(rows=[])
    sc = PC.statcast_frame(); done = {r["i"] for r in st["rows"]}
    for i in range(i0, i0 + n):
        if i in done:
            continue
        f, _ = __import__("coverage_stream").behave(pd.read_parquet("data/called_2026_w.parquet"), scen, 900 + i)
        r = PC.one_split(f, sc, ana, 20000 + i, R, [(1.0, 0.0)], world=gen_variant(scen))
        st["rows"].append(dict(i=i, bench=r["policies"]["1.0|0.0|bench"]["wins"], ok=r["ok"]))
        json.dump(st, open(path, "w"))
        print("truthK", scen, ana, i, round(st["rows"][-1]["bench"], 4), flush=True)


def clopper(k, n):
    from scipy.stats import beta
    return [float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0, float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


def summary():
    rows = {}
    for p in glob.glob("data/covpcf_rows_*.json"):
        for r in json.load(open(p))["rows"]:
            rows[(r["scen"], r["analysis"], r["s"], r["b"])] = r
    out = {}
    for scen, ana in sorted({(k[0], k[1]) for k in rows}):
        tp = f"data/covpcf_truth_{scen}.json"
        if not os.path.exists(tp):
            continue
        tj = json.load(open(tp)); gap = tj["gap"]
        bk = [r["bench"] for p in glob.glob(f"data/covpcf_truthK_{scen}_{ana}_*.json") for r in json.load(open(p))["rows"]]
        gapK = None
        if bk:
            gapK = float(np.mean(bk) - tj["behaviour"]); gapK_se = float(np.std(bk, ddof=1) / np.sqrt(len(bk))) if len(bk) > 1 else None
        per = []
        for s in sorted({k[2] for k in rows if k[:2] == (scen, ana)}):
            pt = rows.get((scen, ana, s, -1))
            bs = np.array([r["lost"] for k, r in rows.items() if k[:3] == (scen, ana, s) and k[3] >= 0])
            if pt is None or len(bs) < 10:
                continue
            p = pt["lost"]; lo, hi = np.percentile(bs, [2.5, 97.5]); sd = float(bs.std(ddof=1))
            ints = dict(percentile=(lo, hi), basic=(2 * p - hi, 2 * p - lo), normal=(p - 1.96 * sd, p + 1.96 * sd))
            per.append(dict(s=s, point=p, B=len(bs), boot_sd=sd, boot_mean=float(bs.mean()),
                            **{k: [float(a), float(b)] for k, (a, b) in ints.items()},
                            **{f"cover_{k}": bool(a <= gap <= b) for k, (a, b) in ints.items()},
                            **({f"coverK_{k}": bool(a <= gapK <= b) for k, (a, b) in ints.items()} if gapK is not None else {})))
        n = len(per)
        if not n:
            continue
        pts = np.array([x["point"] for x in per])
        cov = {}
        for k in ("percentile", "basic", "normal"):
            c = sum(x[f"cover_{k}"] for x in per)
            cov[k] = dict(covered=c, rate=c / n, ci=clopper(c, n), miss_low=sum(x[k][1] < gap for x in per),
                          miss_high=sum(x[k][0] > gap for x in per), mean_width=float(np.mean([x[k][1] - x[k][0] for x in per])))
        covK = {}
        if gapK is not None:
            for k in ("percentile", "basic", "normal"):
                c = sum(x[f"coverK_{k}"] for x in per)
                covK[k] = dict(covered=c, rate=c / n, ci=clopper(c, n), miss_low=sum(x[k][1] < gapK for x in per),
                               miss_high=sum(x[k][0] > gapK for x in per))
        out[f"{scen}|{ana}"] = dict(scen=scen, analysis=ana, gap=gap, gapK=gapK, gapK_se=gapK_se if gapK is not None else None,
                                    coverageK=covK, point_bias_K=float(pts.mean() - gapK) if gapK is not None else None, S=n, B=int(np.median([x["B"] for x in per])),
                                    coverage=cov, point_mean=float(pts.mean()), point_bias=float(pts.mean() - gap),
                                    point_bias_se=float(pts.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
                                    point_sd_across_seasons=float(pts.std(ddof=1)) if n > 1 else None,
                                    boot_sd_mean=float(np.mean([x["boot_sd"] for x in per])),
                                    boot_mean_minus_point=float(np.mean([x["boot_mean"] - x["point"] for x in per])), seasons=per)
    json.dump(dict(design="2026 stream fixed, behaviour simulated; refit bootstrap of the full policy cross-fit",
                   results=out), open(OUT, "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "seasons"} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "truth":
        truth(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8, int(sys.argv[4]) if len(sys.argv) > 4 else 20)
    elif mode == "run":
        scen, ana, B = sys.argv[2], sys.argv[3], int(sys.argv[5])
        s0, S = (map(int, sys.argv[4].split(":")) if ":" in sys.argv[4] else (0, int(sys.argv[4])))   # seasons s0..S-1
        R = int(sys.argv[6]) if len(sys.argv) > 6 else 4; workers = int(sys.argv[7]) if len(sys.argv) > 7 else 8
        shard, nshard = (map(int, sys.argv[8].split("/")) if len(sys.argv) > 8 else (0, 1))
        path = f"data/covpcf_rows_{scen}_{ana}" + (f"_s{s0}" if s0 else "") + (f"_{shard}" if nshard > 1 else "") + ".json"
        cfg = dict(scen=scen, analysis=ana, R=R)
        st = json.load(open(path)) if os.path.exists(path) else dict(config=cfg, rows=[])
        if st["config"] != cfg:
            raise SystemExit(f"{path}: configuration differs")
        done = {(r["s"], r["b"]) for r in st["rows"]}
        jobs = [(scen, ana, s, b, R) for s in range(s0, S) for b in range(-1, B)]
        jobs = [j for i, j in enumerate(jobs) if i % nshard == shard and (j[2], j[3]) not in done]
        with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(job, jobs):
                st["rows"].append(r); json.dump(st, open(path, "w"))
                print(len(st["rows"]), r["s"], r["b"], r["secs"], round(r["lost"], 3), flush=True)
    elif mode == "truthK":
        truth_k(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 8,
                int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    elif mode == "summary":
        summary()
