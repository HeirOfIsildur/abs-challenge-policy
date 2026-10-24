"""the mechanism with uncertainty.

Point (cross-fitted split 5000, main model refitted on its decision stakes) and B joint-bootstrap replicates (games
and their Statcast resampled, stakes cross-fitted inside the resample, main model refitted -- as joint_boot.job) of
  slopes  threshold slope on log g, teams' implied p* vs the benchmark's p* = D/(g + D) (robustness_checks.slopes), with the
          criterion's controls ("full"), with the twelve exact count cells ("count_fe"), and without controls;
  decomp  the gap closed by re-tuning only the intercept, only the stake terms (linear or scaled), or both, of the
          teams' own rule s > kappa(x), tuned on half the draws and scored on the other half (robustness_checks.decomp).
The frequency-matched one-line rule comes from the joint bootstrap (joint_boot.freq_matched).
usage: python src/mechanism_boot.py point [R_decomp] [R_slopes]
       python src/mechanism_boot.py boot <B> [R_decomp] [R_slopes] [workers]  -> results/mechanism_boot.json"""
import json, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, "src")

ROWS = "data/mechanism_boot_rows.json"
OUT = "results/mechanism_boot.json"
DECOMP_ROWS = ["intercept_only", "stake_linear_only", "stake_scale_only", "both_linear", "both_scale"]


def extract(sl, dc):
    out = dict(slopes={}, decomp={})
    for side, S in sl["sides"].items():
        for rg in ("all", "high"):
            for ctl in ("full", "count_fe", "none"):
                out["slopes"][f"{side}_{rg}_{ctl}_teams"] = S[rg][f"teams_logit_{ctl}"]
                out["slopes"][f"{side}_{rg}_{ctl}_opt"] = S[rg][f"opt_logit_{ctl}"]
    for k in DECOMP_ROWS:
        r = dc["rows"][k]["out_of_sample"]
        out["decomp"][k] = dict(gap_closed=r["gap_closed"], n=r["n"])
    out["decomp"]["observed_n"] = dc["observed"]["n"]
    return out


def frame(b):
    """b < 0: the point frame (2026 games, cross-fit split 5000); b >= 0: a joint-bootstrap resample."""
    import wp, lik, crossfit as CF
    from joint_boot import resample_frame, run_laws
    from headline import spec_variant, MAIN
    m = pd.read_parquet("data/called_2026_w.parquet")
    if b < 0:
        mb = CF.crossfit_frame(m, CF.statcast(), np.random.default_rng(5000))
    else:
        by = {g: d for g, d in m.groupby("game_pk")}; games = np.array(sorted(by))
        pick = np.random.default_rng(6000 + b).choice(games, len(games))
        mb = resample_frame(by, pick)
        sc = pd.read_parquet("data/statcast_2026.parquet", columns=sorted(set(wp.SC_COLS) | {"home_score", "away_score", "post_bat_score"}))
        bys = {g: d for g, d in sc.groupby("game_pk")}
        scb = pd.concat([bys[g].assign(game_pk=i) for i, g in enumerate(pick) if g in bys], ignore_index=True)
        run_laws(scb)
        mb = CF.crossfit_frame(mb, scb[wp.SC_COLS], np.random.default_rng([6000 + b, 1]))
    v = spec_variant(MAIN)
    r, _ = lik.fit(lik.prepare(mb), v["spec"], th0=np.load(v["theta"]), maxiter=60)
    return mb, r, dict(v, theta="override", theta_override=r.x, frame=mb)


def job(args):
    b, Rd, Rs = args
    import robustness_checks as RF
    t = time.time()
    mb, r, var = frame(b)
    ctx = RF.context(frame=mb, variant=var)
    sl = RF.slopes(Rs, ctx=ctx, th=r.x, store=False, seed=1 + max(b, 0))
    dc = RF.decomp(Rd, ctx=ctx, th=r.x, store=False, seed=1 + max(b, 0))
    return dict(b=b, ok=bool(r.success), secs=round(time.time() - t, 1), **extract(sl, dc),
                full=dict(slopes=sl, decomp=dc) if b < 0 else None)


def summarise(st):
    pt = [r for r in st["rows"] if r["b"] < 0]; bs = [r for r in st["rows"] if r["b"] >= 0 and r["ok"]]
    pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) > 1 else None
    out = dict(B=len(bs), config=st.get("config"), point=pt[0] if pt else None, slopes={}, decomp={})
    keys = (pt[0] if pt else bs[0])["slopes"].keys()
    for k in keys:
        x = [r["slopes"][k] for r in bs]
        out["slopes"][k] = dict(point=pt[0]["slopes"][k] if pt else None, ci=pct(x), sd=float(np.std(x, ddof=1)) if len(x) > 1 else None)
    for side in ("off", "def"):
        x = [r["slopes"][f"{side}_all_full_opt"] - r["slopes"][f"{side}_all_full_teams"] for r in bs]
        out["slopes"][f"{side}_all_full_opt_minus_teams"] = dict(ci=pct(x))
    for k in DECOMP_ROWS:
        x = [r["decomp"][k]["gap_closed"] for r in bs]
        out["decomp"][k] = dict(point=pt[0]["decomp"][k]["gap_closed"] if pt else None, ci=pct(x),
                                n=pt[0]["decomp"][k]["n"] if pt else None)
    x = [r["decomp"]["stake_linear_only"]["gap_closed"] - r["decomp"]["intercept_only"]["gap_closed"] for r in bs]
    out["decomp"]["stake_minus_intercept"] = dict(ci=pct(x), share_positive=float(np.mean(np.array(x) > 0)) if x else None)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    st = json.load(open(ROWS)) if os.path.exists(ROWS) else dict(rows=[])
    if mode == "point":
        Rd = int(sys.argv[2]) if len(sys.argv) > 2 else 20; Rs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        st["rows"] = [r for r in st["rows"] if r["b"] >= 0] + [job((-1, Rd, Rs))]
        st.setdefault("config", {})["point"] = dict(R_decomp=Rd, R_slopes=Rs)
    elif mode == "boot":
        B = int(sys.argv[2]); Rd = int(sys.argv[3]) if len(sys.argv) > 3 else 4; Rs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
        workers = int(sys.argv[5]) if len(sys.argv) > 5 else 5
        import crossfit
        if st.get("config", {}).get("boot", {}).get("folds") != crossfit.FOLDS and any(r["b"] >= 0 for r in st["rows"]):
            raise SystemExit(f"{ROWS}: bootstrap rows from another fold assignment; drop them to start over")
        st.setdefault("config", {})["boot"] = dict(R_decomp=Rd, R_slopes=Rs, folds=crossfit.FOLDS)
        done = {r["b"] for r in st["rows"]}
        with mp.get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(job, [(b, Rd, Rs) for b in range(B) if b not in done]):
                st["rows"].append(r); json.dump(st, open(ROWS, "w"))
                print("mech", len(st["rows"]), r["b"], r["secs"], flush=True)
    json.dump(st, open(ROWS, "w"))
    json.dump(summarise(st), open(OUT, "w"), indent=1, default=float)
