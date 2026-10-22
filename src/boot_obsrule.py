"""Re-evaluate the simulated behavioural rule (obs_rule) in every refit-bootstrap replicate after the
per-count kappa fix (sim.kappa_by_count), without redoing the refits and the DPs, which the fix does not touch.

Each replicate b repeats headline.one exactly: the same game resample (final_ci.resample, seed 1000 + b), the stored
refitted theta, and the same draws (rng seed b; sim.draw is the only consumer of the generator in evaluate_all, so
the R draws are identical). Only obs_rule is re-run; every other policy's summary is kept. Checked against replicates
recomputed in full with headline.py boot (exact agreement required).
usage: python src/boot_obsrule.py <old boot json> [workers] [check json]  -> results/headline_boot.json"""
import json, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, "src")

MAIN = "kappa_sr_region_g"        # BIC-best homogeneous model (criterion piecewise-linear in log g)


def one(args):
    row, R = args
    import sim, dp_score, policies as P
    from final_ci import spec_of, resample
    b = row["b"]; spec = spec_of(MAIN); t = time.time()
    m0 = pd.read_parquet("data/called_2026_w.parquet"); by = {g: d for g, d in m0.groupby("game_pk")}; games = np.array(sorted(by))
    mb = resample(by, games, np.random.default_rng(1000 + b))
    v = dict(theta="override", theta_override=np.array(row["theta"]), spec=spec, frame=mb, re=False)
    m, _ = sim.load(v)
    m = dp_score.add_dstart(m)
    m = sim.opportunities(m)
    m["new_extra"] = sim.extra_starts(m)
    kin = sim.kin_index(m).astype(np.int64)
    ng = int(m.gidx.max() + 1)
    rng = np.random.default_rng(b)
    isoff = (m.side == "off").values
    base = (m.gidx.values, m.is_home_team.values, m.new_extra.values, m.slot.values.astype(np.int64), kin,
            m.inning.values.astype(np.int64), m.balls.values.astype(np.int64), m.strikes.values.astype(np.int64),
            m.dbin.values.astype(np.int64))
    z3 = np.zeros((19, 80, 3)); zc = np.zeros((19, 3)); zcard = np.zeros((4, 3, 2, 2)); z4 = np.zeros((19, 13, 80, 3))
    kap = sim.kappa_by_count(m)
    v_ = np.zeros((ng, 2, 2)); n_ = np.zeros((ng, 2, 2))
    for r in range(R):
        W, s, pt = sim.draw(m, rng, v)
        val, n, _, _, _ = P.run_path(*base, W, pt, s, m.g.values, m.RV.values, kap, isoff, 4, z3, z3, zc, zc, zcard,
                                     0.0, 1.0, 0.0, z4, z4, ng)
        v_ += val / R; n_ += n / R
    ngames = len(games)
    obs = dict(wins=float(v_.sum() / 30), batter=float(v_[:, :, 1].sum() / 30), fielder=float(v_[:, :, 0].sum() / 30),
               n=float(n_.sum() / (2 * ngames)), n_batter=float(n_[:, :, 1].sum() / (2 * ngames)),
               n_fielder=float(n_[:, :, 0].sum() / (2 * ngames)))
    out = dict(row); out["summary"] = dict(row["summary"], obs_rule=obs); out["secs_obsrule"] = round(time.time() - t, 1)
    return out


if __name__ == "__main__":
    old = json.load(open(sys.argv[1])); workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    R = old["R"]
    rows = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(one, [(row, R) for row in old["rows"]]):
            rows.append(r)
            if len(rows) % 20 == 0:
                print(len(rows), flush=True)
    rows.sort(key=lambda r: r["b"])
    if len(sys.argv) > 3:                                   # exact agreement with fully recomputed replicates
        full = {r["b"]: r for r in json.load(open(sys.argv[3]))["rows"]}
        diffs = [max(abs(r["summary"][p][k] - full[r["b"]]["summary"][p][k]) for p in r["summary"] if p != "best_stake"
                     for k in r["summary"][p]) for r in rows if r["b"] in full]
        print("checked", len(diffs), "replicates against full recomputation; max abs diff", max(diffs), flush=True)
        assert max(diffs) < 1e-9
    json.dump(dict(variant=MAIN, R=R, B_done=len(rows), rows=rows,
                   note="obs_rule re-evaluated after the per-count kappa fix (src/boot_obsrule.py)"),
              open("results/headline_boot.json", "w"))
