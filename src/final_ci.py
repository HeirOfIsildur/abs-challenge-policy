"""Wins lost against the optimum with a CI that carries parameter uncertainty.
Full bootstrap (CI_MODE=refit, default): resample the 2026 games with replacement, refit the perception + truth
model on the resample (lik.fit, warm start), re-solve the DP and simulate on the resample (R draws), record
opt - observed and opt - simulated fitted rule per team-season.
Parametric check (CI_MODE=param): theta* ~ N(theta_hat, H^-1) on game-resampled data, no refit.
usage: python src/final_ci.py <B> [variant] [R] [workers]  -> results/final_ci_<variant>[_param].json
variant: a key of robust_fit.VARIANTS (fitted by robust_fit.py), default "normal" (the baseline)."""
import json, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, "src")


def spec_of(variant):
    from lik import spec_of as s
    return s(variant)


def resample(by, games, rng):
    pick = rng.choice(games, len(games))
    return pd.concat([by[g].assign(game_pk=i) for i, g in enumerate(pick)], ignore_index=True)


def one(args):
    b, variant, R, mode = args
    import lik, sim
    spec = spec_of(variant)
    m = pd.read_parquet("data/called_2026_w.parquet"); by = {g: d for g, d in m.groupby("game_pk")}; games = np.array(sorted(by))
    rng = np.random.default_rng(1000 + b)
    mb = resample(by, games, rng)
    th = np.load(f"data/th_{variant}.npy")
    t = time.time()
    if mode == "refit":
        r, _ = lik.fit(lik.prepare(mb, players=bool(spec.get("re"))), spec, th0=th, maxiter=60)
        thb, ok = r.x, bool(r.success)
    else:
        thb, ok = rng.multivariate_normal(th, np.load(f"data/cov_{variant}.npy")), True
    # random-effects variant: player effects from the full-sample EB posteriors (not refitted per replicate)
    _, out, _ = sim.evaluate(dict(theta="override", theta_override=thb, spec=spec, frame=mb, re=bool(spec.get("re"))), R=R, seed=b)
    tot = {k: v[0].sum() / 30 for k, v in out.items()}
    rd = lik.readable(thb, spec)
    return dict(b=b, ok=ok, opt=tot["opt"], observed=tot["observed"], obs_rule=tot["obs_rule"],
                lost=tot["opt"] - tot["observed"], lost_vs_rule=tot["opt"] - tot["obs_rule"],
                best_stake=max(v for k, v in tot.items() if k.startswith("stake_")),
                off_sr=rd["off_sr"], def_sr=rd["def_sr"], secs=round(time.time() - t, 1))


if __name__ == "__main__":
    B = int(sys.argv[1]); variant = sys.argv[2] if len(sys.argv) > 2 else "normal"
    R = int(sys.argv[3]) if len(sys.argv) > 3 else 2; workers = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    mode = os.environ.get("CI_MODE", "refit")
    path = f"results/final_ci_{variant}{'' if mode == 'refit' else '_' + mode}.json"
    rows = json.load(open(path))["rows"] if os.path.exists(path) else []        # resume: keep finished replicates
    done = {r["b"] for r in rows}
    with mp.get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(one, [(b, variant, R, mode) for b in range(B) if b not in done]):
            rows.append(r)
            if len(rows) % 10 == 0 or len(rows) >= B:
                df = pd.DataFrame(rows)
                summ = {k: [round(float(df[k].mean()), 3), *np.round(np.percentile(df[k], [2.5, 97.5]), 3).tolist()]
                        for k in ("opt", "observed", "obs_rule", "lost", "lost_vs_rule", "best_stake", "off_sr", "def_sr")}
                json.dump(dict(variant=variant, mode=mode, B_done=len(rows), R=R, summary=summ, rows=rows), open(path, "w"), indent=1)
                print(len(rows), json.dumps(summ), flush=True)
