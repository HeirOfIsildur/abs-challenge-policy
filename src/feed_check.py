"""MLB 2026 rebuilt on the Stats API feed's own pitch coordinates and strike-zone limits (build_ext.py,
tag mlb2026feed) instead of Statcast.  In Triple-A and spring the feed geometry reproduces the ABS rulings almost
exactly (sigma_m -> 0), so the truth model drops out and sigma_r is identified by the challenge curve alone.
Fits the baseline and the BIC-best perception model on this table and recomputes wins lost (DP + simulation).
usage: python src/feed_check.py [R]  -> results/feed_check.json"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import lik, sim
from final_ci import spec_of

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    m = pd.read_parquet("data/called_mlb2026feed_w.parquet")
    data = lik.prepare(m); res = {}
    for variant in ("normal", "kappa_sr_region_g"):
        spec = spec_of(variant)
        th0 = np.load(f"data/th_{variant}.npy")       # Deterministic truth, no sigma_m to start near 0
        r, nll = lik.fit(data, spec, th0=th0, maxiter=30, verbose=True)
        cov = np.linalg.inv(lik.hessian(nll, r.x)); nt = lik.n_truth(spec); ns = lik.n_side(spec)
        p = lik.readable(r.x, spec)
        out = dict(nll=float(r.fun), converged=bool(r.success), params={k: v for k, v in p.items() if "beta" not in k})
        for i, side in ((nt + 6, "off"), (nt + ns + 6, "def")):
            se = float(np.sqrt(abs(cov[i, i]))); out[f"{side}_sr_ci95"] = [p[f"{side}_sr"] * np.exp(-1.96 * se), p[f"{side}_sr"] * np.exp(1.96 * se)]
        from headline import evaluate_all
        _, _, o, _ = evaluate_all(dict(theta="override", theta_override=r.x, spec=spec, frame=m), R=R, seed=1)
        tot = {k: v[0].sum() / 30 for k, v in o.items()}      # Score-aware DP benchmark (was the pooled DP)
        out["wins"] = dict(opt=tot["opt_score"], observed=tot["observed"], obs_rule=tot["obs_rule"], lost=tot["opt_score"] - tot["observed"],
                           lost_vs_rule=tot["opt_score"] - tot["obs_rule"], best_stake=max(v for k, v in tot.items() if k.startswith("stake_")))
        res[variant] = out
        print(variant, json.dumps(out, default=float), flush=True)
        json.dump(res, open("results/feed_check.json", "w"), indent=1, default=float)
