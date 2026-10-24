"""HONZA-132 (Sol #2, Astra #4; "make the hidden-information intervention the methodological centre"): wins lost as a
function of the two things the data cannot decide.

The challenge data identify the decision noise sigma_r of the fitted rule s > kappa(x) (and, in the lapse models, the
lapse rate), not how that noise splits. Two shares are not identified:
  rho    the share of sigma_r^2 that is PERCEPTUAL -- noise in what the player sees, which no decision rule removes.
         The rest, 1 - rho, is criterion noise (the threshold jitters from pitch to pitch), which a disciplined rule
         removes. The benchmark player sees s = w + sqrt(rho) sigma_r e and holds a posterior calibrated to that signal;
         the behavioural rule adds the criterion noise back (sim.draw), so the fitted model is reproduced for every rho.
  lambda the share of the lapses (challenge curve topping out below 1) that a rule change removes (lapse models only):
         the benchmark keeps the share 1 - lambda of them, coupled to the behavioural rule's (headline.evaluate_all
         "bench").
PRIMARY intervention: change only the decision rule, keep the revealed information and attention -- rho = 1,
lambda = 0. Every other (rho, lambda) is a secondary intervention (removing lapses: lambda > 0; removing criterion
noise: rho < 1). For each model at least as good as the main model by BIC the grid is evaluated on ONE cross-fitted
split (crossfit.crossfit_frame, seed 5000, the first split of crossfit.py point), so its (1, 0) cell is comparable
to that split's headline value.
usage: python src/intervention.py [R] [models]  -> results/intervention.json"""
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "src")

RHOS = [1.0, 0.75, 0.5, 0.25]
LAMS = [0.0, 0.5, 1.0]
MODELS = ["kappa_sr_region_g", "main_lapse", "re_gh11", "re_lapse_gh11", "main_time", "main_time_lapse"]
PATH = "results/intervention.json"


def bic_set(models):
    """the models at least as good as the main model by BIC (results/perception_variants.json)."""
    pv = json.load(open("results/perception_variants.json"))
    b0 = pv["kappa_sr_region_g"]["bic"]
    return [k for k in models if k in pv and pv[k]["bic"] <= b0 + 1e-9]


def main(R=6, models=None):
    import lik, crossfit as CF
    from headline import evaluate_all, summarise, spec_variant
    st = json.load(open(PATH)) if os.path.exists(PATH) else {}
    res = st.get("models", {})
    m0 = pd.read_parquet("data/called_2026_w.parquet")
    mb = CF.crossfit_frame(m0, CF.statcast(), np.random.default_rng(5000))
    pv = json.load(open("results/perception_variants.json"))
    for name in bic_set(models or MODELS):
        v = spec_variant(name); spec = v["spec"]; is_re = bool(spec.get("re"))
        t = time.time()
        r, _ = lik.fit(lik.prepare(mb, players=is_re), spec, th0=np.load(v["theta"]), maxiter=60)
        base = dict(v, theta_override=r.x, frame=mb)
        if is_re:
            data = lik.prepare(mb, players=True); tag = f"iv_{name}"
            for side in ("off", "def"):
                lik.re_posterior(data, r.x, side, spec).to_parquet(f"data/{tag}_{side}.parquet")
            base["re_tag"] = tag
        lapse = bool(spec.get("lapse"))
        cells = []
        for rho in RHOS:
            for lam in (LAMS if lapse else [0.0]):
                mm, _, out, dr = evaluate_all(dict(base, rho=rho, lam=lam), R=R, seed=1)
                sm = summarise(out, int(mm.gidx.max() + 1))
                lost_d = np.array([d["totals"]["bench"] for d in dr]) - sm["observed"]["wins"]
                cells.append(dict(rho=rho, lam=lam, lost=sm["bench"]["wins"] - sm["observed"]["wins"],
                                  lost_mc_se=float(lost_d.std(ddof=1) / np.sqrt(R)), bench=sm["bench"]["wins"],
                                  observed=sm["observed"]["wins"], obs_rule=sm["obs_rule"]["wins"],
                                  stake_0008=sm["stake_0.008"]["wins"], n_bench=sm["bench"]["n"]))
                print(name, rho, lam, round(cells[-1]["lost"], 3), round(time.time() - t), flush=True)
        res[name] = dict(bic=pv[name]["bic"], delta_bic_vs_main=pv[name]["bic"] - pv["kappa_sr_region_g"]["bic"],
                         lapse=lapse, R=R, ok=bool(r.success), cells=cells,
                         primary=[c for c in cells if c["rho"] == 1.0 and c["lam"] == 0.0][0]["lost"])
        prim = [x["primary"] for x in res.values()]
        json.dump(dict(R=R, rhos=RHOS, lams=LAMS, split_seed=5000, models=res,
                       primary_range=[float(min(prim)), float(max(prim))]), open(PATH, "w"), indent=1)


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    main(R, sys.argv[2].split(",") if len(sys.argv) > 2 else None)
